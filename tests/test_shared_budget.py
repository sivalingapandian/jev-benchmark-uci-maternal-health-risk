"""Durable shared-budget tests: one cumulative exposure across paid phases.

The single ``$20`` software guard must survive process restarts: a bound ledger
flushes every reservation/settlement to disk, an interrupted session's pending
reservation is recovered (never assumed free), prior development-run spend is
reconciled into a fresh ledger, and a configured cap above the approved ceiling
is refused. All fixtures are local files; no provider calls are made.
"""

import json

import pytest

from jev_benchmark.budget import (
    CAP_USD,
    PLANNING_SEED_USD,
    open_shared_budget,
    reconcile_historical_exposure,
    validate_cap,
)
from jev_benchmark.errors import BudgetConfigError, HistoricalExposureError
from jev_benchmark.providers.base import Usage


def test_shared_ledger_persists_across_sessions(tmp_path):
    path = tmp_path / "runs" / "shared-budget.json"
    ledger = open_shared_budget(path, runs_dir=tmp_path / "runs")
    token = ledger.reserve()
    ledger.settle_billed(token, 0.001)
    del ledger

    reloaded = open_shared_budget(path)
    assert reloaded.billed_openrouter == pytest.approx(0.001)
    assert reloaded.planning_seed == pytest.approx(PLANNING_SEED_USD)
    assert reloaded.exposure == pytest.approx(PLANNING_SEED_USD + 0.001)


def test_pending_reservation_survives_interruption(tmp_path):
    # A reserved-but-never-settled attempt (process killed mid-call) must keep
    # its conservative exposure on reload, folded into retained-unknown.
    path = tmp_path / "runs" / "shared-budget.json"
    ledger = open_shared_budget(path, runs_dir=tmp_path / "runs")
    ledger.reserve()  # in-flight when the process "dies"; never settled

    reloaded = open_shared_budget(path)
    assert reloaded.outstanding_reserved == 0.0
    reservation = reloaded.pricing.per_attempt_reservation
    assert reloaded.reserved_unknown == pytest.approx(reservation)
    assert reloaded.exposure == pytest.approx(PLANNING_SEED_USD + reservation)


def test_fresh_shared_ledger_reconciles_prior_run_spend(tmp_path):
    # Prior development/pilot runs recorded budget snapshots; a fresh shared
    # ledger must fold their marginal spend in instead of starting from scratch.
    runs_dir = tmp_path / "runs"
    for name, billed, estimated in (("pilot", 0.001, 0.00004), ("run", 0.002, 0.0)):
        run_dir = runs_dir / name
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "budget": {
                        "planning_seed": PLANNING_SEED_USD,  # seed not re-counted
                        "billed_openrouter": billed,
                        "estimated_jev": estimated,
                        "reserved_unknown": 0.0,
                        "outstanding_reserved": 0.0,
                    }
                }
            )
        )
    ledger = open_shared_budget(runs_dir / "shared-budget.json", runs_dir=runs_dir)
    historical = 0.001 + 0.00004 + 0.002
    assert ledger.reserved_unknown == pytest.approx(historical)
    assert ledger.exposure == pytest.approx(PLANNING_SEED_USD + historical)


def test_reconcile_raises_when_history_cannot_be_bounded(tmp_path):
    runs_dir = tmp_path / "runs"
    bad = runs_dir / "broken"
    bad.mkdir(parents=True)
    (bad / "run.json").write_text("{not json")
    with pytest.raises(HistoricalExposureError):
        reconcile_historical_exposure(runs_dir)


def test_reconcile_ignores_missing_runs_dir(tmp_path):
    assert reconcile_historical_exposure(tmp_path / "nope") == 0.0


def test_validate_cap_refuses_above_approved_ceiling():
    with pytest.raises(BudgetConfigError):
        validate_cap(CAP_USD + 0.01)
    # Tightening the guard is allowed.
    assert validate_cap(5.0) == 5.0
    assert validate_cap(CAP_USD) == CAP_USD


def test_shared_ledger_load_clamps_stored_cap_to_ceiling(tmp_path):
    path = tmp_path / "runs" / "shared-budget.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "cap": 500.0,  # stale/forged over-ceiling cap must not widen the guard
                "planning_seed": PLANNING_SEED_USD,
                "billed_openrouter": 0.0,
                "estimated_jev": 0.0,
                "reserved_unknown": 0.0,
                "pending": [],
            }
        )
    )
    ledger = open_shared_budget(path, cap=CAP_USD)
    assert ledger.cap == CAP_USD


def test_bound_ledger_flushes_jev_estimate_to_shared_file(tmp_path):
    path = tmp_path / "runs" / "shared-budget.json"
    ledger = open_shared_budget(path, runs_dir=tmp_path / "runs")
    token = ledger.reserve()
    ledger.settle_estimated(token, Usage(input_tokens=1000, output_tokens=100))

    saved = json.loads(path.read_text())
    assert saved["estimated_jev"] == pytest.approx(ledger.estimated_jev)
    assert saved["estimated_jev"] > 0.0
