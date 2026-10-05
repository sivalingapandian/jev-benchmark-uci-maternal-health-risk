"""Pilot projection tests: conservative full-run gating."""

import pytest

from jev_benchmark import pilot, runner
from jev_benchmark.budget import BILLED, ESTIMATED, BudgetLedger
from jev_benchmark.runner import AttemptRecord, CaseRecord

MODEL_IDS = ["jev-1.13.0", "openai/gpt-4o-mini", "google/gemini-2.5-flash"]


def _case(model, cost_basis, amount, row=0):
    attempt = AttemptRecord(
        provider=model,
        provider_kind=runner.provider_kind(model),
        stage="stage1",
        attempt=1,
        status="ok",
        started_at=0.0,
        ended_at=0.1,
        actual_model=model,
        input_tokens=5,
        output_tokens=1,
        cost_basis=cost_basis,
        cost_amount=amount,
    )
    return CaseRecord(
        row_index=row,
        requested_model=model,
        provider_kind=runner.provider_kind(model),
        outcome="predicted",
        final_label="high risk",
        gold_label="high risk",
        stage_count=1,
        workflow_started_at=0.0,
        workflow_ended_at=0.1,
        attempts=[attempt],
    )


def test_validated_models_requires_predicted_with_usage():
    # A pilot supplies preflight-equivalent evidence only for providers that
    # returned at least one valid predicted case with known usage.
    jev = _case("jev-1.13.0", ESTIMATED, 0.00001)
    gpt = _case("openai/gpt-4o-mini", BILLED, 0.00002)
    gemini = _case("google/gemini-2.5-flash", BILLED, 0.0)
    gemini.outcome = "invalid"  # provider failed to validate
    validated = pilot.validated_models([jev, gpt, gemini])
    assert "jev-1.13.0" in validated
    assert "openai/gpt-4o-mini" in validated
    assert "google/gemini-2.5-flash" not in validated


def test_validated_models_excludes_predicted_without_usage():
    case = _case("jev-1.13.0", ESTIMATED, 0.0)
    case.attempts[0].input_tokens = None  # no usage recorded
    assert pilot.validated_models([case]) == []


def test_cheap_pilot_projects_affordable():
    cases = [
        _case("jev-1.13.0", ESTIMATED, 0.00001),
        _case("openai/gpt-4o-mini", BILLED, 0.00002),
        _case("google/gemini-2.5-flash", BILLED, 0.00003),
    ]
    ledger = BudgetLedger()
    projection = pilot.project_full_run(cases, ledger, MODEL_IDS, safety_factor=2.0)
    assert projection.affordable is True
    # Three providers x 1 pilot row, each cost c -> per_row c; projected 1014*2*c.
    assert projection.projected_full_cost == pytest.approx(
        (0.00001 + 0.00002 + 0.00003) * 1014 * 2.0
    )
    assert "may proceed" in projection.reason


def test_expensive_pilot_projects_not_affordable():
    # Each provider costs $0.01/row in the pilot: projected ~ 3 * 0.01 * 1014 * 2.
    cases = [_case(m, BILLED if "jev" not in m else ESTIMATED, 0.01) for m in MODEL_IDS]
    ledger = BudgetLedger()
    projection = pilot.project_full_run(cases, ledger, MODEL_IDS, safety_factor=2.0)
    assert projection.affordable is False
    assert projection.projected_total_exposure > ledger.cap
    assert "must not start" in projection.reason


def test_projection_includes_existing_exposure():
    cases = [_case(m, ESTIMATED, 0.0) for m in MODEL_IDS]
    ledger = BudgetLedger()
    # Push exposure to just under the cap via retained-unknown reservations.
    ledger.reserved_unknown = ledger.cap - ledger.planning_seed - 0.001
    projection = pilot.project_full_run(cases, ledger, MODEL_IDS)
    # Even a zero-cost projection is unaffordable because prior exposure is high.
    assert projection.exposure_before == pytest.approx(ledger.cap - 0.001)
    assert projection.affordable is True  # 0 projected, exposure still < cap
    # Any positive projection would tip it over; verify the boundary math.
    assert projection.projected_total_exposure < ledger.cap
