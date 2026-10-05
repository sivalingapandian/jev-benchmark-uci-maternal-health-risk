"""Sequential runner tests: retries, budget stop, durable resume, timing.

A scripted inner adapter yields a queued result (or raises) per stage call, so
retry/interrupt/resume behaviour is deterministic. These complement the live
pilot CLI, which exercises the real three-provider dispatch.
"""

import pytest

from jev_benchmark import runner, spec, workflow
from jev_benchmark.budget import BudgetLedger, Pricing
from jev_benchmark.errors import (
    InvalidDecisionError,
    ProviderRequestError,
    RunIdentityError,
)
from jev_benchmark.providers.base import ProviderAdapter, StageDecision, Usage

MODEL_IDS = list(spec.MODEL_IDS)
STATE = {
    "Age": 30,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "BS": 7.5,
    "BodyTemp": 98.0,
    "HeartRate": 76,
}


def _decision(stage_id, choice, model, *, cost=None, in_tok=5, out_tok=1):
    return StageDecision(
        stage=stage_id,
        requested_model=model,
        actual_model=model,
        choice=choice,
        confidence=0.9,
        probabilities=None,
        usage=Usage(input_tokens=in_tok, output_tokens=out_tok, cost=cost),
    )


class ScriptedInner(ProviderAdapter):
    """Yields queued results per stage id; an Exception item is raised."""

    def __init__(self, model_id, per_stage):
        self.model_id = model_id
        self._per_stage = {k: list(v) for k, v in per_stage.items()}
        self.calls = []

    def decide(self, stage, state):
        self.calls.append(stage["id"])
        item = self._per_stage[stage["id"]].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass


class ConstantInner(ProviderAdapter):
    """Stateless inner: returns a fresh decision per call (like a live adapter)."""

    def __init__(self, model_id, choice, cost=None):
        self.model_id = model_id
        self._choice = choice
        self._cost = cost
        self.calls = 0

    def decide(self, stage, state):
        self.calls += 1
        return _decision(stage["id"], self._choice, self.model_id, cost=self._cost)

    def close(self):
        pass


def _high_only(model, cost=0.00001):
    return ConstantInner(model, "high", cost=cost)


def _three_providers_all_high():
    return {
        "jev-1.13.0": _high_only("jev-1.13.0", cost=None),
        "openai/gpt-4o-mini": _high_only("openai/gpt-4o-mini", cost=0.00002),
        "google/gemini-2.5-flash": _high_only("google/gemini-2.5-flash", cost=0.00003),
    }


def _rows(n):
    return [{"index": i, "features": dict(STATE), "label": "high risk"} for i in range(n)]


# --------------------------------------------------------------------------- #
# MeasuredAdapter: retries, settlement, timing
# --------------------------------------------------------------------------- #
def test_measured_adapter_retries_transport_error_then_succeeds():
    inner = ScriptedInner(
        "openai/gpt-4o-mini",
        {
            "stage1": [
                ProviderRequestError("timeout"),
                _decision("stage1", "high", "openai/gpt-4o-mini", cost=0.00002),
            ]
        },
    )
    ledger = BudgetLedger()
    measured = runner.MeasuredAdapter(inner, ledger, max_retries=2)
    decision = measured.decide(spec.STAGE1, STATE)
    assert decision.choice == "high"
    # Two attempts recorded: one failed (retained unknown), one ok (billed).
    assert [a.status for a in measured.attempts] == ["error", "ok"]
    assert ledger.reserved_unknown > 0.0  # the timed-out attempt is not free
    assert ledger.billed_openrouter == pytest.approx(0.00002)


def test_measured_adapter_exhausts_retries_and_raises():
    inner = ScriptedInner(
        "openai/gpt-4o-mini",
        {
            "stage1": [
                ProviderRequestError("t1"),
                ProviderRequestError("t2"),
                ProviderRequestError("t3"),
            ]
        },
    )
    ledger = BudgetLedger()
    measured = runner.MeasuredAdapter(inner, ledger, max_retries=2)
    with pytest.raises(ProviderRequestError):
        measured.decide(spec.STAGE1, STATE)
    assert len(measured.attempts) == 3  # 1 try + 2 retries, all recorded


def test_measured_adapter_does_not_retry_invalid():
    inner = ScriptedInner("openai/gpt-4o-mini", {"stage1": [InvalidDecisionError("bad json")]})
    ledger = BudgetLedger()
    measured = runner.MeasuredAdapter(inner, ledger, max_retries=3)
    with pytest.raises(InvalidDecisionError):
        measured.decide(spec.STAGE1, STATE)
    assert len(measured.attempts) == 1  # terminal, not retried


def test_measured_adapter_records_monotonic_duration():
    inner = _high_only("openai/gpt-4o-mini", cost=0.00002)
    measured = runner.MeasuredAdapter(inner, BudgetLedger(), max_retries=0)
    measured.decide(spec.STAGE1, STATE)
    attempt = measured.attempts[0]
    assert attempt.duration is not None
    assert attempt.duration >= 0.0


# --------------------------------------------------------------------------- #
# run(): population, durability, timing
# --------------------------------------------------------------------------- #
def test_run_covers_every_pair_and_persists(tmp_path):
    store = runner.RunStore(tmp_path / "run1")
    result = runner.run(
        store=store,
        adapters=_three_providers_all_high(),
        rows=_rows(2),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=BudgetLedger(),
    )
    assert result.state.status == runner.COMPLETE
    assert result.newly_completed == 6  # 2 rows x 3 providers
    assert store.run_path.is_file()
    cases = store.load_cases()
    assert len(cases) == 6
    assert all(c.outcome == workflow.PREDICTED for c in cases)
    # Workflow timing captured for each case.
    assert all(c.workflow_duration is not None for c in cases)


def test_run_separates_billed_and_estimated(tmp_path):
    store = runner.RunStore(tmp_path / "run2")
    ledger = BudgetLedger()
    runner.run(
        store=store,
        adapters=_three_providers_all_high(),
        rows=_rows(1),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=ledger,
    )
    # Jev has no billed cost (estimated), both OpenRouter models are billed.
    assert ledger.billed_openrouter == pytest.approx(0.00002 + 0.00003)
    assert ledger.estimated_jev > 0.0


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #
def test_resume_skips_completed_and_preserves_accounting(tmp_path):
    run_dir = tmp_path / "run3"
    store = runner.RunStore(run_dir)
    # First session: only one provider available, fails the other two so they are
    # terminal outcomes (still "completed" cases). Simpler: run 1 row fully.
    runner.run(
        store=store,
        adapters=_three_providers_all_high(),
        rows=_rows(1),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=BudgetLedger(),
    )
    first_cases = store.load_cases()
    assert len(first_cases) == 3

    # Resume with 2 rows: the first row's 3 pairs are skipped, 3 new pairs added.
    store2 = runner.RunStore(run_dir)
    result = runner.run(
        store=store2,
        adapters=_three_providers_all_high(),
        rows=_rows(2),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=BudgetLedger(),
        resume=True,
    )
    assert result.newly_completed == 3
    assert len({c.key for c in store2.load_cases()}) == 6
    # Accounting was rebuilt from replay and includes the first session's spend.
    assert result.state.budget["billed_openrouter"] == pytest.approx(2 * (0.00002 + 0.00003))


def test_resume_refuses_on_changed_config(tmp_path):
    run_dir = tmp_path / "run4"
    store = runner.RunStore(run_dir)
    runner.run(
        store=store,
        adapters=_three_providers_all_high(),
        rows=_rows(1),
        model_ids=MODEL_IDS,
        config_hash="cfg-A",
        ordered_record_sha256="hash",
        ledger=BudgetLedger(),
    )
    with pytest.raises(RunIdentityError):
        runner.run(
            store=runner.RunStore(run_dir),
            adapters=_three_providers_all_high(),
            rows=_rows(1),
            model_ids=MODEL_IDS,
            config_hash="cfg-B",  # changed specification
            ordered_record_sha256="hash",
            ledger=BudgetLedger(),
            resume=True,
        )


def test_resume_refuses_on_changed_dataset_hash(tmp_path):
    run_dir = tmp_path / "run5"
    store = runner.RunStore(run_dir)
    runner.run(
        store=store,
        adapters=_three_providers_all_high(),
        rows=_rows(1),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash-A",
        ledger=BudgetLedger(),
    )
    with pytest.raises(RunIdentityError):
        runner.run(
            store=runner.RunStore(run_dir),
            adapters=_three_providers_all_high(),
            rows=_rows(1),
            model_ids=MODEL_IDS,
            config_hash="cfg",
            ordered_record_sha256="hash-B",  # changed data
            ledger=BudgetLedger(),
            resume=True,
        )


# --------------------------------------------------------------------------- #
# Budget stop
# --------------------------------------------------------------------------- #
def _three_providers_high_cost(cost=1.0):
    # Each attempt bills ~$1 so settled spend accumulates toward the cap even
    # though reservations release; the per-attempt reservation is sized to match.
    return {
        "jev-1.13.0": _high_only("jev-1.13.0", cost=None),
        "openai/gpt-4o-mini": _high_only("openai/gpt-4o-mini", cost=cost),
        "google/gemini-2.5-flash": _high_only("google/gemini-2.5-flash", cost=cost),
    }


def test_run_stops_before_cap_and_preserves_completed(tmp_path):
    store = runner.RunStore(tmp_path / "run6")
    # Reservation sized to the real ~$1 bill, so the guard is genuinely
    # conservative: dispatch is refused before exposure reaches the $3.5 cap.
    pricing = Pricing(per_attempt_reservation=1.0)
    ledger = BudgetLedger(cap=3.5, pricing=pricing, planning_seed=0.0)
    result = runner.run(
        store=store,
        adapters=_three_providers_high_cost(cost=1.0),
        rows=_rows(5),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=ledger,
    )
    assert result.stopped is True
    assert result.state.status == runner.STOPPED_BUDGET
    assert result.state.stop_reason
    # Some pairs completed before the stop and remain recorded.
    saved = store.load_cases()
    assert 0 < len(saved) < 15
    # The persisted exposure never reached the cap (stop-before, not stop-at).
    assert result.state.budget["exposure"] < ledger.cap


def test_run_marks_incomplete_when_stopped_before_all_pairs(tmp_path):
    store = runner.RunStore(tmp_path / "run7")
    pricing = Pricing(per_attempt_reservation=1.0)
    ledger = BudgetLedger(cap=2.5, pricing=pricing, planning_seed=0.0)
    result = runner.run(
        store=store,
        adapters=_three_providers_high_cost(cost=1.0),
        rows=_rows(3),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=ledger,
    )
    assert result.state.status in (runner.STOPPED_BUDGET, runner.INCOMPLETE)
    assert len({c.key for c in store.load_cases()}) < result.state.planned_pairs
