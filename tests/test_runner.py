"""Sequential runner tests: retries, budget stop, durable resume, timing.

A scripted inner adapter yields a queued result (or raises) per stage call, so
retry/interrupt/resume behaviour is deterministic. These complement the live
pilot CLI, which exercises the real three-provider dispatch.
"""

import json

import pytest

from jev_benchmark import runner, spec, workflow
from jev_benchmark.budget import BudgetLedger, Pricing, open_shared_budget
from jev_benchmark.errors import (
    InvalidDecisionError,
    ProviderRequestError,
    RunDirectoryError,
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


def test_resume_refuses_when_task_wording_changes_identity(tmp_path, monkeypatch):
    # A reworded stage instruction changes the frozen experiment identity
    # (config_hash), which must block an otherwise-matching resume.
    run_dir = tmp_path / "run_identity"
    original_hash = spec.config_hash()
    runner.run(
        store=runner.RunStore(run_dir),
        adapters=_three_providers_all_high(),
        rows=_rows(1),
        model_ids=MODEL_IDS,
        config_hash=original_hash,
        ordered_record_sha256="hash",
        ledger=BudgetLedger(),
    )
    monkeypatch.setitem(
        spec.STAGE1, "instructions", spec.STAGE1_INSTRUCTIONS + " (reworded)"
    )
    reworded_hash = spec.config_hash()
    assert reworded_hash != original_hash
    with pytest.raises(RunIdentityError):
        runner.run(
            store=runner.RunStore(run_dir),
            adapters=_three_providers_all_high(),
            rows=_rows(1),
            model_ids=MODEL_IDS,
            config_hash=reworded_hash,
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


# --------------------------------------------------------------------------- #
# Stage journal and occupied-directory guard
# --------------------------------------------------------------------------- #
def test_stage_journal_records_each_completed_stage(tmp_path):
    store = runner.RunStore(tmp_path / "run8")
    runner.run(
        store=store,
        adapters=_three_providers_all_high(),
        rows=_rows(1),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=BudgetLedger(),
    )
    entries = store.load_stage_entries()
    # One all-high case per provider, so one stage-1 entry each.
    assert len(entries) == 3
    assert {e["requested_model"] for e in entries} == set(MODEL_IDS)
    assert all(e["stage"] == "stage1" for e in entries)
    assert all(e["decision"]["choice"] == "high" for e in entries)
    assert all(len(e["attempts"]) == 1 for e in entries)


def test_fresh_run_refuses_occupied_directory(tmp_path):
    run_dir = tmp_path / "run9"
    store = runner.RunStore(run_dir)
    runner.run(
        store=store,
        adapters=_three_providers_all_high(),
        rows=_rows(1),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=BudgetLedger(),
    )
    cases_before = store.load_cases()
    # A non-resume run on the same directory must refuse, not append duplicates.
    with pytest.raises(RunDirectoryError):
        runner.run(
            store=runner.RunStore(run_dir),
            adapters=_three_providers_all_high(),
            rows=_rows(1),
            model_ids=MODEL_IDS,
            config_hash="cfg",
            ordered_record_sha256="hash",
            ledger=BudgetLedger(),
        )
    assert runner.RunStore(run_dir).load_cases() == cases_before


class CrashAfterStage1(ProviderAdapter):
    """Answers stage 1 then crashes hard (non-provider fault) on stage 2."""

    def __init__(self, model_id, cost):
        self.model_id = model_id
        self._cost = cost
        self.calls = []

    def decide(self, stage, state):
        self.calls.append(stage["id"])
        if stage["id"] == "stage2":
            raise RuntimeError("simulated process crash mid-workflow")
        return _decision("stage1", "other", self.model_id, cost=self._cost)

    def close(self):
        pass


def _openrouter_providers_stage2(model_results):
    """Build a three-provider dict; ``model_results`` maps model -> stage2 outcome."""
    adapters = {}
    for model_id in MODEL_IDS:
        if model_id == "jev-1.13.0":
            adapters[model_id] = _high_only("jev-1.13.0", cost=None)
        else:
            result = model_results.get(model_id, "high")
            if result == "high":
                adapters[model_id] = _high_only(model_id, cost=0.00002)
            else:
                adapters[model_id] = result
    return adapters


def test_resume_recovers_interrupted_stage1_without_redispatch(tmp_path):
    run_dir = tmp_path / "run10"
    store = runner.RunStore(run_dir)
    crashing = CrashAfterStage1("openai/gpt-4o-mini", cost=0.00002)
    adapters = _openrouter_providers_stage2({"openai/gpt-4o-mini": crashing})

    # Session 1 crashes mid-workflow after stage 1 succeeded and was journaled.
    with pytest.raises(RuntimeError):
        runner.run(
            store=store,
            adapters=adapters,
            rows=_rows(1),
            model_ids=MODEL_IDS,
            config_hash="cfg",
            ordered_record_sha256="hash",
            ledger=BudgetLedger(),
        )
    # Row-major order means jev (first provider) completed before the crash;
    # the gpt case was interrupted, but its stage-1 success is durably journaled.
    completed = store.load_cases()
    assert [c.requested_model for c in completed] == ["jev-1.13.0"]
    journal = store.load_stage_entries()
    partial = [e for e in journal if e["requested_model"] == "openai/gpt-4o-mini"]
    assert len(partial) == 1 and partial[0]["stage"] == "stage1"

    # Session 2 resumes with a healthy adapter for the previously-crashing model.
    healthy = ScriptedInner(
        "openai/gpt-4o-mini",
        {"stage2": [_decision("stage2", "mid", "openai/gpt-4o-mini", cost=0.00003)]},
    )
    adapters2 = _openrouter_providers_stage2({"openai/gpt-4o-mini": healthy})
    result = runner.run(
        store=runner.RunStore(run_dir),
        adapters=adapters2,
        rows=_rows(1),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=BudgetLedger(),
        resume=True,
    )
    # Stage 1 was NOT re-dispatched for the resumed case; only stage 2 ran.
    assert healthy.calls == ["stage2"]
    assert result.state.status == runner.COMPLETE

    cases = runner.RunStore(run_dir).load_cases()
    resumed = next(c for c in cases if c.requested_model == "openai/gpt-4o-mini")
    assert resumed.outcome == workflow.PREDICTED
    assert resumed.final_label == "mid risk"
    assert resumed.stage_count == 2
    # The case carries both the journaled stage-1 attempt and the new stage-2
    # attempt, in order.
    assert [a.stage for a in resumed.attempts] == ["stage1", "stage2"]


def test_resume_does_not_double_bill_interrupted_stage(tmp_path):
    run_dir = tmp_path / "run11"
    store = runner.RunStore(run_dir)
    crashing = CrashAfterStage1("openai/gpt-4o-mini", cost=0.00002)
    adapters = _openrouter_providers_stage2({"openai/gpt-4o-mini": crashing})
    with pytest.raises(RuntimeError):
        runner.run(
            store=store,
            adapters=adapters,
            rows=_rows(1),
            model_ids=MODEL_IDS,
            config_hash="cfg",
            ordered_record_sha256="hash",
            ledger=BudgetLedger(planning_seed=0.0),
        )

    healthy = ScriptedInner(
        "openai/gpt-4o-mini",
        {"stage2": [_decision("stage2", "mid", "openai/gpt-4o-mini", cost=0.00003)]},
    )
    adapters2 = _openrouter_providers_stage2({"openai/gpt-4o-mini": healthy})
    result = runner.run(
        store=runner.RunStore(run_dir),
        adapters=adapters2,
        rows=_rows(1),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=BudgetLedger(planning_seed=0.0),
        resume=True,
    )
    # The journaled stage-1 billed cost is replayed exactly once, then the new
    # session's costs are added: gpt stage1 $0.00002 (journaled, session 1) +
    # gpt stage2 $0.00003 + gemini $0.00002 (both session 2). Jev is estimated,
    # not billed, and was completed in session 1 so it is skipped on resume.
    billed = result.state.budget["billed_openrouter"]
    assert billed == pytest.approx(0.00002 + 0.00003 + 0.00002)


def test_resume_with_bound_shared_ledger_keeps_cumulative_exposure(tmp_path):
    # When the ledger is bound to a durable shared store, resume must trust that
    # store (already flushed during the interrupted session) and never replay
    # settlements a second time.
    run_dir = tmp_path / "run12"
    budget_path = tmp_path / "shared-budget.json"
    ledger = open_shared_budget(budget_path, runs_dir=tmp_path / "empty-runs")
    runner.run(
        store=runner.RunStore(run_dir),
        adapters=_three_providers_all_high(),
        rows=_rows(1),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=ledger,
    )
    exposure_after_first = open_shared_budget(budget_path).exposure

    ledger2 = open_shared_budget(budget_path)
    result = runner.run(
        store=runner.RunStore(run_dir),
        adapters=_three_providers_all_high(),
        rows=_rows(1),  # all pairs already completed: resume is a no-op
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=ledger2,
        resume=True,
    )
    assert result.newly_completed == 0
    assert result.ledger.exposure == pytest.approx(exposure_after_first)


def test_run_json_budget_snapshot_includes_pending_key_via_shared_store(tmp_path):
    # Sanity: the shared store file written during a run is valid JSON with the
    # expected buckets.
    budget_path = tmp_path / "shared-budget.json"
    ledger = open_shared_budget(budget_path, runs_dir=tmp_path / "empty-runs")
    runner.run(
        store=runner.RunStore(tmp_path / "run13"),
        adapters=_three_providers_all_high(),
        rows=_rows(1),
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=ledger,
    )
    saved = json.loads(budget_path.read_text())
    assert saved["billed_openrouter"] == pytest.approx(0.00002 + 0.00003)
    assert saved["estimated_jev"] > 0.0
    assert saved["pending"] == []
