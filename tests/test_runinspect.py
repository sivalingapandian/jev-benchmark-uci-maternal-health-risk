"""Offline run-inspection tests: accounting split, latency scopes, completeness."""

import pytest

from jev_benchmark import runinspect, runner, spec, workflow
from jev_benchmark.budget import BudgetLedger
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


def _decision(stage_id, choice, model, cost=None):
    return StageDecision(
        stage=stage_id,
        requested_model=model,
        actual_model=model,
        choice=choice,
        confidence=0.9,
        probabilities=None,
        usage=Usage(input_tokens=5, output_tokens=1, cost=cost),
    )


class _Inner(ProviderAdapter):
    def __init__(self, model_id, per_stage):
        self.model_id = model_id
        self._per_stage = {k: list(v) for k, v in per_stage.items()}

    def decide(self, stage, state):
        return self._per_stage[stage["id"]].pop(0)

    def close(self):
        pass


def _adapters():
    return {
        "jev-1.13.0": _Inner(
            "jev-1.13.0", {"stage1": [_decision("stage1", "high", "jev-1.13.0")]}
        ),
        "openai/gpt-4o-mini": _Inner(
            "openai/gpt-4o-mini",
            {"stage1": [_decision("stage1", "high", "openai/gpt-4o-mini", 0.00002)]},
        ),
        "google/gemini-2.5-flash": _Inner(
            "google/gemini-2.5-flash",
            {"stage1": [_decision("stage1", "high", "google/gemini-2.5-flash", 0.00003)]},
        ),
    }


def test_inspection_reports_accounting_and_latency(tmp_path):
    store = runner.RunStore(tmp_path / "run")
    # Build fresh adapters per row by re-running rows one at a time via resume.
    for i in range(2):
        runner.run(
            store=runner.RunStore(tmp_path / "run"),
            adapters=_adapters(),
            rows=[{"index": i, "features": dict(STATE), "label": "high risk"}],
            model_ids=MODEL_IDS,
            config_hash="cfg",
            ordered_record_sha256="hash",
            ledger=BudgetLedger(),
            resume=(i > 0),
        )
    summary = runinspect.inspect_run(store)
    assert summary.completed_pairs == 6
    assert summary.complete is True
    # Accounting splits billed OpenRouter from estimated Jev.
    assert summary.accounting["billed_openrouter"] == pytest.approx(2 * (0.00002 + 0.00003))
    assert summary.accounting["estimated_jev"] > 0.0
    # Latency summarised at both scopes with sample counts.
    assert summary.overall_latency.api_attempt["count"] == 6
    assert summary.overall_latency.workflow["count"] == 6
    for prov in summary.providers:
        assert prov.outcome_counts[workflow.PREDICTED] == 2
        assert prov.latency.api_attempt["count"] == 2


def test_incomplete_run_is_reported_incomplete(tmp_path):
    store = runner.RunStore(tmp_path / "run")
    # Only run row 0 but plan for 2 rows by hand-editing planned indices.
    runner.run(
        store=store,
        adapters=_adapters(),
        rows=[{"index": 0, "features": dict(STATE), "label": "high risk"}],
        model_ids=MODEL_IDS,
        config_hash="cfg",
        ordered_record_sha256="hash",
        ledger=BudgetLedger(),
    )
    # Simulate a larger planned population so completion is partial.
    state = store.load_state()
    state.planned_row_indices = [0, 1, 2]
    state.status = runner.INCOMPLETE
    store.save_state(state)

    summary = runinspect.inspect_run(store)
    assert summary.complete is False
    assert summary.completed_pairs == 3
    assert summary.planned_pairs == 9
