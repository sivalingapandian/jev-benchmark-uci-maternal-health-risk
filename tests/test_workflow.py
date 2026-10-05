"""Two-stage branch-sequencing and invalid-outcome tests.

A scripted adapter lets these assert, deterministically, that stage 2 runs only
on a provider's own ``other`` answer and that malformed/refused/mismatched
responses become explicit outcomes with no fabricated label and no unearned
stage 2.
"""

import pytest

from jev_benchmark import spec, workflow
from jev_benchmark.errors import (
    InvalidDecisionError,
    ModelMismatchError,
    ProviderRequestError,
    RefusalError,
)
from jev_benchmark.providers.base import ProviderAdapter, StageDecision, Usage


def _decision(stage_id, choice, *, model="prov", actual="prov"):
    return StageDecision(
        stage=stage_id,
        requested_model=model,
        actual_model=actual,
        choice=choice,
        confidence=0.8,
        probabilities=None,
        usage=Usage(input_tokens=5, output_tokens=1),
    )


class ScriptedAdapter(ProviderAdapter):
    """Returns or raises a scripted result per stage id, recording call order."""

    def __init__(self, model_id, script):
        self.model_id = model_id
        self._script = script
        self.calls = []

    def decide(self, stage, state):
        self.calls.append(stage["id"])
        item = self._script[stage["id"]]
        if isinstance(item, Exception):
            raise item
        return item


def test_high_ends_at_high_risk_without_stage2():
    adapter = ScriptedAdapter("prov", {"stage1": _decision("stage1", "high")})
    outcome = workflow.evaluate_state(adapter, {"Age": 30})
    assert outcome.outcome == workflow.PREDICTED
    assert outcome.final_label == "high risk"
    assert outcome.stage_count == 1
    assert adapter.calls == ["stage1"]  # stage 2 never requested


def test_other_then_mid_maps_to_mid_risk():
    adapter = ScriptedAdapter(
        "prov",
        {"stage1": _decision("stage1", "other"), "stage2": _decision("stage2", "mid")},
    )
    outcome = workflow.evaluate_state(adapter, {"Age": 30})
    assert outcome.final_label == "mid risk"
    assert adapter.calls == ["stage1", "stage2"]


def test_other_then_low_maps_to_low_risk():
    adapter = ScriptedAdapter(
        "prov",
        {"stage1": _decision("stage1", "other"), "stage2": _decision("stage2", "low")},
    )
    outcome = workflow.evaluate_state(adapter, {"Age": 30})
    assert outcome.final_label == "low risk"
    assert adapter.calls == ["stage1", "stage2"]


def test_branch_follows_provider_not_gold_or_peer():
    # Same state, two providers with different stage-1 answers select different
    # branches: the branch depends only on each provider's own answer.
    high_adapter = ScriptedAdapter("a", {"stage1": _decision("stage1", "high")})
    other_adapter = ScriptedAdapter(
        "b",
        {"stage1": _decision("stage1", "other"), "stage2": _decision("stage2", "low")},
    )
    state = {"Age": 30}
    assert workflow.evaluate_state(high_adapter, state).final_label == "high risk"
    assert workflow.evaluate_state(other_adapter, state).final_label == "low risk"
    assert high_adapter.calls == ["stage1"]
    assert other_adapter.calls == ["stage1", "stage2"]


def test_evaluate_state_signature_has_no_label_input():
    # The gold label cannot enter the workflow: evaluate_state takes only state.
    import inspect

    params = list(inspect.signature(workflow.evaluate_state).parameters)
    assert params == ["adapter", "state"]


def test_stage1_invalid_blocks_stage2_and_has_no_label():
    adapter = ScriptedAdapter("prov", {"stage1": InvalidDecisionError("bad json")})
    outcome = workflow.evaluate_state(adapter, {"Age": 30})
    assert outcome.outcome == workflow.INVALID
    assert outcome.final_label is None
    assert outcome.stage_count == 1
    assert adapter.calls == ["stage1"]  # no unearned stage 2


def test_stage1_refusal_is_visible():
    adapter = ScriptedAdapter("prov", {"stage1": RefusalError("refused")})
    outcome = workflow.evaluate_state(adapter, {"Age": 30})
    assert outcome.outcome == workflow.REFUSED
    assert outcome.final_label is None


def test_stage1_model_mismatch_is_visible():
    adapter = ScriptedAdapter(
        "prov", {"stage1": ModelMismatchError("swap", actual_model="other-model")}
    )
    outcome = workflow.evaluate_state(adapter, {"Age": 30})
    assert outcome.outcome == workflow.MISMATCH
    assert outcome.actual_model == "other-model"
    assert outcome.final_label is None


def test_stage1_request_error_is_visible():
    adapter = ScriptedAdapter("prov", {"stage1": ProviderRequestError("timeout")})
    outcome = workflow.evaluate_state(adapter, {"Age": 30})
    assert outcome.outcome == workflow.ERROR
    assert outcome.final_label is None


def test_stage2_invalid_records_two_stages_without_label():
    adapter = ScriptedAdapter(
        "prov",
        {"stage1": _decision("stage1", "other"), "stage2": InvalidDecisionError("bad")},
    )
    outcome = workflow.evaluate_state(adapter, {"Age": 30})
    assert outcome.outcome == workflow.INVALID
    assert outcome.final_label is None
    assert outcome.stage_count == 2
    assert adapter.calls == ["stage1", "stage2"]


def test_outcome_usage_is_collected_from_every_stage():
    adapter = ScriptedAdapter(
        "prov",
        {"stage1": _decision("stage1", "other"), "stage2": _decision("stage2", "mid")},
    )
    outcome = workflow.evaluate_state(adapter, {"Age": 30})
    assert [u.input_tokens for u in outcome.usages] == [5, 5]


def test_compose_label_contract_matches_spec():
    assert spec.compose_label("high") == "high risk"
    assert spec.compose_label("other", "mid") == "mid risk"
    assert spec.compose_label("other", "low") == "low risk"
    with pytest.raises(ValueError):
        spec.compose_label("other", None)
