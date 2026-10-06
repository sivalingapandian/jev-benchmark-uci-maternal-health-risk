"""Preflight aggregation tests with scripted adapters.

These check the pass/fail aggregation and redacted record shape. They do not
replace the live three-provider preflight that VAL-FLOW-001 requires; that is a
real paid CLI run.
"""

import pytest

from jev_benchmark import spec, workflow
from jev_benchmark.budget import BudgetLedger
from jev_benchmark.cli import _charge_preflight
from jev_benchmark.errors import InvalidDecisionError
from jev_benchmark.preflight import run_preflight, run_preflight_from_env
from jev_benchmark.providers.base import ProviderAdapter, StageDecision, Usage


def _decision(stage_id, choice, model, actual=None):
    return StageDecision(
        stage=stage_id,
        requested_model=model,
        actual_model=actual or model,
        choice=choice,
        confidence=0.8,
        probabilities=None,
        usage=Usage(input_tokens=5, output_tokens=1, cost=0.0000005),
    )


class ScriptedAdapter(ProviderAdapter):
    def __init__(self, model_id, script):
        self.model_id = model_id
        self._script = script

    def decide(self, stage, state):
        item = self._script[stage["id"]]
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass


def _all_ok_adapters():
    return [
        ScriptedAdapter("jev-1.13.0", {"stage1": _decision("stage1", "high", "jev-1.13.0")}),
        ScriptedAdapter(
            "openai/gpt-4o-mini",
            {
                "stage1": _decision("stage1", "other", "openai/gpt-4o-mini"),
                "stage2": _decision("stage2", "mid", "openai/gpt-4o-mini"),
            },
        ),
        ScriptedAdapter(
            "google/gemini-2.5-flash",
            {"stage1": _decision("stage1", "high", "google/gemini-2.5-flash")},
        ),
    ]


def test_preflight_passes_when_all_three_validate():
    result = run_preflight(_all_ok_adapters())
    assert result.ok is True
    assert result.covered_models == set(spec.MODEL_IDS)
    labels = {p.requested_model: p.final_label for p in result.providers}
    assert labels["jev-1.13.0"] == "high risk"
    assert labels["openai/gpt-4o-mini"] == "mid risk"


def test_preflight_fails_when_one_provider_is_invalid():
    adapters = _all_ok_adapters()
    adapters[1] = ScriptedAdapter(
        "openai/gpt-4o-mini", {"stage1": InvalidDecisionError("bad schema")}
    )
    result = run_preflight(adapters)
    assert result.ok is False
    bad = next(p for p in result.providers if p.requested_model == "openai/gpt-4o-mini")
    assert bad.ok is False
    assert bad.outcome == workflow.INVALID


def test_preflight_fails_when_usage_is_missing():
    # A valid predicted decision that reports no usage tokens is treated as
    # unknown, not a pass: missing usage cannot be a complete preflight.
    no_usage = StageDecision(
        stage="stage1",
        requested_model="jev-1.13.0",
        actual_model="jev-1.13.0",
        choice="high",
        confidence=0.8,
        probabilities=None,
        usage=Usage(),
    )
    adapters = _all_ok_adapters()
    adapters[0] = ScriptedAdapter("jev-1.13.0", {"stage1": no_usage})
    result = run_preflight(adapters)
    assert result.ok is False
    jev = next(p for p in result.providers if p.requested_model == "jev-1.13.0")
    assert jev.ok is False
    assert jev.outcome == workflow.PREDICTED  # it predicted, but usage was unknown


def test_usage_totals_keep_missing_tokens_unknown_not_zero():
    # Missing token counts must stay None in the redacted totals: laundering
    # them into 0 would let the budget charge such an attempt at a $0.00
    # estimate instead of retaining its reservation as unknown.
    no_usage = StageDecision(
        stage="stage1",
        requested_model="jev-1.13.0",
        actual_model="jev-1.13.0",
        choice="high",
        confidence=0.8,
        probabilities=None,
        usage=Usage(),
    )
    adapters = _all_ok_adapters()
    adapters[0] = ScriptedAdapter("jev-1.13.0", {"stage1": no_usage})
    result = run_preflight(adapters)
    jev = next(p for p in result.providers if p.requested_model == "jev-1.13.0")
    assert jev.usage_totals["input_tokens"] is None
    assert jev.usage_totals["output_tokens"] is None


def test_preflight_charge_retains_unknown_when_jev_usage_is_missing():
    # Path: the CLI preflight charge. A Jev preflight that returned a label but
    # no usage keeps its reservation as unknown in the shared ledger; it is
    # never settled at a $0.00 token estimate. The other providers' known usage
    # still settles to their own bases.
    no_usage = StageDecision(
        stage="stage1",
        requested_model="jev-1.13.0",
        actual_model="jev-1.13.0",
        choice="high",
        confidence=0.8,
        probabilities=None,
        usage=Usage(),
    )
    adapters = _all_ok_adapters()
    adapters[0] = ScriptedAdapter("jev-1.13.0", {"stage1": no_usage})
    result = run_preflight(adapters)

    ledger = BudgetLedger()
    _charge_preflight(ledger, result)
    assert ledger.estimated_jev == 0.0
    assert ledger.reserved_unknown == pytest.approx(ledger.pricing.per_attempt_reservation)
    assert ledger.billed_openrouter > 0.0  # known OpenRouter usage still bills
    assert ledger.exposure > ledger.planning_seed


def test_preflight_fails_when_a_model_is_missing():
    adapters = _all_ok_adapters()[:2]  # drop gemini
    result = run_preflight(adapters)
    assert result.ok is False
    assert result.covered_models != set(spec.MODEL_IDS)


def test_preflight_record_is_redacted_and_serialisable():
    result = run_preflight(_all_ok_adapters())
    data = result.as_dict()
    text = str(data)
    assert "api_key" not in text.lower()
    assert "authorization" not in text.lower()
    # Usage is summarised per provider.
    jev = next(p for p in data["providers"] if p["requested_model"] == "jev-1.13.0")
    assert jev["usage_totals"]["cost"] is not None
    assert jev["stage_count"] == 1


def test_run_preflight_from_env_records_missing_credentials_as_failure():
    # No keys in the environment: every provider is reported as a failure,
    # which blocks bulk execution rather than raising.
    result = run_preflight_from_env({})
    assert result.ok is False
    assert result.covered_models == set(spec.MODEL_IDS)
    assert all(p.ok is False for p in result.providers)
