"""Deterministic offline fault-path tests for the provider adapters.

These exercise schema validation, model-identity checks, refusals, truncation,
and transport failures with injected fake clients. They complement — and do not
replace — the live three-provider preflight.
"""

import json
from types import SimpleNamespace

import pytest

from jev_benchmark import spec
from jev_benchmark.errors import (
    InvalidDecisionError,
    ModelMismatchError,
    ProviderRequestError,
    RefusalError,
)
from jev_benchmark.providers import JevAdapter, OpenRouterAdapter
from jev_benchmark.providers.base import validate_choice_response

# A complete synthetic (non-patient) state; the OpenRouter adapter serializes
# all six permitted measurements into its prompt.
STATE = {
    "Age": 30,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "BS": 7.5,
    "BodyTemp": 98.0,
    "HeartRate": 76,
}


# --------------------------------------------------------------------------- #
# Shared choice-response validation
# --------------------------------------------------------------------------- #
def test_validate_choice_accepts_valid_choice_and_probabilities():
    choice, conf, probs = validate_choice_response(
        "high",
        ("high", "other"),
        confidence=0.9,
        probabilities={"high": 0.8, "other": 0.2},
    )
    assert choice == "high"
    assert conf == 0.9
    assert probs == {"high": 0.8, "other": 0.2}


def test_validate_choice_rejects_choice_outside_allowed():
    with pytest.raises(InvalidDecisionError):
        validate_choice_response("maybe", ("high", "other"))


def test_malformed_probabilities_are_optional_by_default():
    # A valid choice is preserved even when probabilities are not well-formed;
    # probability fields are not mandatory solely for reporting calibration.
    choice, conf, probs = validate_choice_response(
        "high", ("high", "other"), probabilities={"high": "oops"}
    )
    assert choice == "high"
    assert probs is None


def test_malformed_probabilities_fail_when_explicitly_required():
    with pytest.raises(InvalidDecisionError):
        validate_choice_response(
            "high", ("high", "other"), probabilities=None, require_probabilities=True
        )


def test_confidence_out_of_range_becomes_none():
    _, conf, _ = validate_choice_response("high", ("high", "other"), confidence=5.0)
    assert conf is None


# --------------------------------------------------------------------------- #
# Jev adapter
# --------------------------------------------------------------------------- #
class FakeJevClient:
    def __init__(self, answers, *, model="jev-1.13.0", usage=(10, 2), raise_exc=None):
        self._answers = answers
        self._model = model
        self._usage = usage
        self._raise_exc = raise_exc
        self.calls = []

    def system_one(self, state, questions, model=None):
        ((stage_id, _question),) = questions.items()
        self.calls.append((stage_id, dict(state), model))
        if self._raise_exc is not None:
            raise self._raise_exc
        choice, conf, probs = self._answers[stage_id]
        answer = SimpleNamespace(choice=choice, confidence=conf, probabilities=probs)
        usage = SimpleNamespace(input_tokens=self._usage[0], output_tokens=self._usage[1])
        return SimpleNamespace(model=self._model, usage=usage, answers={stage_id: answer})


def test_jev_returns_validated_decision_and_usage():
    client = FakeJevClient({"stage1": ("high", 0.9, {"high": 0.9, "other": 0.1})})
    adapter = JevAdapter(client=client, model_id="jev-1.13.0")
    decision = adapter.decide(spec.STAGE1, {"Age": 30})
    assert decision.choice == "high"
    assert decision.actual_model == "jev-1.13.0"
    assert decision.usage.input_tokens == 10
    assert decision.usage.output_tokens == 2
    assert decision.usage.cost is None  # Jev has no billed-dollar field


def test_jev_rejects_model_identity_mismatch():
    client = FakeJevClient({"stage1": ("high", 0.9, None)}, model="jev-9.9.9")
    adapter = JevAdapter(client=client, model_id="jev-1.13.0")
    with pytest.raises(ModelMismatchError) as exc:
        adapter.decide(spec.STAGE1, {"Age": 30})
    assert exc.value.actual_model == "jev-9.9.9"


def test_jev_rejects_choice_outside_criteria():
    client = FakeJevClient({"stage1": ("maybe", 0.5, None)})
    adapter = JevAdapter(client=client, model_id="jev-1.13.0")
    with pytest.raises(InvalidDecisionError):
        adapter.decide(spec.STAGE1, {"Age": 30})


def test_jev_maps_sdk_error_to_request_error():
    import typesafe_sdk as ts

    client = FakeJevClient({}, raise_exc=ts.TypeSafeAPIConnectionError("boom"))
    adapter = JevAdapter(client=client, model_id="jev-1.13.0")
    with pytest.raises(ProviderRequestError):
        adapter.decide(spec.STAGE1, {"Age": 30})


# --------------------------------------------------------------------------- #
# OpenRouter adapter
# --------------------------------------------------------------------------- #
def _make_openrouter_response(
    *, content, model="openai/gpt-4o-mini", finish_reason="stop", refusal=None, cost=0.0000012
):
    message = SimpleNamespace(content=content, refusal=refusal)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=20, completion_tokens=5, cost=cost)
    return SimpleNamespace(model=model, usage=usage, choices=[choice])


class FakeOpenRouterClient:
    def __init__(self, response=None, *, raise_exc=None):
        self._response = response
        self._raise_exc = raise_exc
        self.create_kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.create_kwargs = kwargs
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response


def test_openrouter_returns_validated_decision_and_cost():
    payload = json.dumps(
        {"choice": "other", "confidence": 0.7, "probabilities": {"high": 0.3, "other": 0.7}}
    )
    client = FakeOpenRouterClient(_make_openrouter_response(content=payload))
    adapter = OpenRouterAdapter("openai/gpt-4o-mini", client=client)
    decision = adapter.decide(spec.STAGE1, STATE)
    assert decision.choice == "other"
    assert decision.actual_model == "openai/gpt-4o-mini"
    assert decision.usage.input_tokens == 20
    assert decision.usage.cost == pytest.approx(0.0000012)
    # Structured output and provider-parameter support were requested.
    body = client.create_kwargs["extra_body"]
    assert body["provider"]["require_parameters"] is True
    assert body["usage"]["include"] is True
    assert client.create_kwargs["response_format"]["type"] == "json_schema"


def test_openrouter_never_sends_risklevel():
    payload = json.dumps(
        {"choice": "high", "confidence": 0.9, "probabilities": {"high": 0.9, "other": 0.1}}
    )
    client = FakeOpenRouterClient(_make_openrouter_response(content=payload))
    adapter = OpenRouterAdapter("openai/gpt-4o-mini", client=client)
    adapter.decide(spec.STAGE1, {**STATE, "RiskLevel": "high risk"})
    user_message = client.create_kwargs["messages"][-1]["content"]
    # The gold label field is dropped: it never appears in the serialized
    # measurements block sent to the provider.
    measurements_block = user_message.split("Record measurements (JSON): ", 1)[1]
    assert "RiskLevel" not in measurements_block
    assert "high risk" not in measurements_block


def test_openrouter_accepts_dated_snapshot_of_requested_model():
    payload = json.dumps(
        {"choice": "high", "confidence": 0.9, "probabilities": {"high": 0.9, "other": 0.1}}
    )
    resp = _make_openrouter_response(content=payload, model="openai/gpt-4o-mini-2024-07-18")
    adapter = OpenRouterAdapter("openai/gpt-4o-mini", client=FakeOpenRouterClient(resp))
    decision = adapter.decide(spec.STAGE1, STATE)
    assert decision.actual_model == "openai/gpt-4o-mini-2024-07-18"


def test_openrouter_rejects_substituted_model():
    payload = json.dumps(
        {"choice": "high", "confidence": 0.9, "probabilities": {"high": 0.9, "other": 0.1}}
    )
    resp = _make_openrouter_response(content=payload, model="anthropic/claude-3-haiku")
    adapter = OpenRouterAdapter("openai/gpt-4o-mini", client=FakeOpenRouterClient(resp))
    with pytest.raises(ModelMismatchError):
        adapter.decide(spec.STAGE1, STATE)


def test_openrouter_refusal_becomes_refusal_error():
    resp = _make_openrouter_response(content=None, refusal="I cannot help with that.")
    adapter = OpenRouterAdapter("openai/gpt-4o-mini", client=FakeOpenRouterClient(resp))
    with pytest.raises(RefusalError):
        adapter.decide(spec.STAGE1, STATE)


def test_openrouter_truncated_response_is_invalid():
    payload = json.dumps({"choice": "high"})
    resp = _make_openrouter_response(content=payload, finish_reason="length")
    adapter = OpenRouterAdapter("openai/gpt-4o-mini", client=FakeOpenRouterClient(resp))
    with pytest.raises(InvalidDecisionError):
        adapter.decide(spec.STAGE1, STATE)


def test_openrouter_non_json_content_is_invalid():
    resp = _make_openrouter_response(content="not json at all")
    adapter = OpenRouterAdapter("openai/gpt-4o-mini", client=FakeOpenRouterClient(resp))
    with pytest.raises(InvalidDecisionError):
        adapter.decide(spec.STAGE1, STATE)


def test_openrouter_choice_outside_enum_is_invalid():
    payload = json.dumps(
        {"choice": "maybe", "confidence": 0.5, "probabilities": {"high": 0.5, "other": 0.5}}
    )
    resp = _make_openrouter_response(content=payload)
    adapter = OpenRouterAdapter("openai/gpt-4o-mini", client=FakeOpenRouterClient(resp))
    with pytest.raises(InvalidDecisionError):
        adapter.decide(spec.STAGE1, STATE)


def test_openrouter_transport_error_becomes_request_error():
    from openai import OpenAIError

    client = FakeOpenRouterClient(raise_exc=OpenAIError("network down"))
    adapter = OpenRouterAdapter("openai/gpt-4o-mini", client=client)
    with pytest.raises(ProviderRequestError):
        adapter.decide(spec.STAGE1, STATE)
