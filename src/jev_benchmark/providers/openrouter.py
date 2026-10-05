"""Direct structured-output adapter for OpenRouter-served LLMs.

Pinned to ``openai/gpt-4o-mini`` and ``google/gemini-2.5-flash``. Requests use
an OpenAI-compatible client against OpenRouter with a strict JSON schema and the
``require_parameters`` provider flag, so routing only reaches a backend that
honours structured output. The schema is still validated locally, the exposed
model identity is checked, and truncated or refused responses become explicit
invalid/refusal outcomes instead of fabricated labels.
"""

from __future__ import annotations

import json

from .. import spec
from ..errors import (
    InvalidDecisionError,
    ModelMismatchError,
    ProviderRequestError,
    RefusalError,
)
from .base import ProviderAdapter, StageDecision, Usage, validate_choice_response

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_MODELS = ("openai/gpt-4o-mini", "google/gemini-2.5-flash")

_SYSTEM_PROMPT = (
    "You are a careful classifier. Use only the provided measurements. "
    "Return a single structured JSON object matching the requested schema and "
    "nothing else."
)


def _choice_schema(choices: tuple[str, ...]) -> dict:
    """Strict JSON schema for a bounded choice with confidence and probabilities."""
    return {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": list(choices)},
            "confidence": {"type": "number"},
            "probabilities": {
                "type": "object",
                "properties": {c: {"type": "number"} for c in choices},
                "required": list(choices),
                "additionalProperties": False,
            },
        },
        "required": ["choice", "confidence", "probabilities"],
        "additionalProperties": False,
    }


def _model_matches(requested: str, actual: str | None) -> bool:
    if not actual:
        # Identity not exposed: it cannot be checked, so it is not a mismatch.
        return True
    a = actual.strip().lower()
    r = requested.strip().lower()
    if a == r:
        return True
    # Accept only a dated/versioned snapshot of the *same* model: the requested
    # id followed by a separator and a digit (e.g. "...-2024-07-18"). A lexical
    # variant such as "...-lite" or "...-high" is a different model and must be
    # rejected rather than matched as an arbitrary prefix.
    for sep in ("-", ":", "@"):
        prefix = r + sep
        if a.startswith(prefix) and a[len(prefix) : len(prefix) + 1].isdigit():
            return True
    return False


class OpenRouterAdapter(ProviderAdapter):
    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        client: object | None = None,
    ) -> None:
        self.model_id = model_id
        if client is not None:
            self._client = client
            self._owns_client = False
            return
        from openai import OpenAI

        if not api_key:
            raise ProviderRequestError("OpenRouter adapter requires OPENROUTER_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._owns_client = True

    def decide(self, stage: dict, state: dict) -> StageDecision:
        from openai import OpenAIError

        prompt = spec.build_user_prompt(stage, state)
        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{stage['id']}_decision",
                        "strict": True,
                        "schema": _choice_schema(stage["choices"]),
                    },
                },
                extra_body={
                    "provider": {"require_parameters": True},
                    "usage": {"include": True},
                },
            )
        except OpenAIError as exc:
            raise ProviderRequestError(f"OpenRouter request failed: {exc}") from exc

        actual_model = getattr(response, "model", None)
        usage = _usage_from_response(response)

        if not _model_matches(self.model_id, actual_model):
            raise ModelMismatchError(
                f"requested model {self.model_id!r} but OpenRouter returned {actual_model!r}",
                usage=usage,
                actual_model=actual_model,
            )

        choices = getattr(response, "choices", None)
        if not choices:
            raise InvalidDecisionError(
                "OpenRouter response contained no choices",
                usage=usage,
                actual_model=actual_model,
            )
        top = choices[0]
        message = getattr(top, "message", None)
        finish_reason = getattr(top, "finish_reason", None)

        refusal = getattr(message, "refusal", None) if message else None
        if refusal:
            raise RefusalError(
                f"model refused the bounded question: {refusal}",
                usage=usage,
                actual_model=actual_model,
            )
        if finish_reason in ("length", "content_filter"):
            raise InvalidDecisionError(
                f"response was incomplete (finish_reason={finish_reason!r})",
                usage=usage,
                actual_model=actual_model,
            )

        content = getattr(message, "content", None) if message else None
        if not content:
            raise InvalidDecisionError(
                "OpenRouter response had empty content",
                usage=usage,
                actual_model=actual_model,
            )
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise InvalidDecisionError(
                f"response content was not valid JSON: {exc}",
                usage=usage,
                actual_model=actual_model,
            ) from exc
        if not isinstance(payload, dict):
            raise InvalidDecisionError(
                "response JSON was not an object",
                usage=usage,
                actual_model=actual_model,
            )

        try:
            # The structured schema marks ``probabilities`` required, so the
            # local parser requires them too: the schema and parser agree, and a
            # response missing/malforming required output is an invalid outcome
            # rather than a usable label.
            choice, confidence, probabilities = validate_choice_response(
                payload.get("choice"),
                stage["choices"],
                confidence=payload.get("confidence"),
                probabilities=payload.get("probabilities"),
                require_probabilities=True,
            )
        except InvalidDecisionError as exc:
            exc.usage = usage
            exc.actual_model = actual_model
            raise

        return StageDecision(
            stage=stage["id"],
            requested_model=self.model_id,
            actual_model=actual_model,
            choice=choice,
            confidence=confidence,
            probabilities=probabilities,
            usage=usage,
        )

    def close(self) -> None:
        if getattr(self, "_owns_client", False):
            close = getattr(self._client, "close", None)
            if callable(close):
                close()


def _usage_from_response(response: object) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()
    cost = getattr(usage, "cost", None)
    if cost is None and hasattr(usage, "model_extra"):
        extra = usage.model_extra or {}
        cost = extra.get("cost")
    try:
        cost = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost = None
    return Usage(
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        cost=cost,
    )
