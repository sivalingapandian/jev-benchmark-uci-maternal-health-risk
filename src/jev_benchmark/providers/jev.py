"""Jev adapter built on the TypeSafe ``system_one`` choice question type.

Pinned to ``jev-1.13.0``. Jev returns a typed choice with confidence and a full
probability map and exposes token usage but no billed-dollar amount. The exposed
response model is checked against the requested one so a silent substitution
cannot be scored.
"""

from __future__ import annotations

from ..errors import (
    InvalidDecisionError,
    ModelMismatchError,
    ProviderRequestError,
)
from .base import ProviderAdapter, StageDecision, Usage, validate_choice_response

DEFAULT_MODEL = "jev-1.13.0"


class JevAdapter(ProviderAdapter):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model_id: str = DEFAULT_MODEL,
        client: object | None = None,
    ) -> None:
        self.model_id = model_id
        if client is not None:
            self._client = client
            self._owns_client = False
            return
        import typesafe_sdk as ts

        if not api_key:
            raise ProviderRequestError("Jev adapter requires TYPESAFE_API_KEY")
        self._client = ts.TypeSafeClient(api_key=api_key, model=model_id)
        self._owns_client = True

    def decide(self, stage: dict, state: dict) -> StageDecision:
        import typesafe_sdk as ts

        question = ts.Choice(
            instructions=stage["instructions"],
            criteria={choice: None for choice in stage["choices"]},
        )
        try:
            response = self._client.system_one(
                state=state,
                questions={stage["id"]: question},
                model=self.model_id,
            )
        except ts.TypeSafeAPIResponseValidationError as exc:
            raise InvalidDecisionError(f"Jev response failed validation: {exc}") from exc
        except ts.TypeSafeError as exc:
            raise ProviderRequestError(f"Jev request failed: {exc}") from exc

        actual_model = getattr(response, "model", None)
        usage = _usage_from_response(response)

        if actual_model != self.model_id:
            raise ModelMismatchError(
                f"requested model {self.model_id!r} but Jev returned {actual_model!r}",
                usage=usage,
                actual_model=actual_model,
            )

        answer = response.answers.get(stage["id"]) if response.answers else None
        if answer is None:
            raise InvalidDecisionError(
                f"Jev response is missing an answer for stage {stage['id']!r}",
                usage=usage,
                actual_model=actual_model,
            )

        try:
            choice, confidence, probabilities = validate_choice_response(
                getattr(answer, "choice", None),
                stage["choices"],
                confidence=getattr(answer, "confidence", None),
                probabilities=getattr(answer, "probabilities", None),
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
    return Usage(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        cost=None,
    )
