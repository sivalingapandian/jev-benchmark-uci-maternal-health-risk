"""Shared adapter contract and response validation.

A provider adapter answers exactly one frozen stage question at a time. It
returns a :class:`StageDecision` only when the response is well-formed and the
exposed model identity matches what was requested; otherwise it raises a typed
:class:`~jev_benchmark.errors.ProviderError`. Keeping validation here (rather
than in each adapter) means Jev and OpenRouter apply the same rules for what a
usable decision is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..errors import InvalidDecisionError


@dataclass(frozen=True)
class Usage:
    """Redacted usage for one stage request.

    ``cost`` is OpenRouter's billed amount when the response reports it; Jev has
    no billed-dollar field, so it stays ``None`` and spend is estimated from
    published token rates elsewhere.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> Usage:
        if not data:
            return cls()
        return cls(
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            cost=data.get("cost"),
        )


@dataclass(frozen=True)
class StageDecision:
    """A validated choice for one stage of the two-stage task."""

    stage: str
    requested_model: str
    actual_model: str | None
    choice: str
    confidence: float | None
    probabilities: dict[str, float] | None
    usage: Usage

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "requested_model": self.requested_model,
            "actual_model": self.actual_model,
            "choice": self.choice,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "usage": self.usage.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> StageDecision:
        return cls(
            stage=data["stage"],
            requested_model=data["requested_model"],
            actual_model=data.get("actual_model"),
            choice=data["choice"],
            confidence=data.get("confidence"),
            probabilities=data.get("probabilities"),
            usage=Usage.from_dict(data.get("usage")),
        )


def _coerce_confidence(value: object) -> float | None:
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if 0.0 <= conf <= 1.0:
        return conf
    return None


def _coerce_probabilities(
    value: object, allowed: tuple[str, ...]
) -> dict[str, float] | None:
    """Return a clean {choice: prob} map, or ``None`` when not well-formed.

    A malformed probability block does not, on its own, invalidate an otherwise
    valid choice (see ``require_probabilities`` in
    :func:`validate_choice_response`); callers that only report calibration keep
    the valid choice and simply record ``None`` here.
    """
    if not isinstance(value, dict):
        return None
    cleaned: dict[str, float] = {}
    for key in allowed:
        if key not in value:
            return None
        try:
            prob = float(value[key])
        except (TypeError, ValueError):
            return None
        if not 0.0 <= prob <= 1.0:
            return None
        cleaned[key] = prob
    return cleaned


def validate_choice_response(
    raw_choice: object,
    allowed: tuple[str, ...],
    *,
    confidence: object = None,
    probabilities: object = None,
    require_probabilities: bool = False,
) -> tuple[str, float | None, dict[str, float] | None]:
    """Validate a decision payload against a stage's allowed choices.

    The ``choice`` is always mandatory and must be one of ``allowed``. Confidence
    and probabilities are recorded when well-formed. They are treated as optional
    for label selection unless ``require_probabilities`` is set, which lets a
    caller that genuinely needs probabilities turn a missing/invalid block into an
    invalid outcome rather than silently accepting it.
    """
    if not isinstance(raw_choice, str) or raw_choice not in allowed:
        raise InvalidDecisionError(
            f"choice {raw_choice!r} is not one of the allowed choices {list(allowed)}"
        )
    probs = _coerce_probabilities(probabilities, allowed)
    if require_probabilities and probs is None:
        raise InvalidDecisionError(
            "required probabilities are missing or not well-formed for all choices"
        )
    conf = _coerce_confidence(confidence)
    return raw_choice, conf, probs


class ProviderAdapter(ABC):
    """A thin live adapter that answers one bounded stage question at a time."""

    model_id: str

    @abstractmethod
    def decide(self, stage: dict, state: dict) -> StageDecision:
        """Call the provider for one stage and return a validated decision.

        ``stage`` is :data:`jev_benchmark.spec.STAGE1` or ``STAGE2``; ``state`` is
        the six permitted measurements. Raises a
        :class:`~jev_benchmark.errors.ProviderError` subclass on any transport
        failure, refusal, identity mismatch, or unusable response.
        """

    def close(self) -> None:  # noqa: B027 - optional no-op hook, not abstract
        """Release any underlying client resources (optional override)."""
        return None
