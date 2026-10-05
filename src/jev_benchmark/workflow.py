"""Shared two-stage decision workflow.

Every provider runs the same frozen task against the same six measurements:
stage 1 asks ``high`` vs ``other``; only the provider's *own* ``other`` answer
triggers its stage 2 (``mid`` vs ``low``). The gold label never enters here and
cannot select a branch. Invalid, refused, or failed responses become explicit
outcomes with no fabricated label and no unearned stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import spec
from .errors import (
    InvalidDecisionError,
    ModelMismatchError,
    ProviderError,
    ProviderRequestError,
    RefusalError,
)
from .providers.base import ProviderAdapter, StageDecision, Usage

# Final per-row outcome categories.
PREDICTED = "predicted"
INVALID = "invalid"
REFUSED = "refused"
MISMATCH = "mismatch"
ERROR = "error"


def _status_for_error(exc: ProviderError) -> str:
    if isinstance(exc, RefusalError):
        return REFUSED
    if isinstance(exc, ModelMismatchError):
        return MISMATCH
    if isinstance(exc, InvalidDecisionError):
        return INVALID
    if isinstance(exc, ProviderRequestError):
        return ERROR
    return ERROR


@dataclass
class StageRecord:
    """What happened on one stage request: a usable decision or a typed failure."""

    stage: str
    requested_model: str
    actual_model: str | None
    status: str  # "ok" | "invalid" | "refused" | "mismatch" | "error"
    choice: str | None
    confidence: float | None
    probabilities: dict[str, float] | None
    usage: Usage
    detail: str | None = None

    @classmethod
    def from_decision(cls, decision: StageDecision) -> StageRecord:
        return cls(
            stage=decision.stage,
            requested_model=decision.requested_model,
            actual_model=decision.actual_model,
            status="ok",
            choice=decision.choice,
            confidence=decision.confidence,
            probabilities=decision.probabilities,
            usage=decision.usage,
        )

    @classmethod
    def from_error(cls, stage: dict, requested_model: str, exc: ProviderError) -> StageRecord:
        usage = exc.usage if isinstance(getattr(exc, "usage", None), Usage) else Usage()
        return cls(
            stage=stage["id"],
            requested_model=requested_model,
            actual_model=getattr(exc, "actual_model", None),
            status=_status_for_error(exc),
            choice=None,
            confidence=None,
            probabilities=None,
            usage=usage,
            detail=str(exc),
        )

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "requested_model": self.requested_model,
            "actual_model": self.actual_model,
            "status": self.status,
            "choice": self.choice,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "usage": self.usage.as_dict(),
            "detail": self.detail,
        }


@dataclass
class RowOutcome:
    """Terminal outcome of the whole two-stage workflow for one provider/row."""

    requested_model: str
    actual_model: str | None
    outcome: str  # PREDICTED | INVALID | REFUSED | MISMATCH | ERROR
    final_label: str | None
    stages: list[StageRecord] = field(default_factory=list)
    detail: str | None = None

    @property
    def usages(self) -> list[Usage]:
        return [s.usage for s in self.stages]

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    def as_dict(self) -> dict:
        return {
            "requested_model": self.requested_model,
            "actual_model": self.actual_model,
            "outcome": self.outcome,
            "final_label": self.final_label,
            "stages": [s.as_dict() for s in self.stages],
            "detail": self.detail,
        }


def evaluate_state(
    adapter: ProviderAdapter,
    state: dict,
    prior_stages: dict | None = None,
) -> RowOutcome:
    """Run the two-stage workflow for one provider against the six measurements.

    ``state`` carries only the permitted measurements; no label is ever passed.
    Stage 2 runs only when this provider's own stage-1 answer is ``other``.

    ``prior_stages`` optionally maps a stage id (``"stage1"``/``"stage2"``) to a
    previously-completed :class:`StageDecision` recovered from a run's stage
    journal. A stage present there is *not* re-dispatched (so a crash-resumed
    workflow never re-bills a stage that already succeeded); only missing stages
    are sent to the provider.
    """
    prior_stages = prior_stages or {}
    stages: list[StageRecord] = []

    # Stage 1.
    prior_d1 = prior_stages.get(spec.STAGE1["id"])
    if prior_d1 is not None:
        d1 = prior_d1
        stages.append(StageRecord.from_decision(d1))
    else:
        try:
            d1 = adapter.decide(spec.STAGE1, state)
        except ProviderError as exc:
            rec = StageRecord.from_error(spec.STAGE1, adapter.model_id, exc)
            stages.append(rec)
            return RowOutcome(
                requested_model=adapter.model_id,
                actual_model=rec.actual_model,
                outcome=rec.status,
                final_label=None,
                stages=stages,
                detail=rec.detail,
            )
        stages.append(StageRecord.from_decision(d1))

    if d1.choice == "high":
        return RowOutcome(
            requested_model=adapter.model_id,
            actual_model=d1.actual_model,
            outcome=PREDICTED,
            final_label=spec.compose_label("high"),
            stages=stages,
        )

    # Only this provider's own ``other`` reaches here and earns stage 2.
    prior_d2 = prior_stages.get(spec.STAGE2["id"])
    if prior_d2 is not None:
        stages.append(StageRecord.from_decision(prior_d2))
        return RowOutcome(
            requested_model=adapter.model_id,
            actual_model=prior_d2.actual_model,
            outcome=PREDICTED,
            final_label=spec.compose_label("other", prior_d2.choice),
            stages=stages,
        )
    try:
        d2 = adapter.decide(spec.STAGE2, state)
    except ProviderError as exc:
        rec = StageRecord.from_error(spec.STAGE2, adapter.model_id, exc)
        stages.append(rec)
        return RowOutcome(
            requested_model=adapter.model_id,
            actual_model=d1.actual_model,
            outcome=rec.status,
            final_label=None,
            stages=stages,
            detail=rec.detail,
        )

    stages.append(StageRecord.from_decision(d2))
    return RowOutcome(
        requested_model=adapter.model_id,
        actual_model=d2.actual_model,
        outcome=PREDICTED,
        final_label=spec.compose_label("other", d2.choice),
        stages=stages,
    )
