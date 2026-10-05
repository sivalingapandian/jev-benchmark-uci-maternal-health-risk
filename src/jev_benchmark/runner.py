"""Sequential, budgeted, resumable runner around the shared two-stage workflow.

The runner dispatches the frozen two-stage task for a prespecified set of rows
to each provider, one paid request at a time. It wraps every provider in a
:class:`MeasuredAdapter` that adds, around the unchanged
:func:`jev_benchmark.workflow.evaluate_state` branch logic:

* a conservative budget reservation per attempt and settlement afterwards;
* bounded retries for transport/timeout failures (genuine invalid, refusal and
  identity-mismatch responses are terminal, not retried);
* client-side monotonic timing for each API attempt and the whole workflow.

Completed row-provider cases are appended to ``cases.jsonl`` and the run header
(identity, status, cumulative run time, budget snapshot) to ``run.json`` after
every case. Each *completed stage* is additionally journaled to ``stages.jsonl``
right after it succeeds and before the next stage dispatches, so an interruption
mid-workflow never silently re-bills a stage that already returned. Resume
rebuilds partial cases from the stage journal (skipping their stages), rebuilds
the ledger by replaying recorded settlements unless the ledger is already bound
to a durable shared store, skips completed cases, and refuses to continue if the
dataset or task specification changed. A fresh (non-resume) run refuses an
already-occupied directory instead of duplicating cases on top of it.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import timing, workflow
from .budget import NONE, BudgetLedger, Settlement
from .errors import (
    BudgetExceeded,
    InvalidDecisionError,
    ModelMismatchError,
    ProviderError,
    ProviderRequestError,
    RefusalError,
    RunDirectoryError,
    RunIdentityError,
)
from .providers.base import ProviderAdapter, StageDecision, Usage

# Run status values.
RUNNING = "running"
COMPLETE = "complete"
INCOMPLETE = "incomplete"
STOPPED_BUDGET = "stopped_budget"

# Which provider id maps to which cost model.
JEV_KIND = "jev"
OPENROUTER_KIND = "openrouter"


def provider_kind(model_id: str) -> str:
    return JEV_KIND if model_id == "jev-1.13.0" else OPENROUTER_KIND


def _attempt_status(exc: ProviderError) -> str:
    if isinstance(exc, RefusalError):
        return "refused"
    if isinstance(exc, ModelMismatchError):
        return "mismatch"
    if isinstance(exc, InvalidDecisionError):
        return "invalid"
    return "error"


@dataclass
class AttemptRecord:
    """One API stage attempt (a retry is its own attempt).

    ``ended_at`` is ``None`` only for an attempt interrupted before completion;
    such an attempt has no measured duration and is excluded from latency stats.
    """

    provider: str
    provider_kind: str
    stage: str
    attempt: int  # 1-based try number within this stage
    status: str  # "ok" | "error" | "invalid" | "refused" | "mismatch"
    started_at: float
    ended_at: float | None
    actual_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_basis: str  # BILLED | ESTIMATED | RESERVED_UNKNOWN | NONE
    cost_amount: float
    detail: str | None = None

    @property
    def duration(self) -> float | None:
        return timing.elapsed(self.started_at, self.ended_at)

    def settlement(self) -> Settlement:
        return Settlement(self.cost_basis, self.cost_amount)

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "provider_kind": self.provider_kind,
            "stage": self.stage,
            "attempt": self.attempt,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration": self.duration,
            "actual_model": self.actual_model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_basis": self.cost_basis,
            "cost_amount": self.cost_amount,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AttemptRecord:
        return cls(
            provider=data["provider"],
            provider_kind=data["provider_kind"],
            stage=data["stage"],
            attempt=data["attempt"],
            status=data["status"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
            actual_model=data.get("actual_model"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            cost_basis=data.get("cost_basis", NONE),
            cost_amount=data.get("cost_amount", 0.0),
            detail=data.get("detail"),
        )


@dataclass
class CaseRecord:
    """Terminal outcome and timing of one row-provider two-stage workflow."""

    row_index: int
    requested_model: str
    provider_kind: str
    outcome: str  # workflow.PREDICTED | INVALID | REFUSED | MISMATCH | ERROR
    final_label: str | None
    gold_label: str | None
    stage_count: int
    workflow_started_at: float
    workflow_ended_at: float | None
    attempts: list[AttemptRecord] = field(default_factory=list)
    detail: str | None = None

    @property
    def workflow_duration(self) -> float | None:
        return timing.elapsed(self.workflow_started_at, self.workflow_ended_at)

    @property
    def key(self) -> tuple[int, str]:
        return (self.row_index, self.requested_model)

    def settlements(self) -> list[Settlement]:
        return [a.settlement() for a in self.attempts]

    def as_dict(self) -> dict:
        return {
            "row_index": self.row_index,
            "requested_model": self.requested_model,
            "provider_kind": self.provider_kind,
            "outcome": self.outcome,
            "final_label": self.final_label,
            "gold_label": self.gold_label,
            "stage_count": self.stage_count,
            "workflow_started_at": self.workflow_started_at,
            "workflow_ended_at": self.workflow_ended_at,
            "workflow_duration": self.workflow_duration,
            "attempts": [a.as_dict() for a in self.attempts],
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CaseRecord:
        return cls(
            row_index=data["row_index"],
            requested_model=data["requested_model"],
            provider_kind=data.get("provider_kind", provider_kind(data["requested_model"])),
            outcome=data["outcome"],
            final_label=data.get("final_label"),
            gold_label=data.get("gold_label"),
            stage_count=data.get("stage_count", 0),
            workflow_started_at=data["workflow_started_at"],
            workflow_ended_at=data.get("workflow_ended_at"),
            attempts=[AttemptRecord.from_dict(a) for a in data.get("attempts", [])],
            detail=data.get("detail"),
        )


class MeasuredAdapter(ProviderAdapter):
    """Wraps a live adapter with budget reservation, retries, and timing.

    The branch logic in :func:`jev_benchmark.workflow.evaluate_state` is reused
    unchanged; it simply calls ``decide`` here, which may retry transport
    failures and records one :class:`AttemptRecord` per try. A
    :class:`~jev_benchmark.errors.BudgetExceeded` from the ledger is allowed to
    propagate (it is not a provider error) so the whole run stops.
    """

    def __init__(
        self,
        inner: ProviderAdapter,
        ledger: BudgetLedger,
        *,
        max_retries: int = 2,
        stage_sink: Callable[[StageDecision, list[AttemptRecord]], None] | None = None,
    ) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.kind = provider_kind(inner.model_id)
        self.ledger = ledger
        self.max_retries = max_retries
        self.attempts: list[AttemptRecord] = []
        # Invoked with (decision, attempts-for-this-stage) right after a stage
        # succeeds, so the runner can durably journal a completed stage before
        # the next one dispatches.
        self.stage_sink = stage_sink

    def _settle(self, token: int, usage: Usage | None) -> Settlement:
        if self.kind == JEV_KIND:
            return self.ledger.settle_estimated(token, usage)
        return self.ledger.settle_openrouter(token, usage)

    def _record(
        self,
        stage: dict,
        attempt: int,
        status: str,
        started: float,
        ended: float | None,
        settlement: Settlement,
        *,
        usage: Usage | None,
        actual_model: str | None,
        detail: str | None,
    ) -> None:
        self.attempts.append(
            AttemptRecord(
                provider=self.model_id,
                provider_kind=self.kind,
                stage=stage["id"],
                attempt=attempt,
                status=status,
                started_at=started,
                ended_at=ended,
                actual_model=actual_model,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                cost_basis=settlement.basis,
                cost_amount=settlement.amount,
                detail=detail,
            )
        )

    def decide(self, stage: dict, state: dict) -> StageDecision:
        stage_start = len(self.attempts)
        for attempt in range(1, self.max_retries + 2):
            token = self.ledger.reserve()  # BudgetExceeded propagates (stops run)
            started = timing.monotonic()
            try:
                decision = self.inner.decide(stage, state)
            except ProviderRequestError as exc:
                ended = timing.monotonic()
                settlement = self.ledger.settle_unknown(token)
                self._record(
                    stage, attempt, "error", started, ended, settlement,
                    usage=getattr(exc, "usage", None),
                    actual_model=getattr(exc, "actual_model", None),
                    detail=str(exc),
                )
                if attempt <= self.max_retries:
                    continue
                raise
            except ProviderError as exc:
                # Invalid / refusal / mismatch: a response was received, so settle
                # on its usage. These are genuine outcomes, never retried.
                ended = timing.monotonic()
                usage = getattr(exc, "usage", None)
                settlement = self._settle(token, usage)
                self._record(
                    stage, attempt, _attempt_status(exc), started, ended, settlement,
                    usage=usage,
                    actual_model=getattr(exc, "actual_model", None),
                    detail=str(exc),
                )
                raise
            else:
                ended = timing.monotonic()
                settlement = self._settle(token, decision.usage)
                self._record(
                    stage, attempt, "ok", started, ended, settlement,
                    usage=decision.usage,
                    actual_model=decision.actual_model,
                    detail=None,
                )
                if self.stage_sink is not None:
                    self.stage_sink(decision, self.attempts[stage_start:])
                return decision
        raise AssertionError("unreachable retry loop")  # pragma: no cover

    def close(self) -> None:
        self.inner.close()


@dataclass
class RunState:
    """Durable run header: identity, population, status, timing, budget."""

    run_id: str
    mode: str  # "pilot" | "run"
    config_hash: str
    ordered_record_sha256: str
    model_ids: list[str]
    planned_row_indices: list[int]
    status: str = RUNNING
    total_run_seconds: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    budget: dict = field(default_factory=dict)
    stop_reason: str | None = None

    @property
    def planned_pairs(self) -> int:
        return len(self.planned_row_indices) * len(self.model_ids)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "config_hash": self.config_hash,
            "ordered_record_sha256": self.ordered_record_sha256,
            "model_ids": list(self.model_ids),
            "planned_row_indices": list(self.planned_row_indices),
            "planned_pairs": self.planned_pairs,
            "status": self.status,
            "total_run_seconds": self.total_run_seconds,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "budget": self.budget,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RunState:
        return cls(
            run_id=data["run_id"],
            mode=data.get("mode", "run"),
            config_hash=data["config_hash"],
            ordered_record_sha256=data["ordered_record_sha256"],
            model_ids=list(data["model_ids"]),
            planned_row_indices=list(data["planned_row_indices"]),
            status=data.get("status", RUNNING),
            total_run_seconds=data.get("total_run_seconds", 0.0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            budget=data.get("budget", {}),
            stop_reason=data.get("stop_reason"),
        )


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunStore:
    """On-disk persistence for a single run directory.

    ``run.json`` holds the header and budget snapshot; ``cases.jsonl`` appends
    one completed case per line. Reading the JSONL gives both the completed-case
    set (for resume) and the full settlement history (to rebuild the ledger).
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.dir = Path(run_dir)
        self.run_path = self.dir / "run.json"
        self.cases_path = self.dir / "cases.jsonl"
        # One line per completed stage of a (row, provider) workflow, written
        # before the next stage dispatches so an interrupted multi-stage case
        # can resume without re-dispatching a stage that already succeeded.
        self.stages_path = self.dir / "stages.jsonl"

    def exists(self) -> bool:
        return self.run_path.is_file()

    def init_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> RunState:
        return RunState.from_dict(json.loads(self.run_path.read_text()))

    def save_state(self, state: RunState) -> None:
        state.updated_at = _now_iso()
        self.run_path.write_text(json.dumps(state.as_dict(), indent=2) + "\n")

    def load_cases(self) -> list[CaseRecord]:
        if not self.cases_path.is_file():
            return []
        cases: list[CaseRecord] = []
        for line in self.cases_path.read_text().splitlines():
            line = line.strip()
            if line:
                cases.append(CaseRecord.from_dict(json.loads(line)))
        return cases

    def append_case(self, case: CaseRecord) -> None:
        with self.cases_path.open("a") as handle:
            handle.write(json.dumps(case.as_dict()) + "\n")

    def append_stage(
        self,
        row_index: int,
        requested_model: str,
        decision: StageDecision,
        attempts: list[AttemptRecord],
    ) -> None:
        """Durably journal one completed stage of a (row, provider) workflow."""
        entry = {
            "row_index": row_index,
            "requested_model": requested_model,
            "stage": decision.stage,
            "decision": decision.as_dict(),
            "attempts": [a.as_dict() for a in attempts],
        }
        with self.stages_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def load_stage_entries(self) -> list[dict]:
        if not self.stages_path.is_file():
            return []
        entries: list[dict] = []
        for line in self.stages_path.read_text().splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries


def _build_case(
    row_index: int,
    gold_label: str | None,
    measured: MeasuredAdapter,
    outcome: workflow.RowOutcome,
    wf_start: float,
    wf_end: float,
    *,
    prior_attempts: list[AttemptRecord] | None = None,
) -> CaseRecord:
    # Attempts from a resumed stage (already journaled, never re-dispatched) come
    # first, then this session's new attempts, preserving stage order.
    attempts = list(prior_attempts or []) + list(measured.attempts)
    return CaseRecord(
        row_index=row_index,
        requested_model=measured.model_id,
        provider_kind=measured.kind,
        outcome=outcome.outcome,
        final_label=outcome.final_label,
        gold_label=gold_label,
        stage_count=outcome.stage_count,
        workflow_started_at=wf_start,
        workflow_ended_at=wf_end,
        attempts=attempts,
        detail=outcome.detail,
    )


@dataclass
class RunResult:
    state: RunState
    completed_cases: list[CaseRecord]
    newly_completed: int
    stopped: bool
    stop_reason: str | None
    ledger: BudgetLedger


def run(
    *,
    store: RunStore,
    adapters: dict[str, ProviderAdapter],
    rows: list[dict],  # each: {"index": int, "features": {...}, "label": str|None}
    model_ids: list[str],
    config_hash: str,
    ordered_record_sha256: str,
    ledger: BudgetLedger,
    mode: str = "run",
    run_id: str | None = None,
    max_retries: int = 2,
    resume: bool = False,
) -> RunResult:
    """Dispatch the two-stage workflow sequentially for every (row, provider).

    Row-major, frozen order. Already-completed pairs are skipped on resume. A
    budget refusal stops the run with status ``stopped_budget``; completed cases
    stay recorded and are never re-billed. Returns the final state and cases.
    """
    # A fresh run must never be started on top of an existing one: that would
    # duplicate cases and overwrite the prior header. The caller must pick a new
    # directory or pass resume=True.
    if not resume and store.exists():
        raise RunDirectoryError(
            f"run directory {store.dir} is already occupied by a previous run; "
            "choose a fresh directory or resume the existing run"
        )

    store.init_dir()

    partial_stages: dict[tuple[int, str], dict[str, tuple[StageDecision, list[AttemptRecord]]]] = {}
    if resume and store.exists():
        state = store.load_state()
        if state.config_hash != config_hash:
            raise RunIdentityError(
                "cannot resume: frozen configuration hash changed "
                f"(run={state.config_hash!r} current={config_hash!r})"
            )
        if state.ordered_record_sha256 != ordered_record_sha256:
            raise RunIdentityError(
                "cannot resume: dataset ordered-record hash changed "
                f"(run={state.ordered_record_sha256!r} current={ordered_record_sha256!r})"
            )
        completed = store.load_cases()
        done_keys_resume = {c.key for c in completed}
        # Rebuild any partially-completed cases (a stage that succeeded before an
        # interruption) from the stage journal, skipping pairs already terminal.
        for entry in store.load_stage_entries():
            key = (entry["row_index"], entry["requested_model"])
            if key in done_keys_resume:
                continue
            decision = StageDecision.from_dict(entry["decision"])
            attempts = [AttemptRecord.from_dict(a) for a in entry.get("attempts", [])]
            partial_stages.setdefault(key, {})[decision.stage] = (decision, attempts)
        if ledger.store is None:
            # Unbound ledger: rebuild accounting by replaying every durable
            # settlement (completed cases plus journaled partial stages). A bound
            # shared ledger is already authoritative, so it must not be replayed
            # (its settlements were flushed during the interrupted session).
            settlements: list[Settlement] = [s for c in completed for s in c.settlements()]
            for stages in partial_stages.values():
                for _decision, attempts in stages.values():
                    settlements.extend(a.settlement() for a in attempts)
            ledger = BudgetLedger.replay(
                settlements,
                cap=ledger.cap,
                pricing=ledger.pricing,
                planning_seed=ledger.planning_seed,
            )
        # Grow (never shrink) the planned population to include any rows this
        # resume adds, so the completion check uses the current intended scope.
        merged = sorted(set(state.planned_row_indices) | {r["index"] for r in rows})
        state.planned_row_indices = merged
    else:
        state = RunState(
            run_id=run_id or _dt.datetime.now(_dt.UTC).strftime("run-%Y%m%dT%H%M%SZ"),
            mode=mode,
            config_hash=config_hash,
            ordered_record_sha256=ordered_record_sha256,
            model_ids=list(model_ids),
            planned_row_indices=[r["index"] for r in rows],
            created_at=_now_iso(),
        )
        # A fresh run starts from zero. Continuing prior work requires resume=True
        # so completed cases and their accounting are replayed, never silently
        # mixed with a new ledger.
        completed = []

    done_keys = {c.key for c in completed}
    state.status = RUNNING
    state.budget = ledger.as_dict()
    store.save_state(state)

    session_start = timing.monotonic()
    newly_completed = 0
    stopped = False
    stop_reason: str | None = None

    try:
        for row in rows:
            for model_id in model_ids:
                key = (row["index"], model_id)
                if key in done_keys:
                    continue
                inner = adapters[model_id]
                prior = partial_stages.get(key, {})
                prior_decisions = {sid: dec for sid, (dec, _atts) in prior.items()}
                prior_attempts = [a for _sid, (_dec, atts) in prior.items() for a in atts]

                def _sink(decision: StageDecision, attempts: list[AttemptRecord],
                          _row=row["index"], _model=model_id) -> None:
                    store.append_stage(_row, _model, decision, attempts)

                measured = MeasuredAdapter(
                    inner, ledger, max_retries=max_retries, stage_sink=_sink
                )
                wf_start = timing.monotonic()
                try:
                    outcome = workflow.evaluate_state(
                        measured, row["features"], prior_stages=prior_decisions
                    )
                except BudgetExceeded as exc:
                    stopped = True
                    stop_reason = str(exc)
                    state.status = STOPPED_BUDGET
                    state.stop_reason = stop_reason
                    raise
                wf_end = timing.monotonic()
                case = _build_case(
                    row["index"], row.get("label"), measured, outcome, wf_start, wf_end,
                    prior_attempts=prior_attempts,
                )
                store.append_case(case)
                completed.append(case)
                done_keys.add(key)
                newly_completed += 1
                state.budget = ledger.as_dict()
                state.total_run_seconds += timing.monotonic() - session_start
                session_start = timing.monotonic()
                store.save_state(state)
    except BudgetExceeded:
        state.total_run_seconds += timing.monotonic() - session_start
        state.budget = ledger.as_dict()
        store.save_state(state)
        return RunResult(state, completed, newly_completed, stopped, stop_reason, ledger)

    state.total_run_seconds += timing.monotonic() - session_start
    if len(done_keys) >= state.planned_pairs:
        state.status = COMPLETE
    else:
        state.status = INCOMPLETE
    state.budget = ledger.as_dict()
    store.save_state(state)
    return RunResult(state, completed, newly_completed, stopped, stop_reason, ledger)
