"""Offline inspection of a saved run or pilot directory.

Reads ``run.json`` and ``cases.jsonl`` without making any paid request and
summarises, per provider and overall:

* outcome counts (predicted/invalid/refused/mismatch/error) and completion;
* conservative accounting split into billed OpenRouter, rate-estimated Jev, and
  retained-unknown exposure, plus the planning seed and remaining budget;
* client-side latency p50/p95 with explicit sample counts at two scopes - each
  API attempt and each terminal row-provider workflow - and the total run time.

A run that has not covered every planned pair is reported as explicitly
incomplete so partial results are never mistaken for a finished benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import timing, workflow
from .budget import BudgetLedger, Settlement
from .runner import COMPLETE, CaseRecord, RunState, RunStore

OUTCOMES = (
    workflow.PREDICTED,
    workflow.INVALID,
    workflow.REFUSED,
    workflow.MISMATCH,
    workflow.ERROR,
)


@dataclass
class ScopeLatency:
    api_attempt: dict
    workflow: dict

    def as_dict(self) -> dict:
        return {"api_attempt": self.api_attempt, "workflow": self.workflow}


@dataclass
class ProviderInspection:
    model_id: str
    attempted_cases: int
    outcome_counts: dict
    api_attempts: int
    retried_attempts: int
    failed_attempts: int
    latency: ScopeLatency
    accounting: dict

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "attempted_cases": self.attempted_cases,
            "outcome_counts": self.outcome_counts,
            "api_attempts": self.api_attempts,
            "retried_attempts": self.retried_attempts,
            "failed_attempts": self.failed_attempts,
            "latency": self.latency.as_dict(),
            "accounting": self.accounting,
        }


@dataclass
class RunInspection:
    run_id: str
    mode: str
    status: str
    complete: bool
    config_hash: str
    ordered_record_sha256: str
    planned_pairs: int
    completed_pairs: int
    total_run_seconds: float
    accounting: dict
    overall_latency: ScopeLatency
    providers: list[ProviderInspection] = field(default_factory=list)
    stop_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": self.status,
            "complete": self.complete,
            "config_hash": self.config_hash,
            "ordered_record_sha256": self.ordered_record_sha256,
            "planned_pairs": self.planned_pairs,
            "completed_pairs": self.completed_pairs,
            "total_run_seconds": self.total_run_seconds,
            "accounting": self.accounting,
            "overall_latency": self.overall_latency.as_dict(),
            "providers": [p.as_dict() for p in self.providers],
            "stop_reason": self.stop_reason,
        }


def _accounting_from_cases(cases: list[CaseRecord], state: RunState) -> tuple[BudgetLedger, dict]:
    settlements: list[Settlement] = [s for c in cases for s in c.settlements()]
    budget = state.budget or {}
    ledger = BudgetLedger.replay(
        settlements,
        cap=budget.get("cap", 20.0),
        planning_seed=budget.get("planning_seed", 0.10),
    )
    return ledger, ledger.as_dict()


def _latency(cases: list[CaseRecord]) -> ScopeLatency:
    attempt_durations = [a.duration for c in cases for a in c.attempts]
    workflow_durations = [c.workflow_duration for c in cases]
    return ScopeLatency(
        api_attempt=timing.summarize(attempt_durations),
        workflow=timing.summarize(workflow_durations),
    )


def _provider_inspection(model_id: str, cases: list[CaseRecord]) -> ProviderInspection:
    subset = [c for c in cases if c.requested_model == model_id]
    outcome_counts = {o: 0 for o in OUTCOMES}
    for case in subset:
        outcome_counts[case.outcome] = outcome_counts.get(case.outcome, 0) + 1

    attempts = [a for c in subset for a in c.attempts]
    retried = sum(1 for a in attempts if a.attempt > 1)
    failed = sum(1 for a in attempts if a.status != "ok")

    # Per-provider accounting omits the shared seed (seed is run-level only).
    accounting = {
        "billed_openrouter": sum(
            a.cost_amount for a in attempts if a.cost_basis == "billed"
        ),
        "estimated_jev": sum(
            a.cost_amount for a in attempts if a.cost_basis == "estimated"
        ),
        "reserved_unknown": sum(
            a.cost_amount for a in attempts if a.cost_basis == "reserved_unknown"
        ),
    }

    return ProviderInspection(
        model_id=model_id,
        attempted_cases=len(subset),
        outcome_counts=outcome_counts,
        api_attempts=len(attempts),
        retried_attempts=retried,
        failed_attempts=failed,
        latency=_latency(subset),
        accounting=accounting,
    )


def inspect_run(store: RunStore) -> RunInspection:
    """Build an offline inspection summary from a saved run directory."""
    state = store.load_state()
    cases = store.load_cases()
    _ledger, accounting = _accounting_from_cases(cases, state)
    completed_pairs = len({c.key for c in cases})

    providers = [_provider_inspection(m, cases) for m in state.model_ids]

    return RunInspection(
        run_id=state.run_id,
        mode=state.mode,
        status=state.status,
        complete=(state.status == COMPLETE and completed_pairs >= state.planned_pairs),
        config_hash=state.config_hash,
        ordered_record_sha256=state.ordered_record_sha256,
        planned_pairs=state.planned_pairs,
        completed_pairs=completed_pairs,
        total_run_seconds=state.total_run_seconds,
        accounting=accounting,
        overall_latency=_latency(cases),
        providers=providers,
        stop_reason=state.stop_reason,
    )
