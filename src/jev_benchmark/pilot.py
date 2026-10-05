"""Pilot projection: gate the full run on a small real sample.

A pilot is an ordinary measured run over a few rows. From its recorded cases
this module computes a *conservative* projection of the full 1,014-row run cost
and decides whether the whole run can start without the shared ``$20`` exposure
guard being breached. If it cannot, the full run must stop with a clear reason
rather than quietly shrinking the population or retuning prompts.

Conservatism: per-provider per-row cost uses the pilot's observed billed
(OpenRouter) or token-rate-estimated (Jev) spend plus any unknown-reserved
amount, divided by the number of pilot rows, then multiplied by a safety
factor. The projection is added to current exposure (which already includes the
planning seed and any preflight/pilot spend) before comparison with the cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import workflow
from .budget import BILLED, ESTIMATED, RESERVED_UNKNOWN, BudgetLedger
from .runner import CaseRecord

FULL_POPULATION_ROWS = 1014
DEFAULT_SAFETY_FACTOR = 2.0


def validated_models(cases: list[CaseRecord]) -> list[str]:
    """Model IDs with preflight-equivalent evidence from the pilot.

    A provider counts as validated only when at least one pilot case returned a
    complete, schema-valid prediction with known usage. Invalid/refused/error
    outcomes or a prediction without accountable usage do not count, so a pilot
    cannot authorize a bulk run for a provider that never actually validated.
    """
    ok: set[str] = set()
    for case in cases:
        if case.outcome != workflow.PREDICTED:
            continue
        if any(a.input_tokens is not None for a in case.attempts):
            ok.add(case.requested_model)
    return sorted(ok)


@dataclass
class ProviderProjection:
    model_id: str
    pilot_rows: int
    billed: float
    estimated: float
    reserved_unknown: float
    per_row_cost: float
    projected_full_cost: float

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "pilot_rows": self.pilot_rows,
            "billed": self.billed,
            "estimated": self.estimated,
            "reserved_unknown": self.reserved_unknown,
            "per_row_cost": self.per_row_cost,
            "projected_full_cost": self.projected_full_cost,
        }


@dataclass
class Projection:
    full_population_rows: int
    safety_factor: float
    exposure_before: float
    cap: float
    projected_full_cost: float
    projected_total_exposure: float
    affordable: bool
    reason: str
    providers: list[ProviderProjection] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "full_population_rows": self.full_population_rows,
            "safety_factor": self.safety_factor,
            "exposure_before": self.exposure_before,
            "cap": self.cap,
            "projected_full_cost": self.projected_full_cost,
            "projected_total_exposure": self.projected_total_exposure,
            "affordable": self.affordable,
            "reason": self.reason,
            "providers": [p.as_dict() for p in self.providers],
        }


def _provider_costs(cases: list[CaseRecord], model_id: str) -> tuple[float, float, float, int]:
    billed = estimated = reserved = 0.0
    rows: set[int] = set()
    for case in cases:
        if case.requested_model != model_id:
            continue
        rows.add(case.row_index)
        for attempt in case.attempts:
            if attempt.cost_basis == BILLED:
                billed += attempt.cost_amount
            elif attempt.cost_basis == ESTIMATED:
                estimated += attempt.cost_amount
            elif attempt.cost_basis == RESERVED_UNKNOWN:
                reserved += attempt.cost_amount
    return billed, estimated, reserved, len(rows)


def project_full_run(
    cases: list[CaseRecord],
    ledger: BudgetLedger,
    model_ids: list[str],
    *,
    full_population_rows: int = FULL_POPULATION_ROWS,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> Projection:
    """Conservatively project the full-run cost and whether it fits the cap."""
    providers: list[ProviderProjection] = []
    projected_full_cost = 0.0
    for model_id in model_ids:
        billed, estimated, reserved, pilot_rows = _provider_costs(cases, model_id)
        total = billed + estimated + reserved
        per_row = (total / pilot_rows) if pilot_rows else 0.0
        projected = per_row * full_population_rows * safety_factor
        projected_full_cost += projected
        providers.append(
            ProviderProjection(
                model_id=model_id,
                pilot_rows=pilot_rows,
                billed=billed,
                estimated=estimated,
                reserved_unknown=reserved,
                per_row_cost=per_row,
                projected_full_cost=projected,
            )
        )

    exposure_before = ledger.exposure
    projected_total = exposure_before + projected_full_cost
    affordable = projected_total < ledger.cap
    if affordable:
        reason = (
            f"projected total exposure ${projected_total:.6f} stays under the "
            f"${ledger.cap:.2f} cap; full {full_population_rows}-row run may proceed"
        )
    else:
        reason = (
            f"projected total exposure ${projected_total:.6f} would reach the "
            f"${ledger.cap:.2f} cap; full run must not start"
        )
    return Projection(
        full_population_rows=full_population_rows,
        safety_factor=safety_factor,
        exposure_before=exposure_before,
        cap=ledger.cap,
        projected_full_cost=projected_full_cost,
        projected_total_exposure=projected_total,
        affordable=affordable,
        reason=reason,
        providers=providers,
    )
