"""Durable, conservative budget accounting for paid provider calls.

A single shared software guard caps *estimated aggregate exposure* across
planning readiness, preflight, pilot, measured run, and retries at ``$20``.
This is a conservative local guard, not a vendor-enforced account cap (Jev in
particular has no billed-dollar field).

Exposure is tracked in three deliberately separate buckets so OpenRouter's
billed amounts are never conflated with Jev's published-token-rate estimate or
with amounts that are simply unknown:

* ``billed_openrouter`` - OpenRouter ``usage.cost`` values actually reported.
* ``estimated_jev`` - Jev spend estimated from token counts at a published
  rate, because Jev exposes tokens but no billed-dollar field.
* ``reserved_unknown`` - worst-case reservations retained for attempts whose
  billable amount is unknown (timeouts, transport failures, or a missing cost
  field). These are never assumed free.

A fixed ``$0.10`` planning-readiness seed accounts for the three real
readiness calls already made during planning (OpenRouter reported
``$0.00002655`` combined; Jev reported tokens with no billed cost), so later
work cannot spend the entire ``$20`` anew.

Before every stage attempt the runner ``reserve``\\ s a conservative worst-case
amount. Dispatch is refused when ``spent + outstanding reservations + the new
reservation`` would reach the cap. After the attempt the reservation is
``settle``\\ d to a billed amount, a token-rate estimate, or retained as
unknown. The ledger serialises to JSON and can be rebuilt by replaying the
per-attempt records of a run, so resume preserves accounting exactly.

One ledger can be bound to a :class:`SharedBudgetStore`, a durable JSON file
shared across planning readiness, preflight, pilot, measured run and retries.
Every reservation and settlement is flushed to that file (pending reservations
included) so an interrupted session's in-flight exposure is recovered on the
next load instead of being assumed free. :func:`open_shared_budget` initialises
a fresh file from the planning seed plus historical exposure reconciled from
prior run directories, and :func:`validate_cap` refuses any configured cap above
the approved ``$20`` ceiling before paid calls begin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import BudgetConfigError, BudgetExceeded, HistoricalExposureError
from .providers.base import Usage

# The approved shared software guard. A user may configure a *lower* cap for a
# tighter test, but never a higher one: $20 is the mission ceiling.
CAP_USD = 20.0
PLANNING_SEED_USD = 0.10

# Jev published its rate as $0.042 per million input tokens at planning time;
# refresh for reporting rather than trusting a live bill. Jev exposes no output
# price, so output tokens are charged at the same (conservative) rate here.
JEV_INPUT_RATE_PER_TOKEN = 0.042 / 1_000_000
JEV_OUTPUT_RATE_PER_TOKEN = 0.042 / 1_000_000

# Conservative worst-case reservation for a single stage API attempt. Real
# attempts cost a tiny fraction of this (planning readiness was ~$1.3e-5 per
# OpenRouter call); the headroom is intentional while still leaving the full
# 1,014-row run affordable because reservations are released on settlement.
PER_ATTEMPT_RESERVATION_USD = 0.005

# Cost bases recorded per settled attempt.
BILLED = "billed"
ESTIMATED = "estimated"
RESERVED_UNKNOWN = "reserved_unknown"
NONE = "none"


@dataclass(frozen=True)
class Pricing:
    """Rates and the conservative per-attempt reservation."""

    jev_input_rate: float = JEV_INPUT_RATE_PER_TOKEN
    jev_output_rate: float = JEV_OUTPUT_RATE_PER_TOKEN
    per_attempt_reservation: float = PER_ATTEMPT_RESERVATION_USD

    def jev_estimate(self, usage: Usage | None) -> float:
        """Published-token-rate estimate for one Jev attempt (never a billed cost)."""
        if usage is None:
            return 0.0
        return (usage.input_tokens or 0) * self.jev_input_rate + (
            usage.output_tokens or 0
        ) * self.jev_output_rate

    def as_dict(self) -> dict:
        return {
            "jev_input_rate": self.jev_input_rate,
            "jev_output_rate": self.jev_output_rate,
            "per_attempt_reservation": self.per_attempt_reservation,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> Pricing:
        if not data:
            return cls()
        return cls(
            jev_input_rate=data.get("jev_input_rate", JEV_INPUT_RATE_PER_TOKEN),
            jev_output_rate=data.get("jev_output_rate", JEV_OUTPUT_RATE_PER_TOKEN),
            per_attempt_reservation=data.get(
                "per_attempt_reservation", PER_ATTEMPT_RESERVATION_USD
            ),
        )


@dataclass
class Settlement:
    """How one reservation was resolved: a basis plus its dollar amount."""

    basis: str  # BILLED | ESTIMATED | RESERVED_UNKNOWN | NONE
    amount: float

    def as_dict(self) -> dict:
        return {"basis": self.basis, "amount": self.amount}


class BudgetLedger:
    """Mutable, serialisable record of conservative paid-call exposure."""

    def __init__(
        self,
        *,
        cap: float = CAP_USD,
        pricing: Pricing | None = None,
        planning_seed: float = PLANNING_SEED_USD,
        billed_openrouter: float = 0.0,
        estimated_jev: float = 0.0,
        reserved_unknown: float = 0.0,
    ) -> None:
        self.cap = cap
        self.pricing = pricing or Pricing()
        self.planning_seed = planning_seed
        self.billed_openrouter = billed_openrouter
        self.estimated_jev = estimated_jev
        self.reserved_unknown = reserved_unknown
        self._outstanding: dict[int, float] = {}
        self._next_token = 1
        # When bound to a durable store, every reservation and settlement is
        # flushed so an interrupted process still accounts for in-flight calls.
        self.store: SharedBudgetStore | None = None

    def bind(self, store: SharedBudgetStore) -> BudgetLedger:
        self.store = store
        return self

    def _flush(self) -> None:
        if self.store is not None:
            self.store.save(self)

    # -- derived exposure ---------------------------------------------------- #
    @property
    def spent(self) -> float:
        """Settled exposure: seed + billed + estimated + retained unknown."""
        return (
            self.planning_seed
            + self.billed_openrouter
            + self.estimated_jev
            + self.reserved_unknown
        )

    @property
    def outstanding_reserved(self) -> float:
        """In-flight reservations not yet settled this session."""
        return sum(self._outstanding.values())

    @property
    def exposure(self) -> float:
        """Conservative total: settled spend plus still-outstanding reservations."""
        return self.spent + self.outstanding_reserved

    @property
    def remaining(self) -> float:
        return self.cap - self.exposure

    # -- reservation / settlement ------------------------------------------- #
    def would_exceed(self, amount: float) -> bool:
        """True when reserving ``amount`` more would reach the cap (stop-before)."""
        return self.exposure + amount >= self.cap

    def reserve(self, amount: float | None = None) -> int:
        """Reserve a worst-case amount for the next attempt or refuse at the cap.

        Refusal is *before* exposure reaches the cap: spent + outstanding + the
        new reservation must stay strictly under it. Timeouts and other unknown
        billable attempts keep their reservation, so they count here too.
        """
        amt = self.pricing.per_attempt_reservation if amount is None else amount
        if self.would_exceed(amt):
            raise BudgetExceeded(
                f"dispatch refused: exposure ${self.exposure:.6f} + reservation "
                f"${amt:.6f} would reach the ${self.cap:.2f} shared cap",
                exposure=self.exposure + amt,
                cap=self.cap,
            )
        token = self._next_token
        self._next_token += 1
        self._outstanding[token] = amt
        # Journal the pending reservation *before* the paid call so an in-flight
        # attempt that never returns stays accounted for on reload.
        self._flush()
        return token

    def _pop(self, token: int) -> float:
        return self._outstanding.pop(token)

    def settle_billed(self, token: int, cost: float) -> Settlement:
        """Settle an OpenRouter attempt to its reported billed cost."""
        self._pop(token)
        self.billed_openrouter += cost
        self._flush()
        return Settlement(BILLED, cost)

    def settle_estimated(self, token: int, usage: Usage | None) -> Settlement:
        """Settle a Jev attempt to a published-token-rate estimate."""
        self._pop(token)
        estimate = self.pricing.jev_estimate(usage)
        self.estimated_jev += estimate
        self._flush()
        return Settlement(ESTIMATED, estimate)

    def settle_unknown(self, token: int) -> Settlement:
        """Retain a reservation as unknown billable exposure (not assumed free)."""
        amount = self._pop(token)
        self.reserved_unknown += amount
        self._flush()
        return Settlement(RESERVED_UNKNOWN, amount)

    def settle_openrouter(self, token: int, usage: Usage | None) -> Settlement:
        """Bill an OpenRouter attempt when a cost is reported, else keep it unknown."""
        if usage is not None and usage.cost is not None:
            return self.settle_billed(token, float(usage.cost))
        return self.settle_unknown(token)

    def release(self, token: int) -> Settlement:
        """Release a reservation with no charge (only when provably not billable)."""
        self._pop(token)
        self._flush()
        return Settlement(NONE, 0.0)

    # -- durability --------------------------------------------------------- #
    def as_dict(self) -> dict:
        return {
            "cap": self.cap,
            "pricing": self.pricing.as_dict(),
            "planning_seed": self.planning_seed,
            "billed_openrouter": self.billed_openrouter,
            "estimated_jev": self.estimated_jev,
            "reserved_unknown": self.reserved_unknown,
            "spent": self.spent,
            "outstanding_reserved": self.outstanding_reserved,
            "exposure": self.exposure,
            "remaining": self.remaining,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BudgetLedger:
        return cls(
            cap=data.get("cap", CAP_USD),
            pricing=Pricing.from_dict(data.get("pricing")),
            planning_seed=data.get("planning_seed", PLANNING_SEED_USD),
            billed_openrouter=data.get("billed_openrouter", 0.0),
            estimated_jev=data.get("estimated_jev", 0.0),
            reserved_unknown=data.get("reserved_unknown", 0.0),
        )

    def add_settlement(self, settlement: Settlement) -> None:
        """Fold a recorded settlement straight into the right bucket (replay)."""
        if settlement.basis == BILLED:
            self.billed_openrouter += settlement.amount
        elif settlement.basis == ESTIMATED:
            self.estimated_jev += settlement.amount
        elif settlement.basis == RESERVED_UNKNOWN:
            self.reserved_unknown += settlement.amount

    @classmethod
    def replay(
        cls,
        settlements: list[Settlement],
        *,
        cap: float = CAP_USD,
        pricing: Pricing | None = None,
        planning_seed: float = PLANNING_SEED_USD,
    ) -> BudgetLedger:
        """Rebuild a ledger from durable per-attempt settlements plus the seed."""
        ledger = cls(cap=cap, pricing=pricing, planning_seed=planning_seed)
        for settlement in settlements:
            ledger.add_settlement(settlement)
        return ledger

    # -- shared durable state ----------------------------------------------- #
    def shared_state_dict(self) -> dict:
        """Durable snapshot for the cross-phase shared ledger file.

        Unlike :meth:`as_dict` (a human-readable run header), this carries the
        raw pending reservations so an interrupted session's in-flight exposure
        is preserved on reload.
        """
        return {
            "cap": self.cap,
            "pricing": self.pricing.as_dict(),
            "planning_seed": self.planning_seed,
            "billed_openrouter": self.billed_openrouter,
            "estimated_jev": self.estimated_jev,
            "reserved_unknown": self.reserved_unknown,
            "pending": list(self._outstanding.values()),
            "exposure": self.exposure,
            "remaining": self.remaining,
        }


def validate_cap(requested: float) -> float:
    """Return a usable cap or refuse one above the approved ceiling.

    A user may tighten the guard (a smaller cap) but never raise it: a cap above
    ``$20`` is refused before any paid call so an over-budget run cannot start.
    """
    if requested > CAP_USD:
        raise BudgetConfigError(
            f"budget cap ${requested:.2f} exceeds the approved ${CAP_USD:.2f} "
            "shared ceiling; configure a cap at or below the approved amount"
        )
    if requested <= 0:
        raise BudgetConfigError("budget cap must be positive")
    return requested


def reconcile_historical_exposure(runs_dir: str | Path) -> float:
    """Sum settled marginal spend recorded by prior run directories.

    Each ``run.json`` carries a budget snapshot. Its settled spend *beyond* the
    shared planning seed (billed + estimated + retained-unknown + any still
    outstanding) is folded in so a fresh shared ledger does not forget paid
    development or planning attempts. A snapshot that cannot be parsed raises
    :class:`HistoricalExposureError` so the caller can stop rather than guess.
    """
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        return 0.0
    total = 0.0
    for run_json in sorted(runs_path.glob("*/run.json")):
        try:
            data = json.loads(run_json.read_text())
            budget = data["budget"]
            spent = (
                float(budget.get("billed_openrouter", 0.0))
                + float(budget.get("estimated_jev", 0.0))
                + float(budget.get("reserved_unknown", 0.0))
                + float(budget.get("outstanding_reserved", 0.0))
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HistoricalExposureError(
                f"cannot bound historical exposure from {run_json}: {exc}"
            ) from exc
        total += max(spent, 0.0)
    return total


class SharedBudgetStore:
    """Durable JSON file for the single cross-phase shared exposure.

    Planning readiness, preflight, pilot, measured run and retries all load and
    update the same file, so the ``$20`` guard is one cumulative exposure rather
    than a fresh allowance per command. Pending reservations recovered from an
    interrupted session are folded into retained-unknown exposure on load, since
    an in-flight paid call must never be assumed free.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def save(self, ledger: BudgetLedger) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(ledger.shared_state_dict(), indent=2) + "\n")

    def load(self) -> BudgetLedger:
        data = json.loads(self.path.read_text())
        pending = data.get("pending", []) or []
        ledger = BudgetLedger(
            cap=min(float(data.get("cap", CAP_USD)), CAP_USD),
            pricing=Pricing.from_dict(data.get("pricing")),
            planning_seed=float(data.get("planning_seed", PLANNING_SEED_USD)),
            billed_openrouter=float(data.get("billed_openrouter", 0.0)),
            estimated_jev=float(data.get("estimated_jev", 0.0)),
            reserved_unknown=float(data.get("reserved_unknown", 0.0))
            + sum(float(p) for p in pending),
        )
        ledger.store = self
        return ledger


def open_shared_budget(
    path: str | Path,
    *,
    cap: float = CAP_USD,
    runs_dir: str | Path | None = None,
) -> BudgetLedger:
    """Open (or initialise) the one durable shared ledger bound to ``path``.

    An existing file is loaded and its cap clamped to the approved ceiling and
    the (possibly lower) requested cap. A fresh file starts from the planning
    seed plus any historical exposure reconciled from ``runs_dir``.
    """
    cap = validate_cap(cap)
    store = SharedBudgetStore(path)
    if store.exists():
        ledger = store.load()
        ledger.cap = min(ledger.cap, cap)
        return ledger
    historical = reconcile_historical_exposure(runs_dir) if runs_dir else 0.0
    ledger = BudgetLedger(cap=cap, reserved_unknown=historical)
    ledger.store = store
    store.save(ledger)
    return ledger
