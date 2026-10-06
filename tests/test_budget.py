"""Deterministic budget-accounting tests.

These cover the conservative stop-before guard, the three separate exposure
buckets (billed OpenRouter vs rate-estimated Jev vs retained-unknown), the
planning seed, and durable replay. They stand in for the arithmetic that keeps
a live run inside the shared $20 cap; the live spend itself is exercised by the
pilot CLI.
"""

import pytest

from jev_benchmark.budget import (
    BILLED,
    ESTIMATED,
    PLANNING_SEED_USD,
    RESERVED_UNKNOWN,
    BudgetLedger,
    Pricing,
    Settlement,
)
from jev_benchmark.errors import BudgetExceeded
from jev_benchmark.providers.base import Usage


def test_seed_is_initial_exposure():
    ledger = BudgetLedger()
    assert ledger.planning_seed == PLANNING_SEED_USD
    assert ledger.exposure == pytest.approx(PLANNING_SEED_USD)
    assert ledger.remaining == pytest.approx(20.0 - PLANNING_SEED_USD)


def test_openrouter_billed_and_jev_estimate_stay_separate():
    ledger = BudgetLedger()
    t1 = ledger.reserve()
    ledger.settle_openrouter(t1, Usage(input_tokens=10, output_tokens=2, cost=0.00001))
    t2 = ledger.reserve()
    ledger.settle_estimated(t2, Usage(input_tokens=1_000_000, output_tokens=0))
    assert ledger.billed_openrouter == pytest.approx(0.00001)
    # 1,000,000 input tokens at $0.042/M = $0.042 Jev estimate.
    assert ledger.estimated_jev == pytest.approx(0.042)
    assert ledger.reserved_unknown == 0.0


def test_unknown_billable_attempt_is_not_free():
    ledger = BudgetLedger()
    token = ledger.reserve()
    settlement = ledger.settle_unknown(token)
    assert settlement.basis == RESERVED_UNKNOWN
    assert ledger.reserved_unknown == pytest.approx(ledger.pricing.per_attempt_reservation)
    # A timeout keeps exposure above the seed; it is not assumed free.
    assert ledger.exposure > PLANNING_SEED_USD


def test_openrouter_without_cost_is_retained_as_unknown():
    ledger = BudgetLedger()
    token = ledger.reserve()
    settlement = ledger.settle_openrouter(token, Usage(input_tokens=5, cost=None))
    assert settlement.basis == RESERVED_UNKNOWN
    assert ledger.billed_openrouter == 0.0
    assert ledger.reserved_unknown > 0.0


def test_jev_settle_without_usage_is_retained_as_unknown():
    # A Jev attempt whose usage never arrived cannot be priced: its reservation
    # is retained as unknown billable exposure, never booked at a $0.00
    # estimate (mirroring the OpenRouter missing-cost retention).
    ledger = BudgetLedger()
    token = ledger.reserve()
    settlement = ledger.settle_estimated(token, None)
    assert settlement.basis == RESERVED_UNKNOWN
    assert settlement.amount == pytest.approx(ledger.pricing.per_attempt_reservation)
    assert ledger.estimated_jev == 0.0
    assert ledger.reserved_unknown == pytest.approx(ledger.pricing.per_attempt_reservation)
    # The attempt is not assumed free: exposure stays above the seed.
    assert ledger.exposure > PLANNING_SEED_USD


def test_jev_settle_with_missing_token_counts_is_retained_as_unknown():
    # Either missing count (input or output) makes the token-rate price basis
    # unknown, so the reservation is retained rather than settled at $0.
    ledger = BudgetLedger()
    for usage in (
        Usage(),  # no usage block at all
        Usage(input_tokens=5),  # output token count missing
        Usage(output_tokens=5),  # input token count missing
    ):
        token = ledger.reserve()
        settlement = ledger.settle_estimated(token, usage)
        assert settlement.basis == RESERVED_UNKNOWN
    assert ledger.estimated_jev == 0.0
    assert ledger.reserved_unknown == pytest.approx(3 * ledger.pricing.per_attempt_reservation)


def test_jev_settle_with_complete_token_counts_is_estimated():
    # Present counts price normally even when the output count is zero: the
    # price basis is fully known, so the reservation settles to the estimate.
    ledger = BudgetLedger()
    token = ledger.reserve()
    settlement = ledger.settle_estimated(token, Usage(input_tokens=1_000, output_tokens=0))
    assert settlement.basis == ESTIMATED
    assert ledger.estimated_jev == pytest.approx(1_000 * 0.042 / 1_000_000)
    assert ledger.reserved_unknown == 0.0


def test_reserve_refused_before_reaching_cap():
    pricing = Pricing(per_attempt_reservation=1.0)
    ledger = BudgetLedger(cap=5.0, pricing=pricing, planning_seed=0.0)
    tokens = [ledger.reserve() for _ in range(4)]  # 4.0 outstanding, under 5.0
    assert ledger.exposure == pytest.approx(4.0)
    # A fifth reservation would make exposure 5.0 == cap, which must be refused.
    with pytest.raises(BudgetExceeded) as excinfo:
        ledger.reserve()
    assert excinfo.value.cap == 5.0
    assert excinfo.value.exposure >= 5.0
    assert len(tokens) == 4


def test_exposure_counts_outstanding_plus_spent_plus_new():
    pricing = Pricing(per_attempt_reservation=2.0)
    ledger = BudgetLedger(cap=10.0, pricing=pricing, planning_seed=1.0)
    t1 = ledger.reserve()
    ledger.settle_billed(t1, 3.0)  # spent now 1.0 seed + 3.0 billed = 4.0
    ledger.reserve()  # outstanding 2.0 -> exposure 6.0
    # Next reservation of 2.0 -> would be 8.0 < 10.0, allowed.
    assert ledger.would_exceed(2.0) is False
    # A reservation of 4.0 -> would be 10.0 == cap, refused.
    assert ledger.would_exceed(4.0) is True


def test_replay_rebuilds_buckets_from_settlements():
    settlements = [
        Settlement(BILLED, 0.001),
        Settlement(ESTIMATED, 0.002),
        Settlement(RESERVED_UNKNOWN, 0.005),
    ]
    ledger = BudgetLedger.replay(settlements, planning_seed=0.10)
    assert ledger.billed_openrouter == pytest.approx(0.001)
    assert ledger.estimated_jev == pytest.approx(0.002)
    assert ledger.reserved_unknown == pytest.approx(0.005)
    assert ledger.exposure == pytest.approx(0.10 + 0.001 + 0.002 + 0.005)


def test_as_dict_roundtrip_preserves_spent():
    ledger = BudgetLedger()
    t = ledger.reserve()
    ledger.settle_billed(t, 0.5)
    restored = BudgetLedger.from_dict(ledger.as_dict())
    assert restored.billed_openrouter == pytest.approx(0.5)
    assert restored.spent == pytest.approx(ledger.spent)
