"""Deterministic metric-definition tests for the offline report.

These pin the conservative scoring semantics required by the validation
contract (VAL-REPORT-001..004): invalid/abstained outcomes stay in the
all-attempted denominator, an invalid prediction is a false negative for its
true class but never a false positive for another class, zero-division is
explicit, latency scopes keep distinct denominators, cost bases are never
conflated, and uncertainty/duplicate-sensitivity methods are deterministic.
"""

from __future__ import annotations

import pytest
from scipy.stats import binomtest

from jev_benchmark import metrics, spec
from jev_benchmark.dataset import Record
from jev_benchmark.metrics import (
    INVALID_COLUMN,
    all_attempted_metrics,
    bootstrap_intervals,
    confusion_matrix,
    cost_summary,
    dedup_row_indices,
    latency_summary,
    mcnemar_exact,
    success_only_metrics,
)
from jev_benchmark.runner import AttemptRecord, CaseRecord
from jev_benchmark.workflow import ERROR, INVALID, PREDICTED

MODEL = "jev-1.13.0"
OTHER = "openai/gpt-4o-mini"


def att(
    model: str = MODEL,
    stage: str = "stage1",
    attempt: int = 1,
    status: str = "ok",
    started: float = 0.0,
    dur: float | None = 1.0,
    in_tok: int | None = 100,
    out_tok: int | None = 10,
    basis: str = "estimated",
    amount: float = 0.00001,
) -> AttemptRecord:
    ended = None if dur is None else started + dur
    return AttemptRecord(
        provider=model,
        provider_kind="jev" if model == "jev-1.13.0" else "openrouter",
        stage=stage,
        attempt=attempt,
        status=status,
        started_at=started,
        ended_at=ended,
        actual_model=model if status == "ok" else None,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_basis=basis,
        cost_amount=amount,
    )


def case(
    row: int,
    model: str = MODEL,
    outcome: str = PREDICTED,
    final: str | None = "low risk",
    gold: str = "low risk",
    attempts: list[AttemptRecord] | None = None,
    stage_count: int = 1,
    wf: tuple[float, float | None] = (0.0, 2.0),
) -> CaseRecord:
    return CaseRecord(
        row_index=row,
        requested_model=model,
        provider_kind="jev" if model == "jev-1.13.0" else "openrouter",
        outcome=outcome,
        final_label=final,
        gold_label=gold,
        stage_count=stage_count,
        workflow_started_at=wf[0],
        workflow_ended_at=wf[1],
        attempts=attempts if attempts is not None else [att(model)],
    )


def records() -> list[Record]:
    f = {
        "Age": 25, "SystolicBP": 130, "DiastolicBP": 80,
        "BS": 15.0, "BodyTemp": 98.0, "HeartRate": 86,
    }
    g = {
        "Age": 35, "SystolicBP": 140, "DiastolicBP": 90,
        "BS": 13.0, "BodyTemp": 98.0, "HeartRate": 70,
    }
    h = {
        "Age": 30, "SystolicBP": 100, "DiastolicBP": 70,
        "BS": 8.0, "BodyTemp": 98.0, "HeartRate": 80,
    }
    i = {
        "Age": 22, "SystolicBP": 120, "DiastolicBP": 60,
        "BS": 6.1, "BodyTemp": 98.0, "HeartRate": 76,
    }
    return [
        Record(0, f, "high risk"),
        Record(1, f, "high risk"),  # exact duplicate of row 0
        Record(2, g, "mid risk"),
        Record(3, h, "mid risk"),
        Record(4, i, "low risk"),
        Record(5, i, "high risk"),  # conflicting feature group with row 4
    ]


# --------------------------------------------------------------------------- #
# All-attempted (conservative) metrics.
# --------------------------------------------------------------------------- #

def test_confusion_invalid_is_false_negative_not_false_positive():
    cases = [
        case(0, gold="high risk", final="high risk"),
        case(1, gold="high risk", outcome=INVALID, final=None),
        case(2, gold="mid risk", final="low risk"),
    ]
    cm = confusion_matrix(cases, MODEL)
    assert set(cm) == set(spec.RISK_LABELS)
    for row in cm.values():
        assert set(row) == {*spec.RISK_LABELS, INVALID_COLUMN}
    assert cm["high risk"][INVALID_COLUMN] == 1
    assert cm["high risk"]["high risk"] == 1
    assert cm["mid risk"]["low risk"] == 1
    # The invalid outcome added no false positive to any other risk class.
    assert cm["mid risk"]["high risk"] == 0
    assert cm["low risk"]["high risk"] == 0


def test_all_attempted_denominator_keeps_invalid_as_incorrect():
    cases = [
        case(0, gold="high risk", final="high risk"),
        case(1, gold="high risk", outcome=INVALID, final=None),
        case(2, gold="mid risk", final="low risk"),
    ]
    m = all_attempted_metrics(cases, MODEL)
    assert m["coverage"]["attempted"] == 3
    assert m["coverage"]["predicted"] == 2
    assert m["coverage"]["invalid_or_abstained"] == 1
    assert m["accuracy"] == pytest.approx(1 / 3)
    # high-risk recall keeps the invalid outcome in its denominator: 1/2.
    assert m["high_risk_recall"] == pytest.approx(0.5)
    # balanced accuracy averages the three class recalls (low has no support).
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 0.0 + 0.0) / 3)
    # macro-F1 averages the three published-class F1 values.
    f1s = [m["classes"][label]["f1"] for label in spec.RISK_LABELS]
    assert m["macro_f1"] == pytest.approx(sum(f1s) / 3)


def test_zero_division_is_explicit():
    cases = [
        case(0, gold="high risk", final="high risk"),
        case(1, gold="high risk", outcome=INVALID, final=None),
        case(2, gold="mid risk", final="low risk"),
    ]
    m = all_attempted_metrics(cases, MODEL)
    # mid risk: nothing predicted mid -> precision defined as 0 with a note.
    assert m["classes"]["mid risk"]["precision"] == 0.0
    assert m["classes"]["mid risk"]["predicted"] == 0
    # low risk: no gold support in this subset -> recall defined as 0 with a note.
    assert m["classes"]["low risk"]["recall"] == 0.0
    assert m["classes"]["low risk"]["support"] == 0
    notes = " ".join(m["zero_division_notes"])
    assert "mid risk" in notes and "precision" in notes
    assert "low risk" in notes and "recall" in notes
    assert "0.0" in notes  # the applied convention is stated


def test_class_metrics_reconcile_with_confusion_cells():
    cases = [
        case(0, gold="high risk", final="high risk"),
        case(1, gold="mid risk", final="mid risk"),
        case(2, gold="mid risk", final="low risk"),
        case(3, gold="low risk", final="mid risk"),
    ]
    m = all_attempted_metrics(cases, MODEL)
    mid = m["classes"]["mid risk"]
    assert mid["tp"] == 1
    assert mid["predicted"] == 2
    assert mid["support"] == 2
    assert mid["precision"] == pytest.approx(0.5)
    assert mid["recall"] == pytest.approx(0.5)
    assert mid["f1"] == pytest.approx(0.5)
    low = m["classes"]["low risk"]
    assert low["tp"] == 0
    assert low["precision"] == 0.0
    assert low["recall"] == 0.0
    # accuracy = correct/attempted = 2/4; confusion totals reconcile.
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["coverage"]["attempted"] == 4


def test_success_only_metrics_report_narrower_coverage():
    cases = [
        case(0, gold="high risk", final="high risk"),
        case(1, gold="high risk", outcome=INVALID, final=None),
        case(2, gold="mid risk", final="low risk"),
    ]
    s = success_only_metrics(cases, MODEL)
    assert s["attempted"] == 3
    assert s["predicted"] == 2
    assert s["coverage_rate"] == pytest.approx(2 / 3)
    # Only the two predicted rows are scored: 1 of 2 correct.
    assert s["accuracy"] == pytest.approx(0.5)
    cm = s["confusion"]
    assert sum(cm[g][INVALID_COLUMN] for g in cm) == 0
    # The invalid row is absent from the success-only confusion entirely.
    assert cm["high risk"]["high risk"] == 1
    assert cm["mid risk"]["low risk"] == 1


def test_unattempted_counts_reconcile_with_planned_population():
    # Provider attempted only rows 0 and 1 of a planned population of 3.
    cases = [case(0), case(1)]
    cov = metrics.coverage_counts(cases, MODEL, planned_rows=3)
    assert cov["attempted"] == 2
    assert cov["unattempted"] == 1
    assert cov["predicted"] + cov["invalid_or_abstained"] + cov["failed"] == cov["attempted"]


def test_error_outcomes_are_failed_not_invalid():
    cases = [
        case(0, outcome=ERROR, final=None, gold="mid risk"),
        case(1, outcome=INVALID, final=None, gold="low risk"),
        case(2, final="low risk", gold="low risk"),
    ]
    cov = metrics.coverage_counts(cases, MODEL, planned_rows=3)
    assert cov["error"] == 1
    assert cov["invalid"] == 1
    assert cov["invalid_or_abstained"] == 1  # invalid (+refused)
    assert cov["failed"] == 1  # error (+mismatch)
    m = all_attempted_metrics(cases, MODEL)
    # Both non-predicted outcomes land in the invalid column of their gold row.
    assert m["confusion"]["mid risk"][INVALID_COLUMN] == 1
    assert m["confusion"]["low risk"][INVALID_COLUMN] == 1


# --------------------------------------------------------------------------- #
# Latency scopes.
# --------------------------------------------------------------------------- #

def test_latency_scopes_keep_distinct_denominators():
    cases = [
        case(0, attempts=[att(dur=1.0), att(dur=2.0, attempt=2)], wf=(0.0, 4.0)),
        case(1, attempts=[att(dur=0.5)], wf=(0.0, 1.5)),
        case(2, attempts=[att(dur=10.0, status="error")],
             outcome=ERROR, final=None, wf=(0.0, 12.0)),
        case(3, attempts=[att(dur=None)], outcome=ERROR, final=None, wf=(0.0, None)),
    ]
    lat = latency_summary(cases, MODEL)
    # API-attempt scope: every finished attempt, including the failed one; the
    # interrupted attempt (no end time) is excluded, not invented.
    assert lat["api_attempt"]["count"] == 4
    assert lat["api_attempt"]["max"] == pytest.approx(10.0)
    assert lat["interrupted_attempts"] == 1
    # Workflow scope: whole row-provider workflows only.
    assert lat["workflow"]["count"] == 3
    assert lat["workflow"]["max"] == pytest.approx(12.0)
    # Retried and failed attempts are counted separately.
    assert lat["retried_attempts"] == 1
    assert lat["failed_attempts"] == 1
    assert lat["failed_attempt_latency"]["count"] == 1
    assert lat["failed_attempt_latency"]["max"] == pytest.approx(10.0)
    assert lat["retried_attempt_latency"]["count"] == 1
    assert lat["retried_attempt_latency"]["max"] == pytest.approx(2.0)


def test_latency_is_per_provider():
    a = case(0, attempts=[att(dur=1.0)])
    b = case(0, model=OTHER, attempts=[att(OTHER, dur=5.0, basis="billed", amount=0.0001)])
    lat_model = latency_summary([a, b], MODEL)
    assert lat_model["api_attempt"]["count"] == 1
    assert lat_model["api_attempt"]["p50"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Cost aggregation.
# --------------------------------------------------------------------------- #

def test_cost_summary_separates_bases_and_missing_usage():
    cases = [
        case(
            0,
            attempts=[
                att(basis="estimated", amount=0.00002, in_tok=400, out_tok=32),
                att(dur=31.5, attempt=2, status="error", basis="reserved_unknown",
                    amount=0.005, in_tok=None, out_tok=None),
            ],
        ),
        case(
            1,
            model=OTHER,
            attempts=[
                att(OTHER, stage="stage1", basis="billed",
                    amount=0.00005, in_tok=272, out_tok=26),
                att(OTHER, stage="stage2", basis="billed",
                    amount=0.00005, in_tok=256, out_tok=26),
            ],
        ),
    ]
    jev_cost = cost_summary(cases, MODEL)
    assert jev_cost["estimated_jev"] == pytest.approx(0.00002)
    assert jev_cost["billed_openrouter"] == 0.0
    assert jev_cost["reserved_unknown"] == pytest.approx(0.005)
    assert jev_cost["input_tokens"] == 400
    assert jev_cost["output_tokens"] == 32
    assert jev_cost["attempts_missing_usage"] == 1

    or_cost = cost_summary(cases, OTHER)
    assert or_cost["billed_openrouter"] == pytest.approx(0.0001)
    assert or_cost["estimated_jev"] == 0.0
    assert or_cost["reserved_unknown"] == 0.0
    assert or_cost["input_tokens"] == 528


# --------------------------------------------------------------------------- #
# Paired uncertainty.
# --------------------------------------------------------------------------- #

def test_bootstrap_intervals_are_deterministic_and_bracket_point_estimate():
    cases = [
        case(0, gold="high risk", final="high risk"),
        case(1, gold="high risk", final="high risk"),
        case(2, gold="mid risk", final="mid risk"),
        case(3, gold="mid risk", outcome=INVALID, final=None),
    ]
    first = bootstrap_intervals(cases, MODEL, n_boot=500, seed=20261006)
    second = bootstrap_intervals(cases, MODEL, n_boot=500, seed=20261006)
    assert first == second
    assert first["method"].startswith("nonparametric row bootstrap")
    assert first["n_rows"] == 4
    acc = first["accuracy"]
    assert acc["lo"] <= 0.75 <= acc["hi"]
    hr = first["high_risk_recall"]
    assert hr["lo"] <= 1.0 <= hr["hi"]
    # A different seed may resample differently, but the point estimates hold.
    other = bootstrap_intervals(cases, MODEL, n_boot=500, seed=7)
    assert other["accuracy"]["point"] == acc["point"]


def test_mcnemar_exact_matches_binomial_test():
    a = [True, True, True, False, True, True]
    b = [True, False, False, False, True, False]
    res = mcnemar_exact(a, b)
    # a-only-correct = 3 (idx 1, 2, 5), b-only-correct = 0 -> discordant n=3.
    assert res["a_only_correct"] == 3
    assert res["b_only_correct"] == 0
    expected = binomtest(0, 3, 0.5).pvalue
    assert res["p_value"] == pytest.approx(expected)
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0

    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])


# --------------------------------------------------------------------------- #
# Duplicate-feature sensitivity.
# --------------------------------------------------------------------------- #

def test_dedup_row_indices_two_views():
    dedup = dedup_row_indices(records())
    # Complete-row view: identical (features, label) rows collapse to first.
    # (i, low risk) and (i, high risk) are distinct *complete* rows.
    assert dedup["distinct_complete_rows"] == 5
    assert dedup["distinct_complete_row_indices"] == [0, 2, 3, 4, 5]
    # Feature-vector view: conflicting-label groups are excluded entirely.
    assert dedup["conflicting_feature_groups"] == 1
    assert dedup["conflicting_rows"] == 2
    assert dedup["nonconflicting_feature_vectors"] == 3
    assert sorted(dedup["nonconflicting_feature_vector_indices"]) == [0, 2, 3]


def test_dedup_population_changes_denominator_but_not_method():
    recs = records()
    dedup = dedup_row_indices(recs)
    cases = [
        case(0, gold="high risk", final="high risk"),
        case(1, gold="high risk", final="mid risk"),
        case(2, gold="mid risk", final="mid risk"),
        case(3, gold="mid risk", final="low risk"),
        case(4, gold="low risk", final="low risk"),
        case(5, gold="high risk", final="high risk"),
    ]
    primary = all_attempted_metrics(cases, MODEL)
    subset_metrics = metrics.all_attempted_metrics_for_rows(
        cases, MODEL, dedup["distinct_complete_row_indices"]
    )
    assert primary["coverage"]["attempted"] == 6
    assert subset_metrics["coverage"]["attempted"] == 5
    # Rows 0,2,4,5 are correct in the deduped view: 4/5.
    assert subset_metrics["accuracy"] == pytest.approx(4 / 5)
    conflict_free = metrics.all_attempted_metrics_for_rows(
        cases, MODEL, dedup["nonconflicting_feature_vector_indices"]
    )
    assert conflict_free["coverage"]["attempted"] == 3
