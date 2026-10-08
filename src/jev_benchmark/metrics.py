"""Conservative metric definitions for the offline benchmark report.

Every function is deterministic and pure: it derives its numbers from recorded
:class:`~jev_benchmark.runner.CaseRecord` values without touching a provider.

Scoring semantics (frozen for the published analysis):

* **All-attempted (primary, conservative):** every attempted row-provider case
  is scored. An invalid/abstained/failed outcome is *incorrect* and a **false
  negative for its true class**, but never a false positive for another risk
  class: it lands in a dedicated ``invalid`` column of its gold-label row.
  Accuracy is ``correct / attempted``; macro-F1 averages the F1 of the three
  published risk classes; balanced accuracy averages their recalls.
* **Success-only (secondary):** the same metric family restricted to cases
  with a valid predicted label, always reported with its narrower coverage.
* **Zero division** is explicit: precision with no predicted positives, recall
  with no support, F1 with zero precision+recall, or accuracy with no attempted
  case is defined as ``0.0`` and recorded in ``zero_division_notes``.
* **Latency scopes stay distinct:** an API attempt (including completed
  failed/retried attempts) is never conflated with a whole two-stage row
  workflow; interrupted attempts without an end time are excluded rather than
  invented.
* **Cost bases are never conflated:** OpenRouter's billed ``usage.cost``, Jev's
  published-token-rate estimate, and retained-unknown exposure stay separate.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import binomtest

from . import spec, timing
from .budget import BILLED, ESTIMATED, RESERVED_UNKNOWN
from .dataset import Record
from .runner import CaseRecord
from .workflow import PREDICTED

# The fourth prediction column of the 3x4 all-attempted confusion display.
INVALID_COLUMN = "invalid"

DEFAULT_BOOTSTRAP_N = 10_000
DEFAULT_BOOTSTRAP_SEED = 20261006

_CONFUSION_COLUMNS = (*spec.RISK_LABELS, INVALID_COLUMN)
_LABEL_INDEX = {label: i for i, label in enumerate(spec.RISK_LABELS)}


# --------------------------------------------------------------------------- #
# Coverage.
# --------------------------------------------------------------------------- #

def coverage_counts(
    cases: list[CaseRecord], model_id: str, planned_rows: int | None = None
) -> dict:
    """Outcome counts for one provider, reconciling against the saved records.

    ``invalid_or_abstained`` groups invalid and refused outcomes (a response
    that could not become a label); ``failed`` groups mismatch and error
    outcomes. ``unattempted`` is the planned-per-provider population minus the
    attempted cases (``None`` when no planned population is given).
    """
    subset = [c for c in cases if c.requested_model == model_id]
    counts = {outcome: 0 for outcome in ("predicted", "invalid", "refused", "mismatch", "error")}
    for case in subset:
        counts[case.outcome] = counts.get(case.outcome, 0) + 1
    return {
        "attempted": len(subset),
        "predicted": counts["predicted"],
        "invalid": counts["invalid"],
        "refused": counts["refused"],
        "mismatch": counts["mismatch"],
        "error": counts["error"],
        "invalid_or_abstained": counts["invalid"] + counts["refused"],
        "failed": counts["mismatch"] + counts["error"],
        "unattempted": None if planned_rows is None else planned_rows - len(subset),
    }


# --------------------------------------------------------------------------- #
# Confusion and class metrics.
# --------------------------------------------------------------------------- #

def _usable_label(value: str | None) -> str | None:
    return value if value in _LABEL_INDEX else None


def confusion_matrix(cases: list[CaseRecord], model_id: str) -> dict[str, dict[str, int]]:
    """All-attempted 3x4 confusion: three truth rows, four prediction columns.

    Rows are the three published gold classes; columns are the three published
    classes plus ``invalid``. A non-predicted outcome (invalid/refused/mismatch/
    error) or a final label outside the published classes lands in the ``invalid``
    column of its gold row: incorrect, a false negative for its true class, and
    never a false positive for another risk class.
    """
    matrix = {
        truth: {column: 0 for column in _CONFUSION_COLUMNS}
        for truth in spec.RISK_LABELS
    }
    for case in cases:
        if case.requested_model != model_id:
            continue
        gold = _usable_label(case.gold_label)
        if gold is None:
            raise ValueError(
                f"case (row {case.row_index}, {model_id}) has gold label "
                f"{case.gold_label!r} outside the published classes and cannot be scored"
            )
        predicted = _usable_label(case.final_label) if case.outcome == PREDICTED else None
        column = predicted if predicted is not None else INVALID_COLUMN
        matrix[gold][column] += 1
    return matrix


def _class_metrics(matrix: dict[str, dict[str, int]]) -> tuple[dict, list[str]]:
    classes: dict[str, dict] = {}
    notes: list[str] = []
    for label in spec.RISK_LABELS:
        tp = matrix[label][label]
        predicted = sum(matrix[truth][label] for truth in spec.RISK_LABELS)
        support = sum(matrix[label].values())
        precision = tp / predicted if predicted else 0.0
        if not predicted:
            notes.append(
                f"{label} precision: no predicted positives in the denominator; "
                "defined as 0.0"
            )
        recall = tp / support if support else 0.0
        if not support:
            notes.append(
                f"{label} recall: no gold support in the denominator; defined as 0.0"
            )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        classes[label] = {
            "tp": tp,
            "predicted": predicted,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return classes, notes


def _headline(matrix: dict[str, dict[str, int]], classes: dict) -> tuple[dict, list[str]]:
    denominator = sum(
        matrix[truth][col] for truth in spec.RISK_LABELS for col in _CONFUSION_COLUMNS
    )
    correct = sum(classes[label]["tp"] for label in spec.RISK_LABELS)
    notes: list[str] = []
    accuracy = correct / denominator if denominator else 0.0
    if not denominator:
        notes.append("accuracy: no attempted cases in the denominator; defined as 0.0")
    recalls = [classes[label]["recall"] for label in spec.RISK_LABELS]
    f1s = [classes[label]["f1"] for label in spec.RISK_LABELS]
    return (
        {
            "correct": correct,
            "denominator": denominator,
            "accuracy": accuracy,
            "macro_f1": sum(f1s) / len(f1s),
            "balanced_accuracy": sum(recalls) / len(recalls),
            "high_risk_recall": classes["high risk"]["recall"],
        },
        notes,
    )


def all_attempted_metrics(
    cases: list[CaseRecord],
    model_id: str,
    planned_rows: int | None = None,
) -> dict:
    """Primary conservative metrics over every attempted case for one provider."""
    matrix = confusion_matrix(cases, model_id)
    classes, notes = _class_metrics(matrix)
    headline, headline_notes = _headline(matrix, classes)
    notes.extend(headline_notes)
    coverage = coverage_counts(cases, model_id, planned_rows=planned_rows)
    return {
        "coverage": coverage,
        "confusion": matrix,
        "classes": classes,
        "zero_division_notes": notes,
        **headline,
    }


def success_only_metrics(cases: list[CaseRecord], model_id: str) -> dict:
    """Secondary metrics restricted to valid predicted labels, with coverage.

    The invalid/abstained rows are excluded from every denominator here; the
    narrower coverage makes clear this view is complementary, not primary.
    """
    coverage = coverage_counts(cases, model_id)
    predicted_cases = [
        c for c in cases
        if c.requested_model == model_id
        and c.outcome == PREDICTED
        and _usable_label(c.final_label) is not None
    ]
    matrix = confusion_matrix(predicted_cases, model_id)
    classes, notes = _class_metrics(matrix)
    headline, headline_notes = _headline(matrix, classes)
    notes.extend(headline_notes)
    n_predicted = len(predicted_cases)
    attempted = coverage["attempted"]
    return {
        "attempted": attempted,
        "predicted": n_predicted,
        "coverage_rate": (n_predicted / attempted) if attempted else 0.0,
        "confusion": matrix,
        "classes": classes,
        "zero_division_notes": notes,
        **headline,
    }


def all_attempted_metrics_for_rows(
    cases: list[CaseRecord], model_id: str, row_indices: list[int]
) -> dict:
    """All-attempted metrics restricted to a subset of the row population.

    Used only for the duplicate-feature sensitivity views; the primary
    published population is always the full frozen row set.
    """
    allowed = set(row_indices)
    subset = [c for c in cases if c.row_index in allowed]
    return all_attempted_metrics(subset, model_id, planned_rows=len(allowed))


# --------------------------------------------------------------------------- #
# Latency.
# --------------------------------------------------------------------------- #

def latency_summary(cases: list[CaseRecord], model_id: str) -> dict:
    """Per-call and per-workflow latency with explicit, distinct denominators.

    ``api_attempt`` covers every *finished* API attempt, including completed
    failed and retried attempts; ``workflow`` covers whole row-provider
    two-stage workflows. Attempts interrupted before an end time have no
    measured duration and are counted separately, never invented.
    """
    subset = [c for c in cases if c.requested_model == model_id]
    attempts = [a for case in subset for a in case.attempts]
    retried = [a for a in attempts if a.attempt > 1]
    failed = [a for a in attempts if a.status != "ok"]
    interrupted = [a for a in attempts if a.duration is None]
    return {
        "api_attempt": timing.summarize([a.duration for a in attempts]),
        "workflow": timing.summarize([c.workflow_duration for c in subset]),
        "api_attempts": len(attempts),
        "workflow_cases": len(subset),
        "retried_attempts": len(retried),
        "failed_attempts": len(failed),
        "interrupted_attempts": len(interrupted),
        "retried_attempt_latency": timing.summarize([a.duration for a in retried]),
        "failed_attempt_latency": timing.summarize([a.duration for a in failed]),
    }


# --------------------------------------------------------------------------- #
# Cost aggregation (bases kept separate by definition).
# --------------------------------------------------------------------------- #

def cost_summary(cases: list[CaseRecord], model_id: str) -> dict:
    """Billed vs rate-estimated vs unknown exposure for one provider.

    OpenRouter's billed ``usage.cost`` amounts, Jev's published-token-rate
    estimates, and retained-unknown reservations are summed separately and
    never conflated. Attempts whose usage never arrived are counted in
    ``attempts_missing_usage`` and their exposure stays in ``reserved_unknown``.
    """
    attempts = [a for c in cases if c.requested_model == model_id for a in c.attempts]
    return {
        "billed_openrouter": sum(a.cost_amount for a in attempts if a.cost_basis == BILLED),
        "estimated_jev": sum(a.cost_amount for a in attempts if a.cost_basis == ESTIMATED),
        "reserved_unknown": sum(
            a.cost_amount for a in attempts if a.cost_basis == RESERVED_UNKNOWN
        ),
        "input_tokens": sum(a.input_tokens or 0 for a in attempts),
        "output_tokens": sum(a.output_tokens or 0 for a in attempts),
        "api_attempts": len(attempts),
        "attempts_missing_usage": sum(
            1 for a in attempts if a.input_tokens is None or a.output_tokens is None
        ),
    }


# --------------------------------------------------------------------------- #
# Paired uncertainty.
# --------------------------------------------------------------------------- #

def _scoring_arrays(cases: list[CaseRecord], model_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Gold/predicted index arrays in frozen row order; predicted -1 = invalid."""
    subset = sorted(
        (c for c in cases if c.requested_model == model_id), key=lambda c: c.row_index
    )
    gold = np.empty(len(subset), dtype=np.int64)
    pred = np.empty(len(subset), dtype=np.int64)
    for i, case in enumerate(subset):
        g = _usable_label(case.gold_label)
        if g is None:
            raise ValueError(
                f"case (row {case.row_index}, {model_id}) has gold label "
                f"{case.gold_label!r} outside the published classes"
            )
        gold[i] = _LABEL_INDEX[g]
        p = _usable_label(case.final_label) if case.outcome == PREDICTED else None
        pred[i] = _LABEL_INDEX[p] if p is not None else -1
    return gold, pred


def _metrics_from_confusion(conf: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised metric columns from (n, 3, 4) confusion tensors.

    Column layout per gold row: [invalid, high, mid, low] (predicted index -1
    shifts to column 0). Applies the same explicit zero-division conventions as
    the scalar metrics.
    """
    tp = conf[:, np.arange(3), np.arange(3) + 1]
    predicted = conf[:, :, 1:].sum(axis=1)  # column sums over gold rows
    support = conf.sum(axis=2)  # row sums over prediction columns
    zeros = np.zeros_like(tp, dtype=float)
    precision = np.divide(tp, predicted, out=zeros.copy(), where=predicted > 0)
    recall = np.divide(tp, support, out=zeros.copy(), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=zeros.copy(),
        where=(precision + recall) > 0,
    )
    correct = tp.sum(axis=1)
    attempted = conf.sum(axis=(1, 2))
    return {
        "accuracy": correct / attempted,
        "macro_f1": f1.mean(axis=1),
        "balanced_accuracy": recall.mean(axis=1),
        "high_risk_recall": recall[:, 0],
    }


def bootstrap_intervals(
    cases: list[CaseRecord],
    model_id: str,
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Nonparametric row bootstrap 95% percentile intervals, deterministic.

    Rows are resampled with replacement at fixed seed; each resample is scored
    with the same all-attempted semantics (invalid outcomes stay incorrect).
    Denominator: this provider's attempted cases in frozen row order.
    """
    gold, pred = _scoring_arrays(cases, model_id)
    n_rows = len(gold)
    point_conf = np.zeros((1, 3, 4), dtype=np.int64)
    np.add.at(point_conf[0], (gold, pred + 1), 1)
    point_metrics = {k: float(v[0]) for k, v in _metrics_from_confusion(point_conf).items()}

    result: dict = {
        "method": (
            "nonparametric row bootstrap (resample attempted rows with replacement; "
            "percentile 95% interval; all-attempted scoring semantics)"
        ),
        "confidence_level": 0.95,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_rows": n_rows,
    }
    if n_rows == 0:
        for name in ("accuracy", "macro_f1", "balanced_accuracy", "high_risk_recall"):
            result[name] = {"point": None, "lo": None, "hi": None}
        return result

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_rows, size=(n_boot, n_rows))
    rows = np.broadcast_to(np.arange(n_boot)[:, None], idx.shape)
    conf = np.zeros((n_boot, 3, 4), dtype=np.int64)
    np.add.at(conf, (rows, gold[idx], pred[idx] + 1), 1)
    sampled = _metrics_from_confusion(conf)
    for name in ("accuracy", "macro_f1", "balanced_accuracy", "high_risk_recall"):
        values = sampled[name]
        result[name] = {
            "point": point_metrics[name],
            "lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5)),
        }
    return result


def mcnemar_exact(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """Exact McNemar test on paired all-attempted correctness.

    Both providers scored the same frozen rows, so the comparison is paired:
    the test uses only the discordant pairs (a-only-correct vs b-only-correct)
    with an exact two-sided binomial p-value. No multiple-comparison
    correction is applied anywhere in this report.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("paired correctness vectors must have the same length")
    a_only = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if a and not b)
    b_only = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if b and not a)
    n_discordant = a_only + b_only
    p_value = (
        1.0
        if n_discordant == 0
        else float(binomtest(min(a_only, b_only), n_discordant, 0.5).pvalue)
    )
    return {
        "method": "exact McNemar (two-sided binomial on discordant pairs; no correction)",
        "n_pairs": len(correct_a),
        "both_correct": sum(1 for a, b in zip(correct_a, correct_b, strict=True) if a and b),
        "both_wrong": sum(1 for a, b in zip(correct_a, correct_b, strict=True) if not a and not b),
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "p_value": p_value,
    }


def paired_correctness(cases: list[CaseRecord], model_id: str) -> list[bool]:
    """All-attempted correctness per row in frozen row order for one provider."""
    gold, pred = _scoring_arrays(cases, model_id)
    return [bool(g == p) for g, p in zip(gold, pred, strict=True)]


def provider_pairs() -> list[tuple[str, str]]:
    return list(combinations(spec.MODEL_IDS, 2))


# --------------------------------------------------------------------------- #
# Duplicate-feature sensitivity.
# --------------------------------------------------------------------------- #

def dedup_row_indices(records: list[Record]) -> dict:
    """Row subsets for the two duplicate-sensitivity views.

    * ``distinct_complete_row_indices`` keeps the first source row of each
      distinct (features, label) complete row, so no label conflict can occur.
    * ``nonconflicting_feature_vector_indices`` keeps the first source row of
      each distinct feature vector whose repeated occurrences never disagree on
      the label; conflicting-label groups are excluded entirely because their
      gold label is ambiguous.

    Both views are sensitivity analyses only; the primary population is always
    the full frozen row set with duplicates retained.
    """
    complete: dict[tuple, int] = {}
    feature_groups: dict[tuple, list[tuple[int, str]]] = {}
    for record in records:
        features = tuple(record.features[c] for c in spec.FEATURE_COLUMNS)
        complete.setdefault((features, record.label), record.index)
        feature_groups.setdefault(features, []).append((record.index, record.label))

    conflicting = {
        features: members
        for features, members in feature_groups.items()
        if len({label for _idx, label in members}) > 1
    }
    nonconflicting_indices = [
        members[0][0]
        for features, members in feature_groups.items()
        if features not in conflicting
    ]
    return {
        "method": (
            "first source row retained per distinct complete row "
            "(features+label); the feature-vector view excludes "
            "conflicting-label groups entirely"
        ),
        "distinct_complete_rows": len(complete),
        "distinct_complete_row_indices": sorted(complete.values()),
        "distinct_feature_vectors": len(feature_groups),
        "conflicting_feature_groups": len(conflicting),
        "conflicting_rows": sum(len(m) for m in conflicting.values()),
        "nonconflicting_feature_vectors": len(nonconflicting_indices),
        "nonconflicting_feature_vector_indices": sorted(nonconflicting_indices),
    }
