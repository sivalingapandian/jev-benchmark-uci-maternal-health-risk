"""Offline assembly of the published benchmark report from a completed run.

Reads a completed run's durable, non-secret artifacts (``run.json``,
``cases.jsonl`` and, when present, ``stages.jsonl``) and derives every published
figure deterministically - coverage, all-attempted and success-only metrics, the
3x4 confusion with an explicit invalid column, latency at both scopes, billed vs
rate-estimated vs unknown cost, paired uncertainty, duplicate-feature
sensitivity, and exploratory stage-1 probability behaviour. No paid request is
ever made, no credential is read, and nothing is written to the artifact
directory.

A run that is not complete, or whose dataset / frozen-task identity does not
match the locally prepared data, is refused with a clear message: partial or
mismatched results must never be presented as the completed benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import metrics, spec
from .dataset import PreparedDataset
from .errors import ReportInputError
from .runner import COMPLETE, CaseRecord, RunState, provider_kind


@dataclass
class RunArtifacts:
    """A loaded, validated completed run."""

    dir: Path
    state: RunState
    cases: list[CaseRecord]
    stage_entries: list[dict] | None


# --------------------------------------------------------------------------- #
# Artifact loading and validation.
# --------------------------------------------------------------------------- #

def load_run_artifacts(from_dir: str | Path) -> RunArtifacts:
    """Load and validate a completed run's durable artifacts (offline).

    Refuses, with a clear message, a missing directory, a missing/empty
    ``cases.jsonl``, a non-``complete`` status, duplicate row-provider pairs,
    cases outside the planned population, or cases whose gold labels are not
    the published classes.
    """
    run_dir = Path(from_dir)
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise ReportInputError(
            f"no run.json under {run_dir}; a report needs a completed run's "
            "durable artifacts (see 'jev-benchmark run' or the committed published/ directory)"
        )
    state = RunState.from_dict(json.loads(run_path.read_text()))

    cases_path = run_dir / "cases.jsonl"
    if not cases_path.is_file():
        raise ReportInputError(
            f"no cases.jsonl under {run_dir}; the run header exists but no "
            "row-provider outcomes were found"
        )
    cases: list[CaseRecord] = []
    for line in cases_path.read_text().splitlines():
        if line.strip():
            cases.append(CaseRecord.from_dict(json.loads(line)))
    if not cases:
        raise ReportInputError(f"cases.jsonl under {run_dir} is empty; nothing to report")

    if state.status != COMPLETE:
        raise ReportInputError(
            f"run {state.run_id} has status {state.status!r}, not 'complete'; "
            "partial results cannot be reported as the completed benchmark"
        )

    keys = [c.key for c in cases]
    if len(keys) != len(set(keys)):
        seen: set = set()
        duplicates = [k for k in keys if k in seen or seen.add(k)]
        raise ReportInputError(
            f"duplicate row-provider pairs in {cases_path} "
            f"(e.g. {duplicates[:3]}); the artifact file is inconsistent"
        )

    planned_rows = set(state.planned_row_indices)
    planned_models = set(state.model_ids)
    outside = [
        c.key for c in cases
        if c.row_index not in planned_rows or c.requested_model not in planned_models
    ]
    if outside:
        raise ReportInputError(
            f"cases outside the planned population in {cases_path} (e.g. {outside[:3]}); "
            "the run header and its outcomes disagree"
        )
    covered = {(c.row_index, c.requested_model) for c in cases}
    if len(covered) != state.planned_pairs:
        raise ReportInputError(
            f"run {state.run_id} covers {len(covered)} of {state.planned_pairs} planned "
            "row-provider pairs; only a complete run can be reported"
        )

    bad_gold = [
        (c.row_index, c.requested_model, c.gold_label)
        for c in cases
        if c.gold_label not in spec.RISK_LABELS
    ]
    if bad_gold:
        raise ReportInputError(
            f"cases with gold labels outside the published classes cannot be scored "
            f"(first offenders: {bad_gold[:3]})"
        )

    stages_path = run_dir / "stages.jsonl"
    stage_entries: list[dict] | None = None
    if stages_path.is_file():
        stage_entries = [
            json.loads(line)
            for line in stages_path.read_text().splitlines()
            if line.strip()
        ]

    return RunArtifacts(dir=run_dir, state=state, cases=cases, stage_entries=stage_entries)


def _as_state(source: RunArtifacts | RunState | str | Path) -> RunState:
    if isinstance(source, RunState):
        return source
    if isinstance(source, RunArtifacts):
        return source.state
    return load_run_artifacts(source).state


def verify_identity(source: RunArtifacts | RunState | str | Path, dataset: PreparedDataset) -> None:
    """Refuse unless the run's frozen identity matches the locally prepared data.

    The ordered-record hash ties the run to exactly this dataset content and
    order; the config hash ties it to the current frozen task definition
    (wording, serialization, schema, models). A mismatch means the saved
    outcomes belong to a different experiment and must not be re-reported.
    """
    state = _as_state(source)
    expected = dataset.provenance
    if state.ordered_record_sha256 != expected["ordered_record_sha256"]:
        raise ReportInputError(
            "dataset identity mismatch: the run was recorded for ordered-record "
            f"sha256 {state.ordered_record_sha256}, but the locally prepared data has "
            f"{expected['ordered_record_sha256']}; re-prepare the same dataset the run used"
        )
    if state.config_hash != expected["config_hash"]:
        raise ReportInputError(
            "frozen task identity mismatch: the run was recorded for config hash "
            f"{state.config_hash}, but the current task definition hashes to "
            f"{expected['config_hash']}; the saved outcomes belong to a different "
            "task wording/schema and cannot be re-reported under this one"
        )


# --------------------------------------------------------------------------- #
# Stage-journal gating verification.
# --------------------------------------------------------------------------- #

def verify_stage_journal(
    stage_entries: list[dict] | None, cases: list[CaseRecord]
) -> dict:
    """Re-verify the provider-owned two-stage gating from the stage journal.

    Checks that every journaled stage 2 followed this provider's own stage-1
    ``other`` answer (never a ``high``), that no stage 2 exists after a
    ``high``, and that the journal reconciles with the successful attempts
    recorded in the cases. A gold label or another provider's answer can never
    appear here, so this re-proves the branch semantics from saved data alone.
    """
    if stage_entries is None:
        return {
            "provided": False,
            "gating_consistent": None,
            "journaled_stage_entries": 0,
            "keys_with_stage2": 0,
            "stage1_high_keys": 0,
            "note": "no stages.jsonl in this run directory; gating not re-verified",
        }

    by_key: dict[tuple[int, str], dict[str, dict]] = {}
    for entry in stage_entries:
        key = (entry["row_index"], entry["requested_model"])
        stage = entry["stage"]
        stages = by_key.setdefault(key, {})
        if stage in stages:
            raise ReportInputError(
                f"stage journal has duplicate {stage} entries for (row {key[0]}, {key[1]})"
            )
        stages[stage] = entry["decision"]

    keys_with_stage2 = 0
    stage1_high_keys = 0
    for key, stages in by_key.items():
        stage1 = stages.get("stage1")
        if stage1 is None:
            raise ReportInputError(
                f"stage journal has a stage 2 for (row {key[0]}, {key[1]}) without its stage 1"
            )
        if "stage2" in stages:
            keys_with_stage2 += 1
            if stage1["choice"] != "other":
                raise ReportInputError(
                    f"stage journal gating violation: (row {key[0]}, {key[1]}) ran a stage 2 "
                    f"after this provider's own stage-1 choice {stage1['choice']!r}; only "
                    "an 'other' answer may trigger stage 2"
                )
        if stage1["choice"] == "high":
            stage1_high_keys += 1
            if "stage2" in stages:
                raise ReportInputError(
                    f"stage journal gating violation: (row {key[0]}, {key[1]}) journaled a "
                    "stage 2 after a stage-1 'high' answer"
                )

    # The journal must reconcile with the successful attempts in the cases: one
    # entry per completed stage, no extras, none missing.
    successful: dict[tuple[int, str], int] = {}
    for case in cases:
        n_ok = sum(1 for a in case.attempts if a.status == "ok")
        if n_ok:
            successful[case.key] = n_ok
    journaled = {key: len(stages) for key, stages in by_key.items()}
    if journaled != successful:
        extra = sorted(set(journaled) - set(successful))[:3]
        missing = sorted(set(successful) - set(journaled))[:3]
        raise ReportInputError(
            "stage journal does not reconcile with the recorded case attempts "
            f"(keys only in journal: {extra}; keys only in cases' successful attempts: {missing})"
        )

    return {
        "provided": True,
        "gating_consistent": True,
        "journaled_stage_entries": len(stage_entries),
        "keys_with_stage2": keys_with_stage2,
        "stage1_high_keys": stage1_high_keys,
    }


# --------------------------------------------------------------------------- #
# Exploratory stage-1 probability behaviour.
# --------------------------------------------------------------------------- #

def exploratory_calibration(
    stage_entries: list[dict] | None, cases: list[CaseRecord]
) -> dict:
    """Exploratory stage-1 reported P(high) vs the gold high-vs-other split.

    Self-reported probabilities are not validated calibrated estimates, are not
    guaranteed to agree with the provider's own choice, and Jev's and
    OpenRouter's numbers are not known to share a scale; this section is
    explicitly exploratory and never part of the headline comparison.
    """
    providers: dict[str, dict] = {}
    if stage_entries is None:
        return {
            "qualification": "exploratory only; not a calibration claim",
            "provided": False,
            "providers": providers,
        }
    gold_by_key = {c.key: c.gold_label for c in cases}
    for model in spec.MODEL_IDS:
        p_high: list[float] = []
        gold_flags: list[int] = []
        missing = 0
        for entry in stage_entries:
            if entry["requested_model"] != model or entry["stage"] != "stage1":
                continue
            probabilities = entry["decision"].get("probabilities")
            value = probabilities.get("high") if isinstance(probabilities, dict) else None
            if not isinstance(value, int | float):
                missing += 1
                continue
            gold = gold_by_key.get((entry["row_index"], model))
            if gold is None:
                continue
            p_high.append(float(value))
            gold_flags.append(1 if gold == "high risk" else 0)
        if not p_high:
            providers[model] = {
                "n": 0,
                "brier": None,
                "rows_without_reported_probabilities": missing,
            }
            continue
        import numpy as np

        p = np.asarray(p_high)
        y = np.asarray(gold_flags)
        providers[model] = {
            "n": int(len(p)),
            "brier": float(np.mean((p - y) ** 2)),
            "mean_p_high_when_gold_high": float(p[y == 1].mean()) if (y == 1).any() else None,
            "mean_p_high_when_gold_other": float(p[y == 0].mean()) if (y == 0).any() else None,
            "rows_without_reported_probabilities": missing,
        }
    return {
        "qualification": (
            "Exploratory only: providers' self-reported probabilities are not "
            "validated calibrated estimates, are not guaranteed to be "
            "argmax-consistent with their recorded choices, and Jev's and "
            "OpenRouter's probability scales are not known to be comparable. "
            "Brier = mean((reported P(high) - gold high-vs-other)^2) over "
            "stage-1 rows with reported probabilities."
        ),
        "provided": True,
        "providers": providers,
    }


# --------------------------------------------------------------------------- #
# Report assembly.
# --------------------------------------------------------------------------- #

def _stage_counts(cases: list[CaseRecord], model_id: str) -> dict:
    subset = [c for c in cases if c.requested_model == model_id]
    return {
        "single_stage_cases": sum(1 for c in subset if c.stage_count == 1),
        "two_stage_cases": sum(1 for c in subset if c.stage_count == 2),
        "other_stage_counts": sum(1 for c in subset if c.stage_count not in (1, 2)),
    }


def _accounting(artifacts: RunArtifacts) -> dict:
    cases = artifacts.cases
    billed = sum(metrics.cost_summary(cases, m)["billed_openrouter"] for m in spec.MODEL_IDS)
    estimated = sum(metrics.cost_summary(cases, m)["estimated_jev"] for m in spec.MODEL_IDS)
    unknown = sum(metrics.cost_summary(cases, m)["reserved_unknown"] for m in spec.MODEL_IDS)
    accounting: dict = {
        "full_run_marginal": {
            "billed_openrouter": billed,
            "estimated_jev": estimated,
            "reserved_unknown": unknown,
            "total": billed + estimated + unknown,
            "note": (
                "replayed from this run's per-attempt settlements; excludes the "
                "planning seed, preflight, pilot and other phases"
            ),
        },
        "unknown_usage_attempts": {
            m: metrics.cost_summary(cases, m)["attempts_missing_usage"]
            for m in spec.MODEL_IDS
        },
        "pricing": (artifacts.state.budget or {}).get("pricing"),
        "cost_basis_note": (
            "OpenRouter amounts are billed usage.cost values; Jev amounts are "
            "published-token-rate estimates (Jev exposes no billed-dollar field); "
            "reserved_unknown are conservative retained reservations for attempts "
            "whose billable amount is unknown and are never assumed free"
        ),
    }
    shared_path = artifacts.dir / "shared_budget.json"
    if shared_path.is_file():
        accounting["shared_cumulative"] = {
            **json.loads(shared_path.read_text()),
            "note": (
                "cumulative cross-phase guard state (planning seed + preflight + "
                "pilots + measured runs), committed as a snapshot; wider than this "
                "run's marginal spend and not derivable from its cases alone"
            ),
        }
    return accounting


def build_report(
    artifacts: RunArtifacts,
    dataset: PreparedDataset,
    *,
    n_boot: int = metrics.DEFAULT_BOOTSTRAP_N,
    seed: int = metrics.DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    """Assemble the full JSON-serializable report from validated artifacts."""
    state = artifacts.state
    cases = artifacts.cases
    prov = dataset.provenance
    audit = dataset.audit
    planned_rows = len(state.planned_row_indices)

    providers: dict[str, dict] = {}
    for model in state.model_ids:
        providers[model] = {
            "provider_kind": provider_kind(model),
            "coverage": metrics.coverage_counts(cases, model, planned_rows=planned_rows),
            "all_attempted": metrics.all_attempted_metrics(cases, model, planned_rows=planned_rows),
            "success_only": metrics.success_only_metrics(cases, model),
            "stage_counts": _stage_counts(cases, model),
            "latency": metrics.latency_summary(cases, model),
            "cost": metrics.cost_summary(cases, model),
            "uncertainty": metrics.bootstrap_intervals(cases, model, n_boot=n_boot, seed=seed),
        }

    paired_tests = []
    for model_a, model_b in metrics.provider_pairs():
        if model_a not in state.model_ids or model_b not in state.model_ids:
            continue
        result = metrics.mcnemar_exact(
            metrics.paired_correctness(cases, model_a),
            metrics.paired_correctness(cases, model_b),
        )
        paired_tests.append({"a": model_a, "b": model_b, **result})

    dedup = metrics.dedup_row_indices(dataset.records)
    duplicate_sensitivity = {
        "method": dedup["method"],
        "note": (
            "sensitivity only; the primary published population is always the full "
            "frozen row set with duplicates retained"
        ),
        "primary_population_rows": prov["row_count"],
        "distinct_complete_rows": dedup["distinct_complete_rows"],
        "distinct_feature_vectors": dedup["distinct_feature_vectors"],
        "conflicting_feature_groups": dedup["conflicting_feature_groups"],
        "conflicting_rows": dedup["conflicting_rows"],
        "nonconflicting_feature_vectors": dedup["nonconflicting_feature_vectors"],
        "complete_row_view": {
            m: metrics.all_attempted_metrics_for_rows(
                cases, m, dedup["distinct_complete_row_indices"]
            )
            for m in state.model_ids
        },
        "feature_vector_view": {
            m: metrics.all_attempted_metrics_for_rows(
                cases, m, dedup["nonconflicting_feature_vector_indices"]
            )
            for m in state.model_ids
        },
    }

    return {
        "run": {
            "run_id": state.run_id,
            "mode": state.mode,
            "status": state.status,
            "complete": True,
            "config_hash": state.config_hash,
            "ordered_record_sha256": state.ordered_record_sha256,
            "model_ids": list(state.model_ids),
            "planned_rows": planned_rows,
            "planned_pairs": state.planned_pairs,
            "completed_pairs": len({c.key for c in cases}),
            "total_run_seconds": state.total_run_seconds,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "artifact_dir": str(artifacts.dir),
        },
        "dataset": {
            "source_dataset_url": prov["source_dataset_url"],
            "source_data_url": prov["source_data_url"],
            "license": prov["license"],
            "citation": prov["citation"],
            "retrieval_date": prov["retrieval_date"],
            "is_official_population": prov["is_official_population"],
            "row_count": prov["row_count"],
            "metadata_reported_instances": prov["metadata_reported_instances"],
            "metadata_discrepancy": prov["metadata_discrepancy"],
            "class_distribution": prov["class_distribution"],
            "distinct_complete_rows": audit["distinct_complete_rows"],
            "distinct_feature_vectors": audit["distinct_feature_vectors"],
            "conflicting_label_groups": audit["conflicting_label_groups"],
        },
        "providers": providers,
        "paired_tests": paired_tests,
        "duplicate_sensitivity": duplicate_sensitivity,
        "stage_gating": verify_stage_journal(artifacts.stage_entries, cases),
        "exploratory_calibration": exploratory_calibration(
            artifacts.stage_entries, cases
        ),
        "accounting": _accounting(artifacts),
        "interpretation_note": (
            "Retrospective replication of existing public dataset labels; no "
            "clinical validity, patient-care, or calibration claim is made or implied."
        ),
    }


# --------------------------------------------------------------------------- #
# Human-readable printing.
# --------------------------------------------------------------------------- #

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def _print_confusion(matrix: dict[str, dict[str, int]]) -> None:
    header = "        truth \\ predicted"
    for col in (*spec.RISK_LABELS, metrics.INVALID_COLUMN):
        header += f" {col:>11s}"
    print(header)
    for truth in spec.RISK_LABELS:
        line = f"        {truth:>19s}"
        for col in (*spec.RISK_LABELS, metrics.INVALID_COLUMN):
            line += f" {matrix[truth][col]:>11d}"
        print(line)


def print_report(report: dict) -> None:
    """Print the report as human-readable tables (values only, no secrets)."""
    run = report["run"]
    data = report["dataset"]
    print(f"Benchmark report - run {run['run_id']} (mode {run['mode']}, status {run['status']})")
    print(f"  completed row-provider pairs : {run['completed_pairs']}/{run['planned_pairs']}")
    print(f"  config hash                  : {run['config_hash']}")
    print(f"  ordered-record sha256        : {run['ordered_record_sha256']}")
    population = (
        "official UCI population" if data["is_official_population"]
        else "NON-OFFICIAL local CSV"
    )
    print(
        f"  dataset                      : {data['row_count']} rows "
        f"({population}), "
        f"classes {data['class_distribution']}"
    )
    print(
        f"  UCI metadata reports {data['metadata_reported_instances']} instances; "
        f"the served CSV has {data['row_count']} rows - all rows retained"
    )
    print(
        f"  duplicates                    : "
        f"{data['distinct_complete_rows']} distinct complete rows, "
        f"{data['distinct_feature_vectors']} distinct feature vectors, "
        f"{data['conflicting_label_groups']} conflicting-label groups "
        "(primary population unchanged)"
    )
    print(
        f"  total run time               : {run['total_run_seconds']:.1f}s "
        "(cumulative active dispatch; excludes any wall-clock gap between segments)"
    )
    print()

    for model, prov in report["providers"].items():
        cov = prov["coverage"]
        aa = prov["all_attempted"]
        so = prov["success_only"]
        lat = prov["latency"]
        cost = prov["cost"]
        unc = prov["uncertainty"]
        print(f"== {model} ({prov['provider_kind']}) ==")
        unattempted = cov["unattempted"]
        unattempted_str = "n/a" if unattempted is None else str(unattempted)
        print(
            f"  coverage        : attempted {cov['attempted']}, predicted {cov['predicted']}, "
            f"invalid/abstained {cov['invalid_or_abstained']}, failed {cov['failed']}, "
            f"unattempted {unattempted_str}"
        )
        print(
            f"  all-attempted   : accuracy {aa['accuracy']:.4f} "
            f"({aa['correct']}/{aa['denominator']}), macro-F1 {aa['macro_f1']:.4f}, "
            f"balanced accuracy {aa['balanced_accuracy']:.4f}, "
            f"high-risk recall {aa['high_risk_recall']:.4f}"
        )
        print("  confusion (rows = truth high/mid/low; invalid column = non-predicted):")
        _print_confusion(aa["confusion"])
        print("  per class (all-attempted):")
        for label in spec.RISK_LABELS:
            cls = aa["classes"][label]
            print(
                f"    {label:<10s} precision {_pct(cls['precision'])}  "
                f"recall {_pct(cls['recall'])}  f1 {_pct(cls['f1'])}  "
                f"(predicted {cls['predicted']}, support {cls['support']})"
            )
        for note in aa["zero_division_notes"]:
            print(f"    zero division: {note}")
        print(
            f"  success-only    : coverage {so['predicted']}/{so['attempted']} "
            f"({so['coverage_rate'] * 100:.1f}%), accuracy {so['accuracy']:.4f}, "
            f"macro-F1 {so['macro_f1']:.4f}, high-risk recall {so['high_risk_recall']:.4f}"
        )
        stages = prov["stage_counts"]
        print(
            f"  stage counts    : {stages['single_stage_cases']} single-stage, "
            f"{stages['two_stage_cases']} two-stage cases"
        )
        api, wf = lat["api_attempt"], lat["workflow"]
        print(
            f"  latency         : api-attempt p50 {_pct(api['p50'])}s p95 {_pct(api['p95'])}s "
            f"(n={api['count']} finished attempts; retried {lat['retried_attempts']}, "
            f"failed {lat['failed_attempts']}, interrupted {lat['interrupted_attempts']})"
        )
        if lat["failed_attempts"]:
            failed = lat["failed_attempt_latency"]
            print(
                f"                    failed-attempt timing: p50 {_pct(failed['p50'])}s "
                f"max {_pct(failed['max'])}s (n={failed['count']})"
            )
        print(
            f"                    whole workflow p50 {_pct(wf['p50'])}s p95 {_pct(wf['p95'])}s "
            f"(n={wf['count']} terminal row workflows)"
        )
        print(
            f"  cost            : OpenRouter billed {_usd(cost['billed_openrouter'])} | "
            f"Jev rate-estimated {_usd(cost['estimated_jev'])} | "
            f"unknown reserved {_usd(cost['reserved_unknown'])} | "
            f"tokens in {cost['input_tokens']} out {cost['output_tokens']} "
            f"(attempts missing usage: {cost['attempts_missing_usage']})"
        )
        print(
            f"  95% bootstrap CI: accuracy "
            f"[{unc['accuracy']['lo']:.4f}, {unc['accuracy']['hi']:.4f}], "
            f"macro-F1 [{unc['macro_f1']['lo']:.4f}, {unc['macro_f1']['hi']:.4f}], "
            f"high-risk recall "
            f"[{unc['high_risk_recall']['lo']:.4f}, {unc['high_risk_recall']['hi']:.4f}] "
            f"(n_boot={unc['n_boot']}, rows={unc['n_rows']}, seed={unc['seed']})"
        )
        print()

    if report["paired_tests"]:
        print(
            "Paired McNemar tests "
            "(all-attempted correctness; no multiple-comparison correction):"
        )
        for pair in report["paired_tests"]:
            print(
                f"  {pair['a']} vs {pair['b']}: {pair['a']}-only correct {pair['a_only_correct']}, "
                f"{pair['b']}-only correct {pair['b_only_correct']}, p = {pair['p_value']:.6f} "
                f"(n={pair['n_pairs']} paired rows)"
            )
        print()

    sens = report["duplicate_sensitivity"]
    print("Duplicate-feature sensitivity (primary population unchanged):")
    print(
        f"  distinct complete rows: {sens['distinct_complete_rows']} | "
        f"distinct feature vectors: {sens['distinct_feature_vectors']} | "
        f"conflicting-label groups excluded in feature-vector view: "
        f"{sens['conflicting_feature_groups']} ({sens['conflicting_rows']} rows)"
    )
    for model, view in sens["complete_row_view"].items():
        print(
            f"  {model} on {sens['distinct_complete_rows']} distinct complete rows: "
            f"accuracy {view['accuracy']:.4f}, macro-F1 {view['macro_f1']:.4f}, "
            f"high-risk recall {view['high_risk_recall']:.4f}"
        )
    for model, view in sens["feature_vector_view"].items():
        print(
            f"  {model} on {sens['nonconflicting_feature_vectors']} non-conflicting "
            f"feature vectors: "
            f"accuracy {view['accuracy']:.4f}, macro-F1 {view['macro_f1']:.4f}, "
            f"high-risk recall {view['high_risk_recall']:.4f}"
        )
    print()

    gating = report["stage_gating"]
    if gating["provided"]:
        print(
            "Stage-journal gating re-verified from saved data: consistent; "
            f"{gating['journaled_stage_entries']} journaled stages, "
            f"{gating['keys_with_stage2']} stage-2 dispatches all following this "
            f"provider's own stage-1 'other', "
            f"{gating['stage1_high_keys']} stage-1 'high' answers with no stage 2."
        )
    else:
        print(f"Stage gating: {gating['note']}")
    print()

    calib = report["exploratory_calibration"]
    if calib.get("provided"):
        print(
            "Exploratory stage-1 reported-probability behaviour (NOT a calibration claim):"
        )
        for model, values in calib["providers"].items():
            if values.get("n"):
                print(
                    f"  {model}: Brier {values['brier']:.4f} (n={values['n']}), "
                    f"mean P(high) when gold high {values['mean_p_high_when_gold_high']:.4f} "
                    f"vs when gold other {values['mean_p_high_when_gold_other']:.4f}"
                )
            else:
                print(f"  {model}: no reported stage-1 probabilities")
        print()

    acc = report["accounting"]
    marginal = acc["full_run_marginal"]
    print("Accounting (bases kept separate; never conflated):")
    print(
        f"  full-run marginal: OpenRouter billed {_usd(marginal['billed_openrouter'])}, "
        f"Jev rate-estimated {_usd(marginal['estimated_jev'])}, "
        f"unknown reserved {_usd(marginal['reserved_unknown'])}, "
        f"total {_usd(marginal['total'])}"
    )
    if "shared_cumulative" in acc:
        shared = acc["shared_cumulative"]
        print(
            f"  shared cumulative guard: exposure {_usd(shared['exposure'])} of "
            f"${shared['cap']:.2f} "
            f"(seed {shared['planning_seed']:.6f} + billed {shared['billed_openrouter']:.6f} "
            f"+ Jev est {shared['estimated_jev']:.6f} + unknown {shared['reserved_unknown']:.6f})"
        )
    print()
    print(f"Interpretation: {report['interpretation_note']}")
