"""Command-line entry point.

Subcommands available in this milestone:

* ``prepare`` - fetch/read the official UCI CSV, validate it, and save
  provenance + audit (no paid requests).
* ``inspect`` - show the frozen pre-run definition, provenance, and audit so a
  reader can review the planned population, source, input features, pinned
  model IDs, and two-stage task before any paid evaluation.

* ``preflight`` - make one small, genuine live request to Jev and both
  OpenRouter models with the shared two-stage task, validate each response, and
  record redacted model/schema/usage before any bulk run (needs API keys).

* ``pilot`` - run the two-stage workflow for a small real sample across all
  three providers, record schema/usage/errors and timing, and project a
  conservative full-run cost to decide whether the full run fits the shared
  ``$20`` budget (needs API keys).

* ``run`` - sequential, budgeted, resumable measured run over the prespecified
  rows for all three providers. Refuses to start a full-population run without
  an affordable pilot projection, and stops before the shared cap (needs keys).

* ``run-inspect`` - offline summary of a saved run/pilot directory: outcome
  counts, conservative accounting, latency p50/p95 at both scopes, and whether
  the run is complete (no paid requests).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

from . import pilot as pilot_mod
from . import runinspect, runner, spec
from .budget import CAP_USD, open_shared_budget, validate_cap
from .dataset import PreparedDataset, prepare_dataset, save_prepared
from .errors import (
    BenchmarkError,
    BudgetConfigError,
    HistoricalExposureError,
    ProviderRequestError,
    RunDirectoryError,
    RunIdentityError,
    SourceError,
)
from .preflight import run_preflight_from_env
from .providers import JevAdapter, OpenRouterAdapter
from .providers.base import Usage
from .runner import RunStore

DEFAULT_BUDGET_STATE = "runs/shared-budget.json"


def _print_header(title: str) -> None:
    print(title)
    print("=" * len(title))


def cmd_prepare(args: argparse.Namespace) -> int:
    dataset = prepare_dataset(source=args.source)
    paths = save_prepared(dataset, args.data_dir)
    prov = dataset.provenance
    audit = dataset.audit

    if prov["is_official_population"]:
        _print_header("Prepared official UCI Maternal Health Risk dataset")
    else:
        _print_header("Prepared NON-OFFICIAL local CSV (not the official UCI population)")
        print(
            f"NOTE: this CSV has {prov['row_count']} rows, not the official "
            f"{prov['expected_row_count']}-row UCI population. It cannot authorize a "
            "full paid evaluation or be reported as the official benchmark population."
        )
    print(f"source origin     : {prov['source_origin']}")
    print(f"source            : {prov['source_data_url']}")
    print(f"license           : {prov['license']}")
    print(f"retrieval date    : {prov['retrieval_date']}")
    print(f"rows preserved    : {prov['row_count']}")
    print(f"metadata reports  : {prov['metadata_reported_instances']} instances")
    print(f"  discrepancy     : {prov['metadata_discrepancy']}")
    print(f"bytes sha256      : {prov['bytes_sha256']}")
    print(f"ordered-rec sha256: {prov['ordered_record_sha256']}")
    print(f"config hash       : {prov['config_hash']}")
    print(f"class distribution: {prov['class_distribution']}")
    print(
        "duplicate audit   : "
        f"{audit['distinct_complete_rows']} distinct complete rows, "
        f"{audit['conflicting_label_groups']} conflicting feature groups "
        f"(all rows retained)"
    )
    print("written files:")
    for name, path in paths.items():
        print(f"  {name:12s}: {path}")
    return 0


def _load_for_inspect(args: argparse.Namespace) -> PreparedDataset:
    """Load saved artifacts if present; otherwise prepare offline."""
    data_dir = Path(args.data_dir)
    provenance_path = data_dir / "provenance.json"
    audit_path = data_dir / "audit.json"
    raw_path = data_dir / "raw" / "maternal_health_risk.csv"

    if provenance_path.is_file() and audit_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        audit = json.loads(audit_path.read_text())
        return PreparedDataset(records=[], provenance=provenance, audit=audit)

    # No saved artifacts: prepare offline from an explicit source or the cached
    # raw CSV. inspect never issues paid requests, but may fetch the public CSV
    # only when explicitly given no local copy.
    if args.source:
        return prepare_dataset(source=args.source)
    if raw_path.is_file():
        return prepare_dataset(source=raw_path)
    raise SourceError(
        f"no prepared artifacts under {data_dir} and no cached CSV at {raw_path}; "
        "run 'jev-benchmark prepare' first or pass --source"
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    dataset = _load_for_inspect(args)
    prov = dataset.provenance
    audit = dataset.audit
    task = spec.task_spec().as_dict()

    n_models = len(task["model_ids"])
    _print_header("Frozen pre-run benchmark definition")
    print("Task: reproduce existing UCI Maternal Health Risk labels (retrospective, non-clinical).")
    print()
    print(f"Planned population : {prov['row_count']} source rows x {n_models} providers")
    print(f"                     = {prov['row_count'] * n_models} row-provider pairs")
    print(f"Source dataset     : {prov['source_dataset_url']}")
    print(f"Source CSV         : {prov['source_data_url']}")
    print(f"License/citation   : {prov['license']} | {prov['citation']}")
    print()
    print(f"Input features ({len(task['feature_columns'])}): {task['feature_columns']}")
    print(
        f"Local-truth labels : {task['risk_labels']} "
        f"(column '{task['label_column']}', never sent to a provider)"
    )
    print()
    print("Pinned model IDs:")
    for model in task["model_ids"]:
        print(f"  - {model}")
    print()
    print("Two-stage task definition:")
    print(f"  stage 1: {task['stage1']['name']} -> choices {task['stage1']['choices']}")
    print(f"  stage 2: {task['stage2']['name']} -> choices {task['stage2']['choices']}")
    print(f"           gated by: {task['stage2']['gated_by']}")
    print(f"  mapping: {task['label_mapping']}")
    print()
    _print_header("Frozen identifiers")
    print(f"config hash        : {prov['config_hash']}")
    print(f"bytes sha256       : {prov['bytes_sha256']}")
    print(f"ordered-rec sha256 : {prov['ordered_record_sha256']}")
    print()
    _print_header("Provenance and anomalies")
    print(
        f"rows preserved     : {prov['row_count']} "
        f"(UCI metadata reports {prov['metadata_reported_instances']})"
    )
    print(f"  {prov['metadata_discrepancy']}")
    print(f"class distribution : {prov['class_distribution']}")
    print()
    _print_header("Repeated-row audit")
    print(f"total rows                : {audit['total_rows']}")
    print(f"distinct complete rows    : {audit['distinct_complete_rows']}")
    print(f"distinct feature vectors  : {audit['distinct_feature_vectors']}")
    print(f"repeated feature groups   : {audit['repeated_feature_vector_groups']}")
    print(f"conflicting label groups  : {audit['conflicting_label_groups']}")
    print(f"note: {audit['note']}")
    return 0


def _load_env() -> dict:
    """Read the process environment, loading a git-ignored .env if available.

    Keys are only read to authenticate intended API calls; they are never
    printed, logged, or persisted.
    """
    env = dict(os.environ)
    try:
        from dotenv import dotenv_values

        for key, value in dotenv_values().items():
            if value is not None and not env.get(key):
                env[key] = value
    except Exception:
        # python-dotenv is optional at runtime; the process env still applies.
        pass
    return env


def cmd_preflight(args: argparse.Namespace) -> int:
    env = _load_env()
    result = run_preflight_from_env(env)

    _print_header("Live three-provider preflight")
    print(f"config hash : {result.config_hash}")
    print("Bounded shared two-stage task against a synthetic, non-patient record.")
    print()
    for provider in result.providers:
        status = "OK" if provider.ok else "FAILED"
        print(f"[{status}] {provider.requested_model}")
        print(f"    returned model : {provider.actual_model}")
        print(f"    outcome        : {provider.outcome}")
        print(f"    final label    : {provider.final_label}")
        print(f"    stages run     : {provider.stage_count}")
        usage = provider.usage_totals
        cost = usage["cost"]
        cost_str = f"${cost:.8f}" if cost is not None else "n/a (estimated from token rate)"
        print(
            f"    usage          : in={usage['input_tokens']} "
            f"out={usage['output_tokens']} cost={cost_str}"
        )
        if provider.detail:
            print(f"    detail         : {provider.detail}")
        print()

    # Preflight spends real money, so its exposure joins the one shared ledger
    # before any pilot/run can draw on the remaining budget.
    try:
        ledger = open_shared_budget(
            args.budget_state, cap=CAP_USD, runs_dir=Path(args.budget_state).parent
        )
    except (HistoricalExposureError, BudgetConfigError) as exc:
        print(f"preflight blocked: {exc}", file=sys.stderr)
        return 1
    _charge_preflight(ledger, result)
    print()
    _print_accounting(ledger.as_dict())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"preflight-{stamp}.json"
    out_path.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    print(f"redacted preflight record written to {out_path}")

    if result.ok:
        print("preflight passed: all three approved models validated; bulk run may proceed.")
        return 0
    print(
        "preflight FAILED: at least one approved model could not be validated; "
        "bulk execution is blocked.",
        file=sys.stderr,
    )
    return 1


# --------------------------------------------------------------------------- #
# Shared helpers for the paid pilot/run commands.
# --------------------------------------------------------------------------- #
def _load_dataset_offline(args: argparse.Namespace) -> PreparedDataset:
    """Rebuild the prepared dataset deterministically from the cached raw CSV.

    No network request is made when a cached CSV (or explicit --source) exists,
    so the frozen ordered-record and config hashes come straight from the
    already-prepared official data.
    """
    source = getattr(args, "source", None)
    if source:
        return prepare_dataset(source=source)
    raw_path = Path(args.data_dir) / "raw" / "maternal_health_risk.csv"
    if raw_path.is_file():
        return prepare_dataset(source=raw_path)
    raise SourceError(
        f"no cached CSV at {raw_path}; run 'jev-benchmark prepare' first or pass --source"
    )


def _dataset_rows(dataset: PreparedDataset) -> list[dict]:
    return [
        {"index": r.index, "features": r.features, "label": r.label}
        for r in dataset.records
    ]


def _build_live_adapters(env: dict) -> dict:
    """Build the three approved live adapters or raise if a credential is absent."""
    typesafe_key = env.get("TYPESAFE_API_KEY")
    openrouter_key = env.get("OPENROUTER_API_KEY")
    return {
        "jev-1.13.0": JevAdapter(typesafe_key),
        "openai/gpt-4o-mini": OpenRouterAdapter("openai/gpt-4o-mini", openrouter_key),
        "google/gemini-2.5-flash": OpenRouterAdapter(
            "google/gemini-2.5-flash", openrouter_key
        ),
    }


def _close_adapters(adapters: dict) -> None:
    for adapter in adapters.values():
        try:
            adapter.close()
        except Exception:
            pass


def _charge_preflight(ledger, result) -> None:
    """Fold observed preflight usage into the shared ledger.

    Each provider's aggregate usage is reserved then settled: OpenRouter to its
    billed cost when reported (else retained-unknown), Jev to a published-rate
    token estimate. A near-cap refusal would raise, which is acceptable because
    preflight must not proceed past the shared guard either.
    """
    for provider in result.providers:
        totals = provider.usage_totals
        usage = Usage(
            input_tokens=totals.get("input_tokens"),
            output_tokens=totals.get("output_tokens"),
            cost=totals.get("cost"),
        )
        token = ledger.reserve()
        if runner.provider_kind(provider.requested_model) == runner.JEV_KIND:
            ledger.settle_estimated(token, usage)
        else:
            ledger.settle_openrouter(token, usage)


def _print_accounting(budget: dict) -> None:
    print("accounting (conservative, redacted):")
    print(f"  planning seed        : ${budget.get('planning_seed', 0.0):.6f}")
    print(f"  OpenRouter billed    : ${budget.get('billed_openrouter', 0.0):.6f}")
    print(f"  Jev rate-estimated   : ${budget.get('estimated_jev', 0.0):.6f}")
    print(f"  unknown reserved     : ${budget.get('reserved_unknown', 0.0):.6f}")
    print(f"  exposure             : ${budget.get('exposure', 0.0):.6f}")
    print(f"  cap                  : ${budget.get('cap', CAP_USD):.2f}")
    print(f"  remaining            : ${budget.get('remaining', 0.0):.6f}")


def cmd_pilot(args: argparse.Namespace) -> int:
    env = _load_env()
    dataset = _load_dataset_offline(args)
    rows = _dataset_rows(dataset)[: args.rows]
    config_hash = dataset.provenance["config_hash"]
    ordered_hash = dataset.provenance["ordered_record_sha256"]

    # Fail safely before any paid call: refuse an over-cap guard or an occupied
    # directory, and open the one shared ledger (seed + reconciled history).
    try:
        cap = validate_cap(args.budget_cap)
    except BudgetConfigError as exc:
        print(f"pilot blocked: {exc}", file=sys.stderr)
        return 1
    store = RunStore(args.run_dir)
    if store.exists():
        print(
            f"pilot blocked: run directory {store.dir} is already occupied; "
            "choose a fresh --run-dir.",
            file=sys.stderr,
        )
        return 1
    try:
        ledger = open_shared_budget(
            args.budget_state, cap=cap, runs_dir=Path(args.budget_state).parent
        )
    except (HistoricalExposureError, BudgetConfigError) as exc:
        print(f"pilot blocked: {exc}", file=sys.stderr)
        return 1

    try:
        adapters = _build_live_adapters(env)
    except ProviderRequestError as exc:
        print(f"pilot blocked: {exc}", file=sys.stderr)
        return 1
    try:
        result = runner.run(
            store=store,
            adapters=adapters,
            rows=rows,
            model_ids=list(spec.MODEL_IDS),
            config_hash=config_hash,
            ordered_record_sha256=ordered_hash,
            ledger=ledger,
            mode="pilot",
            max_retries=args.max_retries,
        )
    finally:
        _close_adapters(adapters)

    cases = result.completed_cases
    ledger = result.ledger
    projection = pilot_mod.project_full_run(
        cases, ledger, list(spec.MODEL_IDS), safety_factor=args.safety_factor
    )

    _print_header("Live pilot")
    print(f"config hash   : {config_hash}")
    print(f"pilot rows    : {len(rows)}  (row-provider pairs attempted: {len(cases)})")
    print(f"run dir       : {store.dir}")
    print()
    for case in cases:
        last = case.attempts[-1] if case.attempts else None
        print(
            f"  row {case.row_index} {case.requested_model}: {case.outcome} "
            f"-> {case.final_label} (gold {case.gold_label}, stages {case.stage_count}, "
            f"actual {last.actual_model if last else None})"
        )
    print()
    _print_accounting(ledger.as_dict())
    print()
    _print_header("Conservative full-run projection")
    for prov in projection.providers:
        print(
            f"  {prov.model_id}: per-row ${prov.per_row_cost:.8f} "
            f"-> projected ${prov.projected_full_cost:.6f}"
        )
    print(
        f"projected full-run cost     : ${projection.projected_full_cost:.6f} "
        f"(x{projection.safety_factor} safety, {projection.full_population_rows} rows)"
    )
    print(f"projected total exposure    : ${projection.projected_total_exposure:.6f}")
    print(f"affordable within ${projection.cap:.2f}: {projection.affordable}")
    print(f"decision: {projection.reason}")
    validated = pilot_mod.validated_models(cases)
    print(f"providers with valid live evidence: {validated}")
    missing_validation = sorted(set(spec.MODEL_IDS) - set(validated))
    if missing_validation:
        print(
            "note: these providers lack valid live evidence and cannot be "
            f"bulk-dispatched: {missing_validation}"
        )

    out = {
        "config_hash": config_hash,
        "ordered_record_sha256": ordered_hash,
        "is_official_population": bool(dataset.provenance.get("is_official_population")),
        "validated_models": validated,
        "run_dir": str(store.dir),
        "projection": projection.as_dict(),
        "budget": ledger.as_dict(),
    }
    proj_path = store.dir / "projection.json"
    proj_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nprojection written to {proj_path}")

    # The pilot itself ran; a non-affordable projection means the full run is
    # gated off, which the run command enforces. The pilot still exits 0 so its
    # recorded evidence is preserved; a stopped-budget pilot exits non-zero.
    if result.stopped:
        print(f"pilot stopped early: {result.stop_reason}", file=sys.stderr)
        return 1
    return 0


def _load_projection(pilot_dir: str) -> dict:
    path = Path(pilot_dir) / "projection.json"
    if not path.is_file():
        raise BenchmarkError(f"no pilot projection at {path}; run 'jev-benchmark pilot' first")
    return json.loads(path.read_text())


def cmd_run(args: argparse.Namespace) -> int:
    env = _load_env()
    dataset = _load_dataset_offline(args)
    all_rows = _dataset_rows(dataset)
    rows = all_rows if args.rows is None else all_rows[: args.rows]
    config_hash = dataset.provenance["config_hash"]
    ordered_hash = dataset.provenance["ordered_record_sha256"]
    is_official_population = bool(dataset.provenance.get("is_official_population"))
    # Default (``--rows`` unset) means the complete benchmark over every source
    # row; that is a full paid evaluation and is only valid for the official
    # 1,014-row population.
    wants_full_population = args.rows is None

    # Fail safely before any paid call: refuse an over-cap guard first.
    try:
        cap = validate_cap(args.budget_cap)
    except BudgetConfigError as exc:
        print(f"run blocked: {exc}", file=sys.stderr)
        return 1

    if wants_full_population and not is_official_population:
        print(
            "run blocked: a full paid evaluation requires the official "
            f"{spec.EXPECTED_ROW_COUNT}-row UCI population, but the prepared data has "
            f"{len(rows)} rows and is not the official population. "
            "Prepare the official dataset first.",
            file=sys.stderr,
        )
        return 1

    # Every measured run is a bulk paid dispatch and must be authorized by an
    # affordable, config-matching pilot that validated all three providers.
    # Neither a narrowed ``--rows`` scope nor any bypass flag can skip this gate.
    if not args.pilot_dir:
        print(
            "run blocked: a measured run requires an affordable pilot projection "
            "(pass --pilot-dir); no bulk paid run starts before a pilot validates "
            "all three providers.",
            file=sys.stderr,
        )
        return 1
    projection = _load_projection(args.pilot_dir)
    if projection.get("config_hash") != config_hash:
        print(
            "run blocked: pilot projection was made for a different frozen "
            "configuration; re-run the pilot.",
            file=sys.stderr,
        )
        return 1
    if projection.get("ordered_record_sha256") != ordered_hash:
        print(
            "run blocked: pilot projection was made for a different dataset "
            "(ordered-record hash mismatch); re-run the pilot on this data.",
            file=sys.stderr,
        )
        return 1
    if not projection.get("projection", {}).get("affordable", False):
        print(
            "run blocked: pilot projects the full run would reach the shared "
            f"budget cap. {projection.get('projection', {}).get('reason', '')}",
            file=sys.stderr,
        )
        return 1
    # Bulk dispatch requires successful validating evidence for all three real
    # provider IDs; an empty, incomplete, or mismatched pilot cannot authorize it.
    validated = set(projection.get("validated_models", []))
    missing = sorted(set(spec.MODEL_IDS) - validated)
    if missing:
        print(
            "run blocked: pilot did not supply valid live evidence for all "
            f"three approved providers (missing/invalid: {missing}). Re-run "
            "the pilot or preflight so every provider validates before any "
            "bulk paid dispatch.",
            file=sys.stderr,
        )
        return 1

    store = RunStore(args.run_dir)
    if not args.resume and store.exists():
        print(
            f"run blocked: run directory {store.dir} is already occupied by a "
            "previous run; choose a fresh --run-dir or pass --resume.",
            file=sys.stderr,
        )
        return 1

    try:
        ledger = open_shared_budget(
            args.budget_state, cap=cap, runs_dir=Path(args.budget_state).parent
        )
    except (HistoricalExposureError, BudgetConfigError) as exc:
        print(f"run blocked: {exc}", file=sys.stderr)
        return 1

    try:
        adapters = _build_live_adapters(env)
    except ProviderRequestError as exc:
        print(f"run blocked: {exc}", file=sys.stderr)
        return 1

    try:
        result = runner.run(
            store=store,
            adapters=adapters,
            rows=rows,
            model_ids=list(spec.MODEL_IDS),
            config_hash=config_hash,
            ordered_record_sha256=ordered_hash,
            ledger=ledger,
            mode="run",
            max_retries=args.max_retries,
            resume=args.resume,
        )
    except RunIdentityError as exc:
        print(f"resume blocked: {exc}", file=sys.stderr)
        return 1
    except RunDirectoryError as exc:
        print(f"run blocked: {exc}", file=sys.stderr)
        return 1
    finally:
        _close_adapters(adapters)

    state = result.state
    _print_header("Measured run")
    print(f"run id        : {state.run_id}")
    print(f"status        : {state.status}")
    print(f"planned pairs : {state.planned_pairs}")
    print(f"newly done    : {result.newly_completed}")
    print(f"total run time: {state.total_run_seconds:.3f}s")
    print()
    _print_accounting(result.ledger.as_dict())
    if result.stopped:
        print(f"\nrun stopped before the cap: {result.stop_reason}", file=sys.stderr)
        print("completed cases remain recorded and will not be re-billed on resume.")
        return 1
    if state.status != runner.COMPLETE:
        print("\nrun is INCOMPLETE: not all planned pairs are covered.")
        print("partial results must not be reported as a completed benchmark.")
    else:
        print("\nrun complete: every planned row-provider pair was evaluated.")
    return 0


def cmd_run_inspect(args: argparse.Namespace) -> int:
    store = RunStore(args.run_dir)
    if not store.exists():
        print(f"error: no run.json under {store.dir}", file=sys.stderr)
        return 2
    summary = runinspect.inspect_run(store)

    _print_header(f"Run inspection: {summary.run_id} ({summary.mode})")
    print(f"status        : {summary.status}")
    print(f"complete      : {summary.complete}")
    print(f"config hash   : {summary.config_hash}")
    print(f"planned pairs : {summary.planned_pairs}")
    print(f"completed     : {summary.completed_pairs}")
    print(f"total run time: {summary.total_run_seconds:.3f}s")
    if summary.stop_reason:
        print(f"stop reason   : {summary.stop_reason}")
    print()
    _print_accounting(summary.accounting)
    print()
    _print_header("Latency p50/p95 (seconds), with sample counts")
    for prov in summary.providers:
        api = prov.latency.api_attempt
        wf = prov.latency.workflow
        print(f"  {prov.model_id}:")
        print(
            f"    api-attempt : p50={_fmt(api['p50'])} p95={_fmt(api['p95'])} "
            f"n={api['count']} (attempts={prov.api_attempts}, retried={prov.retried_attempts}, "
            f"failed={prov.failed_attempts})"
        )
        print(
            f"    workflow    : p50={_fmt(wf['p50'])} p95={_fmt(wf['p95'])} "
            f"n={wf['count']} (cases={prov.attempted_cases})"
        )
        print(f"    outcomes    : {prov.outcome_counts}")
    overall = summary.overall_latency
    print("  overall:")
    print(
        f"    api-attempt : p50={_fmt(overall.api_attempt['p50'])} "
        f"p95={_fmt(overall.api_attempt['p95'])} n={overall.api_attempt['count']}"
    )
    print(
        f"    workflow    : p50={_fmt(overall.workflow['p50'])} "
        f"p95={_fmt(overall.workflow['p95'])} n={overall.workflow['count']}"
    )

    out_path = store.dir / "inspection.json"
    out_path.write_text(json.dumps(summary.as_dict(), indent=2) + "\n")
    print(f"\ninspection written to {out_path}")
    if not summary.complete:
        print("note: this run is NOT a completed full benchmark.")
    return 0


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-benchmark",
        description=(
            "Retrospective benchmark of Jev and two OpenRouter LLMs on the public UCI "
            "Maternal Health Risk labels. Not a clinical tool."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p_prepare = sub.add_parser(
        "prepare",
        help="Fetch/read the official UCI CSV and save provenance + duplicate audit.",
        description="Fetch (or read) the official dataset 863 CSV, validate the frozen "
        "schema and labels, preserve all source rows, and save provenance and audit.",
    )
    p_prepare.add_argument(
        "--source",
        default=None,
        help="Local CSV path to read instead of fetching the official URL.",
    )
    p_prepare.add_argument(
        "--data-dir",
        default="data",
        help="Directory to write raw CSV, provenance.json and audit.json (default: data).",
    )
    p_prepare.set_defaults(func=cmd_prepare)

    p_inspect = sub.add_parser(
        "inspect",
        help="Show the frozen pre-run definition, provenance and duplicate audit.",
        description="Display the planned population, source, input features, pinned model "
        "IDs, two-stage task and frozen identifiers from saved artifacts (no paid requests).",
    )
    p_inspect.add_argument(
        "--data-dir",
        default="data",
        help="Directory holding prepared artifacts (default: data).",
    )
    p_inspect.add_argument(
        "--source",
        default=None,
        help="CSV path to prepare from if no saved artifacts exist.",
    )
    p_inspect.set_defaults(func=cmd_inspect)

    p_preflight = sub.add_parser(
        "preflight",
        help="Make one small live request to all three models and validate it (needs keys).",
        description="Run the bounded shared two-stage task once against Jev and both "
        "OpenRouter models using a synthetic non-patient record. Validate schema, model "
        "identity and usage; a failure blocks bulk execution. Requires API keys and makes "
        "a small paid request.",
    )
    p_preflight.add_argument(
        "--out-dir",
        default="runs/preflight",
        help="Directory for the redacted preflight record (default: runs/preflight).",
    )
    p_preflight.add_argument(
        "--budget-state",
        default=DEFAULT_BUDGET_STATE,
        help="Shared durable budget file (default: runs/shared-budget.json).",
    )
    p_preflight.set_defaults(func=cmd_preflight)

    p_pilot = sub.add_parser(
        "pilot",
        help="Run a small real sample and project a conservative full-run cost (needs keys).",
        description="Dispatch the two-stage workflow for a few rows to all three live "
        "providers, record schema/usage/errors and timing, and project whether the full "
        "1,014-row run fits the shared $20 budget guard. Makes small paid requests.",
    )
    p_pilot.add_argument("--data-dir", default="data", help="Prepared-data directory.")
    p_pilot.add_argument(
        "--source", default=None, help="CSV path to prepare from if no cached CSV."
    )
    p_pilot.add_argument(
        "--run-dir", default="runs/pilot", help="Directory for pilot artifacts."
    )
    p_pilot.add_argument(
        "--rows", type=int, default=2, help="Number of leading rows to pilot (default: 2)."
    )
    p_pilot.add_argument(
        "--max-retries", type=int, default=2, help="Bounded retries per stage (default: 2)."
    )
    p_pilot.add_argument(
        "--budget-cap",
        type=float,
        default=CAP_USD,
        help="Shared budget cap; a value above the approved $20 is refused (default: 20).",
    )
    p_pilot.add_argument(
        "--budget-state",
        default=DEFAULT_BUDGET_STATE,
        help="Shared durable budget file (default: runs/shared-budget.json).",
    )
    p_pilot.add_argument(
        "--safety-factor",
        type=float,
        default=pilot_mod.DEFAULT_SAFETY_FACTOR,
        help="Conservative multiplier on projected per-row cost (default: 2.0).",
    )
    p_pilot.set_defaults(func=cmd_pilot)

    p_run = sub.add_parser(
        "run",
        help="Sequential, budgeted, resumable measured run across all three models (needs keys).",
        description="Dispatch the frozen two-stage task sequentially for the prespecified "
        "rows to every provider, with bounded retries, conservative stop-before-$20 budget "
        "accounting, durable per-attempt records and same-config resume. A full-population "
        "run requires an affordable pilot projection.",
    )
    p_run.add_argument("--data-dir", default="data", help="Prepared-data directory.")
    p_run.add_argument(
        "--source", default=None, help="CSV path to prepare from if no cached CSV."
    )
    p_run.add_argument(
        "--run-dir", required=True, help="Directory for this run's durable artifacts."
    )
    p_run.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Limit to the leading N rows (default: full 1,014-row population).",
    )
    p_run.add_argument(
        "--pilot-dir",
        default=None,
        help="Pilot directory whose affordable projection gates this run (required).",
    )
    p_run.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run, skipping completed pairs (same config required).",
    )
    p_run.add_argument(
        "--max-retries", type=int, default=2, help="Bounded retries per stage (default: 2)."
    )
    p_run.add_argument(
        "--budget-cap",
        type=float,
        default=CAP_USD,
        help="Shared budget cap; a value above the approved $20 is refused (default: 20).",
    )
    p_run.add_argument(
        "--budget-state",
        default=DEFAULT_BUDGET_STATE,
        help="Shared durable budget file (default: runs/shared-budget.json).",
    )
    p_run.set_defaults(func=cmd_run)

    p_run_inspect = sub.add_parser(
        "run-inspect",
        help="Offline summary of a saved run/pilot: outcomes, accounting, latency p50/p95.",
        description="Read run.json and cases.jsonl from a run directory and report outcome "
        "counts, conservative accounting (billed OpenRouter vs rate-estimated Jev vs unknown "
        "reserved), latency p50/p95 at API-attempt and workflow scopes with sample counts, "
        "and whether the run is complete. No paid requests.",
    )
    p_run_inspect.add_argument(
        "--run-dir", required=True, help="Run directory to inspect."
    )
    p_run_inspect.set_defaults(func=cmd_run_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
