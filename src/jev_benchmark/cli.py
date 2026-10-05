"""Command-line entry point.

Subcommands available in this milestone:

* ``prepare`` - fetch/read the official UCI CSV, validate it, and save
  provenance + audit (no paid requests).
* ``inspect`` - show the frozen pre-run definition, provenance, and audit so a
  reader can review the planned population, source, input features, pinned
  model IDs, and two-stage task before any paid evaluation.

Live preflight/pilot/run/report subcommands are added by later milestones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import spec
from .dataset import PreparedDataset, prepare_dataset, save_prepared
from .errors import BenchmarkError, SourceError


def _print_header(title: str) -> None:
    print(title)
    print("=" * len(title))


def cmd_prepare(args: argparse.Namespace) -> int:
    dataset = prepare_dataset(source=args.source)
    paths = save_prepared(dataset, args.data_dir)
    prov = dataset.provenance
    audit = dataset.audit

    _print_header("Prepared official UCI Maternal Health Risk dataset")
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
