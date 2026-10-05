"""Deterministic preparation and inspection of the official UCI CSV.

Preparation fetches (or reads) the official dataset 863 CSV, validates it
against the frozen schema, preserves every published row in source order, and
records provenance (byte and ordered-record hashes, class distribution) plus a
repeated-row audit. RiskLevel is retained only as local truth.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import spec
from .errors import SchemaError, SourceError


@dataclass(frozen=True)
class Record:
    """One published row: a stable index, six features, and local-truth label."""

    index: int
    features: dict
    label: str


@dataclass
class PreparedDataset:
    records: list[Record]
    provenance: dict
    audit: dict
    raw_bytes: bytes = field(repr=False, default=b"")

    @property
    def row_count(self) -> int:
        return len(self.records)

    def feature_rows(self) -> list[dict]:
        """Provider-facing view: six measurements only, never the label."""
        return [spec.serialize_features(r.features) for r in self.records]

    def labels(self) -> list[str]:
        return [r.label for r in self.records]


def _native(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _read_source_bytes(source: str | Path | None) -> bytes:
    """Return raw CSV bytes from the official URL (source=None) or a local path."""
    if source is None:
        try:
            import httpx

            response = httpx.get(spec.SOURCE_DATA_URL, timeout=60, follow_redirects=True)
            response.raise_for_status()
            return response.content
        except Exception as exc:  # network, HTTP status, DNS, etc.
            raise SourceError(
                f"could not fetch official source {spec.SOURCE_DATA_URL}: {exc}"
            ) from exc

    path = Path(source)
    if not path.is_file():
        raise SourceError(f"source file not found: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SourceError(f"could not read source file {path}: {exc}") from exc


def _parse(raw: bytes) -> pd.DataFrame:
    try:
        # utf-8-sig strips the BOM the official CSV carries.
        frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except Exception as exc:
        raise SourceError(f"source is not a readable CSV: {exc}") from exc
    if frame.empty:
        raise SourceError("source CSV contains no rows")
    return frame


def _validate_schema(frame: pd.DataFrame) -> None:
    columns = list(frame.columns)
    if columns != list(spec.EXPECTED_COLUMNS):
        missing = [c for c in spec.EXPECTED_COLUMNS if c not in columns]
        extra = [c for c in columns if c not in spec.EXPECTED_COLUMNS]
        raise SchemaError(
            "source schema does not match the frozen specification "
            f"(expected columns in order {list(spec.EXPECTED_COLUMNS)}; "
            f"missing={missing}; unexpected={extra}; got={columns})"
        )
    null_counts = frame.isnull().sum()
    offending = {c: int(null_counts[c]) for c in spec.EXPECTED_COLUMNS if null_counts[c]}
    if offending:
        raise SchemaError(f"source contains null values in required columns: {offending}")


def _validate_labels(frame: pd.DataFrame) -> None:
    observed = set(frame[spec.LABEL_COLUMN].unique())
    invalid = sorted(observed - set(spec.RISK_LABELS))
    if invalid:
        raise SchemaError(
            f"source contains invalid {spec.LABEL_COLUMN} values {invalid}; "
            f"local truth is restricted to {list(spec.RISK_LABELS)}"
        )


def _build_records(frame: pd.DataFrame) -> list[Record]:
    records: list[Record] = []
    for idx, row in enumerate(frame.to_dict(orient="records")):
        features = {col: _native(row[col]) for col in spec.FEATURE_COLUMNS}
        label = str(row[spec.LABEL_COLUMN])
        records.append(Record(index=idx, features=features, label=label))
    return records


def _ordered_record_hash(records: list[Record]) -> str:
    ordered = [
        [*(r.features[c] for c in spec.FEATURE_COLUMNS), r.label] for r in records
    ]
    payload = json.dumps(ordered, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _class_distribution(records: list[Record]) -> dict:
    counts = {label: 0 for label in spec.RISK_LABELS}
    for r in records:
        counts[r.label] += 1
    return counts


def _build_provenance(raw: bytes, records: list[Record], retrieval_date: str | None) -> dict:
    if retrieval_date is None:
        retrieval_date = _dt.datetime.now(_dt.UTC).date().isoformat()
    row_count = len(records)
    discrepancy = (
        f"UCI dataset metadata reports {spec.METADATA_REPORTED_INSTANCES} instances, "
        f"but the served CSV contains {row_count} rows. All {row_count} rows are kept; "
        "no row is silently removed."
    )
    return {
        "source_dataset_url": spec.SOURCE_DATASET_URL,
        "source_data_url": spec.SOURCE_DATA_URL,
        "license": spec.SOURCE_LICENSE,
        "citation": spec.SOURCE_CITATION,
        "retrieval_date": retrieval_date,
        "bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes_length": len(raw),
        "ordered_record_sha256": _ordered_record_hash(records),
        "row_count": row_count,
        "metadata_reported_instances": spec.METADATA_REPORTED_INSTANCES,
        "metadata_discrepancy": discrepancy,
        "schema": list(spec.EXPECTED_COLUMNS),
        "feature_columns": list(spec.FEATURE_COLUMNS),
        "label_column": spec.LABEL_COLUMN,
        "class_distribution": _class_distribution(records),
        "config_hash": spec.config_hash(),
        "spec": spec.task_spec().as_dict(),
    }


def _audit(frame: pd.DataFrame) -> dict:
    features = list(spec.FEATURE_COLUMNS)
    total = int(frame.shape[0])
    distinct_complete = int(frame.drop_duplicates().shape[0])
    distinct_vectors = int(frame[features].drop_duplicates().shape[0])

    group_sizes = frame.groupby(features, sort=False).size()
    repeated_groups = int((group_sizes > 1).sum())
    rows_in_repeated_groups = int(group_sizes[group_sizes > 1].sum())

    label_nunique = frame.groupby(features, sort=False)[spec.LABEL_COLUMN].nunique()
    conflicting_groups = int((label_nunique > 1).sum())

    conflicting_examples = []
    if conflicting_groups:
        conflicting_keys = label_nunique[label_nunique > 1].index[:5]
        for key in conflicting_keys:
            key_tuple = key if isinstance(key, tuple) else (key,)
            mask = np.logical_and.reduce(
                [frame[col] == val for col, val in zip(features, key_tuple, strict=True)]
            )
            labels = sorted(frame.loc[mask, spec.LABEL_COLUMN].unique())
            conflicting_examples.append(
                {
                    "features": {
                        col: _native(val)
                        for col, val in zip(features, key_tuple, strict=True)
                    },
                    "labels": labels,
                }
            )

    return {
        "total_rows": total,
        "distinct_complete_rows": distinct_complete,
        "distinct_feature_vectors": distinct_vectors,
        "repeated_feature_vector_groups": repeated_groups,
        "rows_in_repeated_feature_groups": rows_in_repeated_groups,
        "conflicting_label_groups": conflicting_groups,
        "conflicting_examples": conflicting_examples,
        "note": (
            "Repeated feature vectors and complete-row duplicates remain in the primary "
            "1,014-row population. Deduplication/reweighting is reserved for a separate "
            "sensitivity analysis and must not replace the primary denominator."
        ),
    }


def prepare_dataset(
    source: str | Path | None = None,
    *,
    retrieval_date: str | None = None,
) -> PreparedDataset:
    """Fetch/read, validate, and prepare the official dataset.

    ``source=None`` fetches the official URL; a path reads a local copy. Raises
    :class:`SourceError` for an unreachable/invalid source and
    :class:`SchemaError` for a schema or label violation.
    """
    raw = _read_source_bytes(source)
    frame = _parse(raw)
    _validate_schema(frame)
    _validate_labels(frame)
    records = _build_records(frame)
    provenance = _build_provenance(raw, records, retrieval_date)
    audit = _audit(frame)
    return PreparedDataset(records=records, provenance=provenance, audit=audit, raw_bytes=raw)


def save_prepared(dataset: PreparedDataset, out_dir: str | Path) -> dict:
    """Persist the raw CSV, provenance, and audit under ``out_dir``.

    Returns the written file paths. The raw CSV is written verbatim so the
    recorded ``bytes_sha256`` reproduces from the saved copy.
    """
    out = Path(out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw" / "maternal_health_risk.csv"
    raw_path.write_bytes(dataset.raw_bytes)

    provenance_path = out / "provenance.json"
    provenance_path.write_text(json.dumps(dataset.provenance, indent=2) + "\n")

    audit_path = out / "audit.json"
    audit_path.write_text(json.dumps(dataset.audit, indent=2) + "\n")

    return {
        "raw_csv": str(raw_path),
        "provenance": str(provenance_path),
        "audit": str(audit_path),
    }
