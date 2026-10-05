import pytest

from jev_benchmark import spec
from jev_benchmark.dataset import prepare_dataset
from jev_benchmark.errors import SchemaError, SourceError

OFFICIAL_BYTES_SHA256 = "a1f7025719f84715096e0d1f95ae2e56b57809b9b15449e1836c96a7d976ae9b"
OFFICIAL_ORDERED_SHA256 = "ea285a63e5a4c16a5cfa8622db164edd932dc97046dff455b70ba784b43dcffe"


def test_prepare_small_fixture_preserves_rows_and_duplicates(fixtures_dir):
    ds = prepare_dataset(source=fixtures_dir / "valid_small.csv", retrieval_date="2026-10-05")
    # Five rows including an identical feature vector with a conflicting label.
    assert ds.row_count == 5
    assert ds.audit["total_rows"] == 5
    assert ds.audit["conflicting_label_groups"] == 1
    # Duplicate feature vector is kept, not collapsed.
    assert ds.audit["distinct_feature_vectors"] == 4


def test_feature_rows_never_include_risklevel(fixtures_dir):
    ds = prepare_dataset(source=fixtures_dir / "valid_small.csv")
    for row in ds.feature_rows():
        assert set(row) == set(spec.FEATURE_COLUMNS)
        assert "RiskLevel" not in row
    # Local truth is still available separately.
    assert ds.labels()[0] == "high risk"


def test_missing_required_column_fails(fixtures_dir):
    with pytest.raises(SchemaError):
        prepare_dataset(source=fixtures_dir / "missing_column.csv")


def test_renamed_required_column_fails(fixtures_dir):
    with pytest.raises(SchemaError):
        prepare_dataset(source=fixtures_dir / "renamed_column.csv")


def test_invalid_label_fails(fixtures_dir):
    with pytest.raises(SchemaError):
        prepare_dataset(source=fixtures_dir / "bad_label.csv")


def test_null_value_fails(fixtures_dir):
    with pytest.raises(SchemaError):
        prepare_dataset(source=fixtures_dir / "null_value.csv")


def test_missing_source_file_fails():
    with pytest.raises(SourceError):
        prepare_dataset(source="tests/fixtures/does_not_exist.csv")


def test_official_dataset_counts_and_hashes(official_csv):
    ds = prepare_dataset(source=official_csv, retrieval_date="2026-10-05")
    prov = ds.provenance

    # All 1,014 rows retained despite the 1,013 metadata figure.
    assert prov["row_count"] == 1014
    assert ds.row_count == 1014
    assert prov["metadata_reported_instances"] == 1013
    assert "1,013" in prov["metadata_discrepancy"] or "1013" in prov["metadata_discrepancy"]

    # Provenance attribution and hashes.
    assert prov["license"] == "CC BY 4.0"
    assert prov["source_data_url"] == spec.SOURCE_DATA_URL
    assert prov["bytes_sha256"] == OFFICIAL_BYTES_SHA256
    assert prov["ordered_record_sha256"] == OFFICIAL_ORDERED_SHA256
    assert prov["schema"] == list(spec.EXPECTED_COLUMNS)
    assert prov["class_distribution"] == {"high risk": 272, "mid risk": 336, "low risk": 406}
    assert sum(prov["class_distribution"].values()) == 1014

    # Duplicate audit keeps the primary population intact.
    assert ds.audit["distinct_complete_rows"] == 452
    assert ds.audit["conflicting_label_groups"] == 35
    assert ds.audit["total_rows"] == 1014
