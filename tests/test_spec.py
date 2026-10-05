import pytest

from jev_benchmark import spec


def test_feature_columns_are_the_six_measurements():
    assert spec.FEATURE_COLUMNS == (
        "Age",
        "SystolicBP",
        "DiastolicBP",
        "BS",
        "BodyTemp",
        "HeartRate",
    )
    assert spec.LABEL_COLUMN == "RiskLevel"
    assert spec.LABEL_COLUMN not in spec.FEATURE_COLUMNS


def test_model_ids_are_exactly_the_three_approved():
    assert spec.MODEL_IDS == (
        "jev-1.13.0",
        "openai/gpt-4o-mini",
        "google/gemini-2.5-flash",
    )


def test_risk_labels_are_the_three_published_classes():
    assert set(spec.RISK_LABELS) == {"high risk", "mid risk", "low risk"}


def test_serialize_features_returns_only_six_and_drops_risklevel():
    row = {
        "Age": 25,
        "SystolicBP": 130,
        "DiastolicBP": 80,
        "BS": 15.0,
        "BodyTemp": 98.0,
        "HeartRate": 86,
        "RiskLevel": "high risk",
        "extra": "noise",
    }
    out = spec.serialize_features(row)
    assert set(out) == set(spec.FEATURE_COLUMNS)
    assert "RiskLevel" not in out
    assert "extra" not in out


def test_serialize_features_rejects_missing_feature():
    with pytest.raises(KeyError):
        spec.serialize_features({"Age": 25})


def test_compose_label_high_ends_without_stage2():
    assert spec.compose_label("high") == "high risk"


def test_compose_label_other_requires_stage2():
    assert spec.compose_label("other", "mid") == "mid risk"
    assert spec.compose_label("other", "low") == "low risk"
    with pytest.raises(ValueError):
        spec.compose_label("other", None)


def test_compose_label_rejects_unknown_stage1():
    with pytest.raises(ValueError):
        spec.compose_label("maybe")


def test_config_hash_is_stable():
    assert spec.config_hash() == spec.config_hash()
    assert len(spec.config_hash()) == 64
