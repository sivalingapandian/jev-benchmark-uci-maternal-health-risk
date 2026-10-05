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


def test_identity_includes_effective_instructions_serialization_and_schema():
    payload = spec.task_spec().as_dict()
    # Effective two-stage wording is part of the inspectable frozen definition.
    assert spec.STAGE1_INSTRUCTIONS in payload["stage1"]["instructions"]
    assert spec.STAGE2_INSTRUCTIONS in payload["stage2"]["instructions"]
    # Feature serialization and output-schema semantics are captured too.
    assert payload["feature_serialization"]["order"] == list(spec.FEATURE_COLUMNS)
    assert set(payload["output_schema"]) >= {"choice", "confidence", "probabilities"}
    assert "stage1" in payload["effective_prompts"]


def test_config_hash_changes_when_instruction_wording_changes(monkeypatch):
    before = spec.config_hash()
    monkeypatch.setitem(spec.STAGE1, "instructions", spec.STAGE1_INSTRUCTIONS + " (reworded)")
    after = spec.config_hash()
    assert before != after


def test_config_hash_changes_when_output_schema_changes(monkeypatch):
    before = spec.config_hash()
    changed = dict(spec.OUTPUT_SCHEMA_FIELDS)
    changed["choice"] = "a different schema description"
    monkeypatch.setattr(spec, "OUTPUT_SCHEMA_FIELDS", changed)
    after = spec.config_hash()
    assert before != after


def test_config_hash_changes_when_feature_serialization_changes(monkeypatch):
    before = spec.config_hash()
    monkeypatch.setattr(spec, "FEATURE_SERIALIZATION_SORT_KEYS", False)
    after = spec.config_hash()
    assert before != after
