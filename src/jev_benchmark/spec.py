"""Frozen experiment specification.

Everything a reader must be able to inspect *before* any paid run lives here:
the permitted input features, the local label truth, the three pinned model
IDs, the two-stage question definition, and the class mapping. These values
are frozen on purpose; a stable ``config_hash`` lets later resume/report code
detect an incompatible specification instead of silently reusing old results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

# The only measurements that may ever reach a provider request. RiskLevel is
# deliberately absent: it is local truth, never model input.
FEATURE_COLUMNS: tuple[str, ...] = (
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
)

LABEL_COLUMN = "RiskLevel"

# Published UCI classes. Local truth is restricted to exactly these strings.
RISK_LABELS: tuple[str, ...] = ("high risk", "mid risk", "low risk")

# All seven columns the official source CSV must present, in order.
EXPECTED_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, LABEL_COLUMN)

# Exactly the approved providers; never silently substituted.
MODEL_IDS: tuple[str, ...] = (
    "jev-1.13.0",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
)

# Canonical two-stage task. Each provider answers stage 1 against the same row;
# only its own ``other`` answer triggers its own stage 2. The instructions are
# frozen prose shared verbatim by every provider so the question semantics are
# equivalent; they are never tuned against measured outcomes.
STAGE1_INSTRUCTIONS = (
    "You are classifying a single maternal-health record described only by six "
    "routine measurements. Decide whether this record is HIGH risk or not. "
    "Answer with exactly one choice: 'high' if it is high risk, or 'other' if it "
    "is not high risk (that is, mid or low risk)."
)
STAGE2_INSTRUCTIONS = (
    "This maternal-health record has already been judged to be NOT high risk. "
    "Using only the six routine measurements, decide whether it is mid risk or "
    "low risk. Answer with exactly one choice: 'mid' or 'low'."
)

STAGE1 = {
    "id": "stage1",
    "name": "high risk rather than mid/low risk?",
    "choices": ("high", "other"),
    "instructions": STAGE1_INSTRUCTIONS,
}
STAGE2 = {
    "id": "stage2",
    "name": "mid risk rather than low risk?",
    "choices": ("mid", "low"),
    "gated_by": "stage1 == other",
    "instructions": STAGE2_INSTRUCTIONS,
}

STAGES = (STAGE1, STAGE2)

# How stage answers compose into a published label.
LABEL_MAPPING = {
    "high": "high risk",
    ("other", "mid"): "mid risk",
    ("other", "low"): "low risk",
}

SOURCE_DATASET_URL = "https://archive.ics.uci.edu/dataset/863/maternal+health+risk"
SOURCE_DATA_URL = "https://archive.ics.uci.edu/static/public/863/data.csv"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_CITATION = (
    "Ahmed, Marzia. (2020). Maternal Health Risk [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5DP5D"
)

# UCI's dataset metadata reports 1,013 instances while the served CSV has 1,014
# rows. The extra row is kept; this value documents the known discrepancy.
METADATA_REPORTED_INSTANCES = 1013
EXPECTED_ROW_COUNT = 1014

# Output-schema semantics every provider request shares: a bounded choice, a
# confidence scalar, and a per-choice probability map. These are folded into the
# experiment identity (see ``config_hash``) so a schema change cannot silently
# share a run or resume with a run built on a different schema.
OUTPUT_SCHEMA_FIELDS: dict[str, str] = {
    "choice": "exactly one of the stage's allowed choices",
    "confidence": "a number in [0, 1]",
    "probabilities": "a mapping from each allowed choice to a number in [0, 1]",
}

# Whether the measurement JSON embedded in a provider prompt is key-sorted. It
# is part of the effective serialization, so a change here changes the identity.
FEATURE_SERIALIZATION_SORT_KEYS = True

# A fixed non-patient sample used only to fingerprint the effective prompt
# wording and feature serialization in the experiment identity. It is never sent
# to a provider during a measured run.
_IDENTITY_SAMPLE_STATE: dict[str, float] = {
    "Age": 30,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "BS": 7.5,
    "BodyTemp": 98.0,
    "HeartRate": 76,
}


@dataclass(frozen=True)
class TaskSpec:
    """Immutable snapshot of the frozen specification."""

    feature_columns: tuple[str, ...]
    label_column: str
    risk_labels: tuple[str, ...]
    model_ids: tuple[str, ...]
    stage1: dict
    stage2: dict

    def as_dict(self) -> dict:
        # The identity deliberately includes the *effective* two-stage
        # instructions, the rendered prompts (which embed feature serialization
        # and the requested output schema), and the output-schema field
        # semantics. A change to any of these must alter ``config_hash`` so a
        # reworded task or an altered schema cannot silently share a run/resume.
        return {
            "feature_columns": list(self.feature_columns),
            "label_column": self.label_column,
            "risk_labels": list(self.risk_labels),
            "model_ids": list(self.model_ids),
            "stage1": {
                "name": self.stage1["name"],
                "choices": list(self.stage1["choices"]),
                "instructions": self.stage1["instructions"],
            },
            "stage2": {
                "name": self.stage2["name"],
                "choices": list(self.stage2["choices"]),
                "gated_by": self.stage2["gated_by"],
                "instructions": self.stage2["instructions"],
            },
            "label_mapping": {
                "high": "high risk",
                "other+mid": "mid risk",
                "other+low": "low risk",
            },
            "output_schema": dict(OUTPUT_SCHEMA_FIELDS),
            "feature_serialization": {
                "order": list(self.feature_columns),
                "sort_keys": FEATURE_SERIALIZATION_SORT_KEYS,
                "sample": serialize_features(_IDENTITY_SAMPLE_STATE),
            },
            "effective_prompts": {
                "stage1": build_user_prompt(self.stage1, _IDENTITY_SAMPLE_STATE),
                "stage2": build_user_prompt(self.stage2, _IDENTITY_SAMPLE_STATE),
            },
        }


def task_spec() -> TaskSpec:
    return TaskSpec(
        feature_columns=FEATURE_COLUMNS,
        label_column=LABEL_COLUMN,
        risk_labels=RISK_LABELS,
        model_ids=MODEL_IDS,
        stage1=STAGE1,
        stage2=STAGE2,
    )


def config_hash() -> str:
    """Deterministic identifier for the frozen specification."""
    payload = json.dumps(task_spec().as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def serialize_features(row: dict) -> dict:
    """Return only the six permitted measurements for a provider request.

    Raises if a required feature is missing. RiskLevel (or any other key) is
    dropped so that label-derived information can never reach a provider.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in row]
    if missing:
        raise KeyError(f"row is missing required feature(s): {missing}")
    return {c: row[c] for c in FEATURE_COLUMNS}


def build_user_prompt(stage: dict, state: dict) -> str:
    """Build the equivalent natural-language prompt for a text-completion provider.

    The same frozen instructions and the same six serialized measurements that a
    structured provider receives are rendered here as one user message. The
    ``RiskLevel`` label is never part of ``state`` and so can never appear.
    """
    choices = stage["choices"]
    measurements = json.dumps(
        serialize_features(state), sort_keys=FEATURE_SERIALIZATION_SORT_KEYS
    )
    return (
        f"{stage['instructions']}\n\n"
        f"Record measurements (JSON): {measurements}\n\n"
        f"Respond with structured JSON. The 'choice' field must be exactly one of "
        f"{list(choices)}. Also report 'confidence' (0-1) and 'probabilities', a "
        f"mapping from each of {list(choices)} to its probability (0-1)."
    )


def compose_label(stage1_answer: str, stage2_answer: str | None = None) -> str:
    """Map provider stage answers to a published label.

    ``high`` ends at ``high risk`` with no stage 2. ``other`` requires a stage-2
    ``mid``/``low`` answer. Any other combination is rejected.
    """
    if stage1_answer == "high":
        return LABEL_MAPPING["high"]
    if stage1_answer == "other":
        if stage2_answer in ("mid", "low"):
            return LABEL_MAPPING[("other", stage2_answer)]
        raise ValueError(
            f"stage-1 'other' requires stage-2 'mid' or 'low', got {stage2_answer!r}"
        )
    raise ValueError(f"invalid stage-1 answer: {stage1_answer!r}")
