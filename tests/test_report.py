"""Offline report tests: artifact loading, identity verification, assembly,
CLI behavior, and reconciliation of the committed published artifacts.

The published-artifacts and notebook tests are skipped when those committed
paths are absent (e.g. before the results milestone) so the pipeline
milestone's suite stays runnable; the results milestone requires them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jev_benchmark import cli, spec
from jev_benchmark import report as report_mod
from jev_benchmark.dataset import prepare_dataset
from jev_benchmark.errors import ReportInputError
from jev_benchmark.runner import COMPLETE, AttemptRecord, CaseRecord, RunState
from jev_benchmark.workflow import INVALID, PREDICTED

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_DIR = REPO_ROOT / "published"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "benchmark.ipynb"
SMALL_CSV = REPO_ROOT / "tests" / "fixtures" / "valid_small.csv"

MODELS = list(spec.MODEL_IDS)

PUBLISHED_READY = (PUBLISHED_DIR / "cases.jsonl").is_file()
NOTEBOOK_READY = PUBLISHED_READY and NOTEBOOK_PATH.is_file()


def make_attempt(model: str, status: str = "ok", dur: float = 1.0) -> AttemptRecord:
    kind = "jev" if model == "jev-1.13.0" else "openrouter"
    return AttemptRecord(
        provider=model,
        provider_kind=kind,
        stage="stage1",
        attempt=1,
        status=status,
        started_at=100.0,
        ended_at=100.0 + dur,
        actual_model=model if status == "ok" else None,
        input_tokens=400 if status == "ok" else None,
        output_tokens=32 if status == "ok" else None,
        cost_basis="estimated" if kind == "jev" else "billed",
        cost_amount=0.00002 if status == "ok" else 0.005,
    )


def make_case(row: int, model: str, outcome: str = PREDICTED, final: str | None = "high risk",
              gold: str = "high risk") -> CaseRecord:
    return CaseRecord(
        row_index=row,
        requested_model=model,
        provider_kind="jev" if model == "jev-1.13.0" else "openrouter",
        outcome=outcome,
        final_label=final,
        gold_label=gold,
        stage_count=1,
        workflow_started_at=100.0,
        workflow_ended_at=102.0,
        attempts=[make_attempt(model, status="ok" if outcome == PREDICTED else "invalid")],
    )


@pytest.fixture
def small_dataset():
    return prepare_dataset(source=SMALL_CSV)


def full_small_cases(dataset) -> list[CaseRecord]:
    """One case per (row, model) pair of the small fixture, mixed outcomes.

    Every provider mispredicts row 3 (gold ``low risk`` -> ``high risk``);
    Gemini additionally returns an invalid outcome on row 4 (gold ``high risk``).
    """
    cases = []
    labels = [r.label for r in dataset.records]
    for row, gold in enumerate(labels):
        for model in MODELS:
            if model == MODELS[2] and row == 4:
                cases.append(make_case(row, model, outcome=INVALID, final=None, gold=gold))
            else:
                final = "high risk" if row == 3 else gold
                cases.append(make_case(row, model, final=final, gold=gold))
    return cases


def write_run_dir(tmp_path: Path, dataset, cases: list[CaseRecord], *, status: str = COMPLETE,
                  planned_rows: list[int] | None = None,
                  config_hash: str | None = None, ordered_hash: str | None = None) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = RunState(
        run_id="run-test",
        mode="run",
        config_hash=config_hash if config_hash is not None else dataset.provenance["config_hash"],
        ordered_record_sha256=(
            ordered_hash if ordered_hash is not None
            else dataset.provenance["ordered_record_sha256"]
        ),
        model_ids=MODELS,
        planned_row_indices=(
            planned_rows if planned_rows is not None
            else sorted({c.row_index for c in cases})
        ),
        status=status,
        total_run_seconds=12.5,
    )
    (run_dir / "run.json").write_text(json.dumps(state.as_dict(), indent=2) + "\n")
    with (run_dir / "cases.jsonl").open("w") as handle:
        for c in cases:
            handle.write(json.dumps(c.as_dict()) + "\n")
    with (run_dir / "stages.jsonl").open("w") as handle:
        for entry in journal_from_cases(cases):
            handle.write(json.dumps(entry) + "\n")
    return run_dir


def write_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "raw" / "maternal_health_risk.csv").write_bytes(SMALL_CSV.read_bytes())
    return data_dir


def write_manifest(run_dir: Path, files: dict, **extra) -> Path:
    manifest = {"description": "test manifest", "files": files, **extra}
    path = run_dir / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def manifest_entry(raw: bytes, *, size_delta: int = 0, sha: str | None = None) -> dict:
    return {
        "bytes": len(raw) + size_delta,
        "sha256": sha if sha is not None else hashlib.sha256(raw).hexdigest(),
    }


def journal_entry(row: int, model: str, stage: str, choice: str) -> dict:
    return {
        "row_index": row,
        "requested_model": model,
        "stage": stage,
        "decision": {
            "stage": stage,
            "requested_model": model,
            "actual_model": model,
            "choice": choice,
            "confidence": 0.5,
            "probabilities": {choice: 0.6},
            "usage": {"input_tokens": 10, "output_tokens": 5, "cost": None},
        },
        "attempts": [],
    }


def journal_from_cases(cases: list[CaseRecord]) -> list[dict]:
    """A journal entry for every successful attempt, as the runner writes."""
    entries = []
    for c in cases:
        for a in c.attempts:
            if a.status == "ok":
                entries.append(journal_entry(c.row_index, c.requested_model, a.stage, "high"))
    return entries


# --------------------------------------------------------------------------- #
# Artifact loading and completeness.
# --------------------------------------------------------------------------- #

def test_missing_run_json_is_a_clear_error(tmp_path):
    with pytest.raises(ReportInputError, match="run.json"):
        report_mod.load_run_artifacts(tmp_path / "nope")


def test_missing_cases_file_is_a_clear_error(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = RunState(run_id="r", mode="run", config_hash="x", ordered_record_sha256="y",
                     model_ids=MODELS, planned_row_indices=[0], status=COMPLETE)
    (run_dir / "run.json").write_text(json.dumps(state.as_dict()))
    with pytest.raises(ReportInputError, match="cases.jsonl"):
        report_mod.load_run_artifacts(run_dir)


def test_incomplete_run_is_refused(tmp_path, small_dataset):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(
        tmp_path, small_dataset, cases[:6], planned_rows=[0, 1, 2]
    )
    with pytest.raises(ReportInputError, match="complete"):
        report_mod.load_run_artifacts(run_dir)


def test_non_complete_status_is_refused(tmp_path, small_dataset):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases, status="incomplete")
    with pytest.raises(ReportInputError, match="complete"):
        report_mod.load_run_artifacts(run_dir)


def test_duplicate_pairs_are_refused(tmp_path, small_dataset):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases + cases[:1])
    with pytest.raises(ReportInputError, match="duplicate"):
        report_mod.load_run_artifacts(run_dir)


def test_scoring_requires_known_gold_labels(tmp_path, small_dataset):
    cases = full_small_cases(small_dataset)
    cases[0].gold_label = "unknown risk"
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    with pytest.raises(ReportInputError, match="gold"):
        report_mod.load_run_artifacts(run_dir)


# --------------------------------------------------------------------------- #
# Identity verification.
# --------------------------------------------------------------------------- #

def test_dataset_identity_mismatch_is_refused(tmp_path, small_dataset):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases, ordered_hash="0" * 64)
    with pytest.raises(ReportInputError, match="dataset"):
        report_mod.verify_identity(run_dir, small_dataset)


def test_config_identity_mismatch_is_refused(tmp_path, small_dataset):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases, config_hash="1" * 64)
    with pytest.raises(ReportInputError, match="task"):
        report_mod.verify_identity(run_dir, small_dataset)


# --------------------------------------------------------------------------- #
# Official-population scope gate.
# --------------------------------------------------------------------------- #

def test_reduced_scope_is_refused(small_dataset, tmp_path):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    artifacts = report_mod.load_run_artifacts(run_dir)
    with pytest.raises(ReportInputError, match="official"):
        report_mod.verify_official_scope(artifacts)


def test_pilot_mode_is_refused(small_dataset, tmp_path):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    state = json.loads((run_dir / "run.json").read_text())
    state["mode"] = "pilot"
    (run_dir / "run.json").write_text(json.dumps(state))
    artifacts = report_mod.load_run_artifacts(run_dir)
    with pytest.raises(ReportInputError, match="mode"):
        report_mod.verify_official_scope(artifacts)


def test_scope_flags_a_doctored_planned_pairs_header(small_dataset, tmp_path):
    """A subset whose header is made self-consistent is still a reduced scope."""
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    artifacts = report_mod.load_run_artifacts(run_dir)
    scope = report_mod.benchmark_scope(artifacts)
    assert scope["is_completed_official_benchmark"] is False
    assert scope["planned_rows"] == 5
    assert scope["official_planned_pairs"] == 3042
    assert scope["reason"]


@pytest.mark.skipif(
    not PUBLISHED_READY,
    reason="published/ artifacts are committed in the results milestone",
)
def test_doctored_published_subset_is_refused(tmp_path, monkeypatch, capsys):
    """A 100-row, planned_pairs=300 subset of published/ is refused, not reported."""
    import shutil

    dst = tmp_path / "subset"
    shutil.copytree(PUBLISHED_DIR, dst)
    (dst / "MANIFEST.json").unlink()  # test the scope gate, not the manifest gate
    state = json.loads((dst / "run.json").read_text())
    state["planned_row_indices"] = list(range(100))
    (dst / "run.json").write_text(json.dumps(state, indent=2) + "\n")
    kept = [
        line for line in (dst / "cases.jsonl").read_text().splitlines()
        if line.strip() and json.loads(line)["row_index"] < 100
    ]
    (dst / "cases.jsonl").write_text("\n".join(kept) + "\n")

    monkeypatch.chdir(REPO_ROOT)
    rc = cli.main(["report", "--from", str(dst), "--data-dir", "data", "--n-boot", "20"])
    assert rc != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "official" in captured.err.lower()
    assert "100" in captured.err


# --------------------------------------------------------------------------- #
# MANIFEST.json integrity.
# --------------------------------------------------------------------------- #

def test_manifest_absent_is_skipped(small_dataset, tmp_path):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    artifacts = report_mod.load_run_artifacts(run_dir)
    assert artifacts.manifest is None


def test_manifest_consistent_files_are_verified(small_dataset, tmp_path):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    write_manifest(run_dir, {
        "run.json": manifest_entry((run_dir / "run.json").read_bytes()),
        "cases.jsonl": manifest_entry((run_dir / "cases.jsonl").read_bytes()),
    }, run_id="run-test")
    artifacts = report_mod.load_run_artifacts(run_dir)
    assert artifacts.manifest["files_verified"] == 2


def test_manifest_size_mismatch_is_refused(small_dataset, tmp_path):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    raw = (run_dir / "cases.jsonl").read_bytes()
    write_manifest(run_dir, {"cases.jsonl": manifest_entry(raw, size_delta=1)})
    with pytest.raises(ReportInputError, match="bytes"):
        report_mod.load_run_artifacts(run_dir)


def test_manifest_hash_mismatch_is_refused(small_dataset, tmp_path):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    raw = (run_dir / "cases.jsonl").read_bytes()
    write_manifest(run_dir, {"cases.jsonl": manifest_entry(raw, sha="0" * 64)})
    with pytest.raises(ReportInputError, match="sha256"):
        report_mod.load_run_artifacts(run_dir)


def test_manifest_missing_listed_file_is_refused(small_dataset, tmp_path):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    write_manifest(run_dir, {"absent.json": {"bytes": 1, "sha256": "0" * 64}})
    with pytest.raises(ReportInputError, match="missing"):
        report_mod.load_run_artifacts(run_dir)


def test_manifest_identity_mismatch_is_refused(small_dataset, tmp_path):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    write_manifest(run_dir, {
        "cases.jsonl": manifest_entry((run_dir / "cases.jsonl").read_bytes()),
    }, run_id="a-different-run")
    with pytest.raises(ReportInputError, match="run_id"):
        report_mod.load_run_artifacts(run_dir)


# --------------------------------------------------------------------------- #
# None-valued optional display fields.
# --------------------------------------------------------------------------- #

def test_exploratory_calibration_none_when_a_gold_class_absent(small_dataset):
    cases = full_small_cases(small_dataset)
    labels = {r.index: r.label for r in small_dataset.records}

    def entry(row: int, gold: str) -> dict:
        return {
            "row_index": row,
            "requested_model": MODELS[0],
            "stage": "stage1",
            "decision": {
                "stage": "stage1",
                "requested_model": MODELS[0],
                "actual_model": MODELS[0],
                "choice": "other" if gold != "high risk" else "high",
                "confidence": 0.5,
                "probabilities": {"high": 0.3, "other": 0.7},
                "usage": {"input_tokens": 10, "output_tokens": 5, "cost": None},
            },
            "attempts": [],
        }

    only_other = [
        entry(c.row_index, labels[c.row_index])
        for c in cases
        if c.requested_model == MODELS[0] and labels[c.row_index] != "high risk"
    ]
    assert only_other  # the fixture has non-high rows
    calib = report_mod.exploratory_calibration(only_other, cases)
    prov = calib["providers"][MODELS[0]]
    assert prov["n"] == len(only_other)
    assert prov["mean_p_high_when_gold_high"] is None
    assert prov["mean_p_high_when_gold_other"] is not None

    only_high = [
        entry(c.row_index, labels[c.row_index])
        for c in cases
        if c.requested_model == MODELS[0] and labels[c.row_index] == "high risk"
    ]
    calib_high = report_mod.exploratory_calibration(only_high, cases)
    prov_high = calib_high["providers"][MODELS[0]]
    assert prov_high["mean_p_high_when_gold_high"] is not None
    assert prov_high["mean_p_high_when_gold_other"] is None


def test_print_report_labels_none_probability_means(small_dataset, tmp_path, capsys):
    """A provider with no stage-1 rows of one gold class prints 'n/a', not a crash."""
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    artifacts = report_mod.load_run_artifacts(run_dir)
    report_mod.verify_identity(artifacts, small_dataset)
    report = report_mod.build_report(artifacts, small_dataset, n_boot=20, seed=1)
    report["exploratory_calibration"]["providers"][MODELS[0]] = {
        "n": 3,
        "brier": 0.25,
        "mean_p_high_when_gold_high": None,
        "mean_p_high_when_gold_other": 0.4,
        "rows_without_reported_probabilities": 0,
    }
    report_mod.print_report(report)  # must not raise
    out = capsys.readouterr().out
    assert "n/a" in out


# --------------------------------------------------------------------------- #
# Stage journal gating verification.
# --------------------------------------------------------------------------- #

def test_stage_journal_gating_consistency(small_dataset):
    cases = full_small_cases(small_dataset)
    good = journal_from_cases(cases)
    summary = report_mod.verify_stage_journal(good, cases)
    assert summary["gating_consistent"] is True
    assert summary["journaled_stage_entries"] == sum(
        1 for c in cases for a in c.attempts if a.status == "ok"
    )

    # A stage 2 must never follow this provider's own stage-1 'high'.
    bad = good + [journal_entry(0, MODELS[0], "stage2", "mid")]
    with pytest.raises(ReportInputError, match="stage"):
        report_mod.verify_stage_journal(bad, cases)


def test_stage_journal_must_match_case_attempts(small_dataset):
    cases = full_small_cases(small_dataset)
    # Two journaled stage-1 successes for one case that recorded only one.
    journal = [journal_entry(0, MODELS[0], "stage1", "high")] * 2
    with pytest.raises(ReportInputError, match="journal"):
        report_mod.verify_stage_journal(journal, cases)


def test_stage_journal_absent_is_reported_not_failed(small_dataset):
    cases = full_small_cases(small_dataset)
    summary = report_mod.verify_stage_journal(None, cases)
    assert summary["provided"] is False


# --------------------------------------------------------------------------- #
# Report assembly.
# --------------------------------------------------------------------------- #

def test_build_report_reconciles_counts(tmp_path, small_dataset):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    artifacts = report_mod.load_run_artifacts(run_dir)
    report_mod.verify_identity(artifacts, small_dataset)
    report = report_mod.build_report(artifacts, small_dataset, n_boot=200, seed=20261006)

    assert report["run"]["run_id"] == "run-test"
    assert report["run"]["planned_pairs"] == 15
    assert report["run"]["complete"] is True
    assert report["dataset"]["row_count"] == 5
    assert report["dataset"]["distinct_complete_rows"] == 5
    assert report["dataset"]["distinct_feature_vectors"] == 4
    assert report["dataset"]["conflicting_label_groups"] == 1
    assert len(report["providers"]) == 3

    for model in MODELS:
        prov = report["providers"][model]
        cov = prov["coverage"]
        assert cov["attempted"] == 5
        assert cov["unattempted"] == 0
        # attempted reconciles with the outcome breakdown
        assert (cov["predicted"] + cov["invalid_or_abstained"] + cov["failed"]) == cov["attempted"]

    gemini = report["providers"][MODELS[2]]
    assert gemini["coverage"]["invalid_or_abstained"] == 1
    assert gemini["all_attempted"]["accuracy"] == pytest.approx(3 / 5)
    assert gemini["success_only"]["coverage_rate"] == pytest.approx(4 / 5)
    assert gemini["success_only"]["accuracy"] == pytest.approx(3 / 4)
    # latency + cost sections exist with explicit denominators
    assert gemini["latency"]["api_attempt"]["count"] == 5
    assert gemini["latency"]["workflow"]["count"] == 5
    assert "billed_openrouter" in gemini["cost"]
    assert gemini["uncertainty"]["n_boot"] == 200

    assert len(report["paired_tests"]) == 3  # three unordered provider pairs
    assert report["duplicate_sensitivity"]["distinct_complete_rows"] == 5
    assert report["stage_gating"]["provided"] is True


def test_report_is_json_serializable(tmp_path, small_dataset):
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    artifacts = report_mod.load_run_artifacts(run_dir)
    report_mod.verify_identity(artifacts, small_dataset)
    report = report_mod.build_report(artifacts, small_dataset, n_boot=50, seed=1)
    text = json.dumps(report, allow_nan=False, indent=2)
    assert "run-test" in text


# --------------------------------------------------------------------------- #
# CLI report command (offline; no keys, no paid calls).
# --------------------------------------------------------------------------- #

def test_cli_report_refuses_reduced_run(tmp_path, small_dataset, capsys, monkeypatch):
    """A reduced, self-consistent artifact set is never presented as complete."""
    cases = full_small_cases(small_dataset)
    run_dir = write_run_dir(tmp_path, small_dataset, cases)
    write_data_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = cli.main([
        "report", "--from", str(run_dir), "--data-dir", "data",
        "--n-boot", "100",
    ])
    assert rc != 0
    captured = capsys.readouterr()
    assert captured.out == ""  # no partial table is printed before refusing
    assert "official" in captured.err.lower()
    assert "run-inspect" in captured.err


def test_cli_report_missing_dir_fails_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["report", "--from", str(tmp_path / "missing"), "--data-dir", "data"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "report" in err.lower()


@pytest.mark.skipif(
    not PUBLISHED_READY,
    reason="published/ artifacts are committed in the results milestone",
)
def test_cli_report_writes_optional_json_for_official_run(tmp_path, monkeypatch, capsys):
    out_path = tmp_path / "report.json"
    monkeypatch.chdir(REPO_ROOT)
    rc = cli.main([
        "report", "--from", "published", "--data-dir", "data",
        "--n-boot", "50", "--out", str(out_path),
    ])
    assert rc == 0
    saved = json.loads(out_path.read_text())
    assert saved["run"]["planned_pairs"] == 3042
    assert saved["scope"]["is_completed_official_benchmark"] is True
    assert saved["manifest"]["files_verified"] == 5


# --------------------------------------------------------------------------- #
# Committed published artifacts (results milestone).
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not PUBLISHED_READY,
    reason="published/ artifacts are committed in the results milestone",
)
class TestPublishedArtifacts:
    def test_published_run_is_complete_and_official(self):
        state = json.loads((PUBLISHED_DIR / "run.json").read_text())
        assert state["status"] == COMPLETE
        assert state["planned_pairs"] == 3042
        assert state["mode"] == "run"
        cases = [json.loads(line) for line in
                 (PUBLISHED_DIR / "cases.jsonl").read_text().splitlines() if line]
        assert len(cases) == 3042
        keys = {(c["row_index"], c["requested_model"]) for c in cases}
        assert len(keys) == 3042

    def test_report_reconciles_with_independent_recount(self, monkeypatch, capsys):
        """Cross-check CLI report values against an independent inline recount."""
        correct: dict[str, int] = {}
        attempted: dict[str, int] = {}
        nonpredicted: dict[str, int] = {}
        for line in (PUBLISHED_DIR / "cases.jsonl").read_text().splitlines():
            c = json.loads(line)
            m = c["requested_model"]
            attempted[m] = attempted.get(m, 0) + 1
            if c["outcome"] != "predicted":
                nonpredicted[m] = nonpredicted.get(m, 0) + 1
            elif c["final_label"] == c["gold_label"]:
                correct[m] = correct.get(m, 0) + 1

        monkeypatch.chdir(REPO_ROOT)
        rc = cli.main(["report", "--from", "published", "--data-dir", "data", "--n-boot", "200"])
        assert rc == 0
        out = capsys.readouterr().out  # a human-readable report was printed
        assert "official UCI population" in out
        assert "5 files verified" in out

        dataset = prepare_dataset(source=REPO_ROOT / "data" / "raw" / "maternal_health_risk.csv")
        artifacts = report_mod.load_run_artifacts(PUBLISHED_DIR)
        report_mod.verify_identity(artifacts, dataset)
        report = report_mod.build_report(artifacts, dataset, n_boot=200, seed=20261006)
        for model in MODELS:
            prov = report["providers"][model]
            assert prov["coverage"]["attempted"] == attempted[model]
            assert (prov["coverage"]["invalid_or_abstained"]
                    + prov["coverage"]["failed"]) == nonpredicted.get(model, 0)
            assert prov["all_attempted"]["accuracy"] == pytest.approx(
                correct.get(model, 0) / attempted[model]
            )
            assert prov["coverage"]["unattempted"] == 0

    def test_published_artifacts_contain_no_secret_like_markers(self):
        blob = "\n".join(
            p.read_text(errors="replace")
            for p in sorted(PUBLISHED_DIR.iterdir())
            if p.is_file() and p.suffix in {".json", ".jsonl"}
        )
        for marker in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "Authorization",
                       "Bearer", "api_key", "sk-"):
            assert marker not in blob, f"secret-like marker {marker!r} found in published/"

    def test_modified_published_artifact_is_refused(self, tmp_path, monkeypatch, capsys):
        """A tampered published/ file fails the MANIFEST SHA-256/size check."""
        import shutil

        dst = tmp_path / "published"
        shutil.copytree(PUBLISHED_DIR, dst)
        with (dst / "cases.jsonl").open("ab") as handle:
            handle.write(b" \n")
        monkeypatch.chdir(REPO_ROOT)
        rc = cli.main(["report", "--from", str(dst), "--data-dir", "data", "--n-boot", "20"])
        assert rc != 0
        assert "MANIFEST" in capsys.readouterr().err

    def test_tampered_published_manifest_is_refused(self, tmp_path, monkeypatch, capsys):
        import shutil

        dst = tmp_path / "published"
        shutil.copytree(PUBLISHED_DIR, dst)
        manifest = json.loads((dst / "MANIFEST.json").read_text())
        manifest["files"]["cases.jsonl"]["sha256"] = "0" * 64
        (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        monkeypatch.chdir(REPO_ROOT)
        rc = cli.main(["report", "--from", str(dst), "--data-dir", "data", "--n-boot", "20"])
        assert rc != 0
        assert "sha256" in capsys.readouterr().err


@pytest.mark.skipif(
    not NOTEBOOK_READY,
    reason="notebook and published/ are committed in the results milestone",
)
def test_notebook_executes_offline_without_keys(monkeypatch, tmp_path):
    """VAL-REPORT-005: the notebook runs with API keys unavailable, offline."""
    import os

    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    for var in list(os.environ):
        if "API_KEY" in var or "TOKEN" in var:
            monkeypatch.delenv(var, raising=False)

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()

    text_by_cell: list[str] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                pytest.fail(
                    f"notebook cell raised {output.get('ename')}: {output.get('evalue')}"
                )
            if output.get("output_type") == "stream":
                text_by_cell.append(output.get("text", ""))
            data = output.get("data", {})
            if isinstance(data, dict) and "text/plain" in data:
                text_by_cell.append(str(data["text/plain"]))
    joined = "\n".join(text_by_cell)
    state = json.loads((PUBLISHED_DIR / "run.json").read_text())
    assert state["run_id"] in joined, "executed notebook did not surface the completed run id"
    # The notebook reads only committed artifacts; no paid request path exists.
    # (The committed pre-repair preflight COST record is a JSON file the
    # notebook may legitimately read; no live command or client may appear.)
    sources = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for banned in (
        "httpx",
        "requests.post",
        "OpenRouterAdapter",
        "JevAdapter",
        "jev_benchmark.cli",
        "run_preflight",
        "MeasuredAdapter",
    ):
        assert banned not in sources, f"notebook code references paid path {banned!r}"
    # Every filesystem read in the notebook stays inside the checkout.
    for forbidden in ("runs/", "/.env", "os.environ"):
        assert forbidden not in sources, f"notebook code references {forbidden!r}"
