"""Offline figure tests: reconciliation with the report/artifacts, byte-identical
regeneration, and the constraint that the notebook never writes into docs/figures/.

Everything here is offline. The tests derive their expectations from the same
committed ``published/`` artifacts the report uses, so a figure that disagrees
with a published table fails the suite. Tests that need the committed
``published/`` directory or the committed notebook are skipped when those paths
are absent (e.g. before the results milestone).
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from jev_benchmark import cli, figures, metrics, spec
from jev_benchmark import report as report_mod
from jev_benchmark.dataset import prepare_dataset
from jev_benchmark.runner import AttemptRecord, CaseRecord, RunState

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_DIR = REPO_ROOT / "published"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "benchmark.ipynb"
SMALL_CSV = REPO_ROOT / "tests" / "fixtures" / "valid_small.csv"

MODELS = list(spec.MODEL_IDS)

PUBLISHED_READY = (PUBLISHED_DIR / "cases.jsonl").is_file()
FIGURES_READY = FIGURES_DIR.is_dir() and all(
    (FIGURES_DIR / name).is_file() for name in figures.FIGURE_FILENAMES
)
NOTEBOOK_READY = PUBLISHED_READY and NOTEBOOK_PATH.is_file()

requires_published = pytest.mark.skipif(
    not PUBLISHED_READY, reason="published/ artifacts are not committed in this checkout"
)
requires_figures = pytest.mark.skipif(
    not FIGURES_READY, reason="docs/figures/ PNGs are not committed in this checkout"
)


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def published_dataset():
    return prepare_dataset(source=REPO_ROOT / "data" / "raw" / "maternal_health_risk.csv")


@pytest.fixture(scope="module")
def published_cases():
    return [
        CaseRecord.from_dict(json.loads(line))
        for line in (PUBLISHED_DIR / "cases.jsonl").read_text().splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def published_report(published_dataset):
    artifacts = report_mod.load_run_artifacts(PUBLISHED_DIR)
    return report_mod.build_report(artifacts, published_dataset)


def _png_size(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    width, height = struct.unpack(">II", raw[16:24])
    return width, height


def _independent_confusion(cases: list[CaseRecord], model: str) -> list[list[int]]:
    """Recount the 3x4 confusion straight from the case records (no metric code)."""
    columns = [*spec.RISK_LABELS, metrics.INVALID_COLUMN]
    matrix = {truth: dict.fromkeys(columns, 0) for truth in spec.RISK_LABELS}
    for case in cases:
        if case.requested_model != model:
            continue
        final = case.final_label if case.outcome == "predicted" else None
        column = final if final in spec.RISK_LABELS else metrics.INVALID_COLUMN
        matrix[case.gold_label][column] += 1
    return [[matrix[truth][column] for column in columns] for truth in spec.RISK_LABELS]


# --------------------------------------------------------------------------- #
# Figure data reconciliation (VAL-FIG-002/003/004).
# --------------------------------------------------------------------------- #

@requires_published
def test_confusion_figure_data_reconciles_with_report_and_artifacts(
    published_report, published_cases
):
    matrices = figures.confusion_matrices(published_report)
    assert set(matrices) == set(MODELS)
    for model in MODELS:
        matrix = matrices[model]
        assert len(matrix) == 3, "confusion figure must have three truth rows"
        assert all(len(row) == 4 for row in matrix), "four prediction columns incl. invalid"
        report_confusion = published_report["providers"][model]["all_attempted"]["confusion"]
        for row_index, truth in enumerate(spec.RISK_LABELS):
            expected = [report_confusion[truth][column] for column in figures.CONFUSION_COLUMNS]
            assert matrix[row_index] == expected, f"{model} {truth} disagrees with the report"
        assert matrix == _independent_confusion(published_cases, model), (
            f"{model} confusion figure disagrees with an independent recount of cases.jsonl"
        )


@requires_published
def test_confusion_figure_reconciles_with_published_readme_values(published_report):
    """Spot-check the committed README/table values so the figure cannot drift."""
    matrices = figures.confusion_matrices(published_report)
    assert matrices["jev-1.13.0"][0] == [208, 40, 24, 0]
    assert matrices["openai/gpt-4o-mini"][1] == [136, 1, 199, 0]
    # Gemini's invalid column is where its 12 non-predicted rows live (7 + 5).
    assert matrices["google/gemini-2.5-flash"][1][3] == 7
    assert matrices["google/gemini-2.5-flash"][2][3] == 5


@requires_published
def test_latency_figure_annotations_reconcile_with_report(published_report):
    annotations = figures.latency_annotations(published_report)
    for model in MODELS:
        expected = published_report["providers"][model]["latency"]
        assert annotations[model] == expected
        for scope in ("api_attempt", "workflow"):
            assert annotations[model][scope]["count"] > 0
            assert annotations[model][scope]["p50"] is not None
            assert annotations[model][scope]["p95"] is not None


@requires_published
def test_latency_samples_keep_the_two_scopes_distinct_and_include_failures(
    published_report, published_cases
):
    samples = figures.latency_samples(published_cases)
    for model in MODELS:
        latency = published_report["providers"][model]["latency"]
        api = samples[model]["api"]
        workflow = samples[model]["workflow"]
        # Counts match the report's percentile denominators exactly.
        assert len(api) == latency["api_attempt"]["count"]
        assert len(workflow) == latency["workflow"]["count"]
        # The API-attempt scope includes completed failed/retried attempts.
        attempt_durations = [
            a.duration for c in published_cases if c.requested_model == model for a in c.attempts
        ]
        assert sorted(api) == sorted(d for d in attempt_durations if d is not None)
        assert latency["failed_attempts"] == 0 or len(api) > len(workflow)
    gemini = published_report["providers"]["google/gemini-2.5-flash"]["latency"]
    assert gemini["failed_attempts"] == 12
    assert gemini["retried_attempts"] == 0
    jev = published_report["providers"]["jev-1.13.0"]["latency"]
    assert jev["retried_attempts"] == 1 and jev["failed_attempts"] == 1
    # The two scopes are separate series; the workflow scope is never polluted
    # by the extra failed/retried attempts.
    assert len(samples["google/gemini-2.5-flash"]["api"]) == 1575
    assert len(samples["google/gemini-2.5-flash"]["workflow"]) == 1014


@requires_published
def test_cost_figure_bases_reconcile_with_report(published_report):
    bases = figures.cost_bases(published_report)
    assert set(bases["providers"]) == set(MODELS)
    for model in MODELS:
        assert bases["providers"][model] == published_report["providers"][model]["cost"]
    assert bases["marginal"] == published_report["accounting"]["full_run_marginal"]
    # The three bases stay separate: no single conflated "total cost" is exposed.
    for model in MODELS:
        assert set(bases["providers"][model]) >= {
            "billed_openrouter",
            "estimated_jev",
            "reserved_unknown",
        }
    assert bases["providers"]["jev-1.13.0"]["billed_openrouter"] == 0.0
    assert bases["providers"]["openai/gpt-4o-mini"]["estimated_jev"] == 0.0
    assert bases["providers"]["jev-1.13.0"]["reserved_unknown"] == pytest.approx(0.005)


# --------------------------------------------------------------------------- #
# Figure structure (the three series/columns the contract requires).
# --------------------------------------------------------------------------- #

@requires_published
def test_confusion_figure_draws_three_truth_by_four_prediction_cells(published_report):
    figure = figures.confusion_figure(published_report)
    try:
        panels = [ax for ax in figure.get_axes() if ax.get_label() != "<colorbar>"]
        assert len(panels) == len(MODELS)
        for ax in panels:
            assert len(ax.texts) == 12, "each heatmap annotates a 3x4 grid"
            assert [t.get_text() for t in ax.get_xticklabels()][-1] == metrics.INVALID_COLUMN
    finally:
        figure.clf()


@requires_published
def test_latency_figure_draws_two_rows_and_marks_both_percentiles(
    published_report, published_cases
):
    figure = figures.latency_figure(published_report, published_cases)
    try:
        axes = figure.get_axes()
        assert len(axes) == 2 * len(MODELS), "one row per scope, one column per provider"
        for ax in axes:
            # Each panel marks p50 and p95 with vertical lines.
            assert len(ax.get_lines()) == 2
    finally:
        figure.clf()


@requires_published
def test_cost_figure_keeps_the_three_bases_as_separate_series(published_report):
    figure = figures.cost_figure(published_report)
    try:
        per_provider, marginal = figure.get_axes()
        assert len(per_provider.patches) == 3 * len(MODELS)
        labels = [text.get_text() for text in per_provider.get_legend().get_texts()]
        assert len(labels) == 3
        assert len(marginal.patches) == 3
    finally:
        figure.clf()


# --------------------------------------------------------------------------- #
# Byte-identical regeneration and PNG metadata (VAL-FIG-001).
# --------------------------------------------------------------------------- #

@requires_published
def test_regeneration_is_byte_identical(published_report, published_cases, tmp_path):
    first = figures.write_figures(published_report, published_cases, tmp_path / "a")
    second = figures.write_figures(published_report, published_cases, tmp_path / "b")
    assert set(first) == set(figures.FIGURE_FILENAMES)
    for name in figures.FIGURE_FILENAMES:
        assert first[name].read_bytes() == second[name].read_bytes(), (
            f"{name} is not byte-identical across regenerations"
        )


@requires_published
@requires_figures
def test_regeneration_reproduces_the_committed_pngs(
    published_report, published_cases, tmp_path
):
    written = figures.write_figures(published_report, published_cases, tmp_path)
    for name in figures.FIGURE_FILENAMES:
        committed = (FIGURES_DIR / name).read_bytes()
        assert written[name].read_bytes() == committed, (
            f"committed {name} differs from a fresh regeneration"
        )


@requires_published
def test_pngs_carry_no_software_metadata_and_fixed_dimensions(
    published_report, published_cases, tmp_path
):
    written = figures.write_figures(published_report, published_cases, tmp_path)
    expected_sizes = {
        figures.CONFUSION_FILENAME: figures.CONFUSION_FIGSIZE,
        figures.LATENCY_FILENAME: figures.LATENCY_FIGSIZE,
        figures.COST_FILENAME: figures.COST_FIGSIZE,
    }
    for name, (width_in, height_in) in expected_sizes.items():
        raw = written[name].read_bytes()
        assert b"Software" not in raw, "matplotlib Software text chunk was not stripped"
        assert b"tEXt" not in raw
        width, height = _png_size(written[name])
        assert (width, height) == (
            round(width_in * figures.FIGURE_DPI),
            round(height_in * figures.FIGURE_DPI),
        )


@requires_published
def test_figures_command_leaves_published_byte_identical(tmp_path, monkeypatch):
    import os

    for var in list(os.environ):
        if "API_KEY" in var or "TOKEN" in var:
            monkeypatch.delenv(var, raising=False)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PUBLISHED_DIR.iterdir()}

    rc = cli.main(["figures", "--from", str(PUBLISHED_DIR), "--out", str(tmp_path / "figs")])

    assert rc == 0
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PUBLISHED_DIR.iterdir()}
    assert before == after, "the figures command modified published/ artifacts"
    manifest = report_mod.verify_manifest(
        PUBLISHED_DIR, RunState.from_dict(json.loads((PUBLISHED_DIR / "run.json").read_text()))
    )
    assert manifest is not None and manifest["files_verified"] == 5


# --------------------------------------------------------------------------- #
# CLI surface.
# --------------------------------------------------------------------------- #

def test_figures_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["figures", "--help"])
    assert excinfo.value.code == 0
    assert "docs/figures" in capsys.readouterr().out


@requires_published
def test_cli_figures_writes_three_pngs(tmp_path, capsys):
    out = tmp_path / "out"
    rc = cli.main(["figures", "--from", str(PUBLISHED_DIR), "--out", str(out)])
    assert rc == 0
    assert sorted(p.name for p in out.glob("*.png")) == sorted(figures.FIGURE_FILENAMES)
    assert "figures written" in capsys.readouterr().out


@requires_published
def test_cli_figures_refuses_out_inside_artifact_dir(capsys):
    rc = cli.main(["figures", "--from", str(PUBLISHED_DIR), "--out", str(PUBLISHED_DIR / "figs")])
    assert rc != 0
    assert "artifact directory" in capsys.readouterr().err
    assert not (PUBLISHED_DIR / "figs").exists()


def _write_reduced_run_dir(tmp_path: Path) -> Path:
    dataset = prepare_dataset(source=SMALL_CSV)
    run_dir = tmp_path / "reduced"
    run_dir.mkdir()
    cases = []
    for record in dataset.records:
        for model in MODELS:
            kind = "jev" if model == "jev-1.13.0" else "openrouter"
            cases.append(
                CaseRecord(
                    row_index=record.index,
                    requested_model=model,
                    provider_kind=kind,
                    outcome="predicted",
                    final_label=record.label,
                    gold_label=record.label,
                    stage_count=1,
                    workflow_started_at=0.0,
                    workflow_ended_at=1.0,
                    attempts=[
                        AttemptRecord(
                            provider=model,
                            provider_kind=kind,
                            stage="stage1",
                            attempt=1,
                            status="ok",
                            started_at=0.0,
                            ended_at=1.0,
                            actual_model=model,
                            input_tokens=10,
                            output_tokens=5,
                            cost_basis="estimated" if kind == "jev" else "billed",
                            cost_amount=0.0001,
                        )
                    ],
                )
            )
    state = RunState(
        run_id="run-reduced",
        mode="run",
        config_hash=dataset.provenance["config_hash"],
        ordered_record_sha256=dataset.provenance["ordered_record_sha256"],
        model_ids=MODELS,
        planned_row_indices=[record.index for record in dataset.records],
        status="complete",
    )
    (run_dir / "run.json").write_text(json.dumps(state.as_dict(), indent=2) + "\n")
    with (run_dir / "cases.jsonl").open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case.as_dict()) + "\n")
    return run_dir


def test_cli_figures_refuses_reduced_scope(tmp_path, capsys):
    run_dir = _write_reduced_run_dir(tmp_path)
    out = tmp_path / "out"
    rc = cli.main(
        ["figures", "--from", str(run_dir), "--out", str(out), "--source", str(SMALL_CSV)]
    )
    assert rc != 0
    assert "official" in capsys.readouterr().err
    assert not out.exists()


# --------------------------------------------------------------------------- #
# Notebook constraint (must not write into docs/figures/).
# --------------------------------------------------------------------------- #

def _notebook_sources() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


@requires_published
def test_notebook_renders_inline_figures_without_writing_committed_files():
    sources = _notebook_sources()
    assert "figures_mod" in sources, "notebook does not render the shared figures inline"
    for writer in ("savefig", "write_figures", "mkdir"):
        assert writer not in sources, f"notebook references a filesystem writer {writer!r}"


@requires_published
@requires_figures
@pytest.mark.skipif(not NOTEBOOK_READY, reason="notebook is not committed in this checkout")
def test_notebook_execution_does_not_dirty_docs_figures():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    before = {
        name: hashlib.sha256((FIGURES_DIR / name).read_bytes()).hexdigest()
        for name in figures.FIGURE_FILENAMES
    }
    before_listing = sorted(p.name for p in FIGURES_DIR.iterdir())

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()

    after = {
        name: hashlib.sha256((FIGURES_DIR / name).read_bytes()).hexdigest()
        for name in figures.FIGURE_FILENAMES
    }
    assert after == before, "executing the notebook modified committed docs/figures PNGs"
    assert sorted(p.name for p in FIGURES_DIR.iterdir()) == before_listing
