"""Deterministic offline figures for the published benchmark results.

A figure can never disagree with a published table because the plotting code
reuses the *report's* numbers: every annotated value is read from the same
report dictionary the CLI ``report`` command prints (the confusion matrices,
the latency p50/p95 with their sample counts, and the three cost bases).
The only raw values read directly from the artifacts are the per-attempt and
per-workflow durations needed to draw a *distribution* (the report only carries
the percentile summaries), and they come from the same loaded
``published/cases.jsonl`` records.

Rendering is byte-for-byte reproducible from an unchanged artifact set:
matplotlib's non-interactive ``Agg`` backend, an explicit fixed figure size and
DPI, a fixed colormap and normalization, no timestamps, and PNG metadata with
matplotlib's ``Software`` text chunk stripped (``metadata={"Software": None}``).

Everything here is offline. No credential is read, no network call is made and
no paid request is sent; the figures are derived solely from the hash-pinned
committed artifacts. Nothing is ever written into the artifact directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from . import metrics, spec
from .runner import CaseRecord

# Deterministic rendering knobs. These are module constants, not parameters, so
# regenerating from an unchanged artifact set cannot drift.
BACKEND = "Agg"
FIGURE_DPI = 100
CONFUSION_FIGSIZE = (12.0, 4.2)
LATENCY_FIGSIZE = (12.0, 6.4)
COST_FIGSIZE = (12.0, 4.6)

CONFUSION_CMAP = "Blues"
# Fixed log-spaced histogram bins spanning the full observed latency range
# (0.11 s - 50 s) so the same edges are used regardless of the data.
LATENCY_BIN_EDGES = tuple(float(v) for v in np.logspace(np.log10(0.1), np.log10(60.0), 41))

LATENCY_API_COLOR = "#1f77b4"
LATENCY_WORKFLOW_COLOR = "#d62728"
COST_BILLED_COLOR = "#2ca02c"
COST_ESTIMATED_COLOR = "#1f77b4"
COST_UNKNOWN_COLOR = "#ff7f0e"

CONFUSION_COLUMNS = (*spec.RISK_LABELS, metrics.INVALID_COLUMN)

CONFUSION_FILENAME = "confusion-matrices.png"
LATENCY_FILENAME = "latency-distributions.png"
COST_FILENAME = "cost-by-basis.png"

# The three committed figures, in the order the README embeds them.
FIGURE_FILENAMES = (CONFUSION_FILENAME, LATENCY_FILENAME, COST_FILENAME)


# --------------------------------------------------------------------------- #
# Figure data accessors (the exact numbers the plots draw).
# --------------------------------------------------------------------------- #

def _model_ids(report: dict) -> list[str]:
    return list(report["run"]["model_ids"])


def confusion_matrices(report: dict) -> dict[str, list[list[int]]]:
    """The 3x4 all-attempted confusion counts straight from the report.

    Rows are the three published truth classes; columns are the three published
    predicted classes followed by the explicit ``invalid`` column. These are the
    same dictionaries the report prints, so a heatmap cell equals its table cell.
    """
    matrices: dict[str, list[list[int]]] = {}
    for model in _model_ids(report):
        confusion = report["providers"][model]["all_attempted"]["confusion"]
        matrices[model] = [
            [int(confusion[truth][column]) for column in CONFUSION_COLUMNS]
            for truth in spec.RISK_LABELS
        ]
    return matrices


def latency_samples(cases: list[CaseRecord]) -> dict[str, dict[str, list[float]]]:
    """Raw per-attempt and per-workflow durations per provider, from the artifacts.

    ``api`` covers every finished API attempt, including completed failed and
    retried attempts; ``workflow`` covers each whole row-provider two-stage
    workflow. Attempts with no measured duration are excluded, exactly as the
    report's percentile denominators do.
    """
    samples: dict[str, dict[str, list[float]]] = {}
    for model in spec.MODEL_IDS:
        subset = [c for c in cases if c.requested_model == model]
        api = [a.duration for case in subset for a in case.attempts if a.duration is not None]
        workflow = [c.workflow_duration for c in subset if c.workflow_duration is not None]
        samples[model] = {"api": api, "workflow": workflow}
    return samples


def latency_annotations(report: dict) -> dict[str, dict]:
    """The p50/p95 values and sample counts the latency figure marks, from the report."""
    return {
        model: dict(report["providers"][model]["latency"]) for model in _model_ids(report)
    }


def cost_bases(report: dict) -> dict[str, dict]:
    """The three cost bases kept separate: per-provider and full-run marginal.

    ``providers`` are the per-model billed/estimated/unknown amounts; ``marginal``
    is the report's full-run marginal breakdown. The three bases are never summed
    into a single "total cost" anywhere in the figure.
    """
    return {
        "providers": {
            model: dict(report["providers"][model]["cost"]) for model in _model_ids(report)
        },
        "marginal": dict(report["accounting"]["full_run_marginal"]),
    }


# --------------------------------------------------------------------------- #
# Figure builders (return a matplotlib Figure; never touch the filesystem).
# --------------------------------------------------------------------------- #

def _confusion_text_color(value: int, vmax: int) -> str:
    return "white" if vmax and value > 0.55 * vmax else "black"


def confusion_figure(report: dict) -> plt.Figure:
    """Per-model 3x4 confusion heatmaps (rows = truth, columns = predicted + invalid)."""
    model_ids = _model_ids(report)
    matrices = confusion_matrices(report)
    vmax = max((max(max(row) for row in mat) for mat in matrices.values()), default=0) or 1
    norm = Normalize(vmin=0.0, vmax=float(vmax))

    fig, axes = plt.subplots(
        1, len(model_ids), figsize=CONFUSION_FIGSIZE, dpi=FIGURE_DPI, layout="constrained"
    )
    for ax, model in zip(np.atleast_1d(axes), model_ids, strict=True):
        matrix = np.asarray(matrices[model], dtype=float)
        image = ax.imshow(matrix, cmap=CONFUSION_CMAP, norm=norm, aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = int(matrix[row, column])
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color=_confusion_text_color(value, vmax),
                    fontsize=10,
                )
        ax.set_xticks(range(len(CONFUSION_COLUMNS)))
        ax.set_xticklabels(CONFUSION_COLUMNS, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(spec.RISK_LABELS)))
        ax.set_yticklabels(spec.RISK_LABELS, fontsize=8)
        accuracy = report["providers"][model]["all_attempted"]["accuracy"]
        ax.set_title(f"{model}\naccuracy {accuracy:.4f} (all-attempted)", fontsize=9)
        ax.set_xlabel("predicted (invalid = non-predicted)", fontsize=8)
        ax.set_ylabel("gold truth", fontsize=8)
    fig.colorbar(image, ax=list(np.atleast_1d(axes)), fraction=0.03, pad=0.02)
    fig.suptitle(
        "Confusion matrices (all-attempted): the invalid column keeps "
        "non-predicted rows in the denominator",
        fontsize=10,
    )
    return fig


def _mark_percentiles(ax, summary: dict, color: str) -> None:
    for label, key, style in (
        (f"p50 {summary['p50']:.3f}s", "p50", "--"),
        (f"p95 {summary['p95']:.3f}s", "p95", ":"),
    ):
        value = summary[key]
        if value is None:
            continue
        ax.axvline(value, color=color, linestyle=style, linewidth=1.2, label=label)


def latency_figure(report: dict, cases: list[CaseRecord]) -> plt.Figure:
    """Latency distributions at both scopes, kept visually distinct.

    Top row: per-attempt API calls (including completed failed/retried attempts).
    Bottom row: whole row-provider two-stage workflows. p50/p95 and sample counts
    are annotated from the report's latency summary.
    """
    model_ids = _model_ids(report)
    samples = latency_samples(cases)
    annotations = latency_annotations(report)
    bins = np.asarray(LATENCY_BIN_EDGES)

    fig, axes = plt.subplots(
        2, len(model_ids), figsize=LATENCY_FIGSIZE, dpi=FIGURE_DPI, sharex="col", squeeze=False
    )
    for column, model in enumerate(model_ids):
        api_ax = axes[0][column]
        workflow_ax = axes[1][column]
        api_summary = annotations[model]["api_attempt"]
        workflow_summary = annotations[model]["workflow"]

        api_ax.hist(samples[model]["api"], bins=bins, color=LATENCY_API_COLOR)
        _mark_percentiles(api_ax, api_summary, LATENCY_API_COLOR)
        api_ax.set_xscale("log")
        api_ax.set_title(
            f"{model}\nAPI attempts (n={api_summary['count']}; "
            f"retried {annotations[model]['retried_attempts']}, "
            f"failed {annotations[model]['failed_attempts']})",
            fontsize=9,
        )
        api_ax.set_ylabel("finished attempts", fontsize=8)
        api_ax.legend(fontsize=7, loc="upper right")

        workflow_ax.hist(samples[model]["workflow"], bins=bins, color=LATENCY_WORKFLOW_COLOR)
        _mark_percentiles(workflow_ax, workflow_summary, LATENCY_WORKFLOW_COLOR)
        workflow_ax.set_xscale("log")
        workflow_ax.set_title(
            f"whole-row workflows (n={workflow_summary['count']})", fontsize=9
        )
        workflow_ax.set_xlabel("seconds (client-observed, monotonic; log scale)", fontsize=8)
        workflow_ax.set_ylabel("terminal workflows", fontsize=8)
        workflow_ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Latency by scope: API attempts (top, includes failed/retried attempts) vs "
        "whole-row workflows (bottom)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


def cost_figure(report: dict) -> plt.Figure:
    """Cost by basis: billed, rate-estimated and reserved-unknown kept separate."""
    model_ids = _model_ids(report)
    bases = cost_bases(report)
    providers = bases["providers"]
    marginal = bases["marginal"]

    fig, (per_provider_ax, marginal_ax) = plt.subplots(
        1, 2, figsize=COST_FIGSIZE, dpi=FIGURE_DPI
    )

    positions = np.arange(len(model_ids), dtype=float)
    width = 0.25
    series = (
        ("OpenRouter billed (observed)", "billed_openrouter", COST_BILLED_COLOR),
        ("Jev rate-estimated (published rates)", "estimated_jev", COST_ESTIMATED_COLOR),
        ("reserved-unknown (never assumed free)", "reserved_unknown", COST_UNKNOWN_COLOR),
    )
    for offset_index, (label, key, color) in enumerate(series):
        offset = (offset_index - 1) * width
        values = [float(providers[model][key]) for model in model_ids]
        bars = per_provider_ax.bar(positions + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values, strict=True):
            if value > 0.0:
                per_provider_ax.annotate(
                    f"${value:.6f}",
                    (bar.get_x() + bar.get_width() / 2.0, value),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
    per_provider_ax.set_xticks(positions)
    per_provider_ax.set_xticklabels(model_ids, rotation=15, ha="right", fontsize=8)
    per_provider_ax.set_ylabel("USD", fontsize=8)
    per_provider_ax.set_title("Per provider, by basis", fontsize=9)
    per_provider_ax.legend(fontsize=7)

    marginal_series = (
        ("OpenRouter billed", "billed_openrouter", COST_BILLED_COLOR),
        ("Jev rate-estimated", "estimated_jev", COST_ESTIMATED_COLOR),
        ("reserved-unknown", "reserved_unknown", COST_UNKNOWN_COLOR),
    )
    marginal_labels = [label for label, _key, _color in marginal_series]
    marginal_values = [float(marginal[key]) for _label, key, _color in marginal_series]
    marginal_colors = [color for _label, _key, color in marginal_series]
    marginal_bars = marginal_ax.bar(
        marginal_labels, marginal_values, color=marginal_colors, width=0.6
    )
    marginal_ax.bar_label(marginal_bars, fmt="$%.6f", fontsize=7, padding=1)
    marginal_ax.set_ylabel("USD", fontsize=8)
    marginal_ax.set_title("Full-run marginal, by basis (never summed)", fontsize=9)
    marginal_ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "Cost by basis: billed, rate-estimated and reserved-unknown are shown "
        "separately, never conflated into one total",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig


def build_figures(report: dict, cases: list[CaseRecord]) -> dict[str, plt.Figure]:
    """Build all three figures (in memory) from a report and the loaded cases."""
    return {
        CONFUSION_FILENAME: confusion_figure(report),
        LATENCY_FILENAME: latency_figure(report, cases),
        COST_FILENAME: cost_figure(report),
    }


# --------------------------------------------------------------------------- #
# Deterministic file output.
# --------------------------------------------------------------------------- #

def _force_agg_backend() -> None:
    """Select the non-interactive Agg backend for deterministic headless rendering."""
    if matplotlib.get_backend().lower() != BACKEND.lower():
        matplotlib.use(BACKEND, force=True)
    plt.rcParams["figure.dpi"] = FIGURE_DPI
    plt.rcParams["savefig.dpi"] = FIGURE_DPI


def write_figures(
    report: dict, cases: list[CaseRecord], out_dir: str | Path
) -> dict[str, Path]:
    """Render and save the three committed PNGs into ``out_dir``.

    Output is byte-identical for an unchanged artifact set: fixed size/DPI,
    fixed colormap and normalization, no timestamps, and the matplotlib
    ``Software`` text chunk stripped. The artifact directory is never touched.
    """
    _force_agg_backend()
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    figures = build_figures(report, cases)
    for filename, figure in figures.items():
        try:
            path = directory / filename
            figure.savefig(
                path, format="png", dpi=FIGURE_DPI, metadata={"Software": None}
            )
            written[filename] = path
        finally:
            plt.close(figure)
    return written
