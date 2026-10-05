# jev-benchmark

Retrospective benchmark comparing **Jev** (`jev-1.13.0`) and two OpenRouter LLMs
(`openai/gpt-4o-mini`, `google/gemini-2.5-flash`) on their ability to reproduce
the existing public **UCI Maternal Health Risk** labels (dataset 863, CC BY 4.0).

> This project replicates existing dataset labels. It is **not** a clinical tool
> and makes **no** diagnosis, triage, treatment, or patient-safety claim.

## Status

This repository is being built milestone by milestone. The current milestone
provides the local Python package, pinned dependencies, and the deterministic
data preparation / inspection CLI. Live provider adapters, the budgeted runner,
and the results notebook/README arrive in later milestones.

## Setup

Requires **Python 3.12**. From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Secrets live in a git-ignored `.env` (names only in `.env.example`):

```
TYPESAFE_API_KEY=
OPENROUTER_API_KEY=
```

Preparation and inspection below need **no keys and make no paid requests**.

## CLI

The package installs a `jev-benchmark` console script (equivalently
`.venv/bin/python -m jev_benchmark`).

```sh
# Show all commands
.venv/bin/jev-benchmark --help

# Fetch the official UCI CSV, validate it, and save provenance + duplicate audit
.venv/bin/jev-benchmark prepare

# Review the frozen pre-run definition: planned population, source, input
# features, pinned model IDs, two-stage task, and frozen identifiers
.venv/bin/jev-benchmark inspect
```

`prepare` writes to `data/`:

- `data/raw/maternal_health_risk.csv` - the official CSV, verbatim
- `data/provenance.json` - source URL, retrieval date, CC BY 4.0 citation,
  byte and ordered-record SHA-256 hashes, row count, class distribution,
  schema, and the frozen `config_hash`
- `data/audit.json` - repeated-row audit (distinct complete rows, repeated
  feature vectors, conflicting-label groups)

Read a local copy instead of fetching with `--source path/to.csv`.

## Dataset facts and known discrepancy

The served CSV contains **1,014 rows** with six predictors
(`Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`) and the
`RiskLevel` label (`high risk` 272 / `mid risk` 336 / `low risk` 406, no nulls).
UCI's dataset **metadata reports 1,013 instances**; all 1,014 rows are preserved
in source order and the discrepancy is recorded rather than silently resolved.
The audit finds **452 distinct complete rows** and **35** feature vectors with
conflicting labels; duplicates remain in the primary population and any
deduplication is a separate sensitivity analysis.

`RiskLevel` is local truth only and is never included in provider requests;
only the six measurements are serialized for a model.

## Testing

```sh
.venv/bin/pytest -q
.venv/bin/ruff check .
```

## Attribution

Ahmed, Marzia. (2020). *Maternal Health Risk* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5DP5D — licensed CC BY 4.0.
