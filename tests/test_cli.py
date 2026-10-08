import json

import pytest

from jev_benchmark.cli import main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "prepare" in out
    assert "inspect" in out


@pytest.mark.parametrize(
    "sub",
    ["prepare", "inspect", "preflight", "pilot", "run", "run-inspect", "report"],
)
def test_subcommand_help_exits_zero(sub, capsys):
    # argparse interpolates help strings as %-format templates; a bare '%' (e.g.
    # '95%' in report --n-boot) used to raise TypeError and exit 1.
    with pytest.raises(SystemExit) as exc:
        main([sub, "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert sub in out


def test_prepare_writes_artifacts_and_preserves_rows(tmp_path, official_csv, capsys):
    rc = main(["prepare", "--source", str(official_csv), "--data-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1014" in out

    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["row_count"] == 1014
    assert provenance["license"] == "CC BY 4.0"
    assert (tmp_path / "audit.json").is_file()
    assert (tmp_path / "raw" / "maternal_health_risk.csv").is_file()

    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["distinct_complete_rows"] == 452


def test_inspect_shows_pre_run_definition(tmp_path, official_csv, capsys):
    main(["prepare", "--source", str(official_csv), "--data-dir", str(tmp_path)])
    capsys.readouterr()
    rc = main(["inspect", "--data-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # Planned population, source, pinned models, features, two-stage task.
    assert "jev-1.13.0" in out
    assert "openai/gpt-4o-mini" in out
    assert "google/gemini-2.5-flash" in out
    assert "3042" in out.replace(",", "")
    assert "Age" in out and "HeartRate" in out
    assert "high risk rather than mid/low" in out
    assert "config hash" in out


def test_prepare_failed_source_nonzero_exit(tmp_path, capsys):
    rc = main(["prepare", "--source", str(tmp_path / "nope.csv"), "--data-dir", str(tmp_path)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "error" in err.lower()


def test_prepare_bad_schema_nonzero_exit(tmp_path, fixtures_dir, capsys):
    rc = main(
        ["prepare", "--source", str(fixtures_dir / "bad_label.csv"), "--data-dir", str(tmp_path)]
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "error" in err.lower()


def test_prepare_labels_short_csv_as_non_official(tmp_path, fixtures_dir, capsys):
    rc = main(
        ["prepare", "--source", str(fixtures_dir / "valid_small.csv"), "--data-dir", str(tmp_path)]
    )
    assert rc == 0
    out = capsys.readouterr().out.lower()
    # A short arbitrary CSV must not be presented as the official population.
    assert "not the official" in out or "non-official" in out

    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["is_official_population"] is False


def test_run_blocks_when_pilot_lacks_three_provider_validation(tmp_path, official_csv, capsys):
    # Bulk dispatch requires successful validating evidence for all three real
    # provider IDs. An affordable pilot that only validated two providers must
    # not authorize the full paid run. The gate fires before any live adapter is
    # built, so no API keys are needed.
    data_dir = tmp_path / "data"
    main(["prepare", "--source", str(official_csv), "--data-dir", str(data_dir)])
    capsys.readouterr()
    provenance = json.loads((data_dir / "provenance.json").read_text())

    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    projection = {
        "config_hash": provenance["config_hash"],
        "ordered_record_sha256": provenance["ordered_record_sha256"],
        "is_official_population": True,
        "validated_models": ["jev-1.13.0", "openai/gpt-4o-mini"],  # gemini missing
        "projection": {"affordable": True, "reason": "projected cost fits"},
    }
    (pilot_dir / "projection.json").write_text(json.dumps(projection))

    rc = main(
        [
            "run",
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(tmp_path / "run"),
            "--pilot-dir",
            str(pilot_dir),
        ]
    )
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "gemini" in err
    assert "provider" in err


def test_run_blocks_full_eval_on_non_official_population(tmp_path, fixtures_dir, capsys):
    # A full paid evaluation (default --rows => full population) must be refused
    # when the prepared data is not the official 1,014-row population. The guard
    # fires before any live adapter is built, so no API keys are needed.
    rc = main(
        [
            "run",
            "--source",
            str(fixtures_dir / "valid_small.csv"),
            "--run-dir",
            str(tmp_path / "run"),
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "official" in err and "population" in err


def _write_valid_pilot(tmp_path, data_dir):
    """A pilot projection that passes every run gate for the prepared data."""
    provenance = json.loads((data_dir / "provenance.json").read_text())
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir(exist_ok=True)
    (pilot_dir / "projection.json").write_text(
        json.dumps(
            {
                "config_hash": provenance["config_hash"],
                "ordered_record_sha256": provenance["ordered_record_sha256"],
                "is_official_population": True,
                "validated_models": [
                    "jev-1.13.0",
                    "openai/gpt-4o-mini",
                    "google/gemini-2.5-flash",
                ],
                "projection": {"affordable": True, "reason": "projected cost fits"},
            }
        )
    )
    return pilot_dir


def test_run_requires_pilot_even_with_narrowed_rows(tmp_path, official_csv, capsys):
    # --rows must not become a bulk-dispatch bypass: every measured run needs an
    # affordable pilot that validated all three providers. The gate fires before
    # any live adapter is built, so no API keys are needed.
    data_dir = tmp_path / "data"
    main(["prepare", "--source", str(official_csv), "--data-dir", str(data_dir)])
    capsys.readouterr()

    rc = main(
        [
            "run",
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(tmp_path / "run"),
            "--rows",
            "2",
            "--budget-state",
            str(tmp_path / "shared-budget.json"),
        ]
    )
    assert rc != 0
    assert "pilot" in capsys.readouterr().err.lower()


def test_run_rejects_removed_no_require_pilot_flag(tmp_path, official_csv, capsys):
    # The bypass flag was removed; argparse must reject it outright.
    data_dir = tmp_path / "data"
    main(["prepare", "--source", str(official_csv), "--data-dir", str(data_dir)])
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--data-dir",
                str(data_dir),
                "--run-dir",
                str(tmp_path / "run"),
                "--no-require-pilot",
            ]
        )
    assert exc.value.code != 0


def test_run_refuses_cap_above_approved(tmp_path, official_csv, capsys):
    # A user-configured cap above the approved $20 fails safely before any paid
    # call (this guard fires before dataset-gated work touches a provider).
    data_dir = tmp_path / "data"
    main(["prepare", "--source", str(official_csv), "--data-dir", str(data_dir)])
    capsys.readouterr()

    rc = main(
        [
            "run",
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(tmp_path / "run"),
            "--budget-cap",
            "25",
        ]
    )
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "cap" in err and "20" in err


def test_run_refuses_occupied_run_dir(tmp_path, official_csv, capsys):
    # A fresh run on an occupied directory must refuse rather than append
    # duplicate cases; the guard fires before any live adapter is built.
    data_dir = tmp_path / "data"
    main(["prepare", "--source", str(official_csv), "--data-dir", str(data_dir)])
    capsys.readouterr()
    pilot_dir = _write_valid_pilot(tmp_path, data_dir)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"run_id": "existing"}))

    rc = main(
        [
            "run",
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(run_dir),
            "--pilot-dir",
            str(pilot_dir),
            "--budget-state",
            str(tmp_path / "shared-budget.json"),
        ]
    )
    assert rc != 0
    assert "occupied" in capsys.readouterr().err.lower()


def test_run_refuses_empty_pilot_projection(tmp_path, official_csv, capsys):
    # An empty pilot (no providers validated) cannot authorize bulk dispatch.
    data_dir = tmp_path / "data"
    main(["prepare", "--source", str(official_csv), "--data-dir", str(data_dir)])
    capsys.readouterr()
    provenance = json.loads((data_dir / "provenance.json").read_text())

    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    (pilot_dir / "projection.json").write_text(
        json.dumps(
            {
                "config_hash": provenance["config_hash"],
                "ordered_record_sha256": provenance["ordered_record_sha256"],
                "is_official_population": True,
                "validated_models": [],  # empty evidence
                "projection": {"affordable": True, "reason": "projected cost fits"},
            }
        )
    )

    rc = main(
        [
            "run",
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(tmp_path / "run"),
            "--pilot-dir",
            str(pilot_dir),
        ]
    )
    assert rc != 0
    assert "pilot" in capsys.readouterr().err.lower()


def test_pilot_refuses_cap_above_approved(tmp_path, official_csv, capsys):
    rc = main(
        [
            "pilot",
            "--source",
            str(official_csv),
            "--data-dir",
            str(tmp_path / "data"),
            "--run-dir",
            str(tmp_path / "pilot"),
            "--budget-cap",
            "30",
            "--budget-state",
            str(tmp_path / "shared-budget.json"),
        ]
    )
    assert rc != 0
    err = capsys.readouterr().err.lower()
    assert "cap" in err and "20" in err


def test_pilot_refuses_occupied_run_dir(tmp_path, official_csv, capsys):
    run_dir = tmp_path / "pilot"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"run_id": "existing"}))

    rc = main(
        [
            "pilot",
            "--source",
            str(official_csv),
            "--data-dir",
            str(tmp_path / "data"),
            "--run-dir",
            str(run_dir),
            "--budget-state",
            str(tmp_path / "shared-budget.json"),
        ]
    )
    assert rc != 0
    assert "occupied" in capsys.readouterr().err.lower()
