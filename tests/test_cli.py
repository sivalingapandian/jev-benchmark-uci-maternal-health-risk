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
