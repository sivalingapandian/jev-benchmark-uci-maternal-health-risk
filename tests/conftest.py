from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
OFFICIAL_CSV = REPO_ROOT / "data" / "raw" / "maternal_health_risk.csv"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def official_csv() -> Path:
    if not OFFICIAL_CSV.is_file():
        pytest.skip(f"official cached CSV missing at {OFFICIAL_CSV}; run 'jev-benchmark prepare'")
    return OFFICIAL_CSV
