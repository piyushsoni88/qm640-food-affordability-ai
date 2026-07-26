import csv
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_gzip_csv(name: str):
    path = ROOT / "data" / "curated" / name
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_interim_panel_has_expected_grain_and_coverage():
    rows = read_gzip_csv("india_food_affordability_panel_15x8_2005_2025.csv.gz")
    keys = {(row["date"], row["region"], row["commodity"]) for row in rows}

    assert len(rows) == 30_240
    assert len(keys) == len(rows)
    assert len({row["region"] for row in rows}) == 15
    assert len({row["commodity"] for row in rows}) == 8
    assert min(row["date"] for row in rows).startswith("2005-01-01")
    assert max(row["date"] for row in rows).startswith("2025-12-01")


def test_quality_report_matches_committed_outputs():
    quality_path = ROOT / "data" / "metadata" / "interim_data_quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    national = read_gzip_csv("india_food_affordability_national_monthly.csv.gz")

    assert quality["national_rows"] == len(national) == 252
    assert quality["panel_rows"] == 30_240
    assert quality["national_duplicate_rows"] == 0
    assert quality["panel_duplicate_keys"] == 0
