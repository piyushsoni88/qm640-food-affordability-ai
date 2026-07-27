from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "12_collect_all_india_state_daily_climate.py"
)
SPEC = importlib.util.spec_from_file_location("all_india_collector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def test_region_inventory_has_all_states_and_union_territories():
    assert len(COLLECTOR.REGIONS) == 36
    assert sum(value[0] == "State" for value in COLLECTOR.REGIONS.values()) == 28
    assert (
        sum(
            value[0] == "Union Territory"
            for value in COLLECTOR.REGIONS.values()
        )
        == 8
    )


def test_monthly_aggregation_sums_rainfall_and_flags_partial_month():
    sample = pd.DataFrame(
        {
            "region": ["Delhi", "Delhi"],
            "admin_type": ["Union Territory", "Union Territory"],
            "reference_location": ["New Delhi", "New Delhi"],
            "spatial_representation": [
                "representative administrative-capital point",
                "representative administrative-capital point",
            ],
            "latitude": [28.6139, 28.6139],
            "longitude": [77.2090, 77.2090],
            "date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
            "prectotcorr": [2.5, 3.5],
            "t2m": [30.0, 32.0],
            "t2m_max": [35.0, 37.0],
            "t2m_min": [25.0, 27.0],
            "rh2m": [60.0, 70.0],
        }
    )
    result = COLLECTOR.aggregate_monthly(sample).iloc[0]
    assert result["rainfall_mm"] == 6.0
    assert result["temperature_c"] == 31.0
    assert result["observed_days"] == 2
    assert bool(result["is_partial_month"])
