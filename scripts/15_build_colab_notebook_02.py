"""Build the Google Colab version of Notebook 02."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "02_data_quality_and_cleaning.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
    markdown(
        """# QM640 Food Price Affordability AI
## Notebook 02 — Data Quality, Cleaning, and Time-Series Feature Integrity

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Prerequisite:** Notebook 01 completed successfully

### Why this notebook exists

Reliable models require more than removing null values. This notebook:

1. verifies Notebook 01 outputs and key uniqueness;
2. profiles missingness and reporting coverage;
3. checks economic and physical validity rules;
4. flags extreme prices conservatively while preserving raw values;
5. calculates changes using exact calendar lags rather than row offsets;
6. constructs state-specific monthly climate anomalies;
7. cleans the national modeling panel without future-information leakage; and
8. writes auditable cleaned datasets and quality reports.

No missing price observation is invented. Mandi reporting gaps remain explicit.
"""
    ),
    markdown(
        """## 1. Runtime and persistent-storage setup

Notebook 01 saved its outputs in `MyDrive/QM640_Food_Affordability`. This notebook
uses the same location so files survive Colab restarts. pandas and NumPy are
already installed in Colab, avoiding unnecessary installation time.
"""
    ),
    code(
        """from pathlib import Path
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 170)

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

SAVE_TO_DRIVE = True

if IN_COLAB and SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/QM640_Food_Affordability")
else:
    # The override supports local validation without writing test outputs into Git.
    default_root = Path.cwd().resolve()
    if default_root.name == "notebooks":
        default_root = default_root.parent
    OUTPUT_ROOT = Path(os.environ.get("QM640_OUTPUT_ROOT", str(default_root)))

PROCESSED = OUTPUT_ROOT / "data" / "processed"
REPORT_OUTPUT = OUTPUT_ROOT / "reports" / "notebook_outputs"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"Running in Google Colab: {IN_COLAB}")
print(f"Input/output root: {OUTPUT_ROOT}")
print(f"pandas: {pd.__version__}; NumPy: {np.__version__}")
"""
    ),
    markdown(
        """## 2. Load and verify Notebook 01 outputs

Only the columns required for cleaning are loaded. The state file is already a
compact monthly aggregate, so it is safe to load in memory. Explicit prerequisite
checks produce a helpful error instead of a later, ambiguous failure.
"""
    ),
    code(
        """STATE_INPUT = PROCESSED / "modeling_state_monthly.csv.gz"
NATIONAL_INPUT = PROCESSED / "modeling_monthly_national.csv.gz"
NB01_SUMMARY = REPORT_OUTPUT / "01_execution_summary.json"

required_files = [STATE_INPUT, NATIONAL_INPUT, NB01_SUMMARY]
missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        "Notebook 01 outputs are missing. Run Notebook 01 completely first.\\n"
        + "\\n".join(missing_files)
    )

state = pd.read_csv(
    STATE_INPUT,
    parse_dates=["date"],
    dtype={"region": "category", "Commodity": "category"},
    low_memory=False,
)
national = pd.read_csv(NATIONAL_INPUT, parse_dates=["date"], low_memory=False)
nb01_summary = json.loads(NB01_SUMMARY.read_text(encoding="utf-8"))

print("Notebook 01 status:", nb01_summary.get("status"))
print(f"State panel rows loaded: {len(state):,}")
print(f"National panel rows loaded: {len(national):,}")
print(f"State date range: {state['date'].min().date()} to {state['date'].max().date()}")
print(f"National date range: {national['date'].min().date()} to {national['date'].max().date()}")
"""
    ),
    markdown(
        """## 3. Structural integrity checks

The intended state-panel key is `region + date + Commodity`. Duplicates would
distort statistics and train/test evaluation, so the notebook stops rather than
silently deleting them. Dates must be monthly timestamps at the first day.
"""
    ),
    code(
        """state_key = ["region", "date", "Commodity"]
duplicate_keys = int(state.duplicated(state_key).sum())
national_duplicate_dates = int(national.duplicated(["date"]).sum())
non_month_start_state = int(state["date"].dt.day.ne(1).sum())
non_month_start_national = int(national["date"].dt.day.ne(1).sum())

structural_checks = {
    "state_key_duplicates": duplicate_keys,
    "national_date_duplicates": national_duplicate_dates,
    "state_dates_not_month_start": non_month_start_state,
    "national_dates_not_month_start": non_month_start_national,
}
print(pd.Series(structural_checks).to_string())

if any(structural_checks.values()):
    raise AssertionError(
        "Structural checks failed. Do not continue until duplicate/date problems are resolved."
    )
"""
    ),
    markdown(
        """## 4. Missingness and reporting-coverage profile

Missingness is reported both overall and by commodity. Coverage counts are kept
because an unbalanced official reporting panel is not equivalent to random
missing data. The analysis does not replace absent prices with zero or interpolate
them across reporting gaps.
"""
    ),
    code(
        """state_missingness = (
    state.isna()
    .mean()
    .mul(100)
    .rename("missing_percent")
    .rename_axis("variable")
    .reset_index()
    .sort_values("missing_percent", ascending=False)
)
national_missingness = (
    national.isna()
    .mean()
    .mul(100)
    .rename("missing_percent")
    .rename_axis("variable")
    .reset_index()
    .sort_values("missing_percent", ascending=False)
)

commodity_coverage = (
    state.groupby("Commodity", observed=True, as_index=False)
    .agg(
        rows=("date", "size"),
        states_uts=("region", "nunique"),
        start_date=("date", "min"),
        end_date=("date", "max"),
        official_observations=("source_rows", "sum"),
        missing_rainfall_pct=("rainfall_mm", lambda x: x.isna().mean() * 100),
        missing_temperature_pct=("temperature_c", lambda x: x.isna().mean() * 100),
    )
    .sort_values("official_observations", ascending=False)
)

state_missingness.to_csv(REPORT_OUTPUT / "02_state_missingness.csv", index=False)
national_missingness.to_csv(REPORT_OUTPUT / "02_national_missingness.csv", index=False)
commodity_coverage.to_csv(REPORT_OUTPUT / "02_commodity_coverage.csv", index=False)

print("Highest state-panel missingness")
print(state_missingness.head(12).round(3).to_string(index=False))
print("\\nCommodity reporting coverage")
print(commodity_coverage.round(3).to_string(index=False))
"""
    ),
    markdown(
        """## 5. Economic and physical validity rules

The rules are deliberately broad:

- mandi price and observation weights must be positive;
- rainfall cannot be negative;
- temperature must lie between −20°C and 60°C; and
- relative humidity must lie between 0% and 100%.

Broad bounds remove impossible values while avoiding the deletion of legitimate
extreme weather or price shocks—the phenomena the assignment aims to study.
"""
    ),
    code(
        """validity_flags = pd.DataFrame(index=state.index)
validity_flags["invalid_price"] = (
    state["modal_price_rs_per_quintal"].isna()
    | state["modal_price_rs_per_quintal"].le(0)
)
validity_flags["invalid_source_rows"] = (
    state["source_rows"].isna() | state["source_rows"].le(0)
)
validity_flags["invalid_rainfall"] = (
    state["rainfall_mm"].notna() & state["rainfall_mm"].lt(0)
)
validity_flags["invalid_temperature"] = (
    state["temperature_c"].notna()
    & ~state["temperature_c"].between(-20, 60)
)
validity_flags["invalid_humidity"] = (
    state["relative_humidity_pct"].notna()
    & ~state["relative_humidity_pct"].between(0, 100)
)
validity_flags["any_invalid"] = validity_flags.any(axis=1)

validity_counts = (
    validity_flags.sum()
    .astype(int)
    .rename("rows")
    .rename_axis("rule")
    .reset_index()
)
validity_counts.to_csv(REPORT_OUTPUT / "02_validity_rule_counts.csv", index=False)
print(validity_counts.to_string(index=False))

# Keep a copy of all valid records; removed counts remain documented.
clean = state.loc[~validity_flags["any_invalid"]].copy()
print(f"Rows retained after validity rules: {len(clean):,} ({len(clean)/len(state):.2%})")
"""
    ),
    markdown(
        """## 6. Conservative extreme-price treatment

Price distributions differ greatly by commodity. Outliers are therefore assessed
within commodity on the log-price scale using the robust median absolute
deviation (MAD). Only observations beyond six robust standard deviations are
flagged.

For model stability, flagged values receive a winsorized companion value bounded
by the commodity's 0.5th and 99.5th percentiles. The original price remains
unchanged in `modal_price_rs_per_quintal`, and `price_extreme_flag` makes every
adjustment auditable.
"""
    ),
    code(
        """def robust_zscore(series: pd.Series) -> pd.Series:
    \"\"\"Median/MAD z-score; fallback to standard deviation if MAD is zero.\"\"\"
    median = series.median()
    mad = (series - median).abs().median()
    robust_scale = 1.4826 * mad
    if not np.isfinite(robust_scale) or robust_scale == 0:
        robust_scale = series.std()
    if not np.isfinite(robust_scale) or robust_scale == 0:
        robust_scale = 1.0
    return (series - median) / robust_scale


clean["log_price_raw"] = np.log(clean["modal_price_rs_per_quintal"])
clean["price_robust_z"] = (
    clean.groupby("Commodity", observed=True)["log_price_raw"]
    .transform(robust_zscore)
)
clean["price_extreme_flag"] = clean["price_robust_z"].abs().gt(6)

commodity_bounds = (
    clean.groupby("Commodity", observed=True)["modal_price_rs_per_quintal"]
    .quantile([0.005, 0.995])
    .unstack()
    .rename(columns={0.005: "price_lower_bound", 0.995: "price_upper_bound"})
)
clean = clean.join(commodity_bounds, on="Commodity")

# Only flagged observations are adjusted. Non-flagged prices remain exactly raw.
clean["price_cleaned_rs_per_quintal"] = clean["modal_price_rs_per_quintal"]
flagged = clean["price_extreme_flag"]
clean.loc[flagged, "price_cleaned_rs_per_quintal"] = clean.loc[
    flagged, "modal_price_rs_per_quintal"
].clip(
    lower=clean.loc[flagged, "price_lower_bound"],
    upper=clean.loc[flagged, "price_upper_bound"],
)
clean["log_price_cleaned"] = np.log(clean["price_cleaned_rs_per_quintal"])

outlier_summary = (
    clean.groupby("Commodity", observed=True, as_index=False)
    .agg(
        rows=("date", "size"),
        extreme_prices=("price_extreme_flag", "sum"),
        raw_min_price=("modal_price_rs_per_quintal", "min"),
        raw_max_price=("modal_price_rs_per_quintal", "max"),
        cleaned_min_price=("price_cleaned_rs_per_quintal", "min"),
        cleaned_max_price=("price_cleaned_rs_per_quintal", "max"),
    )
)
outlier_summary["extreme_percent"] = (
    100 * outlier_summary["extreme_prices"] / outlier_summary["rows"]
)
outlier_summary.to_csv(REPORT_OUTPUT / "02_extreme_price_summary.csv", index=False)
print(outlier_summary.round(3).to_string(index=False))
"""
    ),
    markdown(
        """## 7. Exact calendar-lag price features

`groupby().pct_change(12)` means twelve *rows*, not necessarily twelve calendar
months. In an unbalanced reporting panel that can incorrectly compare dates more
than a year apart.

The code below performs keyed self-merges for exactly one month and twelve months
earlier. If that calendar observation is absent, the change correctly remains
missing. This prevents both timing errors and future leakage.
"""
    ),
    code(
        """clean = clean.sort_values(state_key).reset_index(drop=True)
lag_lookup = clean[
    ["region", "Commodity", "date", "price_cleaned_rs_per_quintal"]
].copy()


def add_exact_calendar_lag(
    frame: pd.DataFrame, lookup: pd.DataFrame, months: int
) -> pd.DataFrame:
    \"\"\"Join the price from exactly `months` calendar months earlier.\"\"\"
    lagged = lookup.copy()
    lagged["date"] = lagged["date"] + pd.DateOffset(months=months)
    lagged = lagged.rename(
        columns={
            "price_cleaned_rs_per_quintal": f"price_lag_{months}m"
        }
    )
    return frame.merge(
        lagged,
        on=["region", "Commodity", "date"],
        how="left",
        validate="one_to_one",
    )


clean = add_exact_calendar_lag(clean, lag_lookup, months=1)
clean = add_exact_calendar_lag(clean, lag_lookup, months=12)
clean["price_mom_pct"] = (
    clean["price_cleaned_rs_per_quintal"] / clean["price_lag_1m"] - 1
) * 100
clean["price_yoy_pct"] = (
    clean["price_cleaned_rs_per_quintal"] / clean["price_lag_12m"] - 1
) * 100
clean["has_exact_1m_lag"] = clean["price_lag_1m"].notna()
clean["has_exact_12m_lag"] = clean["price_lag_12m"].notna()

print(f"Rows with exact one-month lag: {clean['has_exact_1m_lag'].sum():,} ({clean['has_exact_1m_lag'].mean():.2%})")
print(f"Rows with exact twelve-month lag: {clean['has_exact_12m_lag'].sum():,} ({clean['has_exact_12m_lag'].mean():.2%})")
"""
    ),
    markdown(
        """## 8. State-specific seasonal climate anomalies

Climate values repeat once for every reported commodity in the state price panel.
Using those repeated rows would over-weight months with more commodity reporting.
We first deduplicate to one climate record per state-month, estimate each state's
monthly climatology using completed years through 2025, and then merge anomalies
back to the price panel.
"""
    ),
    code(
        """climate_columns = [
    "region",
    "date",
    "rainfall_mm",
    "temperature_c",
    "relative_humidity_pct",
]
state_climate = (
    clean[climate_columns]
    .drop_duplicates(["region", "date"])
    .copy()
)
state_climate["calendar_month"] = state_climate["date"].dt.month

# Exclude partial 2026 from the reference climatology.
reference_climate = state_climate.loc[state_climate["date"].dt.year <= 2025]
climatology = (
    reference_climate.groupby(
        ["region", "calendar_month"], observed=True, as_index=False
    )
    .agg(
        normal_rainfall_mm=("rainfall_mm", "mean"),
        normal_temperature_c=("temperature_c", "mean"),
        normal_humidity_pct=("relative_humidity_pct", "mean"),
    )
)
state_climate = state_climate.merge(
    climatology,
    on=["region", "calendar_month"],
    how="left",
    validate="many_to_one",
)
state_climate["rainfall_anomaly_mm"] = (
    state_climate["rainfall_mm"] - state_climate["normal_rainfall_mm"]
)
state_climate["rainfall_anomaly_pct"] = (
    100
    * state_climate["rainfall_anomaly_mm"]
    / state_climate["normal_rainfall_mm"].replace(0, np.nan)
)
state_climate["temperature_anomaly_c"] = (
    state_climate["temperature_c"]
    - state_climate["normal_temperature_c"]
)

anomaly_columns = [
    "region",
    "date",
    "rainfall_anomaly_mm",
    "rainfall_anomaly_pct",
    "temperature_anomaly_c",
]
clean = clean.merge(
    state_climate[anomaly_columns],
    on=["region", "date"],
    how="left",
    validate="many_to_one",
)
print(f"Rows with rainfall anomaly: {clean['rainfall_anomaly_pct'].notna().sum():,}")
print(f"Rows with temperature anomaly: {clean['temperature_anomaly_c'].notna().sum():,}")
"""
    ),
    markdown(
        """## 9. Clean the national monthly panel

National outcomes are retained as missing when the source does not report them.
No backward fill or interpolation is used because either can leak later
information into earlier dates. Exact date shifts create CPI lags.
"""
    ),
    code(
        """national_clean = national.sort_values("date").copy()
national_clean["is_partial_2026"] = national_clean["date"].dt.year.eq(2026)

# Replace infinite percentage changes, which can arise from a zero denominator,
# with missing values rather than large artificial numbers.
national_clean = national_clean.replace([np.inf, -np.inf], np.nan)

cpi_lookup = national_clean[["date", "food_cpi_2015_100"]].copy()
for months in [1, 3, 6, 12]:
    lagged = cpi_lookup.copy()
    lagged["date"] = lagged["date"] + pd.DateOffset(months=months)
    lagged = lagged.rename(
        columns={"food_cpi_2015_100": f"food_cpi_lag_{months}m_exact"}
    )
    national_clean = national_clean.merge(
        lagged, on="date", how="left", validate="one_to_one"
    )

# Recalculate exact-calendar changes from the level series for consistency.
national_clean["food_cpi_mom_pct_exact"] = (
    national_clean["food_cpi_2015_100"]
    / national_clean["food_cpi_lag_1m_exact"]
    - 1
) * 100
national_clean["food_cpi_yoy_pct_exact"] = (
    national_clean["food_cpi_2015_100"]
    / national_clean["food_cpi_lag_12m_exact"]
    - 1
) * 100

national_validity = {
    "nonpositive_cpi": int(
        (
            national_clean["food_cpi_2015_100"].notna()
            & national_clean["food_cpi_2015_100"].le(0)
        ).sum()
    ),
    "negative_rainfall": int(
        (
            national_clean["state_avg_rainfall_mm"].notna()
            & national_clean["state_avg_rainfall_mm"].lt(0)
        ).sum()
    ),
    "temperature_outside_bounds": int(
        (
            national_clean["state_avg_temperature_c"].notna()
            & ~national_clean["state_avg_temperature_c"].between(-20, 60)
        ).sum()
    ),
}
print(pd.Series(national_validity).to_string())
if any(national_validity.values()):
    raise AssertionError("National validity checks failed.")
"""
    ),
    markdown(
        """## 10. Save cleaned datasets and verify round-trip integrity

Compressed CSV keeps files portable in Colab and GitHub while substantially
reducing storage. Each output is read back immediately to verify its row count.
"""
    ),
    code(
        """STATE_OUTPUT = PROCESSED / "cleaned_state_monthly.csv.gz"
NATIONAL_OUTPUT = PROCESSED / "cleaned_national_monthly.csv.gz"

clean.to_csv(STATE_OUTPUT, index=False, compression="gzip")
national_clean.to_csv(NATIONAL_OUTPUT, index=False, compression="gzip")

# Read only the key columns during the round-trip check to minimize I/O.
state_roundtrip = pd.read_csv(
    STATE_OUTPUT, usecols=["region", "date", "Commodity"]
)
national_roundtrip = pd.read_csv(NATIONAL_OUTPUT, usecols=["date"])

roundtrip_checks = {
    "state_rows_preserved": len(state_roundtrip) == len(clean),
    "national_rows_preserved": len(national_roundtrip) == len(national_clean),
    "state_keys_still_unique": not state_roundtrip.duplicated(
        ["region", "date", "Commodity"]
    ).any(),
    "national_dates_still_unique": not national_roundtrip.duplicated(["date"]).any(),
}
for check, passed in roundtrip_checks.items():
    print(f"{'PASS' if passed else 'FAIL'} — {check}")
if not all(roundtrip_checks.values()):
    raise AssertionError("Saved-file round-trip validation failed.")
"""
    ),
    markdown(
        """## 11. Final execution summary

Share the printed JSON after running all cells. Notebook 03 will use these exact
counts and cleaned files for exploratory analysis and visualization.
"""
    ),
    code(
        """summary = {
    "notebook": "02_data_quality_and_cleaning",
    "status": "completed",
    "input_state_rows": int(len(state)),
    "rows_removed_by_validity_rules": int(validity_flags["any_invalid"].sum()),
    "clean_state_rows": int(len(clean)),
    "retention_rate": float(len(clean) / len(state)),
    "extreme_price_rows_flagged": int(clean["price_extreme_flag"].sum()),
    "extreme_price_percent": float(clean["price_extreme_flag"].mean() * 100),
    "rows_with_exact_1m_price_lag": int(clean["has_exact_1m_lag"].sum()),
    "rows_with_exact_12m_price_lag": int(clean["has_exact_12m_lag"].sum()),
    "climate_anomaly_rows": int(clean["rainfall_anomaly_pct"].notna().sum()),
    "clean_national_rows": int(len(national_clean)),
    "national_rows_with_cpi": int(
        national_clean["food_cpi_2015_100"].notna().sum()
    ),
    "state_start": str(clean["date"].min().date()),
    "state_end": str(clean["date"].max().date()),
    "national_start": str(national_clean["date"].min().date()),
    "national_end": str(national_clean["date"].max().date()),
    "missing_prices_interpolated": False,
    "outlier_raw_values_preserved": True,
    "calendar_lag_method": "exact keyed date joins",
    "output_root": str(OUTPUT_ROOT),
}

SUMMARY_OUTPUT = REPORT_OUTPUT / "02_execution_summary.json"
SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=" * 76)
print("NOTEBOOK 02 EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT")
print("=" * 76)
print(json.dumps(summary, indent=2))
print("\\nGenerated files:")
for path in [
    STATE_OUTPUT,
    NATIONAL_OUTPUT,
    REPORT_OUTPUT / "02_state_missingness.csv",
    REPORT_OUTPUT / "02_national_missingness.csv",
    REPORT_OUTPUT / "02_commodity_coverage.csv",
    REPORT_OUTPUT / "02_validity_rule_counts.csv",
    REPORT_OUTPUT / "02_extreme_price_summary.csv",
    SUMMARY_OUTPUT,
]:
    print(f"- {path} ({path.stat().st_size / 1_000_000:.3f} MB)")
"""
    ),
    markdown(
        """## Notebook 02 conclusion

The cleaned outputs retain the unbalanced structure of official mandi reporting,
preserve every raw price alongside any conservative modeling adjustment, and use
true calendar lags. These decisions reduce leakage and timing errors while
keeping the transformation fully auditable.

### What to send back

After selecting **Runtime → Run all**, send:

1. the final `NOTEBOOK 02 EXECUTION SUMMARY` JSON; and
2. any red error output if a validation check stops the notebook.

Notebook 03 will then create efficient exploratory tables and visualizations using
the verified cleaned datasets.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {
            "name": "02_data_quality_and_cleaning.ipynb",
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    for position, cell in enumerate(cells, start=1):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{OUTPUT.name}:cell-{position}", "exec")

    print(f"Created and syntax-validated: {OUTPUT}")
    print(f"Notebook cells: {len(cells)}")


if __name__ == "__main__":
    main()
