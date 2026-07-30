"""Build the Google Colab version of Notebook 01.

Keeping notebook construction in a small script makes the JSON reproducible and
allows syntax validation without hand-editing the .ipynb container.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_data_acquisition.ipynb"


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
## Notebook 01 — Data Acquisition, Audit, and Integration

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Study period:** 2000–2026 YTD, subject to source availability

### Why this notebook exists

This notebook creates the reproducible data foundation for the assignment. It:

1. downloads the version-controlled curated datasets from GitHub;
2. audits their row counts, schemas, and compressed sizes;
3. processes the large AGMARKNET table in chunks to control memory use;
4. harmonizes legacy state/UT names before spatial joins;
5. aggregates prices to state–commodity–month using official observation counts;
6. joins mandi prices to monthly climate and national macroeconomic variables;
7. validates totals, uniqueness, date coverage, and join quality; and
8. saves modeling-ready datasets and an execution summary to Google Drive.

Raw API credentials are not required because this notebook uses the curated,
versioned research data already collected from the official sources.
"""
    ),
    markdown(
        """## 1. Runtime configuration

Google Colab already includes pandas and NumPy, so no package installation is
needed. Avoiding unnecessary installations makes execution faster and more
reliable.

Set `SAVE_TO_DRIVE = True` to keep results after the Colab session ends. Google
Drive will ask for authorization once.
"""
    ),
    code(
        """# Standard-library imports handle files, Git, JSON, and environment detection.
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import warnings

# pandas provides efficient tabular operations; NumPy provides vectorized arithmetic.
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)

# Change this to False if you only want temporary files inside the Colab session.
SAVE_TO_DRIVE = True

REPO_URL = "https://github.com/piyushsoni88/qm640-food-affordability-ai.git"
REPO_NAME = "qm640-food-affordability-ai"

# Import detection is more reliable than checking sys.modules because Colab may
# not preload google.colab before the first user cell runs.
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Running in Google Colab: {IN_COLAB}")
print(f"pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
"""
    ),
    markdown(
        """## 2. Connect Google Drive and obtain the repository

The repository is cloned with `--depth 1` because earlier Git history is not
needed for analysis. This reduces download time and temporary storage. If the
repository already exists in the current Colab runtime, it is updated instead.
"""
    ),
    code(
        """# Mount Drive only when requested and when the notebook is running in Colab.
if IN_COLAB and SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

# Colab uses /content for fast temporary computation. When run locally, this cell
# detects the repository from the current working directory instead.
if IN_COLAB:
    REPO_ROOT = Path("/content") / REPO_NAME
    if not REPO_ROOT.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_ROOT)],
            check=True,
        )
    else:
        # A fast-forward-only pull avoids accidental merge commits in a notebook.
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "pull", "--ff-only"],
            check=True,
        )
else:
    candidate = Path.cwd().resolve()
    REPO_ROOT = candidate.parent if candidate.name == "notebooks" else candidate
    if not (REPO_ROOT / "data" / "curated").exists():
        raise FileNotFoundError(
            "Run this notebook from the repository root or its notebooks folder."
        )

CURATED = REPO_ROOT / "data" / "curated"

# Persist outputs in Drive, while using /content for the cloned input repository.
if IN_COLAB and SAVE_TO_DRIVE:
    OUTPUT_ROOT = Path("/content/drive/MyDrive/QM640_Food_Affordability")
else:
    # The optional override is useful for non-Colab validation and automated tests.
    OUTPUT_ROOT = Path(os.environ.get("QM640_OUTPUT_ROOT", str(REPO_ROOT)))

PROCESSED = OUTPUT_ROOT / "data" / "processed"
REPORT_OUTPUT = OUTPUT_ROOT / "reports" / "notebook_outputs"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"Input repository: {REPO_ROOT}")
print(f"Curated inputs:   {CURATED}")
print(f"Saved outputs:    {OUTPUT_ROOT}")
"""
    ),
    markdown(
        """## 3. Audit all curated input files

Each CSV is read in chunks. Chunking caps peak memory rather than loading every
dataset at once. The resulting inventory supplies evidence for the interim report
and detects accidentally missing or empty files.
"""
    ),
    code(
        """def audit_csv_gz(path: Path, chunksize: int = 250_000) -> dict:
    \"\"\"Return actual rows/columns for one compressed CSV with bounded memory.\"\"\"
    row_count = 0
    columns = None
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        row_count += len(chunk)
        if columns is None:
            columns = list(chunk.columns)
    return {
        "file": path.name,
        "rows": row_count,
        "columns": len(columns or []),
        "compressed_bytes": path.stat().st_size,
        "compressed_mb": path.stat().st_size / 1_000_000,
    }


started = time.perf_counter()
curated_files = sorted(CURATED.glob("*.csv.gz"))
if not curated_files:
    raise FileNotFoundError(f"No curated .csv.gz files were found in {CURATED}")

inventory = pd.DataFrame([audit_csv_gz(path) for path in curated_files])
inventory = inventory.sort_values("rows", ascending=False).reset_index(drop=True)
inventory.to_csv(PROCESSED / "source_inventory.csv", index=False)

print(inventory.to_string(index=False, formatters={"compressed_mb": "{:.2f}".format}))
print(f"\\nFiles audited: {len(inventory)}")
print(f"Curated analytical rows: {inventory['rows'].sum():,}")
print(f"Compressed input size: {inventory['compressed_mb'].sum():,.2f} MB")
print(f"Audit elapsed time: {time.perf_counter() - started:,.1f} seconds")
"""
    ),
    markdown(
        """## 4. Efficiently aggregate historical AGMARKNET prices

The curated AGMARKNET file contains daily state–commodity aggregates representing
the much larger official raw dataset. We use:

- `usecols` to avoid loading unused columns;
- explicit dtypes to reduce memory;
- `chunksize` to bound peak RAM;
- vectorized weighted-price calculation; and
- two-stage aggregation, which is mathematically equivalent to a full group-by.

Weights are `source_rows`, so a daily mean based on many official observations
contributes more than a daily mean based on one observation.
"""
    ),
    code(
        """AGMARKNET_FILE = (
    CURATED / "agmarknet_historical_8_commodities_daily_state_2000_2026.csv.gz"
)
AG_COLS = [
    "date",
    "State",
    "Commodity",
    "source_rows",
    "mean_modal_price_rs_per_quintal",
]

# Legacy labels are mapped to the labels used by the 36-state/UT climate panel.
# This prevents avoidable join failures without changing the underlying geography.
STATE_NAME_MAP = {
    "Andaman and Nicobar": "Andaman and Nicobar Islands",
    "Chattisgarh": "Chhattisgarh",
    "Keralam": "Kerala",
    "NCT of Delhi": "Delhi",
    "Uttrakhand": "Uttarakhand",
}

partial_aggregates = []
daily_aggregate_rows = 0
underlying_source_rows = 0
started = time.perf_counter()

reader = pd.read_csv(
    AGMARKNET_FILE,
    usecols=AG_COLS,
    dtype={
        "State": "category",
        "Commodity": "category",
        "source_rows": "int64",
        "mean_modal_price_rs_per_quintal": "float64",
    },
    parse_dates=["date"],
    chunksize=250_000,
)

for part_number, chunk in enumerate(reader, start=1):
    daily_aggregate_rows += len(chunk)
    underlying_source_rows += int(chunk["source_rows"].sum())

    # Convert to strings before replacement because the mapped labels introduce
    # values that are not necessarily present in the original categorical dtype.
    chunk["region"] = chunk["State"].astype(str).replace(STATE_NAME_MAP)
    chunk["date"] = chunk["date"].dt.to_period("M").dt.to_timestamp()

    # Weighted sums let us combine chunks exactly in a second aggregation pass.
    chunk["weighted_price"] = (
        chunk["mean_modal_price_rs_per_quintal"] * chunk["source_rows"]
    )
    grouped = (
        chunk.groupby(
            ["date", "region", "Commodity"],
            as_index=False,
            observed=True,
            sort=False,
        )
        .agg(
            weighted_price=("weighted_price", "sum"),
            source_rows=("source_rows", "sum"),
            daily_records=("State", "size"),
        )
    )
    partial_aggregates.append(grouped)
    print(f"Processed chunk {part_number}: {len(chunk):,} rows")

# The second pass combines groups that occurred in different input chunks.
monthly_prices = (
    pd.concat(partial_aggregates, ignore_index=True)
    .groupby(
        ["date", "region", "Commodity"],
        as_index=False,
        observed=True,
        sort=False,
    )
    .agg(
        weighted_price=("weighted_price", "sum"),
        source_rows=("source_rows", "sum"),
        daily_records=("daily_records", "sum"),
    )
)
monthly_prices["modal_price_rs_per_quintal"] = (
    monthly_prices["weighted_price"] / monthly_prices["source_rows"]
)
monthly_prices = monthly_prices.drop(columns="weighted_price").sort_values(
    ["date", "region", "Commodity"]
)

monthly_price_path = PROCESSED / "agmarknet_monthly_state_commodity.csv.gz"
monthly_prices.to_csv(monthly_price_path, index=False, compression="gzip")

print("\\nAGMARKNET aggregation results")
print(f"Daily aggregate rows read: {daily_aggregate_rows:,}")
print(f"Underlying official observations: {underlying_source_rows:,}")
print(f"Monthly state–commodity rows: {len(monthly_prices):,}")
print(f"States/UT labels: {monthly_prices['region'].nunique()}")
print(f"Commodities: {monthly_prices['Commodity'].nunique()}")
print(f"Date range: {monthly_prices['date'].min().date()} to {monthly_prices['date'].max().date()}")
print(f"Elapsed time: {time.perf_counter() - started:,.1f} seconds")
print(f"Saved: {monthly_price_path}")
"""
    ),
    markdown(
        """## 5. Validate the AGMARKNET aggregation

Assertions intentionally stop execution if key totals, uniqueness, or price rules
fail. A failed assertion is evidence that the source or transformation changed
and should be investigated before modeling.
"""
    ),
    code(
        """# The known source total comes from the downloaded official historical snapshot.
EXPECTED_UNDERLYING_OBSERVATIONS = 18_836_462

validation = {
    "underlying_total_matches_snapshot": (
        underlying_source_rows == EXPECTED_UNDERLYING_OBSERVATIONS
    ),
    "source_rows_preserved_after_aggregation": (
        int(monthly_prices["source_rows"].sum()) == underlying_source_rows
    ),
    "unique_state_month_commodity_keys": (
        not monthly_prices.duplicated(["date", "region", "Commodity"]).any()
    ),
    "all_prices_positive": (
        monthly_prices["modal_price_rs_per_quintal"].gt(0).all()
    ),
    "all_weights_positive": monthly_prices["source_rows"].gt(0).all(),
}

for check, passed in validation.items():
    print(f"{'PASS' if passed else 'FAIL'} — {check}")

if not all(validation.values()):
    raise AssertionError("One or more AGMARKNET validation checks failed.")

commodity_coverage = (
    monthly_prices.groupby("Commodity", observed=True, as_index=False)
    .agg(
        monthly_rows=("date", "size"),
        states_uts=("region", "nunique"),
        start_date=("date", "min"),
        end_date=("date", "max"),
        underlying_observations=("source_rows", "sum"),
    )
    .sort_values("underlying_observations", ascending=False)
)
commodity_coverage.to_csv(REPORT_OUTPUT / "01_commodity_coverage.csv", index=False)
print("\\nCommodity coverage")
print(commodity_coverage.to_string(index=False))
"""
    ),
    markdown(
        """## 6. Join monthly prices to NASA POWER climate data

This is a many-to-one join: each state–commodity–month price record receives one
state/UT-month climate record. `validate="many_to_one"` makes pandas raise an
error if the climate keys are unexpectedly duplicated.

Climate values are representative administrative-capital points, not
area-weighted state averages; this limitation must remain in the report.
"""
    ),
    code(
        """CLIMATE_FILE = (
    CURATED / "nasa_power_india_all_states_uts_monthly_2000_2026_ytd.csv.gz"
)
climate_columns = [
    "region",
    "admin_type",
    "reference_location",
    "spatial_representation",
    "date",
    "rainfall_mm",
    "temperature_c",
    "temperature_max_c",
    "temperature_min_c",
    "relative_humidity_pct",
    "observed_days",
    "expected_days",
    "is_partial_month",
    "data_status",
]
climate = pd.read_csv(
    CLIMATE_FILE,
    usecols=climate_columns,
    parse_dates=["date"],
)

if climate.duplicated(["region", "date"]).any():
    raise AssertionError("Climate data contains duplicate region-date keys.")

state_modeling_panel = monthly_prices.merge(
    climate,
    on=["region", "date"],
    how="left",
    validate="many_to_one",
    indicator=True,
)
climate_match_rate = state_modeling_panel["_merge"].eq("both").mean()
unmatched_regions = sorted(
    state_modeling_panel.loc[
        state_modeling_panel["_merge"].ne("both"), "region"
    ].unique()
)
state_modeling_panel = state_modeling_panel.drop(columns="_merge")

state_panel_path = PROCESSED / "modeling_state_monthly.csv.gz"
state_modeling_panel.to_csv(state_panel_path, index=False, compression="gzip")

print(f"State modeling rows: {len(state_modeling_panel):,}")
print(f"Climate match rate: {climate_match_rate:.2%}")
print(f"Unmatched price-region labels: {unmatched_regions or 'None'}")
print(f"Saved: {state_panel_path}")
"""
    ),
    markdown(
        """## 7. Construct the national monthly modeling panel

Commodity prices have different units and price levels. Each commodity series is
therefore rebased to its own 2015 mean (=100), then averaged across observed
commodities. This avoids letting an intrinsically expensive commodity dominate
the national mandi index.

The index is joined to national food CPI, World Bank commodity prices, and mean
state climate. Observation counts and commodity counts are retained as coverage
indicators.
"""
    ),
    code(
        """# First calculate weighted national prices within commodity and month.
national_commodity = monthly_prices.assign(
    weighted_price=lambda frame: (
        frame["modal_price_rs_per_quintal"] * frame["source_rows"]
    )
)
national_commodity = (
    national_commodity.groupby(
        ["date", "Commodity"],
        as_index=False,
        observed=True,
        sort=False,
    )
    .agg(
        weighted_price=("weighted_price", "sum"),
        source_rows=("source_rows", "sum"),
    )
)
national_commodity["national_price"] = (
    national_commodity["weighted_price"] / national_commodity["source_rows"]
)

# Use the 2015 mean as the common base. All eight assignment commodities have
# 2015 observations; the explicit check protects against silent missing bases.
base_2015 = (
    national_commodity.loc[national_commodity["date"].dt.year.eq(2015)]
    .groupby("Commodity", observed=True)["national_price"]
    .mean()
)
missing_bases = sorted(
    set(national_commodity["Commodity"].astype(str).unique()) - set(base_2015.index.astype(str))
)
if missing_bases:
    raise AssertionError(f"Missing 2015 base prices for: {missing_bases}")

# Mapping from a categorical key can preserve a categorical result in newer
# pandas versions. Convert explicitly to float before arithmetic for portability.
national_commodity["base_2015"] = (
    national_commodity["Commodity"].map(base_2015).astype("float64")
)
national_commodity["commodity_price_index_2015_100"] = (
    100 * national_commodity["national_price"] / national_commodity["base_2015"]
)

national_mandi = (
    national_commodity.groupby("date", as_index=False)
    .agg(
        mandi_price_index_2015_100=("commodity_price_index_2015_100", "mean"),
        mandi_source_rows=("source_rows", "sum"),
        commodities_observed=("Commodity", "nunique"),
    )
    .sort_values("date")
)

# This curated table already contains food CPI, World Bank prices, and national
# climate features. We add the independent AGMARKNET mandi index.
national_existing = pd.read_csv(
    CURATED / "india_food_affordability_national_monthly_2000_2026_ytd.csv.gz",
    parse_dates=["date"],
)
climate_national = (
    climate.groupby("date", as_index=False)
    .agg(
        state_avg_rainfall_mm=("rainfall_mm", "mean"),
        state_avg_temperature_c=("temperature_c", "mean"),
        reporting_regions=("region", "nunique"),
    )
)

national_modeling = (
    national_existing.merge(national_mandi, on="date", how="left")
    .merge(climate_national, on="date", how="left")
    .sort_values("date")
)

# pct_change uses only past values, preserving time direction for later models.
national_modeling["mandi_index_mom_pct"] = (
    national_modeling["mandi_price_index_2015_100"].pct_change() * 100
)
national_modeling["mandi_index_yoy_pct"] = (
    national_modeling["mandi_price_index_2015_100"].pct_change(12) * 100
)

national_path = PROCESSED / "modeling_monthly_national.csv.gz"
national_modeling.to_csv(national_path, index=False, compression="gzip")

print(f"National monthly rows: {len(national_modeling):,}")
print(f"Date range: {national_modeling['date'].min().date()} to {national_modeling['date'].max().date()}")
print(f"Rows with CPI: {national_modeling['food_cpi_2015_100'].notna().sum():,}")
print(f"Rows with mandi index: {national_modeling['mandi_price_index_2015_100'].notna().sum():,}")
print(f"Saved: {national_path}")
"""
    ),
    markdown(
        """## 8. Final quality summary and hand-off

The JSON summary below is the key result to share before Notebook 02 is created.
It records the exact processed counts, source total, temporal coverage, match
rate, and output locations.
"""
    ),
    code(
        """summary = {
    "notebook": "01_data_acquisition",
    "status": "completed",
    "curated_files_audited": int(len(inventory)),
    "curated_analytical_rows": int(inventory["rows"].sum()),
    "agmarknet_daily_aggregate_rows": int(daily_aggregate_rows),
    "agmarknet_underlying_official_observations": int(underlying_source_rows),
    "agmarknet_monthly_state_commodity_rows": int(len(monthly_prices)),
    "agmarknet_states_uts": int(monthly_prices["region"].nunique()),
    "agmarknet_commodities": int(monthly_prices["Commodity"].nunique()),
    "agmarknet_start": str(monthly_prices["date"].min().date()),
    "agmarknet_end": str(monthly_prices["date"].max().date()),
    "state_modeling_rows": int(len(state_modeling_panel)),
    "climate_match_rate": float(climate_match_rate),
    "national_modeling_rows": int(len(national_modeling)),
    "national_start": str(national_modeling["date"].min().date()),
    "national_end": str(national_modeling["date"].max().date()),
    "partial_2026_warning": True,
    "climate_spatial_limitation": (
        "Representative administrative-capital points; not area-weighted state averages."
    ),
    "output_root": str(OUTPUT_ROOT),
}

summary_path = REPORT_OUTPUT / "01_execution_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=" * 72)
print("NOTEBOOK 01 EXECUTION SUMMARY — PLEASE SHARE THIS OUTPUT")
print("=" * 72)
print(json.dumps(summary, indent=2))
print("\\nGenerated files:")
for path in [
    PROCESSED / "source_inventory.csv",
    monthly_price_path,
    state_panel_path,
    national_path,
    REPORT_OUTPUT / "01_commodity_coverage.csv",
    summary_path,
]:
    print(f"- {path} ({path.stat().st_size / 1_000_000:.2f} MB)")
"""
    ),
    markdown(
        """## Notebook 01 conclusion

The acquisition stage preserves the full official AGMARKNET observation count
represented by the curated historical extract, harmonizes geography, and creates
two efficient modeling tables:

- a state–commodity–month price and climate panel; and
- a national monthly CPI, mandi, world-price, and climate panel.

### What to send back

After running **Runtime → Run all** in Colab, please provide:

1. the final `NOTEBOOK 01 EXECUTION SUMMARY` JSON printed above; and
2. any red error message if a validation check fails.

Notebook 02 will use these verified counts to perform missingness analysis,
outlier treatment, time-series cleaning, and data-quality reporting.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {
            "name": "01_data_acquisition.ipynb",
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

    # Compile every code cell now so syntax errors are caught before Colab use.
    for position, cell in enumerate(cells, start=1):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{OUTPUT.name}:cell-{position}", "exec")

    print(f"Created and syntax-validated: {OUTPUT}")
    print(f"Notebook cells: {len(cells)}")


if __name__ == "__main__":
    main()
