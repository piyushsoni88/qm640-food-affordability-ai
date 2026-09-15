"""Build synopsis-aligned Notebook 02 from the tested cleaning notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "02_data_quality_and_cleaning.ipynb"
OUTPUT = ROOT / "notebooks" / "02_data_quality_and_cleaning_synopsis_aligned.ipynb"


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


ALIGNMENT_MARKDOWN = r"""
## 10. Original-synopsis targets and optional confirmatory-source integration

The synopsis defines future essential-food price change at one-, two-, and
three-month horizons. These targets must be created by exact keyed calendar
joins, not by row offsets. This section also conditionally integrates
standardized arrivals, state crop production/yield, and rural wages when the
official files requested by Notebook 01 are present.

The following rules prevent conceptual drift:

- reporting counts remain coverage variables and are never renamed as arrivals;
- future targets use the state-commodity price panel, while national CPI remains
  a supplementary benchmark;
- unavailable confirmatory predictors remain explicitly missing;
- no missing official value is synthetically generated or silently zero-filled.

The saved confirmatory target columns are
`future_price_change_h1m_pct`, `future_price_change_h2m_pct`, and
`future_price_change_h3m_pct`.
"""


ALIGNMENT_CODE = r"""
EXTERNAL_REQUIRED = OUTPUT_ROOT / "data" / "external_required"
EXTERNAL_REQUIRED.mkdir(parents=True, exist_ok=True)

# Add exact price lags required for transparent autoregressive controls.
for months in [2, 3, 6]:
    if f"price_lag_{months}m" not in clean.columns:
        clean = add_exact_calendar_lag(clean, lag_lookup, months=months)

# Create exact 1-, 2-, and 3-month future targets for RQ1 and RQ2.
current_lookup = clean[
    ["region", "Commodity", "date", "price_cleaned_rs_per_quintal"]
].copy()
for horizon in [1, 2, 3]:
    future = current_lookup.copy()
    future["date"] = future["date"] - pd.DateOffset(months=horizon)
    future = future.rename(columns={
        "price_cleaned_rs_per_quintal": f"future_price_h{horizon}m"
    })
    clean = clean.merge(
        future,
        on=["region", "Commodity", "date"],
        how="left",
        validate="one_to_one",
    )
    clean[f"future_price_change_h{horizon}m_pct"] = (
        clean[f"future_price_h{horizon}m"]
        / clean["price_cleaned_rs_per_quintal"]
        - 1
    ) * 100
    clean[f"has_exact_h{horizon}m_target"] = clean[
        f"future_price_h{horizon}m"
    ].notna()

# National future targets remain supplementary and use the same exact-date rule.
national_lookup = national_clean[["date", "food_cpi_2015_100"]].copy()
for horizon in [1, 2, 3]:
    future = national_lookup.copy()
    future["date"] = future["date"] - pd.DateOffset(months=horizon)
    future = future.rename(columns={
        "food_cpi_2015_100": f"future_food_cpi_h{horizon}m"
    })
    national_clean = national_clean.merge(
        future, on="date", how="left", validate="one_to_one"
    )
    national_clean[f"future_food_cpi_change_h{horizon}m_pct"] = (
        national_clean[f"future_food_cpi_h{horizon}m"]
        / national_clean["food_cpi_2015_100"]
        - 1
    ) * 100


def normalize_label(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )

def normalize_commodity(series):
    normalized = normalize_label(series)
    aliases = {
        "arhar/tur": "arhar (tur/red gram)(whole)",
        "gram": "bengal gram(gram)(whole)",
        "soybean": "soyabean",
    }
    return normalized.replace(aliases)


integration_rows = []

# Genuine arrivals: integrate only the explicitly named standardized file.
arrivals_path = EXTERNAL_REQUIRED / "agmarknet_enam_arrivals.csv.gz"
if arrivals_path.exists():
    arrivals = pd.read_csv(arrivals_path, low_memory=False)
    required = {"date", "state", "commodity", "arrival_quantity"}
    missing = required.difference(arrivals.columns)
    if missing:
        raise ValueError(f"Arrivals file missing required columns: {sorted(missing)}")
    arrivals["date"] = pd.to_datetime(arrivals["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    arrivals["region_key"] = normalize_label(arrivals["state"])
    arrivals["commodity_key"] = normalize_commodity(arrivals["commodity"])
    arrivals["arrival_quantity"] = pd.to_numeric(
        arrivals["arrival_quantity"], errors="coerce"
    )
    arrivals_monthly = (
        arrivals.dropna(subset=["date", "region_key", "commodity_key", "arrival_quantity"])
        .groupby(["region_key", "commodity_key", "date"], as_index=False)
        .agg(
            market_arrival_quantity=("arrival_quantity", "sum"),
            arrival_source_rows=("arrival_quantity", "size"),
        )
    )
    clean["region_key"] = normalize_label(clean["region"])
    clean["commodity_key"] = normalize_commodity(clean["Commodity"])
    clean = clean.merge(
        arrivals_monthly,
        on=["region_key", "commodity_key", "date"],
        how="left",
        validate="many_to_one",
    )
    integration_rows.append(["market_arrivals", "integrated", len(arrivals), clean["market_arrival_quantity"].notna().mean()])
else:
    clean["market_arrival_quantity"] = np.nan
    clean["arrival_source_rows"] = np.nan
    integration_rows.append(["market_arrivals", "missing", 0, 0.0])

# State crop area/production/yield: join at state-commodity-year resolution.
apy_path = EXTERNAL_REQUIRED / "des_state_crop_area_production_yield.csv.gz"
if apy_path.exists():
    apy = pd.read_csv(apy_path, low_memory=False)
    required = {"state", "crop", "year", "production_tonne", "yield_kg_per_hectare"}
    missing = required.difference(apy.columns)
    if missing:
        raise ValueError(f"DES APY file missing required columns: {sorted(missing)}")
    apy["region_key"] = normalize_label(apy["state"])
    apy["commodity_key"] = normalize_commodity(apy["crop"])
    apy["analysis_year"] = pd.to_numeric(apy["year"], errors="coerce").astype("Int64")
    apy_year = (
        apy.groupby(["region_key", "commodity_key", "analysis_year"], as_index=False)
        .agg(
            production_tonne=("production_tonne", "sum"),
            yield_kg_per_hectare=("yield_kg_per_hectare", "mean"),
        )
    )
    if "region_key" not in clean:
        clean["region_key"] = normalize_label(clean["region"])
        clean["commodity_key"] = normalize_commodity(clean["Commodity"])
    clean["analysis_year"] = clean["date"].dt.year.astype("Int64")
    clean = clean.merge(
        apy_year,
        on=["region_key", "commodity_key", "analysis_year"],
        how="left",
        validate="many_to_one",
    )
    integration_rows.append(["state_crop_apy", "integrated", len(apy), clean["production_tonne"].notna().mean()])
else:
    clean["production_tonne"] = np.nan
    clean["yield_kg_per_hectare"] = np.nan
    integration_rows.append(["state_crop_apy", "missing", 0, 0.0])

# Rural wages: aggregate occupations to state-month purchasing-power growth.
wage_path = EXTERNAL_REQUIRED / "labour_bureau_rural_wages_monthly.csv.gz"
if wage_path.exists():
    wages = pd.read_csv(wage_path, low_memory=False)
    required = {"state"}
    missing = required.difference(wages.columns)
    if missing:
        raise ValueError(f"Wage file missing required columns: {sorted(missing)}")
    wage_columns = [
        column for column in
        ["male_wage_rs_per_day", "female_wage_rs_per_day"]
        if column in wages.columns
    ]
    if not wage_columns:
        raise ValueError("Wage file requires a male and/or female daily-wage column.")
    for column in wage_columns:
        wages[column] = pd.to_numeric(wages[column], errors="coerce")
    wages["rural_wage_rs_per_day"] = wages[wage_columns].mean(axis=1)
    if "date" in wages.columns:
        wages["date"] = pd.to_datetime(wages["date"], errors="coerce")
    else:
        if not {"year", "month"}.issubset(wages.columns):
            raise ValueError("Wage file requires either date or year/month columns.")
        month_number = pd.to_numeric(wages["month"], errors="coerce")
        if month_number.isna().any():
            month_number = pd.to_datetime(
                wages["month"].astype(str), format="%b", errors="coerce"
            ).dt.month
        wages["date"] = pd.to_datetime(dict(
            year=pd.to_numeric(wages["year"], errors="coerce"),
            month=month_number,
            day=1,
        ), errors="coerce")
    wages["region_key"] = normalize_label(wages["state"])
    wage_monthly = (
        wages.dropna(subset=["date", "region_key", "rural_wage_rs_per_day"])
        .groupby(["region_key", "date"], as_index=False)
        .agg(rural_wage_rs_per_day=("rural_wage_rs_per_day", "mean"))
    )
    if "region_key" not in clean:
        clean["region_key"] = normalize_label(clean["region"])
    clean = clean.merge(
        wage_monthly, on=["region_key", "date"], how="left", validate="many_to_one"
    )
    clean["rural_wage_yoy_pct"] = (
        clean.sort_values(["region", "Commodity", "date"])
        .groupby(["region", "Commodity"], observed=True)["rural_wage_rs_per_day"]
        .pct_change(12, fill_method=None)
        * 100
    )
    wage_status = (
        "integrated_partial_coverage"
        if "coverage_status" in wages.columns
        else "integrated"
    )
    integration_rows.append(["rural_wages", wage_status, len(wages), clean["rural_wage_rs_per_day"].notna().mean()])
else:
    clean["rural_wage_rs_per_day"] = np.nan
    clean["rural_wage_yoy_pct"] = np.nan
    integration_rows.append(["rural_wages", "missing", 0, 0.0])

hces_candidates = list(EXTERNAL_REQUIRED.glob("hces*segment*aggregate*.csv*"))
hces_status = "available_aggregate_for_notebook_08" if hces_candidates else "missing"
integration_rows.append(["hces_microdata", hces_status, 0, float(bool(hces_candidates))])

integration_audit = pd.DataFrame(
    integration_rows,
    columns=["source", "status", "input_rows", "state_panel_match_rate"],
)
integration_audit.to_csv(REPORT_OUTPUT / "02_confirmatory_source_integration.csv", index=False)

target_coverage = pd.DataFrame([
    {
        "horizon_months": horizon,
        "rows_with_target": int(clean[f"has_exact_h{horizon}m_target"].sum()),
        "coverage_rate": float(clean[f"has_exact_h{horizon}m_target"].mean()),
        "target_column": f"future_price_change_h{horizon}m_pct",
    }
    for horizon in [1, 2, 3]
])
target_coverage.to_csv(REPORT_OUTPUT / "02_future_target_coverage.csv", index=False)

print("Exact future-target coverage")
print(target_coverage.to_string(index=False))
print("\nConfirmatory source integration")
print(integration_audit.to_string(index=False))

# Remove temporary normalized keys; the human-readable controlled labels remain.
clean = clean.drop(
    columns=["region_key", "commodity_key", "analysis_year"],
    errors="ignore",
)
"""


SUMMARY_CODE = r"""
summary = {
    "notebook": "02_data_quality_and_cleaning_synopsis_aligned",
    "status": "completed",
    "original_research_questions_preserved": True,
    "input_state_rows": int(len(state)),
    "rows_removed_by_validity_rules": int(validity_flags["any_invalid"].sum()),
    "clean_state_rows": int(len(clean)),
    "retention_rate": float(len(clean) / len(state)),
    "extreme_price_rows_flagged": int(clean["price_extreme_flag"].sum()),
    "extreme_price_percent": float(clean["price_extreme_flag"].mean() * 100),
    "rows_with_exact_1m_price_lag": int(clean["has_exact_1m_lag"].sum()),
    "rows_with_exact_12m_price_lag": int(clean["has_exact_12m_lag"].sum()),
    "future_target_rows": {
        f"{horizon}_month": int(clean[f"has_exact_h{horizon}m_target"].sum())
        for horizon in [1, 2, 3]
    },
    "future_target_coverage": {
        f"{horizon}_month": float(clean[f"has_exact_h{horizon}m_target"].mean())
        for horizon in [1, 2, 3]
    },
    "confirmatory_source_status": dict(
        zip(integration_audit["source"], integration_audit["status"])
    ),
    "market_arrivals_is_genuine_quantity_only": True,
    "reporting_counts_not_used_as_arrivals": True,
    "climate_anomaly_rows": int(clean["rainfall_anomaly_pct"].notna().sum()),
    "clean_national_rows": int(len(national_clean)),
    "national_rows_with_cpi": int(national_clean["food_cpi_2015_100"].notna().sum()),
    "state_start": str(clean["date"].min().date()),
    "state_end": str(clean["date"].max().date()),
    "national_start": str(national_clean["date"].min().date()),
    "national_end": str(national_clean["date"].max().date()),
    "missing_prices_interpolated": False,
    "outlier_raw_values_preserved": True,
    "calendar_lag_and_target_method": "exact keyed date joins",
    "output_root": str(OUTPUT_ROOT),
}

SUMMARY_OUTPUT = REPORT_OUTPUT / "02_execution_summary.json"
SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=" * 78)
print("NOTEBOOK 02 ALIGNED EXECUTION SUMMARY - PLEASE SHARE THIS OUTPUT")
print("=" * 78)
print(json.dumps(summary, indent=2))
print("\nGenerated files:")
for path in [
    STATE_OUTPUT,
    NATIONAL_OUTPUT,
    REPORT_OUTPUT / "02_state_missingness.csv",
    REPORT_OUTPUT / "02_national_missingness.csv",
    REPORT_OUTPUT / "02_commodity_coverage.csv",
    REPORT_OUTPUT / "02_validity_rule_counts.csv",
    REPORT_OUTPUT / "02_extreme_price_summary.csv",
    REPORT_OUTPUT / "02_confirmatory_source_integration.csv",
    REPORT_OUTPUT / "02_future_target_coverage.csv",
    SUMMARY_OUTPUT,
]:
    print(f"- {path} ({path.stat().st_size / 1_000_000:.3f} MB)")
"""


CONCLUSION = r"""
## Notebook 02 conclusion

The cleaned panel now contains exact one-, two-, and three-month future
commodity-price targets required by the original RQ1 and RQ2. Missing
confirmatory sources are visible in the integration audit and remain missing
rather than being replaced with fabricated values. Notebook 03 may proceed for
interim EDA while the official downloads continue in parallel.
"""


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    cells[0] = markdown(
        """
# QM640 Food Price Affordability AI
## Notebook 02 - Synopsis-Aligned Quality, Cleaning, and Target Engineering

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Required predecessor:** Notebook 01 synopsis-aligned version

This notebook preserves raw values, performs exact calendar joins, creates the
original one-, two-, and three-month future price targets, and conditionally
integrates official confirmatory data without silently substituting proxies.
"""
    )

    # Insert alignment cells before the existing save-and-roundtrip section.
    insert_at = 19
    cells[insert_at:insert_at] = [
        markdown(ALIGNMENT_MARKDOWN),
        code(ALIGNMENT_CODE),
    ]

    # Existing final summary moves from index 22 to 24 after the insertion.
    cells[24] = code(SUMMARY_CODE)
    cells[25] = markdown(CONCLUSION)
    notebook["metadata"].setdefault("colab", {})
    notebook["metadata"]["colab"]["name"] = OUTPUT.name
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()
