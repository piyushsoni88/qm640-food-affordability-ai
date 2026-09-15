"""Build the synopsis-aligned Colab version of Notebook 01.

The existing acquisition notebook remains the tested foundation. This builder
adds a fail-visible audit for every source required by the original synopsis,
creates upload templates, and prevents proxy variables from being mislabeled
as confirmatory inputs.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "01_data_acquisition.ipynb"
OUTPUT = ROOT / "notebooks" / "01_data_acquisition_synopsis_aligned.ipynb"


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
## 8. Original-synopsis source readiness and data-gap audit

The original synopsis remains the controlling research design. This section
does **not** redefine a research question to match currently available data.
Instead, it distinguishes:

1. a confirmatory source that is available;
2. an interim proxy that may support preliminary analysis only; and
3. a missing source that must be acquired before the relevant null hypothesis
   can be finally tested.

Important interpretation rules:

- `source_rows` and `daily_records` measure reporting coverage; they are **not**
  market-arrival quantities.
- NASA POWER climate data are a reproducible interim climate source; they are
  not relabeled as IMD observations.
- FAOSTAT national annual production is not a substitute for state-commodity
  area, production, and yield.
- Published rural/urban HCES shares support a prototype HFASI but do not provide
  household-level or expenditure-fractile validity evidence.

The audit preserves the four original hypothesis pairs:

- **H0-1/H1-1:** joint food-price driver coefficients equal zero versus at
  least one nonzero coefficient.
- **H0-2/H1-2:** machine-learning forecast loss is not lower versus lower than
  the best statistical baseline.
- **H0-3/H1-3:** HFASI lacks versus demonstrates positive criterion validity
  and stable household-segment rankings.
- **H0-4/H1-4:** model-driven warnings do not reduce versus reduce paired
  retrospective decision loss.
"""


ALIGNMENT_CODE = r"""
# Store manually downloaded confirmatory files in this stable Drive folder.
EXTERNAL_REQUIRED = OUTPUT_ROOT / "data" / "external_required"
INPUT_TEMPLATES = OUTPUT_ROOT / "data" / "input_templates"
EXTERNAL_REQUIRED.mkdir(parents=True, exist_ok=True)
INPUT_TEMPLATES.mkdir(parents=True, exist_ok=True)

# The URLs are official discovery/download pages. Interactive portals may require
# a one-time manual export because their session tokens are not stable APIs.
required_sources = [
    {
        "source_id": "agmarknet_prices",
        "rq": "RQ1/RQ2",
        "required_variables": "date, state, market/region, commodity, modal price",
        "status_without_upload": "available_confirmatory",
        "current_evidence": "Curated official AGMARKNET price aggregate",
        "upload_patterns": [],
        "official_url": "https://agmarknet.gov.in/",
        "target_filename": "not_required_currently.csv",
        "critical": True,
    },
    {
        "source_id": "market_arrivals",
        "rq": "RQ1/RQ2",
        "required_variables": "date, state, market, commodity, arrival quantity, unit",
        "status_without_upload": "missing_confirmatory",
        "current_evidence": "No genuine arrival-quantity field; source_rows is coverage only",
        "upload_patterns": ["*arrival*.csv", "*arrival*.csv.gz", "*arrival*.xlsx"],
        "official_url": "https://enam.gov.in/web/dashboard/Historical",
        "target_filename": "agmarknet_enam_arrivals.csv.gz",
        "critical": True,
    },
    {
        "source_id": "state_crop_apy",
        "rq": "RQ1",
        "required_variables": "state, crop, year/season, area, production, yield",
        "status_without_upload": "provisional_proxy",
        "current_evidence": "Official DES state/district crop APY acquired and normalized",
        "upload_patterns": ["*apy*.csv", "*apy*.csv.gz", "*apy*.xlsx", "*production_yield*.csv*"],
        "official_url": "https://data.desagri.gov.in/website/apy-index-report-web",
        "target_filename": "des_state_crop_area_production_yield.csv.gz",
        "critical": True,
    },
    {
        "source_id": "rural_wages",
        "rq": "RQ1/RQ3",
        "required_variables": "year, month, state, occupation, male wage, female wage",
        "status_without_upload": "missing_confirmatory",
        "current_evidence": "Official Labour Bureau FY2025-26 portal export; partial historical coverage",
        "upload_patterns": ["*rural*wage*.csv", "*rural*wage*.csv.gz", "*rural*wage*.xlsx"],
        "official_url": "https://www.labourbureau.gov.in/rural-wages",
        "target_filename": "labour_bureau_rural_wages_monthly.csv.gz",
        "critical": True,
    },
    {
        "source_id": "hces_microdata",
        "rq": "RQ3",
        "required_variables": "household key, sector, state, survey weight, food expenditure, MPCE",
        "status_without_upload": "missing_confirmatory",
        "current_evidence": "HCES 2022-23/2023-24 microdata acquired locally; disclosure-safe segment aggregate prepared",
        "upload_patterns": ["*hces*.csv", "*hces*.csv.gz", "*hces*.dta", "*hces*.sav", "*hces*.zip"],
        "official_url": "https://microdata.gov.in/NADA/index.php/catalog/224",
        "target_filename": "hces_2022_23_unit_level_files.zip",
        "critical": True,
    },
    {
        "source_id": "mospi_cpi_cfpi",
        "rq": "RQ1/RQ2/RQ3",
        "required_variables": "month, CPI/CFPI index, group, sector/state where available",
        "status_without_upload": "provisional_proxy",
        "current_evidence": "National food-CPI series available; official MOSPI confirmation preferred",
        "upload_patterns": ["*mospi*cpi*.csv", "*mospi*cpi*.xlsx", "*cfpi*.csv", "*cfpi*.xlsx"],
        "official_url": "https://cpi.mospi.gov.in/",
        "target_filename": "mospi_cpi_cfpi_monthly.csv.gz",
        "critical": False,
    },
    {
        "source_id": "climate",
        "rq": "RQ1",
        "required_variables": "date, state, rainfall, temperature, humidity",
        "status_without_upload": "available_interim_proxy",
        "current_evidence": "NASA POWER state/UT climate panel",
        "upload_patterns": ["*imd*rain*.csv", "*imd*climate*.csv", "*imd*.xlsx"],
        "official_url": "https://mausam.imd.gov.in/",
        "target_filename": "imd_state_monthly_climate.csv.gz",
        "critical": False,
    },
    {
        "source_id": "global_food_energy",
        "rq": "RQ1/RQ2",
        "required_variables": "month, wheat/rice/oil/sugar prices, crude-oil price",
        "status_without_upload": "available_confirmatory",
        "current_evidence": "World Bank Pink Sheet monthly indicators",
        "upload_patterns": [],
        "official_url": "https://www.worldbank.org/en/research/commodity-markets",
        "target_filename": "not_required_currently.csv",
        "critical": False,
    },
]


def uploaded_matches(patterns):
    # Return matching user-supplied files without reading large microdata.
    matches = []
    for pattern in patterns:
        matches.extend(EXTERNAL_REQUIRED.rglob(pattern))
    return sorted({str(path) for path in matches if path.is_file()})


audit_rows = []
for item in required_sources:
    matches = uploaded_matches(item["upload_patterns"])
    if matches and item["source_id"] == "rural_wages":
        status = "partial_confirmatory_coverage"
    elif matches and item["source_id"] == "hces_microdata":
        status = "available_confirmatory_aggregate"
    else:
        status = "available_confirmatory_upload" if matches else item["status_without_upload"]
    audit_rows.append({
        "source_id": item["source_id"],
        "research_question": item["rq"],
        "required_variables": item["required_variables"],
        "status": status,
        "critical_for_final_hypothesis": item["critical"],
        "current_evidence": item["current_evidence"],
        "uploaded_files": " | ".join(matches),
        "official_url": item["official_url"],
        "target_filename": item["target_filename"],
    })

data_gap_audit = pd.DataFrame(audit_rows)
data_gap_audit.to_csv(REPORT_OUTPUT / "01_required_data_audit.csv", index=False)

# Header-only templates make the expected schema explicit without inventing data.
templates = {
    "agmarknet_enam_arrivals_template.csv": [
        "date", "state", "district", "market", "commodity",
        "arrival_quantity", "arrival_unit", "source_url",
    ],
    "des_state_crop_apy_template.csv": [
        "state", "crop", "season", "year", "area_hectare",
        "production_tonne", "yield_kg_per_hectare", "source_url",
    ],
    "labour_bureau_rural_wages_template.csv": [
        "year", "month", "state", "occupation_group", "occupation_item",
        "male_wage_rs_per_day", "female_wage_rs_per_day", "source_url",
    ],
    "hces_household_analysis_template.csv": [
        "household_id", "state", "sector", "expenditure_fractile",
        "household_size", "food_expenditure_monthly", "mpce",
        "survey_weight", "source_file",
    ],
    "mospi_cpi_cfpi_template.csv": [
        "date", "sector", "state", "group", "subgroup",
        "index_base", "index_value", "source_url",
    ],
}
for filename, columns in templates.items():
    pd.DataFrame(columns=columns).to_csv(INPUT_TEMPLATES / filename, index=False)

# This matrix links every original hypothesis to at least four measurable variables.
hypothesis_variable_coverage = pd.DataFrame([
    ["RQ1", "future_price_change_h1_h3", "lagged_price", "available", "confirmatory after Notebook 02"],
    ["RQ1", "future_price_change_h1_h3", "market_arrivals", "missing", "download required"],
    ["RQ1", "future_price_change_h1_h3", "rainfall_temperature_anomaly", "available_proxy", "NASA POWER interim"],
    ["RQ1", "future_price_change_h1_h3", "production_yield", "available", "official DES state-crop APY"],
    ["RQ1", "future_price_change_h1_h3", "fuel_logistics_cost", "available", "World Bank crude oil"],
    ["RQ1", "future_price_change_h1_h3", "wage_macro_growth", "partial", "Labour Bureau FY2025-26 only"],
    ["RQ2", "forecast_loss", "model_family", "derived_later", "Notebook 05"],
    ["RQ2", "forecast_loss", "forecast_horizon", "derived_later", "1, 2, 3 months"],
    ["RQ2", "forecast_loss", "commodity_region", "available", "state panel keys"],
    ["RQ2", "forecast_loss", "forecast_origin", "derived_later", "rolling validation"],
    ["RQ3", "hfasi_validity", "forecast_food_cost_growth", "derived_later", "Notebook 05/08"],
    ["RQ3", "hfasi_validity", "food_expenditure_share", "available_aggregate", "HCES-derived state/sector/decile"],
    ["RQ3", "hfasi_validity", "wage_income_growth", "partial", "Labour Bureau FY2025-26 plus PLFS"],
    ["RQ3", "hfasi_validity", "household_segment_fractile", "available_aggregate", "HCES expenditure deciles"],
    ["RQ4", "decision_loss_gain", "shock_probability", "derived_later", "Notebook 06"],
    ["RQ4", "decision_loss_gain", "hfasi", "derived_later", "Notebook 08"],
    ["RQ4", "decision_loss_gain", "warning_strategy", "derived_later", "Notebook 09"],
    ["RQ4", "decision_loss_gain", "scenario_severity_lead_time", "derived_later", "Notebook 09"],
], columns=["research_question", "outcome", "variable", "status", "action"])
hypothesis_variable_coverage.to_csv(
    REPORT_OUTPUT / "01_hypothesis_variable_coverage.csv", index=False
)

critical_gap_mask = (
    data_gap_audit["critical_for_final_hypothesis"]
    & ~data_gap_audit["status"].str.startswith("available_confirmatory")
)
critical_data_gaps = data_gap_audit.loc[
    critical_gap_mask, ["source_id", "status", "official_url", "target_filename"]
].copy()

print("ORIGINAL-SYNOPSIS DATA READINESS")
print(data_gap_audit[
    ["source_id", "research_question", "status", "critical_for_final_hypothesis"]
].to_string(index=False))

if len(critical_data_gaps):
    print("\nACTION REQUIRED FOR FINAL CONFIRMATORY TESTS")
    print(critical_data_gaps.to_string(index=False))
    print(f"\nUpload downloaded files to: {EXTERNAL_REQUIRED}")
else:
    print("\nAll critical confirmatory source classes are present.")
"""


SUMMARY_CODE = r"""
summary = {
    "notebook": "01_data_acquisition_synopsis_aligned",
    "status": (
        "completed_with_confirmatory_data_gaps"
        if len(critical_data_gaps)
        else "completed_confirmatory_sources_present"
    ),
    "original_research_questions_preserved": True,
    "hypothesis_variable_groups_audited": 4,
    "required_source_classes_audited": int(len(data_gap_audit)),
    "critical_confirmatory_data_gaps": int(len(critical_data_gaps)),
    "critical_gap_sources": critical_data_gaps["source_id"].tolist(),
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
    "source_rows_is_not_market_arrivals": True,
    "hces_aggregate_is_not_household_validation": True,
    "partial_2026_warning": True,
    "climate_spatial_limitation": (
        "Representative administrative-capital points; not area-weighted state averages."
    ),
    "external_required_folder": str(EXTERNAL_REQUIRED),
    "output_root": str(OUTPUT_ROOT),
}

summary_path = REPORT_OUTPUT / "01_execution_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=" * 78)
print("NOTEBOOK 01 ALIGNED EXECUTION SUMMARY - PLEASE SHARE THIS OUTPUT")
print("=" * 78)
print(json.dumps(summary, indent=2))
print("\nGenerated files:")
generated_paths = [
    PROCESSED / "source_inventory.csv",
    monthly_price_path,
    state_panel_path,
    national_path,
    REPORT_OUTPUT / "01_commodity_coverage.csv",
    REPORT_OUTPUT / "01_required_data_audit.csv",
    REPORT_OUTPUT / "01_hypothesis_variable_coverage.csv",
    summary_path,
]
for path in generated_paths:
    print(f"- {path} ({path.stat().st_size / 1_000_000:.3f} MB)")
print(f"- {INPUT_TEMPLATES} ({len(templates)} schema templates)")
"""


CONCLUSION_MARKDOWN = r"""
## Notebook 01 conclusion and hand-off

Notebook 01 has created the existing modeling panels and audited the additional
source classes required by the original synopsis. A status of
`completed_with_confirmatory_data_gaps` is not a notebook failure: it means the
current data support interim analysis while one or more final confirmatory
variables still require an official download.

Before Notebook 02:

1. share the printed execution-summary block;
2. if possible, download the files listed under **ACTION REQUIRED**;
3. place them in the printed `external_required` Google Drive folder; and
4. do not rename reporting-coverage fields as market arrivals.
"""


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    # Clear previous execution state so Colab users run the aligned version cleanly.
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    cells[0] = markdown(
        """
# QM640 Food Price Affordability AI
## Notebook 01 - Synopsis-Aligned Data Acquisition, Audit, and Integration

**Student:** Piyush Soni  
**Environment:** Google Colab  
**Controlling design:** Original approved synopsis  
**Study period:** Maximum official history through 2026 YTD, subject to source availability

This notebook preserves the original RQ1-RQ4 variable requirements while
reproducing the tested acquisition pipeline. It creates current modeling panels,
audits every confirmatory data source, generates input-schema templates, and
prints missing-data actions immediately.
"""
    )

    # Insert the new audit before the existing final summary.
    summary_index = 16
    cells[summary_index:summary_index] = [
        markdown(ALIGNMENT_MARKDOWN),
        code(ALIGNMENT_CODE),
    ]
    cells[summary_index + 2] = code(SUMMARY_CODE)
    cells[summary_index + 3] = markdown(CONCLUSION_MARKDOWN)

    notebook["metadata"].setdefault("colab", {})
    notebook["metadata"]["colab"]["name"] = OUTPUT.name
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()
