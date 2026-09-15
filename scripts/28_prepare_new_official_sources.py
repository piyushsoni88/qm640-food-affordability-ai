"""Create compact, disclosure-safe analytical inputs from newly supplied sources."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "external_required"
OUT.mkdir(parents=True, exist_ok=True)


def clean_label(value: object) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", str(value)).strip()


def prepare_apy() -> Path:
    source = next((RAW / "des" / "apy").glob("*horizontal_crop_vertical_year*.xls"))
    table = pd.read_html(str(source))[0]
    id_columns = list(table.columns[:3])
    base = table[id_columns].copy()
    base.columns = ["state", "district", "year"]
    base["state"] = base["state"].map(clean_label)
    base["district"] = base["district"].map(clean_label)
    base["year"] = base["year"].astype(str).str.extract(r"(\d{4})")[0]
    rows: list[pd.DataFrame] = []
    crops = sorted({str(c[0]) for c in table.columns[3:]})
    for crop in crops:
        crop_columns = [c for c in table.columns[3:] if str(c[0]) == crop]
        whole = [c for c in crop_columns if str(c[1]).strip().lower() == "whole year"]
        selected = whole if whole else crop_columns
        metrics = {}
        for metric, output_name in [
            ("Area", "area_hectare"),
            ("Production", "production_tonne"),
        ]:
            columns = [c for c in selected if metric.lower() in str(c[2]).lower()]
            numeric = table[columns].apply(pd.to_numeric, errors="coerce")
            metrics[output_name] = numeric.sum(axis=1, min_count=1)
        part = base.copy()
        part["crop"] = clean_label(crop)
        part["area_hectare"] = metrics["area_hectare"]
        part["production_tonne"] = metrics["production_tonne"]
        part = part.dropna(subset=["year", "area_hectare", "production_tonne"], how="all")
        rows.append(part)
    district = pd.concat(rows, ignore_index=True)
    state = (
        district.groupby(["state", "crop", "year"], as_index=False)
        .agg(
            area_hectare=("area_hectare", "sum"),
            production_tonne=("production_tonne", "sum"),
            contributing_districts=("district", "nunique"),
        )
    )
    state["yield_kg_per_hectare"] = np.where(
        state["area_hectare"].gt(0),
        state["production_tonne"] * 1000 / state["area_hectare"],
        np.nan,
    )
    state["source_url"] = "https://data.desagri.gov.in/website/apy-index-report-web"
    target = OUT / "des_state_crop_area_production_yield.csv.gz"
    state.to_csv(target, index=False, compression="gzip")
    print(f"APY: {len(state):,} state-crop-year rows -> {target}")
    return target


def prepare_wages() -> Path:
    source = RAW / "labour_bureau" / "labour_bureau_rural_wages_2025_26_partial.csv"
    wages = pd.read_csv(source, na_values=["-", ""])
    wages = wages.rename(
        columns={
            "Year": "financial_year",
            "Month": "month_name",
            "State": "state",
            "Occupation": "occupation_group",
            "Item": "occupation_item",
            "Men": "male_wage_rs_per_day",
            "Women": "female_wage_rs_per_day",
        }
    )
    month_number = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    wages["month"] = wages["month_name"].map(month_number)
    # FY 2025-26: Apr-Dec belong to 2025; Jan-Mar belong to 2026.
    wages["year"] = np.where(wages["month"].ge(4), 2025, 2026)
    wages["date"] = pd.to_datetime(
        dict(year=wages["year"], month=wages["month"], day=1), errors="coerce"
    )
    for column in ["male_wage_rs_per_day", "female_wage_rs_per_day"]:
        wages[column] = pd.to_numeric(wages[column], errors="coerce")
    wages["rural_wage_rs_per_day"] = wages[
        ["male_wage_rs_per_day", "female_wage_rs_per_day"]
    ].mean(axis=1)
    wages["coverage_status"] = "partial_FY2025_26_portal_export"
    wages["source_url"] = "https://www.labourbureau.gov.in/rural-wages"
    keep = [
        "date", "financial_year", "state", "occupation_group", "occupation_item",
        "male_wage_rs_per_day", "female_wage_rs_per_day",
        "rural_wage_rs_per_day", "coverage_status", "source_url",
    ]
    target = OUT / "labour_bureau_rural_wages_monthly.csv.gz"
    wages[keep].to_csv(target, index=False, compression="gzip")
    print(f"Wages: {len(wages):,} partial rows -> {target}")
    return target


def prepare_hces() -> Path:
    folder = RAW / "hces" / "HCES_2023_24"
    food_file = folder / "LEVEL - 05 ( Sec 5  6).csv"
    total_file = folder / "LEVEL - 15 (Section 1_1, A2,B2  C2).csv"
    key = [
        "FSU_Serial_No", "Sector", "State", "Sample_SU_No",
        "Sample_Sub_Division_No", "Second_Stage_Stratum_No",
        "Sample_Household_No", "Questionnaire_No",
    ]
    total = pd.read_csv(total_file, low_memory=False)
    total = total.loc[
        total["Questionnaire_No"].eq("F")
        & total["SECTION"].astype(str).str.upper().eq("A2")
    ].copy()
    total["MONTHLY_CONSUMPTION_EXP"] = pd.to_numeric(
        total["MONTHLY_CONSUMPTION_EXP"], errors="coerce"
    )
    total["HOUSEHOLD_SIZE"] = pd.to_numeric(total["HOUSEHOLD_SIZE"], errors="coerce")
    total["MULTIPLIER"] = pd.to_numeric(total["MULTIPLIER"], errors="coerce")
    total = total.dropna(
        subset=["MONTHLY_CONSUMPTION_EXP", "HOUSEHOLD_SIZE", "MULTIPLIER"]
    )

    pieces = []
    usecols = key + ["Total_Consumption_Value"]
    for chunk in pd.read_csv(food_file, usecols=usecols, chunksize=750_000, low_memory=False):
        chunk["Total_Consumption_Value"] = pd.to_numeric(
            chunk["Total_Consumption_Value"], errors="coerce"
        ).fillna(0)
        pieces.append(
            chunk.groupby(key, dropna=False, as_index=False)["Total_Consumption_Value"].sum()
        )
    food = (
        pd.concat(pieces, ignore_index=True)
        .groupby(key, dropna=False, as_index=False)["Total_Consumption_Value"].sum()
        .rename(columns={"Total_Consumption_Value": "food_expenditure_monthly"})
    )
    households = total.merge(food, on=key, how="inner", validate="one_to_one")
    households["mpce"] = (
        households["MONTHLY_CONSUMPTION_EXP"] / households["HOUSEHOLD_SIZE"]
    )
    households["food_expenditure_share"] = (
        households["food_expenditure_monthly"]
        / households["MONTHLY_CONSUMPTION_EXP"]
    ).clip(0, 1)
    households = households.loc[
        households["mpce"].gt(0) & households["food_expenditure_share"].between(0, 1)
    ].copy()
    households["sector"] = households["Sector"].map({1: "Rural", 2: "Urban"})
    households["expenditure_decile"] = (
        households.groupby("sector")["mpce"]
        .transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) + 1)
        .astype(int)
    )
    state_codes = pd.read_excel(folder / "tabulation_state_code.xlsx")
    # HCES `State` uses the official NSS state code (`st`), not the compact
    # tabulation sequence (`ast`). Mapping the wrong code system can leave
    # Telangana (37) numeric or attach an incorrect label to another state.
    state_map = dict(zip(state_codes["st"], state_codes["stn"]))
    households["state"] = households["State"].map(state_map).fillna(
        households["State"].astype(str)
    )
    households["weight"] = households["MULTIPLIER"]

    def weighted_mean(group: pd.DataFrame, column: str) -> float:
        return float(np.average(group[column], weights=group["weight"]))

    aggregate_rows = []
    for (state_name, sector, decile), group in households.groupby(
        ["state", "sector", "expenditure_decile"], observed=True
    ):
        aggregate_rows.append(
            {
                "survey_year": "2023-24",
                "state": state_name,
                "sector": sector,
                "expenditure_decile": int(decile),
                "sample_households": int(len(group)),
                "weighted_mpce_rs": weighted_mean(group, "mpce"),
                "weighted_food_expenditure_monthly_rs": weighted_mean(
                    group, "food_expenditure_monthly"
                ),
                "weighted_food_expenditure_share": weighted_mean(
                    group, "food_expenditure_share"
                ),
                "sum_survey_weight": float(group["weight"].sum()),
                "source_url": "https://microdata.gov.in/NADA/index.php/catalog/224",
                "disclosure_status": "aggregated_state_sector_decile",
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    target = OUT / "hces_2023_24_segment_aggregates.csv.gz"
    aggregate.to_csv(target, index=False, compression="gzip")
    print(
        f"HCES: {len(households):,} valid household records -> "
        f"{len(aggregate):,} disclosure-safe aggregate rows -> {target}"
    )
    return target


if __name__ == "__main__":
    outputs = [prepare_apy(), prepare_wages(), prepare_hces()]
    for output in outputs:
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"Output validation failed: {output}")
