from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CURATED = ROOT / "data" / "curated"
METADATA = ROOT / "data" / "metadata"

FAOSTAT_FILES = {
    "consumer_price_indices": (
        RAW / "faostat" / "ConsumerPriceIndices_E_All_Data_Normalized.zip",
        "https://bulks-faostat.fao.org/production/ConsumerPriceIndices_E_All_Data_(Normalized).zip",
    ),
    "producer_prices": (
        RAW / "faostat" / "Prices_E_All_Data_Normalized.zip",
        "https://bulks-faostat.fao.org/production/Prices_E_All_Data_(Normalized).zip",
    ),
    "crop_production": (
        RAW / "faostat" / "Production_Crops_Livestock_E_All_Data_Normalized.zip",
        "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip",
    ),
}

REGIONS = {
    "Delhi": (28.6139, 77.2090),
    "Maharashtra": (19.0760, 72.8777),
    "Karnataka": (12.9716, 77.5946),
    "West Bengal": (22.5726, 88.3639),
    "Tamil Nadu": (13.0827, 80.2707),
    "Telangana": (17.3850, 78.4867),
    "Gujarat": (23.0225, 72.5714),
    "Rajasthan": (26.9124, 75.7873),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Bihar": (25.5941, 85.1376),
    "Madhya Pradesh": (23.2599, 77.4126),
    "Punjab": (30.7333, 76.7794),
    "Odisha": (20.2961, 85.8245),
    "Assam": (26.1445, 91.7362),
    "Kerala": (8.5241, 76.9366),
}

NASA_PARAMETERS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M"]

COMMODITY_PATTERNS = {
    "rice": r"\brice\b",
    "wheat": r"\bwheat\b",
    "onion": r"\bonion",
    "potato": r"\bpotato",
    "tomato": r"\btomato",
    "pulses": r"chick peas|lentils|pigeon peas|beans, dry|peas, dry|pulses",
    "edible_oil": r"oil,|oil$|oilseed|groundnuts|soybeans|rapeseed|sunflower",
    "sugar": r"sugar cane|sugar beet|sugar",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_member(archive: zipfile.ZipFile) -> str:
    members = [
        name
        for name in archive.namelist()
        if "All_Data_(Normalized).csv" in name
    ]
    if len(members) != 1:
        raise ValueError(f"Expected one normalized data member, found {members}")
    return members[0]


def extract_faostat_india(zip_path: Path, output_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        member = normalized_member(archive)
        with archive.open(member) as stream:
            for chunk in pd.read_csv(
                stream,
                chunksize=250_000,
                encoding="utf-8-sig",
                low_memory=False,
            ):
                india = chunk.loc[chunk["Area"].eq("India")].copy()
                if not india.empty:
                    frames.append(india)
    result = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, compression="gzip")
    return result


def fetch_nasa_monthly(start_year: int, end_year: int) -> pd.DataFrame:
    raw_dir = RAW / "nasa_power"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    session = requests.Session()
    session.headers["User-Agent"] = "QM640-Food-Affordability-Research/1.0"
    endpoint = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    for region, (latitude, longitude) in REGIONS.items():
        params = {
            "parameters": ",".join(NASA_PARAMETERS),
            "community": "AG",
            "longitude": longitude,
            "latitude": latitude,
            "format": "JSON",
            "start": start_year,
            "end": end_year,
        }
        response = session.get(endpoint, params=params, timeout=180)
        response.raise_for_status()
        payload = response.json()
        (raw_dir / f"{region.lower().replace(' ', '_')}_{start_year}_{end_year}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        parameter_data = payload["properties"]["parameter"]
        all_periods = sorted(
            {
                period
                for values in parameter_data.values()
                for period in values
                if (
                    len(period) == 6
                    and period.isdigit()
                    and 1 <= int(period[4:]) <= 12
                )
            }
        )
        for period in all_periods:
            row = {
                "region": region,
                "latitude": latitude,
                "longitude": longitude,
                "date": f"{period[:4]}-{period[4:]}-01",
            }
            for parameter in NASA_PARAMETERS:
                row[parameter.lower()] = parameter_data.get(parameter, {}).get(period)
            rows.append(row)
    result = pd.DataFrame(rows)
    for column in [parameter.lower() for parameter in NASA_PARAMETERS]:
        result[column] = pd.to_numeric(result[column], errors="coerce").replace(-999, np.nan)
    return result.sort_values(["region", "date"]).reset_index(drop=True)


def parse_pink_sheet(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Monthly Prices", header=4)
    raw.columns = [column.strip() if isinstance(column, str) else column for column in raw.columns]
    date_column = raw.columns[0]
    result = raw.rename(columns={date_column: "period"}).copy()
    result["date"] = pd.to_datetime(
        result["period"].astype(str).str.replace("M", "-", regex=False) + "-01",
        errors="coerce",
    )
    result = result.loc[result["date"].notna()].copy()
    selected = {
        "Rice, Thai 5%": "world_rice_usd_per_mt",
        "Wheat, US SRW": "world_wheat_usd_per_mt",
        "Palm oil": "world_palm_oil_usd_per_mt",
        "Soybean oil": "world_soybean_oil_usd_per_mt",
        "Sugar, world": "world_sugar_usd_per_kg",
        "Crude oil, average": "world_crude_oil_usd_per_bbl",
    }
    keep = ["date"] + [column for column in selected if column in result]
    result = result[keep].rename(columns=selected)
    for column in result.columns.drop("date"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values("date").reset_index(drop=True)


def select_food_cpi(cpi: pd.DataFrame) -> pd.DataFrame:
    result = cpi.loc[
        cpi["Item"].str.contains("Food Indices", case=False, na=False)
        & cpi["Months"].ne("Annual value")
    ].copy()
    month_number = {
        month: index
        for index, month in enumerate(
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            1,
        )
    }
    result["month"] = result["Months"].map(month_number)
    result["date"] = pd.to_datetime(
        {"year": result["Year"], "month": result["month"], "day": 1},
        errors="coerce",
    )
    result["food_cpi_2015_100"] = pd.to_numeric(result["Value"], errors="coerce")
    return (
        result.loc[result["date"].notna(), ["date", "food_cpi_2015_100", "Flag"]]
        .drop_duplicates("date", keep="last")
        .sort_values("date")
    )


def map_commodity(item: pd.Series) -> pd.Series:
    output = pd.Series(pd.NA, index=item.index, dtype="string")
    for commodity, pattern in COMMODITY_PATTERNS.items():
        output = output.mask(
            output.isna() & item.str.contains(pattern, case=False, na=False, regex=True),
            commodity,
        )
    return output


def annual_india_features(
    producer_prices: pd.DataFrame, production: pd.DataFrame
) -> pd.DataFrame:
    prices = producer_prices.copy()
    prices["commodity"] = map_commodity(prices["Item"])
    prices["Value"] = pd.to_numeric(prices["Value"], errors="coerce")
    prices = prices.loc[
        prices["commodity"].notna()
        & prices["Element"].str.contains("Producer Price", case=False, na=False)
        & prices["Unit"].eq("LCU")
    ]
    price_annual = (
        prices.groupby(["Year", "commodity"], as_index=False)["Value"]
        .median()
        .rename(columns={"Value": "producer_price_lcu_per_tonne"})
    )

    prod = production.copy()
    prod["commodity"] = map_commodity(prod["Item"])
    prod["Value"] = pd.to_numeric(prod["Value"], errors="coerce")
    prod = prod.loc[
        prod["commodity"].notna()
        & prod["Element"].eq("Production")
        & prod["Unit"].isin(["t", "tonnes"])
    ]
    production_annual = (
        prod.groupby(["Year", "commodity"], as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": "production_tonnes"})
    )
    return price_annual.merge(production_annual, on=["Year", "commodity"], how="outer")


def build_monthly_panel(
    climate: pd.DataFrame,
    food_cpi: pd.DataFrame,
    pink: pd.DataFrame,
    annual: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DataFrame(
        {"date": pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")}
    )
    national_climate = (
        climate.assign(date=pd.to_datetime(climate["date"]))
        .groupby("date", as_index=False)[[p.lower() for p in NASA_PARAMETERS]]
        .mean()
        .rename(
            columns={
                "prectotcorr": "rainfall_mm",
                "t2m": "temperature_c",
                "t2m_max": "temperature_max_c",
                "t2m_min": "temperature_min_c",
                "rh2m": "relative_humidity_pct",
            }
        )
    )
    national = (
        dates.merge(food_cpi, on="date", how="left")
        .merge(pink, on="date", how="left")
        .merge(national_climate, on="date", how="left")
        .sort_values("date")
    )
    national["food_cpi_mom_pct"] = national["food_cpi_2015_100"].pct_change() * 100
    national["food_cpi_yoy_pct"] = national["food_cpi_2015_100"].pct_change(12) * 100
    for lag in [1, 2, 3, 6, 12]:
        national[f"food_cpi_lag_{lag}"] = national["food_cpi_2015_100"].shift(lag)
    national["month_sin"] = np.sin(2 * np.pi * national["date"].dt.month / 12)
    national["month_cos"] = np.cos(2 * np.pi * national["date"].dt.month / 12)

    commodities = pd.DataFrame({"commodity": list(COMMODITY_PATTERNS)})
    climate_monthly = climate.assign(date=pd.to_datetime(climate["date"]))
    panel = (
        climate_monthly.merge(commodities, how="cross")
        .merge(
            national.drop(
                columns=[
                    "rainfall_mm",
                    "temperature_c",
                    "temperature_max_c",
                    "temperature_min_c",
                    "relative_humidity_pct",
                ]
            ),
            on="date",
            how="left",
        )
    )
    panel["Year"] = panel["date"].dt.year
    panel = panel.merge(annual, on=["Year", "commodity"], how="left")
    panel = panel.rename(
        columns={
            "prectotcorr": "rainfall_mm",
            "t2m": "temperature_c",
            "t2m_max": "temperature_max_c",
            "t2m_min": "temperature_min_c",
            "rh2m": "relative_humidity_pct",
        }
    )
    return national, panel


def write_manifest(records: list[dict]) -> None:
    manifest_path = METADATA / "interim_data_manifest.csv"
    METADATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(manifest_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--skip-nasa", action="store_true")
    args = parser.parse_args()
    CURATED.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    extracted: dict[str, pd.DataFrame] = {}
    for name, (path, url) in FAOSTAT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run the documented bulk download first.")
        output = CURATED / f"faostat_india_{name}.csv.gz"
        extracted[name] = extract_faostat_india(path, output)
        manifest.append(
            {
                "source": f"FAOSTAT {name}",
                "url": url,
                "raw_path": str(path.relative_to(ROOT)),
                "curated_path": str(output.relative_to(ROOT)),
                "raw_size_bytes": path.stat().st_size,
                "raw_sha256": sha256(path),
                "curated_rows": len(extracted[name]),
                "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    climate_path = CURATED / "nasa_power_india_15_regions_monthly_2005_2025.csv.gz"
    if args.skip_nasa and climate_path.exists():
        climate = pd.read_csv(climate_path)
    else:
        climate = fetch_nasa_monthly(args.start_year, args.end_year)
        climate.to_csv(climate_path, index=False, compression="gzip")
    manifest.append(
        {
            "source": "NASA POWER monthly climate",
            "url": "https://power.larc.nasa.gov/api/temporal/monthly/point",
            "raw_path": "data/raw/nasa_power/",
            "curated_path": str(climate_path.relative_to(ROOT)),
            "raw_size_bytes": sum(
                p.stat().st_size for p in (RAW / "nasa_power").glob("*.json")
            ),
            "raw_sha256": "per-file; see source manifest",
            "curated_rows": len(climate),
            "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    pink_path = RAW / "world_bank" / "CMO-Historical-Data-Monthly.xlsx"
    pink = parse_pink_sheet(pink_path)
    pink_output = CURATED / "world_bank_pink_sheet_food_energy_monthly.csv.gz"
    pink.to_csv(pink_output, index=False, compression="gzip")
    manifest.append(
        {
            "source": "World Bank Pink Sheet",
            "url": "https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/related/CMO-Historical-Data-Monthly.xlsx",
            "raw_path": str(pink_path.relative_to(ROOT)),
            "curated_path": str(pink_output.relative_to(ROOT)),
            "raw_size_bytes": pink_path.stat().st_size,
            "raw_sha256": sha256(pink_path),
            "curated_rows": len(pink),
            "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    food_cpi = select_food_cpi(extracted["consumer_price_indices"])
    annual = annual_india_features(
        extracted["producer_prices"], extracted["crop_production"]
    )
    annual_output = CURATED / "india_commodity_annual_production_prices.csv.gz"
    annual.to_csv(annual_output, index=False, compression="gzip")

    national, panel = build_monthly_panel(
        climate,
        food_cpi,
        pink,
        annual,
        args.start_year,
        args.end_year,
    )
    national_output = CURATED / "india_food_affordability_national_monthly.csv.gz"
    panel_output = CURATED / "india_food_affordability_panel_15x8_2005_2025.csv.gz"
    national.to_csv(national_output, index=False, compression="gzip")
    panel.to_csv(panel_output, index=False, compression="gzip")

    quality = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "national_rows": len(national),
        "panel_rows": len(panel),
        "regions": int(panel["region"].nunique()),
        "commodities": int(panel["commodity"].nunique()),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "national_duplicate_rows": int(national.duplicated().sum()),
        "panel_duplicate_keys": int(
            panel.duplicated(["date", "region", "commodity"]).sum()
        ),
        "national_missing_pct": {
            column: round(float(national[column].isna().mean() * 100), 2)
            for column in national.columns
        },
        "panel_note": (
            "The panel combines observed region-specific NASA climate with national "
            "FAOSTAT CPI/producer-price/production and World Bank benchmark series. "
            "National series are explicitly proxies, not regional mandi prices."
        ),
    }
    (METADATA / "interim_data_quality_report.json").write_text(
        json.dumps(quality, indent=2),
        encoding="utf-8",
    )
    write_manifest(manifest)
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
