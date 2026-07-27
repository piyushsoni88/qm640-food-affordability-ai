from __future__ import annotations

import argparse
import calendar
import hashlib
import importlib.util
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nasa_power_all_india"
CURATED = ROOT / "data" / "curated"
METADATA = ROOT / "data" / "metadata"

NASA_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMETERS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M"]

# Representative administrative-capital points. These are reproducible regional
# indicators, not area-weighted state averages.
REGIONS = {
    "Andhra Pradesh": ("State", "Amaravati", 16.5062, 80.6480),
    "Arunachal Pradesh": ("State", "Itanagar", 27.0844, 93.6053),
    "Assam": ("State", "Dispur", 26.1433, 91.7898),
    "Bihar": ("State", "Patna", 25.5941, 85.1376),
    "Chhattisgarh": ("State", "Raipur", 21.2514, 81.6296),
    "Goa": ("State", "Panaji", 15.4909, 73.8278),
    "Gujarat": ("State", "Gandhinagar", 23.2156, 72.6369),
    "Haryana": ("State", "Chandigarh", 30.7333, 76.7794),
    "Himachal Pradesh": ("State", "Shimla", 31.1048, 77.1734),
    "Jharkhand": ("State", "Ranchi", 23.3441, 85.3096),
    "Karnataka": ("State", "Bengaluru", 12.9716, 77.5946),
    "Kerala": ("State", "Thiruvananthapuram", 8.5241, 76.9366),
    "Madhya Pradesh": ("State", "Bhopal", 23.2599, 77.4126),
    "Maharashtra": ("State", "Mumbai", 19.0760, 72.8777),
    "Manipur": ("State", "Imphal", 24.8170, 93.9368),
    "Meghalaya": ("State", "Shillong", 25.5788, 91.8933),
    "Mizoram": ("State", "Aizawl", 23.7271, 92.7176),
    "Nagaland": ("State", "Kohima", 25.6751, 94.1086),
    "Odisha": ("State", "Bhubaneswar", 20.2961, 85.8245),
    "Punjab": ("State", "Chandigarh", 30.7333, 76.7794),
    "Rajasthan": ("State", "Jaipur", 26.9124, 75.7873),
    "Sikkim": ("State", "Gangtok", 27.3389, 88.6065),
    "Tamil Nadu": ("State", "Chennai", 13.0827, 80.2707),
    "Telangana": ("State", "Hyderabad", 17.3850, 78.4867),
    "Tripura": ("State", "Agartala", 23.8315, 91.2868),
    "Uttar Pradesh": ("State", "Lucknow", 26.8467, 80.9462),
    "Uttarakhand": ("State", "Dehradun", 30.3165, 78.0322),
    "West Bengal": ("State", "Kolkata", 22.5726, 88.3639),
    "Andaman and Nicobar Islands": (
        "Union Territory",
        "Port Blair",
        11.6234,
        92.7265,
    ),
    "Chandigarh": ("Union Territory", "Chandigarh", 30.7333, 76.7794),
    "Dadra and Nagar Haveli and Daman and Diu": (
        "Union Territory",
        "Daman",
        20.3974,
        72.8328,
    ),
    "Delhi": ("Union Territory", "New Delhi", 28.6139, 77.2090),
    "Jammu and Kashmir": ("Union Territory", "Srinagar", 34.0837, 74.7973),
    "Ladakh": ("Union Territory", "Leh", 34.1526, 77.5771),
    "Lakshadweep": ("Union Territory", "Kavaratti", 10.5593, 72.6358),
    "Puducherry": ("Union Territory", "Puducherry", 11.9416, 79.8083),
}


def slug(value: str) -> str:
    return (
        value.lower()
        .replace(" and ", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_with_retries() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "QM640-Food-Affordability-Research/2.0"
    return session


def year_chunks(start_year: int, end_year: int, years_per_chunk: int = 10):
    chunk_start = start_year
    while chunk_start <= end_year:
        chunk_end = min(chunk_start + years_per_chunk - 1, end_year)
        yield chunk_start, chunk_end
        chunk_start = chunk_end + 1


def fetch_chunk(
    session: requests.Session,
    region: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    force: bool,
) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / (
        f"{slug(region)}_{start_date.strftime('%Y%m%d')}_"
        f"{end_date.strftime('%Y%m%d')}.json"
    )
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    params = {
        "parameters": ",".join(NASA_PARAMETERS),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "format": "JSON",
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
    }
    response = session.get(NASA_DAILY_URL, params=params, timeout=240)
    response.raise_for_status()
    payload = response.json()
    if "properties" not in payload or "parameter" not in payload["properties"]:
        raise ValueError(f"Unexpected NASA response for {region}: {payload}")
    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(0.35)
    return payload


def payload_rows(
    payload: dict,
    region: str,
    admin_type: str,
    reference_location: str,
    latitude: float,
    longitude: float,
) -> list[dict]:
    parameter_data = payload["properties"]["parameter"]
    periods = sorted(
        {
            period
            for values in parameter_data.values()
            for period in values
            if len(period) == 8 and period.isdigit()
        }
    )
    rows = []
    for period in periods:
        row = {
            "region": region,
            "admin_type": admin_type,
            "reference_location": reference_location,
            "spatial_representation": "representative administrative-capital point",
            "latitude": latitude,
            "longitude": longitude,
            "date": pd.to_datetime(period, format="%Y%m%d"),
        }
        for parameter in NASA_PARAMETERS:
            row[parameter.lower()] = parameter_data.get(parameter, {}).get(period)
        rows.append(row)
    return rows


def collect_daily(
    start_year: int, end_date: date, force: bool, years_per_chunk: int
) -> pd.DataFrame:
    session = session_with_retries()
    rows: list[dict] = []
    for index, (region, details) in enumerate(REGIONS.items(), 1):
        admin_type, reference_location, latitude, longitude = details
        print(f"[{index:02d}/{len(REGIONS)}] {region}", flush=True)
        for chunk_start, chunk_end in year_chunks(
            start_year, end_date.year, years_per_chunk
        ):
            start_date = date(chunk_start, 1, 1)
            final_date = (
                end_date if chunk_end == end_date.year else date(chunk_end, 12, 31)
            )
            payload = fetch_chunk(
                session,
                region,
                latitude,
                longitude,
                start_date,
                final_date,
                force,
            )
            rows.extend(
                payload_rows(
                    payload,
                    region,
                    admin_type,
                    reference_location,
                    latitude,
                    longitude,
                )
            )
    result = pd.DataFrame(rows)
    value_columns = [parameter.lower() for parameter in NASA_PARAMETERS]
    for column in value_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").replace(
            -999, np.nan
        )
    result = result.drop_duplicates(["region", "date"], keep="last")
    return result.sort_values(["region", "date"]).reset_index(drop=True)


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.copy()
    data["month"] = data["date"].dt.to_period("M").dt.to_timestamp()
    keys = [
        "region",
        "admin_type",
        "reference_location",
        "spatial_representation",
        "latitude",
        "longitude",
        "month",
    ]
    monthly = (
        data.groupby(keys, as_index=False, dropna=False)
        .agg(
            rainfall_mm=("prectotcorr", "sum"),
            temperature_c=("t2m", "mean"),
            temperature_max_c=("t2m_max", "mean"),
            temperature_min_c=("t2m_min", "mean"),
            relative_humidity_pct=("rh2m", "mean"),
            observed_days=("date", "nunique"),
        )
        .rename(columns={"month": "date"})
    )
    monthly["expected_days"] = monthly["date"].map(
        lambda value: calendar.monthrange(value.year, value.month)[1]
    )
    monthly["is_partial_month"] = (
        monthly["observed_days"] < monthly["expected_days"]
    )
    monthly["data_status"] = np.where(
        monthly["is_partial_month"], "provisional_or_incomplete", "complete_month"
    )
    return monthly.sort_values(["region", "date"]).reset_index(drop=True)


def load_builder_module():
    path = ROOT / "scripts" / "09_build_interim_research_dataset.py"
    spec = importlib.util.spec_from_file_location("interim_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_expanded_panel(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    builder = load_builder_module()
    cpi = pd.read_csv(CURATED / "faostat_india_consumer_price_indices.csv.gz")
    production = pd.read_csv(CURATED / "faostat_india_crop_production.csv.gz")
    prices = pd.read_csv(CURATED / "faostat_india_producer_prices.csv.gz")
    pink = pd.read_csv(
        CURATED / "world_bank_pink_sheet_food_energy_monthly.csv.gz",
        parse_dates=["date"],
    )
    food_cpi = builder.select_food_cpi(cpi)
    annual = builder.annual_india_features(prices, production)

    date_index = pd.DataFrame(
        {
            "date": pd.date_range(
                monthly["date"].min(), monthly["date"].max(), freq="MS"
            )
        }
    )
    climate_columns = [
        "rainfall_mm",
        "temperature_c",
        "temperature_max_c",
        "temperature_min_c",
        "relative_humidity_pct",
    ]
    national_climate = (
        monthly.groupby("date", as_index=False)[climate_columns].mean()
    )
    national = (
        date_index.merge(food_cpi, on="date", how="left")
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

    commodities = pd.DataFrame({"commodity": list(builder.COMMODITY_PATTERNS)})
    panel = (
        monthly.merge(commodities, how="cross")
        .merge(
            national.drop(columns=climate_columns),
            on="date",
            how="left",
        )
    )
    panel["Year"] = panel["date"].dt.year
    panel = panel.merge(annual, on=["Year", "commodity"], how="left")
    panel["price_geography"] = "India national proxy"
    panel["climate_geography"] = "state/UT representative point"
    return national, panel


def write_outputs(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    national: pd.DataFrame,
    panel: pd.DataFrame,
    requested_end: date,
) -> None:
    CURATED.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    daily_path = (
        CURATED / "nasa_power_india_all_states_uts_daily_2000_2026_ytd.csv.gz"
    )
    monthly_path = (
        CURATED / "nasa_power_india_all_states_uts_monthly_2000_2026_ytd.csv.gz"
    )
    national_path = (
        CURATED / "india_food_affordability_national_monthly_2000_2026_ytd.csv.gz"
    )
    panel_path = (
        CURATED / "india_food_affordability_panel_36x8_2000_2026_ytd.csv.gz"
    )
    daily.to_csv(daily_path, index=False, compression="gzip")
    monthly.to_csv(monthly_path, index=False, compression="gzip")
    national.to_csv(national_path, index=False, compression="gzip")
    panel.to_csv(panel_path, index=False, compression="gzip")

    raw_files = list(RAW.glob("*.json"))
    quality = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_end_date": requested_end.isoformat(),
        "actual_date_min": str(daily["date"].min().date()),
        "actual_date_max": str(daily["date"].max().date()),
        "states": int(
            daily.loc[daily["admin_type"].eq("State"), "region"].nunique()
        ),
        "union_territories": int(
            daily.loc[
                daily["admin_type"].eq("Union Territory"), "region"
            ].nunique()
        ),
        "daily_rows": len(daily),
        "monthly_rows": len(monthly),
        "panel_rows": len(panel),
        "commodities": int(panel["commodity"].nunique()),
        "daily_duplicate_keys": int(
            daily.duplicated(["region", "date"]).sum()
        ),
        "monthly_duplicate_keys": int(
            monthly.duplicated(["region", "date"]).sum()
        ),
        "panel_duplicate_keys": int(
            panel.duplicated(["region", "date", "commodity"]).sum()
        ),
        "partial_month_rows": int(monthly["is_partial_month"].sum()),
        "raw_files": len(raw_files),
        "raw_bytes": sum(path.stat().st_size for path in raw_files),
        "curated_outputs": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in [daily_path, monthly_path, national_path, panel_path]
        },
        "method_note": (
            "NASA POWER daily observations are tied to one reproducible "
            "administrative-capital point per state/UT. They are not "
            "area-weighted state averages. 2026 is year-to-date and provisional."
        ),
        "national_proxy_note": (
            "FAOSTAT and World Bank price/production fields in the integration "
            "panel are India-level covariates repeated across regions, not "
            "state mandi observations."
        ),
    }
    (METADATA / "all_india_2000_2026_data_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    print(json.dumps(quality, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=4),
        help="ISO date; defaults to four days before today for NASA latency.",
    )
    parser.add_argument("--years-per-chunk", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.start_year > args.end_date.year:
        parser.error("start-year must not be after end-date")
    daily = collect_daily(
        args.start_year, args.end_date, args.force, args.years_per_chunk
    )
    monthly = aggregate_monthly(daily)
    national, panel = build_expanded_panel(monthly)
    write_outputs(daily, monthly, national, panel, args.end_date)


if __name__ == "__main__":
    main()
