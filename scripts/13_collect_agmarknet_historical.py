from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "agmarknet" / "historical_8_commodities"
CURATED = ROOT / "data" / "curated"
METADATA = ROOT / "data" / "metadata"

RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"
URL = f"https://www.data.gov.in/backend/dataapi/v1/resource/{RESOURCE_ID}"

# Counts verified from the official resource on 2026-07-28. Fixing the totals
# makes this a reproducible snapshot even while the live resource continues
# to grow.
TARGETS = {
    "Rice": 1_867_046,
    "Wheat": 3_470_626,
    "Onion": 3_338_857,
    "Potato": 3_406_949,
    "Tomato": 2_968_850,
    "Bengal Gram(Gram)(Whole)": 1_534_906,
    "Arhar (Tur/Red Gram)(Whole)": 987_907,
    "Soyabean": 1_261_321,
}

THREAD_LOCAL = threading.local()


def slug(value: str) -> str:
    return (
        value.lower()
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is not None:
        return session
    retry = Retry(
        total=8,
        connect=5,
        read=8,
        status=8,
        backoff_factor=2,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "QM640-Food-Affordability-Research/3.0"
    THREAD_LOCAL.session = session
    return session


def tasks(page_size: int):
    for commodity, total in TARGETS.items():
        directory = RAW / f"commodity={slug(commodity)}"
        for offset in range(0, total, page_size):
            yield commodity, offset, min(page_size, total - offset), directory


def download_page(
    api_key: str,
    commodity: str,
    offset: int,
    expected_rows: int,
    directory: Path,
    page_size: int,
) -> tuple[Path, int, bool]:
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"part_{offset:09d}.csv.gz"
    if output.exists() and output.stat().st_size > 100:
        return output, expected_rows, True

    params = {
        "api-key": api_key,
        "format": "csv",
        "offset": offset,
        "limit": page_size,
        "filters[Commodity]": commodity,
    }
    response = get_session().get(URL, params=params, timeout=(30, 180))
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"Arrival_Date,Commodity,"):
        raise ValueError(
            f"Unexpected response for {commodity} offset {offset}: "
            f"{content[:120]!r}"
        )
    rows = max(content.count(b"\n") - 1, 0)
    if rows != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for {commodity} offset {offset}, "
            f"received {rows}"
        )
    temporary = output.with_suffix(output.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=6) as stream:
        stream.write(content)
    temporary.replace(output)
    return output, rows, False


def download_all(api_key: str, page_size: int, workers: int) -> dict:
    all_tasks = list(tasks(page_size))
    completed = 0
    reused = 0
    rows = 0
    started = time.monotonic()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                download_page,
                api_key,
                commodity,
                offset,
                expected_rows,
                directory,
                page_size,
            ): (commodity, offset)
            for commodity, offset, expected_rows, directory in all_tasks
        }
        for future in as_completed(future_map):
            commodity, offset = future_map[future]
            try:
                _, page_rows, was_reused = future.result()
                completed += 1
                rows += page_rows
                reused += int(was_reused)
                if completed % 25 == 0 or completed == len(all_tasks):
                    elapsed = max(time.monotonic() - started, 0.01)
                    print(
                        f"pages={completed}/{len(all_tasks)} "
                        f"rows={rows:,} reused={reused} "
                        f"pages_per_minute={completed / elapsed * 60:.1f}",
                        flush=True,
                    )
            except Exception as exc:
                failures.append(f"{commodity} offset={offset}: {exc}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} pages failed. Rerun to resume.\n"
            + "\n".join(failures[:20])
        )
    return {
        "pages": completed,
        "rows": rows,
        "reused_pages": reused,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }


def aggregate_daily() -> tuple[pd.DataFrame, dict]:
    partials: list[pd.DataFrame] = []
    raw_rows = 0
    files = sorted(RAW.rglob("part_*.csv.gz"))
    for index, path in enumerate(files, 1):
        frame = pd.read_csv(path, low_memory=False)
        raw_rows += len(frame)
        frame["date"] = pd.to_datetime(
            frame["Arrival_Date"], dayfirst=True, errors="coerce"
        )
        frame = frame.loc[
            frame["date"].dt.year.between(2000, 2026, inclusive="both")
        ].copy()
        for column in ["Min_Price", "Max_Price", "Modal_Price"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["modal_price_sum"] = frame["Modal_Price"].fillna(0)
        frame["price_observation"] = frame["Modal_Price"].notna().astype("int64")
        grouped = (
            frame.groupby(["date", "State", "Commodity"], as_index=False)
            .agg(
                minimum_price_rs_per_quintal=("Min_Price", "min"),
                maximum_price_rs_per_quintal=("Max_Price", "max"),
                modal_price_sum=("modal_price_sum", "sum"),
                price_observations=("price_observation", "sum"),
                market_count_partition=("Market", "nunique"),
                district_count_partition=("District", "nunique"),
                source_rows=("Arrival_Date", "size"),
            )
        )
        partials.append(grouped)
        if index % 100 == 0:
            print(f"aggregated_files={index}/{len(files)}", flush=True)

    combined = pd.concat(partials, ignore_index=True)
    result = (
        combined.groupby(["date", "State", "Commodity"], as_index=False)
        .agg(
            minimum_price_rs_per_quintal=(
                "minimum_price_rs_per_quintal",
                "min",
            ),
            maximum_price_rs_per_quintal=(
                "maximum_price_rs_per_quintal",
                "max",
            ),
            modal_price_sum=("modal_price_sum", "sum"),
            price_observations=("price_observations", "sum"),
            market_count_partition_sum=("market_count_partition", "sum"),
            district_count_partition_sum=("district_count_partition", "sum"),
            source_rows=("source_rows", "sum"),
        )
        .sort_values(["date", "State", "Commodity"])
    )
    result["mean_modal_price_rs_per_quintal"] = (
        result["modal_price_sum"] / result["price_observations"].replace(0, pd.NA)
    )
    result = result.drop(columns=["modal_price_sum"])
    stats = {
        "raw_files": len(files),
        "raw_rows": raw_rows,
        "aggregated_rows": len(result),
        "date_min": str(result["date"].min().date()),
        "date_max": str(result["date"].max().date()),
        "states_and_labels": int(result["State"].nunique()),
        "commodities": int(result["Commodity"].nunique()),
        "duplicate_keys": int(
            result.duplicated(["date", "State", "Commodity"]).sum()
        ),
    }
    return result, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    if not 1 <= args.page_size <= 100_000:
        parser.error("page-size must be between 1 and 100000")

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DATA_GOV_IN_API_KEY")
    if not api_key:
        raise SystemExit("DATA_GOV_IN_API_KEY missing from .env")

    download_stats = None
    if not args.aggregate_only:
        download_stats = download_all(api_key, args.page_size, args.workers)
    if args.download_only:
        print(json.dumps(download_stats, indent=2))
        return

    daily, aggregate_stats = aggregate_daily()
    CURATED.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    output = (
        CURATED
        / "agmarknet_historical_8_commodities_daily_state_2000_2026.csv.gz"
    )
    daily.to_csv(output, index=False, compression="gzip")
    raw_files = list(RAW.rglob("part_*.csv.gz"))
    if download_stats is None:
        download_stats = {
            "pages": len(raw_files),
            "rows": sum(TARGETS.values()),
            "reused_pages": len(raw_files),
            "elapsed_seconds": None,
            "mode": "aggregate-only from complete local archive",
        }
    quality = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "resource_id": RESOURCE_ID,
        "resource_url": URL,
        "snapshot_totals_verified_on": "2026-07-28",
        "target_source_rows": sum(TARGETS.values()),
        "download": download_stats,
        "aggregation": aggregate_stats,
        "raw_compressed_bytes": sum(path.stat().st_size for path in raw_files),
        "curated_bytes": output.stat().st_size,
        "curated_sha256": sha256(output),
        "method_note": (
            "Raw variety/market observations are retained locally in resumable "
            "gzip pages. The GitHub-ready table aggregates observations to "
            "date-state-commodity and preserves source-row counts. Market and "
            "district partition-count sums are diagnostic upper bounds because "
            "the same reporting unit can occur across pagination boundaries."
        ),
    }
    (METADATA / "agmarknet_historical_8_commodities_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
