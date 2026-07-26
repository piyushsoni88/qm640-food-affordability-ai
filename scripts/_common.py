
from __future__ import annotations
import hashlib, json, logging, os, re, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)
        if log_file:
            ensure_dir(log_file.parent)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger

def build_session(retries: int = 5, backoff: float = 2.0) -> requests.Session:
    retry = Retry(
        total=retries, connect=retries, read=retries, status=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "QM640-Food-Affordability-Research/1.0",
        "Accept": "*/*",
    })
    return s

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_name(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE)
    return value.strip("._") or "file"

def save_bytes(content: bytes, path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_bytes(content)
    return path

def append_manifest(row: dict, manifest_path: Path | None = None) -> None:
    import csv
    path = manifest_path or PROJECT_ROOT / "data" / "source_manifest.csv"
    ensure_dir(path.parent)
    exists = path.exists()
    columns = [
        "source", "url", "local_path", "downloaded_at_utc",
        "size_bytes", "sha256", "status", "notes"
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in columns})

def record_download(source: str, url: str, path: Path, status: str = "success", notes: str = ""):
    append_manifest({
        "source": source,
        "url": url,
        "local_path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else "",
        "status": status,
        "notes": notes,
    })

def write_json(obj, path: Path):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
