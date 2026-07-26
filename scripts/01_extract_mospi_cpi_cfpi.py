
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from _common import PROJECT_ROOT, ensure_dir, get_logger, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    log = get_logger("mospi", PROJECT_ROOT/"data/logs/01_mospi_cpi_cfpi.log")
    out = ensure_dir(PROJECT_ROOT/"data/raw/mospi")
    # Prefer the project's validated downloader if available.
    existing = PROJECT_ROOT/"scripts"/"download_5_years.py"
    if existing.exists() and existing.resolve() != Path(__file__).resolve():
        cmd = [sys.executable, str(existing), "--sources", "mospi"]
        log.info("Using existing validated MoSPI orchestrator: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        return
    try:
        import mospi_esankhyiki  # type: ignore
    except Exception:
        log.error("mospi-esankhyiki is not installed or import name changed.")
        log.info("Install with: python -m pip install mospi-esankhyiki")
        raise SystemExit(2)
    # The official client has changed signatures across releases. Capture the
    # package and expose a reproducible discovery report instead of guessing.
    report = {
        "requested_start": args.start, "requested_end": args.end,
        "package": repr(mospi_esankhyiki),
        "next_action": "Use the repository's validated MoSPI extractor or adapt calls after list_datasets/get_indicators/get_metadata discovery.",
    }
    write_json(report, PROJECT_ROOT/"data/metadata/mospi/client_discovery.json")
    log.warning("No validated repository downloader found. Discovery report written; no fabricated data created.")

if __name__ == "__main__":
    main()
