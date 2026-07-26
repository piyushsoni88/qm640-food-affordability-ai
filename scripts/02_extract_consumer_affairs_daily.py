
from __future__ import annotations
import argparse, re, time
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from _common import PROJECT_ROOT, build_session, ensure_dir, get_logger, record_download, save_bytes

RETAIL = "https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx"
WHOLESALE = "https://fcainfoweb.nic.in/Reports/DB/DBprices_W.aspx"

def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def extract_tables(html: bytes, requested_date: date, kind: str, url: str) -> pd.DataFrame:
    tables = pd.read_html(html)
    frames = []
    for i, df in enumerate(tables):
        flat = " ".join(map(str, df.columns))
        body = " ".join(df.astype(str).head(8).values.ravel())
        if "Commodity" in flat + body or "Commodit" in flat + body:
            df.columns = [str(c).strip() for c in df.columns]
            df["requested_date"] = requested_date.isoformat()
            df["price_type"] = kind
            df["source_url"] = url
            df["source_table_index"] = i
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=date.today().isoformat())
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--sleep", type=float, default=0.6)
    args = p.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    session = build_session()
    log = get_logger("consumer_affairs", PROJECT_ROOT/"data/logs/02_consumer_affairs.log")
    raw_root = ensure_dir(PROJECT_ROOT/"data/raw/consumer_affairs")
    processed_root = ensure_dir(PROJECT_ROOT/"data/processed/consumer_affairs")
    all_frames = []
    for d in daterange(start, end):
        for kind, base in [("retail", RETAIL), ("wholesale", WHOLESALE)]:
            # The public ASP.NET page may ignore query-date parameters. We retain
            # requested_date and page_reported_date so this is auditable.
            url = f"{base}?date={d:%d/%m/%Y}"
            try:
                r = session.get(url, timeout=(20, 180))
                r.raise_for_status()
                path = raw_root/f"year={d:%Y}"/f"month={d:%m}"/f"{kind}_{d:%Y%m%d}.html"
                save_bytes(r.content, path)
                record_download("consumer_affairs", url, path)
                text = BeautifulSoup(r.content, "html.parser").get_text(" ", strip=True)
                m = re.search(r"Date:\s*(\d{2}/\d{2}/\d{4})", text)
                reported = m.group(1) if m else ""
                df = extract_tables(r.content, d, kind, url)
                if not df.empty:
                    df["page_reported_date"] = reported
                    all_frames.append(df)
                log.info("%s %s | tables=%s | page date=%s", d, kind, len(df), reported)
            except Exception as exc:
                log.exception("%s %s failed: %s", d, kind, exc)
            time.sleep(args.sleep)
    if all_frames:
        out = pd.concat(all_frames, ignore_index=True, sort=False)
        csv = processed_root/f"consumer_affairs_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        out.to_csv(csv, index=False)
        log.info("Saved %s rows to %s", len(out), csv)
    else:
        log.warning("No tables parsed. Inspect raw HTML and log; site structure may have changed.")

if __name__ == "__main__":
    main()
