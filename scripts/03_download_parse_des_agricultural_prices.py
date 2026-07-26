
from __future__ import annotations
import argparse, re, time
from pathlib import Path
from urllib.parse import urljoin, urlparse
import pandas as pd
from bs4 import BeautifulSoup
from _common import PROJECT_ROOT, build_session, ensure_dir, get_logger, record_download, safe_name, save_bytes

INDEXES = [
 "https://desagri.gov.in/en/document-report/agricultural-prices-in-india-index-2/",
 "https://desagri.gov.in/document-report-category/agricultural-prices-in-india/",
]

def discover(session, year_start, year_end):
    links = {}
    for page in INDEXES:
        r = session.get(page, timeout=(20,180)); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(page, a["href"])
            label = " ".join(a.get_text(" ", strip=True).split())
            hay = f"{label} {href}"
            years = [int(y) for y in re.findall(r"\b(20\d{2})\b", hay)]
            if any(year_start <= y <= year_end for y in years) and (href.lower().endswith(".pdf") or "agricultural-prices" in href.lower()):
                links[href] = label
    return links

def parse_pdf(path: Path, out_dir: Path, log):
    out_dir = ensure_dir(out_dir)
    try:
        import pdfplumber
    except Exception:
        log.warning("pdfplumber unavailable; PDF retained without table extraction: %s", path.name)
        return 0
    rows = 0
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, 1):
            for table_no, table in enumerate(page.extract_tables() or [], 1):
                if not table or len(table) < 2:
                    continue
                width = max(len(r) for r in table)
                clean = [list(r)+[None]*(width-len(r)) for r in table]
                df = pd.DataFrame(clean)
                csv = out_dir/f"{path.stem}_page_{page_no:04d}_table_{table_no:02d}.csv"
                df.to_csv(csv, index=False, header=False)
                rows += len(df)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--start-year",type=int,default=2006); p.add_argument("--end-year",type=int,default=2025)
    args=p.parse_args()
    log=get_logger("des_prices",PROJECT_ROOT/"data/logs/03_des_agricultural_prices.log")
    s=build_session()
    raw=ensure_dir(PROJECT_ROOT/"data/raw/des/agricultural_prices")
    processed=ensure_dir(PROJECT_ROOT/"data/interim/des/agricultural_prices_tables")
    links=discover(s,args.start_year,args.end_year)
    log.info("Discovered %s candidate links",len(links))
    for url,label in links.items():
        try:
            r=s.get(url,timeout=(20,300)); r.raise_for_status()
            ctype=r.headers.get("content-type","").lower()
            ext=".pdf" if "pdf" in ctype or url.lower().endswith(".pdf") else ".html"
            year_match=re.search(r"\b(20\d{2})\b",f"{label} {url}")
            year=year_match.group(1) if year_match else "unknown"
            path=raw/f"year={year}"/(safe_name(Path(urlparse(url).path).name or label)+ext if not Path(urlparse(url).path).suffix else safe_name(Path(urlparse(url).path).name))
            save_bytes(r.content,path); record_download("des_agricultural_prices",url,path,notes=label)
            if path.suffix.lower()==".pdf":
                n=parse_pdf(path,processed/f"year={year}",log)
                log.info("%s | extracted rows=%s",path.name,n)
        except Exception as exc:
            log.exception("Failed %s: %s",url,exc)
        time.sleep(.5)

if __name__=="__main__": main()
