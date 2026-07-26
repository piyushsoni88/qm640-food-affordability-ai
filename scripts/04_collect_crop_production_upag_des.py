
from __future__ import annotations
import argparse, re, time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
from _common import PROJECT_ROOT, build_session, ensure_dir, get_logger, record_download, safe_name, save_bytes

PAGES = [
 "https://desagri.gov.in/document-report-category/agriculture-statistics-at-a-glance/",
 "https://desagri.gov.in/document-report-category/area-production-yield/",
 "https://upag.gov.in/",
]

def parse_tabular(path, processed, log):
    try:
        if path.suffix.lower()==".csv": df=pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx",".xls"}: df=pd.read_excel(path)
        else: return
        out=processed/(path.stem+".csv"); df.to_csv(out,index=False)
        log.info("Parsed %s rows from %s",len(df),path.name)
    except Exception as exc: log.exception("Parse failed %s: %s",path,exc)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--max-files",type=int,default=100)
    args=p.parse_args()
    s=build_session(); log=get_logger("crop_production",PROJECT_ROOT/"data/logs/04_crop_production.log")
    raw=ensure_dir(PROJECT_ROOT/"data/raw/des/crop_production"); processed=ensure_dir(PROJECT_ROOT/"data/interim/des/crop_production")
    seen=set(); count=0
    for page in PAGES:
        try:
            r=s.get(page,timeout=(20,180)); r.raise_for_status()
            page_path=raw/"source_pages"/(safe_name(urlparse(page).netloc+"_"+urlparse(page).path)+".html")
            save_bytes(r.content,page_path); record_download("crop_production",page,page_path)
            soup=BeautifulSoup(r.text,"html.parser")
            for a in soup.find_all("a",href=True):
                url=urljoin(page,a["href"]); lower=url.lower()
                label=" ".join(a.get_text(" ",strip=True).split())
                if url in seen: continue
                if not any(k in (label+" "+lower).lower() for k in ["production","yield","area","agriculture statistics","crop"]): continue
                if not any(lower.endswith(ext) for ext in [".csv",".xlsx",".xls",".pdf",".zip"]): continue
                seen.add(url); count+=1
                if count>args.max_files: break
                rr=s.get(url,timeout=(20,300)); rr.raise_for_status()
                name=safe_name(Path(urlparse(url).path).name or label)
                path=raw/name; save_bytes(rr.content,path); record_download("crop_production",url,path,notes=label)
                parse_tabular(path,processed,log)
                time.sleep(.5)
        except Exception as exc: log.exception("Page failed %s: %s",page,exc)
    log.info("Downloaded %s candidate assets",min(count,args.max_files))

if __name__=="__main__": main()
