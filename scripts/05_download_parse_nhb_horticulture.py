
from __future__ import annotations
import argparse, time
from pathlib import Path
from urllib.parse import urljoin, urlparse
import pandas as pd
from bs4 import BeautifulSoup
from _common import PROJECT_ROOT, build_session, ensure_dir, get_logger, record_download, safe_name, save_bytes

PAGES=[
 "https://nhb.gov.in/statistics/area-production-statistics.html",
 "https://nhb.gov.in/Default.aspx/statistics/HorticultureCropsFinal/statistics/HorticultureCropsFirst/Statistics.aspx?Type=Publication&menu.Menu=131",
]

def parse_pdf(path,out_dir,log):
    try: import pdfplumber
    except Exception: return 0
    rows=0
    with pdfplumber.open(path) as pdf:
        for pn,page in enumerate(pdf.pages,1):
            for tn,t in enumerate(page.extract_tables() or [],1):
                if not t or len(t)<2: continue
                w=max(map(len,t)); clean=[list(r)+[None]*(w-len(r)) for r in t]
                df=pd.DataFrame(clean); df.to_csv(out_dir/f"{path.stem}_p{pn:04d}_t{tn:02d}.csv",index=False,header=False); rows+=len(df)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--max-files",type=int,default=80)
    args=p.parse_args()
    s=build_session(); log=get_logger("nhb",PROJECT_ROOT/"data/logs/05_nhb_horticulture.log")
    raw=ensure_dir(PROJECT_ROOT/"data/raw/nhb"); interim=ensure_dir(PROJECT_ROOT/"data/interim/nhb/tables")
    seen=set(); n=0
    for page in PAGES:
        try:
            r=s.get(page,timeout=(20,180)); r.raise_for_status()
            soup=BeautifulSoup(r.text,"html.parser")
            for a in soup.find_all("a",href=True):
                url=urljoin(page,a["href"]); label=" ".join(a.get_text(" ",strip=True).split()); low=(url+" "+label).lower()
                if url in seen or not any(k in low for k in ["horticulture","area","production","database","statistics"]): continue
                if not any(url.lower().split("?")[0].endswith(e) for e in [".pdf",".xls",".xlsx",".csv",".zip"]): continue
                seen.add(url); n+=1
                if n>args.max_files: break
                rr=s.get(url,timeout=(20,300)); rr.raise_for_status()
                name=safe_name(Path(urlparse(url).path).name or label); path=raw/name
                save_bytes(rr.content,path); record_download("nhb_horticulture",url,path,notes=label)
                if path.suffix.lower()==".pdf": log.info("%s extracted rows=%s",path.name,parse_pdf(path,interim,log))
                elif path.suffix.lower() in {".xls",".xlsx"}:
                    try:
                        xls=pd.ExcelFile(path)
                        for sheet in xls.sheet_names:
                            pd.read_excel(path,sheet_name=sheet).to_csv(interim/f"{path.stem}_{safe_name(sheet)}.csv",index=False)
                    except Exception as exc: log.exception("Excel parse failed %s: %s",path,exc)
                time.sleep(.5)
        except Exception as exc: log.exception("Page failed %s: %s",page,exc)
    log.info("Downloaded %s NHB assets",min(n,args.max_files))

if __name__=="__main__": main()
