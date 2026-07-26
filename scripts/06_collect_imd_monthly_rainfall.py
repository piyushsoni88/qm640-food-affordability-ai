
from __future__ import annotations
import argparse, re
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from _common import PROJECT_ROOT, build_session, ensure_dir, get_logger, record_download, safe_name, save_bytes

PAGES=[
 "https://mausam.imd.gov.in/responsive/rainfallinformation.php",
 "https://mausam.imd.gov.in/responsive/rainfallinformation_msd.php",
 "https://mausam.imd.gov.in/imd_latest/contents/rainfallinformation.php?msg=C",
]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--year",type=int,default=date.today().year); p.add_argument("--month",type=int,default=date.today().month)
    args=p.parse_args()
    s=build_session(); log=get_logger("imd",PROJECT_ROOT/"data/logs/06_imd_monthly_rainfall.log")
    raw=ensure_dir(PROJECT_ROOT/f"data/raw/imd/monthly/year={args.year}/month={args.month:02d}")
    meta=ensure_dir(PROJECT_ROOT/"data/metadata/imd")
    downloaded=0
    for page in PAGES:
        try:
            r=s.get(page,timeout=(20,180)); r.raise_for_status()
            html=raw/(safe_name(urlparse(page).netloc+"_"+urlparse(page).path)+".html")
            save_bytes(r.content,html); record_download("imd_rainfall",page,html)
            soup=BeautifulSoup(r.text,"html.parser")
            for tag in soup.find_all(["a","img"],href=True)+soup.find_all("img",src=True):
                href=tag.get("href") or tag.get("src"); url=urljoin(page,href); low=url.lower()
                if not any(k in low for k in ["rain","monthly","rf_","rainfall"]): continue
                if not any(low.split("?")[0].endswith(e) for e in [".pdf",".csv",".xls",".xlsx",".png",".jpg",".jpeg"]): continue
                try:
                    rr=s.get(url,timeout=(20,300)); rr.raise_for_status()
                    path=raw/safe_name(Path(urlparse(url).path).name)
                    save_bytes(rr.content,path); record_download("imd_rainfall",url,path); downloaded+=1
                except Exception as exc: log.warning("Asset failed %s: %s",url,exc)
        except Exception as exc: log.exception("Page failed %s: %s",page,exc)
    note=meta/"HISTORICAL_ACCESS_REQUIRED.txt"
    note.write_text(
      "The public IMD rainfall pages mainly expose current products. Complete historical monthly data may require the IMD Data Service Portal or an official data request. This collector does not fabricate unavailable history.\n",
      encoding="utf-8")
    log.info("Downloaded %s current monthly assets; see %s for historical limitation",downloaded,note)

if __name__=="__main__": main()
