
from __future__ import annotations
import argparse, os, time
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from _common import PROJECT_ROOT, build_session, ensure_dir, get_logger, record_download, write_json

RESOURCE_ID="9ef84268-d588-465a-a308-a864a43d0070"
URL=f"https://api.data.gov.in/resource/{RESOURCE_ID}"
DEFAULT=["Onion","Tomato","Potato","Wheat","Rice","Gram","Arhar (Tur/Red Gram)(Whole)"]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--page-size",type=int,default=500); p.add_argument("--commodities",nargs="*",default=DEFAULT)
    args=p.parse_args()
    load_dotenv(PROJECT_ROOT/".env")
    key=os.getenv("DATA_GOV_IN_API_KEY")
    if not key: raise SystemExit("DATA_GOV_IN_API_KEY missing from .env")
    s=build_session(); log=get_logger("agmarknet",PROJECT_ROOT/"data/logs/07_agmarknet_current.log")
    all_rows=[]
    for commodity in args.commodities:
        offset=0
        while True:
            params={"api-key":key,"format":"json","offset":offset,"limit":args.page_size,"filters[commodity]":commodity}
            r=s.get(URL,params=params,timeout=(20,300)); r.raise_for_status(); payload=r.json()
            if str(payload.get("status","")).lower()=="error": raise RuntimeError(payload.get("message"))
            rows=payload.get("records") or []; total=int(payload.get("total") or 0)
            if not rows: break
            all_rows.extend(rows); offset+=len(rows)
            log.info("%s downloaded=%s total=%s",commodity,offset,total)
            if len(rows)<args.page_size or (total and offset>=total): break
            time.sleep(.7)
    if not all_rows: log.warning("No records"); return
    df=pd.DataFrame(all_rows).drop_duplicates()
    if "arrival_date" in df: df["arrival_date"]=pd.to_datetime(df["arrival_date"],dayfirst=True,errors="coerce")
    for c in ["min_price","max_price","modal_price"]:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce")
    now=datetime.now(); out=ensure_dir(PROJECT_ROOT/f"data/raw/agmarknet/current/year={now:%Y}/month={now:%m}")
    stem=out/f"agmarknet_current_{now:%Y%m%d_%H%M%S}"
    df.to_csv(stem.with_suffix(".csv"),index=False); df.to_parquet(stem.with_suffix(".parquet"),index=False)
    record_download("agmarknet_current",URL,stem.with_suffix(".csv"),notes=f"resource={RESOURCE_ID}")
    log.info("Saved %s rows",len(df))

if __name__=="__main__": main()
