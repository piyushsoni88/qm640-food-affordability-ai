
from __future__ import annotations
import argparse, json
from pathlib import Path
from urllib.parse import urlparse
from _common import PROJECT_ROOT, build_session, ensure_dir, get_logger, record_download, safe_name, save_bytes

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default=str(PROJECT_ROOT/"config/state_mandi_sources.json"))
    args=p.parse_args()
    sources=json.loads(Path(args.config).read_text(encoding="utf-8"))
    s=build_session(); log=get_logger("state_mandi",PROJECT_ROOT/"data/logs/08_state_mandi.log")
    for item in sources:
        state=item["state"]; url=item["url"]; enabled=item.get("enabled",True)
        if not enabled: continue
        try:
            r=s.get(url,timeout=(20,180)); r.raise_for_status()
            out=ensure_dir(PROJECT_ROOT/"data/raw/state_mandi"/safe_name(state))
            ext=Path(urlparse(url).path).suffix or ".html"
            path=out/(safe_name(item.get("name",state))+ext)
            save_bytes(r.content,path); record_download("state_mandi",url,path,notes=state)
            log.info("%s downloaded %s bytes",state,len(r.content))
        except Exception as exc:
            log.exception("%s failed: %s",state,exc)

if __name__=="__main__": main()
