from pathlib import Path

def ensure_project_directories(root: Path):
    for p in ["data/raw","data/interim","data/processed","models","reports/figures","reports/tables","reports/model_cards"]:
        (root/p).mkdir(parents=True,exist_ok=True)
