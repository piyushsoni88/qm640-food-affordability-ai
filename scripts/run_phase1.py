from pathlib import Path
from food_affordability_ai.utils.logging import get_logger
from food_affordability_ai.utils.paths import ensure_project_directories

def main():
    logger=get_logger(__name__)
    root=Path(__file__).resolve().parents[1]
    ensure_project_directories(root)
    logger.info("Phase 1 repository scaffold is ready.")

if __name__=="__main__": main()
