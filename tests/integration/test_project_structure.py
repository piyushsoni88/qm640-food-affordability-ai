from pathlib import Path

def test_structure():
    root=Path(__file__).resolve().parents[2]
    for p in ["config","data","notebooks","reports","src","tests"]: assert (root/p).exists()
