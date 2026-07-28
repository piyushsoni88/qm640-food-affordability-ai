from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "13_collect_agmarknet_historical.py"
)
SPEC = importlib.util.spec_from_file_location("agmarknet_historical", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def test_verified_target_inventory():
    assert len(COLLECTOR.TARGETS) == 8
    assert sum(COLLECTOR.TARGETS.values()) == 18_836_462


def test_default_size_requires_192_unique_pages():
    pages = list(COLLECTOR.tasks(100_000))
    keys = {(commodity, offset) for commodity, offset, _, _ in pages}
    assert len(pages) == 192
    assert len(keys) == len(pages)
    assert sum(expected for _, _, expected, _ in pages) == 18_836_462
