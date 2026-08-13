from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_s2g_scaleup_thresholds",
    ROOT / "scripts/select_s2g_scaleup_thresholds.py",
)
assert SPEC is not None and SPEC.loader is not None
select = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(select)


def test_threshold_maximizes_recall_under_precision_constraint():
    result = select.select_threshold(
        [0.9, 0.8, 0.7, 0.6],
        [True, True, False, True],
        minimum_precision=0.9,
        minimum_stop_count=1,
    )
    assert result["feasible"]
    assert result["threshold"] == 0.8
    assert result["precision"] == 1.0
    assert result["recall"] == 2 / 3


def test_threshold_falls_back_to_never_stop():
    result = select.select_threshold(
        [0.9, 0.8],
        [False, True],
        minimum_precision=1.0,
        minimum_stop_count=2,
    )
    assert not result["feasible"]
    assert result["fallback"] == "never_stop"
