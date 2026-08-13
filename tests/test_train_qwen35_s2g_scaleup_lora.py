from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_qwen35_s2g_scaleup_lora",
    ROOT
    / "deploy/searchr1-v02/train_qwen35_s2g_scaleup_lora.py",
)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def test_balanced_schedule_is_deterministic_and_epoch_specific():
    labels = [False] * 5 + [True] * 2
    first = trainer.deterministic_balanced_indices(
        labels, seed=7, epoch=1
    )
    repeat = trainer.deterministic_balanced_indices(
        labels, seed=7, epoch=1
    )
    second = trainer.deterministic_balanced_indices(
        labels, seed=7, epoch=2
    )
    assert first == repeat
    assert first != second
    assert len(first) == 10
    assert sum(labels[index] for index in first) == 5
    assert len(first) - sum(labels[index] for index in first) == 5
    assert set(range(5)).issubset(first)
    assert {5, 6}.issubset(first)


def test_checkpoint_selection_uses_validation_nll_and_earliest_tie():
    records = [
        {"epoch": 1, "validation_token_nll": 0.5},
        {"epoch": 2, "validation_token_nll": 0.4},
        {"epoch": 3, "validation_token_nll": 0.4},
    ]
    assert trainer.select_best_epoch(records) == 2


def test_select_smoke_rows_is_balanced_and_natural_ordered():
    rows = [
        {
            "request_id": f"r{index}",
            "teacher_sufficient": index in {2, 5, 7},
        }
        for index in range(8)
    ]
    selected = trainer.select_smoke_rows(rows, total=4)
    assert [row["request_id"] for row in selected] == [
        "r0",
        "r1",
        "r2",
        "r5",
    ]
    assert sum(bool(row["teacher_sufficient"]) for row in selected) == 2
