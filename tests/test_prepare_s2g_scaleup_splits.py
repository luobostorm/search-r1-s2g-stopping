from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_s2g_scaleup_splits",
    ROOT / "scripts/prepare_s2g_scaleup_splits.py",
)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def write_source(path: Path, rows: list[dict]) -> str:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(prefix: str, count: int) -> list[dict]:
    return [
        {
            "_id": f"{prefix}-{index}",
            "question": f"Question {prefix} {index}?",
            "answer": f"Answer {index}",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": [["Title", 0]],
            "context": [["Title", ["Sentence."]]],
        }
        for index in range(count)
    ]


def test_normalize_question_is_unicode_case_and_space_stable():
    assert prepare.normalize_question("  Ａ  TEST\nQuestion ") == "a test question"


def test_select_disjoint_splits_is_deterministic_and_excludes_questions():
    source = prepare.validate_source_rows(
        rows("train", 12), expected_rows=12, split_name="train"
    )
    excluded = {prepare.normalize_question("Question train 1?")}
    specs = (
        ("train", 4, "seed-a"),
        ("validation", 3, "seed-b"),
        ("reserve", 2, "seed-c"),
    )
    first = prepare.select_disjoint_splits(
        source,
        excluded_ids={"train-2"},
        excluded_questions=excluded,
        split_specs=specs,
    )
    second = prepare.select_disjoint_splits(
        source,
        excluded_ids={"train-2"},
        excluded_questions=excluded,
        split_specs=specs,
    )
    assert {
        name: [row["_original_id"] for row in selected]
        for name, selected in first.items()
    } == {
        name: [row["_original_id"] for row in selected]
        for name, selected in second.items()
    }
    selected_ids = [
        row["_original_id"]
        for selected in first.values()
        for row in selected
    ]
    assert len(selected_ids) == len(set(selected_ids))
    assert "train-1" not in selected_ids
    assert "train-2" not in selected_ids


def test_source_allows_duplicate_question_text_but_split_deduplicates():
    source_rows = rows("train", 5)
    source_rows[4]["question"] = source_rows[0]["question"]
    source = prepare.validate_source_rows(
        source_rows, expected_rows=5, split_name="train"
    )
    assert prepare.duplicate_normalized_question_count(source) == 1
    selected = prepare.select_disjoint_splits(
        source,
        excluded_ids=set(),
        excluded_questions=set(),
        split_specs=(("train", 4, "seed"),),
    )
    normalized = [
        row["_normalized_question"] for row in selected["train"]
    ]
    assert len(normalized) == len(set(normalized))


def test_materialize_separates_blind_inputs_and_labels(tmp_path: Path):
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    train_rows = rows("train", 15)
    validation_rows = rows("validation", 6)
    train_sha = write_source(train_path, train_rows)
    validation_sha = write_source(validation_path, validation_rows)
    exclusion = tmp_path / "prior.jsonl"
    exclusion.write_text(
        json.dumps({"question": "Question train 0?"}) + "\n",
        encoding="utf-8",
    )
    manifest = prepare.materialize(
        train_source=train_path,
        validation_source=validation_path,
        output_dir=tmp_path / "output",
        exclude_paths=[exclusion],
        expected_train_sha256=train_sha,
        expected_validation_sha256=validation_sha,
        expected_train_rows=15,
        expected_validation_rows=6,
        split_specs=(
            ("fresh_train_700", 4, "seed-a"),
            ("grouped_validation_100", 3, "seed-b"),
            ("training_reserve_200", 2, "seed-c"),
        ),
        final_count=6,
    )

    assert manifest["all_overlap_counts_zero"]
    assert manifest["input_gold_field_count"] == 0
    input_path = (
        tmp_path / "output" / "fresh_train_700-inputs.jsonl"
    )
    input_rows = [
        json.loads(line) for line in input_path.read_text().splitlines()
    ]
    assert len(input_rows) == 4
    assert all(
        set(row) == {
            "question_id",
            "question",
            "source_index",
            "source_split",
        }
        for row in input_rows
    )
    assert all(row["question"] != "Question train 0?" for row in input_rows)
    final_inputs = (
        tmp_path
        / "output"
        / "s2g_code_aligned_hotpotqa_dev_1000-inputs.jsonl"
    )
    assert len(final_inputs.read_text().splitlines()) == 6


def test_load_verified_json_fails_closed_on_hash_mismatch(tmp_path: Path):
    source = tmp_path / "source.json"
    write_source(source, rows("train", 1))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.load_verified_json(
            source,
            expected_sha256="0" * 64,
            expected_rows=1,
            split_name="train",
        )
