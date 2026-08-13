#!/usr/bin/env python3
"""Materialize question-isolated S2G scale-up and aligned evaluation splits."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


TRAIN_SHA256 = (
    "26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316"
)
VALIDATION_SHA256 = (
    "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea"
)
TRAIN_ROWS = 90447
VALIDATION_ROWS = 7405
SPLIT_SPECS = (
    (
        "fresh_train_700",
        700,
        "20260727-s2g-scaleup-fresh-train-700-v1",
    ),
    (
        "grouped_validation_100",
        100,
        "20260727-s2g-scaleup-grouped-validation-100-v1",
    ),
    (
        "training_reserve_200",
        200,
        "20260727-s2g-scaleup-training-reserve-200-v1",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def original_id(row: dict[str, Any]) -> str:
    value = row.get("_id", row.get("id"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("row has no non-empty _id or id")
    return value.strip()


def validate_source_rows(
    rows: Any, *, expected_rows: int, split_name: str
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{split_name} source must be a JSON array")
    if len(rows) != expected_rows:
        raise ValueError(
            f"{split_name} row count mismatch: {len(rows)} != {expected_rows}"
        )
    ids: set[str] = set()
    output: list[dict[str, Any]] = []
    for source_index, value in enumerate(rows):
        if not isinstance(value, dict):
            raise ValueError(f"{split_name} row {source_index} is not an object")
        question_id = original_id(value)
        question = value.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"{split_name} row {source_index} has no question"
            )
        normalized_question = normalize_question(question)
        if question_id in ids:
            raise ValueError(f"duplicate {split_name} ID: {question_id}")
        ids.add(question_id)
        output.append(
            {
                **value,
                "_source_index": source_index,
                "_original_id": question_id,
                "_normalized_question": normalized_question,
            }
        )
    return output


def load_verified_json(
    path: Path, *, expected_sha256: str, expected_rows: int, split_name: str
) -> list[dict[str, Any]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{split_name} SHA256 mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return validate_source_rows(
        value, expected_rows=expected_rows, split_name=split_name
    )


def iter_json_values(path: Path) -> Iterable[Any]:
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    for candidate in paths:
        if not candidate.is_file() or candidate.suffix not in {
            ".json",
            ".jsonl",
        }:
            continue
        try:
            if candidate.suffix == ".jsonl":
                with candidate.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            yield json.loads(line)
            else:
                with candidate.open(encoding="utf-8") as handle:
                    yield json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue


def collect_questions(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "question" and isinstance(item, str):
                yield item
            else:
                yield from collect_questions(item)
    elif isinstance(value, list):
        for item in value:
            yield from collect_questions(item)


def exclusion_questions(paths: list[Path]) -> set[str]:
    questions: set[str] = set()
    for path in paths:
        for value in iter_json_values(path):
            questions.update(normalize_question(q) for q in collect_questions(value))
    return questions


def stable_rank(seed: str, row: dict[str, Any]) -> str:
    payload = (
        seed
        + "\0"
        + str(row["_original_id"])
        + "\0"
        + str(row["_normalized_question"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_disjoint_splits(
    rows: list[dict[str, Any]],
    *,
    excluded_ids: set[str],
    excluded_questions: set[str],
    split_specs: tuple[tuple[str, int, str], ...] = SPLIT_SPECS,
) -> dict[str, list[dict[str, Any]]]:
    eligible = [
        row
        for row in rows
        if row["_original_id"] not in excluded_ids
        and row["_normalized_question"] not in excluded_questions
    ]
    selected_ids: set[str] = set()
    selected_questions: set[str] = set()
    output: dict[str, list[dict[str, Any]]] = {}
    for name, count, seed in split_specs:
        ranked = sorted(
            (
                row
                for row in eligible
                if row["_original_id"] not in selected_ids
                and row["_normalized_question"] not in selected_questions
            ),
            key=lambda row: (
                stable_rank(seed, row),
                row["_source_index"],
            ),
        )
        if len(ranked) < count:
            raise ValueError(
                f"only {len(ranked)} eligible rows remain for {name}; "
                f"need {count}"
            )
        chosen = ranked[:count]
        output[name] = chosen
        selected_ids.update(str(row["_original_id"]) for row in chosen)
        selected_questions.update(
            str(row["_normalized_question"]) for row in chosen
        )
    return output


def first_unique(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        question_id = str(row["_original_id"])
        if question_id in seen:
            continue
        output.append(row)
        seen.add(question_id)
        if len(output) == count:
            return output
    raise ValueError(f"only {len(output)} unique rows; need {count}")


def duplicate_normalized_question_count(
    rows: list[dict[str, Any]],
) -> int:
    questions = [str(row["_normalized_question"]) for row in rows]
    return len(questions) - len(set(questions))


def input_row(row: dict[str, Any], *, source_split: str) -> dict[str, Any]:
    return {
        "question_id": str(row["_original_id"]),
        "question": str(row["question"]),
        "source_split": source_split,
        "source_index": int(row["_source_index"]),
    }


def label_row(row: dict[str, Any], *, source_split: str) -> dict[str, Any]:
    return {
        "question_id": str(row["_original_id"]),
        "answer": row.get("answer"),
        "type": row.get("type"),
        "level": row.get("level"),
        "supporting_facts": row.get("supporting_facts"),
        "source_split": source_split,
        "source_index": int(row["_source_index"]),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize(
    *,
    train_source: Path,
    validation_source: Path,
    output_dir: Path,
    exclude_paths: list[Path],
    expected_train_sha256: str = TRAIN_SHA256,
    expected_validation_sha256: str = VALIDATION_SHA256,
    expected_train_rows: int = TRAIN_ROWS,
    expected_validation_rows: int = VALIDATION_ROWS,
    split_specs: tuple[tuple[str, int, str], ...] = SPLIT_SPECS,
    final_count: int = 1000,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    train_rows = load_verified_json(
        train_source,
        expected_sha256=expected_train_sha256,
        expected_rows=expected_train_rows,
        split_name="train",
    )
    validation_rows = load_verified_json(
        validation_source,
        expected_sha256=expected_validation_sha256,
        expected_rows=expected_validation_rows,
        split_name="validation",
    )
    prior_questions = exclusion_questions(exclude_paths)
    validation_ids = {str(row["_original_id"]) for row in validation_rows}
    validation_questions = {
        str(row["_normalized_question"]) for row in validation_rows
    }
    excluded_questions = prior_questions | validation_questions
    splits = select_disjoint_splits(
        train_rows,
        excluded_ids=validation_ids,
        excluded_questions=excluded_questions,
        split_specs=split_specs,
    )
    final_rows = first_unique(validation_rows, final_count)

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for split_name, rows in {
        **splits,
        "s2g_code_aligned_hotpotqa_dev_1000": final_rows,
    }.items():
        source_split = (
            "validation"
            if split_name == "s2g_code_aligned_hotpotqa_dev_1000"
            else "train"
        )
        inputs_path = output_dir / f"{split_name}-inputs.jsonl"
        labels_path = output_dir / f"{split_name}-labels.jsonl"
        write_jsonl(
            inputs_path,
            (input_row(row, source_split=source_split) for row in rows),
        )
        write_jsonl(
            labels_path,
            (label_row(row, source_split=source_split) for row in rows),
        )
        files[split_name] = {
            "count": len(rows),
            "inputs": {
                "path": str(inputs_path),
                "sha256": sha256_file(inputs_path),
            },
            "labels": {
                "path": str(labels_path),
                "sha256": sha256_file(labels_path),
                "mount_policy": "isolated_read_only_after_blind_run",
            },
            "question_ids_sha256": hashlib.sha256(
                "\n".join(str(row["_original_id"]) for row in rows).encode()
            ).hexdigest(),
        }

    id_sets = {
        name: {str(row["_original_id"]) for row in rows}
        for name, rows in {**splits, "final": final_rows}.items()
    }
    overlaps = {
        f"{left}__{right}": len(id_sets[left] & id_sets[right])
        for left in sorted(id_sets)
        for right in sorted(id_sets)
        if left < right
    }
    manifest = {
        "artifact_id": "searchr1-s2g-scaleup-aligned-splits-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "sources": {
            "train": {
                "path": str(train_source),
                "sha256": sha256_file(train_source),
                "rows": len(train_rows),
                "duplicate_normalized_question_count": (
                    duplicate_normalized_question_count(train_rows)
                ),
            },
            "validation": {
                "path": str(validation_source),
                "sha256": sha256_file(validation_source),
                "rows": len(validation_rows),
                "duplicate_normalized_question_count": (
                    duplicate_normalized_question_count(validation_rows)
                ),
            },
        },
        "prior_exclusion_path_count": len(exclude_paths),
        "prior_exclusion_question_count": len(prior_questions),
        "selection_rule": (
            "sequential split selection by "
            "SHA256(seed NUL original_id NUL normalized_question)"
        ),
        "files": files,
        "id_overlap_counts": overlaps,
        "all_overlap_counts_zero": all(value == 0 for value in overlaps.values()),
        "input_gold_field_count": 0,
    }
    manifest_path = output_dir / "split-manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", required=True, type=Path)
    parser.add_argument("--validation-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--exclude-path",
        action="append",
        default=[],
        type=Path,
        help="JSON/JSONL file or directory containing prior questions",
    )
    args = parser.parse_args()
    manifest = materialize(
        train_source=args.train_source,
        validation_source=args.validation_source,
        output_dir=args.output_dir,
        exclude_paths=args.exclude_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
