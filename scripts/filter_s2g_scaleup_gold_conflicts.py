#!/usr/bin/env python3
"""Apply the frozen one-way gold supporting-sentence conflict filter.

Run only after the complete label-blind trajectory and Teacher runs pass.
Gold data is used solely to identify the pre-registered strong conflict:
Teacher=insufficient even though every official supporting-fact sentence is
present in the accumulated context.  Gold data is never written to student
model inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_supporting_sentences(
    source_row: dict[str, Any],
    supporting_facts: Any,
) -> tuple[list[str], list[str]]:
    if not isinstance(supporting_facts, list) or not supporting_facts:
        raise ValueError("supporting_facts must be a non-empty list")
    context = source_row.get("context")
    if not isinstance(context, list):
        raise ValueError("source row has invalid context")
    title_to_sentences: dict[str, list[str]] = {}
    for entry in context:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or not isinstance(entry[1], list)
        ):
            raise ValueError("source context entry is invalid")
        title_to_sentences[entry[0]] = entry[1]
    titles: list[str] = []
    sentences: list[str] = []
    seen: set[tuple[str, int]] = set()
    for fact in supporting_facts:
        if (
            not isinstance(fact, list)
            or len(fact) != 2
            or not isinstance(fact[0], str)
            or not isinstance(fact[1], int)
        ):
            raise ValueError("supporting fact is invalid")
        title, sentence_index = fact
        key = (title, sentence_index)
        if key in seen:
            continue
        seen.add(key)
        if title not in title_to_sentences:
            raise ValueError(f"supporting title absent from source: {title}")
        source_sentences = title_to_sentences[title]
        if not 0 <= sentence_index < len(source_sentences):
            raise ValueError(
                f"supporting sentence index out of range: {key}"
            )
        sentence = source_sentences[sentence_index]
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f"supporting sentence is empty: {key}")
        titles.append(title)
        sentences.append(sentence)
    return titles, sentences


def filter_conflicts(
    *,
    student_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    labels = {str(row["question_id"]): row for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("label question IDs are duplicated")
    retained: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    coverage_counts: Counter[str] = Counter()
    questions = {str(row["question_id"]) for row in student_rows}
    if not questions.issubset(labels):
        raise ValueError("student questions are missing from labels")
    for student in student_rows:
        question_id = str(student["question_id"])
        label = labels[question_id]
        source_index = int(label["source_index"])
        if not 0 <= source_index < len(raw_rows):
            raise ValueError(f"{question_id}: source_index out of range")
        source = raw_rows[source_index]
        source_id = str(source.get("_id", source.get("id", "")))
        if source_id != question_id:
            raise ValueError(f"{question_id}: source identity mismatch")
        titles, sentences = source_supporting_sentences(
            source, label.get("supporting_facts")
        )
        normalized_context = normalize_text(str(student["context"]))
        title_complete = all(
            normalize_text(title) in normalized_context for title in titles
        )
        sentence_complete = all(
            normalize_text(sentence) in normalized_context
            for sentence in sentences
        )
        coverage_counts[
            f"title_complete={title_complete},sentence_complete={sentence_complete}"
        ] += 1
        strong_conflict = (
            not bool(student["teacher_sufficient"])
            and sentence_complete
        )
        if strong_conflict:
            conflicts.append(
                {
                    "request_id": str(student["request_id"]),
                    "question_id": question_id,
                    "state_index": int(student["state_index"]),
                    "teacher_sufficient": False,
                    "all_supporting_titles_present": title_complete,
                    "all_supporting_sentences_present": True,
                    "action": "excluded_without_relabeling",
                }
            )
            continue
        retained.append(dict(student))
    for index, row in enumerate(retained):
        row["sequence_index"] = index
    report = {
        "artifact_id": "searchr1-s2g-scaleup-gold-conflict-filter-v1",
        "passed": (
            len(retained) + len(conflicts) == len(student_rows)
            and len({row["request_id"] for row in retained})
            == len(retained)
        ),
        "input_state_count": len(student_rows),
        "retained_state_count": len(retained),
        "strong_conflict_count": len(conflicts),
        "retained_fraction": (
            len(retained) / len(student_rows) if student_rows else 0.0
        ),
        "question_count": len(questions),
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "rule": (
            "exclude only Teacher-insufficient states containing every "
            "official supporting-fact sentence; never relabel"
        ),
        "gold_fields_written_to_student_output": 0,
    }
    return retained, report, conflicts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-input", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--raw-source", required=True, type=Path)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--student-output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--conflicts", required=True, type=Path)
    args = parser.parse_args()
    if any(
        path.exists()
        for path in (args.student_output, args.audit, args.conflicts)
    ):
        raise FileExistsError("one or more outputs already exist")
    if sha256_file(args.labels) != args.expected_labels_sha256:
        raise RuntimeError("labels SHA256 mismatch")
    if sha256_file(args.raw_source) != args.expected_raw_sha256:
        raise RuntimeError("raw source SHA256 mismatch")
    raw_rows = json.loads(args.raw_source.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError("raw source is not a JSON array")
    retained, report, conflicts = filter_conflicts(
        student_rows=read_jsonl(args.student_input),
        label_rows=read_jsonl(args.labels),
        raw_rows=raw_rows,
    )
    if report["passed"]:
        write_jsonl(args.student_output, retained)
        write_jsonl(args.conflicts, conflicts)
        report["sources"] = {
            "student_input_sha256": sha256_file(args.student_input),
            "labels_sha256": sha256_file(args.labels),
            "raw_source_sha256": sha256_file(args.raw_source),
        }
        report["outputs"] = {
            "student_output_sha256": sha256_file(args.student_output),
            "conflicts_sha256": sha256_file(args.conflicts),
        }
    write_json(args.audit, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
