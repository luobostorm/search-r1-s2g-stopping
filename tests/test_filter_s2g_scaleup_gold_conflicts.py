from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "filter_s2g_scaleup_gold_conflicts",
    ROOT / "scripts/filter_s2g_scaleup_gold_conflicts.py",
)
assert SPEC is not None and SPEC.loader is not None
filt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(filt)


def student(request: str, sufficient: bool, context: str) -> dict:
    return {
        "sequence_index": 0,
        "request_id": request,
        "question_id": "q1",
        "state_index": 1,
        "context": context,
        "teacher_sufficient": sufficient,
    }


def test_filter_excludes_only_one_way_strong_conflicts():
    raw = [
        {
            "_id": "q1",
            "context": [
                ["Title A", ["First fact.", "Required fact A."]],
                ["Title B", ["Required fact B."]],
            ],
        }
    ]
    labels = [
        {
            "question_id": "q1",
            "source_index": 0,
            "supporting_facts": [["Title A", 1], ["Title B", 0]],
        }
    ]
    rows = [
        student(
            "complete-insufficient",
            False,
            "Title A Required fact A. Title B Required fact B.",
        ),
        student("incomplete-insufficient", False, "Required fact A."),
        student("incomplete-sufficient", True, "Alternative evidence."),
    ]
    retained, report, conflicts = filt.filter_conflicts(
        student_rows=rows,
        label_rows=labels,
        raw_rows=raw,
    )
    assert report["passed"]
    assert [row["request_id"] for row in retained] == [
        "incomplete-insufficient",
        "incomplete-sufficient",
    ]
    assert [row["request_id"] for row in conflicts] == [
        "complete-insufficient"
    ]
    assert report["strong_conflict_count"] == 1


def test_supporting_sentence_resolution_fails_closed():
    source = {"context": [["Title", ["Only sentence."]]]}
    try:
        filt.source_supporting_sentences(
            source, [["Title", 9]]
        )
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("expected ValueError")
