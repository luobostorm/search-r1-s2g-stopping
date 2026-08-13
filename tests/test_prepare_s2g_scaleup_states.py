from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_s2g_scaleup_states",
    ROOT / "scripts/prepare_s2g_scaleup_states.py",
)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def trajectory(question_id: str, search_count: int = 2):
    steps = []
    for index in range(search_count):
        steps.append(
            {
                "action": "SEARCH",
                "search_executed": True,
                "observation": (
                    f"\n\n<information>Doc 1(Title: T{index}) "
                    f"Evidence {index}.</information>\n\n"
                ),
            }
        )
    steps.append(
        {
            "action": "ANSWER",
            "search_executed": False,
            "observation": "",
        }
    )
    return {
        "record_type": "trajectory",
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "search_calls": search_count,
        "steps": steps,
    }


def test_expand_trajectory_builds_initial_and_cumulative_states():
    states = prepare.expand_trajectory(
        trajectory("q"),
        question_order=0,
        split_name="train",
    )
    assert len(states) == 3
    assert [row["turn_index"] for row in states] == [1, 2, 3]
    assert states[0]["context"] == prepare.NO_EVIDENCE
    assert "Evidence 0." in states[1]["context"]
    assert "Evidence 0." in states[2]["context"]
    assert "Evidence 1." in states[2]["context"]
    assert [row["native_next_action"] for row in states] == [
        "SEARCH",
        "SEARCH",
        "ANSWER",
    ]


def test_expand_excludes_budget_exhausted_post_fourth_search_state():
    states = prepare.expand_trajectory(
        trajectory("q4", search_count=4),
        question_order=0,
        split_name="train",
    )
    assert len(states) == prepare.MAX_DECISION_STATES == 4
    assert [row["state_index"] for row in states] == [0, 1, 2, 3]
    assert states[-1]["native_next_action"] == "SEARCH"
    assert "Evidence 2." in states[-1]["context"]
    assert "Evidence 3." not in states[-1]["context"]


def test_build_states_requires_complete_error_free_run():
    rows = [
        {
            "record_type": "run_start",
            "labels_mounted": False,
            "gold_fields_received": 0,
        },
        trajectory("q"),
        {"record_type": "run_end", "status": "completed"},
    ]
    states, audit = prepare.build_states(
        run_rows=rows,
        split_name="train",
        expected_count=1,
    )
    assert audit["passed"]
    assert audit["question_count"] == 1
    assert audit["state_count"] == 3
    assert all(not (set(prepare.teacher_row(row)) & prepare.FORBIDDEN) for row in states)


def test_missing_isolation_declaration_requires_explicit_compatibility():
    rows = [
        {"record_type": "run_start"},
        trajectory("q"),
        {"record_type": "run_end", "status": "completed"},
    ]
    _, strict = prepare.build_states(
        run_rows=rows,
        split_name="train",
        expected_count=1,
    )
    _, compatible = prepare.build_states(
        run_rows=rows,
        split_name="train",
        expected_count=1,
        allow_missing_isolation_declaration=True,
    )
    assert not strict["passed"]
    assert compatible["passed"]
    assert compatible["missing_isolation_declaration"]


def test_observation_requires_one_information_block():
    with pytest.raises(ValueError, match="exactly one"):
        prepare.observation_information("plain text")
