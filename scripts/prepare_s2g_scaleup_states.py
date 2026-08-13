#!/usr/bin/env python3
"""Expand label-blind Native Search-R1 trajectories into S2G Judge states."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


NO_EVIDENCE = "[NO RETRIEVED EVIDENCE]"
MAX_DECISION_STATES = 4
INFORMATION_RE = re.compile(
    r"<information>(.*?)(?:</information>|$)", flags=re.DOTALL
)
FORBIDDEN = {
    "answer",
    "answers",
    "gold",
    "gold_answer",
    "golden_answers",
    "label",
    "labels",
    "official_em",
    "prediction",
    "reward_model",
    "supporting_facts",
    "target",
    "targets",
}
SYSTEM_PROMPT = """You are a QA/RAG sufficiency judge.
Given a QUESTION and a CONTEXT (documents retrieved so far), decide whether the CONTEXT alone contains enough information to reliably answer the QUESTION. If not, list the gap items that describe what information is still missing.

Rules:
- Base the decision ONLY on the supplied CONTEXT.
- Do not use memorized or external knowledge to fill a missing fact.
- "sufficient" may be true only when the CONTEXT explicitly supports a reliable answer.
- When "sufficient" is false, gap_items must be a non-empty list of concrete missing facts.
- Each gap item must use one category from: bridge entity, attribute, relation, evidence span, other.
- When "sufficient" is true, gap_items must be [].
- Emit the two top-level keys in this exact order: "sufficient", then "gap_items".

Output exactly one JSON object and nothing else, with exactly two top-level keys:
{
  "sufficient": true/false,
  "gap_items": [
    {
      "category": "bridge entity | attribute | relation | evidence span | other",
      "target": "string",
      "slot": "string",
      "description": "string"
    }
  ]
}"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
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


def observation_information(observation: str) -> str:
    blocks = [match.strip() for match in INFORMATION_RE.findall(observation)]
    blocks = [block for block in blocks if block]
    if len(blocks) != 1:
        raise ValueError(
            f"search observation must contain exactly one information block; "
            f"got {len(blocks)}"
        )
    return blocks[0]


def expand_trajectory(
    trajectory: dict[str, Any],
    *,
    question_order: int,
    split_name: str,
) -> list[dict[str, Any]]:
    question_id = trajectory.get("question_id")
    question = trajectory.get("question")
    steps = trajectory.get("steps")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("trajectory has invalid question_id")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{question_id}: trajectory has invalid question")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{question_id}: trajectory has no steps")

    contexts: list[str] = []
    states: list[dict[str, Any]] = []

    def append_state(state_index: int, next_action: str) -> None:
        context = "\n\n".join(contexts) if contexts else NO_EVIDENCE
        states.append(
            {
                "record_type": "state",
                "sequence_index": -1,
                "request_id": f"{question_id}:search-state:{state_index}",
                "state_id": f"{question_id}:search-state:{state_index}",
                "question_order": question_order,
                "question_id": question_id,
                "state_index": state_index,
                "turn_index": state_index + 1,
                "searches_used": state_index,
                "remaining_search_budget": max(0, 3 - state_index),
                "state_role": "native_reachable",
                "native_next_action": next_action,
                "split": split_name,
                "question": question,
                "context": context,
                "context_sha256": sha256_text(context),
            }
        )

    append_state(0, str(steps[0].get("action", "INVALID")))
    search_count = 0
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"{question_id}: step {step_index} is not an object")
        if step.get("action") != "SEARCH" or not step.get("search_executed"):
            continue
        observation = step.get("observation")
        if not isinstance(observation, str) or not observation:
            raise ValueError(
                f"{question_id}: search step {step_index} has no observation"
            )
        contexts.append(observation_information(observation))
        search_count += 1
        # State 4 after the fourth search is a budget-exhausted terminal
        # state, not a STOP-vs-CONTINUE decision.  The frozen protocol keeps
        # at most four decision states: initial plus post-search states 1..3.
        if search_count < MAX_DECISION_STATES:
            next_step = (
                steps[step_index + 1]
                if step_index + 1 < len(steps)
                else {}
            )
            append_state(
                search_count,
                str(next_step.get("action", "INVALID")),
            )
    if search_count != trajectory.get("search_calls"):
        raise ValueError(
            f"{question_id}: reconstructed {search_count} searches, "
            f"trajectory reports {trajectory.get('search_calls')}"
        )
    return states


def build_states(
    *,
    run_rows: list[dict[str, Any]],
    split_name: str,
    expected_count: int,
    allow_missing_isolation_declaration: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    starts = [row for row in run_rows if row.get("record_type") == "run_start"]
    trajectories = [
        row for row in run_rows if row.get("record_type") == "trajectory"
    ]
    errors = [
        row for row in run_rows if row.get("record_type") == "trajectory_error"
    ]
    ends = [row for row in run_rows if row.get("record_type") == "run_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("run must have exactly one run_start and run_end")
    if errors:
        raise ValueError(f"run contains {len(errors)} trajectory errors")
    if ends[0].get("status") != "completed":
        raise ValueError("run_end is not completed")
    if len(trajectories) != expected_count:
        raise ValueError(
            f"trajectory count mismatch: {len(trajectories)} != {expected_count}"
        )
    question_ids = [str(row.get("question_id", "")) for row in trajectories]
    if any(not value for value in question_ids) or len(set(question_ids)) != len(
        question_ids
    ):
        raise ValueError("trajectory IDs are empty or duplicated")
    declared_isolated = (
        starts[0].get("labels_mounted") is False
        and starts[0].get("gold_fields_received") == 0
    )
    missing_isolation_declaration = (
        "labels_mounted" not in starts[0]
        and "gold_fields_received" not in starts[0]
    )
    isolation_compatible = declared_isolated or (
        allow_missing_isolation_declaration
        and missing_isolation_declaration
    )

    states: list[dict[str, Any]] = []
    per_question = Counter()
    for question_order, trajectory in enumerate(trajectories):
        expanded = expand_trajectory(
            trajectory,
            question_order=question_order,
            split_name=split_name,
        )
        per_question[len(expanded)] += 1
        states.extend(expanded)
    for sequence_index, state in enumerate(states):
        state["sequence_index"] = sequence_index

    turn_counts = Counter(int(row["turn_index"]) for row in states)
    next_action_counts = Counter(str(row["native_next_action"]) for row in states)
    model_fields = {"question", "context"}
    audit = {
        "passed": (
            len(trajectories) == expected_count
            and not errors
            and isolation_compatible
            and len({row["request_id"] for row in states}) == len(states)
            and not any(
                set({key: row[key] for key in model_fields}) & FORBIDDEN
                for row in states
            )
        ),
        "split": split_name,
        "question_count": len(trajectories),
        "state_count": len(states),
        "unique_request_id_count": len(
            {str(row["request_id"]) for row in states}
        ),
        "states_per_question": dict(sorted(per_question.items())),
        "maximum_decision_states_per_question": MAX_DECISION_STATES,
        "budget_exhausted_terminal_states_excluded": sum(
            int(row.get("search_calls", 0)) >= MAX_DECISION_STATES
            for row in trajectories
        ),
        "turn_counts": dict(sorted(turn_counts.items())),
        "turn_fractions": {
            str(turn): count / len(states)
            for turn, count in sorted(turn_counts.items())
        },
        "native_next_action_counts": dict(sorted(next_action_counts.items())),
        "model_facing_fields": sorted(model_fields),
        "forbidden_model_fields_absent": True,
        "run_start_labels_mounted": starts[0].get("labels_mounted"),
        "run_start_gold_fields_received": starts[0].get(
            "gold_fields_received"
        ),
        "declared_label_isolation": declared_isolated,
        "allow_missing_isolation_declaration": (
            allow_missing_isolation_declaration
        ),
        "missing_isolation_declaration": missing_isolation_declaration,
        "isolation_compatible": isolation_compatible,
    }
    return states, audit


def teacher_row(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence_index": int(state["sequence_index"]),
        "request_id": str(state["request_id"]),
        "question_id": str(state["question_id"]),
        "state_id": str(state["state_id"]),
        "state_index": int(state["state_index"]),
        "split": str(state["split"]),
        "question": str(state["question"]),
        "context": str(state["context"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--state-output", required=True, type=Path)
    parser.add_argument("--teacher-inputs", required=True, type=Path)
    parser.add_argument("--teacher-smoke-inputs", required=True, type=Path)
    parser.add_argument("--teacher-smoke-count", type=int, default=40)
    parser.add_argument("--teacher-protocol", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument(
        "--allow-missing-isolation-declaration",
        action="store_true",
        help=(
            "Compatibility only for an already-started run whose container "
            "mount audit independently proves no labels were mounted."
        ),
    )
    args = parser.parse_args()
    outputs = (
        args.state_output,
        args.teacher_inputs,
        args.teacher_smoke_inputs,
        args.teacher_protocol,
        args.audit,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("one or more outputs already exist")

    states, audit = build_states(
        run_rows=read_jsonl(args.run),
        split_name=args.split_name,
        expected_count=args.expected_count,
        allow_missing_isolation_declaration=(
            args.allow_missing_isolation_declaration
        ),
    )
    teacher_inputs = [teacher_row(row) for row in states]
    teacher_smoke = teacher_inputs[: args.teacher_smoke_count]
    if len(teacher_smoke) != args.teacher_smoke_count:
        raise ValueError("not enough states for Teacher smoke")
    write_jsonl(args.state_output, states)
    write_jsonl(args.teacher_inputs, teacher_inputs)
    write_jsonl(args.teacher_smoke_inputs, teacher_smoke)
    protocol = {
        "artifact_id": (
            f"searchr1-s2g-scaleup-{args.split_name}-teacher-protocol-v1"
        ),
        "frozen_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "teacher": {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "temperature": 0.0,
            "max_new_tokens": 512,
            "thinking": False,
            "concurrency": 4,
            "system_prompt": SYSTEM_PROMPT,
            "user_template": "QUESTION:\n{question}\n\nCONTEXT:\n{context}",
            "must_not_receive": sorted(FORBIDDEN),
        },
        "sources": {
            "trajectory_run_path": str(args.run),
            "trajectory_run_sha256": sha256_file(args.run),
            "state_manifest_path": str(args.state_output),
            "state_manifest_sha256": sha256_file(args.state_output),
            "teacher_inputs_path": str(args.teacher_inputs),
            "teacher_inputs_sha256": sha256_file(args.teacher_inputs),
            "smoke_inputs_path": str(args.teacher_smoke_inputs),
            "smoke_inputs_sha256": sha256_file(args.teacher_smoke_inputs),
        },
    }
    write_json(args.teacher_protocol, protocol)
    audit["sources"] = {
        "trajectory_run_sha256": sha256_file(args.run),
        "state_manifest_sha256": sha256_file(args.state_output),
        "teacher_inputs_sha256": sha256_file(args.teacher_inputs),
        "teacher_smoke_inputs_sha256": sha256_file(
            args.teacher_smoke_inputs
        ),
        "teacher_protocol_sha256": sha256_file(args.teacher_protocol),
    }
    write_json(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
