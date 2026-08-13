"""Causal stopping-Judge dataset, split, audit, and evaluation utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .searchr1_v02_eval import official_em
from .searchr1_v02_gateh0_eval import token_f1


NO_EVIDENCE = "[NO RETRIEVED EVIDENCE]"
INFORMATION_RE = re.compile(
    r"<information>(.*?)(?:</information>|(?=<\|im_end\|>)|$)",
    flags=re.DOTALL,
)
FORBIDDEN_MODEL_FIELDS = {
    "answer",
    "answers",
    "gold",
    "gold_answer",
    "label",
    "labels",
    "native_answer",
    "native_prediction",
    "official_em",
    "prediction",
    "reward_model",
    "target",
    "targets",
    "token_f1",
}
ACTION_LABELS = {"STOP", "CONTINUE"}
JUDGE_PARSE_FAILURE_EFFECTIVE_SCORE = -1.0e30


def judge_score_is_structured_parse_failure(
    row: Mapping[str, Any],
) -> bool:
    """Return whether a structured Judge result explicitly failed parsing.

    Older binary-Judge records did not carry a ``parsed`` field, so absence of
    the field is not treated as a parse failure.  The structured S2G runner
    always writes the field and uses ``null`` for invalid completions.
    """

    return "parsed" in row and row.get("parsed") is None


def effective_judge_score(row: Mapping[str, Any]) -> float:
    """Return the score consumed by stopping policies.

    A malformed structured completion is fail-closed to CONTINUE regardless
    of its provenance-only first-token margin.  Parseable records, and legacy
    records without a structured ``parsed`` field, retain their raw score.
    """

    raw_score = row.get("score")
    if not isinstance(raw_score, (int, float)) or not math.isfinite(
        float(raw_score)
    ):
        raise ValueError("Judge score is missing or non-finite")
    if judge_score_is_structured_parse_failure(row):
        return JUDGE_PARSE_FAILURE_EFFECTIVE_SCORE
    return float(raw_score)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def core_rank(seed: str, row: Mapping[str, Any]) -> str:
    payload = "\0".join(
        (
            str(seed),
            str(row["data_source"]).strip().lower(),
            str(row["question_id"]),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def materialize_core_final_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: str,
    core_count: int = 200,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != 1000:
        raise ValueError(f"expected frozen final-1000, got {len(rows)}")
    ids = [str(row.get("question_id", "")) for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("final inputs have empty or duplicate question IDs")
    for index, row in enumerate(rows):
        forbidden = set(row) & FORBIDDEN_MODEL_FIELDS
        if forbidden:
            raise ValueError(f"final input {index} has forbidden fields {forbidden}")
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (core_rank(seed, row), str(row["question_id"])),
    )
    core = ordered[:core_count]
    reserve = ordered[core_count:]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "selection_is_label_blind": True,
        "labels_read_or_written": False,
        "seed": seed,
        "rank_rule": (
            "SHA256(seed + NUL + normalized_data_source + NUL + question_id)"
        ),
        "source_count": len(rows),
        "core_count": len(core),
        "extension_reserve_count": len(reserve),
        "unique_source_ids": len(set(ids)),
        "core_reserve_overlap": len(
            {str(row["question_id"]) for row in core}
            & {str(row["question_id"]) for row in reserve}
        ),
        "core_ids_sha256": json_sha256(
            [str(row["question_id"]) for row in core]
        ),
        "reserve_ids_sha256": json_sha256(
            [str(row["question_id"]) for row in reserve]
        ),
    }
    return core, reserve, manifest


def extract_evidence(rolling_state_text: str) -> tuple[str, int]:
    assistant_marker = "<|im_start|>assistant\n"
    trajectory_text = (
        rolling_state_text.split(assistant_marker, 1)[1]
        if assistant_marker in rolling_state_text
        else ""
    )
    blocks = [match.strip() for match in INFORMATION_RE.findall(trajectory_text)]
    blocks = [block for block in blocks if block]
    if not blocks:
        return NO_EVIDENCE, 0
    return "\n\n".join(blocks), len(blocks)


def render_user_text(
    *,
    question: str,
    evidence: str,
    searches_used: int,
    max_searches: int,
) -> str:
    remaining = max(0, int(max_searches) - int(searches_used))
    return (
        "EVIDENCE RETRIEVED SO FAR:\n"
        f"{evidence}\n\n"
        "QUESTION:\n"
        f"{question.strip()}\n\n"
        f"SEARCHES_USED: {int(searches_used)}\n"
        f"REMAINING_SEARCH_BUDGET: {remaining}\n\n"
        "Output exactly one action: STOP or CONTINUE."
    )


def _question_map(forced_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in forced_rows:
        if row.get("record_type") != "trajectory":
            continue
        question_id = str(row.get("question_id", ""))
        question = str(row.get("question", "")).strip()
        if not question_id or not question or question_id in output:
            raise ValueError("forced trajectories have invalid question mapping")
        output[question_id] = question
    return output


def _training_labels(
    question_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, int]]:
    labels: dict[str, str] = {}
    excluded = Counter()
    for question in question_results:
        states = sorted(
            (dict(row) for row in question.get("states", [])),
            key=lambda row: int(row["state_index"]),
        )
        for index, state in enumerate(states):
            if state.get("native_next_action") != "SEARCH":
                continue
            state_id = str(state["state_id"])
            if int(state["official_em"]) == 1:
                labels[state_id] = "STOP"
            elif any(int(later["official_em"]) == 1 for later in states[index + 1 :]):
                labels[state_id] = "CONTINUE"
            else:
                excluded["NO_REACHABLE_CORRECT"] += 1
    excluded["LABELED"] = len(labels)
    return labels, dict(excluded)


def build_causal_inputs(
    *,
    state_rows: Sequence[Mapping[str, Any]],
    forced_rows: Sequence[Mapping[str, Any]],
    detokenize: Callable[[list[int]], str],
    question_results: Sequence[Mapping[str, Any]] | None = None,
    decision_states_only: bool = True,
    max_searches: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions = _question_map(forced_rows)
    labels: dict[str, str] | None = None
    exclusion_counts: dict[str, int] = {}
    if question_results is not None:
        labels, exclusion_counts = _training_labels(question_results)

    selected_states = [
        dict(row)
        for row in state_rows
        if row.get("record_type") == "state"
        and row.get("state_role") == "native_prefix"
        and (
            not decision_states_only
            or row.get("native_next_action") == "SEARCH"
        )
    ]
    if labels is not None:
        selected_states = [
            row for row in selected_states if str(row["state_id"]) in labels
        ]

    outputs: list[dict[str, Any]] = []
    block_counts: Counter[int] = Counter()
    prompt_hash_checks: list[bool] = []
    for sequence_index, state in enumerate(selected_states):
        state_id = str(state["state_id"])
        question_id = str(state["question_id"])
        if question_id not in questions:
            raise ValueError(f"missing question for {question_id}")
        rolling_count = int(state["rolling_state_token_count"])
        probe_ids = list(state["probe_input_ids"])
        if rolling_count <= 0 or rolling_count > len(probe_ids):
            raise ValueError(f"{state_id}: invalid rolling-state slice")
        rolling_ids = probe_ids[:rolling_count]
        prompt_hash_checks.append(
            json_sha256(rolling_ids) == str(state["rolling_state_sha256"])
        )
        rolling_text = detokenize(rolling_ids)
        evidence, block_count = extract_evidence(rolling_text)
        block_counts[block_count] += 1
        question = questions[question_id]
        searches_used = int(state["state_index"])
        row = {
            "sequence_index": sequence_index,
            "request_id": state_id,
            "state_id": state_id,
            "question_id": question_id,
            "state_index": searches_used,
            "searches_used": searches_used,
            "remaining_search_budget": max(0, max_searches - searches_used),
            "question": question,
            "evidence": evidence,
            "user_text": render_user_text(
                question=question,
                evidence=evidence,
                searches_used=searches_used,
                max_searches=max_searches,
            ),
            "rolling_state_sha256": state["rolling_state_sha256"],
        }
        if labels is not None:
            row["action_label"] = labels[state_id]
        outputs.append(row)

    input_model_fields = {
        "question",
        "evidence",
        "searches_used",
        "remaining_search_budget",
        "user_text",
    }
    audit = {
        "passed": (
            bool(outputs)
            and len({row["request_id"] for row in outputs}) == len(outputs)
            and all(prompt_hash_checks)
            and all(
                not (set(row) & FORBIDDEN_MODEL_FIELDS)
                for row in (
                    {key: value for key, value in output.items() if key in input_model_fields}
                    for output in outputs
                )
            )
            and (
                labels is None
                or all(row.get("action_label") in ACTION_LABELS for row in outputs)
            )
        ),
        "label_blind": labels is None,
        "input_count": len(outputs),
        "unique_request_ids": len({row["request_id"] for row in outputs}),
        "unique_question_ids": len({row["question_id"] for row in outputs}),
        "action_counts": dict(
            sorted(Counter(row.get("action_label") for row in outputs).items())
        )
        if labels is not None
        else {},
        "exclusion_counts": exclusion_counts,
        "rolling_state_sha256_all_match": all(prompt_hash_checks),
        "information_block_counts": dict(sorted(block_counts.items())),
        "model_facing_fields": sorted(input_model_fields),
        "model_facing_forbidden_fields_absent": True,
        "finalizer_suffix_excluded": True,
    }
    return outputs, audit


def audit_judge_run(
    *,
    inputs: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    expected_protocol_sha256: str,
    expected_inputs_sha256: str,
    expected_adapter_sha256: str | None,
) -> dict[str, Any]:
    starts = [row for row in records if row.get("record_type") == "run_start"]
    results = [row for row in records if row.get("record_type") == "judge_score"]
    errors = [row for row in records if row.get("record_type") == "judge_error"]
    ends = [row for row in records if row.get("record_type") == "run_end"]
    expected_ids = [str(row["request_id"]) for row in inputs]
    actual_ids = [str(row.get("request_id", "")) for row in results]
    scores_finite = all(
        isinstance(row.get("score"), (int, float))
        and math.isfinite(float(row["score"]))
        for row in results
    )
    passed = (
        len(starts) == 1
        and len(ends) == 1
        and ends[0].get("status") == "complete"
        and not errors
        and len(results) == len(inputs)
        and actual_ids == expected_ids
        and len(set(actual_ids)) == len(actual_ids)
        and [row.get("sequence_index") for row in results]
        == list(range(len(inputs)))
        and scores_finite
        and starts[0].get("labels_mounted") is False
        and starts[0].get("gold_fields_received") == 0
        and starts[0].get("protocol_sha256") == expected_protocol_sha256
        and starts[0].get("inputs_sha256") == expected_inputs_sha256
        and starts[0].get("adapter_sha256") == expected_adapter_sha256
    )
    return {
        "passed": passed,
        "label_file_opened": False,
        "expected_count": len(inputs),
        "run_start_count": len(starts),
        "result_count": len(results),
        "error_count": len(errors),
        "run_end_count": len(ends),
        "request_order_exact": actual_ids == expected_ids,
        "sequence_contiguous": [row.get("sequence_index") for row in results]
        == list(range(len(inputs))),
        "scores_finite": scores_finite,
        "labels_mounted": starts[0].get("labels_mounted") if starts else None,
        "gold_fields_received": (
            starts[0].get("gold_fields_received") if starts else None
        ),
    }


def _wilson_upper(errors: int, total: int, z: float = 1.6448536269514722) -> float:
    if total == 0:
        return 1.0
    p = errors / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return min(1.0, (centre + radius) / denominator)


def _bootstrap_delta(
    values: Sequence[tuple[float, float]],
    *,
    seed: int,
    repeats: int = 10_000,
) -> dict[str, float]:
    rng = random.Random(seed)
    deltas = []
    for _ in range(repeats):
        sample = [values[rng.randrange(len(values))] for _ in values]
        deltas.append(
            sum(right - left for left, right in sample) / len(sample)
        )
    deltas.sort()
    return {
        "estimate": sum(right - left for left, right in values) / len(values),
        "ci95_low": deltas[int(0.025 * repeats)],
        "ci95_high": deltas[int(0.975 * repeats)],
    }


def _average_precision(
    values: Sequence[tuple[float, bool]],
) -> float:
    """Return tie-aware average precision for STOP_SAFE as the positive class."""

    positive_count = sum(is_positive for _, is_positive in values)
    if positive_count == 0:
        return 0.0
    groups: dict[float, list[bool]] = defaultdict(list)
    for score, is_positive in values:
        groups[float(score)].append(bool(is_positive))
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    average_precision = 0.0
    for score in sorted(groups, reverse=True):
        labels = groups[score]
        true_positive += sum(labels)
        false_positive += len(labels) - sum(labels)
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


def _bootstrap_average_precision_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    repeats: int = 10_000,
) -> dict[str, float]:
    by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[str(row["question_id"])].append(row)
    question_ids = sorted(by_question)

    def delta(sample: Sequence[str]) -> float:
        sampled_rows = [
            row for question_id in sample for row in by_question[question_id]
        ]
        lora_ap = _average_precision(
            [
                (float(row["lora_score"]), row["truth"] == "STOP")
                for row in sampled_rows
            ]
        )
        base_ap = _average_precision(
            [
                (float(row["base_score"]), row["truth"] == "STOP")
                for row in sampled_rows
            ]
        )
        return lora_ap - base_ap

    estimate = delta(question_ids) if question_ids else 0.0
    if not question_ids:
        return {"estimate": estimate, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    deltas = []
    for _ in range(repeats):
        sample = [
            question_ids[rng.randrange(len(question_ids))]
            for _ in question_ids
        ]
        deltas.append(delta(sample))
    deltas.sort()
    return {
        "estimate": estimate,
        "ci95_low": deltas[int(0.025 * repeats)],
        "ci95_high": deltas[int(0.975 * repeats)],
    }


def evaluate_core_policies(
    *,
    inputs: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    manifest_states: Sequence[Mapping[str, Any]],
    probe_records: Sequence[Mapping[str, Any]],
    forced_rows: Sequence[Mapping[str, Any]],
    base_records: Sequence[Mapping[str, Any]],
    lora_records: Sequence[Mapping[str, Any]],
    threshold: float,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_order = [str(row["question_id"]) for row in inputs]
    label_by_id = {str(row["question_id"]): list(row["targets"]) for row in labels}
    if list(label_by_id) != input_order:
        raise ValueError("label order does not exactly match frozen inputs")
    trajectories = {
        str(row["question_id"]): row
        for row in forced_rows
        if row.get("record_type") == "trajectory"
    }
    state_by_id = {
        str(row["state_id"]): row
        for row in manifest_states
        if row.get("record_type") == "state"
        and row.get("state_role") == "native_prefix"
    }
    probes = {
        str(row["state_id"]): row
        for row in probe_records
        if row.get("record_type") == "probe"
    }
    base = {
        str(row["request_id"]): row
        for row in base_records
        if row.get("record_type") == "judge_score"
    }
    lora = {
        str(row["request_id"]): row
        for row in lora_records
        if row.get("record_type") == "judge_score"
    }
    search_states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in state_by_id.values():
        if state.get("native_next_action") == "SEARCH":
            search_states[str(state["question_id"])].append(dict(state))
    for rows in search_states.values():
        rows.sort(key=lambda row: int(row["state_index"]))

    question_rows: list[dict[str, Any]] = []
    for question_id in input_order:
        targets = label_by_id[question_id]
        trajectory = trajectories[question_id]
        events = list(trajectory.get("candidate_events", []))
        native_prediction = (
            events[0].get("candidate_answer")
            if events
            else trajectory.get("answer")
        )
        native_searches = int(events[0].get("search_calls", 0)) if events else int(
            trajectory.get("search_calls", 0)
        )
        native_em = official_em(native_prediction, targets)
        native_f1 = token_f1(native_prediction, targets)

        scored_states = []
        for state in search_states.get(question_id, []):
            state_id = str(state["state_id"])
            prediction = probes[state_id].get("answer")
            scored_states.append(
                {
                    "question_id": question_id,
                    "state_id": state_id,
                    "state_index": int(state["state_index"]),
                    "prediction": prediction,
                    "official_em": official_em(prediction, targets),
                    "token_f1": token_f1(prediction, targets),
                    "base_score": float(base[state_id]["score"]),
                    "lora_score": float(lora[state_id]["score"]),
                }
            )

        def policy(prefix: str) -> dict[str, Any]:
            stopped = next(
                (
                    row
                    for row in scored_states
                    if row[f"{prefix}_score"] >= threshold
                ),
                None,
            )
            if stopped is None:
                return {
                    "prediction": native_prediction,
                    "official_em": native_em,
                    "token_f1": native_f1,
                    "search_calls": native_searches,
                    "stopped": False,
                    "stop_state_id": None,
                    "stop_state_em": None,
                }
            return {
                "prediction": stopped["prediction"],
                "official_em": stopped["official_em"],
                "token_f1": stopped["token_f1"],
                "search_calls": stopped["state_index"],
                "stopped": True,
                "stop_state_id": stopped["state_id"],
                "stop_state_index": stopped["state_index"],
                "stop_state_em": stopped["official_em"],
            }

        question_rows.append(
            {
                "question_id": question_id,
                "native": {
                    "prediction": native_prediction,
                    "official_em": native_em,
                    "token_f1": native_f1,
                    "search_calls": native_searches,
                },
                "base": policy("base"),
                "lora": policy("lora"),
                "states": scored_states,
            }
        )

    def aggregate(name: str) -> dict[str, Any]:
        pairs = [(row, row[name]) for row in question_rows]
        rows = [policy_row for _, policy_row in pairs]
        stops = [
            (question, policy_row)
            for question, policy_row in pairs
            if policy_row.get("stopped")
        ]
        wrong_stops = [
            (question, policy_row)
            for question, policy_row in stops
            if policy_row.get("stop_state_em") == 0
        ]
        premature_stops = []
        harmful_over_search_rescues = []
        for question, policy_row in stops:
            stop_index = int(policy_row["stop_state_index"])
            future_correct = bool(question["native"]["official_em"]) or any(
                state["state_index"] > stop_index and state["official_em"] == 1
                for state in question["states"]
            )
            if policy_row["official_em"] == 0 and future_correct:
                premature_stops.append((question, policy_row))
            if (
                policy_row["official_em"] == 1
                and question["native"]["official_em"] == 0
            ):
                harmful_over_search_rescues.append((question, policy_row))
        mean_search_calls = sum(int(row["search_calls"]) for row in rows) / len(
            rows
        )
        native_mean_search_calls = sum(
            int(question["native"]["search_calls"]) for question, _ in pairs
        ) / len(pairs)
        return {
            "question_count": len(rows),
            "official_em": sum(int(row["official_em"]) for row in rows) / len(rows),
            "token_f1": sum(float(row["token_f1"]) for row in rows) / len(rows),
            "mean_search_calls": mean_search_calls,
            "mean_search_calls_delta_vs_native": (
                mean_search_calls - native_mean_search_calls
            ),
            "search_reduction_rate_vs_native": (
                (native_mean_search_calls - mean_search_calls)
                / native_mean_search_calls
                if native_mean_search_calls
                else 0.0
            ),
            "stop_count": len(stops),
            "coverage": len(stops) / len(rows),
            "wrong_stop_count": len(wrong_stops),
            "premature_stop_count": len(premature_stops),
            "harmful_over_search_rescue_count": len(
                harmful_over_search_rescues
            ),
            "accepted_risk": len(wrong_stops) / len(stops) if stops else 0.0,
            "accepted_risk_upper_90": _wilson_upper(
                len(wrong_stops), len(stops)
            ),
        }

    native_metrics = {
        "question_count": len(question_rows),
        "official_em": sum(row["native"]["official_em"] for row in question_rows)
        / len(question_rows),
        "token_f1": sum(row["native"]["token_f1"] for row in question_rows)
        / len(question_rows),
        "mean_search_calls": sum(
            row["native"]["search_calls"] for row in question_rows
        )
        / len(question_rows),
    }
    base_metrics = aggregate("base")
    lora_metrics = aggregate("lora")
    paired_em = _bootstrap_delta(
        [
            (float(row["base"]["official_em"]), float(row["lora"]["official_em"]))
            for row in question_rows
        ],
        seed=bootstrap_seed,
    )
    paired_search = _bootstrap_delta(
        [
            (float(row["base"]["search_calls"]), float(row["lora"]["search_calls"]))
            for row in question_rows
        ],
        seed=bootstrap_seed + 1,
    )

    state_rows = [state for row in question_rows for state in row["states"]]
    high_confidence = []
    for row in question_rows:
        states = row["states"]
        for index, state in enumerate(states):
            if state["official_em"] == 1:
                truth = "STOP"
            elif (
                row["native"]["official_em"] == 1
                or any(
                    later["official_em"] == 1
                    for later in states[index + 1 :]
                )
            ):
                truth = "CONTINUE"
            else:
                continue
            high_confidence.append({**state, "truth": truth})

    def state_metrics(prefix: str) -> dict[str, Any]:
        decisions = [
            "STOP" if row[f"{prefix}_score"] >= threshold else "CONTINUE"
            for row in high_confidence
        ]
        stop_tp = sum(
            decision == "STOP" and row["truth"] == "STOP"
            for decision, row in zip(decisions, high_confidence)
        )
        stop_fp = sum(
            decision == "STOP" and row["truth"] == "CONTINUE"
            for decision, row in zip(decisions, high_confidence)
        )
        stop_fn = sum(
            decision == "CONTINUE" and row["truth"] == "STOP"
            for decision, row in zip(decisions, high_confidence)
        )
        stop_tn = sum(
            decision == "CONTINUE" and row["truth"] == "CONTINUE"
            for decision, row in zip(decisions, high_confidence)
        )
        return {
            "eligible_state_count": len(high_confidence),
            "stop_true_positive": stop_tp,
            "stop_false_positive": stop_fp,
            "stop_false_negative": stop_fn,
            "stop_true_negative": stop_tn,
            "stop_precision": stop_tp / (stop_tp + stop_fp)
            if stop_tp + stop_fp
            else 0.0,
            "stop_recall": stop_tp / (stop_tp + stop_fn)
            if stop_tp + stop_fn
            else 0.0,
            "stop_average_precision": _average_precision(
                [
                    (float(row[f"{prefix}_score"]), row["truth"] == "STOP")
                    for row in high_confidence
                ]
            ),
        }

    report = {
        "schema_version": 1,
        "stage": "T1 first LoRA / Core final test 200",
        "threshold": threshold,
        "claim_status": "confirmatory_core_200_single_frozen_threshold",
        "question_count": len(question_rows),
        "state_count": len(state_rows),
        "high_confidence_state_count": len(high_confidence),
        "native": native_metrics,
        "base": base_metrics,
        "lora": lora_metrics,
        "state_level": {
            "base": state_metrics("base"),
            "lora": state_metrics("lora"),
            "paired_lora_minus_base": {
                "stop_average_precision": _bootstrap_average_precision_delta(
                    high_confidence,
                    seed=bootstrap_seed + 2,
                )
            },
        },
        "paired_lora_minus_base": {
            "official_em": paired_em,
            "search_calls": paired_search,
        },
        "improvement": {
            "em_higher": lora_metrics["official_em"] > base_metrics["official_em"],
            "em_non_decreasing": lora_metrics["official_em"]
            >= base_metrics["official_em"],
            "accepted_risk_lower": lora_metrics["accepted_risk"]
            < base_metrics["accepted_risk"],
            "mean_searches_lower": lora_metrics["mean_search_calls"]
            < base_metrics["mean_search_calls"],
            "stop_precision_higher": state_metrics("lora")["stop_precision"]
            > state_metrics("base")["stop_precision"],
            "stop_recall_higher": state_metrics("lora")["stop_recall"]
            > state_metrics("base")["stop_recall"],
        },
    }
    return report, question_rows
