#!/usr/bin/env python3
"""Evaluate scale-up S2G Judges on SAFE_EARLY_STOP and endpoint policy."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from kstar.searchr1_v02_causal_judge import (
    effective_judge_score,
    judge_score_is_structured_parse_failure,
)
from kstar.searchr1_v02_eval import official_em
from kstar.searchr1_v02_gateh0_eval import token_f1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
                + "\n"
            )


def average_precision(pairs: Sequence[tuple[float, bool]]) -> float:
    positives = sum(label for _, label in pairs)
    if positives == 0:
        return 0.0
    ordered = sorted(
        enumerate(pairs),
        key=lambda item: (-item[1][0], item[0]),
    )
    true_positives = 0
    value = 0.0
    for rank, (_, (_, label)) in enumerate(ordered, 1):
        if label:
            true_positives += 1
            value += true_positives / rank
    return value / positives


def threshold_metrics(
    scores: Sequence[float],
    truths: Sequence[bool],
    threshold: float,
) -> dict[str, Any]:
    if len(scores) != len(truths) or not scores:
        raise ValueError("scores/truths are empty or differ in length")
    predictions = [score >= threshold for score in scores]
    tp = sum(pred and truth for pred, truth in zip(predictions, truths))
    fp = sum(pred and not truth for pred, truth in zip(predictions, truths))
    fn = sum(not pred and truth for pred, truth in zip(predictions, truths))
    tn = sum(
        not pred and not truth for pred, truth in zip(predictions, truths)
    )
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "stop_precision": tp / (tp + fp) if tp + fp else 0.0,
        "stop_recall": tp / (tp + fn) if tp + fn else 0.0,
        "stop_rate": (tp + fp) / len(scores),
        "unsafe_stop_rate": fp / (tp + fp) if tp + fp else 0.0,
    }


def recall_at_precision(
    scores: Sequence[float],
    truths: Sequence[bool],
    minimum_precision: float,
) -> dict[str, Any]:
    candidates = [math.inf, *sorted(set(scores), reverse=True)]
    eligible = []
    for threshold in candidates:
        metrics = threshold_metrics(scores, truths, threshold)
        if (
            metrics["true_positive"] + metrics["false_positive"] > 0
            and metrics["stop_precision"] >= minimum_precision
        ):
            eligible.append(metrics)
    if not eligible:
        return {
            "minimum_precision": minimum_precision,
            "achievable": False,
            "recall": 0.0,
            "precision": None,
            "threshold": None,
        }
    best = max(
        eligible,
        key=lambda row: (
            row["stop_recall"],
            row["stop_precision"],
            row["threshold"],
        ),
    )
    return {
        "minimum_precision": minimum_precision,
        "achievable": True,
        "recall": best["stop_recall"],
        "precision": best["stop_precision"],
        "threshold": best["threshold"],
    }


def calibration_metrics(
    scores: Sequence[float],
    truths: Sequence[bool],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    probabilities = [
        1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, score))))
        for score in scores
    ]
    brier = sum(
        (probability - float(truth)) ** 2
        for probability, truth in zip(probabilities, truths)
    ) / len(probabilities)
    ece = 0.0
    rows = []
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        indices = [
            index
            for index, probability in enumerate(probabilities)
            if probability >= low
            and (
                probability < high
                or (bin_index == bins - 1 and probability <= high)
            )
        ]
        if not indices:
            continue
        confidence = sum(probabilities[index] for index in indices) / len(
            indices
        )
        accuracy = sum(truths[index] for index in indices) / len(indices)
        ece += len(indices) / len(probabilities) * abs(confidence - accuracy)
        rows.append(
            {
                "low": low,
                "high": high,
                "count": len(indices),
                "mean_probability": confidence,
                "observed_rate": accuracy,
            }
        )
    return {"brier": brier, "ece_10": ece, "bins": rows}


def paired_bootstrap(
    values: Sequence[tuple[float, float]],
    *,
    seed: int,
    repeats: int = 10_000,
) -> dict[str, float]:
    if not values:
        raise ValueError("paired values are empty")
    deltas = [right - left for left, right in values]
    rng = random.Random(seed)
    samples = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas)
        / len(deltas)
        for _ in range(repeats)
    ]
    samples.sort()
    return {
        "estimate": sum(deltas) / len(deltas),
        "ci95_low": samples[int(0.025 * repeats)],
        "ci95_high": samples[int(0.975 * repeats)],
    }


def grouped_ap_bootstrap_delta(
    state_rows: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
    seed: int,
    repeats: int = 10_000,
) -> dict[str, float]:
    by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in state_rows:
        by_question[str(row["question_id"])].append(row)
    question_ids = list(by_question)
    if not question_ids:
        raise ValueError("state rows are empty")

    def delta(sample: Sequence[str]) -> float:
        rows = [row for question_id in sample for row in by_question[question_id]]
        truths = [bool(row["safe_early_stop"]) for row in rows]
        return average_precision(
            [(float(row["scores"][right]), truth) for row, truth in zip(rows, truths)]
        ) - average_precision(
            [(float(row["scores"][left]), truth) for row, truth in zip(rows, truths)]
        )

    estimate = delta(question_ids)
    rng = random.Random(seed)
    samples = [
        delta(
            [
                question_ids[rng.randrange(len(question_ids))]
                for _ in question_ids
            ]
        )
        for _ in range(repeats)
    ]
    samples.sort()
    return {
        "estimate": estimate,
        "ci95_low": samples[int(0.025 * repeats)],
        "ci95_high": samples[int(0.975 * repeats)],
    }


def evaluate(
    *,
    inputs: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    native_run: Sequence[Mapping[str, Any]],
    judge_states: Sequence[Mapping[str, Any]],
    probe_states: Sequence[Mapping[str, Any]],
    probe_run: Sequence[Mapping[str, Any]],
    judge_runs: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: Mapping[str, float],
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    input_ids = [str(row["question_id"]) for row in inputs]
    label_ids = [str(row["question_id"]) for row in labels]
    if input_ids != label_ids or len(set(input_ids)) != len(input_ids):
        raise ValueError("input/label order differs or IDs duplicate")
    answers = {
        str(row["question_id"]): row.get("answer") for row in labels
    }
    trajectories = [
        row for row in native_run if row.get("record_type") == "trajectory"
    ]
    if [str(row["question_id"]) for row in trajectories] != input_ids:
        raise ValueError("Native trajectory order differs from inputs")
    state_ids = [str(row["request_id"]) for row in judge_states]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("Judge state IDs are duplicated")
    probe_state_ids = [str(row["state_id"]) for row in probe_states]
    probes = [
        row for row in probe_run if row.get("record_type") == "probe"
    ]
    if [str(row["state_id"]) for row in probes] != probe_state_ids:
        raise ValueError("answer-probe order differs from frozen states")
    probe_by_id = {str(row["state_id"]): row for row in probes}
    probe_meta = {str(row["state_id"]): row for row in probe_states}

    scores_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for name, records in judge_runs.items():
        scores = [
            row for row in records if row.get("record_type") == "judge_score"
        ]
        if [str(row["request_id"]) for row in scores] != state_ids:
            raise ValueError(f"{name}: Judge score order differs from states")
        if any(row.get("score") is None for row in scores):
            raise ValueError(f"{name}: missing decision score")
        scores_by_model[name] = {
            str(row["request_id"]): dict(row) for row in scores
        }
    if set(scores_by_model) != set(thresholds):
        raise ValueError("threshold/model names differ")

    states_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    state_rows: list[dict[str, Any]] = []
    for state_id in probe_state_ids:
        state = probe_meta[state_id]
        question_id = str(state["question_id"])
        prediction = probe_by_id[state_id].get("answer")
        truth = bool(official_em(prediction, answers[question_id]))
        row = {
            "question_id": question_id,
            "state_id": state_id,
            "state_index": int(state["state_index"]),
            "answer_probe_prediction": prediction,
            "safe_early_stop": truth,
            "scores": {
                name: effective_judge_score(
                    scores_by_model[name][state_id]
                )
                for name in scores_by_model
            },
            "raw_scores": {
                name: float(scores_by_model[name][state_id]["score"])
                for name in scores_by_model
            },
            "structured_parse_valid": {
                name: not judge_score_is_structured_parse_failure(
                    scores_by_model[name][state_id]
                )
                for name in scores_by_model
            },
            "greedy_decisions": {
                name: str(
                    scores_by_model[name][state_id].get("decision")
                )
                for name in scores_by_model
            },
        }
        states_by_question[question_id].append(row)
        state_rows.append(row)
    for rows in states_by_question.values():
        rows.sort(key=lambda row: int(row["state_index"]))

    question_rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        question_id = str(trajectory["question_id"])
        gold = answers[question_id]
        native_prediction = trajectory.get("answer")
        native_searches = int(trajectory["search_calls"])
        native = {
            "prediction": native_prediction,
            "official_em": official_em(native_prediction, gold),
            "token_f1": token_f1(native_prediction, [str(gold)]),
            "search_calls": native_searches,
        }
        states = states_by_question.get(question_id, [])
        safe_states = [
            row for row in states if row["safe_early_stop"]
        ]
        earliest_safe = (
            int(safe_states[0]["state_index"]) if safe_states else None
        )
        policies: dict[str, Any] = {}
        for name, threshold in thresholds.items():
            stopped = next(
                (
                    row
                    for row in states
                    if row["scores"][name] >= float(threshold)
                ),
                None,
            )
            if stopped is None:
                policy = {
                    **native,
                    "stopped_early": False,
                    "stop_state_id": None,
                    "stop_state_index": None,
                    "safe_stop": False,
                }
            else:
                prediction = stopped["answer_probe_prediction"]
                policy = {
                    "prediction": prediction,
                    "official_em": official_em(prediction, gold),
                    "token_f1": token_f1(prediction, [str(gold)]),
                    "search_calls": int(stopped["state_index"]),
                    "stopped_early": True,
                    "stop_state_id": stopped["state_id"],
                    "stop_state_index": int(stopped["state_index"]),
                    "safe_stop": bool(stopped["safe_early_stop"]),
                }
            policy["searches_saved"] = (
                native_searches - int(policy["search_calls"])
            )
            policies[name] = policy
        question_rows.append(
            {
                "question_id": question_id,
                "native": native,
                "has_safe_early_stop_opportunity": bool(safe_states),
                "earliest_safe_stop_index": earliest_safe,
                "policies": policies,
            }
        )

    truths = [bool(row["safe_early_stop"]) for row in state_rows]
    state_metrics: dict[str, Any] = {}
    for name, threshold in thresholds.items():
        scores = [float(row["scores"][name]) for row in state_rows]
        state_metrics[name] = {
            **threshold_metrics(scores, truths, float(threshold)),
            "eligible_state_count": len(state_rows),
            "positive_state_count": sum(truths),
            "safe_early_stop_prevalence": sum(truths) / len(truths),
            "stop_average_precision": average_precision(
                list(zip(scores, truths))
            ),
            "recall_at_precision_90": recall_at_precision(
                scores, truths, 0.90
            ),
            "recall_at_precision_95": recall_at_precision(
                scores, truths, 0.95
            ),
            "calibration": calibration_metrics(scores, truths),
        }

    def aggregate(name: str) -> dict[str, Any]:
        policy_rows = [row["policies"][name] for row in question_rows]
        opportunities = [
            row
            for row in question_rows
            if row["has_safe_early_stop_opportunity"]
        ]
        safe_stops = [
            row
            for row in question_rows
            if row["policies"][name]["safe_stop"]
        ]
        unsafe_stops = [
            row
            for row in question_rows
            if row["policies"][name]["stopped_early"]
            and not row["policies"][name]["safe_stop"]
        ]
        safe_delays = [
            int(row["policies"][name]["stop_state_index"])
            - int(row["earliest_safe_stop_index"])
            for row in safe_stops
        ]
        return {
            "question_count": len(question_rows),
            "official_em": sum(
                int(row["official_em"]) for row in policy_rows
            )
            / len(policy_rows),
            "token_f1": sum(
                float(row["token_f1"]) for row in policy_rows
            )
            / len(policy_rows),
            "mean_search_calls": sum(
                int(row["search_calls"]) for row in policy_rows
            )
            / len(policy_rows),
            "mean_searches_saved": sum(
                int(row["searches_saved"]) for row in policy_rows
            )
            / len(policy_rows),
            "early_stop_count": sum(
                bool(row["stopped_early"]) for row in policy_rows
            ),
            "safe_early_stop_count": len(safe_stops),
            "unsafe_early_stop_count": len(unsafe_stops),
            "accepted_risk": (
                len(unsafe_stops)
                / sum(bool(row["stopped_early"]) for row in policy_rows)
                if any(bool(row["stopped_early"]) for row in policy_rows)
                else 0.0
            ),
            "opportunity_question_count": len(opportunities),
            "early_stop_opportunity_recall": (
                len(safe_stops) / len(opportunities)
                if opportunities
                else 0.0
            ),
            "mean_first_safe_stop_delay_on_safe_stops": (
                sum(safe_delays) / len(safe_delays)
                if safe_delays
                else None
            ),
            "correct_to_wrong_count": sum(
                row["native"]["official_em"] == 1
                and row["policies"][name]["official_em"] == 0
                for row in question_rows
            ),
            "wrong_to_correct_count": sum(
                row["native"]["official_em"] == 0
                and row["policies"][name]["official_em"] == 1
                for row in question_rows
            ),
        }

    native_metrics = {
        "question_count": len(question_rows),
        "official_em": sum(
            row["native"]["official_em"] for row in question_rows
        )
        / len(question_rows),
        "token_f1": sum(
            row["native"]["token_f1"] for row in question_rows
        )
        / len(question_rows),
        "mean_search_calls": sum(
            row["native"]["search_calls"] for row in question_rows
        )
        / len(question_rows),
    }
    system_metrics = {name: aggregate(name) for name in thresholds}
    comparisons: dict[str, Any] = {}
    names = list(thresholds)
    for offset, name in enumerate(names):
        comparisons[f"{name}_minus_native"] = {
            "official_em": paired_bootstrap(
                [
                    (
                        float(row["native"]["official_em"]),
                        float(row["policies"][name]["official_em"]),
                    )
                    for row in question_rows
                ],
                seed=bootstrap_seed + 200 + offset,
            ),
            "search_calls": paired_bootstrap(
                [
                    (
                        float(row["native"]["search_calls"]),
                        float(row["policies"][name]["search_calls"]),
                    )
                    for row in question_rows
                ],
                seed=bootstrap_seed + 300 + offset,
            ),
            "native_correct_policy_wrong": sum(
                row["native"]["official_em"] == 1
                and row["policies"][name]["official_em"] == 0
                for row in question_rows
            ),
            "native_wrong_policy_correct": sum(
                row["native"]["official_em"] == 0
                and row["policies"][name]["official_em"] == 1
                for row in question_rows
            ),
        }
    if len(names) >= 2:
        reference = (
            "structured_base"
            if "structured_base" in names
            else names[0]
        )
        comparison_names = [
            name for name in names if name != reference
        ]
        for offset, name in enumerate(comparison_names, 1):
            comparisons[f"{name}_minus_{reference}"] = {
                "official_em": paired_bootstrap(
                    [
                        (
                            float(row["policies"][reference]["official_em"]),
                            float(row["policies"][name]["official_em"]),
                        )
                        for row in question_rows
                    ],
                    seed=bootstrap_seed + offset,
                ),
                "search_calls": paired_bootstrap(
                    [
                        (
                            float(row["policies"][reference]["search_calls"]),
                            float(row["policies"][name]["search_calls"]),
                        )
                        for row in question_rows
                    ],
                    seed=bootstrap_seed + 100 + offset,
                ),
                "stop_average_precision": grouped_ap_bootstrap_delta(
                    state_rows,
                    left=reference,
                    right=name,
                    seed=bootstrap_seed + 400 + offset,
                ),
            }
    report = {
        "schema_version": 1,
        "truth": (
            "SAFE_EARLY_STOP = answer-only official EM correct on a "
            "Native state whose next action was SEARCH"
        ),
        "question_count": len(question_rows),
        "eligible_state_count": len(state_rows),
        "positive_state_count": sum(truths),
        "native": native_metrics,
        "state_level": state_metrics,
        "system_level": system_metrics,
        "paired_comparisons": comparisons,
        "thresholds": dict(thresholds),
        "score_policy": (
            "parseable structured Judge rows retain raw margin; parsed=null "
            "rows use deterministic CONTINUE-only effective score"
        ),
        "masked_parse_failure_counts": {
            name: sum(
                judge_score_is_structured_parse_failure(row)
                for row in records.values()
            )
            for name, records in scores_by_model.items()
        },
    }
    return report, question_rows, state_rows


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH: {value}")
        name, path = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"invalid or duplicate name: {name}")
        output[name] = Path(path)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--native-run", required=True, type=Path)
    parser.add_argument("--judge-states", required=True, type=Path)
    parser.add_argument("--probe-states", required=True, type=Path)
    parser.add_argument("--probe-run", required=True, type=Path)
    parser.add_argument(
        "--judge-run", action="append", required=True
    )
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--question-output", required=True, type=Path)
    parser.add_argument("--state-output", required=True, type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    args = parser.parse_args()
    if any(
        path.exists()
        for path in (
            args.report,
            args.question_output,
            args.state_output,
        )
    ):
        raise FileExistsError("one or more outputs already exist")
    judge_paths = parse_named_paths(args.judge_run)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    report, questions, states = evaluate(
        inputs=read_jsonl(args.inputs),
        labels=read_jsonl(args.labels),
        native_run=read_jsonl(args.native_run),
        judge_states=read_jsonl(args.judge_states),
        probe_states=read_jsonl(args.probe_states),
        probe_run=read_jsonl(args.probe_run),
        judge_runs={
            name: read_jsonl(path) for name, path in judge_paths.items()
        },
        thresholds={
            str(key): float(value) for key, value in thresholds.items()
        },
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json(args.report, report)
    write_jsonl(args.question_output, questions)
    write_jsonl(args.state_output, states)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
