#!/usr/bin/env python3
"""Freeze per-model STOP thresholds on grouped validation only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def point_metrics(
    scores: Sequence[float],
    truths: Sequence[bool],
    threshold: float,
) -> dict[str, Any]:
    predictions = [score >= threshold for score in scores]
    tp = sum(pred and truth for pred, truth in zip(predictions, truths))
    fp = sum(pred and not truth for pred, truth in zip(predictions, truths))
    fn = sum(not pred and truth for pred, truth in zip(predictions, truths))
    return {
        "threshold": threshold,
        "stop_count": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def select_threshold(
    scores: Sequence[float],
    truths: Sequence[bool],
    *,
    minimum_precision: float,
    minimum_stop_count: int,
) -> dict[str, Any]:
    if len(scores) != len(truths) or not scores:
        raise ValueError("scores/truths are empty or differ in length")
    candidates = [math.inf, *sorted(set(scores), reverse=True)]
    points = [
        point_metrics(scores, truths, threshold) for threshold in candidates
    ]
    feasible = [
        row
        for row in points
        if row["stop_count"] >= minimum_stop_count
        and row["precision"] >= minimum_precision
    ]
    if not feasible:
        return {
            **point_metrics(scores, truths, math.inf),
            "feasible": False,
            "fallback": "never_stop",
            "minimum_precision": minimum_precision,
            "minimum_stop_count": minimum_stop_count,
        }
    best = max(
        feasible,
        key=lambda row: (
            row["recall"],
            row["precision"],
            row["threshold"],
        ),
    )
    return {
        **best,
        "feasible": True,
        "fallback": None,
        "minimum_precision": minimum_precision,
        "minimum_stop_count": minimum_stop_count,
    }


def select_all(
    state_rows: Sequence[Mapping[str, Any]],
    *,
    model_names: Sequence[str],
    minimum_precision: float,
    minimum_stop_count: int,
) -> dict[str, Any]:
    truths = [bool(row["safe_early_stop"]) for row in state_rows]
    if not truths or not any(truths) or all(truths):
        raise ValueError("grouped validation must contain both truth classes")
    output = {}
    for name in model_names:
        scores = [float(row["scores"][name]) for row in state_rows]
        output[name] = select_threshold(
            scores,
            truths,
            minimum_precision=minimum_precision,
            minimum_stop_count=minimum_stop_count,
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-states", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--minimum-stop-count", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        raise FileExistsError("output or audit already exists")
    rows = read_jsonl(args.validation_states)
    selected = select_all(
        rows,
        model_names=args.model,
        minimum_precision=args.minimum_precision,
        minimum_stop_count=args.minimum_stop_count,
    )
    thresholds = {
        name: details["threshold"] for name, details in selected.items()
    }
    # JSON has no portable infinity literal.  "never_stop" is represented by
    # max observed score + 1, which remains finite and deterministic.
    for name, details in selected.items():
        if not details["feasible"]:
            maximum = max(float(row["scores"][name]) for row in rows)
            thresholds[name] = maximum + 1.0
            details["threshold"] = thresholds[name]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report = {
        "artifact_id": "searchr1-s2g-scaleup-threshold-freeze-v1",
        "frozen_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "selection_split": "grouped_validation_100",
        "selection_rule": (
            "maximize SAFE_EARLY_STOP recall subject to empirical "
            "precision >= 0.90 and at least 10 predicted STOP states; "
            "tie by precision then higher threshold; otherwise never STOP"
        ),
        "minimum_precision": args.minimum_precision,
        "minimum_stop_count": args.minimum_stop_count,
        "validation_state_count": len(rows),
        "positive_state_count": sum(
            bool(row["safe_early_stop"]) for row in rows
        ),
        "models": selected,
        "thresholds": thresholds,
        "sources": {
            "validation_states_path": str(args.validation_states),
            "validation_states_sha256": sha256_file(
                args.validation_states
            )
        },
        "outputs": {
            "thresholds_path": str(args.output),
            "thresholds_sha256": sha256_file(args.output),
        },
        "final_dev1000_labels_seen": False,
    }
    args.audit.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
