from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_s2g_scaleup",
    ROOT / "scripts/evaluate_s2g_scaleup.py",
)
assert SPEC is not None and SPEC.loader is not None
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


def test_precision_recall_and_average_precision():
    scores = [0.9, 0.8, 0.1, -0.2]
    truths = [True, False, True, False]
    point = evaluate.threshold_metrics(scores, truths, 0.5)
    assert point["true_positive"] == 1
    assert point["false_positive"] == 1
    assert point["stop_precision"] == 0.5
    assert point["stop_recall"] == 0.5
    assert evaluate.average_precision(list(zip(scores, truths))) == (
        1.0 + 2 / 3
    ) / 2


def test_recall_at_precision_selects_highest_recall():
    result = evaluate.recall_at_precision(
        [0.9, 0.8, 0.7, 0.6],
        [True, True, False, True],
        0.9,
    )
    assert result["achievable"]
    assert result["precision"] == 1.0
    assert result["recall"] == 2 / 3


def test_endpoint_policy_uses_first_predicted_stop():
    inputs = [{"question_id": "q1", "question": "Q?"}]
    labels = [{"question_id": "q1", "answer": "good"}]
    native = [
        {"record_type": "run_start"},
        {
            "record_type": "trajectory",
            "question_id": "q1",
            "answer": "good",
            "search_calls": 2,
        },
        {"record_type": "run_end"},
    ]
    judge_states = [
        {"request_id": "q1:s0"},
        {"request_id": "q1:s1"},
        {"request_id": "q1:s2"},
    ]
    probe_states = [
        {"state_id": "q1:s0", "question_id": "q1", "state_index": 0},
        {"state_id": "q1:s1", "question_id": "q1", "state_index": 1},
    ]
    probes = [
        {"record_type": "run_start"},
        {"record_type": "probe", "state_id": "q1:s0", "answer": "bad"},
        {"record_type": "probe", "state_id": "q1:s1", "answer": "good"},
        {"record_type": "run_end"},
    ]
    base = [
        {"record_type": "run_start"},
        {
            "record_type": "judge_score",
            "request_id": "q1:s0",
            "score": -1,
            "decision": "CONTINUE",
        },
        {
            "record_type": "judge_score",
            "request_id": "q1:s1",
            "score": 1,
            "decision": "STOP",
        },
        {
            "record_type": "judge_score",
            "request_id": "q1:s2",
            "score": 1,
            "decision": "STOP",
        },
        {"record_type": "run_end"},
    ]
    report, questions, states = evaluate.evaluate(
        inputs=inputs,
        labels=labels,
        native_run=native,
        judge_states=judge_states,
        probe_states=probe_states,
        probe_run=probes,
        judge_runs={"base": base},
        thresholds={"base": 0.0},
        bootstrap_seed=1,
    )
    assert report["state_level"]["base"]["stop_precision"] == 1.0
    assert report["state_level"]["base"]["stop_recall"] == 1.0
    assert report["system_level"]["base"]["official_em"] == 1.0
    assert report["system_level"]["base"]["mean_search_calls"] == 1.0
    assert questions[0]["policies"]["base"]["safe_stop"]
    assert len(states) == 2


def test_pairwise_comparisons_use_structured_base_despite_json_key_order():
    inputs = [{"question_id": "q1", "question": "Q?"}]
    labels = [{"question_id": "q1", "answer": "good"}]
    native = [
        {
            "record_type": "trajectory",
            "question_id": "q1",
            "answer": "good",
            "search_calls": 2,
        }
    ]
    judge_states = [
        {"request_id": "q1:s0"},
        {"request_id": "q1:s1"},
    ]
    probe_states = [
        {"state_id": "q1:s0", "question_id": "q1", "state_index": 0},
        {"state_id": "q1:s1", "question_id": "q1", "state_index": 1},
    ]
    probes = [
        {
            "record_type": "probe",
            "state_id": "q1:s0",
            "answer": "bad",
        },
        {
            "record_type": "probe",
            "state_id": "q1:s1",
            "answer": "good",
        },
    ]

    def judge_run(scores):
        return [
            {
                "record_type": "judge_score",
                "request_id": state["request_id"],
                "score": score,
                "decision": "STOP" if score >= 0 else "CONTINUE",
            }
            for state, score in zip(judge_states, scores)
        ]

    # Threshold JSON is serialized with sort_keys=True, so the insertion order
    # after loading is alphabetical and expanded_s2g_lora comes first.
    thresholds = {
        "expanded_s2g_lora": 0.0,
        "old_s2g_lora": 0.0,
        "structured_base": 0.0,
    }
    report, _, _ = evaluate.evaluate(
        inputs=inputs,
        labels=labels,
        native_run=native,
        judge_states=judge_states,
        probe_states=probe_states,
        probe_run=probes,
        judge_runs={
            "expanded_s2g_lora": judge_run([-1.0, 1.0]),
            "old_s2g_lora": judge_run([1.0, -1.0]),
            "structured_base": judge_run([-0.5, 0.5]),
        },
        thresholds=thresholds,
        bootstrap_seed=1,
    )
    comparisons = report["paired_comparisons"]
    assert "expanded_s2g_lora_minus_structured_base" in comparisons
    assert "old_s2g_lora_minus_structured_base" in comparisons


def test_endpoint_policy_masks_parse_failure_positive_margin():
    inputs = [{"question_id": "q1", "question": "Q?"}]
    labels = [{"question_id": "q1", "answer": "good"}]
    native = [
        {
            "record_type": "trajectory",
            "question_id": "q1",
            "answer": "good",
            "search_calls": 1,
        }
    ]
    judge_states = [{"request_id": "q1:s0"}]
    probe_states = [
        {"state_id": "q1:s0", "question_id": "q1", "state_index": 0}
    ]
    probes = [
        {"record_type": "probe", "state_id": "q1:s0", "answer": "bad"}
    ]
    malformed = [
        {
            "record_type": "judge_score",
            "request_id": "q1:s0",
            "parsed": None,
            "parse_error": "json_decode",
            "score": 4.0,
            "decision": "CONTINUE",
        }
    ]
    report, questions, states = evaluate.evaluate(
        inputs=inputs,
        labels=labels,
        native_run=native,
        judge_states=judge_states,
        probe_states=probe_states,
        probe_run=probes,
        judge_runs={"base": malformed},
        thresholds={"base": 0.0},
        bootstrap_seed=1,
    )
    assert states[0]["raw_scores"]["base"] == 4.0
    assert states[0]["scores"]["base"] < 0
    assert report["masked_parse_failure_counts"] == {"base": 1}
    assert report["system_level"]["base"]["early_stop_count"] == 0
    assert questions[0]["policies"]["base"]["official_em"] == 1
