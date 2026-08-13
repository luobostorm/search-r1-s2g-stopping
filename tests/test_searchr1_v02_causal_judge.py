from __future__ import annotations

import json

import pytest

from kstar.searchr1_v02_causal_judge import (
    JUDGE_PARSE_FAILURE_EFFECTIVE_SCORE,
    audit_judge_run,
    build_causal_inputs,
    effective_judge_score,
    evaluate_core_policies,
    json_sha256,
    judge_score_is_structured_parse_failure,
    materialize_core_final_split,
)


def test_core_split_is_label_blind_deterministic_and_disjoint():
    rows = [
        {
            "question_id": f"hotpotqa:{index}",
            "data_source": "hotpotqa",
            "source_row_index": index,
            "question": f"q{index}",
            "initial_prompt": f"Question: q{index}",
        }
        for index in range(1000)
    ]
    left = materialize_core_final_split(rows, seed="frozen")
    right = materialize_core_final_split(rows, seed="frozen")
    core, reserve, manifest = left
    assert [row["question_id"] for row in core] == [
        row["question_id"] for row in right[0]
    ]
    assert len(core) == 200
    assert len(reserve) == 800
    assert manifest["labels_read_or_written"] is False
    assert manifest["core_reserve_overlap"] == 0


def _state(question_id: str, state_index: int, next_action: str = "SEARCH"):
    rolling = [10 + state_index, 20 + state_index]
    return {
        "record_type": "state",
        "state_id": f"{question_id}:search-state:{state_index}",
        "question_id": question_id,
        "state_index": state_index,
        "state_role": "native_prefix",
        "native_next_action": next_action,
        "rolling_state_token_count": len(rolling),
        "probe_input_ids": rolling + [99],
        "rolling_state_sha256": json_sha256(rolling),
    }


def test_build_training_inputs_uses_only_high_confidence_states():
    question_id = "hotpotqa:1"
    states = [_state(question_id, index) for index in range(4)]
    forced = [
        {
            "record_type": "trajectory",
            "question_id": question_id,
            "question": "Who?",
        }
    ]
    results = [
        {
            "question_id": question_id,
            "states": [
                {
                    "state_id": states[0]["state_id"],
                    "state_index": 0,
                    "native_next_action": "SEARCH",
                    "official_em": 0,
                },
                {
                    "state_id": states[1]["state_id"],
                    "state_index": 1,
                    "native_next_action": "SEARCH",
                    "official_em": 1,
                },
                {
                    "state_id": states[2]["state_id"],
                    "state_index": 2,
                    "native_next_action": "SEARCH",
                    "official_em": 0,
                },
                {
                    "state_id": states[3]["state_id"],
                    "state_index": 3,
                    "native_next_action": "SEARCH",
                    "official_em": 0,
                },
            ],
        }
    ]
    rows, audit = build_causal_inputs(
        state_rows=states,
        forced_rows=forced,
        detokenize=lambda _: (
            "<|im_start|>assistant\n"
            "<information>retrieved fact</information><|im_end|>"
        ),
        question_results=results,
    )
    assert [row["action_label"] for row in rows] == ["CONTINUE", "STOP"]
    assert audit["exclusion_counts"]["NO_REACHABLE_CORRECT"] == 2
    assert audit["passed"]
    assert all("official_em" not in row["user_text"] for row in rows)


def test_audit_judge_run_requires_exact_order_and_label_blind_start():
    inputs = [{"request_id": "a"}, {"request_id": "b"}]
    records = [
        {
            "record_type": "run_start",
            "labels_mounted": False,
            "gold_fields_received": 0,
            "protocol_sha256": "p",
            "inputs_sha256": "i",
            "adapter_sha256": None,
        },
        {"record_type": "judge_score", "sequence_index": 0, "request_id": "a", "score": 0.1},
        {"record_type": "judge_score", "sequence_index": 1, "request_id": "b", "score": -0.1},
        {"record_type": "run_end", "status": "complete"},
    ]
    audit = audit_judge_run(
        inputs=inputs,
        records=records,
        expected_protocol_sha256="p",
        expected_inputs_sha256="i",
        expected_adapter_sha256=None,
    )
    assert audit["passed"]
    records[2]["request_id"] = "a"
    assert not audit_judge_run(
        inputs=inputs,
        records=records,
        expected_protocol_sha256="p",
        expected_inputs_sha256="i",
        expected_adapter_sha256=None,
    )["passed"]


def test_structured_parse_failure_masks_positive_raw_margin():
    malformed = {
        "parsed": None,
        "parse_error": "json_decode",
        "score": 2.5,
        "decision": "CONTINUE",
    }
    assert judge_score_is_structured_parse_failure(malformed)
    assert effective_judge_score(malformed) == (
        JUDGE_PARSE_FAILURE_EFFECTIVE_SCORE
    )
    assert effective_judge_score(
        {
            "parsed": {"sufficient": True, "gap_items": []},
            "score": 2.5,
            "decision": "STOP",
        }
    ) == 2.5
    assert effective_judge_score({"score": 1.25}) == 1.25


def test_evaluate_core_policies_prefers_earliest_threshold_stop():
    inputs = [{"question_id": "q1"}, {"question_id": "q2"}]
    labels = [
        {"question_id": "q1", "targets": ["alpha"]},
        {"question_id": "q2", "targets": ["beta"]},
    ]
    states = [
        {
            **_state("q1", 0),
            "state_id": "q1:s0",
        },
        {
            **_state("q2", 0),
            "state_id": "q2:s0",
        },
    ]
    probes = [
        {"record_type": "probe", "state_id": "q1:s0", "answer": "alpha"},
        {"record_type": "probe", "state_id": "q2:s0", "answer": "wrong"},
    ]
    forced = [
        {
            "record_type": "trajectory",
            "question_id": "q1",
            "candidate_events": [{"candidate_answer": "alpha", "search_calls": 1}],
        },
        {
            "record_type": "trajectory",
            "question_id": "q2",
            "candidate_events": [{"candidate_answer": "beta", "search_calls": 1}],
        },
    ]
    base = [
        {"record_type": "judge_score", "request_id": "q1:s0", "score": -1.0},
        {"record_type": "judge_score", "request_id": "q2:s0", "score": 1.0},
    ]
    lora = [
        {"record_type": "judge_score", "request_id": "q1:s0", "score": 1.0},
        {"record_type": "judge_score", "request_id": "q2:s0", "score": -1.0},
    ]
    report, rows = evaluate_core_policies(
        inputs=inputs,
        labels=labels,
        manifest_states=states,
        probe_records=probes,
        forced_rows=forced,
        base_records=base,
        lora_records=lora,
        threshold=0.0,
        bootstrap_seed=3,
    )
    assert report["base"]["official_em"] == pytest.approx(0.5)
    assert report["lora"]["official_em"] == pytest.approx(1.0)
    assert report["base"]["accepted_risk"] == pytest.approx(1.0)
    assert report["base"]["premature_stop_count"] == 1
    assert report["lora"]["accepted_risk"] == pytest.approx(0.0)
    assert report["lora"]["search_reduction_rate_vs_native"] == pytest.approx(0.5)
    assert report["state_level"]["base"]["stop_average_precision"] == pytest.approx(
        0.5
    )
    assert report["state_level"]["lora"]["stop_average_precision"] == pytest.approx(
        1.0
    )
    assert report["state_level"]["paired_lora_minus_base"][
        "stop_average_precision"
    ]["estimate"] == pytest.approx(0.5)
    assert report["improvement"]["em_higher"]
    assert rows[0]["lora"]["stopped"]
