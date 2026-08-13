from __future__ import annotations

from kstar.searchr1_v02_gateh0_eval import (
    build_gateh0_report,
    render_markdown,
    resource_decision,
)


def _fixture():
    manifest = []
    probes = []
    labels = []
    native = []
    for order, (qid, native_ok) in enumerate((("q1", 1), ("q2", 0))):
        labels.append({"question_id": qid, "targets": [f"gold-{qid}"]})
        native.append(
            {
                "question_id": qid,
                "native_first_answer": {
                    "official_em": native_ok,
                    "prediction": f"gold-{qid}" if native_ok else f"native-{qid}",
                    "search_calls": 1,
                },
            }
        )
        for index in range(2):
            state_id = f"{qid}:{index}"
            manifest.append(
                {
                    "record_type": "state",
                    "state_id": state_id,
                    "question_id": qid,
                    "question_order": order,
                    "state_index": index,
                    "native_search_count": 1,
                    "state_role": "native_prefix",
                    "native_next_action": "SEARCH" if index == 0 else "ANSWER",
                    "probe_input_ids_sha256": f"h-{state_id}",
                }
            )
            answer = (
                f"gold-{qid}"
                if (qid == "q1" and index == 0)
                or (qid == "q2" and index == 0)
                else "wrong"
            )
            probes.append(
                {
                    "record_type": "probe",
                    "state_id": state_id,
                    "question_id": qid,
                    "action": "ANSWER",
                    "answer": answer,
                    "probe_input_ids_sha256": f"h-{state_id}",
                }
            )
        counterfactual_state_id = f"{qid}:counterfactual"
        manifest.append(
            {
                "record_type": "state",
                "state_id": counterfactual_state_id,
                "question_id": qid,
                "question_order": order,
                "state_index": 2,
                "native_search_count": 1,
                "state_role": "counterfactual_forced_continue",
                "native_next_action": "COUNTERFACTUAL",
                "probe_input_ids_sha256": f"h-{counterfactual_state_id}",
            }
        )
        probes.append(
            {
                "record_type": "probe",
                "state_id": counterfactual_state_id,
                "question_id": qid,
                "action": "ANSWER",
                "answer": f"gold-{qid}" if qid == "q2" else "wrong",
                "probe_input_ids_sha256": f"h-{counterfactual_state_id}",
            }
        )
    run = [
        {
            "record_type": "run_start",
            "labels_mounted": False,
            "gold_fields_received": 0,
        },
        *probes,
        {"record_type": "run_end", "status": "complete"},
    ]
    labels_manifest = {
        "outputs": {"labels": {"sha256": "labels-hash", "count": 2}},
        "final_test_labels_read": False,
    }
    return manifest, run, labels, native, labels_manifest


def test_gateh0_oracles_rescue_quality_and_preserve_native_cost():
    manifest, run, labels, native, labels_manifest = _fixture()
    report, rows = build_gateh0_report(
        manifest_rows=manifest,
        run_rows=run,
        structure_audit={"passed": True},
        labels=labels,
        native_comparisons=native,
        labels_manifest=labels_manifest,
        artifact_hashes={"labels": "labels-hash"},
        bootstrap_repeats=100,
    )

    # Frozen production counts intentionally fail for this tiny fixture, but
    # metric behavior remains directly testable.
    assert report["analysis_status"] == "FAIL"
    assert report["metrics"]["native_em"] == 0.5
    assert report["metrics"]["forced_answer_coverage_em"] == 1.0
    assert report["metrics"]["evidence_ready_search_rate"] is None
    assert (
        report["metrics"]["evidence_ready_search_rate_status"]
        == "UNAVAILABLE"
    )
    assert (
        report["metrics"]["stop_now_answerable_search_state_count"] == 2
    )
    assert (
        report["metrics"]["stop_now_answerable_search_state_denominator"] == 2
    )
    assert report["metrics"]["stop_now_answerable_search_rate"] == 1.0
    assert (
        report["metrics"]["stop_now_answerable_search_question_count"] == 2
    )
    assert (
        report["metrics"]["stop_now_answerable_search_question_rate"] == 1.0
    )
    assert report["metrics"]["quality_stop_oracle_em"] == 1.0
    assert report["metrics"]["quality_headroom"] == 0.5
    assert report["metrics"]["harmful_oversearch_rescue_questions"] == 1
    assert report["metrics"]["avoidable_searches"] == 1
    assert rows[0]["cost_oracle_search_calls"] == 0
    assert rows[1]["cost_oracle_search_calls"] == 1
    assert report["native_search_action_cells"] == {
        "stop_correct_native_correct": 1,
        "stop_correct_native_wrong": 1,
        "stop_wrong_native_correct": 0,
        "stop_wrong_native_wrong": 0,
    }
    assert report["native_search_question_cells"] == {
        "has_stop_correct_native_correct": 1,
        "has_stop_correct_native_wrong": 1,
    }
    assert report["fixed_budget_curve"][0]["exact_state_coverage"] == 1.0
    assert report["fixed_budget_curve"][1]["exact_state_coverage"] == 1.0
    assert report["fixed_budget_curve"][2]["exact_state_coverage"] == 0.0
    assert report["fixed_budget_curve"][2][
        "last_reachable_imputed_count"
    ] == 2
    assert report["counterfactual_supplement"] == {
        "state_count": 2,
        "question_count": 2,
        "official_em_correct_state_count": 1,
        "official_em_state_rate": 0.5,
        "mean_token_f1": 0.5,
        "questions_with_at_least_one_correct_state": 1,
        "question_answerability_rate": 0.5,
        "included_in_primary_metrics": False,
        "native_oversearch_attribution_allowed": False,
    }
    markdown = render_markdown(report)
    assert "## Material Passport" in markdown
    assert "不归因于停止" in markdown
    assert "## Native SEARCH 问题级分解" in markdown
    assert "## 问题级 paired bootstrap 描述性区间" in markdown
    assert "## forced-continue 反事实补充" in markdown


def test_resource_decision_matrix_is_exhaustive():
    assert resource_decision(0.05, 0.0) == "STRONG_QUALITY_HEADROOM"
    assert resource_decision(0.03, 0.20) == "QUALITY_COST_PARETO"
    assert resource_decision(0.03, 0.19) == "MODERATE_QUALITY_HEADROOM"
    assert resource_decision(0.01, 0.20) == "EFFICIENCY_ONLY"
    assert resource_decision(0.01, 0.19) == "LOW_STOPPING_HEADROOM"
