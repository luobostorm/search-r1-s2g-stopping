"""Isolated labeled evaluation for Gate H0 v2 answer-only probes."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .searchr1_v02_eval import normalize_answer, official_em


BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20260726
STRONG_QUALITY_HEADROOM = 0.05
MODERATE_QUALITY_HEADROOM = 0.02
MATERIAL_AVOIDABLE_SEARCH_FRACTION = 0.20


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_f1(prediction: str | None, targets: Sequence[str]) -> float:
    if prediction is None:
        return 0.0
    predicted = normalize_answer(prediction).split()
    if not predicted:
        return 0.0
    best = 0.0
    for target in targets:
        golden = normalize_answer(target).split()
        if not golden:
            continue
        common = Counter(predicted) & Counter(golden)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(predicted)
        recall = overlap / len(golden)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def build_gateh0_report(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    run_rows: Sequence[Mapping[str, Any]],
    structure_audit: Mapping[str, Any],
    labels: Sequence[Mapping[str, Any]],
    native_comparisons: Sequence[Mapping[str, Any]],
    labels_manifest: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    states = [
        dict(row) for row in manifest_rows if row.get("record_type") == "state"
    ]
    probes = [dict(row) for row in run_rows if row.get("record_type") == "probe"]
    starts = [row for row in run_rows if row.get("record_type") == "run_start"]
    ends = [row for row in run_rows if row.get("record_type") == "run_end"]
    errors = [
        row for row in run_rows if row.get("record_type") == "probe_error"
    ]
    label_by_id = {str(row.get("question_id")): row for row in labels}
    native_by_id = {
        str(row.get("question_id")): row.get("native_first_answer", {})
        for row in native_comparisons
    }
    state_by_id = {str(row["state_id"]): row for row in states}
    probe_by_id = {str(row["state_id"]): row for row in probes}
    question_order = list(
        dict.fromkeys(str(row["question_id"]) for row in states)
    )
    label_order = [str(row.get("question_id")) for row in labels]
    native_order = [str(row.get("question_id")) for row in native_comparisons]
    native_states = [
        row for row in states if row.get("state_role") == "native_prefix"
    ]
    counterfactual_states = [
        row
        for row in states
        if row.get("state_role") == "counterfactual_forced_continue"
    ]
    native_search_states = [
        row
        for row in native_states
        if row.get("native_next_action") == "SEARCH"
    ]

    materialized_labels = (
        (labels_manifest.get("outputs") or {}).get("labels") or {}
    )
    checks = {
        "structure_audit_passed": structure_audit.get("passed") is True,
        "one_run_start": len(starts) == 1,
        "one_run_end": len(ends) == 1,
        "run_complete": len(ends) == 1 and ends[0].get("status") == "complete",
        "zero_probe_errors": len(errors) == 0,
        "probe_inference_declared_label_blind": (
            len(starts) == 1
            and starts[0].get("labels_mounted") is False
            and starts[0].get("gold_fields_received") == 0
        ),
        "expected_state_count": len(states) == 798,
        "expected_native_state_count": len(native_states) == 723,
        "expected_counterfactual_state_count": len(counterfactual_states) == 75,
        "expected_native_search_state_count": len(native_search_states) == 523,
        "expected_probe_count": len(probes) == 798,
        "all_probes_are_answers": all(
            row.get("action") == "ANSWER" and isinstance(row.get("answer"), str)
            for row in probes
        ),
        "unique_state_ids": len(state_by_id) == len(states),
        "unique_probe_ids": len(probe_by_id) == len(probes),
        "state_probe_ids_exact": list(state_by_id) == list(probe_by_id),
        "prompt_hashes_exact": all(
            probe_by_id[state_id].get("probe_input_ids_sha256")
            == state.get("probe_input_ids_sha256")
            for state_id, state in state_by_id.items()
        ),
        "expected_question_count": len(question_order) == 200,
        "expected_label_count": len(labels) == 200,
        "expected_native_comparison_count": len(native_comparisons) == 200,
        "unique_label_ids": len(label_by_id) == 200,
        "unique_native_ids": len(native_by_id) == 200,
        "label_order_exact": label_order == question_order,
        "native_order_exact": native_order == question_order,
        "all_labels_have_targets": all(
            isinstance(row.get("targets"), list) and bool(row["targets"])
            for row in labels
        ),
        "native_search_counts_match_manifest": all(
            int(native_by_id[question_id].get("search_calls", -1))
            == max(
                int(row["native_search_count"])
                for row in native_states
                if row["question_id"] == question_id
            )
            for question_id in question_order
        ),
        "labels_hash_matches_materialization": (
            materialized_labels.get("sha256") == artifact_hashes.get("labels")
        ),
        "labels_materialization_count": materialized_labels.get("count") == 200,
        "final_test_labels_not_used": labels_manifest.get(
            "final_test_labels_read"
        )
        is False,
    }

    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in native_states:
        probe = probe_by_id.get(str(state["state_id"]), {})
        by_question[str(state["question_id"])].append(
            {
                "state_id": state["state_id"],
                "state_index": int(state["state_index"]),
                "native_next_action": state.get("native_next_action"),
                "prediction": probe.get("answer"),
            }
        )
    for rows in by_question.values():
        rows.sort(key=lambda row: row["state_index"])

    question_rows: list[dict[str, Any]] = []
    stored_native_em_matches: list[bool] = []
    for question_id in question_order:
        targets = list(label_by_id.get(question_id, {}).get("targets", []))
        native = native_by_id.get(question_id, {})
        native_correct = official_em(native.get("prediction"), targets)
        stored_native_em_matches.append(
            native_correct == int(native.get("official_em", -1))
        )
        native_searches = int(native.get("search_calls", 0))
        scored_states = []
        for state in by_question[question_id]:
            prediction = state["prediction"]
            scored_states.append(
                {
                    **state,
                    "official_em": official_em(prediction, targets),
                    "token_f1": token_f1(prediction, targets),
                }
            )
        correct_states = [
            state for state in scored_states if state["official_em"] == 1
        ]
        earliest = correct_states[0] if correct_states else None
        correct_search_states = [
            state
            for state in scored_states
            if state["native_next_action"] == "SEARCH"
            and state["official_em"] == 1
        ]
        earliest_early_stop = (
            correct_search_states[0] if correct_search_states else None
        )
        terminal_state = next(
            (
                state
                for state in scored_states
                if state["native_next_action"] != "SEARCH"
            ),
            None,
        )
        probe_any_correct = int(earliest is not None)
        rescue = int(native_correct == 0 and earliest_early_stop is not None)
        terminal_only_rescue = int(
            native_correct == 0
            and earliest_early_stop is None
            and terminal_state is not None
            and terminal_state["official_em"] == 1
        )
        quality_oracle_correct = int(native_correct == 1 or rescue == 1)

        if native_correct and earliest_early_stop is not None:
            cost_searches = min(
                native_searches, int(earliest_early_stop["state_index"])
            )
        else:
            cost_searches = native_searches
        avoidable = native_searches - cost_searches

        search_states = [
            state
            for state in scored_states
            if state["native_next_action"] == "SEARCH"
        ]
        question_rows.append(
            {
                "question_id": question_id,
                "native_prediction": native.get("prediction"),
                "native_official_em": native_correct,
                "native_search_calls": native_searches,
                "probe_any_correct": probe_any_correct,
                "quality_oracle_official_em": quality_oracle_correct,
                "harmful_oversearch_rescue": rescue,
                "terminal_only_answerer_rescue": terminal_only_rescue,
                "earliest_correct_state_index": (
                    int(earliest["state_index"]) if earliest is not None else None
                ),
                "earliest_correct_prediction": (
                    earliest["prediction"] if earliest is not None else None
                ),
                "earliest_correct_early_stop_state_index": (
                    int(earliest_early_stop["state_index"])
                    if earliest_early_stop is not None
                    else None
                ),
                "cost_oracle_search_calls": cost_searches,
                "avoidable_searches": avoidable,
                "native_search_state_count": len(search_states),
                "correct_native_search_state_count": len(correct_search_states),
                "has_cost_only_oversearch": bool(
                    native_correct and correct_search_states
                ),
                "has_harmful_oversearch_state": bool(
                    not native_correct and correct_search_states
                ),
                "states": scored_states,
            }
        )
    checks["native_em_recomputed_exact"] = all(stored_native_em_matches)

    native_correct_count = sum(row["native_official_em"] for row in question_rows)
    probe_coverage_count = sum(row["probe_any_correct"] for row in question_rows)
    quality_oracle_count = sum(
        row["quality_oracle_official_em"] for row in question_rows
    )
    rescue_count = sum(
        row["harmful_oversearch_rescue"] for row in question_rows
    )
    terminal_only_rescue_count = sum(
        row["terminal_only_answerer_rescue"] for row in question_rows
    )
    native_search_total = sum(row["native_search_calls"] for row in question_rows)
    cost_search_total = sum(
        row["cost_oracle_search_calls"] for row in question_rows
    )
    avoidable_total = native_search_total - cost_search_total
    stop_now_answerable_search_state_count = sum(
        int(row["correct_native_search_state_count"]) for row in question_rows
    )
    stop_now_answerable_search_question_count = sum(
        int(row["correct_native_search_state_count"] > 0)
        for row in question_rows
    )

    action_cells = Counter()
    question_cells: Counter[str] = Counter()
    for row in question_rows:
        native_correct = bool(row["native_official_em"])
        correct_search_count = int(row["correct_native_search_state_count"])
        total_search_count = int(row["native_search_state_count"])
        action_cells[
            "stop_correct_native_correct"
            if native_correct
            else "stop_correct_native_wrong"
        ] += correct_search_count
        action_cells[
            "stop_wrong_native_correct"
            if native_correct
            else "stop_wrong_native_wrong"
        ] += total_search_count - correct_search_count
        if correct_search_count:
            question_cells[
                "has_stop_correct_native_correct"
                if native_correct
                else "has_stop_correct_native_wrong"
            ] += 1
        else:
            question_cells[
                "no_stop_correct_native_correct"
                if native_correct
                else "no_stop_correct_native_wrong"
            ] += 1

    fixed_budget = []
    for budget in range(5):
        selected = []
        exact_state_available = 0
        for row in question_rows:
            states_for_question = row["states"]
            if budget < len(states_for_question):
                exact_state_available += 1
            chosen = states_for_question[min(budget, len(states_for_question) - 1)]
            selected.append(chosen)
        fixed_budget.append(
            {
                "budget_k": budget,
                "selection_rule": "last_reachable",
                "question_count": len(selected),
                "exact_state_available_count": exact_state_available,
                "exact_state_coverage": exact_state_available / len(selected),
                "last_reachable_imputed_count": (
                    len(selected) - exact_state_available
                ),
                "official_em": sum(item["official_em"] for item in selected)
                / len(selected),
                "token_f1": sum(item["token_f1"] for item in selected)
                / len(selected),
                "mean_search_calls": sum(
                    item["state_index"] for item in selected
                )
                / len(selected),
            }
        )

    counterfactual_scored_states = []
    for state in counterfactual_states:
        question_id = str(state["question_id"])
        probe = probe_by_id.get(str(state["state_id"]), {})
        prediction = probe.get("answer")
        targets = list(label_by_id.get(question_id, {}).get("targets", []))
        counterfactual_scored_states.append(
            {
                "state_id": state["state_id"],
                "question_id": question_id,
                "state_index": int(state["state_index"]),
                "prediction": prediction,
                "official_em": official_em(prediction, targets),
                "token_f1": token_f1(prediction, targets),
            }
        )
    counterfactual_question_ids = list(
        dict.fromkeys(
            row["question_id"] for row in counterfactual_scored_states
        )
    )
    counterfactual_correct_question_ids = {
        row["question_id"]
        for row in counterfactual_scored_states
        if row["official_em"] == 1
    }

    bootstrap = _bootstrap_intervals(
        question_rows,
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
    )
    question_count = len(question_rows)
    if question_count == 0:
        raise ValueError("Gate H0 evaluation has no questions")
    native_em = native_correct_count / question_count
    probe_coverage_em = probe_coverage_count / question_count
    quality_oracle_em = quality_oracle_count / question_count
    quality_headroom = quality_oracle_em - native_em
    avoidable_fraction = (
        avoidable_total / native_search_total if native_search_total else 0.0
    )
    decision = resource_decision(quality_headroom, avoidable_fraction)
    analysis_status = "PASS" if all(checks.values()) else "FAIL"
    if analysis_status != "PASS":
        decision = "ANALYSIS_FAILED"

    report = {
        "schema_version": 1,
        "stage": "Gate H0 v2 answer-only exploratory stop-headroom audit",
        "analysis_status": analysis_status,
        "decision": decision,
        "claim_boundary": {
            "confirmatory": False,
            "dataset": "HotpotQA exploratory pilot 200",
            "viewed_questions": 200,
            "final_test_labels_used": False,
            "same_model_answer_only_conditional_upper_bound": True,
            "natural_native_behavior": False,
            "counterfactual_states_used_in_primary_metrics": False,
            "evidence_oracle": "UNAVAILABLE",
            "resource_decision_rules_frozen_after_labelblind_probe_count": 318,
            "gold_correctness_seen_before_resource_decision_freeze": False,
        },
        "checks": checks,
        "metrics": {
            "question_count": question_count,
            "native_em": native_em,
            "forced_answer_coverage_em": probe_coverage_em,
            "evidence_ready_search_rate": None,
            "evidence_ready_search_rate_status": "UNAVAILABLE",
            "stop_now_answerable_search_state_count": (
                stop_now_answerable_search_state_count
            ),
            "stop_now_answerable_search_state_denominator": len(
                native_search_states
            ),
            "stop_now_answerable_search_rate": (
                stop_now_answerable_search_state_count
                / len(native_search_states)
                if native_search_states
                else 0.0
            ),
            "stop_now_answerable_search_question_count": (
                stop_now_answerable_search_question_count
            ),
            "stop_now_answerable_search_question_rate": (
                stop_now_answerable_search_question_count / question_count
            ),
            "quality_stop_oracle_em": quality_oracle_em,
            "quality_headroom": quality_headroom,
            "quality_headroom_pp": quality_headroom * 100,
            "harmful_oversearch_rescue_questions": rescue_count,
            "terminal_only_answerer_rescue_questions": terminal_only_rescue_count,
            "native_mean_search_calls": native_search_total / question_count,
            "cost_oracle_mean_search_calls": cost_search_total / question_count,
            "avoidable_searches": avoidable_total,
            "avoidable_search_fraction": avoidable_fraction,
            "questions_with_cost_only_oversearch": sum(
                row["has_cost_only_oversearch"] for row in question_rows
            ),
            "questions_with_harmful_oversearch_state": sum(
                row["has_harmful_oversearch_state"] for row in question_rows
            ),
        },
        "native_search_action_cells": dict(sorted(action_cells.items())),
        "native_search_question_cells": dict(sorted(question_cells.items())),
        "fixed_budget_curve": fixed_budget,
        "paired_bootstrap_95_ci": bootstrap,
        "bootstrap": {
            "unit": "question",
            "repeats": bootstrap_repeats,
            "seed": bootstrap_seed,
        },
        "counterfactual_supplement": {
            "state_count": len(counterfactual_states),
            "question_count": len(counterfactual_question_ids),
            "official_em_correct_state_count": sum(
                row["official_em"] for row in counterfactual_scored_states
            ),
            "official_em_state_rate": (
                sum(row["official_em"] for row in counterfactual_scored_states)
                / len(counterfactual_scored_states)
                if counterfactual_scored_states
                else 0.0
            ),
            "mean_token_f1": (
                sum(row["token_f1"] for row in counterfactual_scored_states)
                / len(counterfactual_scored_states)
                if counterfactual_scored_states
                else 0.0
            ),
            "questions_with_at_least_one_correct_state": len(
                counterfactual_correct_question_ids
            ),
            "question_answerability_rate": (
                len(counterfactual_correct_question_ids)
                / len(counterfactual_question_ids)
                if counterfactual_question_ids
                else 0.0
            ),
            "included_in_primary_metrics": False,
            "native_oversearch_attribution_allowed": False,
        },
        "artifact_hashes": dict(artifact_hashes),
    }
    return report, question_rows


def resource_decision(
    quality_headroom: float, avoidable_search_fraction: float
) -> str:
    """Apply the frozen exhaustive exploratory resource-allocation matrix."""

    if quality_headroom >= STRONG_QUALITY_HEADROOM:
        return "STRONG_QUALITY_HEADROOM"
    if quality_headroom >= MODERATE_QUALITY_HEADROOM:
        if (
            avoidable_search_fraction
            >= MATERIAL_AVOIDABLE_SEARCH_FRACTION
        ):
            return "QUALITY_COST_PARETO"
        return "MODERATE_QUALITY_HEADROOM"
    if avoidable_search_fraction >= MATERIAL_AVOIDABLE_SEARCH_FRACTION:
        return "EFFICIENCY_ONLY"
    return "LOW_STOPPING_HEADROOM"


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    curve = report["fixed_budget_curve"]
    cells = report["native_search_action_cells"]
    question_cells = report["native_search_question_cells"]
    bootstrap = report["paired_bootstrap_95_ci"]
    counterfactual = report["counterfactual_supplement"]
    checks = report["checks"]
    lines = [
        "# Gate H0 v2：Answer-Only 停止上界审计报告",
        "",
        "## Material Passport",
        "",
        "- **Artifact ID**：`searchr1-v02-gate-h0-v2-quality-cost-report`",
        "- **Artifact type**：探索性停止条件上界审计",
        "- **Dataset**：HotpotQA exploratory pilot 200",
        "- **Primary state universe**：723 个 Native 可达状态",
        "- **Counterfactual supplement**：75 个 forced-continue 状态，"
        "不进入主指标",
        "- **Answerer**：Search-R1 Qwen2.5-7B v0.2，同模型 "
        "answer-only constrained decoding",
        "- **Label order**：先完成 798-state 无标签结构审计，"
        "后隔离连接已查看的 pilot labels",
        "- **Claim status**：exploratory / non-confirmatory / "
        "not final-test evidence",
        "- **Resource decision clarification**：在 318 个无标签输出之后、"
        "Gate H0 gold 评测之前冻结；未据 gold 改规则",
        "",
        "## 结论",
        "",
        f"- 分析状态：`{report['analysis_status']}`",
        f"- 资源决策：`{report['decision']}`",
        f"- Native EM：{metrics['native_em']:.1%}",
        f"- Forced-answer coverage EM：{metrics['forced_answer_coverage_em']:.1%}",
        "- Evidence-ready SEARCH rate：`UNAVAILABLE`（supporting-fact labels "
        "未 materialize）",
        f"- Stop-now answerable SEARCH rate："
        f"{metrics['stop_now_answerable_search_state_count']} / "
        f"{metrics['stop_now_answerable_search_state_denominator']} "
        f"（{metrics['stop_now_answerable_search_rate']:.1%}）",
        f"- 至少一个 stop-now answerable SEARCH 状态的问题："
        f"{metrics['stop_now_answerable_search_question_count']} / "
        f"{metrics['question_count']} "
        f"（{metrics['stop_now_answerable_search_question_rate']:.1%}）",
        f"- Quality Stop Oracle EM：{metrics['quality_stop_oracle_em']:.1%}",
        f"- 质量上界增量：{metrics['quality_headroom_pp']:.1f} pp",
        f"- harmful over-search rescue：{metrics['harmful_oversearch_rescue_questions']} 题",
        f"- 仅终止状态由 answer-only 解码救回："
        f"{metrics['terminal_only_answerer_rescue_questions']} 题"
        "（不归因于停止）",
        f"- Native 平均搜索：{metrics['native_mean_search_calls']:.3f}",
        f"- 保持 Native EM 的 Cost Oracle 平均搜索：{metrics['cost_oracle_mean_search_calls']:.3f}",
        f"- 可避免搜索：{metrics['avoidable_searches']} / "
        f"{int(metrics['native_mean_search_calls'] * metrics['question_count'])}"
        f"（{metrics['avoidable_search_fraction']:.1%}）",
        f"- 出现 cost-only over-search 的问题："
        f"{metrics['questions_with_cost_only_oversearch']} / "
        f"{metrics['question_count']}",
        f"- 出现 harmful over-search state 的问题："
        f"{metrics['questions_with_harmful_oversearch_state']} / "
        f"{metrics['question_count']}",
        "",
        "这里的 Quality Stop Oracle 只允许在 Native 下一动作仍为 SEARCH "
        "的状态提前停止；若不存在正确早停状态，则保留 Native 原行为。"
        "最后一个终止状态的 answer-only 改善单独报告，不归因于停止。",
        "",
        "## Native SEARCH 动作四格",
        "",
        "| Stop-now v2 | Native 最终 | 状态数 |",
        "|---|---:|---:|",
        f"| 正确 | 正确 | {cells.get('stop_correct_native_correct', 0)} |",
        f"| 正确 | 错误 | {cells.get('stop_correct_native_wrong', 0)} |",
        f"| 错误 | 正确 | {cells.get('stop_wrong_native_correct', 0)} |",
        f"| 错误 | 错误 | {cells.get('stop_wrong_native_wrong', 0)} |",
        "",
        "动作级分母固定为 523 个 Native `SEARCH` 状态；同一问题可能贡献"
        "多个动作，不能将动作视为独立问题。",
        "",
        "## Native SEARCH 问题级分解",
        "",
        "| 是否至少有一个正确的 stop-now 状态 | Native 最终 | 问题数 |",
        "|---|---:|---:|",
        f"| 是 | 正确 | "
        f"{question_cells.get('has_stop_correct_native_correct', 0)} |",
        f"| 是 | 错误 | "
        f"{question_cells.get('has_stop_correct_native_wrong', 0)} |",
        f"| 否 | 正确 | "
        f"{question_cells.get('no_stop_correct_native_correct', 0)} |",
        f"| 否 | 错误 | "
        f"{question_cells.get('no_stop_correct_native_wrong', 0)} |",
        "",
        "## 固定预算曲线",
        "",
        "| k | EM | Token F1 | 平均搜索 | 精确第 k 状态覆盖 | last-reachable 回填 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curve:
        lines.append(
            f"| {row['budget_k']} | {row['official_em']:.1%} | "
            f"{row['token_f1']:.3f} | {row['mean_search_calls']:.3f} | "
            f"{row['exact_state_available_count']}/{row['question_count']} "
            f"({row['exact_state_coverage']:.1%}) | "
            f"{row['last_reachable_imputed_count']} |"
        )
    lines.extend(
        [
            "",
            "## 问题级 paired bootstrap 描述性区间",
            "",
            "| 指标 | 95% 区间 |",
            "|---|---:|",
            f"| Quality headroom | "
            f"[{bootstrap['quality_headroom'][0]:.1%}, "
            f"{bootstrap['quality_headroom'][1]:.1%}] |",
            f"| 可避免搜索比例 | "
            f"[{bootstrap['avoidable_search_fraction'][0]:.1%}, "
            f"{bootstrap['avoidable_search_fraction'][1]:.1%}] |",
            f"| 每题平均减少搜索 | "
            f"[{bootstrap['mean_search_reduction'][0]:.3f}, "
            f"{bootstrap['mean_search_reduction'][1]:.3f}] |",
            "",
            "区间以问题为重采样单位，10,000 次、固定 seed 20260726；"
            "仅作探索性描述，不是确认性显著性检验。",
            "",
            "## forced-continue 反事实补充",
            "",
            f"- 状态数：{counterfactual['state_count']}；涉及问题数："
            f"{counterfactual['question_count']}。",
            f"- 状态级 official EM："
            f"{counterfactual['official_em_correct_state_count']} / "
            f"{counterfactual['state_count']} "
            f"（{counterfactual['official_em_state_rate']:.1%}）。",
            f"- 平均 Token F1：{counterfactual['mean_token_f1']:.3f}。",
            f"- 至少一个正确反事实状态的问题："
            f"{counterfactual['questions_with_at_least_one_correct_state']} / "
            f"{counterfactual['question_count']} "
            f"（{counterfactual['question_answerability_rate']:.1%}）。",
            "- 这些状态不是 Native 实际继续搜索到的状态，仅描述 forced-continue "
            "后的答案可达性；禁止归因成 Native over-search。",
            "",
            "## 审计与边界",
            "",
            f"- 结构与隔离检查：{sum(checks.values())}/{len(checks)} 通过。",
            "- 主指标只使用 723 个 Native 可达状态；75 个 forced-continue "
            "反事实状态不进入 Native over-search 比例。",
            "- 本结果来自已查看过标签的 200 题 exploratory pilot，不是 "
            "final-test 或确认性结论。",
            "- v2 测量同一 Search-R1 answerer 在 answer-only 解码条件下的"
            "可回答性；不等同于 Native 自然生成行为。",
            "- supporting-fact 标签未 materialize，Evidence Oracle 标记为 "
            "`UNAVAILABLE`。",
            "",
        ]
    )
    return "\n".join(lines)


def _bootstrap_intervals(
    rows: Sequence[Mapping[str, Any]], *, repeats: int, seed: int
) -> dict[str, list[float]]:
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    rng = random.Random(seed)
    count = len(rows)
    quality_deltas = []
    avoidable_fractions = []
    search_deltas = []
    for _ in range(repeats):
        sample = [rows[rng.randrange(count)] for _ in range(count)]
        quality_deltas.append(
            sum(
                int(row["quality_oracle_official_em"])
                - int(row["native_official_em"])
                for row in sample
            )
            / count
        )
        native_searches = sum(int(row["native_search_calls"]) for row in sample)
        avoidable = sum(int(row["avoidable_searches"]) for row in sample)
        avoidable_fractions.append(
            avoidable / native_searches if native_searches else 0.0
        )
        search_deltas.append(avoidable / count)
    return {
        "quality_headroom": _percentile_interval(quality_deltas),
        "avoidable_search_fraction": _percentile_interval(avoidable_fractions),
        "mean_search_reduction": _percentile_interval(search_deltas),
    }


def _percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize empty bootstrap values")
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return [lower, upper]
