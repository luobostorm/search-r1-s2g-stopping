"""Generate the three claim-bearing figures for the Search-R1/S2G paper.

The quantitative panels read the frozen JSON evidence directly.  Run from the
repository root with no arguments; outputs are written under paper/arxiv/figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H0 = ROOT / "results/searchr1-v02-reproduction/gateh0/quality-cost-report.json"
DEFAULT_FINAL = (
    ROOT
    / "results/s2g-scaleup-aligned-eval-v1/final-dev1000-v2/"
    "evaluation-confirmatory-suffix800.json"
)
DEFAULT_OUTPUT = ROOT / "paper/arxiv/figures"

BLUE = "#0077BB"
CYAN = "#33BBEE"
TEAL = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
MAGENTA = "#EE3377"
GREY = "#6B7280"
LIGHT_GREY = "#E5E7EB"
BLACK = "#111827"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)
    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    return {
        "pdf": display_path(pdf),
        "pdf_sha256": sha256(pdf),
        "png": display_path(png),
        "png_sha256": sha256(png),
    }


def add_box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    heading: str,
    body: str,
    color: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.67, heading, ha="center", va="center", weight="bold")
    ax.text(x + width / 2, y + height * 0.31, body, ha="center", va="center", fontsize=8, color=GREY)


def add_arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = GREY,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def plot_evaluation_framework(output_dir: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(6.9, 3.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.015, 0.27, 0.525, 0.78]
    width, height, y = 0.205, 0.27, 0.60
    boxes = [
        ("1  Reachability", "Can the current state\nanswer correctly now?", BLUE),
        ("2  State ranking", "Can the judge rank\nsafe stop states?", TEAL),
        ("3  First crossing", "Which state crosses\nthe threshold first?", ORANGE),
        ("4  System effects", "Answer quality, stop risk,\nand retrieval cost", MAGENTA),
    ]
    for x, (heading, body, color) in zip(xs, boxes):
        add_box(ax, (x, y), width, height, heading, body, color)
    for index in range(3):
        add_arrow(ax, (xs[index] + width, y + height / 2), (xs[index + 1], y + height / 2))

    ax.text(0.5, 0.94, "Trajectory-aware evaluation of multi-round RAG stopping", ha="center", weight="bold", fontsize=11)

    ax.text(0.015, 0.465, "Illustrative trajectory: first crossing at s2", ha="left", va="center", fontsize=8, color=GREY)

    # A first-crossing example.  The action labels sit away from the state
    # markers so the diagram remains readable when scaled to a paper column.
    state_x = [0.07, 0.28, 0.49, 0.70, 0.91]
    state_y = 0.29
    for left, right in zip(state_x[:2], state_x[1:3]):
        add_arrow(ax, (left + 0.025, state_y), (right - 0.025, state_y), color=BLUE)
    for left, right in zip(state_x[2:4], state_x[3:5]):
        ax.plot([left + 0.025, right - 0.025], [state_y, state_y], color=LIGHT_GREY, linewidth=1.5, linestyle="--")

    ax.scatter(state_x[:2], [state_y] * 2, s=95, facecolor="white", edgecolor=BLUE, linewidth=1.6, zorder=3)
    ax.scatter([state_x[2]], [state_y], s=115, marker="X", color=RED, linewidth=1.2, zorder=4)
    ax.scatter(state_x[3:], [state_y] * 2, s=85, facecolor="white", edgecolor=GREY, linewidth=1.2, alpha=0.55, zorder=3)

    for index, x in enumerate(state_x):
        color = RED if index == 2 else (GREY if index > 2 else BLACK)
        alpha = 0.6 if index > 2 else 1.0
        ax.text(x, state_y + 0.075, f"s{index}", ha="center", va="bottom", fontsize=8, color=color, alpha=alpha, weight="bold")
    actions = ["SEARCH", "SEARCH", "STOP", "not reached", "not reached"]
    action_colors = [BLUE, BLUE, RED, GREY, GREY]
    for index, (x, action, color) in enumerate(zip(state_x, actions, action_colors)):
        ax.text(x, state_y - 0.075, action, ha="center", va="top", fontsize=8, color=color, alpha=0.55 if index > 2 else 1.0)

    ax.text(
        0.70,
        0.055,
        "a false first crossing truncates every later state",
        ha="center",
        va="center",
        fontsize=8,
        color=RED,
    )
    return save_figure(fig, output_dir, "stopping-evaluation-framework")


def plot_h0_headroom(h0: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    curve = h0["fixed_budget_curve"]
    searches = [float(row["mean_search_calls"]) for row in curve]
    em = [100 * float(row["official_em"]) for row in curve]
    metrics = h0["metrics"]
    ci = h0["paired_bootstrap_95_ci"]

    native_x = float(metrics["native_mean_search_calls"])
    native_y = 100 * float(metrics["native_em"])
    oracle_y = 100 * float(metrics["quality_stop_oracle_em"])
    cost_x = float(metrics["cost_oracle_mean_search_calls"])
    headroom_low, headroom_high = [100 * float(value) for value in ci["quality_headroom"]]
    cost_low, cost_high = [float(value) for value in ci["mean_search_reduction"]]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.15), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    ax.plot(searches, em, color=BLUE, marker="o", linewidth=1.6, markersize=5, label="Fixed budget")
    label_offsets = [(0, 7), (0, 7), (-4, 9), (-13, 9), (13, 7)]
    for row, x, y_value, (x_offset, y_offset) in zip(curve, searches, em, label_offsets):
        ax.annotate(
            f"k={int(row['budget_k'])}",
            (x, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax.scatter([native_x], [native_y], marker="s", s=54, color=BLACK, label="Native")
    ax.scatter([native_x], [oracle_y], marker="D", s=54, color=ORANGE, label="Quality oracle")
    ax.scatter([cost_x], [native_y], marker="P", s=62, color=TEAL, label="Cost oracle")
    ax.annotate(
        "+5.0 pp",
        xy=(native_x, oracle_y),
        xytext=(native_x - 0.42, oracle_y + 4.2),
        fontsize=8,
        color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE),
    )
    ax.annotate(
        "same EM, fewer searches",
        xy=(cost_x, native_y),
        xytext=(cost_x - 0.25, native_y - 8.0),
        fontsize=8,
        color=TEAL,
        arrowprops=dict(arrowstyle="->", color=TEAL),
    )
    ax.set_xlabel("Mean search calls per question")
    ax.set_ylabel("Official EM (%)")
    ax.set_title("a  Fixed budgets versus conditional oracles", loc="left", weight="bold")
    ax.set_xlim(-0.1, 3.05)
    ax.set_ylim(14, 57)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    estimates = [100 * float(metrics["quality_headroom"]), 100 * float(metrics["avoidable_search_fraction"])]
    lows = [headroom_low, 100 * float(ci["avoidable_search_fraction"][0])]
    highs = [headroom_high, 100 * float(ci["avoidable_search_fraction"][1])]
    labels = ["Quality\nheadroom", "Avoidable\nsearches"]
    colors = [ORANGE, TEAL]
    for y_pos, estimate, low, high, color in zip([1, 0], estimates, lows, highs, colors):
        ax.errorbar(
            estimate,
            y_pos,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            markersize=7,
            capsize=4,
            color=color,
            linewidth=1.5,
        )
        ax.text(estimate, y_pos + 0.16, f"{estimate:.2f}%", ha="center", color=color, fontsize=8, weight="bold")
    ax.set_yticks([1, 0], labels)
    ax.set_xlabel("Estimate with question-level 95% CI (%)")
    ax.set_title("b  Exploratory headroom", loc="left", weight="bold")
    ax.axvline(0, color=GREY, linewidth=0.8)
    ax.set_xlim(-1, 35)
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)
    fig.subplots_adjust(wspace=0.40)

    result = save_figure(fig, output_dir, "h0-stopping-headroom")
    result["derived_checks"] = {
        "fixed_budget_points": len(curve),
        "quality_headroom_pp": estimates[0],
        "avoidable_search_fraction_percent": estimates[1],
        "cost_oracle_reduction_ci_calls": [cost_low, cost_high],
    }
    return result


def plot_confirmatory_tradeoff(final: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    comparison = final["paired_comparisons"]["expanded_s2g_lora_minus_native"]
    em = comparison["official_em"]
    search = comparison["search_calls"]
    policy = final["system_level"]["expanded_s2g_lora"]

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9), gridspec_kw={"width_ratios": [1.1, 1.1, 1.35]})

    ax = axes[0]
    estimate = 100 * float(em["estimate"])
    low = 100 * float(em["ci95_low"])
    high = 100 * float(em["ci95_high"])
    ax.axvspan(-2, 0.6, color=TEAL, alpha=0.08)
    ax.axvline(-2, color=RED, linestyle="--", linewidth=1.2)
    ax.axvline(0, color=GREY, linewidth=0.8)
    ax.errorbar(estimate, 0, xerr=[[estimate - low], [high - estimate]], fmt="o", color=ORANGE, capsize=5, markersize=7, linewidth=1.6)
    ax.text(-2, 0.34, "non-inferiority\nmargin", ha="center", va="bottom", fontsize=7, color=RED)
    ax.text(estimate, -0.31, f"{estimate:.3f} pp\n[{low:.2f}, {high:.2f}]", ha="center", va="top", fontsize=8)
    ax.set_xlim(-2.5, 0.65)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([])
    ax.set_xlabel("EM difference (pp)")
    ax.set_title("a  Quality bound", loc="left", weight="bold")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)

    ax = axes[1]
    estimate = float(search["estimate"])
    low = float(search["ci95_low"])
    high = float(search["ci95_high"])
    ax.axvspan(-0.135, 0, color=TEAL, alpha=0.08)
    ax.axvline(0, color=RED, linestyle="--", linewidth=1.2)
    ax.errorbar(estimate, 0, xerr=[[estimate - low], [high - estimate]], fmt="o", color=TEAL, capsize=5, markersize=7, linewidth=1.6)
    ax.text(estimate, -0.31, f"{estimate:.5f}\n[{low:.5f}, {high:.5f}]", ha="center", va="top", fontsize=8)
    ax.set_xlim(-0.135, 0.018)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([])
    ax.set_xlabel("Search-call difference")
    ax.set_title("b  Retrieval reduction", loc="left", weight="bold")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)

    safe = int(policy["safe_early_stop_count"])
    unsafe = int(policy["unsafe_early_stop_count"])
    total = int(policy["early_stop_count"])
    ax = axes[2]
    ax.barh([0], [safe], color=TEAL, height=0.42, label=f"Safe: {safe}")
    ax.barh([0], [unsafe], left=[safe], color=RED, height=0.42, label=f"Unsafe: {unsafe}")
    ax.text(safe / 2, 0, str(safe), ha="center", va="center", color="white", weight="bold", fontsize=9)
    ax.text(safe + unsafe / 2, 0, str(unsafe), ha="center", va="center", color="white", weight="bold", fontsize=9)
    ax.text(total / 2, -0.39, f"unsafe fraction = {100 * unsafe / total:.2f}%", ha="center", va="top", fontsize=8, color=RED)
    ax.set_xlim(0, total)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([])
    ax.set_xlabel("System-level early stops (n)")
    ax.set_title("c  Selected-stop risk", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper center", ncols=1)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)

    fig.subplots_adjust(wspace=0.42)
    result = save_figure(fig, output_dir, "confirmatory-quality-cost-risk")
    result["derived_checks"] = {
        "em_difference_pp": 100 * float(em["estimate"]),
        "search_difference": float(search["estimate"]),
        "early_stop_total": total,
        "safe_plus_unsafe": safe + unsafe,
        "unsafe_fraction": unsafe / total,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h0", type=Path, default=DEFAULT_H0)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    configure_style()
    h0 = load_json(args.h0)
    final = load_json(args.final)
    if h0.get("analysis_status") != "PASS":
        raise ValueError("H0 evidence is not in PASS state")
    if final.get("analysis_role") != "confirmatory_untouched_suffix800":
        raise ValueError("final evidence is not the confirmatory suffix-800 artifact")

    script = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "generator": str(script.relative_to(ROOT)),
        "generator_sha256": sha256(script),
        "source_data": {
            "h0": str(args.h0.resolve().relative_to(ROOT)),
            "h0_sha256": sha256(args.h0),
            "confirmatory": str(args.final.resolve().relative_to(ROOT)),
            "confirmatory_sha256": sha256(args.final),
        },
        "figures": {
            "framework": plot_evaluation_framework(args.output_dir),
            "h0_headroom": plot_h0_headroom(h0, args.output_dir),
            "confirmatory_tradeoff": plot_confirmatory_tradeoff(final, args.output_dir),
        },
        "claim_boundaries": [
            "H0 is exploratory and supports identifiability/headroom only.",
            "Suffix-800 supports retrieval reduction within the frozen EM non-inferiority margin.",
            "The figures do not establish answer-quality superiority, safe stopping, or lower total compute cost.",
        ],
        "figure_table_trace": [
            {
                "artifact": "stopping-evaluation-framework",
                "source_data": ["paper/多轮RAG何时停止-论文核心立意.md"],
                "transformation": "Conceptual synthesis of the four evaluation layers; no quantitative inference.",
                "caption_claim": "Stopping must be evaluated from reachable states through selected system outcomes.",
                "supported_claims": ["Defines the paper's trajectory-aware evaluation framework."],
                "limitations": ["Conceptual diagram; it is not an empirical result."],
            },
            {
                "artifact": "h0-stopping-headroom",
                "source_data": [str(args.h0.resolve().relative_to(ROOT))],
                "transformation": "Fixed-budget curve plus paired question-level bootstrap intervals read from the frozen H0 JSON.",
                "caption_claim": "Exploratory evidence shows conditional stopping headroom and avoidable retrievals.",
                "supported_claims": ["Quality headroom is 5.00 pp.", "Avoidable searches are 25.05%."],
                "limitations": ["Exploratory, conditional-oracle evidence; not a deployable policy result."],
            },
            {
                "artifact": "confirmatory-quality-cost-risk",
                "source_data": [str(args.final.resolve().relative_to(ROOT))],
                "transformation": "Paired suffix-800 differences with frozen confidence intervals and selected-stop counts.",
                "caption_claim": "Expanded S2G reduces retrieval calls within the frozen EM non-inferiority bound, but unsafe selected stops remain.",
                "supported_claims": ["EM difference is -0.625 pp.", "Search-call difference is -0.09625.", "27 of 69 early stops are unsafe."],
                "limitations": ["Does not establish answer-quality superiority, safe stopping, or lower total compute cost."],
            },
        ],
    }
    manifest_path = args.output_dir / "figure-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generated": list(manifest["figures"]), "manifest": str(manifest_path.relative_to(ROOT))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
