from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_s2g_paper_figures.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_s2g_paper_figures", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_figures_are_generated_from_frozen_evidence(tmp_path):
    module = load_module()
    module.configure_style()
    h0 = module.load_json(module.DEFAULT_H0)
    final = module.load_json(module.DEFAULT_FINAL)

    framework = module.plot_evaluation_framework(tmp_path)
    headroom = module.plot_h0_headroom(h0, tmp_path)
    tradeoff = module.plot_confirmatory_tradeoff(final, tmp_path)

    for stem in (
        "stopping-evaluation-framework",
        "h0-stopping-headroom",
        "confirmatory-quality-cost-risk",
    ):
        assert (tmp_path / f"{stem}.pdf").stat().st_size > 0
        assert (tmp_path / f"{stem}.png").stat().st_size > 0

    assert headroom["derived_checks"]["quality_headroom_pp"] == pytest.approx(5.0)
    assert round(headroom["derived_checks"]["avoidable_search_fraction_percent"], 2) == 25.05
    assert tradeoff["derived_checks"]["early_stop_total"] == 69
    assert tradeoff["derived_checks"]["safe_plus_unsafe"] == 69
    assert round(tradeoff["derived_checks"]["unsafe_fraction"], 4) == 0.3913
    assert framework["pdf_sha256"]


def test_source_roles_remain_frozen():
    h0 = json.loads((ROOT / "results/searchr1-v02-reproduction/gateh0/quality-cost-report.json").read_text())
    final = json.loads(
        (
            ROOT
            / "results/s2g-scaleup-aligned-eval-v1/final-dev1000-v2/"
            "evaluation-confirmatory-suffix800.json"
        ).read_text()
    )
    assert h0["claim_boundary"]["confirmatory"] is False
    assert final["analysis_role"] == "confirmatory_untouched_suffix800"
