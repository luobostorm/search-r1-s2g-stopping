from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


DEPLOY = Path(__file__).parents[1] / "deploy" / "searchr1-v02"
sys.path.insert(0, str(DEPLOY))

from qwen35_causal_lora import (  # noqa: E402
    LoRALinear,
    adapter_state,
    apply_lora,
    load_adapter,
    save_adapter,
)


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [nn.ModuleDict({"q_proj": nn.Linear(4, 4, bias=False)})]
        )
        self.visual = nn.ModuleDict({"q_proj": nn.Linear(4, 4, bias=False)})


def test_minimal_lora_targets_only_language_layers_and_roundtrips(tmp_path):
    model = TinyLanguageModel()
    original = model.model.language_model.layers[0]["q_proj"].weight.detach().clone()
    replaced = apply_lora(
        model,
        target_modules=["q_proj"],
        rank=2,
        alpha=4,
        dropout=0.0,
    )
    assert replaced == ["model.language_model.layers.0.q_proj"]
    layer = model.model.language_model.layers[0]["q_proj"]
    assert isinstance(layer, LoRALinear)
    assert not model.visual["q_proj"].weight.requires_grad
    with torch.no_grad():
        layer.lora_B.weight.fill_(0.5)
    state = adapter_state(model)
    assert set(state) == {
        "model.language_model.layers.0.q_proj.lora_A.weight",
        "model.language_model.layers.0.q_proj.lora_B.weight",
    }
    adapter = tmp_path / "adapter"
    save_adapter(
        model,
        adapter,
        config={
            "rank": 2,
            "alpha": 4,
            "dropout": 0.0,
            "target_modules": ["q_proj"],
        },
    )

    restored = TinyLanguageModel()
    restored.model.language_model.layers[0]["q_proj"].weight.data.copy_(original)
    config = load_adapter(restored, str(adapter))
    assert config["replaced_modules"] == replaced
    restored_layer = restored.model.language_model.layers[0]["q_proj"]
    assert torch.equal(restored_layer.lora_B.weight, layer.lora_B.weight)
    inputs = torch.randn(2, 4)
    assert torch.allclose(layer(inputs), restored_layer(inputs))
