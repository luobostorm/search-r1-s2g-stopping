"""Minimal auditable LoRA implementation for Qwen3.5 language Linear layers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from safetensors.torch import load_file, save_file
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        device = base_layer.weight.device
        self.lora_A = nn.Linear(
            base_layer.in_features,
            rank,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        self.lora_B = nn.Linear(
            rank,
            base_layer.out_features,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        delta = self.lora_B(self.lora_A(self.dropout(inputs).float()))
        return base + delta.to(base.dtype) * self.scaling


def apply_lora(
    model: nn.Module,
    *,
    target_modules: Iterable[str],
    rank: int,
    alpha: int,
    dropout: float,
) -> list[str]:
    targets = set(target_modules)
    for parameter in model.parameters():
        parameter.requires_grad = False
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and "language_model.layers." in name
            and name.rsplit(".", 1)[-1] in targets
        ):
            replacements.append((name, module))
    if not replacements:
        raise RuntimeError("no language-model Linear modules matched LoRA targets")
    replaced = []
    for name, module in replacements:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        replaced.append(name)
    return replaced


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    output = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        output[f"{name}.lora_A.weight"] = (
            module.lora_A.weight.detach().cpu().contiguous()
        )
        output[f"{name}.lora_B.weight"] = (
            module.lora_B.weight.detach().cpu().contiguous()
        )
    if not output:
        raise RuntimeError("model has no LoRA adapter parameters")
    return output


def save_adapter(
    model: nn.Module,
    output_dir: Path,
    *,
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    save_file(adapter_state(model), output_dir / "adapter_model.safetensors")
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_adapter(
    model: nn.Module, adapter_dir: str | Path
) -> dict[str, Any]:
    adapter_dir = Path(adapter_dir)
    config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    replaced = apply_lora(
        model,
        target_modules=config["target_modules"],
        rank=int(config["rank"]),
        alpha=int(config["alpha"]),
        dropout=float(config["dropout"]),
    )
    expected = set(adapter_state(model))
    state = load_file(adapter_dir / "adapter_model.safetensors", device="cpu")
    if set(state) != expected:
        raise RuntimeError(
            "adapter key mismatch: "
            f"missing={sorted(expected - set(state))[:5]} "
            f"unexpected={sorted(set(state) - expected)[:5]}"
        )
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(
                value.to(
                    device=parameters[name].device,
                    dtype=parameters[name].dtype,
                )
            )
    config["replaced_modules"] = replaced
    return config
