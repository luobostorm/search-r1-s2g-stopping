#!/usr/bin/env python3
"""Train scale-up Qwen3.5-2B S2G LoRA with grouped validation.

The natural training rows remain immutable.  Each epoch deterministically
oversamples only the minority Teacher class to the majority count.  Validation
is never balanced, and the frozen adapter is selected solely by the lowest
natural-distribution token NLL (ties go to the earliest epoch).
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Any


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


def deterministic_balanced_indices(
    labels: list[bool], *, seed: int, epoch: int
) -> list[int]:
    """Return a deterministic, shuffled, class-balanced epoch schedule."""
    by_class = {
        False: [index for index, value in enumerate(labels) if not value],
        True: [index for index, value in enumerate(labels) if value],
    }
    if not by_class[False] or not by_class[True]:
        raise ValueError("both Teacher classes are required")
    target = max(len(by_class[False]), len(by_class[True]))
    rng = random.Random(f"{seed}:epoch:{epoch}")
    expanded: dict[bool, list[int]] = {}
    for label in (False, True):
        values: list[int] = []
        while len(values) < target:
            cycle = list(by_class[label])
            rng.shuffle(cycle)
            values.extend(cycle)
        expanded[label] = values[:target]
    schedule = expanded[False] + expanded[True]
    rng.shuffle(schedule)
    return schedule


def select_best_epoch(records: list[dict[str, Any]]) -> int:
    if not records:
        raise ValueError("no epoch records")
    for record in records:
        value = float(record["validation_token_nll"])
        if value != value or abs(value) == float("inf"):
            raise ValueError("non-finite validation NLL")
    return int(
        min(
            records,
            key=lambda row: (
                float(row["validation_token_nll"]),
                int(row["epoch"]),
            ),
        )["epoch"]
    )


def select_smoke_rows(
    rows: list[dict[str, Any]], *, total: int
) -> list[dict[str, Any]]:
    """Select an equal, deterministic two-class engineering smoke."""
    if total <= 0 or total % 2:
        raise ValueError("smoke sample count must be positive and even")
    per_class = total // 2
    selected: dict[bool, list[dict[str, Any]]] = {
        False: [],
        True: [],
    }
    for row in rows:
        label = bool(row["teacher_sufficient"])
        if len(selected[label]) < per_class:
            selected[label].append(row)
        if all(len(values) == per_class for values in selected.values()):
            break
    if not all(len(values) == per_class for values in selected.values()):
        raise RuntimeError("insufficient rows for balanced smoke")
    chosen_ids = {
        str(row["request_id"])
        for values in selected.values()
        for row in values
    }
    return [
        row for row in rows if str(row["request_id"]) in chosen_ids
    ]


def hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--train-inputs", required=True, type=Path)
    parser.add_argument(
        "--validation-inputs", required=True, type=Path
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    # Heavy runtime imports are intentionally delayed so CPU-only protocol
    # tests can import the deterministic helpers.
    import torch
    from torch.utils.data import DataLoader, Dataset
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from qwen35_causal_lora import apply_lora, save_adapter

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    training = protocol["training"]
    if (
        sha256_file(args.train_inputs)
        != protocol["sources"]["train_inputs_sha256"]
    ):
        raise RuntimeError("train input SHA256 mismatch")
    if (
        sha256_file(args.validation_inputs)
        != protocol["sources"]["validation_inputs_sha256"]
    ):
        raise RuntimeError("validation input SHA256 mismatch")
    train_rows = read_jsonl(args.train_inputs)
    validation_rows = read_jsonl(args.validation_inputs)
    if args.mode == "smoke":
        if (
            training.get("smoke_selection")
            != "first_equal_per_class_in_natural_order"
        ):
            raise RuntimeError("unsupported smoke selection rule")
        train_rows = select_smoke_rows(
            train_rows, total=int(training["smoke_samples"])
        )
        validation_rows = validation_rows[: int(training["smoke_samples"])]
        epochs = int(training["smoke_epochs"])
    else:
        epochs = int(training["epochs"])
    if {
        row["question_id"] for row in train_rows
    } & {row["question_id"] for row in validation_rows}:
        raise RuntimeError("train/validation question overlap")
    args.output_dir.mkdir(parents=True)

    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required")

    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True
    )

    class S2GDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows
            self.examples: list[dict[str, torch.Tensor]] = []
            tokenizer = processor.tokenizer
            for row in rows:
                prompt = processor.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": protocol["system_prompt"],
                        },
                        {"role": "user", "content": row["user_text"]},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
                prompt_ids = tokenizer.encode(
                    prompt, add_special_tokens=False
                )
                target_ids = tokenizer.encode(
                    str(row["target_json"]), add_special_tokens=False
                )
                if tokenizer.eos_token_id is not None:
                    target_ids.append(int(tokenizer.eos_token_id))
                total_length = len(prompt_ids) + len(target_ids)
                if total_length > int(training["max_length"]):
                    raise RuntimeError(
                        f"prompt_overlength:{row['request_id']}:"
                        f"{total_length}>{training['max_length']}"
                    )
                self.examples.append(
                    {
                        "input_ids": torch.tensor(
                            prompt_ids + target_ids, dtype=torch.long
                        ),
                        "attention_mask": torch.ones(
                            total_length, dtype=torch.long
                        ),
                        "labels": torch.tensor(
                            [-100] * len(prompt_ids) + target_ids,
                            dtype=torch.long,
                        ),
                    }
                )

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return self.examples[index]

    def collate(batch: list[dict[str, torch.Tensor]]):
        width = max(len(row["input_ids"]) for row in batch)
        output = {}
        for key, fill in (
            ("input_ids", processor.tokenizer.pad_token_id),
            ("attention_mask", 0),
            ("labels", -100),
        ):
            output[key] = torch.stack(
                [
                    torch.nn.functional.pad(
                        row[key],
                        (0, width - len(row[key])),
                        value=fill,
                    )
                    for row in batch
                ]
            )
        return output

    train_dataset = S2GDataset(train_rows)
    validation_dataset = S2GDataset(validation_rows)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    replaced_modules = apply_lora(
        model,
        rank=int(training["lora_r"]),
        alpha=int(training["lora_alpha"]),
        dropout=float(training["lora_dropout"]),
        target_modules=list(training["target_modules"]),
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    micro_batch_size = int(training["micro_batch_size"])
    history_path = args.output_dir / "history.jsonl"
    epoch_records: list[dict[str, Any]] = []
    all_losses: list[float] = []
    global_step = 0
    optimizer_step = 0
    started = time.monotonic()

    def append_history(record: dict[str, Any]) -> None:
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    def validation_nll() -> tuple[float, int]:
        loader = DataLoader(
            validation_dataset,
            batch_size=micro_batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        model.eval()
        weighted_loss = 0.0
        target_tokens = 0
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to("cuda") for key, value in batch.items()}
                outputs = model(**batch)
                if not torch.isfinite(outputs.loss):
                    raise RuntimeError("non_finite_validation_loss")
                tokens = int((batch["labels"] != -100).sum().item())
                weighted_loss += float(outputs.loss.detach().cpu()) * tokens
                target_tokens += tokens
        if target_tokens == 0:
            raise RuntimeError("validation_has_zero_target_tokens")
        return weighted_loss / target_tokens, target_tokens

    labels = [bool(row["teacher_sufficient"]) for row in train_rows]
    natural_class_counts = Counter(str(value) for value in labels)
    for epoch in range(1, epochs + 1):
        schedule = deterministic_balanced_indices(
            labels, seed=seed, epoch=epoch
        )
        effective_counts = Counter(str(labels[index]) for index in schedule)
        loader = DataLoader(
            train_dataset,
            batch_size=micro_batch_size,
            sampler=schedule,
            collate_fn=collate,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_batches = 0
        epoch_losses: list[float] = []
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            outputs = model(**batch)
            if not torch.isfinite(outputs.loss):
                raise RuntimeError("non_finite_training_loss")
            (outputs.loss / accumulation).backward()
            raw_loss = float(outputs.loss.detach().cpu())
            all_losses.append(raw_loss)
            epoch_losses.append(raw_loss)
            global_step += 1
            accumulated_batches += 1
            should_step = (
                accumulated_batches == accumulation
                or batch_index + 1 == len(loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    (
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    float(training["max_grad_norm"]),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                accumulated_batches = 0
            append_history(
                {
                    "record_type": "train_step",
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "loss": raw_loss,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )

        val_nll, val_tokens = validation_nll()
        checkpoint = args.output_dir / "checkpoints" / f"epoch-{epoch:03d}"
        adapter_dir = checkpoint / "adapter"
        save_adapter(
            model,
            adapter_dir,
            config={
                "format": "kstar-minimal-lora-v1",
                "rank": int(training["lora_r"]),
                "alpha": int(training["lora_alpha"]),
                "dropout": float(training["lora_dropout"]),
                "target_modules": list(training["target_modules"]),
                "replaced_modules": replaced_modules,
            },
        )
        hashes = hash_tree(adapter_dir)
        record = {
            "record_type": "epoch_end",
            "epoch": epoch,
            "training_mean_batch_loss": sum(epoch_losses)
            / len(epoch_losses),
            "validation_token_nll": val_nll,
            "validation_target_tokens": val_tokens,
            "natural_train_class_counts": dict(
                sorted(natural_class_counts.items())
            ),
            "effective_train_class_counts": dict(
                sorted(effective_counts.items())
            ),
            "effective_train_samples": len(schedule),
            "checkpoint_path": str(checkpoint),
            "adapter_file_hashes": hashes,
            "adapter_hashes_sha256": hashlib.sha256(
                json.dumps(
                    hashes, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "elapsed_seconds": time.monotonic() - started,
        }
        epoch_records.append(record)
        append_history(record)

    selected_epoch = select_best_epoch(epoch_records)
    selected_checkpoint = (
        args.output_dir / "checkpoints" / f"epoch-{selected_epoch:03d}"
    )
    frozen_adapter = args.output_dir / "adapter"
    shutil.copytree(selected_checkpoint / "adapter", frozen_adapter)
    processor.save_pretrained(args.output_dir / "processor")
    frozen_hashes = hash_tree(frozen_adapter)
    manifest = {
        "schema_version": 1,
        "artifact_id": "searchr1-s2g-scaleup-qwen35-2b-lora-run-v1",
        "mode": args.mode,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(args.protocol),
        "training_inputs_sha256": sha256_file(args.train_inputs),
        "validation_inputs_sha256": sha256_file(args.validation_inputs),
        "natural_train_row_count": len(train_rows),
        "natural_train_question_count": len(
            {str(row["question_id"]) for row in train_rows}
        ),
        "validation_row_count": len(validation_rows),
        "validation_question_count": len(
            {str(row["question_id"]) for row in validation_rows}
        ),
        "epochs": epochs,
        "epoch_records": epoch_records,
        "checkpoint_selection_rule": (
            "minimum_natural_validation_token_nll_tie_earliest"
        ),
        "selected_epoch": selected_epoch,
        "selected_checkpoint_path": str(selected_checkpoint),
        "frozen_adapter_file_hashes": frozen_hashes,
        "frozen_adapter_hashes_sha256": hashlib.sha256(
            json.dumps(
                frozen_hashes, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "global_steps": global_step,
        "optimizer_steps": optimizer_step,
        "initial_loss": all_losses[0],
        "final_loss": all_losses[-1],
        "minimum_loss": min(all_losses),
        "maximum_loss": max(all_losses),
        "all_losses_finite": all(
            value == value and abs(value) != float("inf")
            for value in all_losses
        ),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "replaced_module_count": len(replaced_modules),
        "elapsed_seconds": time.monotonic() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    (args.output_dir / "checkpoint-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
