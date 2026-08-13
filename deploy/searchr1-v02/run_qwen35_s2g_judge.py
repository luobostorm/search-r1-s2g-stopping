#!/usr/bin/env python3
"""Generate structured S2G Judge decisions with Base or LoRA weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

from qwen35_causal_lora import load_adapter


ALLOWED_CATEGORIES = {
    "bridge entity",
    "attribute",
    "relation",
    "evidence span",
    "other",
}
GAP_KEYS = {"category", "target", "slot", "description"}
SCORE_PREFIX = '{"sufficient":'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    rows = {
        str(file.relative_to(path)): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def strict_parse(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(raw.strip())
    except Exception as exc:
        return None, f"json_decode:{type(exc).__name__}:{exc}"
    if not isinstance(parsed, dict) or set(parsed) != {
        "sufficient",
        "gap_items",
    }:
        return None, "top_level_schema"
    sufficient = parsed["sufficient"]
    gaps = parsed["gap_items"]
    if not isinstance(sufficient, bool) or not isinstance(gaps, list):
        return None, "top_level_types"
    if sufficient and gaps:
        return None, "sufficient_requires_empty_gap_items"
    if not sufficient and not gaps:
        return None, "insufficient_requires_nonempty_gap_items"
    for item in gaps:
        if not isinstance(item, dict) or set(item) != GAP_KEYS:
            return None, "gap_item_schema"
        if not all(
            isinstance(item[key], str) and item[key].strip()
            for key in GAP_KEYS
        ):
            return None, "gap_item_types_or_empty"
        if item["category"] not in ALLOWED_CATEGORIES:
            return None, "gap_item_category"
    return parsed, None


def binary_prefix_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
) -> list[dict[str, float]]:
    """Score true vs false after the canonical structured-output prefix."""

    candidates = ("true", "false")
    sequences: list[list[int]] = []
    prefix_lengths: list[int] = []
    candidate_lengths: list[int] = []
    for prompt in prompt_texts:
        prefix_ids = tokenizer.encode(
            prompt + SCORE_PREFIX, add_special_tokens=False
        )
        if not prefix_ids:
            raise RuntimeError("empty score prefix")
        for candidate in candidates:
            candidate_ids = tokenizer.encode(
                candidate, add_special_tokens=False
            )
            if not candidate_ids:
                raise RuntimeError(f"empty candidate tokens: {candidate}")
            sequences.append(prefix_ids + candidate_ids)
            prefix_lengths.append(len(prefix_ids))
            candidate_lengths.append(len(candidate_ids))
    width = max(len(ids) for ids in sequences)
    input_ids = torch.full(
        (len(sequences), width),
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, ids in enumerate(sequences):
        input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[index, : len(ids)] = 1
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
        ).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
    values: list[float] = []
    for row_index, ids in enumerate(sequences):
        prefix_length = prefix_lengths[row_index]
        candidate_length = candidate_lengths[row_index]
        value = 0.0
        for offset in range(candidate_length):
            token_position = prefix_length + offset
            token_id = ids[token_position]
            value += float(
                log_probs[
                    row_index, token_position - 1, token_id
                ].detach().cpu()
            )
        values.append(value)
    return [
        {
            "true_logprob": values[offset],
            "false_logprob": values[offset + 1],
            "margin": values[offset] - values[offset + 1],
        }
        for offset in range(0, len(values), 2)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--score-margin",
        action="store_true",
        help=(
            "Record logP(true)-logP(false) after the canonical "
            "structured-output prefix."
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.inputs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_count = len(rows)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    adapter_sha = tree_hash(args.adapter) if args.adapter else None
    append(
        args.output,
        {
            "record_type": "run_start",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "mode": "lora" if args.adapter else "base",
            "expected_count": len(rows),
            "source_count": source_count,
            "labels_mounted": False,
            "gold_fields_received": 0,
            "protocol_sha256": sha256_file(args.protocol),
            "inputs_sha256": sha256_file(args.inputs),
            "adapter_sha256": adapter_sha,
            "batch_size": args.batch_size,
            "score_margin": args.score_margin,
            "score_prefix": SCORE_PREFIX if args.score_margin else None,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    )
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True
    )
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if args.adapter:
        load_adapter(model, args.adapter)
    model = model.to("cuda")
    model.eval()
    started = time.monotonic()
    errors = 0
    parsed_count = 0
    completed = 0
    for batch_start in range(0, len(rows), args.batch_size):
        batch_rows = rows[batch_start : batch_start + args.batch_size]
        batch_started = time.monotonic()
        try:
            texts = []
            for row in batch_rows:
                user_text = row.get("user_text") or (
                    f"QUESTION:\n{row['question']}\n\n"
                    f"CONTEXT:\n"
                    f"{row.get('context', row.get('evidence', ''))}"
                )
                texts.append(
                    processor.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": protocol["system_prompt"],
                            },
                            {"role": "user", "content": user_text},
                        ],
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                    )
                )
            encoded = processor.tokenizer(
                texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            prompt_width = int(encoded["input_ids"].shape[-1])
            prompt_lengths = [
                int(value)
                for value in encoded["attention_mask"].sum(dim=1).tolist()
            ]
            if prompt_width > int(protocol["training"]["max_length"]):
                raise RuntimeError(
                    f"prompt_overlength:{prompt_width}>"
                    f"{protocol['training']['max_length']}"
                )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=int(
                        protocol["inference"]["max_new_tokens"]
                    ),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            margins = (
                binary_prefix_logprobs(
                    model=model,
                    tokenizer=processor.tokenizer,
                    prompt_texts=texts,
                )
                if args.score_margin
                else [None] * len(batch_rows)
            )
            generated_batch = outputs[:, prompt_width:]
            batch_elapsed = time.monotonic() - batch_started
            for offset, (row, generated, prompt_tokens, margin) in enumerate(
                zip(
                    batch_rows,
                    generated_batch,
                    prompt_lengths,
                    margins,
                )
            ):
                index = batch_start + offset
                raw = processor.decode(
                    generated, skip_special_tokens=True
                )
                parsed, parse_error = strict_parse(raw)
                if parsed is not None:
                    parsed_count += 1
                sufficient = (
                    bool(parsed["sufficient"])
                    if parsed is not None
                    else False
                )
                result = {
                    "record_type": "judge_score",
                    "sequence_index": index,
                    "request_id": row["request_id"],
                    "question_id": row["question_id"],
                    "state_id": row["state_id"],
                    "state_index": int(row["state_index"]),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": int(generated.shape[-1]),
                    "raw_completion": raw,
                    "parsed": parsed,
                    "parse_error": parse_error,
                    "generation_error": None,
                    "score": (
                        float(margin["margin"])
                        if margin is not None
                        else (1.0 if sufficient else -1.0)
                    ),
                    "true_logprob": (
                        float(margin["true_logprob"])
                        if margin is not None
                        else None
                    ),
                    "false_logprob": (
                        float(margin["false_logprob"])
                        if margin is not None
                        else None
                    ),
                    "decision": "STOP" if sufficient else "CONTINUE",
                    "batch_latency_seconds": batch_elapsed,
                    "latency_seconds": batch_elapsed / len(batch_rows),
                }
                append(args.output, result)
                completed += 1
                print(
                    json.dumps(
                        {
                            "done": index + 1,
                            "total": len(rows),
                            "parsed": parsed is not None,
                            "decision": result["decision"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        except Exception as exc:
            errors += 1
            append(
                args.output,
                {
                    "record_type": "judge_error",
                    "sequence_index": batch_start,
                    "request_id": batch_rows[0].get("request_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            break
    append(
        args.output,
        {
            "record_type": "run_end",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": (
                "complete"
                if errors == 0 and completed == len(rows)
                else "failed"
            ),
            "expected_count": len(rows),
            "completed_count": completed,
            "parsed_count": parsed_count,
            "parse_rate": parsed_count / len(rows) if rows else 0.0,
            "error_count": errors,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    return 0 if errors == 0 and completed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
