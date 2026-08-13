#!/usr/bin/env python3
"""Run a local OpenAI-compatible Qwen3.6 S2G-style Teacher."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib import request


ALLOWED_CATEGORIES = {
    "bridge entity",
    "attribute",
    "relation",
    "evidence span",
    "other",
}
GAP_KEYS = {"category", "target", "slot", "description"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
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
    gap_items = parsed["gap_items"]
    if not isinstance(sufficient, bool) or not isinstance(gap_items, list):
        return None, "top_level_types"
    if sufficient and gap_items:
        return None, "sufficient_requires_empty_gap_items"
    if not sufficient and not gap_items:
        return None, "insufficient_requires_nonempty_gap_items"
    for item in gap_items:
        if not isinstance(item, dict) or set(item) != GAP_KEYS:
            return None, "gap_item_schema"
        if not all(isinstance(item[key], str) and item[key].strip() for key in GAP_KEYS):
            return None, "gap_item_types_or_empty"
        if item["category"] not in ALLOWED_CATEGORIES:
            return None, "gap_item_category"
    return parsed, None


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def infer_sync(
    *,
    row: dict[str, Any],
    sequence_index: int,
    protocol: dict[str, Any],
    base_url: str,
    model: str,
    timeout: float,
    max_tokens: int,
    json_schema: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    teacher = protocol["teacher"]
    user_text = teacher["user_template"].format(
        question=row["question"],
        context=row["context"],
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": teacher["system_prompt"]},
            {"role": "user", "content": user_text},
        ],
        "temperature": float(teacher["temperature"]),
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "s2g_judge",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sufficient", "gap_items"],
                    "properties": {
                        "sufficient": {"type": "boolean"},
                        "gap_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "category",
                                    "target",
                                    "slot",
                                    "description",
                                ],
                                "properties": {
                                    "category": {
                                        "type": "string",
                                        "enum": sorted(ALLOWED_CATEGORIES),
                                    },
                                    "target": {"type": "string"},
                                    "slot": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        }
    response = post_json(
        base_url.rstrip("/") + "/chat/completions",
        payload,
        timeout,
    )
    choice = response["choices"][0]
    raw = str(choice["message"].get("content") or "")
    parsed, parse_error = strict_parse(raw)
    usage = response.get("usage") or {}
    return {
        "record_type": "teacher_result",
        "sequence_index": sequence_index,
        "source_sequence_index": int(row["sequence_index"]),
        "request_id": row["request_id"],
        "question_id": row["question_id"],
        "state_id": row["state_id"],
        "state_index": int(row["state_index"]),
        "raw_completion": raw,
        "parsed": parsed,
        "parse_error": parse_error,
        "generation_error": None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_seconds": time.monotonic() - started,
    }


async def main_async(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = read_jsonl(args.inputs)
    expected_sha = (
        protocol["sources"]["smoke_inputs_sha256"]
        if args.phase == "smoke"
        else protocol["sources"]["teacher_inputs_sha256"]
    )
    if sha256_file(args.inputs) != expected_sha:
        raise RuntimeError("Teacher input SHA256 mismatch")
    if args.request_ids_file is not None:
        requested = {
            line.strip()
            for line in args.request_ids_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
        rows = [row for row in rows if str(row["request_id"]) in requested]
        if len(rows) != len(requested):
            raise RuntimeError("repair request IDs are missing or duplicated")
    forbidden = {
        "action_label",
        "gold",
        "gold_answer",
        "official_em",
        "prediction",
        "answer",
    }
    if any(set(row) & forbidden for row in rows):
        raise RuntimeError("Teacher inputs contain forbidden fields")

    health = await asyncio.to_thread(
        post_json,
        args.base_url.rstrip("/") + "/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "temperature": 0,
            "max_tokens": 4,
            "stream": False,
        },
        args.timeout,
    )
    if not health.get("choices"):
        raise RuntimeError("Teacher endpoint preflight returned no choices")

    append_jsonl(
        args.output,
        {
            "record_type": "run_start",
            "phase": args.phase,
            "started_at": utc_now(),
            "model": args.model,
            "expected_count": len(rows),
            "inputs_sha256": sha256_file(args.inputs),
            "protocol_sha256": sha256_file(args.protocol),
            "labels_mounted": False,
            "gold_fields_received": 0,
            "concurrency": args.concurrency,
            "json_schema": args.json_schema,
            "max_tokens": (
                args.max_tokens
                if args.max_tokens is not None
                else int(protocol["teacher"]["max_new_tokens"])
            ),
            "request_ids_file_sha256": (
                sha256_file(args.request_ids_file)
                if args.request_ids_file is not None
                else None
            ),
        },
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def infer(index: int, row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            try:
                result = await asyncio.to_thread(
                    infer_sync,
                    row=row,
                    sequence_index=index,
                    protocol=protocol,
                    base_url=args.base_url,
                    model=args.model,
                    timeout=args.timeout,
                    max_tokens=(
                        args.max_tokens
                        if args.max_tokens is not None
                        else int(protocol["teacher"]["max_new_tokens"])
                    ),
                    json_schema=args.json_schema,
                )
            except Exception as exc:
                result = {
                    "record_type": "teacher_result",
                    "sequence_index": index,
                    "source_sequence_index": int(row["sequence_index"]),
                    "request_id": row["request_id"],
                    "question_id": row["question_id"],
                    "state_id": row["state_id"],
                    "state_index": int(row["state_index"]),
                    "raw_completion": None,
                    "parsed": None,
                    "parse_error": None,
                    "generation_error": f"{type(exc).__name__}:{exc}",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "latency_seconds": None,
                }
            print(
                json.dumps(
                    {
                        "done_index": index,
                        "request_id": row["request_id"],
                        "parsed": result["parsed"] is not None,
                        "generation_error": result["generation_error"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return result

    started = time.monotonic()
    tasks = [infer(index, row) for index, row in enumerate(rows)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda row: int(row["sequence_index"]))
    for result in results:
        append_jsonl(args.output, result)
    errors = sum(result["generation_error"] is not None for result in results)
    parsed = sum(result["parsed"] is not None for result in results)
    append_jsonl(
        args.output,
        {
            "record_type": "run_end",
            "phase": args.phase,
            "completed_at": utc_now(),
            "status": (
                "complete"
                if not errors and len(results) == len(rows)
                else "failed"
            ),
            "expected_count": len(rows),
            "completed_count": len(results),
            "parsed_count": parsed,
            "parse_rate": parsed / len(rows) if rows else 0.0,
            "generation_error_count": errors,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    return 0 if errors == 0 and len(results) == len(rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--phase", choices=("smoke", "full", "repair"), required=True
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--request-ids-file", type=Path)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--json-schema", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
