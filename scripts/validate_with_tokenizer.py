#!/usr/bin/env python3
"""Validate converted dataset using the FunctionGemma chat template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = lambda x, **_: x  # type: ignore

Json = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dataset_functiongemma.jsonl with tokenizer.apply_chat_template."
    )
    parser.add_argument(
        "--dataset-path",
        default="output/dataset_functiongemma.jsonl",
        help="Path to the converted dataset JSONL.",
    )
    parser.add_argument(
        "--model",
        default="google/function-gemma-2b",
        help="Model name to load tokenizer from.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on records for quick checks.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error instead of collecting all failures.",
    )
    return parser.parse_args()


def load_records(path: Path, limit: int | None) -> List[Json]:
    records: List[Json] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def normalize_tool_calls(tool_calls: Any) -> List[Json]:
    """Adapt tool_calls to the structure expected by chat templates."""
    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list")
    normalized = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise ValueError("tool_call entry must be an object")
        # Accept minimal {name, arguments} schema and wrap to function style.
        if "function" in call:
            fn_block = call["function"]
            if not isinstance(fn_block, dict) or "name" not in fn_block or "arguments" not in fn_block:
                raise ValueError("function block must contain name and arguments")
            normalized.append({"type": call.get("type", "function"), "function": fn_block})
        else:
            if "name" not in call or "arguments" not in call:
                raise ValueError("tool_call must contain name and arguments")
            normalized.append(
                {"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
            )
    return normalized


def prepare_messages(messages: Any) -> List[Json]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    prepared: List[Json] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("each message must be an object")
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported role: {role}")
        prepared_msg = dict(msg)
        if role == "assistant" and "tool_calls" in msg:
            prepared_msg["tool_calls"] = normalize_tool_calls(msg["tool_calls"])
        prepared.append(prepared_msg)
    return prepared


def validate_record(idx: int, record: Json, tokenizer: AutoTokenizer) -> tuple[bool, str | None]:
    if not isinstance(record, dict):
        return False, "record must be a JSON object"
    if "messages" not in record:
        return False, "missing messages"
    try:
        prepared = prepare_messages(record["messages"])
        # apply_chat_template should not rais
        tokenizer.apply_chat_template(prepared, tokenize=False)
    except Exception as exc:  # broad on purpose to surface tokenizer issues
        return False, f"chat template failure: {exc}"
    return True, None


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = load_records(dataset_path, args.max_records)
    if not records:
        print("Dataset is empty; nothing to validate.", file=sys.stderr)
        sys.exit(1)

    failures: List[str] = []
    for idx, record in enumerate(tqdm(records, desc="Validating")):
        ok, error = validate_record(idx, record, tokenizer)
        if not ok:
            failures.append(f"Record {idx}: {error}")
            if args.fail_fast:
                break

    if failures:
        print("Validation FAILED. Issues:", file=sys.stderr)
        for issue in failures[:20]:
            print(f"- {issue}", file=sys.stderr)
        if len(failures) > 20:
            print(f"... {len(failures) - 20} more", file=sys.stderr)
        sys.exit(1)

    print(f"Validation passed for {len(records)} records using {args.model}")


if __name__ == "__main__":
    main()
