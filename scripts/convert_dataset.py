#!/usr/bin/env python3
"""Convert an Arabic function-calling dataset into FunctionGemma chat format."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from datasets import Dataset, load_dataset


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
Logger = logging.getLogger(__name__)

Json = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Arabic function-calling dataset into FunctionGemma chat format."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face dataset name or path (e.g., org/name or ./local_path).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--raw-path",
        default="raw/dataset.jsonl",
        help="Path to write the raw dataset copy.",
    )
    parser.add_argument(
        "--output-path",
        default="output/dataset_functiongemma.jsonl",
        help="Path to write the converted chat dataset.",
    )
    parser.add_argument(
        "--registry-path",
        default="schemas/registry.json",
        help="Path to write the generated function registry.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on processed rows for debugging.",
    )
    return parser.parse_args()


def to_jsonl(records: Iterable[Json], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def sort_object(value: Any) -> Any:
    """Recursively sort dictionary keys for deterministic output."""
    if isinstance(value, dict):
        return {key: sort_object(value[key]) for key in sorted(value.keys())}
    if isinstance(value, list):
        return [sort_object(item) for item in value]
    return value


def normalize_arguments(raw_args: Any) -> Optional[Json]:
    """Return arguments as a dictionary or None when unusable."""
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        raw_args = raw_args.strip()
        if not raw_args:
            return {}
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            Logger.warning("Skipping arguments that failed to parse JSON: %s", exc)
            return None
        if isinstance(parsed, dict):
            return parsed
        Logger.warning("Arguments JSON is not an object: %s", type(parsed))
        return None
    Logger.warning("Unsupported arguments type: %s", type(raw_args))
    return None


def infer_schema(value: Any) -> Json:
    """Infer a JSON Schema fragment from a Python value."""
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        items_schema: Optional[Json] = None
        for item in value:
            schema = infer_schema(item)
            items_schema = schema if items_schema is None else merge_schema(items_schema, schema)
        if items_schema:
            return {"type": "array", "items": items_schema}
        return {"type": "array"}
    if isinstance(value, dict):
        properties: Dict[str, Json] = {}
        required: List[str] = []
        for key in sorted(value.keys()):
            properties[key] = infer_schema(value[key])
            if value[key] is not None:
                required.append(key)
        schema: Json = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema
    return {"type": "string"}


def merge_types(left: Union[str, List[str]], right: Union[str, List[str]]) -> List[str]:
    lset: Set[str] = set([left] if isinstance(left, str) else left)
    rset: Set[str] = set([right] if isinstance(right, str) else right)
    merged = sorted(lset | rset)
    return merged if len(merged) > 1 else [merged[0]]


def merge_schema(left: Json, right: Json) -> Json:
    """Merge two JSON Schema fragments."""
    merged: Json = {}
    if "type" in left and "type" in right:
        merged["type"] = merge_types(left["type"], right["type"])
    elif "type" in left:
        merged["type"] = left["type"]
    elif "type" in right:
        merged["type"] = right["type"]

    # Merge object schemas
    if left.get("type") in ("object", ["object"]) or right.get("type") in ("object", ["object"]):
        properties: Dict[str, Json] = {}
        left_props = left.get("properties", {})
        right_props = right.get("properties", {})
        for key in sorted(set(left_props.keys()) | set(right_props.keys())):
            if key in left_props and key in right_props:
                properties[key] = merge_schema(left_props[key], right_props[key])
            elif key in left_props:
                properties[key] = left_props[key]
            else:
                properties[key] = right_props[key]
        merged["properties"] = properties

        left_required = set(left.get("required", []))
        right_required = set(right.get("required", []))
        required = sorted(left_required & right_required)
        if required:
            merged["required"] = required

    # Merge array schemas
    if left.get("type") in ("array", ["array"]) or right.get("type") in ("array", ["array"]):
        left_items = left.get("items")
        right_items = right.get("items")
        if left_items and right_items:
            merged["items"] = merge_schema(left_items, right_items)
        elif left_items:
            merged["items"] = left_items
        elif right_items:
            merged["items"] = right_items

    # Carry additional keys if present on either schema
    for key in set(left.keys()) | set(right.keys()):
        if key in merged:
            continue
        if key in left and key not in right:
            merged[key] = left[key]
        if key in right and key not in left:
            merged[key] = right[key]
    return merged


def clean_schema(schema: Json) -> Json:
    """Sort properties deterministically and collapse single-item type lists."""
    cleaned = dict(schema)
    schema_type = cleaned.get("type")
    if isinstance(schema_type, list) and len(schema_type) == 1:
        cleaned["type"] = schema_type[0]
    if cleaned.get("type") == "object" and "properties" in cleaned:
        properties = cleaned["properties"]
        sorted_props = {}
        for key in sorted(properties.keys()):
            sorted_props[key] = clean_schema(properties[key])
        cleaned["properties"] = sorted_props
        if "required" in cleaned:
            cleaned["required"] = sorted(cleaned["required"])
    if cleaned.get("type") == "array" and "items" in cleaned:
        cleaned["items"] = clean_schema(cleaned["items"])
    return cleaned


def build_function_registry(dataset: Dataset) -> Tuple[List[Json], Json]:
    """Return converted messages and registry."""
    converted_rows: List[Json] = []
    registry: Dict[str, Json] = {}

    for idx, row in enumerate(dataset):
        query = row.get("query")
        function_name = row.get("function_name")
        answer = row.get("answer")
        raw_args = row.get("arguments")

        if not isinstance(query, str) or not query.strip():
            Logger.warning("Skipping row %s: missing/empty query", idx)
            continue

        # Tool case
        if isinstance(function_name, str) and function_name.strip():
            arguments = normalize_arguments(raw_args)
            if arguments is None:
                Logger.warning("Skipping row %s: unusable arguments", idx)
                continue
            arguments = sort_object(arguments)

            messages = [
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"name": function_name, "arguments": arguments},
                    ],
                },
            ]
            converted_rows.append({"messages": messages})

            schema = infer_schema(arguments)
            if function_name in registry:
                registry[function_name] = merge_schema(registry[function_name], schema)
            else:
                registry[function_name] = schema
            continue

        # No tool case
        if isinstance(answer, str) and answer.strip():
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ]
            converted_rows.append({"messages": messages})
        else:
            Logger.warning("Skipping row %s: neither function call nor answer found", idx)

    cleaned_registry = {
        "functions": [
            {
                "name": name,
                "description": "Auto-generated schema from dataset samples.",
                "parameters": clean_schema(schema),
            }
            for name, schema in sorted(registry.items(), key=lambda kv: kv[0])
        ]
    }

    return converted_rows, cleaned_registry


def main() -> None:
    args = parse_args()
    Logger.info("Loading dataset %s split=%s", args.dataset, args.split)
    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_rows:
        dataset = dataset.select(range(min(args.max_rows, len(dataset))))
        Logger.info("Capped dataset to %s rows", len(dataset))

    raw_path = Path(args.raw_path)
    output_path = Path(args.output_path)
    registry_path = Path(args.registry_path)

    Logger.info("Writing raw copy to %s", raw_path)
    dataset.to_json(raw_path.as_posix(), orient="records", force_ascii=False)

    Logger.info("Converting dataset")
    converted_rows, registry = build_function_registry(dataset)

    Logger.info("Writing converted dataset to %s", output_path)
    to_jsonl(converted_rows, output_path)

    Logger.info("Writing function registry to %s", registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    Logger.info(
        "Done. Converted rows: %d | Functions discovered: %d",
        len(converted_rows),
        len(registry["functions"]),
    )


if __name__ == "__main__":
    main()
