from __future__ import annotations

import json
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


def _normalize(raw: dict) -> dict:
    item = dict(raw)
    item["id"] = str(raw.get("id") or raw.get("uid") or "").strip()
    if not item["id"]:
        raise ValueError("rubrictext item requires a non-empty id")
    item["task"] = str(raw.get("task") or raw.get("prompt") or "").strip()
    if not item["task"]:
        raise ValueError(f"rubrictext item {item['id']!r} requires task/prompt")
    item["reference_context"] = str(raw.get("reference_context") or raw.get("reference") or "")
    item["criteria"] = list(raw.get("criteria") or [])
    item["required_terms"] = [str(x) for x in (raw.get("required_terms") or [])]
    item["forbidden_terms"] = [str(x) for x in (raw.get("forbidden_terms") or [])]
    item["task_type"] = str(raw.get("task_type") or "rubrictext")
    item["minimum_score"] = float(raw.get("minimum_score", 0.80))
    return item


class RubricTextDataLoader(SplitDataLoader):
    """Load generic rubric-scored text tasks from JSON files."""

    def load_split_items(self, split_path: str) -> list[dict]:
        json_files = sorted(Path(split_path).glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json file found in {split_path}")
        with json_files[0].open(encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"Expected a JSON array in {json_files[0]}")
        return [_normalize(item) for item in raw]
