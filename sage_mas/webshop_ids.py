"""Stable task IDs for WebShop SAGE online (goal_index ↔ URI)."""

from __future__ import annotations

import re

_GOAL_URI_RE = re.compile(r"^webshop://goal/(\d+)$", re.IGNORECASE)


def goal_uri(goal_idx: int) -> str:
    return f"webshop://goal/{int(goal_idx)}"


def parse_goal_idx(task_id: str | int) -> int:
    if isinstance(task_id, int):
        return int(task_id)
    text = str(task_id or "").strip()
    match = _GOAL_URI_RE.match(text)
    if match:
        return int(match.group(1))
    if text.isdigit():
        return int(text)
    raise ValueError(f"Not a WebShop goal id: {task_id!r}")


def goal_indices_from_task_ids(task_ids: list[str]) -> list[int]:
    return [parse_goal_idx(task_id) for task_id in task_ids]
