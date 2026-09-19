"""Idle / empty-spin trajectory filters (domain-agnostic).

A failure is useful for compilation only when it shows some productive
diversity. Pure repetition / single-action loops are treated as idle noise.
"""

from __future__ import annotations

from typing import Any, Sequence

from sage_mas.trajectory.abstraction import (
    abstract_environment_action,
    stage_for_abstracted_action,
)


def _actions(record: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for step in record.get("steps") or []:
        if isinstance(step, dict):
            action = str(step.get("action") or "").strip()
        else:
            action = str(step or "").strip()
        if action:
            actions.append(action)
    return actions


def abstract_productive_protocol(record: dict[str, Any]) -> list[str]:
    protocol: list[str] = []
    for action in _actions(record):
        abstracted = abstract_environment_action(action)
        if stage_for_abstracted_action(abstracted) == "other":
            continue
        if abstracted in {"look", "inventory", "examine <entity>"}:
            continue
        if protocol and protocol[-1] == abstracted:
            continue
        protocol.append(abstracted)
    return protocol


def idle_failure_metrics(record: dict[str, Any]) -> dict[str, float | int | bool]:
    """Compute repetition / diversity stats for a trajectory record."""
    actions = _actions(record)
    protocol = abstract_productive_protocol(record)
    n = len(actions)
    if n <= 0:
        return {
            "n_actions": 0,
            "n_productive": 0,
            "unique_productive": 0,
            "max_run": 0,
            "repeat_ratio": 1.0,
            "is_idle": True,
        }

    # Longest consecutive identical raw action run.
    max_run = 1
    run = 1
    for index in range(1, n):
        if actions[index] == actions[index - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1

    unique = len(set(protocol))
    # Fraction of steps that merely repeat the previous abstract step.
    repeats = 0
    prev = None
    for action in actions:
        abstracted = abstract_environment_action(action)
        if prev is not None and abstracted == prev:
            repeats += 1
        prev = abstracted
    repeat_ratio = repeats / max(1, n - 1) if n > 1 else 1.0

    # Idle if almost no productive diversity or dominated by repetition.
    is_idle = bool(
        unique <= 1
        or (n >= 6 and unique <= 2 and repeat_ratio >= 0.5)
        or (n >= 8 and max_run >= max(4, n // 2))
    )
    return {
        "n_actions": n,
        "n_productive": len(protocol),
        "unique_productive": unique,
        "max_run": max_run,
        "repeat_ratio": float(repeat_ratio),
        "is_idle": is_idle,
    }


def is_idle_failure(record: dict[str, Any]) -> bool:
    if bool(record.get("won") or record.get("success")):
        return False
    return bool(idle_failure_metrics(record)["is_idle"])


def filter_informative_failures(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep losses that are not empty-spin / pure repetition."""
    kept: list[dict[str, Any]] = []
    for record in records:
        if bool(record.get("won") or record.get("success")):
            continue
        if is_idle_failure(record):
            continue
        kept.append(record)
    return kept
