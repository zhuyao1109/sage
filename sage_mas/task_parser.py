"""Task parsing helpers for ALFWorld specialists."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


TASK_FAMILY_OPERATIONS = {
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "look_at_obj_in_light": "inspect_light",
    "pick_two_obj_and_place": "pick_two",
}


@dataclass(frozen=True, slots=True)
class ParsedTask:
    task: str
    gamefile: str
    task_family: str
    operation: str | None
    target: str | None
    destination: str | None


def parse_alfworld_task(task: str, gamefile: str = "") -> ParsedTask:
    """Parse operation, target object, and destination from task text/gamefile.

    ALFWorld gamefile parent directories encode the goal as
    ``family-Object-None-Receptacle-id``. The human task text is still the
    preferred source because it is what the actor sees, but the path provides a
    stable fallback when wording varies.
    """

    family, path_target, path_destination = _parse_gamefile_goal(gamefile)
    operation = TASK_FAMILY_OPERATIONS.get(family)
    text = " ".join(str(task or "").lower().split())

    text_operation, text_target, text_destination = _parse_text_goal(text)
    return ParsedTask(
        task=str(task or ""),
        gamefile=str(gamefile or ""),
        task_family=family,
        operation=operation or text_operation,
        target=_first_non_empty(text_target, path_target),
        destination=_first_non_empty(text_destination, path_destination),
    )


def normalize_entity(text: str | None) -> str | None:
    """Normalize ALFWorld entity names to action-token form.

    Examples: ``ButterKnife`` -> ``butterknife`` and ``coffee machine`` ->
    ``coffeemachine``.
    """

    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _parse_text_goal(text: str) -> tuple[str | None, str | None, str | None]:
    operation = None
    target = None
    destination = None

    patterns = (
        (
            "clean",
            re.compile(
                r"\bclean\s+(?:some|a|the)?\s*([a-z][a-z ]*?)\s+"
                r"and\s+put\s+it\s+(?:in|on|in/on)\s+(?:a|the)?\s*"
                r"([a-z][a-z ]*?)(?:[.?!]|$)"
            ),
        ),
        (
            "clean",
            re.compile(
                r"\bput\s+(?:a|the)?\s*clean\s+([a-z][a-z ]*?)\s+"
                r"(?:in|on|in/on)\s+(?:a|the)?\s*"
                r"([a-z][a-z ]*?)(?:[.?!]|$)"
            ),
        ),
        (
            "heat",
            re.compile(
                r"\bput\s+(?:a|the)?\s*hot\s+([a-z][a-z ]*?)\s+"
                r"(?:in|on|in/on)\s+(?:a|the)?\s*"
                r"([a-z][a-z ]*?)(?:[.?!]|$)"
            ),
        ),
        (
            "cool",
            re.compile(
                r"\bput\s+(?:a|the)?\s*cool\s+([a-z][a-z ]*?)\s+"
                r"(?:in|on|in/on)\s+(?:a|the)?\s*"
                r"([a-z][a-z ]*?)(?:[.?!]|$)"
            ),
        ),
    )
    for candidate_operation, pattern in patterns:
        match = pattern.search(text)
        if match:
            operation = candidate_operation
            target = normalize_entity(match.group(1))
            destination = normalize_entity(match.group(2))
            break
    return operation, target, destination


def _parse_gamefile_goal(gamefile: str) -> tuple[str, str | None, str | None]:
    path = Path(str(gamefile or ""))
    parts = list(path.parts)
    goal_part = ""
    for part in reversed(parts):
        if (
            (part.startswith("pick_") and "_then_place_in_recep-" in part)
            or part.startswith("look_at_obj_in_light-")
            or part.startswith("pick_two_obj_and_place-")
        ):
            goal_part = part
            break
    if not goal_part:
        return "", None, None

    segments = goal_part.split("-")
    if len(segments) < 4:
        return segments[0], None, None
    return (
        segments[0],
        normalize_entity(segments[1]),
        normalize_entity(segments[3]),
    )


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        normalized = normalize_entity(value)
        if normalized:
            return normalized
    return None
