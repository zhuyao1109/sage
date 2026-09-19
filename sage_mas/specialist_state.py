"""State reconstruction for staged ALFWorld specialists."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sage_mas.task_parser import ParsedTask, normalize_entity


ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)


@dataclass(slots=True)
class SpecialistState:
    phase: str
    operation: str | None
    target: str | None
    destination: str | None
    held_target: bool = False
    operation_done: bool = False
    placed_target: bool = False
    target_visible: bool = False
    destination_visible: bool = False
    sink_visible: bool = False
    recent_actions: list[str] = field(default_factory=list)
    visited_locations: set[str] = field(default_factory=set)
    opened_receptacles: set[str] = field(default_factory=set)
    invalid_actions: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "operation": self.operation,
            "target": self.target,
            "destination": self.destination,
            "held_target": self.held_target,
            "operation_done": self.operation_done,
            "placed_target": self.placed_target,
            "target_visible": self.target_visible,
            "destination_visible": self.destination_visible,
            "sink_visible": self.sink_visible,
            "recent_actions": list(self.recent_actions),
            "visited_locations": sorted(self.visited_locations),
            "opened_receptacles": sorted(self.opened_receptacles),
            "invalid_actions": sorted(self.invalid_actions),
        }


def infer_specialist_state(
    parsed: ParsedTask,
    *,
    steps: list[dict[str, Any]],
    observation: str = "",
    admissible_actions: list[str] | None = None,
) -> SpecialistState:
    """Infer the clean/cool/heat phase from recorded environment steps."""

    target = parsed.target
    destination = parsed.destination
    operation = parsed.operation
    held_target = False
    operation_done = False
    placed_target = False
    recent_actions: list[str] = []
    visited_locations: set[str] = set()
    opened_receptacles: set[str] = set()
    invalid_actions: set[str] = set()

    for step in steps:
        action = extract_action_text(step.get("action", ""))
        if not action:
            continue
        valid = bool(step.get("is_action_valid", True))
        recent_actions.append(action)
        if action.startswith("go to "):
            visited_locations.add(_strip_action_prefix(action, "go to "))
        if action.startswith("open ") and valid:
            opened_receptacles.add(_strip_action_prefix(action, "open "))
        if not valid:
            invalid_actions.add(action)
            continue
        observation_after = str(step.get("observation", "")).lower()
        if target and action.startswith("take ") and contains_entity(action, target):
            held_target = True
            placed_target = False
        if target and action.startswith(("put ", "move ")) and contains_entity(action, target):
            held_target = False
            if destination is None or contains_entity(action, destination):
                placed_target = True
        if (
            operation
            and target
            and (
                action.startswith(f"{operation} ")
                or f"you {operation} " in observation_after
            )
            and contains_entity(action + " " + observation_after, target)
        ):
            operation_done = True

    actions = [extract_action_text(action) for action in admissible_actions or []]
    compact_observation = normalize_entity(observation) or ""
    target_visible = bool(
        target
        and (
            any(action.startswith("take ") and contains_entity(action, target) for action in actions)
            or target in compact_observation
        )
    )
    destination_visible = bool(
        destination
        and (
            any(contains_entity(action, destination) for action in actions)
            or destination in compact_observation
        )
    )
    sink_visible = any("sinkbasin" in action or "sink" in action for action in actions) or (
        "sinkbasin" in compact_observation or "sink" in compact_observation
    )

    if placed_target:
        phase = "VERIFY"
    elif not held_target:
        phase = "TAKE_TARGET" if target_visible else "SEARCH_TARGET"
    elif not operation_done:
        phase = f"{str(operation or 'operation').upper()}_TARGET"
    else:
        phase = "PLACE_TARGET" if destination_visible else "GO_TO_DESTINATION"

    return SpecialistState(
        phase=phase,
        operation=operation,
        target=target,
        destination=destination,
        held_target=held_target,
        operation_done=operation_done,
        placed_target=placed_target,
        target_visible=target_visible,
        destination_visible=destination_visible,
        sink_visible=sink_visible,
        recent_actions=recent_actions[-8:],
        visited_locations=visited_locations,
        opened_receptacles=opened_receptacles,
        invalid_actions=invalid_actions,
    )


def extract_action_text(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    match = ACTION_TAG_RE.search(text)
    if match:
        text = match.group(1)
    return " ".join(text.lower().strip().split())


def contains_entity(text: str, entity: str | None) -> bool:
    normalized = normalize_entity(text) or ""
    target = normalize_entity(entity) or ""
    return bool(target and target in normalized)


def _strip_action_prefix(action: str, prefix: str) -> str:
    return action[len(prefix) :].strip()
