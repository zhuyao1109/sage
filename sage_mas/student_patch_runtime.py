"""Runtime binding for student patches: match → fill slots → concrete inject.

The mid-strong actor must see concrete admissible-style actions, not
``<object>/<location>`` placeholders. Trigger matching and slot filling happen
in code; only the bound ``inject_text`` is shown to the model.

Forbidden actions are derived from already-completed protocol steps (do not
repeat), not from hand-written family expert policies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from sage_mas.executable_protocol import (
    action_matches_template,
    parse_admissible_actions,
)
from sage_mas.schemas import Skill
from sage_mas.skill_quality import annotate_skill_identity
from sage_mas.task_parser import parse_alfworld_task


_INVENTORY_RE = re.compile(
    r"you are carrying[:\s]*(nothing|[^\n.]+)",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"<([a-zA-Z_]+)>")
_ENTITY_RE = re.compile(r"\b([a-z][a-z0-9]*)\s+(\d+)\b", re.I)
_VERB_BLOCKLIST = frozenset(
    {
        "go",
        "take",
        "move",
        "put",
        "open",
        "close",
        "cool",
        "heat",
        "clean",
        "use",
        "examine",
        "look",
        "inventory",
        "from",
        "with",
        "to",
        "on",
        "in",
    }
)



@dataclass(slots=True)
class ObservationState:
    task_family: str = ""
    task: str = ""
    gamefile: str = ""
    inventory: str | None = None
    holding: bool = False
    admissible: list[str] = field(default_factory=list)
    target: str | None = None
    destination: str | None = None
    operation: str | None = None


@dataclass(slots=True)
class BoundStudentPatch:
    matched: bool
    skill_name: str
    inject_text: str
    slot_values: dict[str, str]
    concrete_steps: list[str]
    forbidden_actions: list[str]
    current_step_index: int
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "skill_name": self.skill_name,
            "inject_text": self.inject_text,
            "slot_values": dict(self.slot_values),
            "concrete_steps": list(self.concrete_steps),
            "forbidden_actions": list(self.forbidden_actions),
            "current_step_index": self.current_step_index,
            "reason": self.reason,
        }


def extract_observation_state(
    *,
    observation: str,
    task: str = "",
    gamefile: str = "",
    task_family: str | None = None,
) -> ObservationState:
    parsed = parse_alfworld_task(task, gamefile)
    family = str(task_family or parsed.task_family or "").strip()
    admissible = parse_admissible_actions(observation)
    inventory = None
    holding = False
    match = _INVENTORY_RE.search(observation or "")
    if match:
        raw = " ".join(match.group(1).split()).strip().lower()
        if raw and raw != "nothing":
            inventory = raw
            holding = True
    # Fallback: many prompts only expose inventory via "Inventory:" lines.
    if inventory is None:
        for line in str(observation or "").splitlines():
            if "inventory" in line.lower() and ":" in line:
                value = line.split(":", 1)[1].strip().lower()
                if value and value != "nothing":
                    inventory = value
                    holding = True
                break
    return ObservationState(
        task_family=family,
        task=str(task or ""),
        gamefile=str(gamefile or ""),
        inventory=inventory,
        holding=holding,
        admissible=list(admissible),
        target=parsed.target,
        destination=parsed.destination,
        operation=parsed.operation,
    )


def student_patch_of(skill: Skill) -> dict[str, Any] | None:
    md = skill.metadata or {}
    patch = md.get("student_patch")
    return patch if isinstance(patch, dict) else None


def trigger_matches(
    patch: dict[str, Any],
    state: ObservationState,
) -> tuple[bool, str]:
    signature = dict(patch.get("trigger_signature") or {})
    families = [
        str(item).strip()
        for item in (signature.get("task_families") or [])
        if str(item).strip()
    ]
    if families and state.task_family and state.task_family not in families:
        return False, f"family_mismatch:{state.task_family}"
    require_holding = signature.get("require_holding")
    if require_holding is True and not state.holding:
        return False, "require_holding"
    if require_holding is False and state.holding:
        # Optional negative constraint; only enforce when explicitly false.
        pass
    need_admissible = signature.get("require_admissible", True)
    if need_admissible and not state.admissible:
        # Still allow match: some prompts omit the list early; binder degrades.
        pass
    return True, "ok"


def _normalize_entity_token(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _entities_from_admissible(admissible: Sequence[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for action in admissible:
        for match in _ENTITY_RE.finditer(action):
            name = match.group(1).lower()
            if name in _VERB_BLOCKLIST:
                continue
            entity = f"{name} {match.group(2)}"
            key = _normalize_entity_token(entity)
            if key in seen:
                continue
            seen.add(key)
            found.append(entity)
    return found


def _bound_admissible_for_template(
    template: str,
    slots: dict[str, str],
    admissible: Sequence[str],
) -> str | None:
    """Prefer a real admissible action over naive string substitution."""
    bound = bind_template(template, slots)
    if bound and not _PLACEHOLDER_RE.search(bound):
        for action in admissible:
            if action.lower() == bound.lower():
                return action
        for action in admissible:
            if action_matches_template(action, bound):
                return action
    # Verb-guided search with available slot tokens.
    verb = bound.split(" ", 1)[0].lower() if bound else ""
    if not verb:
        return None
    needles = [
        _normalize_entity_token(slots.get(key))
        for key in ("object", "target", "tool", "destination", "receptacle", "location")
        if slots.get(key)
    ]
    needles = [item for item in needles if item]
    best = None
    best_score = -1
    for action in admissible:
        lower = action.lower()
        if not lower.startswith(verb):
            continue
        norm = _normalize_entity_token(action)
        score = sum(1 for needle in needles if needle in norm)
        if score > best_score:
            best_score = score
            best = action
    if best is not None and (best_score > 0 or not needles):
        return best
    return bound if bound and not _PLACEHOLDER_RE.search(bound) else None


def _prefer_entity(
    candidates: Sequence[str],
    *,
    prefer: str | None,
) -> str | None:
    if not candidates:
        return None
    prefer_key = _normalize_entity_token(prefer)
    if prefer_key:
        for item in candidates:
            token = _normalize_entity_token(item)
            if token.startswith(prefer_key) or prefer_key in token:
                return item
    return candidates[0]


def _find_admissible(
    admissible: Sequence[str],
    *,
    verb: str,
    must_contain: Sequence[str] = (),
) -> str | None:
    verb = verb.lower().strip()
    needles = [_normalize_entity_token(item) for item in must_contain if item]
    ranked: list[str] = []
    for action in admissible:
        text = str(action).strip().lower()
        if not text.startswith(verb):
            continue
        norm = _normalize_entity_token(text)
        if needles and not all(needle in norm for needle in needles):
            continue
        ranked.append(str(action).strip())
    return ranked[0] if ranked else None


def resolve_slot_values(
    patch: dict[str, Any],
    state: ObservationState,
) -> dict[str, str]:
    """Fill slot names from task parse + admissible actions (no expert defaults)."""
    entities = _entities_from_admissible(state.admissible)
    target = _prefer_entity(entities, prefer=state.target)
    destination = None
    if state.destination:
        destination = _prefer_entity(entities, prefer=state.destination)
    tool = None
    # Prefer an entity that appears in a matching transform verb if present.
    op = str(state.operation or "").lower()
    if op in {"cool", "heat", "clean"}:
        for action in state.admissible:
            lower = action.lower()
            if lower.startswith(f"{op} ") and " with " in lower:
                tool = lower.split(" with ", 1)[1].strip()
                break
    location = None
    go = _find_admissible(state.admissible, verb="go to")
    if go:
        location = go[len("go to ") :].strip()

    values: dict[str, str] = {}
    bindings = dict(patch.get("slot_bindings") or {})
    # Always expose common slots even if bindings omit them.
    defaults = {
        "object": target,
        "target": target,
        "receptacle": destination or location,
        "destination": destination,
        "location": location or destination,
        "tool": tool,
        "entity": destination or location or tool,
        "source": location or destination,
    }
    for slot, spec in {**{k: {"source": "auto"} for k in defaults}, **bindings}.items():
        source = ""
        if isinstance(spec, dict):
            source = str(spec.get("source") or "auto")
        else:
            source = str(spec or "auto")
        value = None
        if source.startswith("state."):
            attr = source.split(".", 1)[1]
            value = getattr(state, attr, None)
        elif source == "task.target":
            value = state.target
        elif source == "task.destination":
            value = state.destination
        else:
            value = defaults.get(slot)
        if value:
            # If bare type name, try to upgrade to numbered admissible entity.
            upgraded = _prefer_entity(entities, prefer=str(value))
            values[slot] = upgraded or str(value)
    # Drop empties.
    return {key: val for key, val in values.items() if str(val).strip()}


def bind_template(template: str, slots: dict[str, str]) -> str:
    text = str(template or "")
    for slot, value in slots.items():
        text = text.replace(f"<{slot}>", str(value))
        text = text.replace(f"<{slot.upper()}>", str(value))
    # Common aliases
    aliases = {
        "<object>": slots.get("object") or slots.get("target"),
        "<target>": slots.get("target") or slots.get("object"),
        "<receptacle>": slots.get("receptacle") or slots.get("destination"),
        "<destination>": slots.get("destination") or slots.get("receptacle"),
        "<location>": slots.get("location") or slots.get("receptacle"),
        "<tool>": slots.get("tool"),
        "<entity>": slots.get("entity") or slots.get("receptacle") or slots.get("tool"),
        "<source>": slots.get("source") or slots.get("location"),
    }
    for token, value in aliases.items():
        if value:
            text = text.replace(token, str(value))
    return " ".join(text.split())


def _history_actions(history_steps: Sequence[Any] | None) -> list[str]:
    actions: list[str] = []
    for step in history_steps or []:
        if isinstance(step, dict):
            action = step.get("action")
        else:
            action = getattr(step, "action", None)
        text = " ".join(str(action or "").split()).strip().lower()
        # Strip optional <action>...</action> wrappers.
        match = re.search(r"<action>(.*?)</action>", text, re.I | re.S)
        if match:
            text = " ".join(match.group(1).split())
        if text:
            actions.append(text)
    return actions


def infer_bound_step_index(
    concrete_steps: Sequence[str],
    history_steps: Sequence[Any] | None,
) -> int:
    actions = _history_actions(history_steps)
    cursor = 0
    for action in actions:
        if cursor >= len(concrete_steps):
            break
        expected = concrete_steps[cursor].lower()
        if action == expected or action_matches_template(action, expected):
            cursor += 1
    return min(cursor, max(len(concrete_steps) - 1, 0))


def _format_success_demo_lines(
    demos: Sequence[Any],
    *,
    max_demos: int = 2,
) -> list[str]:
    """Backward-compatible wrapper; prefer rich_skill_context helpers."""
    from sage_mas.rich_skill_context import format_success_demo_lines

    return [
        line.lstrip()
        for line in format_success_demo_lines(
            demos, max_demos=max_demos, indent=""
        )
    ]


def render_bound_inject_text(
    *,
    skill_label: str,
    concrete_steps: Sequence[str],
    forbidden_actions: Sequence[str],
    current_step_index: int = 0,
    success_demos: Sequence[Any] | None = None,
    rich_supplement: str = "",
) -> str:
    """Final Flash-facing text: bound steps + rich demos/cues supplement."""
    lines = [
        f"[Skill: {skill_label}]",
        "Follow these steps in order:",
    ]
    for index, step in enumerate(concrete_steps, start=1):
        marker = ""
        if index - 1 < current_step_index:
            marker = " [done]"
        elif index - 1 == current_step_index:
            marker = " [current]"
        lines.append(f"{index}. {step}{marker}")
    if forbidden_actions:
        lines.append("Rules:")
        for action in forbidden_actions[:4]:
            lines.append(f'- Do not output "{action}" again.')
    # Prefer the extracted rich supplement (demos + failure cues + anti).
    # Fall back to demos-only if supplement is empty.
    supplement = str(rich_supplement or "").strip()
    if supplement:
        lines.append(supplement)
    else:
        demo_lines = _format_success_demo_lines(success_demos or [])
        if demo_lines:
            lines.append(
                "Example successful trajectories "
                "(adapt entities to the current observation):"
            )
            lines.extend(demo_lines)
    lines.append("Output exactly one action per turn.")
    return "\n".join(lines)


def bind_student_patch(
    skill: Skill,
    *,
    observation: str,
    task: str = "",
    gamefile: str = "",
    task_family: str | None = None,
    history_steps: Sequence[Any] | None = None,
) -> BoundStudentPatch:
    annotate_skill_identity(skill)
    patch = student_patch_of(skill)
    if not patch:
        return BoundStudentPatch(
            matched=False,
            skill_name=skill.skill_name,
            inject_text="",
            slot_values={},
            concrete_steps=[],
            forbidden_actions=[],
            current_step_index=0,
            reason="no_student_patch",
        )
    state = extract_observation_state(
        observation=observation,
        task=task,
        gamefile=gamefile,
        task_family=task_family,
    )
    ok, reason = trigger_matches(patch, state)
    if not ok:
        return BoundStudentPatch(
            matched=False,
            skill_name=skill.skill_name,
            inject_text="",
            slot_values={},
            concrete_steps=[],
            forbidden_actions=[],
            current_step_index=0,
            reason=reason,
        )

    slots = resolve_slot_values(patch, state)
    templates = [
        str(step).strip()
        for step in (patch.get("action_templates") or patch.get("do") or [])
        if str(step).strip()
    ]
    concrete: list[str] = []
    for template in templates:
        preferred = _bound_admissible_for_template(
            template, slots, state.admissible
        )
        if not preferred:
            continue
        if preferred not in concrete:
            concrete.append(preferred)

    if not concrete:
        return BoundStudentPatch(
            matched=False,
            skill_name=skill.skill_name,
            inject_text="",
            slot_values=slots,
            concrete_steps=[],
            forbidden_actions=[],
            current_step_index=0,
            reason="unresolved_slots",
        )

    step_index = infer_bound_step_index(concrete, history_steps)
    # Hard constraints = already completed concrete steps only (no expert lists).
    forbidden = list(concrete[:step_index])

    label = str(patch.get("skill_label") or _family_label(skill) or skill.skill_name)
    demos = patch.get("success_demos")
    if not isinstance(demos, list) or not demos:
        # Fallback: demos still on skill metadata (pre-enrich banks).
        raw = (skill.metadata or {}).get("specialist_demos") or []
        demos = raw if isinstance(raw, list) else []
    from sage_mas.rich_skill_context import render_rich_supplement_for_student_patch

    rich_supplement = render_rich_supplement_for_student_patch(
        skill,
        demos=demos,
        max_demos=2,
    )
    inject = render_bound_inject_text(
        skill_label=label,
        concrete_steps=concrete,
        forbidden_actions=forbidden,
        current_step_index=step_index,
        success_demos=demos,
        rich_supplement=rich_supplement,
    )
    return BoundStudentPatch(
        matched=True,
        skill_name=skill.skill_name,
        inject_text=inject,
        slot_values=slots,
        concrete_steps=concrete,
        forbidden_actions=forbidden,
        current_step_index=step_index,
        reason="ok",
    )


def _family_label(skill: Skill) -> str:
    annotate_skill_identity(skill)
    family = str(
        (skill.metadata or {}).get("primary_task_family")
        or (
            (skill.applicable_task_families or [""])[0]
            if skill.applicable_task_families
            else ""
        )
    )
    return family or skill.skill_name


def action_hits_forbidden(action: str, forbidden_actions: Sequence[str]) -> bool:
    text = " ".join(str(action or "").split()).strip().lower()
    match = re.search(r"<action>(.*?)</action>", text, re.I | re.S)
    if match:
        text = " ".join(match.group(1).split())
    if not text:
        return False
    forbidden = {str(item).strip().lower() for item in forbidden_actions if str(item).strip()}
    return text in forbidden


def render_skills_bound(
    skills: Sequence[Skill],
    *,
    observation: str,
    task: str = "",
    gamefile: str = "",
    task_family: str | None = None,
    history_steps: Sequence[Any] | None = None,
) -> str:
    """Render only matched, slot-filled student patches for the actor."""
    blocks: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        if skill.skill_name in seen:
            continue
        seen.add(skill.skill_name)
        if student_patch_of(skill) is None:
            continue
        bound = bind_student_patch(
            skill,
            observation=observation,
            task=task,
            gamefile=gamefile,
            task_family=task_family,
            history_steps=history_steps,
        )
        if bound.matched and bound.inject_text:
            blocks.append(bound.inject_text)
    return "\n\n".join(blocks)
