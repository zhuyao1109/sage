"""Environment-action abstraction, step productivity, and stage labels."""

from __future__ import annotations

import re

from sage_mas.schemas import AtomicOp, AtomicStep

def trajectory_id_from_steps(steps: list[AtomicStep]) -> str:
    terminal = steps[-1]
    return str(terminal.metadata.get("trajectory_id", terminal.node_id))


def trajectory_action_steps(
    steps: list[AtomicStep],
) -> list[AtomicStep]:
    """Return environment actions, excluding reasoning/advisor messages."""
    return [
        step
        for step in steps[:-1]
        if step.atomic_op != AtomicOp.COMMUNICATE
        and str(step.action or "").strip()
    ]


def _strip_instance_id(phrase: str) -> str:
    """Drop trailing instance ids (``fridge 1`` → ``fridge``) but keep type."""
    text = " ".join(str(phrase or "").strip().lower().split())
    return re.sub(r"\s+\d+$", "", text).strip()


def _abstract_environment_action(action: str) -> str:
    """Generalize an observed ALFWorld action without inventing new steps.

    Preserve entity **types** (fridge / microwave / apple) so protocols stay
    actionable for weaker actors. Only strip instance numbers. In particular,
    ``use desklamp N`` must stay ``use desklamp``.

    Legacy fully-slotted forms like ``go to <location>`` remain valid inputs
    for stage labeling / tests, but newly distilled protocols should not emit
    interchangeable bare location slots.
    """
    normalized = " ".join(str(action).strip().lower().split())
    if re.match(r"^use\s+desklamp(?:\s+\d+)?$", normalized):
        return "use desklamp"

    match = re.match(r"^go to (.+)$", normalized)
    if match:
        target = _strip_instance_id(match.group(1))
        return f"go to {target or '<location>'}"

    match = re.match(r"^take (.+) from (.+)$", normalized)
    if match:
        obj = _strip_instance_id(match.group(1))
        recep = _strip_instance_id(match.group(2))
        return f"take {obj or '<object>'} from {recep or '<receptacle>'}"

    match = re.match(r"^(?:put|move) (.+) (?:in|on|to) (.+)$", normalized)
    if match:
        obj = _strip_instance_id(match.group(1))
        recep = _strip_instance_id(match.group(2))
        return f"move {obj or '<object>'} to {recep or '<receptacle>'}"

    match = re.match(r"^(clean|heat|cool) (.+) with (.+)$", normalized)
    if match:
        verb, obj, tool = match.group(1), match.group(2), match.group(3)
        return (
            f"{verb} {_strip_instance_id(obj) or '<object>'} "
            f"with {_strip_instance_id(tool) or '<tool>'}"
        )

    match = re.match(r"^(open|close|examine) (.+)$", normalized)
    if match:
        verb, entity = match.group(1), match.group(2)
        return f"{verb} {_strip_instance_id(entity) or '<entity>'}"

    match = re.match(r"^use (.+)$", normalized)
    if match:
        return f"use {_strip_instance_id(match.group(1)) or '<entity>'}"

    return normalized


def abstract_environment_action(action: str) -> str:
    """Public wrapper for grounding protocols in observed environment actions."""
    return _abstract_environment_action(action)


# Observation / action noise that should not enter reusable protocols.
_NOOP_OBSERVATION_MARKERS = (
    "nothing happens",
    "nothing happens.",
)


_PROTOCOL_NOISE_ACTIONS = {
    "inventory",
    "look",
    "examine <entity>",
}


def _is_protocol_noise_action(abstracted: str) -> bool:
    text = str(abstracted or "").strip().lower()
    if text in _PROTOCOL_NOISE_ACTIONS:
        return True
    return text.startswith("examine ")


# Canonical stage order used only to sort observed productive actions.
_STAGE_ORDER = (
    "find",
    "access",
    "pickup",
    "transform",
    "place",
    "other",
)


def _observation_is_noop(observation: str | None) -> bool:
    text = str(observation or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _NOOP_OBSERVATION_MARKERS)


# ALFWorld success observations that prove the action changed environment state.
# These override a false ``is_action_valid`` flag from the env wrapper.
_CONFIRMED_EFFECT_OBSERVATION_MARKERS = (
    "you pick up",
    "you clean",
    "you heat",
    "you cool",
    "you turn on",
    "you move",
    "you put",
    "you open",
    "you close",
    "you arrive at",
)


def _observation_confirms_environment_effect(observation: str | None) -> bool:
    text = str(observation or "").strip().lower()
    if not text or _observation_is_noop(text):
        return False
    return any(marker in text for marker in _CONFIRMED_EFFECT_OBSERVATION_MARKERS)


def is_productive_environment_step(step: AtomicStep) -> bool:
    """True when an environment action changed state usefully."""
    action = str(step.action or "").strip()
    if not action:
        return False
    if step.metadata.get("stalled"):
        return False
    if _observation_is_noop(step.observation):
        return False
    abstracted = _abstract_environment_action(action)
    if _is_protocol_noise_action(abstracted):
        return False
    # Prefer grounded observations over a noisy is_action_valid flag. Env
    # wrappers sometimes mark successful take/clean/heat/cool/use as invalid.
    if _observation_confirms_environment_effect(step.observation):
        return True
    if step.metadata.get("is_action_valid") is False:
        return False
    return True


def stage_for_abstracted_action(abstracted: str) -> str:
    """Map an abstracted environment action to a protocol stage label."""
    text = str(abstracted or "").strip().lower()
    if text.startswith("go to "):
        return "find"
    if text.startswith("open ") or text.startswith("close "):
        return "access"
    if text.startswith("take "):
        return "pickup"
    if re.match(r"^(clean|heat|cool)\b", text) or text.startswith("use "):
        return "transform"
    if text.startswith("move ") or text.startswith("put "):
        return "place"
    return "other"


def _capability_operation_marker(capability: str) -> str:
    """Substring used to anchor a local window in raw trajectory actions."""
    capability = str(capability or "").strip()
    if capability.startswith("transform."):
        return capability.split(".", 1)[1]
    if capability in {"inspect.with_light", "inspect.light"}:
        return "use desklamp"
    return ""
