"""Causal-backbone extraction and causal-coherence checks."""

from __future__ import annotations

import re

from sage_mas.schemas import AtomicStep, Skill
from sage_mas.trajectory.abstraction import (
    _abstract_environment_action,
    is_productive_environment_step,
    stage_for_abstracted_action,
    trajectory_action_steps,
)
from sage_mas.trajectory.protocol_canon import (
    _dedupe_consecutive,
    _drop_oldest_search_step,
    protocol_structure_issues,
)

def _op_receptacle(text: str) -> str:
    """Receptacle a productive op interacts with (take-source / tool / target)."""
    normalized = " ".join(str(text or "").strip().lower().split())
    for pattern in (
        r"^take .+ from (.+)$",
        r"^(?:move|put) .+ to (.+)$",
        r"^(?:clean|heat|cool) .+ with (.+)$",
        r"^use (.+)$",
    ):
        match = re.match(pattern, normalized)
        if match:
            return match.group(1).strip()
    return ""


def _op_object(text: str) -> str:
    """Movable object of a take/place/transform step (``""`` when absent)."""
    normalized = " ".join(str(text or "").strip().lower().split())
    for pattern in (
        r"^take (.+) from .+$",
        r"^(?:move|put) (.+) to .+$",
        r"^(?:clean|heat|cool) (.+) with .+$",
    ):
        match = re.match(pattern, normalized)
        if match:
            return match.group(1).strip()
    return ""


def _core_op_indices(abstracts: list[str], capability: str) -> list[int]:
    """Indices of the causal-chain ops inside a winning trace.

    The chain is the first acquisition of the ultimately-placed object, the
    capability transform, and the terminal place(s). Detour pickups of other
    objects, re-acquisitions after the object was already held, and tidy-up
    closes never enter the chain.
    """
    capability = str(capability or "").strip()
    stages = [stage_for_abstracted_action(text) for text in abstracts]
    pickups = [i for i, stage in enumerate(stages) if stage == "pickup"]
    transforms = [i for i, stage in enumerate(stages) if stage == "transform"]
    places = [i for i, stage in enumerate(stages) if stage == "place"]

    def first_take_of(obj: str, before: int) -> int | None:
        for i in pickups:
            if i >= before:
                continue
            if not obj or _op_object(abstracts[i]) == obj:
                return i
        return None

    if capability in {"inspect.with_light", "inspect.light"}:
        lamp = next(
            (i for i in reversed(transforms) if "desklamp" in abstracts[i]),
            None,
        )
        if lamp is None:
            return []
        # ALFWorld completes look_at_obj_in_light when the object is held
        # while the lamp is on — either order wins. Emit canonical order
        # (acquire, then examine under light) even when the trace lit first.
        take = first_take_of("", len(abstracts))
        if take is None:
            return [lamp]
        return [take, lamp]

    if capability == "track.multiple_objects":
        if len(places) < 2 or len(pickups) < 2:
            return sorted(set(pickups + places))
        terminal = places[-2:]
        obj = _op_object(abstracts[terminal[-1]])
        takes: list[int] = []
        for i in pickups:
            if i >= terminal[-1]:
                continue
            if obj and _op_object(abstracts[i]) != obj:
                continue
            takes.append(i)
            if len(takes) >= 2:
                break
        if len(takes) < 2:
            takes = pickups[:2]
        return sorted(set(takes + terminal))

    place_idx = places[-1] if places else None
    if capability.startswith("transform."):
        marker = capability.split(".", 1)[1].lower()
        marked = [i for i in transforms if marker in abstracts[i]]
        if place_idx is not None:
            before_place = [i for i in marked if i < place_idx]
            marked = before_place or marked
        transform_idx = marked[-1] if marked else None
        if transform_idx is not None:
            obj = _op_object(abstracts[place_idx]) if place_idx is not None else ""
            take_idx = first_take_of(obj, transform_idx)
            core = [i for i in (take_idx, transform_idx, place_idx) if i is not None]
            return sorted(set(core))
        # The name did not match an observed step. Fall through and keep
        # the state change the trajectory actually showed.

    # track.place is take-then-place. Every other name, including an empty
    # name and a model-chosen name, keeps state-changing steps that occurred
    # between the pickup and the place. Those steps are in the trace; they
    # are not chosen from a family table.
    if place_idx is None:
        return sorted(set(pickups[-1:] + transforms[-1:]))
    obj = _op_object(abstracts[place_idx])
    take_idx = first_take_of(obj, place_idx)
    if take_idx is None:
        take_idx = next((i for i in pickups if i < place_idx), None)
    core = {i for i in (take_idx, place_idx) if i is not None}
    if not capability.startswith("track."):
        start = take_idx if take_idx is not None else -1
        core.update(i for i in transforms if start < i < place_idx)
    return sorted(core)


def extract_causal_backbone(
    steps: list[AtomicStep],
    *,
    capability: str = "",
    max_protocol_steps: int = 16,
) -> list[str]:
    """Compress a winning trajectory to its causal backbone.

    Keep the object-acquisition pickup, the capability transform, and the
    terminal place; attach only the last ``go to`` / ``open`` each core op
    causally needs. Search detours, closes, examines, and repeated open
    cycles are dropped. Every kept step remains an observed environment
    action — nothing is invented.
    """
    productive = [
        step
        for step in trajectory_action_steps(steps)
        if is_productive_environment_step(step)
    ]
    abstracts = [
        _abstract_environment_action(str(step.action or ""))
        for step in productive
    ]
    abstracts = [text for text in abstracts if text]
    if not abstracts:
        return []
    core = _core_op_indices(abstracts, capability)
    if not core:
        return []
    out: list[str] = []
    cursor = 0
    for index in core:
        recep = _op_receptacle(abstracts[index])
        if recep:
            nav = next(
                (
                    j
                    for j in range(index - 1, cursor - 1, -1)
                    if abstracts[j] == f"go to {recep}"
                ),
                None,
            )
            if nav is not None:
                out.append(abstracts[nav])
            open_idx = next(
                (
                    j
                    for j in range(
                        index - 1,
                        (nav if nav is not None else cursor) - 1,
                        -1,
                    )
                    if abstracts[j] == f"open {recep}"
                ),
                None,
            )
            if open_idx is not None:
                out.append(abstracts[open_idx])
        out.append(abstracts[index])
        cursor = index + 1
    out = _dedupe_consecutive(out, preserve_stage_repeats=True)
    while len(out) > max(1, int(max_protocol_steps)):
        if _drop_oldest_search_step(out) is None:
            break
    return out


def protocol_causal_issues(
    protocol: list[str],
    *,
    capability: str = "",
) -> list[str]:
    """Causal-coherence reasons for a protocol (beyond stage shape).

    A receptacle must be open before take/place through it, productive ops
    must execute at the receptacle they reference, and navigation must not
    stutter. Slotted steps (``<source>`` / ``<destination>``) are grounded at
    runtime and skipped.
    """
    capability = str(capability or "").strip()
    issues: list[str] = []
    if not protocol:
        return ["empty protocol"]
    for prev, cur in zip(protocol, protocol[1:]):
        if cur == prev and stage_for_abstracted_action(cur) == "find":
            issues.append(f"consecutive duplicate navigation `{cur}`")
    location = ""
    receptacle_state: dict[str, str] = {}
    for index, step in enumerate(protocol):
        stage = stage_for_abstracted_action(step)
        if stage == "find":
            location = step[len("go to ") :].strip()
            continue
        if step.startswith("open "):
            receptacle_state[step[len("open ") :].strip()] = "open"
            continue
        if step.startswith("close "):
            recep = step[len("close ") :].strip()
            if receptacle_state.get(recep) != "open":
                issues.append(
                    f"step {index + 1}: `close {recep}` without prior open"
                )
            receptacle_state[recep] = "closed"
            continue
        if stage not in {"pickup", "transform", "place"}:
            continue
        recep = _op_receptacle(step)
        if not recep or recep.startswith("<"):
            continue
        if receptacle_state.get(recep) == "closed":
            issues.append(
                f"step {index + 1}: `{step}` uses `{recep}` after it was closed"
            )
        # ``use <fixture>`` (desklamp) is co-located with furniture; the
        # fixture is not a navigation target of its own.
        if step.startswith("use "):
            continue
        if location and not location.startswith("<") and recep != location:
            issues.append(
                f"step {index + 1}: `{step}` while current location is `{location}`"
            )
    if capability == "track.multiple_objects":
        takes = sum(
            1 for step in protocol if stage_for_abstracted_action(step) == "pickup"
        )
        places = sum(
            1 for step in protocol if stage_for_abstracted_action(step) == "place"
        )
        if takes < 2 or places < 2:
            issues.append(
                "multi-object protocol needs >=2 pickup and >=2 place steps "
                f"(got {takes} pickup, {places} place)"
            )
    if capability.startswith("transform."):
        marker = capability.split(".", 1)[1].lower()
        for step in protocol:
            if stage_for_abstracted_action(step) != "transform":
                continue
            if marker not in step.lower():
                continue
            tool = _op_receptacle(step)
            if tool.startswith("<"):
                issues.append(
                    f"transform step `{step}` must keep a concrete tool "
                    "(tools are task-family-determined, not scene-specific)"
                )
    return issues


def slot_scene_receptacles(protocol: list[str]) -> list[str]:
    """Slot scene-specific source/target receptacles; keep tools concrete.

    When no receptacle-homogeneous group reaches merge support, the cross-scene
    consensus is the protocol *shape* (go→take→…→place), not the receptacle
    types. Objects slot to ``<object>``, sources to ``<source>``, targets to
    ``<destination>`` (matching the executable layer's role-slot convention,
    where ``<target>`` already means the moved object); transform tools
    (microwave / fridge / sinkbasin / desklamp) stay concrete because they are
    task-family-determined, not scene-specific.
    """
    def next_core_role(start: int, dest: str) -> str:
        for j in range(start + 1, len(protocol)):
            stage = stage_for_abstracted_action(protocol[j])
            if stage not in {"pickup", "transform", "place"}:
                continue
            if _op_receptacle(protocol[j]) != dest:
                return ""
            if stage == "pickup":
                return "source"
            if stage == "place":
                return "destination"
            return ""
        return ""

    out: list[str] = []
    for index, step in enumerate(protocol):
        stage = stage_for_abstracted_action(step)
        if stage == "find":
            dest = step[len("go to ") :].strip()
            role = next_core_role(index, dest)
            out.append(f"go to <{role}>" if role else step)
            continue
        if stage == "access":
            verb, _, entity = step.partition(" ")
            entity = entity.strip()
            role = ""
            for j in range(index + 1, len(protocol)):
                next_stage = stage_for_abstracted_action(protocol[j])
                if next_stage not in {"pickup", "transform", "place"}:
                    continue
                if _op_receptacle(protocol[j]) == entity:
                    role = "source" if next_stage == "pickup" else (
                        "destination" if next_stage == "place" else ""
                    )
                break
            out.append(f"{verb} <{role}>" if role else step)
            continue
        if stage == "pickup":
            out.append("take <object> from <source>")
            continue
        if stage == "place":
            out.append("move <object> to <destination>")
            continue
        if stage == "transform":
            match = re.match(r"^(clean|heat|cool) .+ with (.+)$", step)
            if match:
                out.append(f"{match.group(1)} <object> with {match.group(2)}")
            else:
                out.append(step)
            continue
        out.append(step)
    return out


def protocol_is_org_ready(skill: Skill) -> bool:
    """Whether a grounded skill is strong enough for organization edits."""
    if skill.metadata.get("protocol_alignment_ok") is False:
        return False
    if skill.metadata.get("protocol_structure_ok") is False:
        return False
    if skill.metadata.get("protocol_form_ok") is False:
        return False
    if skill.metadata.get("not_for_add_agent"):
        return False
    # Teacher form alone is not enough: rejected / unset student MU blocks org.
    if skill.metadata.get("inject_ready") is False:
        return False
    if skill.metadata.get("mu_rejected") is True:
        return False
    support = int(skill.metadata.get("protocol_alignment_support") or 0)
    if support < 1:
        return False
    issues = protocol_structure_issues(
        list(skill.action_protocol or []),
        capability=str(skill.capability_key or ""),
    )
    return not issues
