"""Filter, canonicalize, and structurally check staged protocols."""

from __future__ import annotations

from typing import Sequence

from sage_mas.schemas import AtomicStep
from sage_mas.trajectory.abstraction import (
    _abstract_environment_action,
    _capability_operation_marker,
    is_productive_environment_step,
    stage_for_abstracted_action,
    trajectory_action_steps,
)

def _filtered_protocol_from_trajectory(
    steps: list[AtomicStep],
    *,
    capability: str = "",
    max_protocol_steps: int = 16,
) -> list[str]:
    """Compress one trajectory into a productive staged action protocol.

    For ``transform.*`` capabilities, keep the full productive episode so the
    same successful run's find/access prefix stays inside the transform skill
    (not as a separate injectable ``search.*`` skill). Search hops abstract to
    the same ``go to <location>``; they are kept as repeated steps so later
    evolution is not stuck on a 5-verb skeleton.
    """
    action_steps = [
        step
        for step in trajectory_action_steps(steps)
        if is_productive_environment_step(step)
    ]
    capability = str(capability or "").strip()
    marker = _capability_operation_marker(capability)
    # Keep a full productive episode for transform / light-inspect skills so
    # merge consensus retains find/access context instead of collapsing to a
    # 2–3 step effect skeleton (go→take→use desklamp).
    keep_full_episode = capability.startswith("transform.") or capability in {
        "inspect.with_light",
        "inspect.light",
        "track.multiple_objects",
        "track.place",
    }
    if marker and not keep_full_episode:
        anchor_index = next(
            (
                index
                for index, step in enumerate(action_steps)
                if str(step.action or "").strip().lower().startswith(marker)
            ),
            None,
        )
        if anchor_index is not None:
            # Narrow skills keep a local window around the effect.
            action_steps = action_steps[
                max(0, anchor_index - 5) : anchor_index + 5
            ]

    if keep_full_episode:
        max_protocol_steps = max(int(max_protocol_steps), 16)
    # Capability skills need a longer find/access prefix than generic recovery.
    # Search hops abstract to the same ``go to <location>``; collapsing them
    # to one step makes later evolution / student compile follow a 5-verb
    # skeleton instead of the teacher episode.
    find_access_cap = 8 if keep_full_episode else 2
    protocol: list[str] = []
    stage_counts: dict[str, int] = {}
    for step in action_steps:
        abstracted = _abstract_environment_action(str(step.action or ""))
        if not abstracted:
            continue
        stage = stage_for_abstracted_action(abstracted)
        if protocol and protocol[-1] == abstracted and stage != "find":
            continue
        # Cap repeated navigation / access noise inside one episode window.
        if (
            stage in {"find", "access"}
            and stage_counts.get(stage, 0) >= find_access_cap
        ):
            continue
        protocol.append(abstracted)
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if len(protocol) < max_protocol_steps:
            continue
        missing = _core_stages_missing(protocol, capability)
        if not (keep_full_episode and missing):
            break
        dropped = _drop_oldest_search_step(protocol)
        if dropped is None:
            break
        dropped_stage = stage_for_abstracted_action(dropped)
        stage_counts[dropped_stage] = max(
            0, stage_counts.get(dropped_stage, 1) - 1
        )
    return protocol


def _core_stages_missing(protocol: list[str], capability: str) -> set[str]:
    """Return required stages still absent from a capability protocol."""
    capability = str(capability or "").strip()
    stages = {stage_for_abstracted_action(step) for step in protocol}
    if capability.startswith("transform."):
        marker = capability.split(".", 1)[-1].lower()
        missing = {"pickup", "transform", "place"} - stages
        if not any(
            stage_for_abstracted_action(step) == "transform"
            and marker in step.lower()
            for step in protocol
        ):
            missing.add("transform")
        return missing
    if capability.startswith("track."):
        return {"pickup", "place"} - stages
    if capability in {"inspect.with_light", "inspect.light"}:
        missing: set[str] = set()
        if "pickup" not in stages:
            missing.add("pickup")
        if not any("desklamp" in step.lower() for step in protocol):
            missing.add("transform")
        return missing
    return set()


def _drop_oldest_search_step(protocol: list[str]) -> str | None:
    for index, step in enumerate(protocol):
        if stage_for_abstracted_action(step) in {"find", "access"}:
            return protocol.pop(index)
    return None


def _dedupe_consecutive(
    protocol: list[str],
    *,
    preserve_stage_repeats: bool = False,
    preserve_find_repeats: bool = False,
) -> list[str]:
    out: list[str] = []
    for step in protocol:
        if not step:
            continue
        if out and out[-1] == step:
            stage = stage_for_abstracted_action(step)
            if preserve_stage_repeats and stage in {"pickup", "place"}:
                # Keep repeated take/place slots for multi-object protocols.
                out.append(step)
                continue
            if preserve_find_repeats and stage == "find":
                out.append(step)
                continue
            continue
        out.append(step)
    return out


def _protocol_core_order_ok(protocol: Sequence[str], capability: str) -> bool:
    stages = [stage_for_abstracted_action(step) for step in protocol]
    if capability.startswith("transform."):
        marker = capability.split(".", 1)[-1].lower()
        try:
            take_i = next(i for i, stage in enumerate(stages) if stage == "pickup")
            transform_i = next(
                i
                for i, (stage, step) in enumerate(zip(stages, protocol))
                if stage == "transform" and marker in str(step).lower()
            )
            place_i = next(i for i, stage in enumerate(stages) if stage == "place")
        except StopIteration:
            return False
        return take_i < transform_i < place_i
    if capability.startswith("track."):
        try:
            take_i = next(i for i, stage in enumerate(stages) if stage == "pickup")
            place_i = next(i for i, stage in enumerate(stages) if stage == "place")
        except StopIteration:
            return False
        return take_i < place_i
    return True


def canonicalize_protocol_stages(
    protocol: list[str],
    *,
    capability: str = "",
) -> list[str]:
    """Reorder an aligned protocol into a stable stage sequence.

    This does not invent actions. Observed find/search hops that abstract to
    the same ``go to <location>`` are kept (evolution needs that length).
    Transform traces are only reordered when pickup/transform/place is scrambled.
    """
    if not protocol:
        return []
    capability = str(capability or "").strip()
    buckets: dict[str, list[str]] = {
        "find": [],
        "access": [],
        "pickup": [],
        "transform": [],
        "place": [],
        "other": [],
    }
    for step in protocol:
        buckets[stage_for_abstracted_action(step)].append(step)

    if capability == "track.multiple_objects":
        ordered: list[str] = []
        ordered.extend(buckets["find"][:8])
        ordered.extend(buckets["access"][:2])
        pickups = list(buckets["pickup"])
        places = list(buckets["place"])
        # Keep observed pickup/place steps only. Do NOT invent non-executable
        # ``(second)`` markers — typed abstraction should already distinguish
        # two objects when they differ; identical templates may repeat.
        pair_count = max(len(pickups), len(places), 1)
        for index in range(pair_count):
            if index < len(pickups):
                ordered.append(pickups[index])
            if index < len(places):
                ordered.append(places[index])
        return _dedupe_consecutive(
            ordered,
            preserve_stage_repeats=True,
            preserve_find_repeats=True,
        )

    if _protocol_core_order_ok(protocol, capability):
        return _dedupe_consecutive(list(protocol), preserve_find_repeats=True)

    finds = list(buckets["find"])
    n_find = len(finds)
    if n_find <= 1:
        prefix, mid, suffix = finds, [], []
    elif n_find == 2:
        prefix, mid, suffix = finds[:1], finds[1:2], []
    elif n_find == 3:
        prefix, mid, suffix = finds[:1], finds[1:2], finds[2:]
    else:
        prefix = finds[: max(1, n_find - 2)]
        mid = finds[len(prefix) : len(prefix) + 1]
        suffix = finds[len(prefix) + 1 :]

    ordered = []
    ordered.extend(prefix)
    ordered.extend(buckets["access"][:2])
    ordered.extend(buckets["pickup"][:2])
    ordered.extend(mid)
    ordered.extend(buckets["transform"][:1])
    ordered.extend(suffix)
    ordered.extend(buckets["place"][:2])
    return _dedupe_consecutive(ordered, preserve_find_repeats=True)


def protocol_structure_issues(
    protocol: list[str],
    *,
    capability: str = "",
) -> list[str]:
    """Return structural reasons a protocol is too shallow / inconsistent."""
    capability = str(capability or "").strip()
    issues: list[str] = []
    if not protocol:
        return ["empty protocol"]
    stages = [stage_for_abstracted_action(step) for step in protocol]
    if capability.startswith("transform."):
        marker = capability.split(".", 1)[1].lower()
        has_take = any(stage == "pickup" for stage in stages)
        has_transform = any(
            stage == "transform" and marker in step.lower()
            for stage, step in zip(stages, protocol)
        )
        has_place = any(stage == "place" for stage in stages)
        if not has_take:
            issues.append("transform protocol missing take/pickup")
        if not has_transform:
            issues.append(f"transform protocol missing `{marker}` step")
        if not has_place:
            issues.append("transform protocol missing place/move")
        if has_take and has_transform and has_place:
            take_i = next(
                i for i, stage in enumerate(stages) if stage == "pickup"
            )
            transform_i = next(
                i
                for i, (stage, step) in enumerate(zip(stages, protocol))
                if stage == "transform" and marker in step.lower()
            )
            place_i = next(
                i for i, stage in enumerate(stages) if stage == "place"
            )
            if not (take_i < transform_i < place_i):
                issues.append(
                    "transform protocol stage order must be "
                    "pickup -> transform -> place"
                )
    elif capability == "track.multiple_objects":
        takes = sum(1 for stage in stages if stage == "pickup")
        places = sum(1 for stage in stages if stage == "place")
        # Require at least one pickup and one place. Prefer two of each, but do
        # not hard-reject after abstraction collapses duplicate take/place forms.
        if takes < 1:
            issues.append("multi-object protocol missing take/pickup")
        if places < 1:
            issues.append("multi-object protocol missing place/move")
        if takes == 1 and places == 1 and len(protocol) < 3:
            issues.append(
                "multi-object protocol too shallow "
                "(need search/access context around take/place)"
            )
    elif capability in {"inspect.with_light", "inspect.light"}:
        has_take = any(stage == "pickup" for stage in stages)
        has_lamp = any(
            "desklamp" in step.lower() or step.lower().startswith("use desklamp")
            for step in protocol
        )
        if not has_take:
            issues.append("light-inspect protocol missing take/pickup")
        if not has_lamp:
            issues.append("light-inspect protocol missing `use desklamp`")
    # Generic noise guard: a capability protocol that collapsed to 1–2 steps
    # is almost never an injectable plan.
    if capability and len(protocol) < 3 and not issues:
        issues.append(
            f"protocol too shallow for capability `{capability}` "
            f"(len={len(protocol)} < 3)"
        )
    return issues
