"""Cross-trajectory typed stage consensus for winning protocols."""

from __future__ import annotations

import re
from typing import Any

from sage_mas.trajectory.abstraction import (
    _STAGE_ORDER,
    stage_for_abstracted_action,
)
from sage_mas.trajectory.protocol_canon import (
    canonicalize_protocol_stages,
    protocol_structure_issues,
)

def _protocol_stage_counts(protocol: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in protocol:
        stage = stage_for_abstracted_action(step)
        if stage == "other":
            continue
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return int(ordered[len(ordered) // 2])


def _ensure_evidence_stage_slots(
    aligned: list[str],
    protocols: list[list[str]],
    *,
    first_index_sums: dict[str, float],
    first_index_counts: dict[str, int],
    capability: str = "",
) -> list[str]:
    """Fill stage slots learned from cleaned-win stage statistics.

    No capability checklist: a stage is required only when it recurs across
    enough evidence protocols. Target multiplicity is the median count of that
    stage among protocols that contain it. Concrete steps are salvaged from
    evidence, never invented.
    """
    if not protocols:
        return aligned

    soft_support = max(1, (len(protocols) + 2) // 3)
    stage_doc_freq: dict[str, int] = {}
    stage_count_samples: dict[str, list[int]] = {}
    stage_first_index_sums: dict[str, float] = {}
    stage_first_index_counts: dict[str, int] = {}

    for protocol in protocols:
        counts = _protocol_stage_counts(protocol)
        seen_stages: set[str] = set()
        for index, step in enumerate(protocol):
            stage = stage_for_abstracted_action(step)
            if stage == "other" or stage in seen_stages:
                continue
            seen_stages.add(stage)
            stage_first_index_sums[stage] = (
                stage_first_index_sums.get(stage, 0.0) + float(index)
            )
            stage_first_index_counts[stage] = (
                stage_first_index_counts.get(stage, 0) + 1
            )
        for stage, count in counts.items():
            stage_doc_freq[stage] = stage_doc_freq.get(stage, 0) + 1
            stage_count_samples.setdefault(stage, []).append(int(count))

    multi_object = str(capability or "").strip() == "track.multiple_objects"
    # Evidence-derived slots only.
    slots: list[tuple[str, int]] = []
    for stage, doc_freq in stage_doc_freq.items():
        if doc_freq < soft_support:
            continue
        target = max(1, _median_int(stage_count_samples.get(stage, [])))
        # Transform skills almost always have one pickup/transform/place;
        # median>1 usually means failed-then-retry noise, not a second object.
        if (
            not multi_object
            and stage in {"pickup", "transform", "place"}
            and target > 1
        ):
            target = 1
        slots.append((stage, target))

    slots.sort(
        key=lambda item: (
            stage_first_index_sums.get(item[0], 0.0)
            / max(1, stage_first_index_counts.get(item[0], 1)),
            item[0],
        )
    )

    ranked_protocols = sorted(
        protocols,
        key=lambda protocol: (
            -len(_protocol_stage_counts(protocol)),
            -len(protocol),
        ),
    )

    def _count_stage(items: list[str], stage: str) -> int:
        return sum(
            1 for step in items if stage_for_abstracted_action(step) == stage
        )

    out = list(aligned)
    for stage, need in slots:
        while _count_stage(out, stage) < need:
            chosen: str | None = None
            existing = {
                step
                for step in out
                if stage_for_abstracted_action(step) == stage
            }
            for protocol in ranked_protocols:
                for step in protocol:
                    if stage_for_abstracted_action(step) != stage:
                        continue
                    if step in existing and stage != "find":
                        continue
                    support = sum(1 for proto in protocols if step in proto)
                    # Prefer recurrent concrete steps; allow support>=1 only when
                    # the stage is still entirely missing from the merge.
                    # Find hops share one template; duplicate it up to the
                    # median count instead of collapsing search to one step.
                    if (
                        support < soft_support
                        and _count_stage(out, stage) > 0
                        and stage != "find"
                    ):
                        continue
                    if support < 1:
                        continue
                    chosen = step
                    break
                if chosen is not None:
                    break
            if chosen is None and stage == "find":
                for protocol in ranked_protocols:
                    for step in protocol:
                        if stage_for_abstracted_action(step) == "find":
                            chosen = step
                            break
                    if chosen is not None:
                        break
            if chosen is None:
                break
            out.append(chosen)
            existing.add(chosen)
            if chosen not in first_index_sums:
                prior = stage_first_index_sums.get(stage, 0.0) / max(
                    1, stage_first_index_counts.get(stage, 1)
                )
                first_index_sums[chosen] = prior
                first_index_counts[chosen] = 1
    return out


def _keeps_episode_search_hops(capability: str) -> bool:
    capability = str(capability or "").strip()
    return (
        capability.startswith("transform.")
        or capability.startswith("track.")
        or capability in {"inspect.with_light", "inspect.light"}
    )


def _consensus_action_key(step: str) -> str:
    """Cross-trace vote key: keep place/tool types, slot movable objects.

    ``go to fridge`` / ``heat <object> with microwave`` can align across wins
    that used different apples/tomatoes, without collapsing all navigation into
    a single interchangeable ``go to <location>``.
    """
    text = " ".join(str(step or "").strip().lower().split())
    match = re.match(r"^take (.+) from (.+)$", text)
    if match:
        recep = match.group(2).strip()
        return f"take <object> from {recep or '<receptacle>'}"
    match = re.match(r"^move (.+) to (.+)$", text)
    if match:
        recep = match.group(2).strip()
        return f"move <object> to {recep or '<receptacle>'}"
    match = re.match(r"^(clean|heat|cool) (.+) with (.+)$", text)
    if match:
        verb, tool = match.group(1), match.group(3).strip()
        return f"{verb} <object> with {tool or '<tool>'}"
    return text


def _protocols_as_consensus_keys(
    protocols: list[list[str]],
) -> list[list[str]]:
    return [[_consensus_action_key(step) for step in protocol] for protocol in protocols]


def _cap_consensus_find_hops(
    steps: list[str],
    *,
    occurrence: dict[str, int],
    max_finds: int = 4,
) -> list[str]:
    """Keep the most recurrent find hops; drop long search tails."""
    finds = [
        step
        for step in steps
        if stage_for_abstracted_action(step) == "find"
    ]
    if len(finds) <= max_finds:
        return list(steps)
    ranked = sorted(
        finds,
        key=lambda step: (
            -int(occurrence.get(step, 0)),
            finds.index(step),
            step,
        ),
    )
    keep = set(ranked[: max(1, int(max_finds))])
    return [
        step
        for step in steps
        if stage_for_abstracted_action(step) != "find" or step in keep
    ]


def _align_protocols_across_trajectories(
    protocols: list[list[str]],
    *,
    min_support: int | None = None,
    capability: str = "",
    merge_trace: dict[str, Any] | None = None,
) -> list[str]:
    """Merge winning protocols by typed stage consensus.

    Movable objects are slotted to ``<object>`` for voting so different wins can
    agree on ``heat <object> with microwave`` / ``go to fridge``. Place and tool
    types stay concrete. No median-single-trace shortcut.
    """
    if merge_trace is not None:
        merge_trace.clear()
        merge_trace.update(
            {
                "mode": "empty",
                "input_protocols": len(protocols),
                "consensus_support": 0,
            }
        )

    if not protocols:
        return []

    keyed_protocols = _protocols_as_consensus_keys(protocols)
    if len(keyed_protocols) == 1:
        if merge_trace is not None:
            merge_trace.update(
                {
                    "mode": "single_trace",
                    "consensus_support": 1,
                }
            )
        return canonicalize_protocol_stages(
            list(keyed_protocols[0]),
            capability=capability,
        )

    support_needed = min_support
    if support_needed is None:
        support_needed = max(2, (len(keyed_protocols) + 1) // 2)

    occurrence: dict[str, int] = {}
    first_index_sums: dict[str, float] = {}
    first_index_counts: dict[str, int] = {}
    key_count_samples: dict[str, list[int]] = {}
    # Per-key ordinal indices: occurrence[k][i] = indices of the i-th copy
    # across protocols that have at least i+1 copies. Used so return-to-tool
    # hops (2nd ``go to fridge``) land after pickup, not dumped at the front.
    ordinal_index_lists: dict[str, list[list[int]]] = {}
    for protocol in keyed_protocols:
        seen: set[str] = set()
        local_counts: dict[str, int] = {}
        local_indices: dict[str, list[int]] = {}
        for index, step in enumerate(protocol):
            local_counts[step] = local_counts.get(step, 0) + 1
            local_indices.setdefault(step, []).append(index)
            if step in seen:
                continue
            occurrence[step] = occurrence.get(step, 0) + 1
            first_index_sums[step] = first_index_sums.get(step, 0.0) + index
            first_index_counts[step] = first_index_counts.get(step, 0) + 1
            seen.add(step)
        for step, count in local_counts.items():
            key_count_samples.setdefault(step, []).append(int(count))
        for step, indices in local_indices.items():
            buckets = ordinal_index_lists.setdefault(step, [])
            for ordinal, index in enumerate(indices):
                while len(buckets) <= ordinal:
                    buckets.append([])
                buckets[ordinal].append(int(index))

    # Find/access: keep every majority-typed hop (search diversity).
    # Pickup/transform/place: keep the strongest template(s) only — otherwise
    # cool/heat merges accumulate every ``take … from X`` that clears majority.
    core_limits = {
        "pickup": 2 if str(capability or "").strip() == "track.multiple_objects" else 1,
        "transform": 1,
        "place": 2 if str(capability or "").strip() == "track.multiple_objects" else 1,
        "access": 2,
    }
    aligned: list[str] = []
    for stage in ("find", "access", "pickup", "transform", "place", "other"):
        candidates = sorted(
            (
                (step, count)
                for step, count in occurrence.items()
                if stage_for_abstracted_action(step) == stage
                and count >= support_needed
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if not candidates:
            continue
        limit = core_limits.get(stage)
        if limit is None:
            aligned.extend(step for step, _count in candidates)
        else:
            aligned.extend(step for step, _count in candidates[:limit])

    # Soft-fill a second pickup/place for multi-object when two distinct
    # typed cycles recur at least twice (below hard majority is OK).
    if str(capability or "").strip() == "track.multiple_objects":
        soft = max(2, support_needed - 1)
        for stage, limit in (("pickup", 2), ("place", 2)):
            present = [
                step
                for step in aligned
                if stage_for_abstracted_action(step) == stage
            ]
            if len(present) >= limit:
                continue
            ranked = sorted(
                (
                    (step, count)
                    for step, count in occurrence.items()
                    if stage_for_abstracted_action(step) == stage
                    and step not in present
                    and count >= soft
                ),
                key=lambda item: (-item[1], item[0]),
            )
            for step, _count in ranked:
                aligned.append(step)
                present.append(step)
                if len(present) >= limit:
                    break

    # Do NOT fall back to a single-trace clone. Thicken from evidence stages.
    aligned = _ensure_evidence_stage_slots(
        aligned,
        keyed_protocols,
        first_index_sums=first_index_sums,
        first_index_counts=first_index_counts,
        capability=capability,
    )
    # Salvaged raw steps may still carry concrete objects — re-key.
    aligned = [_consensus_action_key(step) for step in aligned]
    # Unique while preserving order after salvage.
    deduped: list[str] = []
    seen_aligned: set[str] = set()
    for step in aligned:
        if step in seen_aligned:
            continue
        seen_aligned.add(step)
        deduped.append(step)
    aligned = deduped

    # Place each consensus key (and recurrent find copies) at the median of
    # its 1st / 2nd / ... occurrence indices across wins.
    positioned: list[tuple[float, int, str]] = []
    for step in aligned:
        ordinals = ordinal_index_lists.get(step) or []
        if stage_for_abstracted_action(step) == "find":
            max_copies = min(
                2,
                max(1, _median_int(key_count_samples.get(step, [1]))),
            )
        else:
            max_copies = 1
        for ordinal in range(max_copies):
            soft = support_needed if ordinal == 0 else max(2, support_needed - 1)
            if ordinal < len(ordinals) and len(ordinals[ordinal]) >= soft:
                pos = float(_median_int(ordinals[ordinal]))
            elif ordinal == 0:
                pos = first_index_sums.get(step, 0.0) / max(
                    1, first_index_counts.get(step, 1)
                )
            else:
                # Missing later ordinal evidence: skip rather than clone front.
                continue
            stage_rank = (
                _STAGE_ORDER.index(stage_for_abstracted_action(step))
                if stage_for_abstracted_action(step) in _STAGE_ORDER
                else len(_STAGE_ORDER)
            )
            positioned.append((pos, stage_rank, step))
    positioned.sort(key=lambda item: (item[0], item[1], item[2]))
    aligned = [step for _pos, _rank, step in positioned]
    max_finds = 4 if _keeps_episode_search_hops(capability) else 6
    aligned = _cap_consensus_find_hops(
        aligned,
        occurrence=occurrence,
        max_finds=max_finds,
    )
    if merge_trace is not None:
        merge_trace.update(
            {
                "mode": "typed_stage_consensus",
                "consensus_support": int(support_needed),
                "evidence_trace_count": len(keyed_protocols),
                "aligned_steps": len(aligned),
                "note": (
                    "Majority typed steps across wins; movable objects "
                    "slotted to <object>, places/tools keep types; "
                    "recurrent finds use ordinal median indices."
                ),
            }
        )
    return canonicalize_protocol_stages(aligned, capability=capability)


def _select_median_length_protocol(
    protocols: list[list[str]],
    *,
    capability: str = "",
) -> list[str]:
    """Pick a representative winning protocol (diagnostic helper only)."""
    if not protocols:
        return []
    complete = [
        protocol
        for protocol in protocols
        if not protocol_structure_issues(protocol, capability=capability)
    ]
    pool = complete or protocols
    ordered = sorted(
        pool,
        key=lambda protocol: (
            len(protocol),
            sum(
                1
                for step in protocol
                if stage_for_abstracted_action(step) == "find"
            ),
        ),
    )
    return list(ordered[len(ordered) // 2])
