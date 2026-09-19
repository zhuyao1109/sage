"""Simplified skill protocol distillation: merge wins, drop detours, online utility.

Pipeline:
1. Cluster trajectories into capability seeds (caller / HeuristicSkillDistiller).
2. Collect successful trajectories for the same capability, shortest causal
   backbone first (a clean 4-step win beats a 39-step flail as evidence).
3. Compress each win to its causal backbone: object-acquisition pickup,
   capability transform, terminal place, plus the go-to / open each core op
   causally needs. Search detours never enter the protocol.
4. Merge receptacle-homogeneous groups only; when no group has support >= 2,
   fall back to the majority slotted shape (``<source>`` / ``<destination>``,
   tools concrete). Merged results that fail the causal gate fall back to
   the group's shortest observed backbone.
   A data-derived search prior (where wins first acquired the target object)
   is distilled alongside the protocol into ``metadata["search_prior"]`` and
   the precondition text.
5. Enter the bank as provisional with initial utility.
6. Online credit updates promote / retire from use outcomes.

Form / noise gates (after merge):
- Require enough winning traces (``min_wins``).
- Prefer capability-marker wins and scene-diverse evidence.
- Reject over-collapsed skeletons (short protocols / high collapse ratio).
- Reject structural gaps via ``protocol_structure_issues`` and causal
  contradictions (closed-receptacle use, location mismatch, navigation
  stutter) via ``protocol_causal_issues``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from sage_mas.executable_protocol import ensure_executable_protocol
from sage_mas.schemas import AtomicOp, Skill, SkillStatus
from sage_mas.skill_credit import SkillCreditPolicy, initialize_skill_credit
from sage_mas.skill_quality import annotate_skill_identity
from sage_mas.trajectory.abstraction import (
    is_productive_environment_step,
    stage_for_abstracted_action,
    trajectory_action_steps,
    trajectory_id_from_steps,
)
from sage_mas.trajectory.causal import (
    extract_causal_backbone,
    protocol_causal_issues,
    slot_scene_receptacles,
)
from sage_mas.trajectory.consensus import _align_protocols_across_trajectories
from sage_mas.trajectory.grounding import (
    checkable_expected_effect_text,
    checkable_precondition_text,
    extract_anti_patterns,
)
from sage_mas.trajectory.protocol_canon import protocol_structure_issues


INITIAL_UTILITY = 0.5
DEFAULT_MIN_PROTOCOL_STEPS = 4
DEFAULT_MAX_COLLAPSE_RATIO = 2.5
DEFAULT_MIN_SCENE_DIVERSITY = 2


def _protocol_stage_richness(
    protocol: list[str],
    *,
    capability: str = "",
) -> tuple[int, int, int]:
    """Return (unique_stage_count, productive_stage_steps, protocol_len).

    Capability is unused: richness is evidence-derived, not a family checklist.
    """
    del capability  # kept for call-site compatibility
    stages = [
        stage_for_abstracted_action(step)
        for step in protocol
        if stage_for_abstracted_action(step) != "other"
    ]
    unique = len(set(stages))
    return (unique, len(stages), len(protocol))

def _trajectory_stage_richness(
    steps: list[Any],
    *,
    capability: str = "",
    max_protocol_steps: int = 16,
) -> tuple[int, int, int]:
    protocol = remove_detours_from_trajectory(
        steps,
        capability=capability,
        max_protocol_steps=max_protocol_steps,
    )
    return _protocol_stage_richness(protocol, capability=capability)


def _trajectory_backbone_length(
    steps: list[Any],
    *,
    capability: str = "",
) -> int:
    """Causal-backbone length; ``10**9`` when no chain is extractable."""
    backbone = remove_detours_from_trajectory(steps, capability=capability)
    if not backbone:
        return 10**9
    return len(backbone)


def _backbone_signature(protocol: list[str]) -> tuple:
    """Concrete (sources, tools, targets) identity of a cleaned backbone."""
    sources: list[str] = []
    tools: list[str] = []
    targets: list[str] = []
    for step in protocol:
        stage = stage_for_abstracted_action(step)
        if stage == "pickup":
            match = re.match(r"^take .+ from (.+)$", step)
            sources.append((match.group(1) if match else "").strip())
        elif stage == "place":
            match = re.match(r"^(?:move|put) .+ to (.+)$", step)
            targets.append((match.group(1) if match else "").strip())
        elif stage == "transform":
            match = re.match(r"^(?:clean|heat|cool) .+ with (.+)$", step)
            if match:
                tools.append(match.group(1).strip())
            elif step.startswith("use "):
                tools.append(step[len("use ") :].strip())
    return (tuple(sources), tuple(tools), tuple(targets))


def select_successful_capability_trajectories(
    seed: Skill,
    trajectories: list[list[Any]],
    *,
    max_trajectories: int = 5,
    require_capability_marker: bool = True,
    min_scene_diversity: int = DEFAULT_MIN_SCENE_DIVERSITY,
) -> list[list[Any]]:
    """Step 2: keep winning same-family / same-capability episodes.

    Prefer stage-rich, capability-marker, scene-diverse wins so merge consensus
    is not dominated by short skeleton traces.
    """
    family = str(seed.metadata.get("primary_task_family", "") or "")
    capability = str(seed.capability_key or "").strip().lower()
    marked: list[list[Any]] = []
    family_wins: list[list[Any]] = []
    for steps in trajectories:
        if not steps:
            continue
        terminal = steps[-1]
        if not bool(terminal.metadata.get("won", False)):
            continue
        step_family = str(terminal.metadata.get("task_family", "") or "")
        if family and step_family and step_family != family:
            continue
        family_wins.append(steps)
        if not capability or _trajectory_mentions_capability(steps, capability):
            marked.append(steps)

    if require_capability_marker:
        if marked:
            wins = marked
        else:
            # Hard fail closed: do not silently fall back to unmarked family wins.
            wins = []
    elif marked:
        wins = marked
    else:
        wins = family_wins

    # Prefer short causal backbones first: a clean 4-step win is better
    # protocol evidence than a 39-step flail that eventually succeeded.
    # Wins with no extractable causal chain carry no protocol signal.
    ranked = [
        (
            _trajectory_backbone_length(steps, capability=capability),
            trajectory_id_from_steps(steps),
            steps,
        )
        for steps in wins
    ]
    ranked.sort(key=lambda item: (item[0], item[1]))
    wins = [item[2] for item in ranked if item[0] < 10**9]
    selected: list[list[Any]] = []
    seen_scenes: set[str] = set()
    limit = max(1, int(max_trajectories))
    # Pass 1: unique scenes, already ordered by richness.
    for steps in wins:
        scene = _scene_key(steps)
        if scene in seen_scenes:
            continue
        selected.append(steps)
        seen_scenes.add(scene)
        if len(selected) >= limit:
            break
    # Pass 2: fill remaining slots even if scenes repeat.
    if len(selected) < limit:
        selected_ids = {trajectory_id_from_steps(steps) for steps in selected}
        for steps in wins:
            tid = trajectory_id_from_steps(steps)
            if tid in selected_ids:
                continue
            selected.append(steps)
            selected_ids.add(tid)
            if len(selected) >= limit:
                break

    seed.metadata["protocol_win_scene_count"] = len(
        {_scene_key(steps) for steps in selected}
    )
    seed.metadata["protocol_marked_win_count"] = len(marked)
    seed.metadata["protocol_family_win_count"] = len(family_wins)
    seed.metadata["protocol_selected_stage_coverage"] = [
        _trajectory_stage_richness(steps, capability=capability)[0]
        for steps in selected
    ]
    # Soft signal only; hard reject happens after merge when diversity is low
    # and the protocol also collapses.
    seed.metadata["protocol_scene_diversity_target"] = max(
        1, int(min_scene_diversity)
    )
    return selected


def remove_detours_from_trajectory(
    steps: list[Any],
    *,
    capability: str = "",
    max_protocol_steps: int = 16,
) -> list[str]:
    """Step 3: compress a win to its causal backbone (no search detours).

    The backbone is the object-acquisition pickup, the capability transform,
    and the terminal place, plus the last go-to / open each core op needs.
    """
    return extract_causal_backbone(
        steps,
        capability=capability,
        max_protocol_steps=max_protocol_steps,
    )


def merge_cleaned_protocols(
    protocols: list[list[str]],
    *,
    capability: str = "",
    merge_trace: dict[str, Any] | None = None,
) -> list[str]:
    """Step 4: merge cleaned wins without collapsing search hops to one step."""
    return _align_protocols_across_trajectories(
        protocols,
        capability=capability,
        merge_trace=merge_trace,
    )


def merge_backbone_group(
    cleaned: list[list[str]],
    *,
    capability: str = "",
    merge_trace: dict[str, Any] | None = None,
) -> list[str]:
    """Step 4 (new): receptacle-aware grouped merge over causal backbones.

    Backbones only merge inside a homogeneous (sources, tools, targets)
    group — cross-scene voting produces causally impossible chimeras. When
    no group reaches support >= 2, the consensus is the protocol *shape*:
    the shortest backbone with source/target slotted to ``<source>`` /
    ``<target>`` (tools stay concrete). A merged result that fails the
    causal gate falls back to the group's shortest member.
    """
    if merge_trace is not None:
        merge_trace.clear()
        merge_trace.update({"mode": "empty", "input_protocols": len(cleaned)})
    if not cleaned:
        return []

    groups: dict[tuple, list[list[str]]] = defaultdict(list)
    for protocol in cleaned:
        groups[_backbone_signature(protocol)].append(protocol)
    ranked_groups = sorted(
        groups.values(),
        key=lambda group: (-len(group), sum(map(len, group)) / len(group)),
    )
    best_group = ranked_groups[0]

    # Concrete receptacle claims need majority evidence; a plurality group
    # only proves the *shape* generalizes, not the receptacle types.
    strong_group = len(best_group) >= 2 and len(best_group) * 2 > len(cleaned)
    if strong_group:
        protocol = merge_cleaned_protocols(
            best_group,
            capability=capability,
            merge_trace=merge_trace,
        )
        gate_issues = protocol_causal_issues(
            protocol, capability=capability
        ) + protocol_structure_issues(protocol, capability=capability)
        if protocol and not gate_issues:
            if merge_trace is not None:
                merge_trace["group_support"] = len(best_group)
                merge_trace["group_count"] = len(groups)
            return protocol
        # Homogeneous-group merge still scrambled causality: trust the
        # shortest observed backbone instead of the chimera.
        if merge_trace is not None:
            merge_trace["mode"] = "grouped_shortest_representative"
            merge_trace["causal_fallback_issues"] = gate_issues
            merge_trace["group_support"] = len(best_group)
            merge_trace["group_count"] = len(groups)
        return list(min(best_group, key=len))

    # All groups are singletons: vote on the slotted shape instead.
    shape_groups: dict[tuple, list[list[str]]] = defaultdict(list)
    for protocol in cleaned:
        shape_groups[tuple(slot_scene_receptacles(protocol))].append(protocol)
    ranked_shapes = sorted(
        shape_groups.values(),
        key=lambda group: (-len(group), sum(map(len, group)) / len(group)),
    )
    best_shape = ranked_shapes[0]
    representative = min(best_shape, key=len)
    if merge_trace is not None:
        merge_trace.update(
            {
                "mode": "slotted_shape_consensus",
                "shape_support": len(best_shape),
                "shape_count": len(shape_groups),
                "group_count": len(groups),
                "evidence_trace_count": len(cleaned),
                "note": (
                    "No receptacle-homogeneous group; source/target slotted "
                    "to <source>/<target>, tools kept concrete; shortest "
                    "clean backbone of the majority shape wins."
                ),
            }
        )
    return slot_scene_receptacles(representative)


def assess_protocol_form_quality(
    protocol: list[str],
    cleaned: list[list[str]],
    *,
    capability: str = "",
    min_protocol_steps: int = DEFAULT_MIN_PROTOCOL_STEPS,
    max_collapse_ratio: float = DEFAULT_MAX_COLLAPSE_RATIO,
    scene_diversity: int = 1,
    min_scene_diversity: int = DEFAULT_MIN_SCENE_DIVERSITY,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Return (ok, reasons, metrics) for merged-protocol form / noise gates."""
    reasons: list[str] = []
    mean_clean = (
        sum(len(item) for item in cleaned) / len(cleaned) if cleaned else 0.0
    )
    collapse_ratio = mean_clean / max(1, len(protocol))
    metrics = {
        "protocol_len": len(protocol),
        "mean_cleaned_len": mean_clean,
        "collapse_ratio": collapse_ratio,
        "scene_diversity": int(scene_diversity),
        "min_protocol_steps": int(min_protocol_steps),
        "max_collapse_ratio": float(max_collapse_ratio),
    }
    structure_issues = protocol_structure_issues(
        protocol,
        capability=capability,
    )
    reasons.extend(structure_issues)
    causal_issues = protocol_causal_issues(protocol, capability=capability)
    reasons.extend(causal_issues)
    metrics["causal_issues"] = len(causal_issues)

    min_steps = max(1, int(min_protocol_steps))
    apply_length_floor = True
    if capability.startswith("transform."):
        # Long teacher episodes must not collapse to a 5-verb skeleton.
        # Short complete wins (take → transform → place) keep the old
        # structure-only gate so toy / already-at-tool traces still distill.
        if mean_clean >= 7:
            min_steps = max(min_steps, 6)
        else:
            apply_length_floor = False
    # The length floor exists to catch over-collapsed noise. A short but
    # causally coherent backbone (e.g. go→take→use desklamp) is a valid
    # minimal recipe, so the floor only bites when the causal gate also
    # flags problems or the protocol is below the hard 3-step minimum.
    if (
        apply_length_floor
        and len(protocol) < min_steps
        and (causal_issues or len(protocol) < 3)
    ):
        reason = (
            f"merged protocol too short "
            f"(len={len(protocol)} < min_protocol_steps={min_steps})"
        )
        if reason not in reasons:
            reasons.append(reason)

    collapse_limit = float(max_collapse_ratio)
    if capability.startswith("transform."):
        collapse_limit = min(collapse_limit, 1.6)
    if cleaned and collapse_ratio > collapse_limit:
        reasons.append(
            f"merge over-collapsed "
            f"(mean_cleaned={mean_clean:.1f} -> merged={len(protocol)}, "
            f"ratio={collapse_ratio:.2f} > {collapse_limit:.2f})"
        )

    if (
        int(scene_diversity) < max(1, int(min_scene_diversity))
        and len(protocol) <= min_steps
    ):
        reasons.append(
            f"low scene diversity ({scene_diversity} < {min_scene_diversity}) "
            "with a short merged protocol"
        )

    return (not reasons), reasons, metrics


def extract_search_prior(
    wins: list[list[Any]],
    *,
    max_sources: int = 3,
) -> dict[str, Any]:
    """Learn a search-order hint from where winning episodes first acquired
    each target object.

    Only the first ``take`` of each object instance per episode counts —
    re-grabs after a transform (e.g. taking the heated plate back out of the
    microwave) are not acquisition evidence. The distribution is distilled
    from data, never hand-coded.
    """
    counts: dict[str, int] = {}
    episodes = 0
    for steps in wins:
        seen_objects: set[str] = set()
        found = False
        for step in trajectory_action_steps(steps):
            action = str(getattr(step, "action", "") or "").strip()
            if not action.startswith("take ") or " from " not in action:
                continue
            obj, _, recep = action[len("take ") :].partition(" from ")
            obj = obj.strip()
            recep = re.sub(r"\s+\d+$", "", recep.strip())
            if not obj or not recep or obj in seen_objects:
                continue
            seen_objects.add(obj)
            counts[recep] = counts.get(recep, 0) + 1
            found = True
        if found:
            episodes += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "sources": [
            {"source": name, "count": count}
            for name, count in ranked[: max(1, int(max_sources))]
        ],
        "episodes": episodes,
        "acquisitions": sum(counts.values()),
    }


def search_prior_sentence(prior: dict[str, Any]) -> str:
    """Human-readable precondition suffix for a distilled search prior."""
    sources = prior.get("sources") or []
    episodes = int(prior.get("episodes") or 0)
    if not sources or episodes <= 0:
        return ""
    parts = ", ".join(f"{entry['source']} ({entry['count']})" for entry in sources)
    return (
        f"Search prior from {episodes} winning episodes: the target object "
        f"was found at {parts}. Check likely sources first, then expand "
        "the search."
    )


def build_protocol_from_successes(
    seed: Skill,
    trajectories: list[list[Any]],
    *,
    max_trajectories: int = 5,
    max_protocol_steps: int = 16,
    min_wins: int = 3,
    min_protocol_steps: int = DEFAULT_MIN_PROTOCOL_STEPS,
    max_collapse_ratio: float = DEFAULT_MAX_COLLAPSE_RATIO,
    min_scene_diversity: int = DEFAULT_MIN_SCENE_DIVERSITY,
    require_capability_marker: bool = True,
) -> Skill:
    """Steps 2–4: multi-win detour-clean merge into one action_protocol.

    Prefer ``min_wins >= 3`` so the shared protocol is not a single-trace
    clone. Fewer wins / shallow / over-collapsed merges → REJECTED.
    """
    skill = deepcopy(seed)
    capability = str(skill.capability_key or "").strip()
    wins = select_successful_capability_trajectories(
        skill,
        trajectories,
        max_trajectories=max_trajectories,
        require_capability_marker=require_capability_marker,
        min_scene_diversity=min_scene_diversity,
    )
    scene_diversity = len({_scene_key(steps) for steps in wins})
    cleaned = [
        protocol
        for protocol in (
            remove_detours_from_trajectory(
                steps,
                capability=capability,
                max_protocol_steps=max_protocol_steps,
            )
            for steps in wins
        )
        if protocol
    ]
    # Backbones that already fail the stage-shape gate carry no protocol
    # signal (e.g. lamp-only wins missing the acquisition pickup).
    cleaned = [
        protocol
        for protocol in cleaned
        if not protocol_structure_issues(protocol, capability=capability)
    ]
    if not cleaned:
        skill.metadata["protocol_alignment_ok"] = False
        skill.metadata["protocol_structure_ok"] = False
        skill.metadata["protocol_form_ok"] = False
        skill.metadata["protocol_quality_reject"] = (
            "no productive actions after detour filtering"
        )
        skill.status = SkillStatus.REJECTED
        return skill

    merge_trace: dict[str, Any] = {}
    protocol = merge_backbone_group(
        cleaned,
        capability=capability,
        merge_trace=merge_trace,
    )
    if not protocol:
        skill.metadata["protocol_alignment_ok"] = False
        skill.metadata["protocol_structure_ok"] = False
        skill.metadata["protocol_form_ok"] = False
        skill.metadata["protocol_quality_reject"] = (
            "empty protocol (no causal backbone extractable from wins)"
        )
        skill.status = SkillStatus.REJECTED
        return skill

    align_support = len(wins)
    consensus_support = int(
        merge_trace.get("consensus_support")
        or merge_trace.get("shape_support")
        or merge_trace.get("group_support")
        or align_support
    )
    merge_mode = str(merge_trace.get("mode") or "unknown")
    alignment_ok = align_support >= max(1, int(min_wins))
    form_ok, form_reasons, form_metrics = assess_protocol_form_quality(
        protocol,
        cleaned,
        capability=capability,
        min_protocol_steps=min_protocol_steps,
        max_collapse_ratio=max_collapse_ratio,
        scene_diversity=scene_diversity,
        min_scene_diversity=min_scene_diversity,
    )
    # Structure issues are included inside form_reasons.
    structure_issues = [
        reason
        for reason in form_reasons
        if "over-collapsed" not in reason
        and "too short" not in reason
        and "scene diversity" not in reason
    ]
    structure_ok = not structure_issues

    family = str(
        skill.metadata.get("primary_task_family")
        or (
            wins[0][-1].metadata.get("task_family", "other")
            if wins
            else "other"
        )
        or "other"
    )
    stages: list[str] = []
    for step in protocol:
        stage = stage_for_abstracted_action(step)
        if not stages or stages[-1] != stage:
            stages.append(stage)

    label = capability.replace(".", " ").replace("_", " ").strip()
    skill.skill_name = (
        skill.skill_name
        if skill.skill_name and not skill.skill_name.startswith("Observed ")
        else f"Merged {label or 'execution'} protocol"
    )
    if merge_mode in {
        "median_length_representative",
        "single_trace",
        "grouped_shortest_representative",
    }:
        skill.description = (
            f"Representative causal-backbone protocol from {align_support} "
            f"successful {family} trajectories after detour removal "
            f"(merge_mode={merge_mode}; consensus_support={consensus_support})."
        )
        skill.trajectory_summary = (
            f"Selected a representative win among {align_support} "
            f"{family} traces → {len(protocol)} typed protocol steps "
            f"(not a step-wise multi-trace intersection)."
        )
    elif merge_mode == "slotted_shape_consensus":
        skill.description = (
            f"Slotted-shape consensus protocol from {align_support} "
            f"successful {family} trajectories after causal-backbone "
            f"extraction (source/target slotted; tools concrete)."
        )
        skill.trajectory_summary = (
            f"Extracted causal backbones from {align_support} winning "
            f"{family} traces; no receptacle-homogeneous group, so the "
            f"majority slotted shape won → {len(protocol)} protocol steps."
        )
    else:
        skill.description = (
            f"Typed stage-consensus protocol from {align_support} "
            f"successful {family} trajectories after causal-backbone "
            f"extraction (merge_mode={merge_mode}; "
            f"majority_support>={consensus_support})."
        )
        skill.trajectory_summary = (
            f"Merged {align_support} winning {family} traces into "
            f"{len(protocol)} consensus protocol steps "
            f"(objects slotted; places/tools typed; mode={merge_mode})."
        )
    skill.precondition = checkable_precondition_text(
        family=family,
        capability=capability,
        stages=stages,
    )
    # Data-derived search prior: where winning episodes first acquired the
    # target object. Surfaces in the injected SOP so the executor searches
    # likely sources first instead of anchoring on the destination.
    search_prior = extract_search_prior(wins)
    prior_sentence = search_prior_sentence(search_prior)
    if prior_sentence:
        skill.precondition = f"{skill.precondition} {prior_sentence}"
    skill.action_protocol = list(protocol)
    skill.applicable_atomic_ops = [AtomicOp.ACT]
    skill.expected_effect = checkable_expected_effect_text(
        capability=capability,
        family=family,
    )
    skill.suggested_role = skill.suggested_role or "Executor"
    skill.metadata["anti_patterns"] = extract_anti_patterns(wins)
    skill.metadata["search_prior"] = search_prior
    skill.metadata["protocol_source"] = "merged_success_detour_clean_v4_backbone"
    skill.metadata["protocol_merge_mode"] = merge_mode
    skill.metadata["protocol_merge_trace"] = dict(merge_trace)
    skill.metadata["protocol_stages"] = stages
    skill.metadata["protocol_alignment_support"] = align_support
    skill.metadata["protocol_consensus_support"] = consensus_support
    skill.metadata["protocol_alignment_ok"] = alignment_ok
    skill.metadata["protocol_structure_ok"] = structure_ok
    skill.metadata["protocol_structure_issues"] = structure_issues
    skill.metadata["protocol_form_ok"] = bool(form_ok and alignment_ok)
    skill.metadata["protocol_form_metrics"] = form_metrics
    skill.metadata["protocol_merge_trace_count"] = len(cleaned)
    skill.metadata["protocol_win_scene_count"] = scene_diversity
    skill.metadata["seed_template_discarded"] = True
    if wins:
        skill.metadata["protocol_source_trajectory_id"] = trajectory_id_from_steps(
            wins[0]
        )

    if alignment_ok and form_ok:
        skill.metadata.pop("protocol_quality_reject", None)
    else:
        reasons: list[str] = []
        if not alignment_ok:
            reasons.append(
                f"need>={min_wins} winning trajectories, got {align_support}"
            )
        reasons.extend(form_reasons)
        skill.metadata["protocol_quality_reject"] = "; ".join(reasons)
        # Noisy / single-trace / over-collapsed protocols must not enter the bank
        # as ADD_AGENT signals.
        skill.status = SkillStatus.REJECTED
        skill.metadata["protocol_form_ok"] = False

    # Ground the executable protocol in the representative winning trajectory
    # so executable_protocol_source="trajectory" (not "action_protocol"). This
    # satisfies the require_trajectory_executable_protocol gate for ADD_AGENT:
    # the executable step schema carries real per-step observation hints from
    # a winning episode, not just a mirror of the abstract merged protocol.
    #
    # Only Act steps carry real environment observations; Communicate steps
    # (think text with an embedded <action> tag) have observation=None. If
    # Communicate steps are allowed through, steps_from_trajectory dedups
    # them ahead of the matching Act step (same action_template), so the
    # real observation is lost and expected_obs_hint stays empty. Filter on
    # atomic_op=="Act" to keep only environment-effect steps.
    representative = wins[0] if wins else []
    trajectory_records = [
        {
            "action": getattr(step, "action", "") or "",
            "observation": getattr(step, "observation", "") or "",
        }
        for step in representative
        if getattr(step, "action", None)
        and getattr(step, "atomic_op", "") == "Act"
        and is_productive_environment_step(step)
    ]
    ensure_executable_protocol(
        skill,
        force=True,
        trajectory_steps=trajectory_records,
        max_steps=max(max_protocol_steps, 16),
    )
    skill.evidence_ids = sorted(
        {
            *skill.evidence_ids,
            *(trajectory_id_from_steps(steps) for steps in wins),
        }
    )
    skill.support_count = max(skill.support_count, len(skill.evidence_ids))
    annotate_skill_identity(skill)
    return skill


def initialize_provisional_utility(
    skill: Skill,
    *,
    policy: SkillCreditPolicy | None = None,
    initial_utility: float = INITIAL_UTILITY,
) -> Skill:
    """Step 5: provisional bank entry with prior utility score.

    Leave ``marginal_utility`` unset until online credit (or a real MU probe)
    observes uses. MU=0.0 would otherwise look like "no advantage" and lose
    injection ranking to older positive-MU skills before the new skill is tried.
    """
    credit_policy = policy or SkillCreditPolicy()
    initialize_skill_credit(skill, credit_policy)
    credit = skill.metadata["skill_credit"]
    credit["uses"] = 0
    credit["successes"] = 0
    credit["score"] = float(initial_utility)
    credit["events"] = []
    skill.status = SkillStatus.PROVISIONAL
    skill.metadata["utility"] = float(initial_utility)
    skill.marginal_utility = None
    skill.metadata.pop("marginal_utility", None)
    skill.metadata["utility_source"] = "online_credit_prior"
    return skill


def sync_utility_from_credit(skill: Skill) -> float:
    """Mirror credit score into utility after online updates (step 6)."""
    credit = skill.metadata.get("skill_credit") or {}
    score = float(credit.get("score", INITIAL_UTILITY))
    skill.metadata["utility"] = score
    uses = int(credit.get("uses", 0) or 0)
    if uses > 0:
        # Advantage vs the provisional prior, only after real online evidence.
        skill.marginal_utility = score - INITIAL_UTILITY
        skill.metadata["marginal_utility"] = score - INITIAL_UTILITY
    else:
        skill.marginal_utility = None
        skill.metadata.pop("marginal_utility", None)
    skill.metadata["utility_source"] = "online_credit"
    return score


def _scene_key(steps: list[Any]) -> str:
    """Coarse scene identity from gamefile / task id for diversity selection."""
    if not steps:
        return "empty"
    tid = str(trajectory_id_from_steps(steps) or "").strip()
    if not tid:
        task = str(steps[-1].metadata.get("task", "") or "").strip().lower()
        return task or "unknown"
    path = Path(tid)
    # .../look_at_obj_in_light-Bowl-None-DeskLamp-308/trial_.../game.tw-pddl
    for part in reversed(path.parts):
        if part.endswith(".tw-pddl") or part.startswith("trial_"):
            continue
        if "-" in part:
            return part
    return path.parent.name or tid


def _trajectory_mentions_capability(
    steps: list[Any],
    capability: str,
) -> bool:
    capability = str(capability or "").strip().lower()
    if not capability:
        return True
    marker = ""
    if capability.startswith("transform."):
        marker = capability.split(".", 1)[1]
    elif capability in {"inspect.with_light", "inspect.light"}:
        marker = "desklamp"
    elif capability.startswith("track."):
        return True
    if not marker:
        return True
    for step in steps:
        action = str(getattr(step, "action", "") or "").lower()
        observation = str(getattr(step, "observation", "") or "").lower()
        if marker in action or marker in observation:
            return True
    return False
