"""Ground skills in aligned successful trajectories."""

from __future__ import annotations

from sage_mas.schemas import AtomicOp, AtomicStep, Skill
from sage_mas.trajectory.abstraction import (
    _abstract_environment_action,
    _capability_operation_marker,
    _observation_is_noop,
    is_productive_environment_step,
    stage_for_abstracted_action,
    trajectory_action_steps,
    trajectory_id_from_steps,
)
from sage_mas.trajectory.consensus import (
    _align_protocols_across_trajectories,
)
from sage_mas.trajectory.protocol_canon import (
    _filtered_protocol_from_trajectory,
    protocol_structure_issues,
)

def extract_anti_patterns(
    trajectories: list[list[AtomicStep]],
    *,
    max_patterns: int = 5,
) -> list[str]:
    """Mine repeated invalid / noop actions as avoid-patterns."""
    scored: dict[str, float] = {}
    for steps in trajectories:
        if not steps:
            continue
        won = bool(steps[-1].metadata.get("won", False))
        for step in trajectory_action_steps(steps):
            action = str(step.action or "").strip()
            if not action:
                continue
            invalid = step.metadata.get("is_action_valid") is False
            stalled = bool(step.metadata.get("stalled"))
            noop = _observation_is_noop(step.observation)
            repeated = bool(step.metadata.get("repeated_action"))
            if not (invalid or stalled or noop or repeated):
                continue
            abstracted = _abstract_environment_action(action)
            if not abstracted:
                continue
            weight = 1.0
            if invalid:
                weight += 1.5
            if stalled or noop:
                weight += 1.0
            if repeated:
                weight += 1.0
            if not won:
                weight += 0.5
            scored[abstracted] = scored.get(abstracted, 0.0) + weight

    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    patterns: list[str] = []
    for abstracted, _score in ranked[:max_patterns]:
        patterns.append(
            f"Avoid repeating `{abstracted}` after no-op / invalid feedback "
            "(Nothing happens / invalid action)."
        )
    return patterns


def checkable_precondition_text(
    *,
    family: str,
    capability: str,
    stages: list[str],
) -> str:
    """Build a precondition that can be checked against task + history."""
    parts = [
        f"Task family matches `{family}`.",
        "Instantiate every `<placeholder>` from the current observation and "
        "admissible actions only.",
    ]
    marker = _capability_operation_marker(capability)
    if marker and marker != "use":
        parts.append(
            f"Task is not yet won. Follow the protocol to achieve a confirmed "
            f"`{marker}` effect and complete `{family}`."
        )
    elif marker == "use":
        parts.append(
            "Task is not yet won. Follow the protocol through desk-lamp use "
            f"until the `{family}` inspection goal completes."
        )
    if stages:
        parts.append(
            "Follow stage order observed in successful evidence: "
            + " -> ".join(stages)
            + "."
        )
    return " ".join(parts)


def checkable_expected_effect_text(*, capability: str, family: str) -> str:
    """Build an effect claim that can be verified from env feedback."""
    marker = _capability_operation_marker(capability)
    if marker and marker != "use":
        return (
            f"Environment observation confirms a successful `{marker}` "
            f"effect, then the object is placed to satisfy `{family}`; "
            "or the episode is marked won."
        )
    if marker == "use":
        return (
            "Environment observation confirms desk-lamp use and the "
            f"inspection goal for `{family}` completes (episode won)."
        )
    return (
        f"Episode reaches a won terminal state on `{family}` after following "
        "the staged protocol with placeholders grounded in admissible actions."
    )


def _trajectory_has_grounded_progress(steps: list[AtomicStep]) -> bool:
    """Won episode or positive goal-progress delta counts as grounding support."""
    if not steps:
        return False
    if bool(steps[-1].metadata.get("won", False)):
        return True
    return any(
        float(step.metadata.get("goal_progress_delta", 0.0) or 0.0) > 0.0
        for step in steps
    )


def ground_skill_in_successful_trajectory(
    skill: Skill,
    trajectories: list[list[AtomicStep]],
    *,
    max_protocol_steps: int = 16,
    min_alignment_wins: int = 1,
) -> Skill:
    """Ground a skill in filtered, cross-trajectory-aligned success protocols."""
    successful = [
        steps
        for steps in trajectories
        if steps and bool(steps[-1].metadata.get("won", False))
    ]
    evidence = successful or [
        steps
        for steps in trajectories
        if steps
    ]
    if not evidence:
        skill.metadata["protocol_alignment_ok"] = False
        skill.metadata["protocol_structure_ok"] = False
        skill.metadata["protocol_quality_reject"] = "no evidence trajectories"
        return skill

    capability = str(skill.capability_key or "").strip()
    previous_align = int(skill.metadata.get("protocol_alignment_support") or 0)
    previous_ok = bool(skill.metadata.get("protocol_alignment_ok", False))
    previous_protocol = list(skill.action_protocol or [])

    # Refuse to overwrite a stronger multi-trace protocol with a weaker one.
    if (
        len(successful) < min_alignment_wins
        and previous_ok
        and previous_align >= min_alignment_wins
        and previous_protocol
    ):
        skill.metadata["protocol_quality_reject"] = (
            f"kept prior protocol; current grounding has only "
            f"{len(successful)} winning trajectories"
        )
        return skill

    per_trace_protocols = [
        protocol
        for protocol in (
            _filtered_protocol_from_trajectory(
                steps,
                capability=capability,
                max_protocol_steps=max_protocol_steps,
            )
            for steps in successful or evidence
        )
        if protocol
    ]
    if not per_trace_protocols:
        skill.metadata["protocol_alignment_ok"] = False
        skill.metadata["protocol_structure_ok"] = False
        skill.metadata["protocol_quality_reject"] = (
            "no productive actions after noise filtering"
        )
        return skill

    protocol = _align_protocols_across_trajectories(
        per_trace_protocols,
        capability=capability,
    )
    if not protocol:
        skill.metadata["protocol_alignment_ok"] = False
        skill.metadata["protocol_structure_ok"] = False
        skill.metadata["protocol_quality_reject"] = "alignment produced empty protocol"
        return skill

    structure_issues = protocol_structure_issues(
        protocol,
        capability=capability,
    )
    align_support = len(successful)
    if align_support < min_alignment_wins:
        # Sparse-win segments: count progress-backed failures so two-obj /
        # cross-segment distillation is not starved into no_candidates.
        progress_backed = [
            steps
            for steps in (successful or evidence)
            if _trajectory_has_grounded_progress(steps)
        ]
        align_support = max(align_support, len(progress_backed))
    alignment_ok = align_support >= min_alignment_wins
    structure_ok = not structure_issues

    # Prefer the shortest winning source that covers the aligned protocol.
    source = min(
        successful or evidence,
        key=lambda steps: (
            0
            if set(protocol).issubset(
                set(
                    _filtered_protocol_from_trajectory(
                        steps,
                        capability=capability,
                        max_protocol_steps=max_protocol_steps,
                    )
                )
            )
            else 1,
            int(steps[-1].metadata.get("num_steps", len(steps))),
            trajectory_id_from_steps(steps),
        ),
    )
    terminal = source[-1]
    family = str(terminal.metadata.get("task_family", "other") or "other")
    source_id = trajectory_id_from_steps(source)
    stages = [stage_for_abstracted_action(step) for step in protocol]
    stage_sketch: list[str] = []
    for stage in stages:
        if not stage_sketch or stage_sketch[-1] != stage:
            stage_sketch.append(stage)

    anti_patterns = extract_anti_patterns(trajectories)
    label = capability.replace(".", " ").replace("_", " ").strip()

    skill.skill_name = f"Trajectory-derived {label or 'execution'} protocol"
    skill.description = (
        "Staged reusable protocol aligned across "
        f"{align_support} winning trajectory(ies) in {family}. "
        "Noise actions (Nothing happens / invalid / examine-only) were removed."
    )
    skill.precondition = checkable_precondition_text(
        family=family,
        capability=capability,
        stages=stage_sketch,
    )
    skill.action_protocol = protocol
    skill.applicable_atomic_ops = [AtomicOp.ACT]
    skill.expected_effect = checkable_expected_effect_text(
        capability=capability,
        family=family,
    )
    skill.suggested_role = "Executor"
    signal = str(skill.metadata.get("source_signal", "") or "")
    skill.target_failure_types = [signal] if signal else []
    skill.metadata["required_tools"] = []
    skill.metadata["protocol_source"] = "aligned_successful_trajectory_stages_v2"
    skill.metadata["protocol_source_trajectory_id"] = source_id
    skill.metadata["protocol_alignment_support"] = align_support
    skill.metadata["protocol_stages"] = stage_sketch
    skill.metadata["anti_patterns"] = anti_patterns
    skill.metadata["seed_template_discarded"] = True
    skill.metadata["protocol_alignment_ok"] = alignment_ok
    skill.metadata["protocol_structure_ok"] = structure_ok
    skill.metadata["protocol_structure_issues"] = structure_issues
    reject_reasons: list[str] = []
    if not alignment_ok:
        reject_reasons.append(
            f"need>={min_alignment_wins} aligned trajectories "
            f"(wins or grounded progress) for alignment, got {align_support}"
        )
    reject_reasons.extend(structure_issues)
    if reject_reasons:
        skill.metadata["protocol_quality_reject"] = "; ".join(reject_reasons)
    else:
        skill.metadata.pop("protocol_quality_reject", None)

    # Build executable step schema from the aligned winning trajectory.
    # Use ``source`` directly (the winning AtomicStep list); a prior refactor
    # referenced an undefined ``indexed`` map and silently skipped grounding.
    try:
        from sage_mas.executable_protocol import ensure_executable_protocol

        trajectory_records = [
            {
                "action": getattr(step, "action", "") or "",
                "observation": getattr(step, "observation", "") or "",
            }
            for step in source
            if getattr(step, "action", None)
            and getattr(step, "atomic_op", "") == "Act"
            and is_productive_environment_step(step)
        ]
        ensure_executable_protocol(
            skill,
            trajectory_steps=trajectory_records,
            force=True,
            max_steps=max(max_protocol_steps, 16),
        )
        skill.metadata["executable_protocol_includes_find_prefix"] = any(
            stage_for_abstracted_action(
                _abstract_environment_action(str(record.get("action") or ""))
            )
            == "find"
            for record in trajectory_records
        )
    except Exception as exc:
        skill.metadata["executable_protocol_error"] = str(exc)
    return skill
