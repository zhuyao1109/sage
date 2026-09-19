"""Evidence-grounding and identity rules for distilled skills.

These checks intentionally prefer rejecting an unsupported candidate over
letting a plausible-sounding protocol enter the active SkillBank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from sage_mas.schemas import AtomicStep, Skill
from sage_mas.trajectory.abstraction import (
    trajectory_action_steps,
    trajectory_id_from_steps,
)
from sage_mas.trajectory.protocol_canon import protocol_structure_issues


_CAPABILITY_BY_SIGNAL = {
    "missing_clean_operation": "transform.clean",
    "clean_operation_success": "transform.clean",
    "missing_heat_operation": "transform.heat",
    "heat_operation_success": "transform.heat",
    "missing_cool_operation": "transform.cool",
    "cool_operation_success": "transform.cool",
    "missing_light_operation": "inspect.with_light",
    "light_operation_success": "inspect.with_light",
    "incomplete_multi_object": "track.multiple_objects",
    "multi_object_success": "track.multiple_objects",
    "search_exhaustion": "search.coverage",
    "admissible_action_guard": "action.validity",
    "stagnation_recovery": "recovery.non_progress",
    "repetition_guard": "recovery.non_progress",
    "progress_checkpoint": "execution.progress",
    "failure_checkpoint": "execution.progress",
    "efficient_execution": "execution.efficiency",
    "task_completion": "execution.completion",
}

_PROTOCOL_MARKERS = {
    "transform.clean": ("clean",),
    "transform.heat": ("heat",),
    "transform.cool": ("cool",),
    # Accept both raw and abstracted lamp activation forms.
    "inspect.with_light": ("desklamp", "use desklamp"),
    "track.place": ("take", "put", "move"),
    "track.multiple_objects": ("take", "put", "move"),
}

_GENERIC_CAPABILITY_PREFIXES = (
    "action.",
    "execution.",
    "recovery.",
    "search.",
)

_GENERIC_EXPECTED_EFFECTS = {
    "",
    "improve task execution.",
    "reproduce the successful completion observed in the cited trajectory.",
}


def capability_key_for_signal(signal: str) -> str:
    normalized = str(signal or "").strip().lower()
    return _CAPABILITY_BY_SIGNAL.get(
        normalized,
        re.sub(r"_(success|operation)$", "", normalized) or "unspecified",
    )


def annotate_skill_identity(skill: Skill) -> Skill:
    family = str(skill.metadata.get("primary_task_family", "") or "").strip()
    families = set(skill.applicable_task_families)
    if family and family != "other":
        families.add(family)
    # Infer family from applicable set when primary is missing.
    if not family and len(families) == 1:
        family = next(iter(families))
        skill.metadata["primary_task_family"] = family

    # Identity is whatever the skill already carries (usually named by the
    # trajectory model). Do not fill it from a signal table or a family rule.
    existing = str(
        skill.capability_key or skill.metadata.get("capability_key") or ""
    ).strip()
    if existing:
        skill.capability_key = existing

    skill.applicable_task_families = sorted(families)
    skill.metadata["capability_key"] = skill.capability_key
    skill.metadata["applicable_task_families"] = list(
        skill.applicable_task_families
    )
    return skill


def _confirmed_environment_effect_tasks(skill: Skill) -> set[str]:
    """Task IDs with an observed action/observation environment effect.

    Prefer explicit ``confirmed_transitions``. Fall back to ``key_fragments``
    from enriched success distillation, which stores the same grounding but
    historically did not copy it into ``confirmed_transitions``.
    """
    tasks: set[str] = set()
    for record in list(skill.metadata.get("confirmed_transitions") or []):
        if not isinstance(record, dict):
            continue
        task_id = str(record.get("task_id", "") or "").strip()
        action = str(record.get("action", "") or "").strip()
        observation = str(record.get("observation", "") or "").strip()
        if task_id and action and observation:
            tasks.add(task_id)
    if tasks:
        return tasks
    for fragment in skill.key_fragments:
        task_id = str(fragment.trajectory_id or "").strip()
        action = str(fragment.action or "").strip()
        observation = str(fragment.observation or "").strip()
        if task_id and action and observation:
            tasks.add(task_id)
    return tasks


def generic_skill_contract_reasons(skill: Skill) -> list[str]:
    """Require environment-grounded contracts for broad procedural skills."""
    annotate_skill_identity(skill)
    if not skill.capability_key.startswith(_GENERIC_CAPABILITY_PREFIXES):
        return []
    reasons: list[str] = []
    anchor = str(skill.metadata.get("anchor_state", "") or "").strip()
    confirmed_tasks = _confirmed_environment_effect_tasks(skill)
    expected = " ".join(str(skill.expected_effect or "").lower().split())
    if not anchor:
        reasons.append("generic skill has no explicit learned anchor state")
    if expected in _GENERIC_EXPECTED_EFFECTS:
        reasons.append(
            "generic skill has no specific observable expected effect"
        )
    if len(confirmed_tasks) < 1:
        reasons.append(
            "generic skill requires an environment effect on at least one task"
        )
    return reasons


def trajectory_actions(steps: list[AtomicStep]) -> list[str]:
    return [
        str(step.action or "").strip().lower()
        for step in trajectory_action_steps(steps)
        if str(step.action or "").strip()
    ]


def trajectory_has_capability(
    steps: list[AtomicStep],
    capability_key: str,
) -> bool:
    markers = _PROTOCOL_MARKERS.get(capability_key)
    if not markers:
        return True
    return any(
        marker in action
        for action in trajectory_actions(steps)
        for marker in markers
    )


def protocol_has_capability(skill: Skill) -> bool:
    markers = _PROTOCOL_MARKERS.get(skill.capability_key)
    if not markers:
        return bool(skill.action_protocol)
    protocol = " ".join(skill.action_protocol).lower()
    return any(marker in protocol for marker in markers)


@dataclass(slots=True)
class SkillEvidenceValidation:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    success_evidence_ids: list[str] = field(default_factory=list)
    failure_evidence_ids: list[str] = field(default_factory=list)
    protocol_coverage_rate: float = 0.0
    protocol_grounding_rate: float = 0.0
    fragment_alignment_rate: float = 0.0

    def as_metadata(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "success_evidence_ids": list(self.success_evidence_ids),
            "failure_evidence_ids": list(self.failure_evidence_ids),
            "protocol_coverage_rate": self.protocol_coverage_rate,
            "protocol_grounding_rate": self.protocol_grounding_rate,
            "fragment_alignment_rate": self.fragment_alignment_rate,
        }


def validate_skill_evidence(
    skill: Skill,
    trajectories_by_id: dict[str, list[AtomicStep]],
) -> SkillEvidenceValidation:
    """Require a successful episode or observed positive environment effect."""
    annotate_skill_identity(skill)
    evidence_ids = sorted(set(skill.evidence_ids))
    missing_ids = [
        evidence_id
        for evidence_id in evidence_ids
        if evidence_id not in trajectories_by_id
    ]
    available = [
        trajectories_by_id[evidence_id]
        for evidence_id in evidence_ids
        if evidence_id in trajectories_by_id
    ]
    positive_trajectories = [
        steps for steps in available if _trajectory_has_positive_outcome(steps)
    ]
    success_ids = sorted(
        trajectory_id_from_steps(steps)
        for steps in positive_trajectories
    )
    failure_ids = sorted(
        trajectory_id_from_steps(steps)
        for steps in available
        if not bool(steps[-1].metadata.get("won", False))
    )

    fragment_ids = [fragment.trajectory_id for fragment in skill.key_fragments]
    aligned_fragments = sum(
        fragment_id in evidence_ids for fragment_id in fragment_ids
    )
    fragment_alignment = (
        aligned_fragments / len(fragment_ids) if fragment_ids else 1.0
    )

    covered_successes = sum(
        trajectory_has_capability(steps, skill.capability_key)
        for steps in positive_trajectories
    )
    coverage = (
        covered_successes / len(positive_trajectories)
        if positive_trajectories
        else 0.0
    )
    protocol_tokens = {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            " ".join(skill.action_protocol).lower(),
        )
        if len(token) >= 3
        and token
        not in {
            "the",
            "and",
            "then",
            "with",
            "from",
            "into",
            "before",
            "after",
            "object",
            "target",
            "action",
            "step",
        }
    }
    successful_action_tokens = {
        token
        for steps in positive_trajectories
        for action in trajectory_actions(steps)
        for token in re.findall(r"[a-z0-9]+", action)
        if len(token) >= 3
    }
    grounded_tokens = protocol_tokens & successful_action_tokens
    grounding_rate = (
        len(grounded_tokens) / len(protocol_tokens)
        if protocol_tokens
        else 0.0
    )

    reasons: list[str] = []
    if not evidence_ids:
        reasons.append("candidate has no trajectory evidence")
    if missing_ids:
        reasons.append(
            "evidence IDs missing from the distillation input: "
            + ", ".join(missing_ids[:3])
        )
    if not success_ids:
        reasons.append(
            "candidate requires a successful episode or a positive "
            "environment-effect transition"
        )
    if fragment_alignment < 1.0:
        reasons.append("one or more key fragments are outside evidence_ids")
    if not protocol_has_capability(skill):
        reasons.append("protocol does not express the claimed capability")
    structure_issues = list(skill.metadata.get("protocol_structure_issues") or [])
    if not structure_issues:
        structure_issues = protocol_structure_issues(
            list(skill.action_protocol or []),
            capability=str(skill.capability_key or ""),
        )
    if structure_issues:
        reasons.extend(structure_issues)
    if skill.metadata.get("protocol_alignment_ok") is False:
        reasons.append(
            str(
                skill.metadata.get("protocol_quality_reject")
                or "protocol failed multi-trajectory alignment gate"
            )
        )
    elif (
        str(skill.metadata.get("protocol_source", "")).startswith(
            "aligned_successful_trajectory_stages"
        )
        and int(skill.metadata.get("protocol_alignment_support") or 0) < 1
    ):
        reasons.append(
            "aligned protocol requires at least 1 grounded trajectory"
        )
    if (
        skill.capability_key in _PROTOCOL_MARKERS
        and positive_trajectories
        and coverage <= 0.0
    ):
        reasons.append(
            "no successful evidence trajectory executes the claimed capability"
        )
    if (
        skill.capability_key not in _PROTOCOL_MARKERS
        and not grounded_tokens
    ):
        reasons.append(
            "protocol has no lexical grounding in successful evidence actions"
        )

    return SkillEvidenceValidation(
        accepted=not reasons,
        reasons=reasons,
        success_evidence_ids=success_ids,
        failure_evidence_ids=failure_ids,
        protocol_coverage_rate=coverage,
        protocol_grounding_rate=grounding_rate,
        fragment_alignment_rate=fragment_alignment,
    )


def _trajectory_has_positive_outcome(steps: list[AtomicStep]) -> bool:
    if not steps:
        return False
    if bool(steps[-1].metadata.get("won", False)):
        return True
    for step in steps[:-1]:
        try:
            if float(step.metadata.get("goal_progress_delta", 0.0) or 0.0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def validate_environment_confirmed_skill(
    skill: Skill,
    trajectories_by_id: dict[str, list[AtomicStep]],
    *,
    min_independent_tasks: int = 3,
) -> SkillEvidenceValidation:
    """Validate local capability transitions without claiming episode wins."""
    annotate_skill_identity(skill)
    transitions = list(skill.metadata.get("confirmed_transitions") or [])
    evidence_ids = sorted(set(skill.evidence_ids))
    transitions_by_task: dict[str, list[dict[str, object]]] = {}
    for record in transitions:
        task_id = str(record.get("task_id", ""))
        if task_id:
            transitions_by_task.setdefault(task_id, []).append(record)
    confirmed_ids: list[str] = []
    reasons: list[str] = []
    for evidence_id in evidence_ids:
        steps = trajectories_by_id.get(evidence_id)
        task_transitions = transitions_by_task.get(evidence_id, [])
        if steps is None or not task_transitions:
            continue
        if any(
            str(transition.get("action", "")).strip().lower()
            in str(step.action or "").lower()
            and str(transition.get("observation", "")).strip().lower()
            in str(step.observation or "").lower()
            and step.metadata.get("is_action_valid", True) is not False
            for transition in task_transitions
            for step in steps
        ):
            confirmed_ids.append(evidence_id)
    if len(evidence_ids) < min_independent_tasks:
        reasons.append(
            f"requires {min_independent_tasks} independent task IDs, "
            f"found {len(evidence_ids)}"
        )
    if len(confirmed_ids) != len(evidence_ids):
        reasons.append(
            "one or more evidence tasks lack the cited environment-confirmed "
            "action/observation transition"
        )
    if not protocol_has_capability(skill):
        reasons.append("protocol does not express the claimed capability")
    return SkillEvidenceValidation(
        accepted=not reasons,
        reasons=reasons,
        success_evidence_ids=sorted(confirmed_ids),
        failure_evidence_ids=[],
        protocol_coverage_rate=(
            len(confirmed_ids) / len(evidence_ids) if evidence_ids else 0.0
        ),
        protocol_grounding_rate=1.0 if confirmed_ids else 0.0,
        fragment_alignment_rate=1.0,
    )
