"""Student-aware skill transfer: adapt, contrastive patches, MU gates.

Teacher-success protocols are raw material. A skill is injectable / org-eligible
only after positive paired MU on the *student* actor. This module:
  1) rewrites drafts into shorter, conditional student-friendly protocols;
  2) distills contrastive patches from (teacher win ∩ student fail) pairs;
  3) revises skills from student-with-skill failure trajectories;
  4) applies a shared inject_ready / add_agent_ready gate from MU results.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

from sage_mas.schemas import Skill, SkillStatus
from sage_mas.skill_injection_policy import (
    skill_has_positive_mu,
    skill_marginal_utility_value,
)
from sage_mas.skill_quality import annotate_skill_identity
from sage_mas.trajectory.abstraction import (
    abstract_environment_action,
    stage_for_abstracted_action,
)


def skill_inject_ready_flag(skill: Skill) -> bool | None:
    """Return explicit inject_ready metadata, or None if unset."""
    md = skill.metadata or {}
    if "inject_ready" in md:
        return bool(md.get("inject_ready"))
    if md.get("mu_rejected") is True:
        return False
    if md.get("mu_promoted") is True:
        return True
    return None


def skill_is_transfer_ready(
    skill: Skill,
    *,
    min_marginal_utility: float = 0.0,
    require_explicit_inject_ready: bool = False,
) -> bool:
    """Whether a skill may be injected or drive ADD_AGENT after student MU."""
    annotate_skill_identity(skill)
    md = skill.metadata or {}
    if md.get("not_for_injection"):
        return False
    flag = skill_inject_ready_flag(skill)
    if flag is False:
        return False
    if require_explicit_inject_ready and flag is not True:
        return False
    if flag is True:
        return skill_has_positive_mu(
            skill, min_marginal_utility=min_marginal_utility
        ) or float(skill_marginal_utility_value(skill) or 0.0) > float(
            min_marginal_utility
        )
    return skill_has_positive_mu(
        skill, min_marginal_utility=min_marginal_utility
    )


def apply_student_mu_gate(
    skill: Skill,
    *,
    delta_sr: float,
    min_delta: float = 0.0,
    local_effect_delta: float = 0.0,
    promote_on_accept: bool = True,
) -> Skill:
    """Mark inject_ready / add_agent_ready from paired student MU.

    Non-negative ΔSR (``>= min_delta``) is enough to verify / ADD / inject.
    Only clearly harmful skills (negative ΔSR and no local-effect gain) stay out.
    """
    clone = deepcopy(skill)
    md = dict(clone.metadata or {})
    d_sr = float(delta_sr)
    d_eff = float(local_effect_delta)
    threshold = float(min_delta)
    accepted = d_sr >= threshold or d_eff > threshold
    clone.marginal_utility = d_sr
    md["marginal_utility"] = d_sr
    md["inject_ready"] = bool(accepted)
    md["add_agent_ready"] = bool(accepted)
    md["executor_optional_injection"] = bool(accepted)
    md["mu_promoted"] = bool(accepted)
    md["mu_rejected"] = not bool(accepted)
    # Negative paired MU blocks bank promotion / ADD / injection.
    if not accepted:
        md["not_for_add_agent"] = True
        md["not_for_injection"] = True
        md["skill_bank_role"] = "compiled_candidate"
        if clone.status == SkillStatus.VERIFIED:
            clone.status = SkillStatus.PROVISIONAL
    elif promote_on_accept:
        if clone.status not in {SkillStatus.REJECTED, SkillStatus.RETIRED}:
            clone.status = SkillStatus.VERIFIED
            md["credit_promoted_pending_org"] = True
        md["skill_bank_role"] = "executable"
        md["not_for_injection"] = False
        md.pop("not_for_add_agent", None)
    clone.metadata = md
    return clone


def _record_actions(record: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    for step in record.get("steps") or []:
        if isinstance(step, dict):
            action = str(step.get("action") or "").strip()
        else:
            action = str(step or "").strip()
        if action:
            actions.append(action)
    return actions


def _abstract_protocol(actions: Sequence[str]) -> list[str]:
    protocol: list[str] = []
    for action in actions:
        abstracted = abstract_environment_action(action)
        stage = stage_for_abstracted_action(abstracted)
        if stage == "other":
            continue
        if abstracted in {"look", "inventory", "examine <entity>"}:
            continue
        if protocol and protocol[-1] == abstracted:
            continue
        protocol.append(abstracted)
    return protocol


def _gamefile_key(record: dict[str, Any]) -> str:
    return str(
        record.get("gamefile")
        or record.get("task_id")
        or record.get("task")
        or ""
    ).strip()


def pair_teacher_wins_student_fails(
    teacher_records: Sequence[dict[str, Any]],
    student_records: Sequence[dict[str, Any]],
    *,
    task_family: str | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair same-gamefile trajectories: teacher won, student lost."""
    teacher_by: dict[str, dict[str, Any]] = {}
    for record in teacher_records:
        if not bool(record.get("won") or record.get("success")):
            continue
        family = str(record.get("task_family") or "")
        if task_family and family != task_family:
            continue
        key = _gamefile_key(record)
        if key:
            teacher_by[key] = record

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for record in student_records:
        if bool(record.get("won") or record.get("success")):
            continue
        family = str(record.get("task_family") or "")
        if task_family and family != task_family:
            continue
        key = _gamefile_key(record)
        if not key or key in seen or key not in teacher_by:
            continue
        seen.add(key)
        pairs.append((teacher_by[key], record))
    return pairs


def first_protocol_divergence(
    teacher_proto: Sequence[str],
    student_proto: Sequence[str],
) -> int:
    n = min(len(teacher_proto), len(student_proto))
    for index in range(n):
        if teacher_proto[index] != student_proto[index]:
            return index
    return n


def trajectory_gamefile_key(record: dict[str, Any]) -> str:
    return _gamefile_key(record)


def distill_contrastive_patch_skill(
    teacher_record: dict[str, Any],
    student_record: dict[str, Any],
    *,
    draft_skill: Skill | None = None,
) -> Skill | None:
    """Build one patch skill from a single teacher-win / student-fail pair."""
    teacher_proto = _abstract_protocol(_record_actions(teacher_record))
    student_proto = _abstract_protocol(_record_actions(student_record))
    if not teacher_proto:
        return None

    diverge = first_protocol_divergence(teacher_proto, student_proto)
    # Patch = teacher steps from divergence onward (trajectory-only).
    remainder = list(teacher_proto[diverge:])
    if not remainder:
        remainder = list(teacher_proto[-3:])

    # Conditional bullets: student-friendly "when/do" form, still from data.
    when_prefix = (
        student_proto[diverge - 1]
        if diverge > 0 and diverge - 1 < len(student_proto)
        else (student_proto[-1] if student_proto else "task start")
    )
    protocol: list[str] = []
    protocol.append(f"when last productive step ≈ `{when_prefix}`")
    protocol.append(
        "do not repeat failed student prefix; resume teacher continuation:"
    )
    for step in remainder[:6]:
        protocol.append(f"then {step}")

    family = str(
        teacher_record.get("task_family")
        or (draft_skill.applicable_task_families or [None])[0]
        or ""
    )
    capability = ""
    if draft_skill is not None:
        annotate_skill_identity(draft_skill)
        capability = str(draft_skill.capability_key or "")

    name_stem = (
        draft_skill.skill_name if draft_skill is not None else family or "task"
    )
    skill = Skill(
        skill_name=f"Contrastive patch: {name_stem}",
        description=(
            "Student-fail vs teacher-win patch. Learn only the divergence "
            "continuation, not the full teacher success skeleton."
        ),
        precondition=(
            f"Task family `{family}`. Use when the student trajectory has "
            f"diverged or stalled near `{when_prefix}`. Ground every "
            "`<placeholder>` from the current observation and admissible "
            "actions only."
        ),
        action_protocol=protocol,
        applicable_atomic_ops=["Act"],
        expected_effect=(
            "Recover onto the teacher continuation after the student "
            "divergence and complete the episode."
        ),
        suggested_role="Executor",
        target_failure_types=["student_teacher_divergence"],
        applicable_task_families=[family] if family else [],
        capability_key=capability or None,
        status=SkillStatus.PROVISIONAL,
        support_count=1,
        evidence_ids=[_gamefile_key(teacher_record)],
        metadata={
            "source_signal": "contrastive_student_patch",
            "primary_task_family": family,
            "protocol_source": "contrastive_teacher_win_student_fail_v1",
            "protocol_form_ok": True,
            "protocol_structure_ok": True,
            "protocol_alignment_ok": True,
            "protocol_alignment_support": 1,
            "contrastive": {
                "diverge_index": diverge,
                "teacher_protocol": teacher_proto,
                "student_protocol": student_proto,
                "teacher_gamefile": _gamefile_key(teacher_record),
                "student_gamefile": _gamefile_key(student_record),
            },
            "student_transfer": True,
            "inject_ready": False,
            "not_for_add_agent": True,
            "draft_skill_name": (
                draft_skill.skill_name if draft_skill is not None else None
            ),
        },
    )
    annotate_skill_identity(skill)
    return skill


def distill_contrastive_patches(
    teacher_records: Sequence[dict[str, Any]],
    student_records: Sequence[dict[str, Any]],
    *,
    draft_skills: Sequence[Skill] | None = None,
    max_patches_per_family: int = 3,
) -> list[Skill]:
    """Distill contrastive patch skills grouped by task family."""
    draft_by_family: dict[str, Skill] = {}
    for skill in draft_skills or []:
        annotate_skill_identity(skill)
        family = str(
            (skill.metadata or {}).get("primary_task_family")
            or (
                (skill.applicable_task_families or [None])[0]
                if skill.applicable_task_families
                else ""
            )
            or ""
        )
        if family and family not in draft_by_family:
            draft_by_family[family] = skill

    families = sorted(
        {
            str(record.get("task_family") or "")
            for record in list(teacher_records) + list(student_records)
            if str(record.get("task_family") or "").strip()
        }
    )
    patches: list[Skill] = []
    for family in families:
        pairs = pair_teacher_wins_student_fails(
            teacher_records,
            student_records,
            task_family=family,
        )
        count = 0
        for teacher, student in pairs:
            if count >= max(1, int(max_patches_per_family)):
                break
            patch = distill_contrastive_patch_skill(
                teacher,
                student,
                draft_skill=draft_by_family.get(family),
            )
            if patch is None:
                continue
            patches.append(patch)
            count += 1
    return patches


def adapt_skill_for_student(
    skill: Skill,
    *,
    student_failure_notes: Sequence[str] | None = None,
    max_protocol_steps: int = 8,
) -> Skill:
    """Rewrite a teacher draft into a shorter, conditional student-facing skill.

    Deterministic and data-preserving: truncates the draft protocol in order
    (no domain stage/token preferences) and optionally appends stall notes
    derived from student failure trajectories.
    """
    clone = deepcopy(skill)
    annotate_skill_identity(clone)
    protocol = [str(step).strip() for step in (clone.action_protocol or []) if str(step).strip()]
    if not protocol:
        clone.metadata = dict(clone.metadata or {})
        clone.metadata["student_aware_adapted"] = False
        clone.metadata["student_adapt_reason"] = "empty_protocol"
        return clone

    limit = max(1, int(max_protocol_steps))
    kept = protocol[:limit]

    adapted_protocol = [
        "follow only when the current task family matches this skill",
        "ground every <placeholder> from observation + admissible actions",
    ]
    for step in kept:
        adapted_protocol.append(f"then {step}")
    notes = [str(note).strip() for note in (student_failure_notes or []) if str(note).strip()]
    for note in notes[:3]:
        adapted_protocol.append(f"if stalled: {note}")

    family = str(
        (clone.metadata or {}).get("primary_task_family")
        or (
            (clone.applicable_task_families or [""])[0]
            if clone.applicable_task_families
            else ""
        )
    )
    clone.skill_name = f"Student-adapted: {clone.skill_name}"
    clone.description = (
        "Student-aware rewrite of a teacher draft protocol. Shorter "
        "conditional steps; placeholders must be grounded locally."
    )
    clone.precondition = (
        f"Task family `{family}`. Instantiate placeholders from the current "
        "observation and admissible actions only. Prefer the `then` steps "
        "in order; skip a step only when already satisfied by observation."
    )
    clone.action_protocol = adapted_protocol
    md = dict(clone.metadata or {})
    md["student_aware_adapted"] = True
    md["student_transfer"] = True
    md["teacher_draft_protocol"] = protocol
    md["protocol_source"] = "student_aware_adapt_v1"
    # Adaptation alone does not prove MU.
    md["inject_ready"] = False
    md["not_for_add_agent"] = True
    md["protocol_form_ok"] = len(kept) >= 1
    md["protocol_structure_ok"] = True
    clone.metadata = md
    clone.status = SkillStatus.PROVISIONAL
    return clone


def adapt_skills_for_student(
    skills: Sequence[Skill],
    *,
    student_failure_notes_by_family: dict[str, Sequence[str]] | None = None,
    max_protocol_steps: int = 8,
) -> list[Skill]:
    notes_map = student_failure_notes_by_family or {}
    adapted: list[Skill] = []
    for skill in skills:
        annotate_skill_identity(skill)
        family = str(
            (skill.metadata or {}).get("primary_task_family")
            or (
                (skill.applicable_task_families or [""])[0]
                if skill.applicable_task_families
                else ""
            )
        )
        adapted.append(
            adapt_skill_for_student(
                skill,
                student_failure_notes=list(notes_map.get(family) or []),
                max_protocol_steps=max_protocol_steps,
            )
        )
    return adapted


def summarize_student_failure_notes(
    student_records: Sequence[dict[str, Any]],
    *,
    task_family: str | None = None,
    max_notes: int = 3,
) -> list[str]:
    """Stall hints from failed student trajectories only (no domain templates)."""
    from collections import Counter

    last_actions: Counter[str] = Counter()
    bigram_ends: Counter[str] = Counter()
    for record in student_records:
        if bool(record.get("won") or record.get("success")):
            continue
        family = str(record.get("task_family") or "")
        if task_family and family != task_family:
            continue
        proto = _abstract_protocol(_record_actions(record))
        if not proto:
            continue
        last_actions[proto[-1]] += 1
        if len(proto) >= 2:
            bigram_ends[f"{proto[-2]} -> {proto[-1]}"] += 1

    notes: list[str] = []
    for action, count in last_actions.most_common(max(1, int(max_notes))):
        notes.append(
            f"observed frequent terminal step `{action}` (n={count})"
        )
        if len(notes) >= max(1, int(max_notes)):
            break
    if len(notes) < max(1, int(max_notes)):
        for pattern, count in bigram_ends.most_common(1):
            notes.append(
                f"observed frequent ending pattern `{pattern}` (n={count})"
            )
            break
    return notes[: max(1, int(max_notes))]


def revise_skill_from_student_failures(
    skill: Skill,
    failed_with_skill_records: Sequence[dict[str, Any]],
    *,
    max_extra_steps: int = 2,
) -> Skill:
    """Light revision: append stall-recovery cues from with-skill failures."""
    clone = deepcopy(skill)
    notes = summarize_student_failure_notes(
        failed_with_skill_records,
        task_family=str(
            (clone.metadata or {}).get("primary_task_family")
            or (
                (clone.applicable_task_families or [""])[0]
                if clone.applicable_task_families
                else ""
            )
        ),
        max_notes=max_extra_steps,
    )
    if not notes:
        md = dict(clone.metadata or {})
        md["skill_revised"] = False
        clone.metadata = md
        return clone

    protocol = list(clone.action_protocol or [])
    for note in notes:
        cue = f"if stalled: {note}"
        if cue not in protocol:
            protocol.append(cue)
    clone.action_protocol = protocol
    md = dict(clone.metadata or {})
    md["skill_revised"] = True
    md["skill_revise_source"] = "student_with_skill_failures_v1"
    md["inject_ready"] = False
    md["not_for_add_agent"] = True
    md["mu_rejected"] = True
    clone.metadata = md
    clone.status = SkillStatus.PROVISIONAL
    return clone


def transfer_policy_from_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    """Read ``sage.skill_transfer`` knobs."""
    mapping = mapping or {}
    return {
        "student_aware_adapt": bool(mapping.get("student_aware_adapt", False)),
        "contrastive_patches": bool(mapping.get("contrastive_patches", False)),
        "revise_on_mu_failure": bool(mapping.get("revise_on_mu_failure", False)),
        "max_protocol_steps": int(mapping.get("max_protocol_steps", 8)),
        "max_patches_per_family": int(mapping.get("max_patches_per_family", 3)),
        # Default off: credit-only online runs omit ``skill_transfer`` and
        # rely on credit promotion for org eligibility. Full dual-bank configs
        # set these keys explicitly (see sage_config.example.yaml).
        "require_inject_ready_for_org": bool(
            mapping.get("require_inject_ready_for_org", False)
        ),
        "require_positive_mu_for_injection": bool(
            mapping.get("require_positive_mu_for_injection", False)
        ),
        "min_executable_coverage_for_add_agent": float(
            mapping.get("min_executable_coverage_for_add_agent", 0.5)
        ),
    }
