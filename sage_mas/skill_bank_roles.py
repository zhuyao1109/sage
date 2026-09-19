"""Canonical vs Executable skill-bank roles for SAGE-MAS.

Canonical skills (often teacher-distilled) describe capability structure for
clustering and role nomination. Executable skills are student-compiled and
paired-MU verified; only they may be injected or drive ADD_AGENT.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

from sage_mas.schemas import Skill
from sage_mas.skill_quality import annotate_skill_identity

ROLE_CANONICAL = "canonical"
ROLE_EXECUTABLE = "executable"
ROLE_COMPILED_CANDIDATE = "compiled_candidate"


def skill_bank_role(skill: Skill) -> str:
    annotate_skill_identity(skill)
    md = skill.metadata or {}
    raw = str(md.get("skill_bank_role") or "").strip().lower()
    if raw in {ROLE_CANONICAL, ROLE_EXECUTABLE, ROLE_COMPILED_CANDIDATE}:
        return raw
    # Legacy teacher drafts without an explicit role stay canonical.
    if md.get("inject_ready") is True and md.get("mu_rejected") is not True:
        return ROLE_EXECUTABLE
    return ROLE_CANONICAL


def mark_canonical(skill: Skill, *, teacher_model: str | None = None) -> Skill:
    clone = deepcopy(skill)
    md = dict(clone.metadata or {})
    md["skill_bank_role"] = ROLE_CANONICAL
    md.setdefault("inject_ready", False)
    md.setdefault("not_for_add_agent", True)
    md.setdefault("not_for_injection", True)
    if teacher_model:
        md["canonical_teacher_model"] = str(teacher_model)
    clone.metadata = md
    return clone


def mark_compiled_candidate(
    skill: Skill,
    *,
    canonical_skill_id: str | None = None,
    compile_variant: str | None = None,
) -> Skill:
    clone = deepcopy(skill)
    md = dict(clone.metadata or {})
    md["skill_bank_role"] = ROLE_COMPILED_CANDIDATE
    md["inject_ready"] = False
    md["not_for_add_agent"] = True
    md["not_for_injection"] = True
    if canonical_skill_id:
        md["canonical_skill_id"] = str(canonical_skill_id)
    if compile_variant:
        md["compile_variant"] = str(compile_variant)
    clone.metadata = md
    return clone


def mark_executable(skill: Skill, *, delta_sr: float | None = None) -> Skill:
    clone = deepcopy(skill)
    md = dict(clone.metadata or {})
    md["skill_bank_role"] = ROLE_EXECUTABLE
    md["inject_ready"] = True
    md["not_for_injection"] = False
    md["mu_promoted"] = True
    md["mu_rejected"] = False
    if delta_sr is not None:
        md["marginal_utility"] = float(delta_sr)
        clone.marginal_utility = float(delta_sr)
    # ADD_AGENT still needs organization-gap checks; executable alone is not
    # sufficient, but it clears the "teacher-only" block.
    md.pop("not_for_add_agent", None)
    clone.metadata = md
    return clone


def filter_skills_by_role(
    skills: Sequence[Skill],
    roles: Iterable[str],
) -> list[Skill]:
    wanted = {str(role).strip().lower() for role in roles if str(role).strip()}
    return [skill for skill in skills if skill_bank_role(skill) in wanted]


def canonical_skills(skills: Sequence[Skill]) -> list[Skill]:
    return filter_skills_by_role(skills, {ROLE_CANONICAL})


def executable_skills(skills: Sequence[Skill]) -> list[Skill]:
    return filter_skills_by_role(skills, {ROLE_EXECUTABLE})


def promote_bank_to_canonical(
    skills: Sequence[Skill],
    *,
    teacher_model: str | None = None,
) -> list[Skill]:
    return [mark_canonical(skill, teacher_model=teacher_model) for skill in skills]


def bank_role_summary(skills: Sequence[Skill]) -> dict[str, Any]:
    counts = {
        ROLE_CANONICAL: 0,
        ROLE_EXECUTABLE: 0,
        ROLE_COMPILED_CANDIDATE: 0,
    }
    for skill in skills:
        role = skill_bank_role(skill)
        counts[role] = counts.get(role, 0) + 1
    return {"n": len(skills), "by_role": counts}
