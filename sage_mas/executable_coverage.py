"""Executable coverage ρ_mini(C) for organization-gap decisions."""

from __future__ import annotations

from typing import Any, Sequence

from sage_mas.schemas import Skill
from sage_mas.skill_bank_roles import (
    ROLE_EXECUTABLE,
    skill_bank_role,
)
from sage_mas.skill_injection_policy import skill_has_positive_mu
from sage_mas.skill_quality import annotate_skill_identity


def skill_is_student_executable(
    skill: Skill,
    *,
    min_marginal_utility: float = 0.0,
) -> bool:
    """True when a skill is in the Executable bank with positive student MU."""
    annotate_skill_identity(skill)
    if skill_bank_role(skill) != ROLE_EXECUTABLE:
        return False
    md = skill.metadata or {}
    if md.get("inject_ready") is False or md.get("mu_rejected") is True:
        return False
    return skill_has_positive_mu(skill, min_marginal_utility=min_marginal_utility)


def executable_coverage_for_cluster(
    cluster_skills: Sequence[Skill],
    *,
    min_marginal_utility: float = 0.0,
) -> dict[str, Any]:
    """ρ_mini(C) = |{s in C : executable & ΔJ>θ}| / |C|."""
    skills = list(cluster_skills)
    n = len(skills)
    executable = [
        skill
        for skill in skills
        if skill_is_student_executable(
            skill, min_marginal_utility=min_marginal_utility
        )
    ]
    rho = (len(executable) / n) if n else 0.0
    return {
        "n_cluster": n,
        "n_executable": len(executable),
        "rho_mini": float(rho),
        "executable_skill_names": [skill.skill_name for skill in executable],
    }


def cluster_executable_coverage(
    skills: Sequence[Skill],
    *,
    cluster_by: str = "capability",
    min_marginal_utility: float = 0.0,
) -> dict[str, dict[str, Any]]:
    from sage_mas.organization import cluster_key_for_skill

    buckets: dict[str, list[Skill]] = {}
    for skill in skills:
        key = cluster_key_for_skill(skill, cluster_by=cluster_by)
        buckets.setdefault(key, []).append(skill)
    return {
        key: executable_coverage_for_cluster(
            group, min_marginal_utility=min_marginal_utility
        )
        for key, group in buckets.items()
    }


def cluster_ready_for_agent_nomination(
    cluster_skills: Sequence[Skill],
    *,
    min_rho: float = 0.5,
    min_marginal_utility: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    """Whether a cluster has enough executable coverage to nominate an agent."""
    stats = executable_coverage_for_cluster(
        cluster_skills,
        min_marginal_utility=min_marginal_utility,
    )
    ready = (
        stats["n_cluster"] > 0
        and float(stats["rho_mini"]) >= float(min_rho)
        and int(stats["n_executable"]) >= 1
    )
    stats["ready"] = bool(ready)
    stats["min_rho"] = float(min_rho)
    return bool(ready), stats
