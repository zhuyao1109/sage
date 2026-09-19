"""Policies for which distilled skills may be injected or drive org edits."""

from __future__ import annotations

from typing import Any, Iterable

from sage_mas.schemas import Skill
from sage_mas.skill_quality import annotate_skill_identity

# Prefix lists are not a default policy. Credit, marginal utility, and
# verification decide what is injected. Callers may still pass an explicit
# list; config files do not turn one on.
DEFAULT_BLOCKED_INJECTION_PREFIXES: tuple[str, ...] = ()
DEFAULT_INJECTION_ALLOWLIST_PREFIXES: tuple[str, ...] = ()


def injection_policy_from_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    """Return no prefix filter.

    ``block_injection_capability_prefixes`` and
    ``inject_capability_allowlist`` in config are ignored. Whether a skill
    is used is decided by credit, marginal utility, and verification.
    """
    del mapping
    return {
        "block_prefixes": (),
        "allow_prefixes": None,
    }


def skill_capability_key(skill: Skill) -> str:
    annotate_skill_identity(skill)
    return str(
        skill.capability_key
        or skill.metadata.get("capability_key")
        or ""
    ).strip().lower()


def skill_matches_prefixes(skill: Skill, prefixes: Iterable[str]) -> bool:
    key = skill_capability_key(skill)
    if not key:
        return False
    return any(key.startswith(prefix) for prefix in prefixes)


def skill_marginal_utility_value(skill: Skill) -> float | None:
    if skill.marginal_utility is not None:
        try:
            return float(skill.marginal_utility)
        except (TypeError, ValueError):
            pass
    raw = skill.metadata.get("marginal_utility")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def skill_has_positive_mu(
    skill: Skill,
    *,
    min_marginal_utility: float = 0.0,
) -> bool:
    """True when paired MU is present and not below the threshold.

    With the default threshold ``0.0``, non-negative ΔSR (including 0) passes.
    """
    mu = skill_marginal_utility_value(skill)
    return mu is not None and mu >= float(min_marginal_utility)


def skill_is_injectable(
    skill: Skill,
    *,
    block_prefixes: Iterable[str] = DEFAULT_BLOCKED_INJECTION_PREFIXES,
    allow_prefixes: Iterable[str] | None = None,
    require_positive_mu: bool = False,
    min_marginal_utility: float = 0.0,
    require_inject_ready: bool = False,
) -> bool:
    """Whether a skill may be prompt-injected or drive specialist creation."""
    key = skill_capability_key(skill)
    md = skill.metadata or {}
    if md.get("not_for_injection"):
        return False
    # Explicit student-MU rejection always blocks injection.
    if md.get("inject_ready") is False or md.get("mu_rejected") is True:
        return False
    # Prefix lists are optional caller filters, not a default policy.
    # An unnamed skill is not rejected for lacking a preset key.
    if key and any(key.startswith(prefix) for prefix in tuple(block_prefixes)):
        return False
    if allow_prefixes is not None:
        allow = tuple(allow_prefixes)
        if not key or not any(key.startswith(prefix) for prefix in allow):
            return False
    if require_inject_ready and md.get("inject_ready") is not True:
        return False
    if require_positive_mu and not skill_has_positive_mu(
        skill,
        min_marginal_utility=min_marginal_utility,
    ):
        return False
    return True


def filter_injectable_skills(
    skills: list[Skill],
    *,
    block_prefixes: Iterable[str] = DEFAULT_BLOCKED_INJECTION_PREFIXES,
    allow_prefixes: Iterable[str] | None = None,
    require_positive_mu: bool = False,
    min_marginal_utility: float = 0.0,
    require_inject_ready: bool = False,
) -> list[Skill]:
    return [
        skill
        for skill in skills
        if skill_is_injectable(
            skill,
            block_prefixes=block_prefixes,
            allow_prefixes=allow_prefixes,
            require_positive_mu=require_positive_mu,
            min_marginal_utility=min_marginal_utility,
            require_inject_ready=require_inject_ready,
        )
    ]


def distill_seed_is_excluded(
    skill: Skill,
    *,
    exclude_prefixes: Iterable[str] | None = None,
    exclude_signals: Iterable[str] | None = None,
) -> bool:
    """Drop low-value seeds before they enter the SkillBank path."""
    signal = str(skill.metadata.get("source_signal", "") or "").strip().lower()
    if exclude_signals and signal in {
        str(item).strip().lower() for item in exclude_signals if str(item).strip()
    }:
        return True
    if exclude_prefixes and skill_matches_prefixes(skill, exclude_prefixes):
        return True
    return False
