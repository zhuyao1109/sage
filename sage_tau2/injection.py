"""τ² skill injection: prompt formatting + multi-domain offer selection."""

from __future__ import annotations

import re
from collections.abc import Collection

from sage_tau2.credit import injectable_skills
from sage_tau2.schemas import Tau2Skill
from sage_tau2.skill_bank import Tau2SkillBank
from sage_tau2.task_context import domains_match, skill_matches_episode
from sage_tau2.tool_sides import strip_user_side_protocol_steps


def _ordered_protocol_lines(
    protocol: list[str],
) -> tuple[list[str], list[str]]:
    """Split protocol into agent-tool vs guide-user lines."""
    agent: list[str] = []
    guides: list[str] = []
    for step in protocol or []:
        text = str(step or "").strip()
        if not text:
            continue
        low = text.lower()
        if low.startswith("guide user:") or low.startswith("guide "):
            guides.append(text)
        elif low.startswith(("enumerate:", "probe:", "select:")):
            # Legacy prose — ignore in prompt.
            continue
        else:
            agent.append(text)
    return agent, guides


def format_skills_for_prompt_legacy(
    skills: list[Tau2Skill],
    *,
    max_chars: int = 3500,
    specialist: bool = False,
) -> str:
    """Original flat skill-patch formatter (kept for compatibility)."""
    if not skills:
        return ""
    header = (
        "Active skill patch:"
        if specialist
        else "Active skill patch (follow concrete steps; do not invent entities):"
    )
    lines = [header]
    seen: set[str] = set()
    for skill in skills:
        name = str(skill.skill_name or skill.skill_id or "skill")
        if name in seen:
            continue
        seen.add(name)
        cleaned = strip_user_side_protocol_steps(
            list(skill.action_protocol or []),
            domain=getattr(skill, "domain", None),
        )
        protocol = "; ".join(cleaned) or "(empty)"
        credit = skill.metadata.get("skill_credit") or {}
        score = credit.get("score", skill.metadata.get("utility"))
        lines.append(f"- {name}")
        if skill.precondition:
            lines.append(f"  Precondition: {skill.precondition}")
        lines.append(f"  Protocol: {protocol}")
        if skill.expected_effect:
            lines.append(f"  Expected effect: {skill.expected_effect}")
        if score is not None:
            lines.append(f"  status={skill.status.value} score={score}")
    lines.append(
        "When a skill matches the user goal, follow its protocol. "
        "Device/phone steps are never agent tools — guide the user to perform "
        "them. Obey domain policy; one tool call XOR one user message per turn."
    )
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n...(truncated)"
    return text


def render_skill_cards(skills: list[Tau2Skill], *, max_chars: int = 6000) -> tuple[str, list[Tau2Skill]]:
    """Pack complete ordered cards. Omitted cards are not recorded as injected."""
    from sage_tau2.contracts import contract_for_skill

    header = "Available learned skills (candidates, not mandatory actions):"
    footer = (
        "Check applicability using current evidence before adopting a skill. "
        "Preserve dependencies across tools and user actions. Bind identifiers from "
        "observed results; never guess. Verify outcomes before declaring completion; "
        "return unresolved work to Executor. Domain policy takes precedence."
    )
    cards: list[str] = []
    offered: list[Tau2Skill] = []
    seen: set[str] = set()
    used = len(header) + len(footer) + 2
    for skill in skills:
        if skill.skill_id in seen:
            continue
        seen.add(skill.skill_id)
        contract = contract_for_skill(skill)
        lines = [f"Skill {skill.skill_id}: {skill.skill_name}",
                 f"Use when: {contract['condition']}",
                 "Protocol (preserve this order, including user actions):"]
        if contract['observed_conditions']:
            lines.insert(2, 'Observed supporting conditions (check now; not proven causal): ' + '; '.join(contract['observed_conditions']))
        if contract['bindings']:
            lines.insert(2, "Bind from observed sources: " + "; ".join(
                f"{key} <- {' / '.join(sources)}" for key, sources in contract['bindings'].items()))
        protocol = strip_user_side_protocol_steps(skill.action_protocol, domain=skill.domain)
        lines.extend(f"  {i}. {step}" for i, step in enumerate(protocol, 1))
        lines.append("Verify: " + contract["verification"])
        lines.append("Stop/return: " + contract["stop_condition"])
        card = "\n".join(lines)
        if used + len(card) + 2 > max_chars:
            continue
        cards.append(card)
        offered.append(skill)
        used += len(card) + 2
    if not cards:
        return "", []
    return "\n\n".join([header, *cards, footer]), offered


def format_skills_for_prompt(
    skills: list[Tau2Skill], *, max_chars: int = 6000,
    specialist: bool = False, layered: bool = True,
) -> str:
    # Keep the call signature for older runners; all modes now preserve order.
    return render_skill_cards(skills, max_chars=max_chars)[0]


def append_skills_to_user_prompt(user_prompt: str, skills_block: str) -> str:
    """Join GiGPO window + skill patch like sage_mas ``_alfworld_user_prompt``."""
    base = str(user_prompt or "").rstrip()
    patch = str(skills_block or "").strip()
    if not patch:
        return base
    if not base:
        return patch
    return f"{base}\n\n{patch}"


def candidate_pool_for_domain(
    bank: Tau2SkillBank,
    *,
    domain: str,
    allow_provisional: bool,
    same_domain_only: bool,
    allowed_skill_ids: Collection[str] | None = None,
    task_id: str = "",
    episode_scope: str = "",
    task_text: str = "",
    require_scope_match: bool = True,
) -> list[Tau2Skill]:
    """Build the full episode-level candidate pool (no ``max_skills`` truncation).

    This is the τ² analogue of ALFWorld's episode-fixed candidate pool:
    domain-filtered + scope-matched, but **not** ranked or capped. The
    step-level matcher (:class:`sage_tau2.step_matcher.Tau2StepMatcher`)
    re-evaluates this pool every step to decide which skills are *active*.

    Returns all skills that pass domain + allowed-id + scope-match filters
    and are injectable (status + provisional gate). Ranking / dedupe / cap
    happens per-step in the matcher, not here.
    """
    from sage_tau2.credit import skill_allowed_for_inject

    active = bank.active()
    if same_domain_only:
        pool = [
            s
            for s in active
            if domains_match(getattr(s, "domain", ""), domain)
        ]
    else:
        pool = list(active)
    if allowed_skill_ids is not None:
        allowed = {str(x) for x in allowed_skill_ids}
        pool = [s for s in pool if str(s.skill_id) in allowed]

    if require_scope_match and (str(task_id or "").strip() or str(episode_scope or "").strip()):
        pool = [
            s
            for s in pool
            if skill_matches_episode(
                s,
                domain=domain,
                task_id=task_id,
                episode_scope=episode_scope,
                task_text=task_text,
            )
        ]

    # Keep only injectable-status skills, but do NOT rank/dedupe/cap here.
    from sage_tau2.credit import CreditPolicy

    policy = CreditPolicy()
    return [s for s in pool if skill_allowed_for_inject(s, allow_provisional=allow_provisional, policy=policy)]


def offer_skills_for_domain(
    bank: Tau2SkillBank,
    *,
    domain: str,
    max_skills: int,
    allow_provisional: bool,
    same_domain_only: bool,
    allowed_skill_ids: Collection[str] | None = None,
    task_id: str = "",
    episode_scope: str = "",
    task_text: str = "",
    require_scope_match: bool = True,
    dedupe_writes: bool = True,
) -> list[Tau2Skill]:
    """Select injectable skills for one domain collection (episode-level, fixed).

    .. deprecated::
        For step-level retrieval, use :func:`candidate_pool_for_domain` to
        build the pool, then :class:`sage_tau2.step_matcher.Tau2StepMatcher`
        to re-evaluate active skills per step. This function is kept for
        backward compatibility (specialists, offline replay, probes).

    When ``same_domain_only`` is True, never fall back to other domains'
    skills (retail protocols cannot help telecom episodes). An empty
    same-domain bank means inject nothing until that domain distills its own.

    ``allowed_skill_ids`` freezes the inject pool (typically skill ids present at
    segment start) so a segment only reuses **prior-segment** experience, not
    skills distilled earlier in the same segment from another domain.

    When ``require_scope_match`` is True and ``task_id`` / ``episode_scope`` is
    provided, only skills that :func:`skill_matches_episode` accepts are offered.
    Domain-level previews (no task_id) keep domain filtering only.

    Ranking uses episode protocol-coverage when ``task_id`` is set, with optional
    primary-write dedupe so duplicate roaming cards cannot crowd out refuel/pay.
    """
    pool = candidate_pool_for_domain(
        bank,
        domain=domain,
        allow_provisional=allow_provisional,
        same_domain_only=same_domain_only,
        allowed_skill_ids=allowed_skill_ids,
        task_id=task_id,
        episode_scope=episode_scope,
        task_text=task_text,
        require_scope_match=require_scope_match,
    )

    return injectable_skills(
        pool,
        max_skills=max_skills,
        allow_provisional=allow_provisional,
        task_id=task_id,
        dedupe_writes=dedupe_writes,
    )


# ---------------------------------------------------------------------------
# Self-retrieval: skill catalog for LLM-driven on-demand skill checkout.
#
# Instead of code-level deterministic gating (ALFWorld-style precondition
# matcher), the catalog approach puts a compact skill directory into the
# prompt. The LLM reads the directory, decides which skill's precondition
# matches the current dialogue state, and requests the full protocol by
# emitting ``<use_skill>skill_name</use_skill>``. The code intercepts the
# tag and injects the full protocol in the next turn.
# ---------------------------------------------------------------------------

_USE_SKILL_RE = re.compile(
    r"<use_skill>\s*([^<\s]+(?:\s[^<\s]+)*)\s*</use_skill>",
    re.IGNORECASE,
)


def parse_use_skill_tags(content: str | None) -> list[str]:
    """Extract skill names from ``<use_skill>...</use_skill>`` tags in content."""
    if not content:
        return []
    return [m.group(1).strip() for m in _USE_SKILL_RE.finditer(content) if m.group(1).strip()]


def strip_use_skill_tags(content: str | None) -> str:
    """Remove ``<use_skill>...</use_skill>`` tags from content (customer-visible)."""
    if not content:
        return ""
    return _USE_SKILL_RE.sub("", content)


def format_skill_catalog(
    skills: list[Tau2Skill],
    *,
    max_chars: int = 2000,
) -> str:
    """Compact skill directory for LLM-driven self-retrieval.

    Each entry: ``name — precondition | one-line summary``.
    No full protocol (that is injected only after the LLM requests it via
    ``<use_skill>``).

    Returns empty string when there are no skills.
    """
    if not skills:
        return ""
    lines = [
        "Available skills (not yet activated). To retrieve a skill's full "
        "protocol, include <use_skill>skill_name</use_skill> in your reply:"
    ]
    for skill in skills:
        name = str(skill.skill_name or skill.skill_id or "skill")
        pre = str(skill.precondition or "").strip()
        # One-line summary: prefer description, fall back to expected_effect.
        summary = str(skill.description or skill.expected_effect or "").strip()
        if len(summary) > 120:
            summary = summary[:117] + "..."
        entry = f"- {name}"
        if pre:
            entry += f" — {pre}"
        if summary and summary != pre:
            entry += f" | {summary}"
        lines.append(entry)
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n...(truncated)"
    return text
