"""Soft skill injection: short SOP reference, no step-k cursor.

Modes:
- soft (option C): full soft SOP every step
- sparse_soft (option D): full soft SOP only at start/core/stall
- hybrid_soft (C+D): short cue every step + full soft SOP at D gates
"""

from __future__ import annotations

import re
from typing import Any

from sage_mas.executable_protocol import ensure_executable_protocol
from sage_mas.schemas import Skill
from sage_mas.trajectory.abstraction import stage_for_abstracted_action


def _history_actions(history_steps: list[Any] | None) -> list[str]:
    actions: list[str] = []
    for step in history_steps or []:
        if isinstance(step, dict):
            action = str(step.get("action") or "").strip()
        else:
            action = str(getattr(step, "action", "") or "").strip()
        if action:
            actions.append(action)
    return actions


def _protocol_lines(skill: Skill, *, max_steps: int | None = None) -> list[str]:
    steps = ensure_executable_protocol(skill)
    if steps:
        lines = [
            str(step.action_template).strip()
            for step in steps
            if str(step.action_template or "").strip()
        ]
    else:
        lines = [
            str(item).strip()
            for item in (skill.action_protocol or [])
            if str(item).strip()
        ]
    return lines if max_steps is None else lines[: max(1, int(max_steps))]


def _search_prior_names(skill: Skill, *, max_sources: int = 2) -> str:
    prior = (skill.metadata or {}).get("search_prior") or {}
    sources = prior.get("sources") or []
    names = [
        str(entry.get("source") or "").strip()
        for entry in sources[: max(1, int(max_sources))]
    ]
    return ", ".join(name for name in names if name)


def _search_prior_line(skill: Skill) -> str:
    prior = (skill.metadata or {}).get("search_prior") or {}
    sources = prior.get("sources") or []
    episodes = int(prior.get("episodes") or 0)
    if not sources or episodes <= 0:
        return ""
    parts = ", ".join(
        f"{entry.get('source')} ×{entry.get('count')}" for entry in sources
    )
    return (
        f"- Search prior (from {episodes} winning episodes): target usually "
        f"found at {parts} — check these first, then expand the search."
    )


_MECHANICAL_ANTI_PATTERN = re.compile(r"^avoid repeating .+ after no-op", re.IGNORECASE)


def _anti_pattern_lines(skill: Skill, *, max_patterns: int = 2) -> list[str]:
    """Semantic anti-patterns only.

    The mechanical "avoid repeating `x` after no-op" entries come from
    heuristic loop detection and add prompt noise without new information;
    LLM-distilled anti-patterns (container confusion, order violations,
    re-grab mistakes) are the ones worth spending tokens on.
    """
    patterns = (skill.metadata or {}).get("anti_patterns") or []
    semantic = [
        str(p).strip()
        for p in patterns
        if str(p).strip() and not _MECHANICAL_ANTI_PATTERN.match(str(p).strip())
    ]
    return [f"- Avoid: {p}" for p in semantic[: max(1, int(max_patterns))]]


def render_soft_skill_block(
    skills: list[Skill],
    *,
    max_protocol_steps: int | None = None,
) -> str:
    """Soft SOP only: reference handbook, no ``Now Step k/N`` cursor."""
    if not skills:
        return ""
    blocks: list[str] = []
    for skill in skills[:1]:
        lines = _protocol_lines(skill, max_steps=max_protocol_steps)
        if not lines:
            continue
        numbered = "\n".join(f"  {i}. {step}" for i, step in enumerate(lines, 1))
        prior_line = _search_prior_line(skill)
        extra = ([prior_line] if prior_line else []) + _anti_pattern_lines(skill)
        blocks.append(
            "Skill reference (soft; observation + admissible actions win on "
            "conflict; do not invent entities):\n"
            f"- {skill.skill_name}\n"
            f"{numbered}"
            + ("\n" + "\n".join(extra) if extra else "")
        )
    return "\n".join(blocks)


def render_soft_skill_brief(
    skills: list[Skill],
    *,
    max_protocol_steps: int = 6,
) -> str:
    """One-line soft cue for every-step (C-lite) injection."""
    if not skills:
        return ""
    skill = skills[0]
    lines = _protocol_lines(skill, max_steps=max_protocol_steps)
    if not lines:
        return ""
    chain = " → ".join(lines)
    prior_hint = _search_prior_names(skill)
    suffix = f"; likely sources: {prior_hint}" if prior_hint else ""
    return (
        "Skill cue (soft; observation + admissible actions win on conflict): "
        f"{skill.skill_name}: {chain}{suffix}"
    )


def _recent_stall(history_steps: list[Any] | None, *, window: int = 3) -> bool:
    actions = _history_actions(history_steps)
    if len(actions) < window:
        return False
    recent = actions[-window:]
    if len({action.lower() for action in recent}) == 1:
        return True
    invalid = 0
    for step in (history_steps or [])[-window:]:
        if isinstance(step, dict) and step.get("is_action_valid") is False:
            invalid += 1
    return invalid >= max(2, window - 1)


def should_attach_sparse_soft(
    *,
    history_steps: list[Any] | None,
    reinject_every: int = 0,
) -> tuple[bool, str]:
    """Whether to attach the full soft SOP this step (D gates).

    Default: start / core-stage / stall only. Optional heartbeat if
    ``reinject_every > 0``.
    """
    actions = _history_actions(history_steps)
    if not actions:
        return True, "episode_start"

    stages = [
        stage
        for stage in (stage_for_abstracted_action(action) for action in actions)
        if stage != "other"
    ]
    core = {"pickup", "transform", "place"}
    if stages and stages[-1] in core and (
        len(stages) == 1 or stages[-2] != stages[-1]
    ):
        return True, "core_stage"

    if _recent_stall(history_steps):
        return True, "stall"

    every = int(reinject_every or 0)
    if every > 0 and len(actions) % every == 0:
        return True, "heartbeat"
    return False, "skip"


def render_hybrid_soft_block(
    skills: list[Skill],
    *,
    history_steps: list[Any] | None,
    reinject_every: int = 0,
) -> str:
    """C+D: brief cue every step; full soft SOP only at D gates."""
    parts: list[str] = []
    brief = render_soft_skill_brief(skills)
    if brief:
        parts.append(brief)
    attach, _reason = should_attach_sparse_soft(
        history_steps=history_steps,
        reinject_every=reinject_every,
    )
    if attach:
        full = render_soft_skill_block(skills)
        if full:
            parts.append(full)
    return "\n\n".join(parts)
