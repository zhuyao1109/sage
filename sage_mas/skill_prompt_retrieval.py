"""Prompt-based skill retrieval for the ALFWorld Executor.

The Executor model itself decides which learned skill protocols (if any)
apply to the current step. Each step it sees a compact catalog — skill name,
learned precondition, learned expected effect — and replies with the names
to mount. No rule-based precondition matching is involved: selection is the
model's own decision, and the raw reply is logged per step.
"""

from __future__ import annotations

import re

from sage_mas.schemas import Skill

_SKILLS_TAG_RE = re.compile(r"<skills>(.*?)</skills>", re.IGNORECASE | re.DOTALL)
_NONE_REPLIES = {"", "none", "no skill", "no skills", "n/a"}


def build_skill_retrieval_prompt(
    observation: str,
    candidates: list[Skill],
    max_skills: int,
    *,
    task: str | None = None,
) -> str:
    """Catalog prompt: names + learned precondition/effect only."""
    lines = [
        "You are deciding which learned skill protocols (if any) apply to "
        "the current step of an interactive household task.",
        "",
    ]
    if task:
        lines.append(f"Task: {task}")
        lines.append("")
    lines.extend(
        [
            "Current situation:",
            observation.strip(),
            "",
            "Learned skill protocols:",
        ]
    )
    for index, skill in enumerate(candidates, start=1):
        precondition = " ".join(str(skill.precondition or "").split())
        effect = " ".join(str(skill.expected_effect or "").split())
        lines.append(f"{index}. {skill.skill_name}")
        lines.append(f"   When to use: {precondition or 'Not specified.'}")
        lines.append(f"   Expected effect: {effect or 'Not specified.'}")
    lines.extend(
        [
            "",
            f"Reply with the exact names of up to {max_skills} protocol(s) "
            "that are useful for deciding the next action, or \"none\" if no "
            "protocol applies.",
            "Format: <skills>name1, name2</skills> or <skills>none</skills>",
            "Use only names from the list above. Do not explain your choice.",
        ]
    )
    return "\n".join(lines)


def parse_skill_selection(
    content: str,
    candidates: list[Skill],
    max_skills: int,
) -> list[Skill]:
    """Parse the Executor's <skills> reply back into Skill objects."""
    if max_skills <= 0 or not candidates:
        return []
    by_name = {skill.skill_name.strip().lower(): skill for skill in candidates}
    text = str(content or "")
    match = _SKILLS_TAG_RE.search(text)
    if match:
        body = match.group(1).strip()
        if body.lower() in _NONE_REPLIES:
            return []
        selected: list[Skill] = []
        for part in re.split(r"[,;\n]+", body):
            name = part.strip().strip("\"'`.").lower()
            skill = by_name.get(name)
            if skill is not None and skill not in selected:
                selected.append(skill)
        return selected[:max_skills]
    # Lenient fallback when the tag is missing: exact name mentions.
    lowered = text.strip().lower()
    if lowered in _NONE_REPLIES:
        return []
    return [
        skill
        for skill in candidates
        if skill.skill_name.strip().lower() in lowered
    ][:max_skills]
