"""Trajectory-derived exclusive context for newly added specialists.

Builds richer agent briefs from teacher-win demos and informative student
failures. No hand-written ALFWorld expert checklists or family hard rules.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from sage_mas.idle_trajectory import (
    abstract_productive_protocol,
    filter_informative_failures,
)
from sage_mas.schemas import Skill
from sage_mas.skill_quality import annotate_skill_identity


def _skill_families(skill: Skill) -> set[str]:
    annotate_skill_identity(skill)
    values = set()
    primary = str((skill.metadata or {}).get("primary_task_family") or "").strip()
    if primary:
        values.add(primary)
    for family in skill.applicable_task_families or []:
        text = str(family).strip()
        if text:
            values.add(text)
    for family in (skill.metadata or {}).get("task_families") or []:
        text = str(family).strip()
        if text:
            values.add(text)
    return values


def _record_actions(record: dict[str, Any], *, max_actions: int = 12) -> list[str]:
    actions: list[str] = []
    for step in record.get("steps") or []:
        if isinstance(step, dict):
            action = str(step.get("action") or "").strip()
        else:
            action = str(step or "").strip()
        if action:
            actions.append(action)
        if len(actions) >= max_actions:
            break
    return actions


def _gamefile_key(record: dict[str, Any]) -> str:
    for key in ("gamefile", "task_id", "game_file", "id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def compress_success_demo(
    record: dict[str, Any],
    *,
    max_actions: int = 12,
) -> dict[str, Any] | None:
    if not bool(record.get("won") or record.get("success")):
        return None
    actions = _record_actions(record, max_actions=max_actions)
    if not actions:
        return None
    return {
        "task_family": str(record.get("task_family") or ""),
        "gamefile": _gamefile_key(record),
        "won": True,
        "actions": actions,
        "abstract_protocol": abstract_productive_protocol(record)[:max_actions],
    }


def select_teacher_demos_for_skills(
    skills: Sequence[Skill],
    teacher_records: Sequence[dict[str, Any]],
    *,
    max_demos_per_skill: int = 2,
    max_actions: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    """Pick short teacher-win demos per skill (family / evidence overlap)."""
    by_skill: dict[str, list[dict[str, Any]]] = {}
    wins = [
        record
        for record in teacher_records
        if bool(record.get("won") or record.get("success"))
    ]
    for skill in skills:
        families = _skill_families(skill)
        evidence = {
            str(value).strip()
            for value in (skill.evidence_ids or [])
            if str(value).strip()
        }
        ranked: list[tuple[int, dict[str, Any]]] = []
        for record in wins:
            family = str(record.get("task_family") or "").strip()
            if families and family not in families:
                continue
            demo = compress_success_demo(record, max_actions=max_actions)
            if demo is None:
                continue
            key = _gamefile_key(record)
            score = 0
            if key and key in evidence:
                score += 5
            if family and family in families:
                score += 2
            score += min(3, len(demo["actions"]) // 3)
            ranked.append((score, demo))
        ranked.sort(key=lambda item: (-item[0], item[1].get("gamefile") or ""))
        demos: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _score, demo in ranked:
            key = str(demo.get("gamefile") or json_fallback_key(demo))
            if key in seen:
                continue
            seen.add(key)
            demos.append(demo)
            if len(demos) >= max(1, int(max_demos_per_skill)):
                break
        by_skill[skill.skill_name] = demos
    return by_skill


def json_fallback_key(demo: dict[str, Any]) -> str:
    actions = demo.get("actions") or []
    return "|".join(str(a) for a in actions[:4])


def select_failure_cues_for_skills(
    skills: Sequence[Skill],
    student_records: Sequence[dict[str, Any]],
    *,
    max_cues_per_skill: int = 2,
) -> dict[str, list[str]]:
    """Data-driven stall cues from informative student fails (no expert rules)."""
    from collections import Counter

    informative = filter_informative_failures(student_records)
    by_skill: dict[str, list[str]] = {}
    for skill in skills:
        families = _skill_families(skill)
        last_actions: Counter[str] = Counter()
        for record in informative:
            family = str(record.get("task_family") or "").strip()
            if families and family not in families:
                continue
            proto = abstract_productive_protocol(record)
            if proto:
                last_actions[proto[-1]] += 1
        cues: list[str] = []
        for action, count in last_actions.most_common(max(1, int(max_cues_per_skill))):
            cues.append(
                f"observed student stall near `{action}` (n={count}); "
                "do not repeat that prefix—continue from the next demo step "
                "grounded in the current observation"
            )
        by_skill[skill.skill_name] = cues
    return by_skill


def attach_trajectory_context_to_skills(
    skills: Sequence[Skill],
    *,
    teacher_records: Sequence[dict[str, Any]] | None = None,
    student_records: Sequence[dict[str, Any]] | None = None,
    max_demos_per_skill: int = 2,
    max_actions: int = 12,
    max_cues_per_skill: int = 2,
) -> list[Skill]:
    """Write specialist_demos / failure cues onto skill metadata (in-place copy)."""
    teacher_records = list(teacher_records or [])
    student_records = list(student_records or [])
    demos = select_teacher_demos_for_skills(
        skills,
        teacher_records,
        max_demos_per_skill=max_demos_per_skill,
        max_actions=max_actions,
    )
    cues = select_failure_cues_for_skills(
        skills,
        student_records,
        max_cues_per_skill=max_cues_per_skill,
    )
    out: list[Skill] = []
    for skill in skills:
        clone = deepcopy(skill)
        md = dict(clone.metadata or {})
        md["specialist_demos"] = list(demos.get(skill.skill_name) or [])
        md["specialist_failure_cues"] = list(cues.get(skill.skill_name) or [])
        md["specialist_context_source"] = "trajectory_demos"
        clone.metadata = md
        out.append(clone)
    return out


def _format_demo_block(demo: dict[str, Any], index: int) -> str:
    actions = [str(a).strip() for a in (demo.get("actions") or []) if str(a).strip()]
    family = str(demo.get("task_family") or "").strip() or "task"
    if not actions:
        return ""
    seq = " -> ".join(actions)
    return f"Demo {index} ({family} win): {seq}"


def build_specialist_acting_brief(
    *,
    agent_name: str,
    skills: Sequence[Skill],
    executor_name: str = "Executor",
    dispatch_only: bool = True,
    teacher_records: Sequence[dict[str, Any]] | None = None,
    student_records: Sequence[dict[str, Any]] | None = None,
    max_demos: int = 2,
    max_actions: int = 10,
    max_cues: int = 2,
    max_chars: int = 2200,
) -> dict[str, Any]:
    """Compose a detailed, trajectory-grounded specialist role brief."""
    skill_list = list(skills)
    # Prefer demos already attached to skills; otherwise select from records.
    attached_demos: list[dict[str, Any]] = []
    attached_cues: list[str] = []
    for skill in skill_list:
        md = skill.metadata or {}
        for demo in md.get("specialist_demos") or []:
            if isinstance(demo, dict):
                attached_demos.append(demo)
        for cue in md.get("specialist_failure_cues") or []:
            text = str(cue).strip()
            if text:
                attached_cues.append(text)

    if not attached_demos and teacher_records:
        by_skill = select_teacher_demos_for_skills(
            skill_list,
            teacher_records,
            max_demos_per_skill=max_demos,
            max_actions=max_actions,
        )
        for skill in skill_list:
            attached_demos.extend(by_skill.get(skill.skill_name) or [])
    if not attached_cues and student_records:
        by_skill = select_failure_cues_for_skills(
            skill_list,
            student_records,
            max_cues_per_skill=max_cues,
        )
        for skill in skill_list:
            attached_cues.extend(by_skill.get(skill.skill_name) or [])

    # Dedup demos / cues while preserving order.
    demos: list[dict[str, Any]] = []
    seen_demo: set[str] = set()
    for demo in attached_demos:
        key = str(demo.get("gamefile") or json_fallback_key(demo))
        if key in seen_demo:
            continue
        seen_demo.add(key)
        demos.append(demo)
        if len(demos) >= max(1, int(max_demos)):
            break
    cues: list[str] = []
    seen_cue: set[str] = set()
    for cue in attached_cues:
        if cue in seen_cue:
            continue
        seen_cue.add(cue)
        cues.append(cue)
        if len(cues) >= max(1, int(max_cues)):
            break

    protocol_blocks: list[str] = []
    for skill in skill_list:
        protocol = [
            str(step).strip()
            for step in (skill.action_protocol or [])
            if str(step).strip()
        ]
        if protocol:
            protocol_blocks.append(
                f"[{skill.skill_name}] " + " -> ".join(protocol[:8])
            )

    parts: list[str] = [
        f"You are {agent_name}, a capability specialist. When dispatched, "
        "complete the full environment task end-to-end using only admissible "
        "actions. Bind placeholders from the current observation.",
        "Use the exclusive trajectory-derived context below; do not invent "
        "off-scope strategies.",
    ]
    if protocol_blocks:
        parts.append("Exclusive protocols: " + " || ".join(protocol_blocks) + ".")
    if demos:
        demo_text = " | ".join(
            block
            for index, demo in enumerate(demos, start=1)
            for block in [_format_demo_block(demo, index)]
            if block
        )
        if demo_text:
            parts.append("Success demos from verified wins: " + demo_text + ".")
    if cues:
        parts.append("Failure cues: " + " ".join(cues) + ".")
    if dispatch_only:
        parts.append(f"Act only when {executor_name} assigns this episode.")

    role_specification = " ".join(parts)
    if len(role_specification) > max_chars:
        role_specification = role_specification[: max_chars - 3].rstrip() + "..."

    return {
        "role_specification": role_specification,
        "n_demos": len(demos),
        "n_cues": len(cues),
        "n_protocol_blocks": len(protocol_blocks),
        "demos": demos,
        "failure_cues": cues,
        "context_source": (
            "trajectory_demos" if demos or cues else "protocol_only"
        ),
    }
