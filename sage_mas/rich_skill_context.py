"""Shared rich skill context for Flash injection (demos / cues / anti).

Extracted from the ``probe_rich_skill_mu`` path (``MASRuntime._render_skills``)
so student_patch_v2 can reuse the same learned trajectory context without
duplicating formatting logic.
"""

from __future__ import annotations

from typing import Any, Sequence

from sage_mas.schemas import Skill


def success_demos_from_metadata(
    metadata: dict[str, Any] | None,
    *,
    max_demos: int = 3,
    max_actions: int = 16,
) -> list[dict[str, Any]]:
    raw = (metadata or {}).get("specialist_demos") or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for demo in raw:
        if not isinstance(demo, dict):
            continue
        if demo.get("won") is False:
            continue
        actions = [
            str(a).strip()
            for a in (demo.get("actions") or [])
            if str(a).strip()
        ]
        if not actions:
            continue
        out.append(
            {
                "task_family": str(demo.get("task_family") or "").strip(),
                "actions": actions[: max(1, int(max_actions))],
                "won": True,
            }
        )
        if len(out) >= max(0, int(max_demos)):
            break
    return out


def failure_cues_from_metadata(
    metadata: dict[str, Any] | None,
    *,
    max_cues: int = 4,
) -> list[str]:
    md = metadata or {}
    cues = list(md.get("specialist_failure_cues") or [])
    if not cues:
        cues = list(md.get("failure_branches") or [])
    out: list[str] = []
    for item in cues:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= max(0, int(max_cues)):
            break
    return out


def anti_patterns_from_metadata(
    metadata: dict[str, Any] | None,
    *,
    max_items: int = 3,
) -> list[str]:
    out: list[str] = []
    for item in (metadata or {}).get("anti_patterns") or []:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= max(0, int(max_items)):
            break
    return out


def format_success_demo_lines(
    demos: Sequence[Any],
    *,
    max_demos: int = 3,
    indent: str = "  ",
) -> list[str]:
    lines: list[str] = []
    count = 0
    for demo in demos:
        if count >= max_demos:
            break
        if not isinstance(demo, dict):
            continue
        actions = [
            str(a).strip()
            for a in (demo.get("actions") or [])
            if str(a).strip()
        ]
        if not actions:
            continue
        count += 1
        family = str(demo.get("task_family") or "").strip() or "task"
        seq = " -> ".join(actions[:16])
        lines.append(f"{indent}Success demo {count} ({family}): {seq}")
    return lines


def format_rich_context_lines(
    skill: Skill,
    *,
    include_header: bool = True,
    include_contract: bool = True,
    include_demos: bool = True,
    include_cues: bool = True,
    include_anti: bool = True,
    max_demos: int = 3,
    demos_override: Sequence[Any] | None = None,
) -> list[str]:
    """Format the rich block used by probe_rich_skill_mu.

    ``include_contract`` controls precondition / protocol / stages / executable.
    Student-patch binding typically sets this False (bound checklist already
    covers the productive steps) and keeps demos + cues + anti.
    """
    md = skill.metadata or {}
    lines: list[str] = []
    if include_header:
        lines.append(f"- {skill.skill_name}")

    if include_contract:
        protocol = "; ".join(skill.action_protocol or [])
        lines.extend(
            [
                f"  Precondition: {skill.precondition}",
                f"  Protocol: {protocol}",
                "  Placeholder note: <object>/<tool>/<receptacle>/<location> "
                "must be replaced by names from the current observation and "
                "admissible actions.",
                f"  Expected effect: "
                f"{skill.expected_effect or 'Improve task execution.'}",
            ]
        )
        stages = md.get("protocol_stages") or []
        if stages:
            lines.append(f"  Stages: {' -> '.join(str(s) for s in stages)}")
        executable = md.get("executable_protocol") or []
        if isinstance(executable, list) and executable:
            step_bits: list[str] = []
            for item in executable[:8]:
                if isinstance(item, dict):
                    template = str(item.get("action_template", "")).strip()
                    if not template:
                        continue
                    hint = str(item.get("expected_obs_hint", "") or "").strip()
                    if hint:
                        step_bits.append(f"{template} [expect: {hint}]")
                    else:
                        step_bits.append(template)
            if step_bits:
                lines.append("  Executable steps: " + " -> ".join(step_bits))

    if include_demos:
        demos = (
            list(demos_override)
            if demos_override is not None
            else success_demos_from_metadata(md, max_demos=max_demos)
        )
        lines.extend(
            format_success_demo_lines(demos, max_demos=max_demos, indent="  ")
        )

    if include_cues:
        cues = failure_cues_from_metadata(md)
        # Prefer cues already copied onto student_patch when present.
        patch = md.get("student_patch") if isinstance(md.get("student_patch"), dict) else {}
        patch_cues = list((patch or {}).get("failure_cues") or [])
        if patch_cues:
            cues = [str(x).strip() for x in patch_cues if str(x).strip()][:4]
        if cues:
            lines.append("  Failure cues: " + " | ".join(cues[:4]))

    if include_anti:
        anti = anti_patterns_from_metadata(md)
        patch = md.get("student_patch") if isinstance(md.get("student_patch"), dict) else {}
        patch_anti = list((patch or {}).get("anti_patterns") or [])
        if patch_anti:
            anti = [str(x).strip() for x in patch_anti if str(x).strip()][:3]
        if anti:
            lines.append("  Anti-patterns: " + " | ".join(anti[:3]))

    return lines


def render_rich_skill_block(
    skill: Skill,
    *,
    include_header: bool = True,
    include_contract: bool = True,
    include_demos: bool = True,
    include_cues: bool = True,
    include_anti: bool = True,
    max_demos: int = 3,
    demos_override: Sequence[Any] | None = None,
) -> str:
    return "\n".join(
        format_rich_context_lines(
            skill,
            include_header=include_header,
            include_contract=include_contract,
            include_demos=include_demos,
            include_cues=include_cues,
            include_anti=include_anti,
            max_demos=max_demos,
            demos_override=demos_override,
        )
    )


def render_rich_supplement_for_student_patch(
    skill: Skill,
    *,
    demos: Sequence[Any] | None = None,
    max_demos: int = 2,
) -> str:
    """Rich extras appended under a bound student_patch checklist.

    Keeps demos / failure cues / anti-patterns from the rich MU path; omits the
    long abstract contract (already replaced by concrete bound steps).
    """
    lines = format_rich_context_lines(
        skill,
        include_header=False,
        include_contract=False,
        include_demos=True,
        include_cues=True,
        include_anti=True,
        max_demos=max_demos,
        demos_override=demos,
    )
    if not lines:
        return ""
    return (
        "Rich trajectory context (adapt entities to the current observation):\n"
        + "\n".join(lines)
    )
