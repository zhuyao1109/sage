"""Expand strong-model canonical skills into short student-facing patches.

Canonical skills from a strong teacher are often abstract skeletons
(``go / take / cool / move``). Mid-strong students need a short, readable
decision patch—not a long manual and not the raw teacher bank dump.

This module builds ``metadata.student_patch`` with a compact ``inject_text``.
Prefer contrastive teacher-win ∩ student-fail remainders when trajectories
are available; otherwise fall back to a shortened protocol reading aid.

Output skills are ``compiled_candidate`` and stay ``inject_ready=false`` until
a paired student MU gate promotes them to executable.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from sage_mas.idle_trajectory import (
    abstract_productive_protocol,
    filter_informative_failures,
)
from sage_mas.schemas import Skill, SkillStatus
from sage_mas.skill_bank_roles import mark_canonical, mark_compiled_candidate
from sage_mas.skill_quality import annotate_skill_identity
from sage_mas.skill_student_transfer import (
    first_protocol_divergence,
    pair_teacher_wins_student_fails,
)


def _family_of(skill: Skill) -> str:
    annotate_skill_identity(skill)
    return str(
        (skill.metadata or {}).get("primary_task_family")
        or (
            (skill.applicable_task_families or [""])[0]
            if skill.applicable_task_families
            else ""
        )
    )


def _protocol_steps(skill: Skill, *, max_steps: int = 8) -> list[str]:
    steps: list[str] = []
    for raw in skill.action_protocol or []:
        text = str(raw or "").strip()
        if not text:
            continue
        # Drop verbose compile wrappers if a stepwise skill is re-enriched.
        lower = text.lower()
        if lower.startswith(("trigger:", "required_state:", "completion_test:")):
            continue
        if lower.startswith("step:"):
            text = text.split(":", 1)[1].strip()
        if lower.startswith("recovery:"):
            continue
        if text not in steps:
            steps.append(text)
        if len(steps) >= max(1, int(max_steps)):
            break
    return steps


def _short_when(skill: Skill, *, stall_near: str | None = None) -> str:
    family = _family_of(skill) or "this task"
    if stall_near:
        return (
            f"family `{family}`; student stalled / diverged near `{stall_near}`"
        )
    precondition = " ".join(str(skill.precondition or "").split())
    if precondition and len(precondition) <= 140:
        return precondition
    effect = " ".join(str(skill.expected_effect or "").split())
    if effect and len(effect) <= 120:
        return f"family `{family}`; aim for: {effect}"
    return f"family `{family}`; follow the Do steps with grounded placeholders"


def _dont_lines(
    skill: Skill,
    *,
    recovery: Sequence[str] | None = None,
    max_dont: int = 2,
) -> list[str]:
    lines: list[str] = []
    for item in recovery or []:
        text = " ".join(str(item or "").split())
        if text and text not in lines:
            lines.append(text)
        if len(lines) >= max(0, int(max_dont)):
            return lines[: max(0, int(max_dont))]
    for item in (skill.metadata or {}).get("anti_patterns") or []:
        text = " ".join(str(item or "").split())
        if not text:
            continue
        # Keep anti-patterns short.
        if len(text) > 110:
            text = text[:107] + "..."
        if text not in lines:
            lines.append(text)
        if len(lines) >= max(0, int(max_dont)):
            break
    if not lines:
        lines.append(
            "Do not repeat an invalid / no-progress action; advance to the "
            "next unfinished Do step."
        )
    return lines[: max(1, int(max_dont))]


def render_student_inject_text(
    *,
    skill_name: str,
    when: str,
    do_steps: Sequence[str],
    dont_steps: Sequence[str],
) -> str:
    """Compact prompt block for mid-strong actors."""
    lines = [
        f"Student skill patch: {skill_name}",
        f"When: {when}",
        "Do (in order; bind <placeholders> from observation/admissible only):",
    ]
    for index, step in enumerate(do_steps, start=1):
        lines.append(f"  {index}. {step}")
    if dont_steps:
        lines.append("Don't:")
        for item in dont_steps:
            lines.append(f"  - {item}")
    lines.append("Emit exactly one admissible environment action this turn.")
    return "\n".join(lines)


def _best_contrastive_remainder(
    skill: Skill,
    teacher_records: Sequence[dict[str, Any]],
    student_records: Sequence[dict[str, Any]],
) -> tuple[list[str], str | None, dict[str, Any]]:
    family = _family_of(skill) or None
    informative = filter_informative_failures(student_records)
    pairs = pair_teacher_wins_student_fails(
        teacher_records,
        informative,
        task_family=family,
    )
    best_remainder: list[str] = []
    stall_near: str | None = None
    trace: dict[str, Any] = {
        "n_pairs": len(pairs),
        "used_contrastive": False,
    }
    for teacher, student in pairs:
        teacher_proto = abstract_productive_protocol(teacher)
        student_proto = abstract_productive_protocol(student)
        if not teacher_proto:
            continue
        diverge = first_protocol_divergence(teacher_proto, student_proto)
        remainder = list(teacher_proto[diverge:]) or list(teacher_proto[-3:])
        if len(remainder) <= len(best_remainder):
            continue
        best_remainder = remainder
        if diverge > 0 and diverge - 1 < len(student_proto):
            stall_near = student_proto[diverge - 1]
        elif student_proto:
            stall_near = student_proto[-1]
        else:
            stall_near = None
        trace = {
            "n_pairs": len(pairs),
            "used_contrastive": True,
            "diverge_index": diverge,
            "teacher_gamefile": str(
                teacher.get("gamefile") or teacher.get("task_id") or ""
            ),
            "student_gamefile": str(
                student.get("gamefile") or student.get("task_id") or ""
            ),
            "remainder": list(remainder),
            "stall_near": stall_near,
        }
    return best_remainder, stall_near, trace


def _recovery_from_student_fails(
    student_records: Sequence[dict[str, Any]],
    *,
    task_family: str | None,
    max_cues: int = 2,
) -> list[str]:
    from collections import Counter

    informative = filter_informative_failures(student_records)
    last_actions: Counter[str] = Counter()
    for record in informative:
        family = str(record.get("task_family") or "")
        if task_family and family != task_family:
            continue
        proto = abstract_productive_protocol(record)
        if proto:
            last_actions[proto[-1]] += 1
    cues: list[str] = []
    for action, count in last_actions.most_common(max(1, int(max_cues))):
        cues.append(
            f"if stuck near `{action}` (n={count}), continue the next Do step "
            "instead of repeating it"
        )
    return cues


def _default_slot_bindings(do_steps: Sequence[str]) -> dict[str, dict[str, str]]:
    joined = " ".join(do_steps).lower()
    bindings: dict[str, dict[str, str]] = {}
    if "<object>" in joined or "<target>" in joined:
        bindings["object"] = {"source": "task.target"}
        bindings["target"] = {"source": "task.target"}
    if "<destination>" in joined or "<receptacle>" in joined:
        bindings["destination"] = {"source": "task.destination"}
        bindings["receptacle"] = {"source": "task.destination"}
    if "<location>" in joined:
        bindings["location"] = {"source": "auto"}
    if "<tool>" in joined:
        bindings["tool"] = {"source": "auto"}
    if "<entity>" in joined:
        bindings["entity"] = {"source": "auto"}
    return bindings


def _forbidden_templates_from_steps(do_steps: Sequence[str]) -> list[str]:
    """Templates that become concrete forbidden actions after the step is done."""
    # All productive steps are candidates; runtime only forbids those already done.
    return [str(step) for step in do_steps if str(step).strip()]


def _success_demos_from_skill(
    skill: Skill,
    *,
    max_demos: int = 2,
    max_actions: int = 16,
) -> list[dict[str, Any]]:
    """Copy winning specialist demos into the patch (concrete action chains)."""
    from sage_mas.rich_skill_context import success_demos_from_metadata

    return success_demos_from_metadata(
        skill.metadata,
        max_demos=max_demos,
        max_actions=max_actions,
    )


def _failure_cues_from_skill(skill: Skill, *, max_cues: int = 4) -> list[str]:
    from sage_mas.rich_skill_context import failure_cues_from_metadata

    return failure_cues_from_metadata(skill.metadata, max_cues=max_cues)


def _anti_patterns_from_skill(skill: Skill, *, max_items: int = 3) -> list[str]:
    from sage_mas.rich_skill_context import anti_patterns_from_metadata

    return anti_patterns_from_metadata(skill.metadata, max_items=max_items)

def build_student_patch(
    skill: Skill,
    *,
    teacher_records: Sequence[dict[str, Any]] | None = None,
    student_records: Sequence[dict[str, Any]] | None = None,
    max_do_steps: int = 6,
    max_dont: int = 2,
    max_success_demos: int = 2,
) -> dict[str, Any]:
    """Build a structured student_patch for code-side bind + short Flash inject.

    Flash should never be asked to interpret family names or ``<placeholders>``.
    Runtime matches ``trigger_signature``, fills ``slot_bindings``, then renders
    a concrete ``inject_text`` (bound steps + optional success demos). The stored
    ``inject_text`` here is only an unbound preview for debugging.
    """
    teacher_records = list(teacher_records or [])
    student_records = list(student_records or [])
    remainder, stall_near, contrastive = _best_contrastive_remainder(
        skill,
        teacher_records,
        student_records,
    )
    if remainder:
        do_steps = remainder[: max(1, int(max_do_steps))]
        source = "contrastive_remainder"
    else:
        do_steps = _protocol_steps(skill, max_steps=max_do_steps)
        source = "canonical_protocol"
    if not do_steps:
        do_steps = ["take <object> from <receptacle>"]

    family = _family_of(skill)
    recovery = _recovery_from_student_fails(
        student_records,
        task_family=family or None,
        max_cues=max_dont,
    )
    when = _short_when(skill, stall_near=stall_near)
    dont = _dont_lines(skill, recovery=recovery, max_dont=max_dont)
    success_demos = _success_demos_from_skill(
        skill, max_demos=max_success_demos
    )
    failure_cues = _failure_cues_from_skill(skill)
    anti_patterns = _anti_patterns_from_skill(skill)
    preview = render_student_inject_text(
        skill_name=skill.skill_name,
        when=when,
        do_steps=do_steps,
        dont_steps=dont,
    )
    trigger_signature = {
        "task_families": [family] if family else [],
        "require_admissible": True,
        # Holding is optional; cooler/heater patches often start before take.
        "require_holding": None,
        "stall_near": stall_near,
    }
    return {
        "schema_version": "student_patch_v2",
        "source": source,
        "skill_label": family or skill.skill_name,
        # Code-side only (not shown to Flash as semantics).
        "trigger_signature": trigger_signature,
        "slot_bindings": _default_slot_bindings(do_steps),
        "action_templates": list(do_steps),
        "forbidden_templates": _forbidden_templates_from_steps(do_steps),
        # Concrete win trajectories + rich MU cues (shared formatter).
        "success_demos": success_demos,
        "failure_cues": failure_cues,
        "anti_patterns": anti_patterns,
        # Legacy/debug fields.
        "when": when,
        "do": list(do_steps),
        "dont": list(dont),
        "inject_text": preview,
        "inject_text_note": (
            "Unbound preview only. Runtime must call bind_student_patch() "
            "and inject BoundStudentPatch.inject_text "
            "(concrete actions + rich demos/cues)."
        ),
        "contrastive": contrastive,
    }


def enrich_skill_for_student(
    skill: Skill,
    *,
    teacher_records: Sequence[dict[str, Any]] | None = None,
    student_records: Sequence[dict[str, Any]] | None = None,
    max_do_steps: int = 6,
    max_dont: int = 2,
    keep_canonical_protocol: bool = True,
) -> Skill:
    """Return a compiled_candidate with structured ``student_patch`` (v2)."""
    annotate_skill_identity(skill)
    patch = build_student_patch(
        skill,
        teacher_records=teacher_records,
        student_records=student_records,
        max_do_steps=max_do_steps,
        max_dont=max_dont,
    )
    clone = deepcopy(skill)
    # Keep abstract skeleton for clustering; injection uses bound student_patch.
    if keep_canonical_protocol:
        clone.action_protocol = list(skill.action_protocol or [])
    else:
        clone.action_protocol = list(patch["action_templates"])
    clone.description = (
        "Student-enriched patch of a strong-model skill: structured trigger + "
        "slot bindings + success demos; runtime renders concrete Flash inject text."
    )
    clone.status = SkillStatus.PROVISIONAL
    ready = mark_compiled_candidate(
        clone,
        canonical_skill_id=skill.skill_id,
        compile_variant="student_patch_v2",
    )
    md = dict(ready.metadata or {})
    md["student_patch"] = patch
    # Do NOT put unbound preview into inject_render_text for production use.
    # Runtime prefers bind_student_patch(); preview stays under student_patch.
    md.pop("inject_render_text", None)
    md["student_enrichment"] = {
        "source": patch["source"],
        "used_contrastive": bool((patch.get("contrastive") or {}).get("used_contrastive")),
        "canonical_skill_name": skill.skill_name,
        "canonical_skill_id": skill.skill_id,
        "schema_version": patch.get("schema_version"),
    }
    # Explicitly not injectable until student MU passes.
    md["inject_ready"] = False
    md["not_for_injection"] = True
    md["not_for_add_agent"] = True
    ready.metadata = md
    return ready


def enrich_skills_for_student(
    skills: Sequence[Skill],
    *,
    teacher_records: Sequence[dict[str, Any]] | None = None,
    student_records: Sequence[dict[str, Any]] | None = None,
    max_do_steps: int = 6,
    max_dont: int = 2,
    mark_inputs_canonical: bool = True,
    teacher_model: str | None = None,
) -> dict[str, list[Skill]]:
    """Enrich a bank; optionally mark inputs canonical first.

    Returns ``{"canonical": [...], "enriched": [...]}``.
    """
    canonical: list[Skill] = []
    enriched: list[Skill] = []
    for skill in skills:
        source = (
            mark_canonical(skill, teacher_model=teacher_model)
            if mark_inputs_canonical
            else deepcopy(skill)
        )
        canonical.append(source)
        enriched.append(
            enrich_skill_for_student(
                source,
                teacher_records=teacher_records,
                student_records=student_records,
                max_do_steps=max_do_steps,
                max_dont=max_dont,
            )
        )
    return {"canonical": canonical, "enriched": enriched}
