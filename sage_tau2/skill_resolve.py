"""Resolve agent.assigned_skills refs (skill_id preferred, name legacy fallback)."""

from __future__ import annotations

from typing import Any

from sage_tau2.schemas import SkillStatus, Tau2Skill
from sage_tau2.task_context import organizational_capability_key

_ACTIVE_STATUSES = {
    SkillStatus.CANDIDATE,
    SkillStatus.PROVISIONAL,
    SkillStatus.VERIFIED,
    SkillStatus.VERIFIED_LOW_SUPPORT,
}

_STATUS_RANK = {
    SkillStatus.VERIFIED: 0,
    SkillStatus.VERIFIED_LOW_SUPPORT: 1,
    SkillStatus.PROVISIONAL: 2,
    SkillStatus.CANDIDATE: 3,
}


def resolve_assigned_skills(
    assigned_refs: list[str] | None,
    skills: list[Tau2Skill],
    *,
    allowed_status: set[SkillStatus] | None = None,
) -> list[Tau2Skill]:
    """Map assigned refs to skills without duplicating by colliding skill_name.

    Prefer ``skill_id`` (unique). Legacy org snapshots that stored ``skill_name``
    fall back to name match, but only keep one skill per name (highest support /
    verified preferred) so duplicate names cannot inflate assigned sets.
    """
    if not assigned_refs:
        return []
    by_id = {s.skill_id: s for s in skills}
    by_name: dict[str, list[Tau2Skill]] = {}
    for skill in skills:
        by_name.setdefault(skill.skill_name, []).append(skill)

    out: list[Tau2Skill] = []
    seen: set[str] = set()
    for ref in assigned_refs:
        ref = str(ref or "").strip()
        if not ref:
            continue
        match: Tau2Skill | None = None
        if ref in by_id:
            match = by_id[ref]
        else:
            candidates = list(by_name.get(ref) or [])
            if allowed_status is not None:
                candidates = [s for s in candidates if s.status in allowed_status]
            if not candidates:
                continue
            candidates.sort(
                key=lambda s: (
                    0 if s.status == SkillStatus.VERIFIED else 1,
                    -int(s.support_count or 0),
                    s.skill_id,
                )
            )
            match = candidates[0]
        if match is None:
            continue
        if allowed_status is not None and match.status not in allowed_status:
            continue
        if match.skill_id in seen:
            continue
        seen.add(match.skill_id)
        out.append(match)
    return out


def heal_assigned_skill_refs(
    agent: Any,
    skills: list[Tau2Skill],
) -> dict[str, Any]:
    """Re-point ``agent.assigned_skills`` refs whose target went dead.

    A ref is *dead* when it resolves to nothing at all, or to a skill whose
    status is REJECTED/RETIRED (credit pruning, coarse dedupe, replacement by
    a richer duplicate). Dead refs silently zero out a specialist's dispatch
    eligibility ("zombie specialist"), so we cascade them onto the best
    active skill carrying the same capability key.

    Mutates ``agent.assigned_skills`` in place. Returns a report::

        {"changed": bool,
         "reassigned": {old_ref: new_skill_id, ...},
         "dropped": [old_ref, ...]}
    """
    report: dict[str, Any] = {"changed": False, "reassigned": {}, "dropped": []}
    refs = [str(r or "").strip() for r in (getattr(agent, "assigned_skills", None) or [])]
    if not refs:
        return report

    by_id = {s.skill_id: s for s in skills}
    by_name: dict[str, list[Tau2Skill]] = {}
    for skill in skills:
        by_name.setdefault(skill.skill_name, []).append(skill)

    def _resolve_any(ref: str) -> Tau2Skill | None:
        if ref in by_id:
            return by_id[ref]
        cands = list(by_name.get(ref) or [])
        if not cands:
            return None
        cands.sort(
            key=lambda s: (
                _STATUS_RANK.get(s.status, 9),
                -int(s.support_count or 0),
                s.skill_id,
            )
        )
        return cands[0]

    def _cap_key(skill: Tau2Skill) -> str:
        # Canonical derivation shared with cluster/nominate/dispatch:
        # metadata.write_capability_key → protocol write tool → capability_key.
        return str(organizational_capability_key(skill) or "").strip().lower()

    def _replacement_for(dead: Tau2Skill | None) -> Tau2Skill | None:
        if dead is None:
            return None
        want = _cap_key(dead)
        if not want:
            return None
        cands = [
            s
            for s in skills
            if s.status in _ACTIVE_STATUSES and _cap_key(s) == want
        ]
        if not cands:
            return None
        cands.sort(
            key=lambda s: (
                _STATUS_RANK.get(s.status, 9),
                -int(s.support_count or 0),
                s.skill_id,
            )
        )
        return cands[0]

    new_refs: list[str] = []
    kept_ids: set[str] = set()
    for ref in refs:
        if not ref:
            report["changed"] = True
            continue
        current = _resolve_any(ref)
        if current is not None and current.status in _ACTIVE_STATUSES:
            if current.skill_id not in kept_ids:
                kept_ids.add(current.skill_id)
                new_refs.append(ref)
            else:
                # Duplicate ref to an already-kept skill: drop silently.
                report["changed"] = True
            continue
        repl = _replacement_for(current)
        if repl is not None:
            # Capability still covered (possibly by a twin ref kept above).
            report["reassigned"][ref] = repl.skill_id
            report["changed"] = True
            if repl.skill_id not in kept_ids:
                kept_ids.add(repl.skill_id)
                new_refs.append(repl.skill_id)
        else:
            report["dropped"].append(ref)
            report["changed"] = True

    if report["changed"]:
        agent.assigned_skills = new_refs
    return report
