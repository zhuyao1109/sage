"""Fast bank entry: form-ok distilled skills become verified without MU."""

from __future__ import annotations

from sage_mas.schemas import Skill, SkillStatus


def skill_is_form_ok_for_fast_verify(skill: Skill) -> bool:
    """Structural gates only — no paired MU required."""
    if skill.status in {SkillStatus.REJECTED, SkillStatus.RETIRED}:
        return False
    md = skill.metadata or {}
    if md.get("protocol_form_ok") is False:
        return False
    if md.get("protocol_structure_ok") is False:
        return False
    if md.get("mu_rejected") is True and float(skill.marginal_utility or 0.0) < 0.0:
        return False
    protocol = list(skill.action_protocol or [])
    if not protocol:
        return False
    return True


def auto_verify_form_ok_skill(skill: Skill) -> bool:
    """Promote a form-ok skill to verified / org-ready. Returns True if changed."""
    if skill.status == SkillStatus.VERIFIED:
        # Still clear injection/ADD blocks left by older MU gates.
        md = dict(skill.metadata or {})
        changed = False
        if md.get("not_for_add_agent") and md.get("mu_rejected") is not True:
            md.pop("not_for_add_agent", None)
            changed = True
        if md.get("inject_ready") is False and md.get("mu_rejected") is not True:
            md["inject_ready"] = True
            changed = True
        if md.get("add_agent_ready") is False and md.get("mu_rejected") is not True:
            md["add_agent_ready"] = True
            changed = True
        if changed:
            skill.metadata = md
        return changed
    if not skill_is_form_ok_for_fast_verify(skill):
        return False
    md = dict(skill.metadata or {})
    skill.status = SkillStatus.VERIFIED
    md["inject_ready"] = True
    md["add_agent_ready"] = True
    md["executor_optional_injection"] = True
    md["auto_verified_form_ok"] = True
    md["skill_bank_role"] = "executable"
    md["not_for_injection"] = False
    md.pop("not_for_add_agent", None)
    md.pop("mu_rejected", None)
    if int(md.get("protocol_alignment_support") or 0) < 1:
        md["protocol_alignment_support"] = max(
            1,
            int(skill.support_count or 1),
        )
    if md.get("protocol_alignment_ok") is None:
        md["protocol_alignment_ok"] = True
    if md.get("protocol_structure_ok") is None:
        md["protocol_structure_ok"] = True
    if md.get("protocol_form_ok") is None:
        md["protocol_form_ok"] = True
    md["credit_promoted_pending_org"] = True
    skill.metadata = md
    return True


def auto_verify_form_ok_skills(skills: list[Skill]) -> list[Skill]:
    """In-place promote form-ok skills; return the same list."""
    for skill in skills:
        auto_verify_form_ok_skill(skill)
    return skills
