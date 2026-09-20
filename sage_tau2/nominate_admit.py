"""Nominate-then-admit for τ² — aligned with sage_mas online600_noseed.

Default org path matches ALFWorld online600:
  OrganizationEditor-style ADD → probation → onboarding (Spec-vs-Exec OFF).

Optional Spec-vs-Exec (actor_promotion_probe) remains available when enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol
from uuid import uuid4

from sage_tau2.distill import is_escalate_only_capability, is_read_only_protocol
from sage_tau2.organization import EXECUTOR_NAME, Organization
from sage_tau2.schemas import (
    AgentSpec,
    OrganizationEdit,
    OrganizationEditType,
    SkillStatus,
    Tau2Skill,
)
from sage_tau2.skill_clustering import SkillCluster, SkillClusterArchive
from sage_tau2.task_context import organizational_capability_key


@dataclass(slots=True)
class NominationPolicy:
    """Organization-gap nomination knobs (mirrors sage_mas OrganizationPolicy)."""

    min_cluster_support: int = 1
    min_member_skills: int = 1
    max_new_agents_per_round: int = 1
    add_when_uncovered_capability: bool = True
    # Match allow_provisional_org_edits=false on online600_noseed.
    allow_provisional: bool = False
    # ALFWorld does not gate ADD on injection MU; keep 0 by default.
    require_min_utility: float = 0.0
    dispatch_only_new_agents: bool = True
    probation_games: int = 3
    require_executable_protocol: bool = True
    allow_bootstrap_add: bool = True
    # Only nominate clusters whose member skills match this domain (multidomain).
    require_same_domain: str | None = None


@dataclass(slots=True)
class AdmissionPolicy:
    """Admission knobs (mirrors sage_mas; Spec optional like actor_promotion_probe)."""

    # online600_noseed sets actor_promotion_probe: false → default OFF.
    enable_spec_vs_exec: bool = False
    min_advantage: float = 0.0
    # When Spec is used, ties should not pass (stricter than prior τ² runs).
    accept_ties: bool = False
    num_tasks: int = 8
    on_pass_acting_status: str = "probation"
    online_trial_games: int = 3
    verify_skills_on_pass: bool = True
    on_fail_action: str = "remove"  # remove | demote | dormant
    # Used only by credit_fallback mode (not the default editor commit).
    min_utility: float = 0.60
    min_support: int = 3
    # When Spec is disabled: commit ADD like OrganizationEditor (default).
    # Set "credit_fallback" to require utility/support gates instead.
    no_spec_mode: str = "editor_commit"  # editor_commit | credit_fallback


@dataclass(slots=True)
class NominateAdmitConfig:
    nomination: NominationPolicy = field(default_factory=NominationPolicy)
    admission: AdmissionPolicy = field(default_factory=AdmissionPolicy)


class SpecVsExecProbe(Protocol):
    def __call__(
        self,
        *,
        specialist: AgentSpec,
        organization: Organization,
        skills: list[Tau2Skill],
        num_tasks: int,
    ) -> dict[str, Any]: ...


def _skill_utility(skill: Tau2Skill) -> float:
    credit = skill.metadata.get("skill_credit") or {}
    if "score" in credit:
        return float(credit["score"])
    if skill.metadata.get("utility") is not None:
        return float(skill.metadata["utility"])
    return 0.5


def _cluster_members(
    cluster: SkillCluster, skills: list[Tau2Skill]
) -> list[Tau2Skill]:
    by_id = {s.skill_id: s for s in skills}
    out = []
    for sid in cluster.member_skill_ids:
        skill = by_id.get(sid)
        if skill is None:
            continue
        if skill.status in {SkillStatus.REJECTED, SkillStatus.RETIRED}:
            continue
        out.append(skill)
    return out


def _cluster_support(members: list[Tau2Skill]) -> int:
    return sum(max(int(s.support_count or 0), len(s.evidence_ids), 1) for s in members)


def _specialist_name(capability_key: str) -> str:
    short = capability_key.replace("tau2.", "").replace("_", " ").strip()
    parts = [p.capitalize() for p in short.split() if p]
    base = "".join(parts) or "Capability"
    return f"{base}Specialist"


def admission_would_pass(
    specialist_sr: float,
    executor_sr: float,
    *,
    min_advantage: float = 0.0,
    accept_ties: bool = True,
) -> bool:
    """Same rule as sage_mas: ΔSR >= min_advantage; ties optional."""
    delta = float(specialist_sr) - float(executor_sr)
    if accept_ties:
        return delta >= float(min_advantage)
    return delta > float(min_advantage)


def propose_add_agents(
    *,
    organization: Organization,
    archive: SkillClusterArchive,
    skills: list[Tau2Skill],
    config: NominateAdmitConfig | None = None,
) -> list[OrganizationEdit]:
    """Stage B: nominate ADD_AGENT for uncovered capability clusters."""
    cfg = config or NominateAdmitConfig()
    nom = cfg.nomination
    edits: list[OrganizationEdit] = []
    only_executor = len(organization.specialists()) == 0
    if only_executor and not nom.allow_bootstrap_add:
        return [
            OrganizationEdit(
                edit_type=OrganizationEditType.DO_NOTHING,
                rationale="Bootstrap ADD disabled and only Executor exists.",
                status="noop",
            )
        ]

    ranked = sorted(
        archive.clusters,
        key=lambda c: (
            -_cluster_support(_cluster_members(c, skills)),
            -c.born_segment,
            c.cluster_id,
        ),
    )
    for cluster in ranked:
        if len(edits) >= nom.max_new_agents_per_round:
            break
        members = _cluster_members(cluster, skills)
        if not nom.allow_provisional:
            members = [s for s in members if s.status == SkillStatus.VERIFIED]
        if nom.require_executable_protocol:
            members = [s for s in members if s.action_protocol]
        # Skip escalate-only. Read-only is allowed for communicate_ok skills.
        filtered: list = []
        for s in members:
            if is_escalate_only_capability(s.capability_key):
                continue
            modes = (s.metadata or {}).get("success_modes") or (
                [s.metadata.get("success_mode")]
                if (s.metadata or {}).get("success_mode")
                else None
            )
            mode_set = {
                str(m).strip()
                for m in (modes or [])
                if m is not None and str(m).strip()
            }
            if is_read_only_protocol(s.action_protocol or []):
                if mode_set and mode_set <= {"communicate_ok"}:
                    filtered.append(s)
                continue
            filtered.append(s)
        members = filtered
        domain_req = str(nom.require_same_domain or "").strip().lower()
        if domain_req:
            members = [
                s
                for s in members
                if str(s.domain or "").strip().lower() == domain_req
            ]
        if len(members) < nom.min_member_skills:
            continue
        support = _cluster_support(members)
        if support < nom.min_cluster_support:
            continue
        mean_u = sum(_skill_utility(s) for s in members) / len(members)
        if mean_u < nom.require_min_utility:
            continue
        leader = next(
            (s for s in members if s.skill_id == cluster.leader_skill_id),
            members[0],
        )
        # Prefer write-spine ownership key over wide episode labels.
        org_capability = (
            organizational_capability_key(leader) or cluster.capability_key
        )
        if nom.add_when_uncovered_capability and organization.has_carrier(
            org_capability
        ):
            continue

        name = _specialist_name(org_capability)
        if any(e.new_agent and e.new_agent.name == name for e in edits):
            name = f"{name}_{cluster.cluster_id[-4:]}"
        if any(a.name == name for a in organization.agents):
            continue

        dispatch_only = bool(nom.dispatch_only_new_agents)
        skill_domains = sorted(
            {
                str(s.domain).strip().lower()
                for s in members
                if str(s.domain or "").strip()
            }
        )
        agent = AgentSpec(
            name=name,
            role="specialist",
            responsibilities=[
                f"Own capability cluster {org_capability}",
                "Follow assigned tool protocols under domain policy",
            ],
            # Prefer unique skill_id refs (skill_name collides across supports).
            assigned_skills=[s.skill_id for s in members],
            capability_keys=[org_capability],
            tool_permissions=["env_action", "tau2_tools"],
            activation_condition=leader.precondition[:300],
            role_specification=(
                f"Specialist for {org_capability}. "
                f"Primary protocol: {' -> '.join(leader.action_protocol[:8])}."
            ),
            acting_status="probation",
            agent_id=str(uuid4()),
            metadata={
                "cluster_id": cluster.cluster_id,
                "capability_key": org_capability,
                "cluster_capability_key": cluster.capability_key,
                "nominated_support": support,
                "nominated_utility": mean_u,
                "source": "sage_tau2_nominate",
                "dispatch_only": dispatch_only,
                "domains": skill_domains,
                "assigned_skill_names": [s.skill_name for s in members],
            },
            shadow_evaluation_record={
                "acting_status": "probation",
                "dispatch_only": dispatch_only,
                "applicable_dispatched_games": 0,
                "trial_games_remaining": int(nom.probation_games),
                # Spec path sets True until probe; editor commit clears it.
                "nominate_admit_awaiting_admission": True,
            },
        )
        edits.append(
            OrganizationEdit(
                edit_type=OrganizationEditType.ADD_AGENT,
                rationale=(
                    f"Uncovered capability {org_capability} "
                    f"(cluster={cluster.cluster_id}, support={support}); "
                    "nominate specialist for org admission."
                ),
                new_agent=agent,
                assigned_skill_names=[s.skill_id for s in members],
                status="nominated",
                cluster_id=cluster.cluster_id,
                capability_key=org_capability,
            )
        )
    if not edits:
        edits.append(
            OrganizationEdit(
                edit_type=OrganizationEditType.DO_NOTHING,
                rationale="No uncovered capability cluster met nomination gates.",
                status="noop",
            )
        )
    return edits


def _credit_admission_fallback(
    *,
    edit: OrganizationEdit,
    skills: list[Tau2Skill],
    archive: SkillClusterArchive,
    adm: AdmissionPolicy,
) -> dict[str, Any]:
    cluster = None
    if edit.cluster_id:
        for c in archive.clusters:
            if c.cluster_id == edit.cluster_id:
                cluster = c
                break
    members = _cluster_members(cluster, skills) if cluster else []
    if not members:
        refs = set(edit.assigned_skill_names or [])
        members = [
            s
            for s in skills
            if s.skill_id in refs or s.skill_name in refs
        ]
    support = _cluster_support(members) if members else 0
    mean_u = (
        sum(_skill_utility(s) for s in members) / len(members) if members else 0.0
    )
    accepted = support >= adm.min_support and mean_u >= adm.min_utility
    return {
        "accepted": accepted,
        "reason": (
            "credit_admission_pass"
            if accepted
            else (
                f"credit admission failed: support={support}<{adm.min_support} or "
                f"utility={mean_u:.3f}<{adm.min_utility}"
            )
        ),
        "support": support,
        "utility": mean_u,
        "specialist_sr": mean_u,
        "executor_sr": adm.min_utility,
        "n_tasks": 0,
        "mode": "credit_fallback",
    }


def apply_admission(
    *,
    organization: Organization,
    edit: OrganizationEdit,
    skills: list[Tau2Skill],
    result: dict[str, Any],
    config: NominateAdmitConfig | None = None,
) -> dict[str, Any]:
    """Mutate org/skills from Spec-vs-Exec (or fallback) admission result."""
    cfg = config or NominateAdmitConfig()
    adm = cfg.admission
    if edit.edit_type != OrganizationEditType.ADD_AGENT or edit.new_agent is None:
        return {
            "accepted": False,
            "reason": "not_an_add_agent_edit",
            "edit": asdict(edit),
        }

    specialist = edit.new_agent
    accepted = bool(result.get("accepted"))
    related = list(edit.assigned_skill_names or specialist.assigned_skills or [])
    verified_names: list[str] = []

    if accepted:
        specialist.acting_status = adm.on_pass_acting_status
        mode = str(result.get("mode") or "")
        # Spec PASS clears dispatch_only + marks probe passed.
        # Editor commit keeps dispatch_only (sage_mas) but clears awaiting;
        # probation quota then allows primary until onboarding accepts.
        record = dict(specialist.shadow_evaluation_record or {})
        record["nominate_admit_awaiting_admission"] = False
        record["last_nominate_admit"] = dict(result)
        record["same_skill_probe_passed"] = (
            mode == "spec_vs_exec" and
            (result.get("probe") or {}).get("comparison") == "same_skills_same_model_same_turn_budget"
        )
        record["trial_games_remaining"] = int(adm.online_trial_games)
        if mode == "editor_commit":
            record["promotion_probe_passed"] = False
            dispatch_only = bool(
                specialist.metadata.get("dispatch_only")
                or record.get("dispatch_only")
                or True
            )
            specialist.metadata["dispatch_only"] = dispatch_only
            record["dispatch_only"] = dispatch_only
        else:
            specialist.metadata["dispatch_only"] = False
            record["promotion_probe_passed"] = True
            record["dispatch_only"] = False
        specialist.shadow_evaluation_record = record
        organization.apply_edit(edit)
        # Spec / credit admission may stamp VERIFIED. editor_commit must NOT —
        # otherwise zero-credit skills become "verified" and pollute inject.
        if adm.verify_skills_on_pass and mode != "editor_commit":
            related_refs = set(related)
            for skill in skills:
                if (
                    skill.skill_id not in related_refs
                    and skill.skill_name not in related_refs
                ):
                    continue
                if skill.status in {SkillStatus.REJECTED, SkillStatus.RETIRED}:
                    continue
                skill.status = SkillStatus.VERIFIED
                meta = dict(skill.metadata or {})
                meta["nominate_admit_awaiting_admission"] = False
                meta["nominate_admit_verified_by"] = str(
                    result.get("mode") or "spec_vs_exec"
                )
                meta["add_agent_ready"] = True
                meta["carrier_agent"] = specialist.name
                skill.metadata = meta
                verified_names.append(skill.skill_name)
        elif mode == "editor_commit":
            related_refs = set(related)
            for skill in skills:
                if (
                    skill.skill_id not in related_refs
                    and skill.skill_name not in related_refs
                ):
                    continue
                meta = dict(skill.metadata or {})
                meta["nominate_admit_awaiting_admission"] = False
                meta["carrier_agent"] = specialist.name
                meta["nominate_admit_pending_credit_verify"] = True
                skill.metadata = meta
        edit.status = "accepted"
        return {
            "accepted": True,
            "reason": result.get("reason") or "admission_pass",
            "agent_name": specialist.name,
            "specialist_sr": result.get("specialist_sr"),
            "executor_sr": result.get("executor_sr"),
            "n_tasks": result.get("n_tasks"),
            "verified_skill_names": verified_names,
            "edit": asdict(edit),
            "mode": result.get("mode"),
        }

    # FAIL
    edit.status = "rejected"
    action = str(adm.on_fail_action or "remove").lower()
    # Ensure nominee is not left on the live roster from a partial apply.
    organization.agents = [
        a for a in organization.agents if a.name != specialist.name
    ]
    if action == "remove":
        return {
            "accepted": False,
            "reason": result.get("reason") or "admission_failed",
            "agent_name": specialist.name,
            "specialist_sr": result.get("specialist_sr"),
            "executor_sr": result.get("executor_sr"),
            "n_tasks": result.get("n_tasks"),
            "removed": True,
            "edit": asdict(edit),
            "mode": result.get("mode"),
        }
    specialist.metadata["dispatch_only"] = True
    record = dict(specialist.shadow_evaluation_record or {})
    record["nominate_admit_awaiting_admission"] = False
    record["promotion_probe_passed"] = False
    record["dispatch_only"] = True
    record["last_nominate_admit"] = dict(result)
    if action == "dormant":
        specialist.acting_status = "dormant"
        record["acting_status"] = "dormant"
    else:
        specialist.acting_status = "probation"
        record["acting_status"] = "probation"
    specialist.shadow_evaluation_record = record
    organization.apply_edit(edit)
    return {
        "accepted": False,
        "reason": result.get("reason") or "admission_failed",
        "agent_name": specialist.name,
        "specialist_sr": result.get("specialist_sr"),
        "executor_sr": result.get("executor_sr"),
        "n_tasks": result.get("n_tasks"),
        "removed": False,
        "new_acting_status": specialist.acting_status,
        "edit": asdict(edit),
        "mode": result.get("mode"),
    }


def run_nominate_admit(
    *,
    organization: Organization,
    archive: SkillClusterArchive,
    skills: list[Tau2Skill],
    config: NominateAdmitConfig | None = None,
    spec_vs_exec_probe: SpecVsExecProbe | Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Nominate → (optional) Spec-vs-Exec admit, matching sage_mas staging."""
    cfg = config or NominateAdmitConfig()
    proposals = propose_add_agents(
        organization=organization,
        archive=archive,
        skills=skills,
        config=cfg,
    )
    admissions: list[dict[str, Any]] = []
    accepted_agents: list[str] = []
    for edit in proposals:
        if edit.edit_type == OrganizationEditType.DO_NOTHING:
            admissions.append(
                {"accepted": False, "reason": "noop", "edit": asdict(edit)}
            )
            continue

        adm = cfg.admission
        if adm.enable_spec_vs_exec and spec_vs_exec_probe is not None:
            probe = spec_vs_exec_probe(
                specialist=edit.new_agent,
                organization=organization,
                skills=skills,
                num_tasks=int(adm.num_tasks),
            )
            specialist_sr = float(probe.get("specialist_sr") or 0.0)
            executor_sr = float(probe.get("executor_sr") or 0.0)
            n_tasks = int(probe.get("n_tasks") or 0)
            if n_tasks <= 0:
                accepted = False
                reason = "spec_vs_exec probe returned no scored tasks; refuse ADD"
            else:
                accepted = admission_would_pass(
                    specialist_sr,
                    executor_sr,
                    min_advantage=adm.min_advantage,
                    accept_ties=adm.accept_ties,
                )
                delta = specialist_sr - executor_sr
                if accepted:
                    reason = (
                        f"promote: specialist {probe.get('specialist_wins', '?')}/{n_tasks} "
                        f"vs Executor {probe.get('executor_wins', '?')}/{n_tasks} "
                        f"(delta={delta:+.3f}, min_advantage={adm.min_advantage}, "
                        "accept_ties=True)"
                        if adm.accept_ties
                        else f"promote: delta={delta:+.3f}"
                    )
                else:
                    reason = (
                        f"keep Executor primary: specialist "
                        f"{probe.get('specialist_wins', '?')}/{n_tasks} vs Executor "
                        f"{probe.get('executor_wins', '?')}/{n_tasks} "
                        f"(delta={delta:+.3f})"
                    )
            result = {
                "accepted": accepted,
                "reason": reason,
                "specialist_sr": specialist_sr,
                "executor_sr": executor_sr,
                "specialist_wins": probe.get("specialist_wins"),
                "executor_wins": probe.get("executor_wins"),
                "n_tasks": n_tasks,
                "mode": "spec_vs_exec",
                "probe": probe,
            }
        elif adm.enable_spec_vs_exec and spec_vs_exec_probe is None:
            result = {
                "accepted": False,
                "reason": "spec_vs_exec enabled but probe backend missing; refuse ADD",
                "specialist_sr": 0.0,
                "executor_sr": 0.0,
                "n_tasks": 0,
                "mode": "spec_vs_exec_missing",
            }
        elif str(adm.no_spec_mode or "editor_commit").strip().lower() == (
            "credit_fallback"
        ):
            result = _credit_admission_fallback(
                edit=edit, skills=skills, archive=archive, adm=adm
            )
        else:
            # Match sage_mas OrganizationEditor: commit ADD to probation;
            # onboarding (not Spec) decides accepted vs dormant.
            result = {
                "accepted": True,
                "reason": (
                    "editor_commit: uncovered capability ADD to probation "
                    "(Spec-vs-Exec off; onboarding will validate)"
                ),
                "specialist_sr": 0.0,
                "executor_sr": 0.0,
                "n_tasks": 0,
                "mode": "editor_commit",
            }

        decision = apply_admission(
            organization=organization,
            edit=edit,
            skills=skills,
            result=result,
            config=cfg,
        )
        admissions.append(decision)
        if decision.get("accepted"):
            accepted_agents.append(str(decision.get("agent_name")))

    return {
        "n_proposals": sum(
            1 for e in proposals if e.edit_type == OrganizationEditType.ADD_AGENT
        ),
        "accepted_agents": accepted_agents,
        "admissions": admissions,
        "organization_size": len(organization.agents),
        "specialists": [a.name for a in organization.specialists()],
    }
