"""Restricted organization edits driven by skill distribution change."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from sage_mas.assignment_gap import AssignmentGapResult
from sage_mas.capability_contract import (
    AgentRoleCompiler,
    compile_capability_contract,
    sync_agent_skill_scopes,
)
from sage_mas.schemas import (
    AgentSpec,
    DistributionShift,
    OrganizationEdit,
    OrganizationEditType,
    Skill,
    SkillStatus,
)


@dataclass(slots=True)
class OrganizationPolicy:
    # Organization edits consume verified capability clusters only; the
    # pipeline enforces held-out MU verification before reaching this policy.
    distribution_shift_threshold: float = 0.15
    min_cluster_support: int = 1
    min_failure_rate: float = 0.0
    min_assignment_gap: float = 0.1
    allow_bootstrap_add: bool = True
    max_bootstrap_agents_per_round: int = 1
    # Cap ADD_AGENT commits per distill round (seeded banks included).
    max_new_agents_per_round: int = 1
    cluster_by: str = "capability"
    # When True, a verified capability cluster with no non-Executor carrier is
    # treated as organizationally novel (enables ADD from Executor-only seeds
    # without relying on distribution-shift vs an already-populated bank).
    # SAGE: Executor owning every cluster is an organization bottleneck; lack of
    # a specialist carrier is itself structural evidence for ADD_AGENT.
    add_when_uncovered_capability: bool = False
    # New specialists only act after Executor dispatch; they never auto-steal
    # a stage from the GiGPO-style Executor.
    dispatch_only_new_agents: bool = True
    probation_games: int = 8
    # ADD_AGENT requires trajectory-derived executable steps + credit adherence.
    require_executable_protocol: bool = True
    require_trajectory_executable_protocol: bool = False
    min_protocol_adherence_for_add_agent: float = 0.34
    # New agents are carriers of already-proven skills, not MU bets.
    require_positive_mu_for_add_agent: bool = False
    min_marginal_utility_for_add_agent: float = 0.0
    # SAGE dual-bank: ADD_AGENT only from Executable skills with cluster coverage.
    require_executable_bank_for_add_agent: bool = False
    min_executable_coverage_for_add_agent: float = 0.5
    # Shadow/onboarding: new agent must be dispatched often enough on scope.
    min_new_agent_call_rate: float = 0.5
    # Prefer ASSIGN only when a *specialist* carrier already owns the cluster.
    # Never use this to park new capabilities on the sole Executor (that blocks
    # SAGE restructuring / ADD_AGENT from Executor-only organizations).
    prefer_assign_before_add: bool = True


def cluster_key_for_skill(skill: Skill, *, cluster_by: str = "capability") -> str:
    """Stable capability-cluster key used for org decisions."""
    mode = (cluster_by or "capability").strip().lower()
    if mode == "role":
        return (skill.suggested_role or "specialist").strip().lower() or "specialist"
    if mode == "capability" and skill.capability_key:
        return skill.capability_key.strip().lower()
    family = str(skill.metadata.get("primary_task_family", "") or "").strip().lower()
    signal = str(skill.metadata.get("source_signal", "") or "").strip().lower()
    if not family and not signal:
        return (skill.suggested_role or "specialist").strip().lower() or "specialist"
    return f"{family or 'other'}|{signal or 'unspecified'}"


def unique_evidence_support(skills: list[Skill]) -> int:
    return len(
        {
            evidence_id
            for skill in skills
            for evidence_id in skill.evidence_ids
        }
    )


class OrganizationEditor:
    """Propose assign/add edits from capability-cluster distribution shift."""

    def __init__(self, policy: OrganizationPolicy | None = None):
        self.policy = policy or OrganizationPolicy()

    @staticmethod
    def _shift_metrics(
        skills: list[Skill],
        cluster_shift: DistributionShift | None = None,
    ) -> tuple[float, float, bool]:
        if cluster_shift is not None:
            return (
                float(cluster_shift.shift_score),
                float(cluster_shift.kl_divergence),
                cluster_shift.baseline_source == "empty_skill_bank",
            )
        shift_score = 0.0
        kl_divergence = 0.0
        empty_bank = False
        for skill in skills:
            shift = skill.distribution_shift
            if shift is None:
                continue
            shift_score = max(shift_score, shift.shift_score)
            kl_divergence = max(kl_divergence, shift.kl_divergence)
            empty_bank = empty_bank or shift.baseline_source == "empty_skill_bank"
        return shift_score, kl_divergence, empty_bank

    def is_deviated(
        self,
        skills: list[Skill],
        *,
        support: int | None = None,
        cluster_shift: DistributionShift | None = None,
    ) -> bool:
        total_support = (
            unique_evidence_support(skills) if support is None else int(support)
        )
        if total_support < self.policy.min_cluster_support:
            return False
        shift_score, kl_divergence, empty_bank = self._shift_metrics(
            skills,
            cluster_shift=cluster_shift,
        )
        threshold = self.policy.distribution_shift_threshold
        if (
            cluster_shift is not None
            and cluster_shift.metric == "cosine_nearest_skill"
            and cluster_shift.failure_rate < self.policy.min_failure_rate
        ):
            return False
        if empty_bank:
            return bool(self.policy.allow_bootstrap_add)
        return shift_score > threshold or kl_divergence > threshold

    @staticmethod
    def _is_executor(agent: AgentSpec) -> bool:
        return "executor" in f"{agent.name} {agent.role}".lower()

    def _cluster_has_carrier(
        self,
        cluster_key: str,
        existing_agents: list[AgentSpec] | None,
    ) -> bool:
        """True if a non-Executor agent already owns this capability cluster."""
        target = str(cluster_key or "").strip().lower()
        if not target or not existing_agents:
            return False
        for agent in existing_agents:
            if self._is_executor(agent):
                continue
            record = agent.shadow_evaluation_record or {}
            contract = record.get("capability_contract") or {}
            markers = {
                str(contract.get("name") or "").strip().lower(),
                str(contract.get("capability_id") or "").strip().lower(),
                str(agent.role or "").strip().lower(),
                str(agent.name or "").strip().lower(),
            }
            markers.update(
                str(value).strip().lower()
                for value in (record.get("capability_keys") or [])
                if str(value).strip()
            )
            if target in markers:
                return True
            # Compact match: transform.heat ↔ HeatTransform / transform heat.
            compact_target = target.replace(".", "").replace("_", "").replace("-", "")
            for marker in markers:
                compact = marker.replace(".", "").replace("_", "").replace("-", "")
                if compact_target and compact_target in compact:
                    return True
            blob = " ".join(
                [
                    str(agent.name or ""),
                    str(agent.role or ""),
                    str(agent.activation_condition or ""),
                    str(contract.get("name") or ""),
                ]
            ).lower()
            if target in blob or target.replace(".", " ") in blob:
                return True
        return False

    def has_skill_change(self, skills: list[Skill]) -> bool:
        return self.is_deviated(skills)

    def propose(
        self,
        skills: list[Skill],
        gap: AssignmentGapResult,
        existing_agent_names: set[str] | None = None,
        executor_name: str = "Executor",
        existing_agents: list[AgentSpec] | None = None,
        *,
        cluster_key: str | None = None,
        cluster_shift: DistributionShift | None = None,
        bootstrap_slots_remaining: int | None = None,
        add_slots_remaining: int | None = None,
    ) -> OrganizationEdit:
        skill_names = [skill.skill_name for skill in skills]
        total_support = unique_evidence_support(skills)
        shift_score, kl_divergence, empty_bank = self._shift_metrics(
            skills,
            cluster_shift=cluster_shift,
        )
        key = cluster_key or (
            cluster_key_for_skill(skills[0], cluster_by=self.policy.cluster_by)
            if skills
            else "empty"
        )
        metric = (
            cluster_shift.metric
            if cluster_shift is not None
            else "legacy_distribution"
        )
        nearest = (
            cluster_shift.nearest_skill_name
            if cluster_shift is not None
            else None
        )
        metrics_txt = (
            f"cluster={key}; support={total_support}; "
            f"shift={shift_score:.3f}; kl={kl_divergence:.3f}; "
            f"metric={metric}; nearest={nearest or 'none'}; "
            f"empty_bank={empty_bank}"
        )
        existing = list(existing_agents or [])
        names = existing_agent_names or {agent.name for agent in existing}

        unverified = [
            skill.skill_name
            for skill in skills
            if skill.status != SkillStatus.VERIFIED
        ]
        if unverified:
            return OrganizationEdit(
                edit_type=OrganizationEditType.DO_NOTHING,
                rationale=(
                    "ADD/ASSIGN blocked because the capability cluster "
                    "contains unverified skills: "
                    f"{', '.join(unverified)}. {metrics_txt}."
                ),
                assigned_skill_names=skill_names,
                assignment_gap=gap.assignment_gap,
            )

        if total_support < self.policy.min_cluster_support:
            return OrganizationEdit(
                edit_type=OrganizationEditType.DO_NOTHING,
                rationale=(
                    f"Cluster support below min_cluster_support "
                    f"({total_support} < {self.policy.min_cluster_support}); "
                    f"{metrics_txt}."
                ),
                assigned_skill_names=skill_names,
                assignment_gap=gap.assignment_gap,
            )

        deviated = self.is_deviated(
            skills,
            support=total_support,
            cluster_shift=cluster_shift,
        )
        # Uncovered is independent of distribution-shift: a high-shift cluster
        # with only Executor as owner is still an organizational gap (SAGE Gr).
        uncovered = bool(
            self.policy.add_when_uncovered_capability
            and not self._cluster_has_carrier(key, existing)
        )
        if uncovered:
            deviated = True
            metrics_txt = f"{metrics_txt}; uncovered_capability=True"
        if not deviated:
            if gap.best_agent_name is not None and skill_names:
                return OrganizationEdit(
                    edit_type=OrganizationEditType.ASSIGN_SKILL,
                    rationale=(
                        "No capability-cluster distribution deviation; assign "
                        f"skills to {gap.best_agent_name}. {metrics_txt}."
                    ),
                    target_agent=gap.best_agent_name,
                    assigned_skill_names=skill_names,
                    assignment_gap=gap.assignment_gap,
                )
            return OrganizationEdit(
                edit_type=OrganizationEditType.DO_NOTHING,
                rationale=(
                    "No capability-cluster distribution deviation and no "
                    f"assignable agent. {metrics_txt}."
                ),
                assigned_skill_names=skill_names,
                assignment_gap=gap.assignment_gap,
            )

        # Uncovered capabilities are organizationally novel even when the
        # Executor remains the gap winner; do not require a large gap to ADD.
        effective_min_gap = (
            0.0
            if uncovered or empty_bank
            else self.policy.min_assignment_gap
        )
        if gap.assignment_gap < effective_min_gap:
            if gap.best_agent_name is not None and skill_names:
                return OrganizationEdit(
                    edit_type=OrganizationEditType.ASSIGN_SKILL,
                    rationale=(
                        "Capability is novel but the existing organization "
                        f"covers it sufficiently (assignment_gap="
                        f"{gap.assignment_gap:.3f} < "
                        f"{effective_min_gap:.3f}); assign to "
                        f"{gap.best_agent_name}. {metrics_txt}."
                    ),
                    target_agent=gap.best_agent_name,
                    assigned_skill_names=skill_names,
                    assignment_gap=gap.assignment_gap,
                )
            return OrganizationEdit(
                edit_type=OrganizationEditType.DO_NOTHING,
                rationale=(
                    "Capability is novel but assignment gap is below the "
                    f"ADD_AGENT threshold. {metrics_txt}."
                ),
                assigned_skill_names=skill_names,
                assignment_gap=gap.assignment_gap,
            )

        readiness = self._skills_ready_for_add_agent(skills)
        if readiness is not None:
            if gap.best_agent_name is not None and skill_names:
                return OrganizationEdit(
                    edit_type=OrganizationEditType.ASSIGN_SKILL,
                    rationale=(
                        f"ADD_AGENT blocked ({readiness}); assign skills to "
                        f"{gap.best_agent_name} instead. {metrics_txt}."
                    ),
                    target_agent=gap.best_agent_name,
                    assigned_skill_names=skill_names,
                    assignment_gap=gap.assignment_gap,
                )
            return OrganizationEdit(
                edit_type=OrganizationEditType.DO_NOTHING,
                rationale=(
                    f"ADD_AGENT blocked ({readiness}). {metrics_txt}."
                ),
                assigned_skill_names=skill_names,
                assignment_gap=gap.assignment_gap,
            )

        # Prefer ASSIGN only onto an existing *specialist* carrier. Parking a
        # new capability on the sole Executor is not SAGE restructuring.
        if (
            self.policy.prefer_assign_before_add
            and not uncovered
            and self._cluster_has_carrier(key, existing)
            and gap.best_agent_name is not None
            and skill_names
        ):
            return OrganizationEdit(
                edit_type=OrganizationEditType.ASSIGN_SKILL,
                rationale=(
                    "Executable skills prefer ASSIGN before ADD_AGENT when a "
                    f"specialist carrier already covers the capability "
                    f"(assign to {gap.best_agent_name}). {metrics_txt}."
                ),
                target_agent=gap.best_agent_name,
                assigned_skill_names=skill_names,
                assignment_gap=gap.assignment_gap,
            )

        probe = self._build_agent(
            self._spawn_skills_for_agent(skills),
            names,
            executor_name=executor_name,
            dispatch_only=self.policy.dispatch_only_new_agents,
            trial_games=self.policy.probation_games,
        )
        same_role = self._find_compatible_agent(probe, existing)
        if same_role is not None:
            return OrganizationEdit(
                edit_type=OrganizationEditType.ASSIGN_SKILL,
                rationale=(
                    f"Cluster deviated, but role `{probe.role}` already exists "
                    f"as {same_role.name}; assign skills instead of adding a "
                    f"duplicate agent. {metrics_txt}."
                ),
                target_agent=same_role.name,
                assigned_skill_names=skill_names,
                assignment_gap=gap.assignment_gap,
            )

        if empty_bank:
            slots = (
                self.policy.max_bootstrap_agents_per_round
                if bootstrap_slots_remaining is None
                else int(bootstrap_slots_remaining)
            )
            if slots <= 0:
                if gap.best_agent_name is not None and skill_names:
                    return OrganizationEdit(
                        edit_type=OrganizationEditType.ASSIGN_SKILL,
                        rationale=(
                            "Empty-bank bootstrap quota exhausted; assign "
                            f"skills to {gap.best_agent_name}. {metrics_txt}."
                        ),
                        target_agent=gap.best_agent_name,
                        assigned_skill_names=skill_names,
                        assignment_gap=gap.assignment_gap,
                    )
                return OrganizationEdit(
                    edit_type=OrganizationEditType.DO_NOTHING,
                    rationale=(
                        "Empty-bank bootstrap quota exhausted and no "
                        f"assignable agent. {metrics_txt}."
                    ),
                    assigned_skill_names=skill_names,
                    assignment_gap=gap.assignment_gap,
                )

        add_slots = (
            self.policy.max_new_agents_per_round
            if add_slots_remaining is None
            else int(add_slots_remaining)
        )
        if add_slots <= 0:
            if gap.best_agent_name is not None and skill_names:
                return OrganizationEdit(
                    edit_type=OrganizationEditType.ASSIGN_SKILL,
                    rationale=(
                        "ADD_AGENT quota exhausted this round; assign skills "
                        f"to {gap.best_agent_name}. {metrics_txt}."
                    ),
                    target_agent=gap.best_agent_name,
                    assigned_skill_names=skill_names,
                    assignment_gap=gap.assignment_gap,
                )
            return OrganizationEdit(
                edit_type=OrganizationEditType.DO_NOTHING,
                rationale=(
                    "ADD_AGENT quota exhausted this round and no assignable "
                    f"agent. {metrics_txt}."
                ),
                assigned_skill_names=skill_names,
                assignment_gap=gap.assignment_gap,
            )

        trigger = (
            "empty SkillBank baseline bootstrap"
            if empty_bank
            else (
                f"{metric} novelty {shift_score:.3f}"
            )
        )
        return OrganizationEdit(
            edit_type=OrganizationEditType.ADD_AGENT,
            rationale=(
                f"Capability-cluster deviation ({trigger}) across "
                f"{total_support} supporting trajectories. {metrics_txt}."
            ),
            new_agent=probe,
            assigned_skill_names=skill_names,
            assignment_gap=gap.assignment_gap,
        )

    @staticmethod
    def _find_compatible_agent(
        probe: AgentSpec,
        existing_agents: list[AgentSpec],
    ) -> AgentSpec | None:
        probe_role = (probe.role or probe.name).lower()
        for agent in existing_agents:
            if "executor" in f"{agent.name} {agent.role}".lower():
                continue
            agent_role = (agent.role or agent.name).lower()
            if agent_role == probe_role:
                return agent
            if probe_role and probe_role in agent.name.lower():
                return agent
        return None

    @classmethod
    def _build_agent(
        cls,
        skills: list[Skill],
        existing_agent_names: set[str],
        *,
        executor_name: str = "Executor",
        dispatch_only: bool = True,
        trial_games: int = 3,
    ) -> AgentSpec:
        contract = compile_capability_contract(skills)
        return AgentRoleCompiler().compile(
            contract,
            existing_agent_names,
            executor_name=executor_name,
            dispatch_only=dispatch_only,
            trial_games=trial_games,
            skills=skills,
        )

    def _spawn_skills_for_agent(self, skills: list[Skill]) -> list[Skill]:
        """Prefer Executable-only skills when dual-bank ADD gate is on."""
        if not self.policy.require_executable_bank_for_add_agent:
            return list(skills)
        from sage_mas.executable_coverage import skill_is_student_executable

        exec_only = [
            skill
            for skill in skills
            if skill_is_student_executable(
                skill,
                min_marginal_utility=float(
                    self.policy.min_marginal_utility_for_add_agent
                ),
            )
        ]
        return exec_only or list(skills)

    def _skills_ready_for_add_agent(self, skills: list[Skill]) -> str | None:
        """Return a block reason, or None when skills may spawn a specialist."""
        from sage_mas.executable_protocol import ensure_executable_protocol
        from sage_mas.skill_injection_policy import skill_has_positive_mu

        if not skills:
            return "empty skill cluster"
        for skill in skills:
            if skill.metadata.get("protocol_form_ok") is False:
                return (
                    f"skill `{skill.skill_name}` failed protocol form/noise gates"
                )
            reject = str(skill.metadata.get("protocol_quality_reject") or "").strip()
            if reject and skill.status == SkillStatus.REJECTED:
                return f"skill `{skill.skill_name}` rejected: {reject}"
        if self.policy.require_executable_protocol:
            for skill in skills:
                steps = ensure_executable_protocol(skill)
                if not steps:
                    return (
                        f"skill `{skill.skill_name}` lacks trajectory-derived "
                        "executable protocol steps"
                    )
                source = str(
                    skill.metadata.get("executable_protocol_source") or ""
                )
                if (
                    self.policy.require_trajectory_executable_protocol
                    and source not in {"trajectory", "confirmed_transitions"}
                ):
                    return (
                        f"skill `{skill.skill_name}` executable protocol is "
                        f"not trajectory-grounded (source={source or 'missing'})"
                    )
        if self.policy.require_positive_mu_for_add_agent:
            mu_subjects = list(skills)
            if self.policy.require_executable_bank_for_add_agent:
                from sage_mas.skill_bank_roles import ROLE_EXECUTABLE, skill_bank_role

                # Canonical siblings in the same cluster need not carry MU.
                mu_subjects = [
                    skill
                    for skill in skills
                    if skill_bank_role(skill) == ROLE_EXECUTABLE
                ] or list(skills)
            for skill in mu_subjects:
                if not skill_has_positive_mu(
                    skill,
                    min_marginal_utility=float(
                        self.policy.min_marginal_utility_for_add_agent
                    ),
                ):
                    mu = skill.marginal_utility
                    if mu is None and skill.metadata.get("marginal_utility") is not None:
                        try:
                            mu = float(skill.metadata["marginal_utility"])
                        except (TypeError, ValueError):
                            mu = None
                    return (
                        f"skill `{skill.skill_name}` lacks positive paired MU "
                        f"(mu={mu}); prove Executor+skill gain before ADD_AGENT"
                    )
        if self.policy.require_executable_bank_for_add_agent:
            from sage_mas.executable_coverage import (
                cluster_ready_for_agent_nomination,
                skill_is_student_executable,
            )

            ready, coverage = cluster_ready_for_agent_nomination(
                skills,
                min_rho=float(self.policy.min_executable_coverage_for_add_agent),
                min_marginal_utility=float(
                    self.policy.min_marginal_utility_for_add_agent
                ),
            )
            if not ready:
                return (
                    "cluster executable coverage too low for ADD_AGENT "
                    f"(rho_mini={coverage.get('rho_mini')}, "
                    f"n_executable={coverage.get('n_executable')}, "
                    f"min_rho={coverage.get('min_rho')}); "
                    "Canonical-only clusters nominate roles but cannot spawn agents"
                )
            # Spawn from Executable members only; Canonical siblings may remain.
            exec_only = [
                skill
                for skill in skills
                if skill_is_student_executable(
                    skill,
                    min_marginal_utility=float(
                        self.policy.min_marginal_utility_for_add_agent
                    ),
                )
            ]
            if not exec_only:
                return "no Executable-bank skills in cluster for ADD_AGENT"
        min_adherence = float(self.policy.min_protocol_adherence_for_add_agent)
        if min_adherence <= 0:
            return None
        adherences: list[float] = []
        for skill in skills:
            observed = self._observed_protocol_adherence(skill)
            if observed is not None:
                adherences.append(observed)
        if not adherences:
            # No online adherence evidence yet: if MU already proved the skill,
            # do not block ADD_AGENT on missing adherence. Otherwise bootstrap
            # only when explicitly allowed.
            if self.policy.require_positive_mu_for_add_agent:
                return None
            return None
        mean_adherence = sum(adherences) / len(adherences)
        if mean_adherence < min_adherence:
            return (
                f"mean protocol adherence {mean_adherence:.3f} < "
                f"{min_adherence:.3f}"
            )
        return None

    @staticmethod
    def _observed_protocol_adherence(skill: Skill) -> float | None:
        """Return measured adherence, or None when there is no evidence yet."""
        credit = skill.metadata.get("skill_credit") or {}
        if not isinstance(credit, dict):
            return None
        uses = int(credit.get("uses") or 0)
        scores = credit.get("adherence_scores") or []
        if uses <= 0 and not scores:
            # Unused skills often carry mean_protocol_adherence=0.0 as a
            # placeholder; that is not a failed measurement.
            return None
        if scores:
            return sum(float(value) for value in scores) / len(scores)
        raw = credit.get("mean_protocol_adherence")
        if raw is None:
            return None
        return float(raw)


class OrganizationStateManager:
    """Materialize candidate organizations without mutating the active state."""

    @staticmethod
    def _executor_agent(agents: list[AgentSpec]) -> AgentSpec | None:
        for agent in agents:
            if "executor" in f"{agent.name} {agent.role}".lower():
                return agent
        return None

    @classmethod
    def _transfer_executor_skill_ownership(
        cls,
        agents: list[AgentSpec],
        skill_names: list[str],
        *,
        owner: AgentSpec | None = None,
    ) -> None:
        """Move owned specialist skills off Executor for exclusive dispatch."""
        if not skill_names:
            return
        executor = cls._executor_agent(agents)
        if executor is None:
            return
        if owner is not None and executor.agent_id == owner.agent_id:
            return
        skill_set = set(skill_names)
        executor.assigned_skills = sorted(
            name for name in (executor.assigned_skills or []) if name not in skill_set
        )

    @classmethod
    def apply_candidate(
        cls,
        active_agents: list[AgentSpec],
        edits: list[OrganizationEdit],
        skills: list[Skill] | None = None,
    ) -> list[AgentSpec]:
        candidate_agents = deepcopy(active_agents)
        for edit in edits:
            if edit.edit_type == OrganizationEditType.ASSIGN_SKILL:
                target = next(
                    (
                        agent
                        for agent in candidate_agents
                        if agent.name == edit.target_agent
                    ),
                    None,
                )
                if target is None:
                    raise ValueError(f"Unknown target agent: {edit.target_agent}")
                target.assigned_skills = sorted(
                    set(target.assigned_skills + edit.assigned_skill_names)
                )
                cls._transfer_executor_skill_ownership(
                    candidate_agents,
                    edit.assigned_skill_names,
                    owner=target,
                )
            elif edit.edit_type == OrganizationEditType.ADD_AGENT:
                if edit.new_agent is None:
                    raise ValueError("add_agent edit requires new_agent")
                if any(
                    agent.name == edit.new_agent.name
                    for agent in candidate_agents
                ):
                    raise ValueError(
                        f"Agent name already exists: {edit.new_agent.name}"
                    )
                new_agent = deepcopy(edit.new_agent)
                candidate_agents.append(new_agent)
                skill_names = list(edit.assigned_skill_names) or list(
                    new_agent.assigned_skills
                )
                cls._transfer_executor_skill_ownership(
                    candidate_agents,
                    skill_names,
                    owner=new_agent,
                )
        if skills:
            for agent in candidate_agents:
                sync_agent_skill_scopes(agent, skills)
        return candidate_agents
