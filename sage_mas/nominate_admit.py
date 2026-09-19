"""Nominate-then-Admit organization gate (independent of online_evolution).

This module is a **parallel** code path. It does not patch
``sage_mas.online_evolution`` or ``sage_mas.pipeline``. Callers that still
use injection-MU-before-ADD keep the old path; this path encodes a different
SAGE gate:

=============================================================================
Stage A — Distill / bank (usually already done by the caller)
=============================================================================
  form_ok skill protocols enter the SkillBank as **provisional**.
  Provisional means "structurally valid enough to consider", not "proven useful".

=============================================================================
Stage B — Nomination (organization gap, NOT effect proof)
=============================================================================
  Question answered:
      Does the current organization lack a specialist carrier for this
      capability cluster (distribution shift and/or uncovered capability)?

  Evidence allowed:
      - capability-cluster distribution shift vs historical bank
      - uncovered capability (only Executor owns the cluster)
      - assignment gap / empty-bank bootstrap quotas

  Evidence **forbidden** as an ADD driver:
      - injection MU  (Executor+skill vs bare Executor)
      - auto-promoting form_ok → verified without a specialist probe

  Side effects of a successful nomination:
      - propose ADD_AGENT
      - commit a **probation** specialist (dispatch_only by default)
      - skills remain **provisional** (NOT verified yet)

=============================================================================
Stage C — Admission (effect proof = Specialist vs bare Executor)
=============================================================================
  Question answered:
      On held-out tasks in the specialist's scope, does the specialist-as-
      primary beat bare Executor-as-primary by ``min_advantage``?

  Evidence allowed:
      - Spec vs Exec paired SR (see ``sage_mas.actor_promotion``)

  Evidence **forbidden** as the admission criterion:
      - injection MU  (that only answers "inject this skill into Executor?")

  Side effects:
      - PASS → mark related skills **verified**; keep specialist
        (probation or accepted per config)
      - FAIL → demote / dormant / remove specialist; skills stay provisional

Why Spec vs Exec (not injection MU) for ADD?
  Injection MU measures whether a *protocol text* helps the *same* Executor
  actor. ADD_AGENT introduces a *new actor*. The right counterfactual is
  therefore "new specialist primary vs Executor primary", matching SAGE's
  organizational innovation (shift nominates; Spec vs Exec admits).
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from sage_mas.actor_promotion import (
    ActorPromotionResult,
    apply_actor_promotion_result,
    probe_actor_promotion,
    promotion_beats_executor,
    select_promotion_gamefiles,
)
from sage_mas.alfworld_evaluator import AlfWorldOrganizationEvaluator
from sage_mas.assignment_gap import AssignmentGapEstimator
from sage_mas.distribution_stats import (
    aggregate_skill_bank_distribution,
    shift_for_skill_cluster,
)
from sage_mas.onboarding import acting_status, restore_executor_baseline
from sage_mas.organization import (
    OrganizationEditor,
    OrganizationPolicy,
    OrganizationStateManager,
    cluster_key_for_skill,
    unique_evidence_support,
)
from sage_mas.schemas import (
    AgentSpec,
    OrganizationEdit,
    OrganizationEditType,
    Skill,
    SkillStatus,
)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NominationPolicy:
    """Organization-gap nomination knobs.

    Intentionally orthogonal to injection MU. Even if every provisional skill
    carries a large positive ``marginal_utility``, that alone must **not**
    force ADD_AGENT — only shift / uncovered / gap logic may nominate.
    """

    distribution_shift_threshold: float = 0.15
    min_cluster_support: int = 1
    min_assignment_gap: float = 0.1
    allow_bootstrap_add: bool = True
    max_bootstrap_agents_per_round: int = 1
    max_new_agents_per_round: int = 1
    cluster_by: str = "capability"
    # Core of this redesign: missing specialist carrier is itself a nomination
    # signal, independent of histogram KL vs an already-populated bank.
    add_when_uncovered_capability: bool = True
    dispatch_only_new_agents: bool = True
    probation_games: int = 8
    prefer_assign_before_add: bool = True
    # Keep trajectory-derived protocol quality, but never require MU.
    require_executable_protocol: bool = True
    require_trajectory_executable_protocol: bool = False
    require_positive_mu_for_add_agent: bool = False
    min_marginal_utility_for_add_agent: float = 0.0
    require_executable_bank_for_add_agent: bool = False
    min_executable_coverage_for_add_agent: float = 0.0
    # Accept provisional / form_ok skills as nomination subjects. The legacy
    # OrganizationEditor still demands VERIFIED; we satisfy that with
    # **ephemeral clones** only (see ``_nomination_clones``).
    allow_provisional_nomination: bool = True

    def to_organization_policy(self) -> OrganizationPolicy:
        """Map onto OrganizationEditor without enabling MU-for-ADD."""
        return OrganizationPolicy(
            distribution_shift_threshold=self.distribution_shift_threshold,
            min_cluster_support=self.min_cluster_support,
            min_assignment_gap=self.min_assignment_gap,
            allow_bootstrap_add=self.allow_bootstrap_add,
            max_bootstrap_agents_per_round=self.max_bootstrap_agents_per_round,
            max_new_agents_per_round=self.max_new_agents_per_round,
            cluster_by=self.cluster_by,
            add_when_uncovered_capability=self.add_when_uncovered_capability,
            dispatch_only_new_agents=self.dispatch_only_new_agents,
            probation_games=self.probation_games,
            require_executable_protocol=self.require_executable_protocol,
            require_trajectory_executable_protocol=(
                self.require_trajectory_executable_protocol
            ),
            # Hard rule for this path: injection MU never gates ADD.
            require_positive_mu_for_add_agent=False,
            min_marginal_utility_for_add_agent=0.0,
            require_executable_bank_for_add_agent=(
                self.require_executable_bank_for_add_agent
            ),
            min_executable_coverage_for_add_agent=(
                self.min_executable_coverage_for_add_agent
            ),
            prefer_assign_before_add=self.prefer_assign_before_add,
        )


@dataclass(slots=True)
class AdmissionPolicy:
    """Spec vs bare Exec admission knobs.

    Cold-start aligned default (``accept_ties=True``): admit when
    ``specialist_sr - executor_sr >= min_advantage`` (ΔSR≥0 ties pass), matching
    ``cold_start.enable_table_from_eval_summary(min_delta_sr=0)``.

    Set ``accept_ties=False`` for the stricter online actor-promotion rule
    (ties never pass when ``min_advantage==0``). Injection MU is intentionally
    absent from this dataclass.
    """

    min_advantage: float = 0.0
    # Cold-start bundle used ΔSR >= 0 (ties enable the specialist).
    accept_ties: bool = True
    num_tasks: int = 8
    # After a PASS, keep the specialist in online probation by default so a
    # short held-out probe cannot permanently steal primary control.
    on_pass_acting_status: str = "probation"
    online_trial_games: int = 3
    verify_skills_on_pass: bool = True
    # FAIL handling for the newly nominated specialist.
    #   demote  → acting_status=probation, dispatch_only=True (retry later)
    #   dormant → acting_status=dormant, dispatch_only=True
    #   remove  → drop agent from roster
    on_fail_action: str = "remove"
    remove_after_rejected_windows: int = 1


@dataclass(slots=True)
class NominateAdmitConfig:
    nomination: NominationPolicy = field(default_factory=NominationPolicy)
    admission: AdmissionPolicy = field(default_factory=AdmissionPolicy)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClusterNomination:
    cluster_key: str
    skill_names: list[str]
    edit_type: str
    rationale: str
    assignment_gap: float | None
    shift_score: float | None
    uncovered: bool
    new_agent_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdmissionDecision:
    agent_name: str
    accepted: bool
    specialist_sr: float
    executor_sr: float
    n_tasks: int
    reason: str
    verified_skill_names: list[str]
    new_acting_status: str | None
    removed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NominateAdmitReport:
    """Serializable end-to-end gate report for a single round."""

    nominations: list[ClusterNomination] = field(default_factory=list)
    committed_adds: list[str] = field(default_factory=list)
    admissions: list[AdmissionDecision] = field(default_factory=list)
    skipped_admission: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nominations": [item.as_dict() for item in self.nominations],
            "committed_adds": list(self.committed_adds),
            "admissions": [item.as_dict() for item in self.admissions],
            "skipped_admission": list(self.skipped_admission),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Eligibility helpers
# ---------------------------------------------------------------------------


def skill_is_nomination_eligible(
    skill: Skill,
    *,
    allow_provisional: bool = True,
) -> bool:
    """Return True when a skill may be considered for ADD nomination.

    Requires structural / form readiness only. Explicitly does **not** inspect
    ``marginal_utility``, ``inject_ready``, or injection-MU probe metadata.
    """
    if skill.status in {SkillStatus.REJECTED, SkillStatus.RETIRED}:
        return False
    if skill.metadata.get("protocol_form_ok") is False:
        return False
    reject = str(skill.metadata.get("protocol_quality_reject") or "").strip()
    if reject and skill.status == SkillStatus.REJECTED:
        return False
    if skill.status == SkillStatus.VERIFIED:
        return True
    if allow_provisional and skill.status in {
        SkillStatus.PROVISIONAL,
        SkillStatus.CANDIDATE,
    }:
        return True
    return False


def _nomination_clones(skills: Sequence[Skill]) -> list[Skill]:
    """Clone skills and mark clones VERIFIED for OrganizationEditor only.

    OrganizationEditor historically refuses non-verified clusters. In the
    nominate-then-admit design, provisional form_ok skills are allowed to
    *nominate* ADD, but verification is deferred until Spec vs Exec passes.

    These clones are never written back into the SkillBank. Real skill objects
    stay provisional until ``apply_admission`` promotes them.
    """
    clones: list[Skill] = []
    for skill in skills:
        clone = deepcopy(skill)
        # Ephemeral status for the editor gate only.
        clone.status = SkillStatus.VERIFIED
        # Make sure leftover MU flags cannot re-enter via other helpers.
        meta = dict(clone.metadata or {})
        meta["nominate_admit_ephemeral_verified"] = True
        meta["nominate_admit_source_status"] = skill.status.value
        clone.metadata = meta
        clones.append(clone)
    return clones


def cluster_skills(
    skills: Sequence[Skill],
    *,
    cluster_by: str = "capability",
) -> list[list[Skill]]:
    buckets: dict[str, list[Skill]] = defaultdict(list)
    for skill in skills:
        key = cluster_key_for_skill(skill, cluster_by=cluster_by)
        buckets[key].append(skill)
    return list(buckets.values())


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class NominateAdmitController:
    """Run nomination → probation ADD → Spec vs Exec admission.

    Designed so a future online loop can call ``run_round`` without modifying
    ``OnlineEvolutionEngine``. Unit tests can call stages in isolation without
    an environment evaluator.
    """

    def __init__(
        self,
        config: NominateAdmitConfig | None = None,
        *,
        gap_estimator: AssignmentGapEstimator | None = None,
    ):
        self.config = config or NominateAdmitConfig()
        self.gap_estimator = gap_estimator or AssignmentGapEstimator()
        self.org_editor = OrganizationEditor(
            self.config.nomination.to_organization_policy()
        )

    # ---- Stage B: nominate -------------------------------------------------

    def nominate(
        self,
        *,
        agents: list[AgentSpec],
        skills: list[Skill],
        bank_before: Sequence[Skill] | None = None,
    ) -> list[OrganizationEdit]:
        """Propose org edits from provisional/form_ok clusters.

        Does not mutate ``skills`` statuses. Does not consult injection MU.
        """
        nom = self.config.nomination
        eligible = [
            skill
            for skill in skills
            if skill_is_nomination_eligible(
                skill,
                allow_provisional=nom.allow_provisional_nomination,
            )
        ]
        clusters = cluster_skills(eligible, cluster_by=nom.cluster_by)
        clusters = sorted(
            clusters,
            key=lambda cluster: (
                -unique_evidence_support(cluster),
                cluster_key_for_skill(cluster[0], cluster_by=nom.cluster_by)
                if cluster
                else "",
            ),
        )

        edits: list[OrganizationEdit] = []
        evolving = list(agents)
        names = {agent.name for agent in evolving}
        executor_name = next(
            (
                agent.name
                for agent in evolving
                if "executor" in f"{agent.name} {agent.role}".lower()
            ),
            evolving[0].name if evolving else "Executor",
        )
        bootstrap_budget = nom.max_bootstrap_agents_per_round
        add_budget = nom.max_new_agents_per_round
        historical = list(bank_before) if bank_before is not None else []

        for cluster in clusters:
            key = cluster_key_for_skill(cluster[0], cluster_by=nom.cluster_by)
            cluster_ids = {skill.skill_id for skill in cluster}
            if not historical:
                cluster_shift = shift_for_skill_cluster(
                    cluster,
                    None,
                    baseline_skills=[],
                )
            else:
                baseline = [
                    skill
                    for skill in historical
                    if skill.skill_id not in cluster_ids
                ] or list(historical)
                cluster_shift = shift_for_skill_cluster(
                    cluster,
                    aggregate_skill_bank_distribution(baseline),
                    baseline_skills=baseline,
                )
            # Attach shift on real skill objects for audit only; status unchanged.
            for skill in cluster:
                skill.distribution_shift = cluster_shift

            gap = self.gap_estimator.estimate(cluster, evolving)
            # Ephemeral verified clones satisfy OrganizationEditor without
            # promoting the real bank entries.
            edit = self.org_editor.propose(
                _nomination_clones(cluster),
                gap,
                existing_agent_names=names,
                executor_name=executor_name,
                existing_agents=evolving,
                cluster_key=key,
                cluster_shift=cluster_shift,
                bootstrap_slots_remaining=bootstrap_budget,
                add_slots_remaining=add_budget,
            )
            # Rewrite rationale to make the gate semantics explicit in logs.
            if edit.edit_type == OrganizationEditType.ADD_AGENT:
                edit.rationale = (
                    "[nominate_admit:nomination] "
                    + edit.rationale
                    + " Skills remain provisional until Spec vs Exec admission."
                )
            edits.append(edit)

            if (
                edit.edit_type == OrganizationEditType.ADD_AGENT
                and edit.new_agent is not None
            ):
                names.add(edit.new_agent.name)
                evolving = list(evolving) + [edit.new_agent]
                add_budget = max(0, add_budget - 1)
                if cluster_shift.baseline_source == "empty_skill_bank":
                    bootstrap_budget = max(0, bootstrap_budget - 1)
        return edits

    def summarize_nominations(
        self,
        edits: Sequence[OrganizationEdit],
        *,
        skills: Sequence[Skill],
        agents: Sequence[AgentSpec],
    ) -> list[ClusterNomination]:
        by_name = {skill.skill_name: skill for skill in skills}
        summaries: list[ClusterNomination] = []
        for edit in edits:
            named = [by_name[name] for name in edit.assigned_skill_names if name in by_name]
            key = (
                cluster_key_for_skill(
                    named[0],
                    cluster_by=self.config.nomination.cluster_by,
                )
                if named
                else "empty"
            )
            shift = named[0].distribution_shift if named else None
            uncovered = bool(
                self.config.nomination.add_when_uncovered_capability
                and not self.org_editor._cluster_has_carrier(key, list(agents))
            )
            summaries.append(
                ClusterNomination(
                    cluster_key=key,
                    skill_names=list(edit.assigned_skill_names),
                    edit_type=edit.edit_type.value,
                    rationale=edit.rationale,
                    assignment_gap=edit.assignment_gap,
                    shift_score=(
                        float(shift.shift_score) if shift is not None else None
                    ),
                    uncovered=uncovered,
                    new_agent_name=(
                        edit.new_agent.name if edit.new_agent is not None else None
                    ),
                )
            )
        return summaries

    # ---- Stage B commit: probation ADD ------------------------------------

    def commit_probation_add(
        self,
        agents: list[AgentSpec],
        edits: Sequence[OrganizationEdit],
        skills: list[Skill],
    ) -> tuple[list[AgentSpec], list[str]]:
        """Apply ADD_AGENT edits as probation specialists; skills stay provisional.

        Returns ``(new_agents, committed_specialist_names)``.
        """
        add_edits = [
            edit
            for edit in edits
            if edit.edit_type == OrganizationEditType.ADD_AGENT
            and edit.new_agent is not None
        ]
        if not add_edits:
            return list(agents), []

        # Ensure new agents start in probation / dispatch_only regardless of
        # whatever the role compiler stamped.
        for edit in add_edits:
            assert edit.new_agent is not None
            record = dict(edit.new_agent.shadow_evaluation_record or {})
            record["acting_status"] = "probation"
            record["dispatch_only"] = bool(
                self.config.nomination.dispatch_only_new_agents
            )
            record["trial_games_remaining"] = int(
                self.config.nomination.probation_games
            )
            record["nominate_admit"] = {
                "stage": "probation_committed",
                "skills_still_provisional": True,
            }
            edit.new_agent.shadow_evaluation_record = record

        candidate = OrganizationStateManager.apply_candidate(
            agents,
            list(add_edits),
            skills=skills,
        )
        committed = [edit.new_agent.name for edit in add_edits if edit.new_agent]
        # Defensive: never verify skills at commit time.
        for skill in skills:
            if skill.skill_name in {
                name
                for edit in add_edits
                for name in edit.assigned_skill_names
            }:
                if skill.status == SkillStatus.VERIFIED:
                    # Caller may have mixed banks; leave verified alone, but
                    # provisional must stay provisional here.
                    continue
                if skill.status not in {
                    SkillStatus.REJECTED,
                    SkillStatus.RETIRED,
                }:
                    skill.status = SkillStatus.PROVISIONAL
                    meta = dict(skill.metadata or {})
                    meta["nominate_admit_awaiting_admission"] = True
                    skill.metadata = meta
        return candidate, committed

    # ---- Stage C: admit ---------------------------------------------------

    def admit_specialist(
        self,
        evaluator: AlfWorldOrganizationEvaluator,
        *,
        agents: list[AgentSpec],
        skills: list[Skill],
        specialist: AgentSpec,
        gamefile_pool: Sequence[str],
    ) -> ActorPromotionResult:
        """Paired Spec-as-primary vs Exec-as-primary on held-out tasks."""
        adm = self.config.admission
        gamefiles = select_promotion_gamefiles(
            gamefile_pool,
            specialist,
            max_tasks=adm.num_tasks,
        )
        # Probe still runs the paired Spec/Exec episodes. Acceptance uses the
        # nominate-admit policy (cold-start ΔSR≥0 by default), not the stricter
        # online ``promotion_beats_executor`` tie-break baked into the probe.
        result = probe_actor_promotion(
            evaluator,
            agents=agents,
            skills=skills,
            specialist=specialist,
            gamefiles=gamefiles,
            min_advantage=adm.min_advantage,
            # Spec+skill vs bare mid-model Exec (cold-start enable contrast).
            bare_executor=True,
        )
        accepted = admission_would_pass(
            result.specialist_sr,
            result.executor_sr,
            min_advantage=adm.min_advantage,
            accept_ties=adm.accept_ties,
        )
        result.accepted = bool(accepted)
        delta = float(result.specialist_sr) - float(result.executor_sr)
        tie_note = "accept_ties=True (cold-start ΔSR>=min)" if adm.accept_ties else "accept_ties=False"
        if accepted:
            result.reason = (
                f"promote: specialist {result.specialist_wins}/{result.n_tasks} vs "
                f"Executor {result.executor_wins}/{result.n_tasks} "
                f"(delta={delta:+.3f}, min_advantage={adm.min_advantage}, {tie_note})"
            )
        else:
            result.reason = (
                f"keep Executor primary: specialist {result.specialist_wins}/"
                f"{result.n_tasks} vs Executor {result.executor_wins}/{result.n_tasks} "
                f"(delta={delta:+.3f}, min_advantage={adm.min_advantage}, {tie_note})"
            )
        return result

    def apply_admission(
        self,
        *,
        agents: list[AgentSpec],
        skills: list[Skill],
        specialist_name: str,
        result: ActorPromotionResult,
        related_skill_names: Sequence[str] | None = None,
    ) -> AdmissionDecision:
        """Mutate org/skills from a Spec vs Exec admission result.

        PASS → verify related skills; keep specialist.
        FAIL → demote/dormant/remove; skills remain provisional.
        """
        adm = self.config.admission
        specialist = next(
            (agent for agent in agents if agent.name == specialist_name),
            None,
        )
        if specialist is None:
            return AdmissionDecision(
                agent_name=specialist_name,
                accepted=False,
                specialist_sr=result.specialist_sr,
                executor_sr=result.executor_sr,
                n_tasks=result.n_tasks,
                reason=f"specialist missing from roster: {result.reason}",
                verified_skill_names=[],
                new_acting_status=None,
                removed=False,
            )

        related = list(related_skill_names or specialist.assigned_skills or [])
        verified_names: list[str] = []
        removed = False
        new_status: str | None = None

        if result.accepted:
            # --- PASS: effect proof succeeded ---
            if adm.verify_skills_on_pass:
                for skill in skills:
                    if skill.skill_name not in related:
                        continue
                    if skill.status in {
                        SkillStatus.REJECTED,
                        SkillStatus.RETIRED,
                    }:
                        continue
                    skill.status = SkillStatus.VERIFIED
                    meta = dict(skill.metadata or {})
                    meta["nominate_admit_awaiting_admission"] = False
                    meta["nominate_admit_verified_by"] = "spec_vs_exec"
                    meta["nominate_admit_admission"] = result.as_dict()
                    skill.metadata = meta
                    verified_names.append(skill.skill_name)

            record = dict(specialist.shadow_evaluation_record or {})
            record["last_nominate_admit"] = result.as_dict()
            record["promotion_probe_passed"] = True
            desired = str(adm.on_pass_acting_status or "probation").strip().lower()
            if desired == "accepted":
                record["acting_status"] = "accepted"
                record["dispatch_only"] = False
                record["trial_games_remaining"] = 0
                record["actor_promoted"] = True
            else:
                # Default: online probation after held-out pass.
                apply_actor_promotion_result(
                    specialist,
                    result,
                    remove_after_rejected_windows=adm.remove_after_rejected_windows,
                    online_trial_games=adm.online_trial_games,
                )
                record = dict(specialist.shadow_evaluation_record or {})
                record["last_nominate_admit"] = result.as_dict()
            specialist.shadow_evaluation_record = record
            new_status = acting_status(specialist)
            return AdmissionDecision(
                agent_name=specialist_name,
                accepted=True,
                specialist_sr=result.specialist_sr,
                executor_sr=result.executor_sr,
                n_tasks=result.n_tasks,
                reason=result.reason,
                verified_skill_names=verified_names,
                new_acting_status=new_status,
                removed=False,
            )

        # --- FAIL: do not verify skills ---
        for skill in skills:
            if skill.skill_name not in related:
                continue
            if skill.status == SkillStatus.VERIFIED:
                # Do not demote previously verified skills from other rounds.
                continue
            if skill.status not in {SkillStatus.REJECTED, SkillStatus.RETIRED}:
                skill.status = SkillStatus.PROVISIONAL
            meta = dict(skill.metadata or {})
            meta["nominate_admit_awaiting_admission"] = False
            meta["nominate_admit_admission_failed"] = result.as_dict()
            skill.metadata = meta

        action = str(adm.on_fail_action or "remove").strip().lower()
        record = dict(specialist.shadow_evaluation_record or {})
        record["last_nominate_admit"] = result.as_dict()
        record["promotion_probe_passed"] = False
        record["actor_promoted"] = False

        if action == "remove":
            agents[:] = [agent for agent in agents if agent.name != specialist_name]
            removed = True
            new_status = "removed"
            restore_executor_baseline(agents)
        elif action == "dormant":
            record["acting_status"] = "dormant"
            record["dispatch_only"] = True
            record["trial_games_remaining"] = 0
            specialist.shadow_evaluation_record = record
            new_status = "dormant"
        else:
            # demote: stay on roster, dispatch_only, may retry later
            record["acting_status"] = "probation"
            record["dispatch_only"] = True
            record["trial_games_remaining"] = 0
            rejects = int(record.get("actor_promotion_rejects") or 0) + 1
            record["actor_promotion_rejects"] = rejects
            specialist.shadow_evaluation_record = record
            new_status = "probation"

        return AdmissionDecision(
            agent_name=specialist_name,
            accepted=False,
            specialist_sr=result.specialist_sr,
            executor_sr=result.executor_sr,
            n_tasks=result.n_tasks,
            reason=result.reason,
            verified_skill_names=[],
            new_acting_status=new_status,
            removed=removed,
        )

    # ---- End-to-end round --------------------------------------------------

    def run_round(
        self,
        *,
        agents: list[AgentSpec],
        skills: list[Skill],
        gamefile_pool: Sequence[str],
        evaluator: AlfWorldOrganizationEvaluator | None = None,
        bank_before: Sequence[Skill] | None = None,
        run_admission: bool = True,
    ) -> tuple[list[AgentSpec], NominateAdmitReport]:
        """Nominate → commit probation ADD → (optional) Spec vs Exec admit.

        When ``run_admission`` is True, ``evaluator`` is required.
        """
        report = NominateAdmitReport()
        report.notes.append(
            "Gate semantics: shift/uncovered nominates ADD; "
            "Spec vs bare Exec admits; injection MU is not used."
        )

        edits = self.nominate(
            agents=agents,
            skills=skills,
            bank_before=bank_before,
        )
        report.nominations = self.summarize_nominations(
            edits,
            skills=skills,
            agents=agents,
        )

        agents_after, committed = self.commit_probation_add(agents, edits, skills)
        report.committed_adds = list(committed)

        if not run_admission:
            report.notes.append("Admission skipped (run_admission=False).")
            return agents_after, report
        if evaluator is None:
            raise ValueError("evaluator is required when run_admission=True")
        if not committed:
            report.notes.append("No ADD_AGENT commits; nothing to admit.")
            return agents_after, report

        # Map specialist → skill names from the ADD edits.
        skill_map = {
            edit.new_agent.name: list(edit.assigned_skill_names)
            for edit in edits
            if edit.edit_type == OrganizationEditType.ADD_AGENT
            and edit.new_agent is not None
        }

        working = list(agents_after)
        for name in committed:
            specialist = next((a for a in working if a.name == name), None)
            if specialist is None:
                report.skipped_admission.append(name)
                continue
            print(
                f"[nominate_admit] phase=admission specialist={name} "
                f"(Spec vs Exec, env reset + up to {self.config.admission.num_tasks} games)...",
                flush=True,
            )
            result = self.admit_specialist(
                evaluator,
                agents=working,
                skills=skills,
                specialist=specialist,
                gamefile_pool=gamefile_pool,
            )
            decision = self.apply_admission(
                agents=working,
                skills=skills,
                specialist_name=name,
                result=result,
                related_skill_names=skill_map.get(name),
            )
            report.admissions.append(decision)
            print(
                f"[nominate_admit] admission {name}: "
                f"accepted={decision.accepted} "
                f"spec_sr={decision.specialist_sr:.2f} "
                f"exec_sr={decision.executor_sr:.2f} "
                f"n={decision.n_tasks}",
                flush=True,
            )

        return working, report


def admission_would_pass(
    specialist_sr: float,
    executor_sr: float,
    *,
    min_advantage: float = 0.0,
    accept_ties: bool = True,
) -> bool:
    """Cold-start-aligned Spec vs Exec gate (default), or strict promotion.

    With ``accept_ties=True`` (cold-start): pass iff
    ``specialist_sr - executor_sr >= min_advantage`` (ties pass at 0).
    With ``accept_ties=False``: defer to ``promotion_beats_executor``.
    """
    if accept_ties:
        return (
            float(specialist_sr) - float(executor_sr)
            >= float(min_advantage) - 1e-12
        )
    return promotion_beats_executor(
        specialist_sr,
        executor_sr,
        min_advantage=min_advantage,
    )


def config_from_mapping(raw: dict[str, Any] | None) -> NominateAdmitConfig:
    """Build config from a yaml ``nominate_admit:`` block."""
    raw = dict(raw or {})
    nom_raw = dict(raw.get("nomination") or {})
    adm_raw = dict(raw.get("admission") or {})
    nomination = NominationPolicy(
        **{
            key: value
            for key, value in nom_raw.items()
            if key in NominationPolicy.__dataclass_fields__
        }
    )
    admission = AdmissionPolicy(
        **{
            key: value
            for key, value in adm_raw.items()
            if key in AdmissionPolicy.__dataclass_fields__
        }
    )
    return NominateAdmitConfig(nomination=nomination, admission=admission)
