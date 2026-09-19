"""End-to-end offline SAGE-MAS evolution pipeline."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from sage_mas.assignment_gap import AssignmentGapEstimator
from sage_mas.distribution_stats import (
    aggregate_skill_bank_distribution,
    attach_distribution_shift,
    build_experience_distribution_snapshot,
    compute_round_exploration_distribution,
    compute_signal_cooccurrence,
    compute_skill_exploration_distribution,
    index_trajectory_steps,
    shift_for_skill_cluster,
)
from sage_mas.enriched_skill_distiller import EnrichedSkillDistiller
from sage_mas.organization import (
    OrganizationEditor,
    OrganizationPolicy,
    OrganizationStateManager,
    cluster_key_for_skill,
    unique_evidence_support,
)
from sage_mas.schemas import (
    AgentSpec,
    ExperienceDistributionSnapshot,
    OrganizationEditType,
    Skill,
    SkillStatus,
)
from sage_mas.serialization import (
    agent_from_dict,
    load_agents,
    load_skills,
    read_json,
    write_json,
)
from sage_mas.skill_bank import SkillBank
from sage_mas.skill_quality import (
    generic_skill_contract_reasons,
    validate_environment_confirmed_skill,
    validate_skill_evidence,
)
from sage_mas.skill_distiller import (
    DistillationConfig,
    HeuristicSkillDistiller,
    LLMSignalExtractor,
)
from sage_mas.skill_credit import (
    initialize_skill_credit,
    skill_credit_policy_from_mapping,
)
from sage_mas.skill_injection_policy import (
    distill_seed_is_excluded,
    filter_injectable_skills,
    injection_policy_from_mapping,
)
from sage_mas.skill_student_transfer import (
    adapt_skills_for_student,
    distill_contrastive_patches,
    summarize_student_failure_notes,
    transfer_policy_from_mapping,
)
from sage_mas.trajectory_adapter import build_trajectory_adapter
from sage_mas.trajectory.causal import protocol_is_org_ready

# Compatibility constant for callers that inspect feature availability. Actual
# execution remains controlled by ``sage.verification.enabled``.
SKILL_VERIFICATION_ACTIVE = True


@dataclass(slots=True)
class PipelineArtifacts:
    run_dir: str
    atomic_trajectories: str
    candidate_skills: str
    skill_bank: str
    assignment_gaps: str
    organization_edits: str
    active_organization: str
    candidate_organization: str
    experience_distribution_snapshot: str | None = None
    capability_contracts: str | None = None
    rejected_skills: str | None = None


class SageEvolutionPipeline:
    def __init__(
        self,
        config: dict[str, Any],
        distillation_backend: Any | None = None,
    ):
        self.config = config
        sage_cfg = config.get("sage", {})

        distill_cfg = sage_cfg.get("distillation", {})
        self.adapter = build_trajectory_adapter(
            sage_cfg.get("trajectory_adapter")
            or distill_cfg.get("trajectory_adapter")
            or "alfworld"
        )
        # LLM-driven signal extraction replaces hardcoded _repair_signal_matches
        # and _CAPABILITY_BY_SIGNAL when a distillation backend is available.
        signal_extractor = None
        if distillation_backend is not None:
            signal_extractor = LLMSignalExtractor(distillation_backend)
        seed_distiller = HeuristicSkillDistiller(
            DistillationConfig(
                high_cost_steps=int(distill_cfg.get("high_cost_steps", 25)),
                min_support=int(distill_cfg.get("min_support", 1)),
                operation_min_support=int(
                    distill_cfg.get("operation_min_support", 1)
                ),
                min_family_support=distill_cfg.get("min_family_support"),
            ),
            signal_extractor=signal_extractor,
        )
        distillation_mode = str(
            distill_cfg.get("mode", "trajectory_enriched")
        )
        use_llm_rewrite = bool(
            distill_cfg.get(
                "use_llm_rewrite",
                distillation_mode == "trajectory_grounded_llm",
            )
        )
        enriched_kwargs = {
            "config": DistillationConfig(
                high_cost_steps=int(distill_cfg.get("high_cost_steps", 25)),
                min_support=int(distill_cfg.get("min_support", 1)),
                operation_min_support=int(
                    distill_cfg.get("operation_min_support", 1)
                ),
                min_family_support=distill_cfg.get("min_family_support"),
            ),
            "backend": distillation_backend if use_llm_rewrite else None,
            "max_evidence_trajectories": int(
                distill_cfg.get("max_evidence_trajectories", 5)
            ),
            "max_steps_per_trajectory": int(
                distill_cfg.get("max_steps_per_trajectory", 15)
            ),
            "max_fragments": int(distill_cfg.get("max_fragments", 5)),
            "max_completion_tokens": int(
                distill_cfg.get("max_completion_tokens", 4096)
            ),
            "max_retries": int(distill_cfg.get("max_retries", 2)),
            "use_llm_rewrite": use_llm_rewrite,
            "credit_policy": skill_credit_policy_from_mapping(
                sage_cfg.get("skill_credit") or {}
            ),
            "initial_utility": float(
                distill_cfg.get("initial_utility", 0.5)
            ),
            "min_wins_for_protocol": int(
                distill_cfg.get("min_wins_for_protocol", 3)
            ),
            "min_protocol_steps": int(
                distill_cfg.get("min_protocol_steps", 4)
            ),
            "max_collapse_ratio": float(
                distill_cfg.get("max_collapse_ratio", 2.5)
            ),
            "min_scene_diversity": int(
                distill_cfg.get("min_scene_diversity", 2)
            ),
            "require_capability_marker_wins": bool(
                distill_cfg.get("require_capability_marker_wins", True)
            ),
            "seed_distiller": seed_distiller,
        }
        if bool(distill_cfg.get("llm_protocol_overlay", False)):
            if distillation_backend is None:
                raise ValueError(
                    "llm_protocol_overlay requires a distillation backend"
                )
            from sage_mas.llm_protocol_distill import LLMProtocolProposer

            enriched_kwargs["protocol_proposer"] = LLMProtocolProposer(
                distillation_backend,
                max_evidence=int(distill_cfg.get("overlay_max_evidence", 6)),
            )
        if distillation_mode in {
            "trajectory_enriched",
            "heuristic",
            "merged_protocol",
        }:
            if distillation_mode == "heuristic":
                enriched_kwargs["backend"] = None
                enriched_kwargs["use_llm_rewrite"] = False
            self.distiller = EnrichedSkillDistiller(**enriched_kwargs)
        elif distillation_mode == "trajectory_grounded_llm":
            if distillation_backend is None:
                raise ValueError(
                    "trajectory_grounded_llm distillation requires a backend"
                )
            enriched_kwargs["use_llm_rewrite"] = True
            enriched_kwargs["backend"] = distillation_backend
            self.distiller = EnrichedSkillDistiller(**enriched_kwargs)
        elif distillation_mode == "heuristic_seed_only":
            self.distiller = seed_distiller
        else:
            raise ValueError(
                f"Unsupported distillation mode: {distillation_mode}"
            )

        online_cfg = sage_cfg.get("online") or {}
        injection_policy = injection_policy_from_mapping(
            {**distill_cfg, **online_cfg}
        )
        self.injection_block_prefixes = tuple(
            injection_policy["block_prefixes"]
        )
        self.injection_allow_prefixes = injection_policy["allow_prefixes"]
        credit_cfg = sage_cfg.get("skill_credit") or {}
        skill_transfer_raw = sage_cfg.get("skill_transfer")
        transfer_cfg = transfer_policy_from_mapping(skill_transfer_raw or {})
        self.skill_transfer = transfer_cfg
        # Credit-only online (no skill_transfer block, paired_mu off) must not
        # inherit MU/inject_ready gates meant for the dual-bank student path.
        if skill_transfer_raw is None:
            default_require_mu = bool(
                credit_cfg.get("require_positive_mu_for_verify", False)
            )
            default_inject_ready = default_require_mu
        else:
            default_require_mu = bool(
                transfer_cfg.get("require_positive_mu_for_injection", True)
            )
            default_inject_ready = bool(
                transfer_cfg.get(
                    "require_inject_ready_for_org",
                    default_require_mu,
                )
            )
        self.require_positive_mu_for_org = bool(
            online_cfg.get(
                "require_positive_mu_for_injection",
                default_require_mu,
            )
        )
        if skill_transfer_raw is None:
            self.require_inject_ready_for_org = bool(
                online_cfg.get(
                    "require_inject_ready_for_org",
                    self.require_positive_mu_for_org,
                )
            )
        else:
            self.require_inject_ready_for_org = bool(
                online_cfg.get(
                    "require_inject_ready_for_org",
                    default_inject_ready,
                )
            )
        self.min_marginal_utility = float(
            online_cfg.get(
                "min_marginal_utility",
                credit_cfg.get("min_marginal_utility", 0.0),
            )
        )
        self.auto_verify_form_ok_skills = bool(
            online_cfg.get("auto_verify_form_ok_skills", False)
        )
        # Prefix and signal blacklists are not applied. A skill stays or
        # goes by credit and verification, not by a preset name.
        self.exclude_capability_prefixes = ()
        self.exclude_signals = set()

        gap_cfg = sage_cfg.get("assignment_gap", {})
        self.gap_estimator = AssignmentGapEstimator(
            role_weight=float(gap_cfg.get("role_weight", 0.7)),
            tool_weight=float(gap_cfg.get("tool_weight", 0.3)),
            empirical_weight=float(
                gap_cfg.get("empirical_weight", 0.7)
            ),
        )

        org_cfg = sage_cfg.get("organization", {})
        self.cluster_by = str(org_cfg.get("cluster_by", "capability"))
        self.org_editor = OrganizationEditor(
            OrganizationPolicy(
                distribution_shift_threshold=float(
                    org_cfg.get("distribution_shift_threshold", 0.15)
                ),
                min_cluster_support=int(org_cfg.get("min_cluster_support", 1)),
                min_failure_rate=float(
                    org_cfg.get("min_failure_rate", 0.0)
                ),
                allow_bootstrap_add=bool(
                    org_cfg.get("allow_bootstrap_add", True)
                ),
                max_bootstrap_agents_per_round=int(
                    org_cfg.get("max_bootstrap_agents_per_round", 1)
                ),
                max_new_agents_per_round=int(
                    org_cfg.get("max_new_agents_per_round", 1)
                ),
                cluster_by=self.cluster_by,
                add_when_uncovered_capability=bool(
                    org_cfg.get("add_when_uncovered_capability", False)
                ),
                min_assignment_gap=float(
                    org_cfg.get("min_assignment_gap", 0.1)
                ),
                dispatch_only_new_agents=bool(
                    org_cfg.get("dispatch_only_new_agents", True)
                ),
                probation_games=int(
                    org_cfg.get("probation_games", 8)
                ),
                require_executable_protocol=bool(
                    org_cfg.get("require_executable_protocol", True)
                ),
                require_trajectory_executable_protocol=bool(
                    org_cfg.get(
                        "require_trajectory_executable_protocol",
                        False,
                    )
                ),
                min_protocol_adherence_for_add_agent=float(
                    org_cfg.get("min_protocol_adherence_for_add_agent", 0.34)
                ),
                require_positive_mu_for_add_agent=bool(
                    org_cfg.get(
                        "require_positive_mu_for_add_agent",
                        online_cfg.get(
                            "require_positive_mu_for_injection",
                            credit_cfg.get(
                                "require_positive_mu_for_verify",
                                bool(
                                    (sage_cfg.get("skill_transfer") or {}).get(
                                        "require_positive_mu_for_injection",
                                        False,
                                    )
                                )
                                or bool(sage_cfg.get("skill_transfer")),
                            ),
                        ),
                    )
                ),
                min_marginal_utility_for_add_agent=float(
                    org_cfg.get(
                        "min_marginal_utility_for_add_agent",
                        credit_cfg.get("min_marginal_utility", 0.0),
                    )
                ),
                require_executable_bank_for_add_agent=bool(
                    org_cfg.get(
                        "require_executable_bank_for_add_agent",
                        bool(
                            (skill_transfer_raw or {}).get(
                                "require_executable_bank_for_add_agent",
                                False,
                            )
                        ),
                    )
                ),
                min_executable_coverage_for_add_agent=float(
                    org_cfg.get(
                        "min_executable_coverage_for_add_agent",
                        (sage_cfg.get("skill_transfer") or {}).get(
                            "min_executable_coverage_for_add_agent",
                            0.5,
                        ),
                    )
                ),
                min_new_agent_call_rate=float(
                    org_cfg.get(
                        "min_new_agent_call_rate",
                        (sage_cfg.get("shadow") or {}).get(
                            "min_new_agent_call_rate",
                            0.5,
                        ),
                    )
                ),
                prefer_assign_before_add=bool(
                    org_cfg.get("prefer_assign_before_add", True)
                ),
            )
        )
        credit_cfg = sage_cfg.get("skill_credit", {})
        self.credit_policy = skill_credit_policy_from_mapping(credit_cfg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SageEvolutionPipeline":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def run(
        self,
        trajectory_path: str | Path,
        output_root: str | Path,
        active_organization_path: str | Path | None = None,
        candidate_skills_path: str | Path | None = None,
        *,
        max_candidates: int | None = None,
        additional_candidates: list[Skill] | None = None,
        additional_evidence_trajectories: list[dict[str, Any]] | None = None,
        skip_organization: bool = False,
        student_trajectory_path: str | Path | None = None,
    ) -> PipelineArtifacts:
        run_dir = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)

        raw_trajectories = self.adapter.load(trajectory_path)
        atomic_trajectories = self.adapter.adapt_many(raw_trajectories)
        sage_cfg = self.config.get("sage", {})
        bank_path = self._resolve_skill_bank_path(sage_cfg, output_root)
        bank = SkillBank(
            bank_path,
            redundancy_threshold=float(sage_cfg.get("redundancy_threshold", 0.9)),
            disable_merge=bool(sage_cfg.get("disable_merge", False)),
        )
        if candidate_skills_path is not None:
            candidates = load_skills(candidate_skills_path)
            self._refresh_skill_distributions(
                candidates,
                atomic_trajectories,
                bank,
            )
        elif isinstance(self.distiller, EnrichedSkillDistiller):
            candidates = self.distiller.distill(
                atomic_trajectories,
                skill_bank=bank,
            )
        else:
            candidates = self.distiller.distill(atomic_trajectories)

        filtered: list[Skill] = []
        excluded: list[Skill] = []
        for skill in candidates:
            if distill_seed_is_excluded(
                skill,
                exclude_prefixes=self.exclude_capability_prefixes or None,
                exclude_signals=self.exclude_signals or None,
            ):
                skill.status = SkillStatus.REJECTED
                skill.metadata["distiller_skip_reason"] = (
                    "excluded by distillation capability/signal policy"
                )
                excluded.append(skill)
            else:
                filtered.append(skill)
        candidates = filtered

        candidates, rejected_candidates = self._validate_candidates(
            candidates,
            atomic_trajectories,
        )
        rejected_candidates = excluded + rejected_candidates

        # Fast path: form_ok skills enter the verified bank without paired MU.
        if self.auto_verify_form_ok_skills:
            from sage_mas.skill_fast_verify import auto_verify_form_ok_skills

            candidates = auto_verify_form_ok_skills(candidates)

        # Student-transfer stage: adapt drafts and/or add contrastive patches
        # before bank merge. Teacher form alone stays provisional unless
        # auto_verify_form_ok_skills is enabled above.
        student_records: list[dict[str, Any]] = []
        if student_trajectory_path is not None:
            student_records = list(self.adapter.load(student_trajectory_path))
        elif additional_evidence_trajectories:
            # Optional: caller may pass student failures via additional evidence
            # when they are tagged metadata.actor_role == student.
            student_records = [
                record
                for record in additional_evidence_trajectories
                if str(
                    (record.get("metadata") or {}).get("actor_role")
                    or record.get("actor_role")
                    or ""
                ).lower()
                in {"student", "mini", "weak"}
            ]

        transfer = self.skill_transfer
        # Teacher-only distill (cold start) must keep canonical protocols.
        # Student-aware rewrite needs student failure traces; otherwise it
        # truncates teacher drafts to ``max_protocol_steps`` and prefixes
        # non-action lines.
        if (
            transfer.get("student_aware_adapt")
            and candidates
            and student_records
        ):
            notes_by_family: dict[str, list[str]] = {}
            families = {
                str(
                    (skill.metadata or {}).get("primary_task_family")
                    or (
                        (skill.applicable_task_families or [""])[0]
                        if skill.applicable_task_families
                        else ""
                    )
                )
                for skill in candidates
            }
            for family in families:
                if not family:
                    continue
                notes_by_family[family] = summarize_student_failure_notes(
                    student_records,
                    task_family=family,
                )
            candidates = adapt_skills_for_student(
                candidates,
                student_failure_notes_by_family=notes_by_family,
                max_protocol_steps=int(transfer.get("max_protocol_steps", 8)),
            )

        if transfer.get("contrastive_patches") and student_records:
            teacher_raw = list(self.adapter.load(trajectory_path))
            patches = distill_contrastive_patches(
                teacher_raw,
                student_records,
                draft_skills=candidates,
                max_patches_per_family=int(
                    transfer.get("max_patches_per_family", 3)
                ),
            )
            if patches:
                patch_accepted, patch_rejected = self._validate_candidates(
                    patches,
                    atomic_trajectories,
                )
                # Contrastive patches are trajectory-derived; keep even if
                # generic evidence validation is strict by relaxing reject.
                if not patch_accepted and patches:
                    for skill in patches:
                        skill.metadata.setdefault(
                            "evidence_validation",
                            {"accepted": True, "reasons": ["contrastive_pair"]},
                        )
                        skill.status = SkillStatus.PROVISIONAL
                    patch_accepted = list(patches)
                    patch_rejected = []
                candidates = list(candidates) + list(patch_accepted)
                rejected_candidates.extend(patch_rejected)

        if max_candidates is not None:
            # Cap only newly distilled candidates; credit/bank org proposals
            # in additional_candidates must not be truncated away.
            candidates = candidates[: max(0, int(max_candidates))]
        extra_candidates = deepcopy(additional_candidates or [])
        if extra_candidates:
            extra_atomic = self.adapter.adapt_many(
                list(additional_evidence_trajectories or [])
            )
            extra_candidates, extra_rejected = self._validate_candidates(
                extra_candidates,
                atomic_trajectories + extra_atomic,
            )
            rejected_candidates.extend(extra_rejected)
            candidates = extra_candidates + candidates

        # Snapshot baseline BEFORE adding this round's candidates so shift is
        # not self-contaminated by the skills just distilled.
        # Deep-copy the baseline: SkillBank.add() merges in place, so a shallow
        # list would let current evidence mutate the supposedly frozen history.
        bank_before_round = deepcopy(bank.skills)
        baseline_snapshot = (
            aggregate_skill_bank_distribution(bank_before_round)
            if bank_before_round
            else None
        )

        agents = (
            load_agents(active_organization_path)
            if active_organization_path
            else self._load_agents()
        )
        for skill in candidates:
            initialize_skill_credit(skill, self.credit_policy)
        # Preserve only this round's grounded prototypes before SkillBank.add()
        # merges their evidence into cumulative canonical skills.
        round_experience_candidates = deepcopy(candidates)
        bank_cfg = sage_cfg.get("skill_lifecycle", {})
        bank.retire_after_negative = int(
            bank_cfg.get("retire_after_negative", bank.retire_after_negative)
        )
        bank.negative_utility_threshold = float(
            bank_cfg.get(
                "negative_utility_threshold",
                bank.negative_utility_threshold,
            )
        )
        bank.negative_utility_decay = float(
            bank_cfg.get(
                "negative_utility_decay",
                bank.negative_utility_decay,
            )
        )
        canonical_candidates: list[Skill] = []
        canonical_ids: set[str] = set()
        from sage_mas.executable_protocol import ensure_executable_protocol

        for skill in candidates:
            ensure_executable_protocol(skill)
            bank.add(skill)
            canonical = bank.canonical_skill(skill)
            if canonical is None or canonical.skill_id in canonical_ids:
                continue
            canonical_ids.add(canonical.skill_id)
            canonical_candidates.append(canonical)
        candidates = canonical_candidates
        bank.save()

        if skip_organization:
            self._update_cluster_archive(bank=bank, bank_path=bank_path, run_dir=run_dir)
            write_json(run_dir / "candidate_skills.json", candidates)
            write_json(run_dir / "rejected_skills.json", rejected_candidates)
            write_json(run_dir / "atomic_trajectories.json", atomic_trajectories)
            write_json(run_dir / "active_organization.json", {"agents": agents})
            write_json(
                run_dir / "organization_edits.json",
                [
                    {
                        "edit_type": "do_nothing",
                        "rationale": (
                            "Organization deferred until paired skill MU "
                            "probes complete."
                        ),
                        "assigned_skill_names": [],
                    }
                ],
            )
            write_json(
                run_dir / "candidate_organization.json",
                {"agents": agents},
            )
            write_json(run_dir / "assignment_gaps.json", [])
            write_json(run_dir / "capability_contracts.json", [])
            return PipelineArtifacts(
                run_dir=str(run_dir),
                atomic_trajectories=str(run_dir / "atomic_trajectories.json"),
                candidate_skills=str(run_dir / "candidate_skills.json"),
                skill_bank=str(bank_path),
                assignment_gaps=str(run_dir / "assignment_gaps.json"),
                organization_edits=str(run_dir / "organization_edits.json"),
                active_organization=str(run_dir / "active_organization.json"),
                candidate_organization=str(
                    run_dir / "candidate_organization.json"
                ),
                experience_distribution_snapshot=None,
                capability_contracts=str(run_dir / "capability_contracts.json"),
                rejected_skills=str(run_dir / "rejected_skills.json"),
            )

        return self._write_organization_round(
            run_dir=run_dir,
            bank=bank,
            bank_path=bank_path,
            bank_before_round=bank_before_round,
            baseline_snapshot=baseline_snapshot,
            agents=agents,
            candidates=candidates,
            rejected_candidates=rejected_candidates,
            atomic_trajectories=atomic_trajectories,
            round_experience_candidates=round_experience_candidates,
            raw_trajectory_count=len(raw_trajectories),
        )

    def repropose_organization(
        self,
        *,
        output_root: str | Path,
        active_organization_path: str | Path | None,
        skill_bank_path: str | Path | None = None,
        bank_before_round: list[Skill] | None = None,
    ) -> PipelineArtifacts:
        """Propose org edits from the current skill bank (after MU probes)."""
        run_dir = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        sage_cfg = self.config.get("sage", {})
        bank_path = Path(
            skill_bank_path
            or self._resolve_skill_bank_path(sage_cfg, output_root)
        )
        bank = SkillBank(
            bank_path,
            redundancy_threshold=float(sage_cfg.get("redundancy_threshold", 0.9)),
            disable_merge=bool(sage_cfg.get("disable_merge", False)),
        )
        agents = (
            load_agents(active_organization_path)
            if active_organization_path
            else self._load_agents()
        )
        snapshot_before = (
            deepcopy(bank_before_round)
            if bank_before_round is not None
            else deepcopy(bank.skills)
        )
        baseline_snapshot = (
            aggregate_skill_bank_distribution(snapshot_before)
            if snapshot_before
            else None
        )
        candidates = [
            skill
            for skill in bank.skills
            if skill.status not in {SkillStatus.REJECTED, SkillStatus.RETIRED}
        ]
        return self._write_organization_round(
            run_dir=run_dir,
            bank=bank,
            bank_path=bank_path,
            bank_before_round=snapshot_before,
            baseline_snapshot=baseline_snapshot,
            agents=agents,
            candidates=candidates,
            rejected_candidates=[],
            atomic_trajectories=[],
            round_experience_candidates=deepcopy(candidates),
            raw_trajectory_count=0,
        )

    def _write_organization_round(
        self,
        *,
        run_dir: Path,
        bank: SkillBank,
        bank_path: Path,
        bank_before_round: list[Skill],
        baseline_snapshot: Any,
        agents: list[AgentSpec],
        candidates: list[Skill],
        rejected_candidates: list[Skill],
        atomic_trajectories: list[Any],
        round_experience_candidates: list[Skill],
        raw_trajectory_count: int,
    ) -> PipelineArtifacts:
        eligible_skills = self._org_eligible_skills(candidates)
        gaps = []
        edits = []
        existing_agent_names = {agent.name for agent in agents}
        executor_name = next(
            (
                agent.name
                for agent in agents
                if "executor" in f"{agent.name} {agent.role}".lower()
            ),
            agents[0].name if agents else "Executor",
        )
        clusters = self._cluster_skills(
            eligible_skills,
            cluster_by=self.cluster_by,
        )
        clusters = sorted(
            clusters,
            key=lambda cluster: (
                -unique_evidence_support(cluster),
                cluster_key_for_skill(
                    cluster[0],
                    cluster_by=self.cluster_by,
                )
                if cluster
                else "",
            ),
        )
        bootstrap_budget = self.org_editor.policy.max_bootstrap_agents_per_round
        add_budget = self.org_editor.policy.max_new_agents_per_round
        deviated_cluster_count = 0
        evolving_agents = list(agents)
        for cluster in clusters:
            key = cluster_key_for_skill(
                cluster[0],
                cluster_by=self.cluster_by,
            )
            cluster_ids = {skill.skill_id for skill in cluster}
            if not bank_before_round:
                cluster_shift = shift_for_skill_cluster(
                    cluster,
                    None,
                    baseline_skills=[],
                )
            else:
                historical_baseline = [
                    skill
                    for skill in bank_before_round
                    if skill.skill_id not in cluster_ids
                ]
                baseline_skills = (
                    historical_baseline
                    if historical_baseline
                    else list(bank_before_round)
                )
                cluster_shift = shift_for_skill_cluster(
                    cluster,
                    aggregate_skill_bank_distribution(baseline_skills),
                    baseline_skills=baseline_skills,
                )
            credit_promotion = any(
                bool(skill.metadata.get("credit_promoted_pending_org"))
                for skill in cluster
            )
            for skill in cluster:
                skill.distribution_shift = cluster_shift
            if self.org_editor.is_deviated(
                cluster,
                cluster_shift=cluster_shift,
            ):
                deviated_cluster_count += 1
            gap = self.gap_estimator.estimate(cluster, evolving_agents)
            edit = self.org_editor.propose(
                cluster,
                gap,
                existing_agent_names=existing_agent_names,
                executor_name=executor_name,
                existing_agents=evolving_agents,
                cluster_key=key,
                cluster_shift=cluster_shift,
                bootstrap_slots_remaining=bootstrap_budget,
                add_slots_remaining=add_budget,
            )
            if credit_promotion and edit.rationale:
                edit.rationale = (
                    f"{edit.rationale} "
                    "Credit proved the Skill usable; organization still used "
                    "independent cluster novelty and assignment gap."
                )
            gaps.append(gap)
            edits.append(edit)
            if (
                edit.edit_type == OrganizationEditType.ADD_AGENT
                and edit.new_agent is not None
            ):
                existing_agent_names.add(edit.new_agent.name)
                evolving_agents = list(evolving_agents) + [edit.new_agent]
                add_budget = max(0, add_budget - 1)
                if cluster_shift.baseline_source == "empty_skill_bank":
                    bootstrap_budget = max(0, bootstrap_budget - 1)
            if credit_promotion:
                for skill in cluster:
                    skill.metadata.pop(
                        "credit_promoted_pending_org",
                        None,
                    )
        bank.save()
        self._update_cluster_archive(bank=bank, bank_path=bank_path, run_dir=run_dir)
        candidate_agents = OrganizationStateManager.apply_candidate(
            agents,
            edits,
            skills=list(bank.skills),
        )

        atomic_path = run_dir / "atomic_trajectories.json"
        candidates_path = run_dir / "candidate_skills.json"
        rejected_path = run_dir / "rejected_skills.json"
        gaps_path = run_dir / "assignment_gaps.json"
        edits_path = run_dir / "organization_edits.json"
        active_org_path = run_dir / "active_organization.json"
        candidate_org_path = run_dir / "candidate_organization.json"
        experience_snapshot_path = (
            run_dir / "experience_distribution_snapshot.json"
        )
        capability_contracts_path = run_dir / "capability_contracts.json"
        write_json(atomic_path, atomic_trajectories)
        write_json(candidates_path, candidates)
        write_json(rejected_path, rejected_candidates)
        write_json(gaps_path, gaps)
        write_json(edits_path, edits)
        write_json(active_org_path, {"agents": agents})
        write_json(candidate_org_path, {"agents": candidate_agents})
        write_json(
            capability_contracts_path,
            [
                edit.new_agent.shadow_evaluation_record["capability_contract"]
                for edit in edits
                if edit.new_agent is not None
                and "capability_contract"
                in edit.new_agent.shadow_evaluation_record
            ],
        )
        round_distribution = getattr(
            self.distiller,
            "last_round_distribution",
            None,
        ) or compute_round_exploration_distribution(atomic_trajectories)
        signal_cooccurrence = getattr(
            self.distiller,
            "last_signal_cooccurrence",
            {},
        )
        add_agent_count = sum(
            1 for edit in edits if edit.edit_type == OrganizationEditType.ADD_AGENT
        )
        experience_snapshot = build_experience_distribution_snapshot(
            round_experience_candidates,
            bank_before_round,
            round_id=run_dir.name,
            novelty_threshold=self.org_editor.policy.distribution_shift_threshold,
            round_distribution=round_distribution,
        )
        write_json(experience_snapshot_path, experience_snapshot)
        write_json(
            run_dir / "exploration_distribution.json",
            {
                "round_distribution": round_distribution,
                "signal_cooccurrence": signal_cooccurrence,
                "bank_baseline_before_round": baseline_snapshot,
            },
        )
        write_json(
            run_dir / "run_summary.json",
            {
                "trajectory_count": raw_trajectory_count,
                "candidate_skill_count": len(candidates),
                "rejected_skill_count": len(rejected_candidates),
                "eligible_skill_count": len(eligible_skills),
                "verified_skill_count": sum(
                    1 for skill in candidates if skill.status == SkillStatus.VERIFIED
                ),
                "organization_edit_count": len(edits),
                "cluster_count": len(clusters),
                "deviated_cluster_count": deviated_cluster_count,
                "add_agent_count": add_agent_count,
                "enriched_skill_count": sum(
                    1 for skill in candidates if skill.trajectory_summary
                ),
                "mean_distribution_shift": (
                    sum(
                        skill.distribution_shift.shift_score
                        for skill in candidates
                        if skill.distribution_shift is not None
                    )
                    / max(
                        1,
                        sum(
                            1
                            for skill in candidates
                            if skill.distribution_shift is not None
                        ),
                    )
                ),
            },
        )

        return PipelineArtifacts(
            run_dir=str(run_dir),
            atomic_trajectories=str(atomic_path),
            candidate_skills=str(candidates_path),
            skill_bank=str(bank_path),
            assignment_gaps=str(gaps_path),
            organization_edits=str(edits_path),
            active_organization=str(active_org_path),
            candidate_organization=str(candidate_org_path),
            experience_distribution_snapshot=str(experience_snapshot_path),
            capability_contracts=str(capability_contracts_path),
            rejected_skills=str(rejected_path),
        )

    def preview_experience_distribution(
        self,
        raw_trajectories: list[dict[str, Any]],
        historical_skills: list[Skill],
        *,
        round_id: str,
    ) -> ExperienceDistributionSnapshot:
        """Distill a non-mutating snapshot for paired organization forks."""
        atomic_trajectories = self.adapter.adapt_many(raw_trajectories)
        if isinstance(self.distiller, EnrichedSkillDistiller):
            skills = self.distiller.distill(
                atomic_trajectories,
                skill_bank=None,
            )
        else:
            skills = self.distiller.distill(atomic_trajectories)
        skills, _ = self._validate_candidates(skills, atomic_trajectories)
        return build_experience_distribution_snapshot(
            skills,
            historical_skills,
            round_id=round_id,
            novelty_threshold=self.org_editor.policy.distribution_shift_threshold,
            round_distribution=compute_round_exploration_distribution(
                atomic_trajectories
            ),
        )

    def _load_agents(self) -> list[AgentSpec]:
        agent_records = self.config.get("sage", {}).get("agents", [])
        return [agent_from_dict(record) for record in agent_records]

    @staticmethod
    def _resolve_skill_bank_path(
        sage_cfg: dict[str, Any],
        output_root: str | Path,
    ) -> Path:
        configured = sage_cfg.get("skill_bank_path")
        if configured:
            return Path(configured)
        return Path(output_root).parent / "skill_bank.json"

    def _update_cluster_archive(
        self,
        *,
        bank: SkillBank,
        bank_path: Path,
        run_dir: Path,
    ) -> None:
        """Observation-only cluster archive update (one call = one segment).

        The archive records cluster births/assignments and per-segment
        snapshots next to the skill bank; it never feeds back into this
        round's decisions.
        """
        clustering_cfg = self.config.get("sage", {}).get("clustering", {})
        if not bool(clustering_cfg.get("enabled", True)):
            return
        from sage_mas.skill_clustering import (
            DEFAULT_NOVELTY_THRESHOLD,
            SkillClusterArchive,
        )

        archive_path = Path(
            clustering_cfg.get("archive_path")
            or bank_path.parent / "skill_clusters.json"
        )
        archive = SkillClusterArchive.load_or_create(
            archive_path,
            novelty_threshold=float(
                clustering_cfg.get(
                    "novelty_threshold", DEFAULT_NOVELTY_THRESHOLD
                )
            ),
        )
        report = archive.update(list(bank.skills))
        archive.save(archive_path)
        write_json(run_dir / "cluster_events.json", report.to_dict())

    def _refresh_skill_distributions(
        self,
        candidates: list[Skill],
        atomic_trajectories: list[list[Any]],
        bank: SkillBank,
    ) -> None:
        if not candidates:
            return
        if not any(
            skill.distribution_shift is None
            or skill.exploration_distribution is None
            for skill in candidates
        ):
            return
        trajectories_by_id = index_trajectory_steps(atomic_trajectories)
        baseline = (
            aggregate_skill_bank_distribution(bank.skills)
            if bank.skills
            else None
        )
        signal_cooccurrence = compute_signal_cooccurrence(candidates)
        for skill in candidates:
            if (
                skill.distribution_shift is not None
                and skill.exploration_distribution is not None
            ):
                continue
            evidence_trajectories = [
                trajectories_by_id[evidence_id]
                for evidence_id in skill.evidence_ids
                if evidence_id in trajectories_by_id
            ]
            if skill.exploration_distribution is None and evidence_trajectories:
                skill.exploration_distribution = compute_skill_exploration_distribution(
                    evidence_trajectories,
                    source_signal=str(skill.metadata.get("source_signal", "")) or None,
                    signal_cooccurrence=signal_cooccurrence,
                )
            if skill.distribution_shift is None:
                attach_distribution_shift(
                    skill,
                    baseline=baseline,
                    signal_cooccurrence=signal_cooccurrence,
                )

    def _org_eligible_skills(self, candidates: list[Skill]) -> list[Skill]:
        # Unverified / shallow single-trace protocols may be persisted, but they
        # cannot alter the active organization. Credit-promoted skills already
        # proved usable online and may lack a fresh alignment pass this round.
        # Broad execution.* skills also cannot spawn specialists.
        # SAGE dual-bank: prefer Executable-bank skills when transfer is on.
        from sage_mas.skill_bank_roles import ROLE_EXECUTABLE, skill_bank_role

        eligible = [
            skill
            for skill in candidates
            if skill.status == SkillStatus.VERIFIED
            and (
                protocol_is_org_ready(skill)
                or bool(skill.metadata.get("credit_promoted_pending_org"))
            )
        ]
        if self.require_inject_ready_for_org:
            eligible = [
                skill
                for skill in eligible
                if skill_bank_role(skill) == ROLE_EXECUTABLE
                or (
                    (skill.metadata or {}).get("inject_ready") is True
                    and (skill.metadata or {}).get("mu_rejected") is not True
                )
            ]
        return filter_injectable_skills(
            eligible,
            block_prefixes=self.injection_block_prefixes,
            allow_prefixes=self.injection_allow_prefixes,
            require_positive_mu=self.require_positive_mu_for_org,
            min_marginal_utility=self.min_marginal_utility,
            require_inject_ready=self.require_inject_ready_for_org,
        )

    @staticmethod
    def _validate_candidates(
        candidates: list[Skill],
        atomic_trajectories: list[list[Any]],
    ) -> tuple[list[Skill], list[Skill]]:
        trajectories_by_id = index_trajectory_steps(atomic_trajectories)
        accepted: list[Skill] = []
        rejected: list[Skill] = []
        for skill in candidates:
            # Credit-promoted / already-verified skills are grounded against
            # historical evidence. Do not re-reject them just because this
            # segment's trajectory batch does not replay their evidence IDs.
            if skill.status == SkillStatus.VERIFIED or bool(
                skill.metadata.get("credit_promoted_pending_org")
            ):
                accepted.append(skill)
                continue
            prior_validation = skill.metadata.get("evidence_validation")
            contract_reasons = generic_skill_contract_reasons(skill)
            if (
                isinstance(skill.metadata.get("skill_credit"), dict)
                and isinstance(prior_validation, dict)
                and bool(prior_validation.get("accepted", False))
                and int(skill.metadata["skill_credit"].get("uses", 0)) >= 0
                and not contract_reasons
            ):
                accepted.append(skill)
                continue
            is_environment_grounded = (
                skill.metadata.get("grounding_protocol")
                == "environment_confirmed_v1"
                or skill.metadata.get("candidate_stage")
                == "environment_confirmed_discovery"
            )
            validation = (
                validate_environment_confirmed_skill(
                    skill,
                    trajectories_by_id,
                )
                if is_environment_grounded
                else validate_skill_evidence(skill, trajectories_by_id)
            )
            if contract_reasons:
                validation.reasons.extend(contract_reasons)
                validation.accepted = False
            skill.metadata["evidence_validation"] = validation.as_metadata()
            if validation.accepted:
                accepted.append(skill)
            else:
                skill.status = SkillStatus.REJECTED
                rejected.append(skill)
        return accepted, rejected

    @staticmethod
    def _cluster_skills(
        skills: list[Skill],
        *,
        cluster_by: str = "capability",
    ) -> list[list[Skill]]:
        clusters: dict[str, list[Skill]] = defaultdict(list)
        for skill in skills:
            key = cluster_key_for_skill(skill, cluster_by=cluster_by)
            clusters[key].append(skill)
        return list(clusters.values())
