"""End-to-end skill distillation: merge wins, drop detours, online utility."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sage_mas.llm_protocol_distill import LLMProtocolProposer

from sage_mas.distribution_stats import (
    aggregate_skill_bank_distribution,
    attach_distribution_shift,
    compute_round_exploration_distribution,
    compute_signal_cooccurrence,
    compute_skill_exploration_distribution,
    index_trajectory_steps,
)
from sage_mas.schemas import Skill, SkillStatus
from sage_mas.skill_bank import SkillBank
from sage_mas.skill_credit import SkillCreditPolicy
from sage_mas.skill_distiller import (
    DistillationConfig,
    HeuristicSkillDistiller,
)
from sage_mas.executable_protocol import ensure_executable_protocol
from sage_mas.skill_protocol_distill import (
    DEFAULT_MAX_COLLAPSE_RATIO,
    DEFAULT_MIN_PROTOCOL_STEPS,
    DEFAULT_MIN_SCENE_DIVERSITY,
    build_protocol_from_successes,
    initialize_provisional_utility,
    protocol_causal_issues,
)
from sage_mas.skill_quality import (
    annotate_skill_identity,
    trajectory_has_capability,
)
from sage_mas.trajectory.finalize import prefer_success_trajectories
from sage_mas.trajectory.fragments import embed_skill, extract_key_fragments
from sage_mas.trajectory.summarizer import (
    SummarizationBackend,
    TrajectorySummarizer,
)


class EnrichedSkillDistiller:
    """Distill skills with the simplified 6-step protocol pipeline.

    1. Cluster trajectories into capability seeds.
    2. Collect multiple successful same-capability trajectories.
    3. Remove detours from each trajectory.
    4. Merge into one shared action_protocol.
    5. Enter provisional with initial utility.
    6. Online credit (caller) updates utility and promote/retire.

    LLM rewrite is optional and off by default to avoid multi-field prompt noise.
    """

    def __init__(
        self,
        config: DistillationConfig | None = None,
        *,
        backend: SummarizationBackend | None = None,
        seed_distiller: HeuristicSkillDistiller | None = None,
        max_evidence_trajectories: int = 5,
        max_steps_per_trajectory: int = 15,
        max_fragments: int = 5,
        max_completion_tokens: int = 4096,
        max_retries: int = 2,
        use_llm_rewrite: bool = False,
        credit_policy: SkillCreditPolicy | None = None,
        initial_utility: float = 0.5,
        min_wins_for_protocol: int = 3,
        min_protocol_steps: int = DEFAULT_MIN_PROTOCOL_STEPS,
        max_collapse_ratio: float = DEFAULT_MAX_COLLAPSE_RATIO,
        min_scene_diversity: int = DEFAULT_MIN_SCENE_DIVERSITY,
        require_capability_marker_wins: bool = True,
        protocol_proposer: LLMProtocolProposer | None = None,
    ):
        self.seed_distiller = seed_distiller or HeuristicSkillDistiller(config)
        self.summarizer = TrajectorySummarizer(
            backend=backend if use_llm_rewrite else None,
            max_evidence_trajectories=max_evidence_trajectories,
            max_steps_per_trajectory=max_steps_per_trajectory,
            max_completion_tokens=max_completion_tokens,
            max_retries=max_retries,
        )
        self.max_fragments = max_fragments
        self.use_llm_rewrite = bool(use_llm_rewrite)
        self.credit_policy = credit_policy or SkillCreditPolicy()
        self.initial_utility = float(initial_utility)
        self.min_wins_for_protocol = max(1, int(min_wins_for_protocol))
        self.min_protocol_steps = max(1, int(min_protocol_steps))
        self.max_collapse_ratio = float(max_collapse_ratio)
        self.min_scene_diversity = max(1, int(min_scene_diversity))
        self.require_capability_marker_wins = bool(require_capability_marker_wins)
        self.protocol_proposer = protocol_proposer
        self.last_round_distribution = None
        self.last_signal_cooccurrence: dict[str, int] = {}

    def distill(
        self,
        trajectories: list[list[Any]],
        skill_bank: SkillBank | None = None,
    ) -> list[Skill]:
        # Step 1: cluster trajectories into capability seeds.
        seeds = self._merge_capability_seeds(
            self.seed_distiller.distill(trajectories)
        )
        trajectories_by_id = index_trajectory_steps(trajectories)
        self.last_round_distribution = compute_round_exploration_distribution(
            trajectories
        )
        self.last_signal_cooccurrence = compute_signal_cooccurrence(seeds)
        baseline = (
            aggregate_skill_bank_distribution(skill_bank.skills)
            if skill_bank is not None and skill_bank.skills
            else None
        )

        enriched: list[Skill] = []
        for seed in seeds:
            evidence_trajectories = self._select_evidence(
                seed,
                trajectories,
                trajectories_by_id,
            )
            if not evidence_trajectories:
                continue

            # Steps 2–4: multi-win detour-clean merge → single protocol.
            skill = build_protocol_from_successes(
                seed,
                evidence_trajectories,
                max_trajectories=max(
                    3, self.summarizer.max_evidence_trajectories
                ),
                min_wins=self.min_wins_for_protocol,
                min_protocol_steps=self.min_protocol_steps,
                max_collapse_ratio=self.max_collapse_ratio,
                min_scene_diversity=self.min_scene_diversity,
                require_capability_marker=self.require_capability_marker_wins,
            )
            if not (
                bool(skill.metadata.get("protocol_alignment_ok", False))
                and bool(skill.metadata.get("protocol_structure_ok", False))
                and bool(skill.metadata.get("protocol_form_ok", False))
            ):
                skill.status = SkillStatus.REJECTED
                skill.metadata["distiller_skip_reason"] = str(
                    skill.metadata.get("protocol_quality_reject")
                    or "protocol quality gate failed"
                )
                enriched.append(skill)
                continue

            # Optional LLM protocol proposal (grounded, gated, falls back
            # to the heuristic merge on any rejection).
            if self.protocol_proposer is not None:
                self._overlay_llm_protocol(skill, evidence_trajectories, trajectories)

            # Optional LLM prose rewrite (disabled by default).
            if self.use_llm_rewrite and self.summarizer.backend is not None:
                fragments = extract_key_fragments(
                    evidence_trajectories,
                    max_fragments=self.max_fragments,
                )
                locked_protocol = list(skill.action_protocol)
                rewritten = self.summarizer.summarize(
                    deepcopy(skill),
                    evidence_trajectories,
                    fragments,
                )
                # Keep the merged protocol as the only executable truth.
                rewritten.action_protocol = locked_protocol
                for key in (
                    "protocol_source",
                    "protocol_source_trajectory_id",
                    "protocol_alignment_support",
                    "protocol_stages",
                    "protocol_alignment_ok",
                    "protocol_structure_ok",
                    "protocol_structure_issues",
                    "protocol_form_ok",
                    "protocol_form_metrics",
                    "protocol_quality_reject",
                    "protocol_win_scene_count",
                    "executable_protocol",
                    "executable_protocol_source",
                    "protocol_merge_trace_count",
                ):
                    if key in skill.metadata:
                        rewritten.metadata[key] = skill.metadata[key]
                skill = rewritten
                skill.key_fragments = fragments
            else:
                skill.key_fragments = extract_key_fragments(
                    evidence_trajectories,
                    max_fragments=self.max_fragments,
                )
                skill.metadata.pop("failure_branches", None)

            # Confirmed transitions for contract checks.
            won_fragments = [
                fragment for fragment in skill.key_fragments if fragment.won
            ]
            transition_source = won_fragments or list(skill.key_fragments)
            skill.metadata["confirmed_transitions"] = [
                {
                    "task_id": str(fragment.trajectory_id or "").strip(),
                    "action": str(fragment.action or "").strip(),
                    "observation": str(fragment.observation or "").strip(),
                }
                for fragment in transition_source
                if str(fragment.trajectory_id or "").strip()
                and str(fragment.action or "").strip()
                and str(fragment.observation or "").strip()
            ]
            skill.exploration_distribution = (
                compute_skill_exploration_distribution(
                    evidence_trajectories,
                    source_signal=str(seed.metadata.get("source_signal", ""))
                    or None,
                    signal_cooccurrence=self.last_signal_cooccurrence,
                )
            )
            skill = attach_distribution_shift(
                skill,
                baseline=baseline,
                signal_cooccurrence=self.last_signal_cooccurrence,
            )
            skill.embedding = embed_skill(skill)

            # Step 5: provisional + initial utility.
            initialize_provisional_utility(
                skill,
                policy=self.credit_policy,
                initial_utility=self.initial_utility,
            )
            skill.metadata["distiller"] = (
                "merged-success-detour-clean-llm-v2"
                if self.use_llm_rewrite and self.summarizer.backend is not None
                else "merged-success-detour-clean-v2"
            )
            annotate_skill_identity(skill)
            enriched.append(skill)
        return enriched

    def _overlay_llm_protocol(
        self,
        skill: Skill,
        wins: list[list[Any]],
        pool: list[list[Any]],
    ) -> None:
        """Replace the heuristic protocol with a grounded LLM proposal.

        The heuristic merge has already run, so ``skill`` always keeps a
        valid fallback: any proposer failure, contract violation, grounding
        rejection, or causal-gate rejection leaves the heuristic protocol
        untouched and records the reason in metadata.
        """
        proposer = self.protocol_proposer
        if proposer is None:
            return
        family = ""
        if wins and wins[0]:
            family = str(getattr(wins[0][-1], "metadata", {}).get("task_family") or "")
        failures: list[list[Any]] = []
        if family and proposer.max_failures > 0:
            win_ids = {id(trajectory) for trajectory in wins}
            for trajectory in pool:
                if len(failures) >= proposer.max_failures:
                    break
                if not trajectory or id(trajectory) in win_ids:
                    continue
                metadata = getattr(trajectory[-1], "metadata", {})
                if metadata.get("won"):
                    continue
                if str(metadata.get("task_family") or "") != family:
                    continue
                failures.append(trajectory)
        proposal = proposer.propose(
            capability=skill.capability_key,
            family=family or skill.capability_key,
            wins=wins,
            failures=failures,
        )
        if proposal is None:
            skill.metadata["llm_protocol"] = "unavailable_or_rejected"
            return
        issues = protocol_causal_issues(proposal.protocol, capability=skill.capability_key)
        if issues:
            skill.metadata["llm_protocol_rejected"] = "; ".join(issues)
            return
        heuristic_protocol = list(skill.action_protocol)
        skill.action_protocol = list(proposal.protocol)
        skill.metadata["protocol_source"] = "llm_proposed_grounded_v1"
        skill.metadata["heuristic_protocol_fallback"] = heuristic_protocol
        skill.metadata["llm_grounding"] = proposal.grounding
        if proposal.search_hint:
            skill.metadata["search_hint"] = proposal.search_hint
        if proposal.object_note:
            skill.precondition = f"{skill.precondition} {proposal.object_note}".strip()
        if proposal.anti_patterns:
            existing = [str(item) for item in (skill.metadata.get("anti_patterns") or [])]
            for pattern in proposal.anti_patterns:
                if pattern.lower() not in {item.lower() for item in existing}:
                    existing.append(pattern)
            skill.metadata["anti_patterns"] = existing
        if proposal.rationale:
            skill.metadata["llm_rationale"] = proposal.rationale
        ensure_executable_protocol(skill, force=True)

    @staticmethod
    def _merge_capability_seeds(seeds: list[Skill]) -> list[Skill]:
        """Collapse success/failure signal variants into one capability seed."""
        grouped: dict[tuple[str, str], list[Skill]] = {}
        for seed in seeds:
            signal = str(seed.metadata.get("source_signal", "") or "")
            family = str(seed.metadata.get("primary_task_family", "") or "other")
            named = str(seed.capability_key or "").strip()
            key = (named or signal, family)
            grouped.setdefault(key, []).append(seed)

        merged: list[Skill] = []
        for (_group_key, family), items in grouped.items():
            base = deepcopy(
                next(
                    (
                        item
                        for item in items
                        if str(item.metadata.get("outcome_class", ""))
                        == "failure"
                    ),
                    items[0],
                )
            )
            named = next(
                (
                    str(item.capability_key or "").strip()
                    for item in items
                    if str(item.capability_key or "").strip()
                ),
                "",
            )
            if named:
                base.capability_key = named
            base.applicable_task_families = (
                [] if family == "other" else [family]
            )
            base.evidence_ids = sorted(
                {
                    evidence_id
                    for item in items
                    for evidence_id in item.evidence_ids
                }
            )
            base.support_count = len(base.evidence_ids)
            base.metadata["source_signals"] = sorted(
                {
                    str(item.metadata.get("source_signal", "") or "")
                    for item in items
                }
            )
            annotate_skill_identity(base)
            merged.append(base)
        return merged

    def _select_evidence(
        self,
        seed: Skill,
        trajectories: list[list[Any]],
        trajectories_by_id: dict[str, list[Any]],
    ) -> list[list[Any]]:
        original_evidence = [
            trajectories_by_id[evidence_id]
            for evidence_id in seed.evidence_ids
            if evidence_id in trajectories_by_id
        ]
        family = str(seed.metadata.get("primary_task_family", ""))
        family_trajectories = [
            steps
            for steps in trajectories
            if steps
            and (
                not family
                or str(steps[-1].metadata.get("task_family", "")) == family
            )
        ]
        positive_trajectories = [
            steps
            for steps in family_trajectories
            if (
                bool(steps[-1].metadata.get("won", False))
                or any(
                    self._positive_progress(
                        step.metadata.get("goal_progress_delta")
                    )
                    for step in steps[:-1]
                )
            )
            and trajectory_has_capability(steps, seed.capability_key)
        ]
        if not positive_trajectories:
            # Fall back to family wins even without capability marker match.
            positive_trajectories = [
                steps
                for steps in family_trajectories
                if steps and bool(steps[-1].metadata.get("won", False))
            ]
        if not positive_trajectories:
            return []

        reference = original_evidence or positive_trajectories
        positive_trajectories.sort(
            key=lambda steps: self._trajectory_similarity(steps, reference),
            reverse=True,
        )
        limit = max(3, self.summarizer.max_evidence_trajectories)
        wins = [
            steps
            for steps in positive_trajectories
            if steps and bool(steps[-1].metadata.get("won", False))
        ]
        ordered = prefer_success_trajectories(
            wins[:limit] if wins else positive_trajectories[:limit]
        )
        return ordered

    @staticmethod
    def _positive_progress(value: Any) -> bool:
        try:
            return float(value or 0.0) > 0.0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _trajectory_similarity(
        candidate: list[Any],
        references: list[list[Any]],
    ) -> float:
        """Task-text Jaccard used only to pair nearby outcomes within a family."""
        if not candidate:
            return 0.0
        candidate_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                str(candidate[-1].metadata.get("task", "")).lower(),
            )
        )
        if not candidate_tokens:
            return 0.0
        best = 0.0
        for steps in references:
            if not steps:
                continue
            reference_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    str(steps[-1].metadata.get("task", "")).lower(),
                )
            )
            union = candidate_tokens | reference_tokens
            if union:
                best = max(
                    best,
                    len(candidate_tokens & reference_tokens) / len(union),
                )
        return best
