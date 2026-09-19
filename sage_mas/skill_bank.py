"""Persistent, deduplicated SkillBank."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sage_mas.distribution_stats import aggregate_skill_bank_distribution
from sage_mas.schemas import AtomicOp, ExplorationDistribution, Skill, SkillStatus
from sage_mas.serialization import skill_from_dict, write_json
from sage_mas.skill_quality import annotate_skill_identity


class SkillBank:
    def __init__(
        self,
        path: str | Path,
        redundancy_threshold: float = 0.9,
        *,
        # 0 disables held-out MU retirement; online skill_credit prune remains.
        retire_after_negative: int = 0,
        negative_utility_threshold: float = 0.0,
        negative_utility_decay: float = 0.5,
        # When True, every incoming skill is appended as a new bank entry
        # instead of being merged into a near-duplicate. This lets the skill
        # cluster archive observe variant births and lets the credit system
        # select between siblings, matching the SAGE "birth then selection"
        # loop. Identical skills still collapse because _find_duplicate
        # returns the exact same object reference; only near-duplicates
        # that would otherwise be absorbed are kept as separate entries.
        disable_merge: bool = False,
    ):
        self.path = Path(path)
        self.redundancy_threshold = redundancy_threshold
        self.disable_merge = bool(disable_merge)
        self.retire_after_negative = max(0, int(retire_after_negative))
        self.negative_utility_threshold = float(negative_utility_threshold)
        self.negative_utility_decay = min(
            1.0,
            max(0.0, float(negative_utility_decay)),
        )
        self.skills: list[Skill] = []
        if self.path.exists():
            loaded = [skill_from_dict(item) for item in self._read()]
            for skill in loaded:
                annotate_skill_identity(skill)
                self.add(skill)

    def add(self, skill: Skill, allow_provisional: bool = True) -> bool:
        if skill.status in {SkillStatus.REJECTED, SkillStatus.RETIRED}:
            return False

        duplicate = self._find_duplicate(skill)
        if duplicate is not None and not self.disable_merge:
            duplicate.evidence_ids = sorted(set(duplicate.evidence_ids + skill.evidence_ids))
            duplicate.support_count = max(
                len(duplicate.evidence_ids),
                duplicate.support_count,
                skill.support_count,
            )
            duplicate.key_fragments = self._merge_fragments(
                duplicate.key_fragments,
                skill.key_fragments,
            )
            if skill.trajectory_summary:
                duplicate.trajectory_summary = skill.trajectory_summary
            if skill.embedding:
                duplicate.embedding = skill.embedding
            duplicate.exploration_distribution = self._merge_distributions(
                duplicate.exploration_distribution,
                skill.exploration_distribution,
            )
            if skill.distribution_shift is not None:
                duplicate.distribution_shift = skill.distribution_shift
            if skill.marginal_utility is not None:
                duplicate.marginal_utility = skill.marginal_utility
                self._record_utility(duplicate, skill.marginal_utility)
            for family in skill.applicable_task_families:
                if family not in duplicate.applicable_task_families:
                    duplicate.applicable_task_families.append(family)
            duplicate.applicable_task_families.sort()
            if skill.capability_key and not duplicate.capability_key:
                duplicate.capability_key = skill.capability_key
            # Refresh executable protocol when the incoming draft is a real
            # consensus update (otherwise metadata can claim typed_stage_consensus
            # while action_protocol stays a stale single-trace clone).
            incoming_protocol = list(skill.action_protocol or [])
            if incoming_protocol and (
                not duplicate.action_protocol
                or str((skill.metadata or {}).get("protocol_merge_mode") or "")
                == "typed_stage_consensus"
                or len(incoming_protocol) != len(duplicate.action_protocol or [])
            ):
                duplicate.action_protocol = incoming_protocol
                if skill.description:
                    duplicate.description = skill.description
                if skill.precondition:
                    duplicate.precondition = skill.precondition
                if skill.expected_effect:
                    duplicate.expected_effect = skill.expected_effect
            if (
                skill.status == SkillStatus.VERIFIED
                and duplicate.status != SkillStatus.RETIRED
            ):
                duplicate.status = SkillStatus.VERIFIED
            existing_credit = duplicate.metadata.get("skill_credit")
            incoming_credit = skill.metadata.get("skill_credit")
            existing_alignment = int(
                duplicate.metadata.get("protocol_alignment_support") or 0
            )
            existing_validation = duplicate.metadata.get("evidence_validation")
            existing_confirmed = list(
                duplicate.metadata.get("confirmed_transitions") or []
            )
            duplicate.metadata.update(skill.metadata)
            incoming_alignment = int(
                duplicate.metadata.get("protocol_alignment_support") or 0
            )
            if existing_alignment > incoming_alignment:
                duplicate.metadata["protocol_alignment_support"] = (
                    existing_alignment
                )
            if (
                isinstance(existing_validation, dict)
                and existing_validation.get("accepted")
                and not (
                    isinstance(duplicate.metadata.get("evidence_validation"), dict)
                    and duplicate.metadata["evidence_validation"].get(
                        "accepted"
                    )
                )
            ):
                duplicate.metadata["evidence_validation"] = existing_validation
            if existing_confirmed and not duplicate.metadata.get(
                "confirmed_transitions"
            ):
                duplicate.metadata["confirmed_transitions"] = existing_confirmed
            if isinstance(existing_credit, dict) and isinstance(
                incoming_credit,
                dict,
            ):
                duplicate.metadata["skill_credit"] = (
                    existing_credit
                    if int(existing_credit.get("uses", 0))
                    >= int(incoming_credit.get("uses", 0))
                    else incoming_credit
                )
            return False

        if skill.marginal_utility is not None and not skill.utility_history:
            self._record_utility(skill, skill.marginal_utility)
        self.skills.append(skill)
        return True

    def save(self) -> None:
        write_json(self.path, {"version": 1, "skills": self.skills})

    def stable_skills(self) -> list[Skill]:
        return [
            skill for skill in self.skills if skill.status == SkillStatus.VERIFIED
        ]

    def candidate_skills(self) -> list[Skill]:
        return [
            skill
            for skill in self.skills
            if skill.status in {SkillStatus.CANDIDATE, SkillStatus.PROVISIONAL}
        ]

    def credit_skills(self) -> list[Skill]:
        return [
            skill
            for skill in self.skills
            if skill.status
            in {SkillStatus.PROVISIONAL, SkillStatus.VERIFIED}
            and isinstance(skill.metadata.get("skill_credit"), dict)
        ]

    def aggregate_distribution(self) -> ExplorationDistribution:
        return aggregate_skill_bank_distribution(self.skills)

    def canonical_skill(self, candidate: Skill) -> Skill | None:
        """Return the persisted identity used by runtime and organization."""
        return self._find_duplicate(candidate)

    def _find_duplicate(self, candidate: Skill) -> Skill | None:
        candidate_text = f"{candidate.skill_name} {candidate.description}".lower()
        for existing in self.skills:
            if not self._metadata_compatible(existing, candidate):
                continue
            if (
                existing.capability_key
                and candidate.capability_key
                and existing.capability_key == candidate.capability_key
            ):
                return existing
            existing_text = f"{existing.skill_name} {existing.description}".lower()
            if SequenceMatcher(None, candidate_text, existing_text).ratio() >= self.redundancy_threshold:
                return existing
        return None

    @staticmethod
    def _metadata_compatible(existing: Skill, candidate: Skill) -> bool:
        """Only merge skills that summarize the same experience regime."""

        if existing.capability_key and candidate.capability_key:
            return existing.capability_key == candidate.capability_key
        for key in ("source_signal", "primary_task_family"):
            left = str(existing.metadata.get(key, "") or "")
            right = str(candidate.metadata.get(key, "") or "")
            if left and right and left != right:
                return False
        return True

    def _record_utility(self, skill: Skill, utility: float) -> None:
        """Track held-out MU history. Retirement is opt-in (retire_after_negative>0).

        Default is disabled so online credit/utility can decide prune/promote
        without held-out MU probes retiring unused skills.
        """
        value = float(utility)
        skill.utility_history.append(value)
        if value <= self.negative_utility_threshold:
            skill.consecutive_negative_evaluations += 1
            skill.reliability_weight *= self.negative_utility_decay
            if skill.status == SkillStatus.VERIFIED:
                skill.status = SkillStatus.PROVISIONAL
        else:
            skill.consecutive_negative_evaluations = 0
            skill.reliability_weight = min(
                1.0,
                skill.reliability_weight + 0.1,
            )
        if self.retire_after_negative <= 0:
            return
        if (
            skill.consecutive_negative_evaluations
            >= self.retire_after_negative
        ):
            skill.status = SkillStatus.RETIRED
            skill.metadata["retirement_reason"] = (
                "consecutive held-out marginal utility did not exceed "
                f"{self.negative_utility_threshold:.4f}"
            )

    def _read(self) -> list[dict[str, Any]]:
        import json

        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        return payload.get("skills", [])

    @staticmethod
    def _merge_fragments(existing, incoming):
        merged = list(existing)
        seen = {
            (fragment.trajectory_id, fragment.step_index, fragment.action)
            for fragment in merged
        }
        for fragment in incoming:
            key = (fragment.trajectory_id, fragment.step_index, fragment.action)
            if key in seen:
                continue
            seen.add(key)
            merged.append(fragment)
        return merged[:8]

    @staticmethod
    def _merge_distributions(
        left: ExplorationDistribution | None,
        right: ExplorationDistribution | None,
    ) -> ExplorationDistribution | None:
        if left is None:
            return right
        if right is None:
            return left

        def merge_counter(
            first: dict[str, int],
            second: dict[str, int],
        ) -> dict[str, int]:
            counter = Counter(first)
            counter.update(second)
            return dict(counter)

        total = left.total_trajectories + right.total_trajectories
        success = left.outcome_histogram.get("success", 0) + right.outcome_histogram.get(
            "success",
            0,
        )
        return ExplorationDistribution(
            task_family_histogram=merge_counter(
                left.task_family_histogram,
                right.task_family_histogram,
            ),
            outcome_histogram=merge_counter(
                left.outcome_histogram,
                right.outcome_histogram,
            ),
            signal_histogram=merge_counter(
                left.signal_histogram,
                right.signal_histogram,
            ),
            signal_cooccurrence=merge_counter(
                left.signal_cooccurrence,
                right.signal_cooccurrence,
            ),
            total_trajectories=total,
            success_rate=(success / total) if total else 0.0,
        )
