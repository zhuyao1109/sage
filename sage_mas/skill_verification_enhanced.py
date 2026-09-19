"""Enhanced skill verification with multi-tiered validation strategy.

Problem: Current verification is too strict, causing most skills to be rejected.
Solution: Use a tiered approach:
  - Tier 1 (Auto-accept): Strong model success + structural quality
  - Tier 2 (Quick probe): 3-5 episodes with weak model, 40%+ SR → provisional
  - Tier 3 (Credit-based): Online usage tracking, promote at 60%+ success rate

This allows more skills to enter the bank for online testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sage_mas.schemas import Skill, SkillStatus


@dataclass(slots=True)
class VerificationResult:
    """Result of skill verification."""
    status: SkillStatus
    confidence: float  # 0.0 to 1.0
    reason: str
    tier: str  # "auto_accept", "quick_probe", "credit_based", "rejected"
    recommendations: list[str]
    should_probe: bool = False
    probe_budget: int = 0  # Number of episodes to test


class EnhancedSkillVerifier:
    """Multi-tiered skill verification for better bank coverage."""

    def __init__(
        self,
        *,
        auto_accept_teacher_sr_threshold: float = 0.85,
        quick_probe_episodes: int = 5,
        quick_probe_sr_threshold: float = 0.40,
        min_protocol_steps: int = 3,
        max_protocol_steps: int = 20,
        require_structural_quality: bool = True,
    ):
        self.auto_accept_teacher_sr = auto_accept_teacher_sr_threshold
        self.quick_probe_episodes = quick_probe_episodes
        self.quick_probe_sr = quick_probe_sr_threshold
        self.min_protocol_steps = min_protocol_steps
        self.max_protocol_steps = max_protocol_steps
        self.require_structural_quality = require_structural_quality

    def verify(
        self,
        skill: Skill,
        *,
        teacher_trajectories: list[dict[str, Any]] | None = None,
    ) -> VerificationResult:
        """
        Verify a skill using multi-tiered strategy.

        Returns a verification result with recommended next steps.
        """
        # Check structural quality first
        structural_issues = self._check_structural_quality(skill)
        if structural_issues and self.require_structural_quality:
            return VerificationResult(
                status=SkillStatus.REJECTED,
                confidence=0.0,
                reason="; ".join(structural_issues),
                tier="rejected",
                recommendations=["Fix protocol structure before re-attempting"],
            )

        # Calculate teacher success metrics
        teacher_metrics = self._calculate_teacher_metrics(
            skill,
            teacher_trajectories or [],
        )

        # Tier 1: Auto-accept high-quality teacher skills
        if self._should_auto_accept(skill, teacher_metrics):
            return VerificationResult(
                status=SkillStatus.PROVISIONAL,
                confidence=0.85,
                reason=(
                    f"Teacher SR {teacher_metrics['success_rate']:.1%} "
                    f"({teacher_metrics['wins']}/{teacher_metrics['total']}); "
                    "high-quality protocol auto-accepted"
                ),
                tier="auto_accept",
                recommendations=[
                    "Monitor online credit during execution",
                    "Promote to verified at 60%+ credit success rate",
                ],
            )

        # Tier 2: Quick probe for medium-quality skills
        if self._should_quick_probe(skill, teacher_metrics):
            return VerificationResult(
                status=SkillStatus.CANDIDATE,
                confidence=0.50,
                reason=(
                    f"Teacher SR {teacher_metrics['success_rate']:.1%}; "
                    "quality acceptable, needs weak model probe"
                ),
                tier="quick_probe",
                recommendations=[
                    f"Run {self.quick_probe_episodes} episodes with weak model",
                    f"Accept if SR >= {self.quick_probe_sr:.0%}",
                ],
                should_probe=True,
                probe_budget=self.quick_probe_episodes,
            )

        # Tier 3: Credit-based for marginal skills
        if self._should_credit_track(skill, teacher_metrics):
            return VerificationResult(
                status=SkillStatus.CANDIDATE,
                confidence=0.30,
                reason=(
                    f"Teacher SR {teacher_metrics['success_rate']:.1%}; "
                    "marginal quality, will track via online credit"
                ),
                tier="credit_based",
                recommendations=[
                    "Add to bank as candidate",
                    "Track online usage and effect-success rate",
                    "Promote at 60%+ credit, prune at <20%",
                ],
            )

        # Reject low-quality skills
        return VerificationResult(
            status=SkillStatus.REJECTED,
            confidence=0.0,
            reason=(
                f"Teacher SR {teacher_metrics['success_rate']:.1%} too low; "
                + ("; ".join(structural_issues) if structural_issues else "quality insufficient")
            ),
            tier="rejected",
            recommendations=["Collect more winning trajectories before re-attempting"],
        )

    def _check_structural_quality(self, skill: Skill) -> list[str]:
        """Check if skill has valid structure (protocol length, placeholders, etc.)."""
        issues: list[str] = []

        protocol = skill.action_protocol or []
        if not protocol:
            issues.append("Empty action protocol")
            return issues

        protocol_length = len(protocol)
        if protocol_length < self.min_protocol_steps:
            issues.append(
                f"Protocol too short ({protocol_length} < {self.min_protocol_steps})"
            )
        if protocol_length > self.max_protocol_steps:
            issues.append(
                f"Protocol too long ({protocol_length} > {self.max_protocol_steps})"
            )

        # Check for placeholder balance (not all placeholders, not all literals)
        protocol_text = " ".join(protocol).lower()
        has_placeholders = any(
            f"<{slot}>" in protocol_text
            for slot in ["target", "source", "destination", "location", "tool", "object"]
        )

        if not has_placeholders:
            # Check if it's overfit to specific instances (e.g., "tomato 1", "cabinet 3")
            import re
            instance_count = len(re.findall(r"\b[a-z]+\s+\d+\b", protocol_text))
            if instance_count > protocol_length * 0.5:
                issues.append("Protocol overfit to specific instances (e.g., 'tomato 1')")

        # Check for essential verbs
        essential_verbs = {"go", "take", "move", "open", "close", "cool", "heat", "clean", "use"}
        found_verbs = set()
        for step in protocol:
            step_lower = step.lower()
            for verb in essential_verbs:
                if step_lower.startswith(verb):
                    found_verbs.add(verb)

        if not found_verbs:
            issues.append("No recognizable action verbs in protocol")

        return issues

    def _calculate_teacher_metrics(
        self,
        skill: Skill,
        trajectories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate success metrics from teacher trajectories."""
        # Use skill's own evidence IDs
        evidence_ids = set(skill.evidence_ids)

        relevant_trajs = [
            traj for traj in trajectories
            if traj.get("trajectory_id") in evidence_ids
        ]

        if not relevant_trajs:
            # Fallback to skill metadata
            support = skill.support_count or 0
            distribution = skill.exploration_distribution
            if distribution:
                total = distribution.total_trajectories
                wins = int(total * distribution.success_rate) if total > 0 else 0
                return {
                    "total": total,
                    "wins": wins,
                    "success_rate": distribution.success_rate,
                    "source": "exploration_distribution",
                }
            return {
                "total": support,
                "wins": support,
                "success_rate": 1.0 if support > 0 else 0.0,
                "source": "support_count_fallback",
            }

        total = len(relevant_trajs)
        wins = sum(
            1 for traj in relevant_trajs
            if traj.get("won", False) or traj.get("reward", 0) > 0
        )

        return {
            "total": total,
            "wins": wins,
            "success_rate": wins / total if total > 0 else 0.0,
            "source": "trajectory_evidence",
        }

    def _should_auto_accept(
        self,
        skill: Skill,
        teacher_metrics: dict[str, Any],
    ) -> bool:
        """Decide if skill should be auto-accepted as provisional."""
        # High teacher success rate + reasonable support
        if (
            teacher_metrics["success_rate"] >= self.auto_accept_teacher_sr
            and teacher_metrics["total"] >= 3
        ):
            return True

        # Or very high support count (merged skill)
        if skill.support_count >= 10 and teacher_metrics["success_rate"] >= 0.75:
            return True

        return False

    def _should_quick_probe(
        self,
        skill: Skill,
        teacher_metrics: dict[str, Any],
    ) -> bool:
        """Decide if skill should get a quick probe test."""
        # Medium teacher success rate
        if (
            0.60 <= teacher_metrics["success_rate"] < self.auto_accept_teacher_sr
            and teacher_metrics["total"] >= 2
        ):
            return True

        # Or medium support with decent success
        if (
            5 <= skill.support_count < 10
            and teacher_metrics["success_rate"] >= 0.70
        ):
            return True

        return False

    def _should_credit_track(
        self,
        skill: Skill,
        teacher_metrics: dict[str, Any],
    ) -> bool:
        """Decide if skill should be tracked via online credit (most permissive)."""
        # Any skill with at least 1 success and reasonable structure
        if teacher_metrics["wins"] >= 1 and teacher_metrics["total"] >= 1:
            return True

        # Or any skill with minimal support
        if skill.support_count >= 1:
            return True

        return False


def batch_verify_skills(
    skills: list[Skill],
    *,
    teacher_trajectories: list[dict[str, Any]] | None = None,
    verifier: EnhancedSkillVerifier | None = None,
) -> dict[str, list[tuple[Skill, VerificationResult]]]:
    """
    Verify a batch of skills and group by tier.

    Returns:
        Dict with keys: "auto_accept", "quick_probe", "credit_based", "rejected"
        Each value is a list of (skill, verification_result) tuples
    """
    if verifier is None:
        verifier = EnhancedSkillVerifier()

    results: dict[str, list[tuple[Skill, VerificationResult]]] = {
        "auto_accept": [],
        "quick_probe": [],
        "credit_based": [],
        "rejected": [],
    }

    for skill in skills:
        result = verifier.verify(skill, teacher_trajectories=teacher_trajectories)

        # Update skill status based on verification
        skill.status = result.status

        # Add to appropriate tier
        results[result.tier].append((skill, result))

    return results


def print_verification_report(
    results: dict[str, list[tuple[Skill, VerificationResult]]],
) -> None:
    """Print a human-readable verification report."""
    total = sum(len(items) for items in results.values())

    print(f"\n{'='*60}")
    print(f"Skill Verification Report (Total: {total})")
    print(f"{'='*60}\n")

    for tier, items in results.items():
        if not items:
            continue

        print(f"{tier.upper().replace('_', ' ')} ({len(items)}):")
        print("-" * 60)

        for skill, result in items:
            print(f"  • {skill.skill_name}")
            print(f"    Status: {result.status.value}")
            print(f"    Confidence: {result.confidence:.0%}")
            print(f"    Reason: {result.reason}")
            if result.should_probe:
                print(f"    → Needs {result.probe_budget} test episodes")
            if result.recommendations:
                print(f"    Recommendations:")
                for rec in result.recommendations:
                    print(f"      - {rec}")
            print()

    print(f"{'='*60}")
    print(f"Summary: "
          f"{len(results['auto_accept'])} auto-accepted, "
          f"{len(results['quick_probe'])} need probing, "
          f"{len(results['credit_based'])} credit-tracked, "
          f"{len(results['rejected'])} rejected")
    print(f"{'='*60}\n")
