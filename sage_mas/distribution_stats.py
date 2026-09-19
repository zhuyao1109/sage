"""Exploration-distribution statistics and SkillBank shift estimation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from sage_mas.schemas import (
    AtomicStep,
    DistributionShift,
    ExperienceDistributionSnapshot,
    ExplorationDistribution,
    Skill,
)
from sage_mas.trajectory.abstraction import trajectory_id_from_steps
from sage_mas.trajectory.fragments import embed_skill


def _normalize_histogram(
    histogram: dict[str, int],
) -> dict[str, float]:
    total = float(sum(histogram.values()))
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in histogram.items()}


def kl_divergence(
    observed: dict[str, float],
    baseline: dict[str, float],
    *,
    epsilon: float = 1e-6,
) -> float:
    keys = set(observed) | set(baseline)
    if not keys:
        return 0.0
    divergence = 0.0
    for key in keys:
        p = observed.get(key, 0.0)
        q = baseline.get(key, epsilon)
        if p > 0.0:
            divergence += p * math.log((p + epsilon) / q)
    return max(0.0, divergence)


def compute_round_exploration_distribution(
    trajectories: list[list[AtomicStep]],
) -> ExplorationDistribution:
    task_family_histogram: Counter[str] = Counter()
    outcome_histogram: Counter[str] = Counter()
    for steps in trajectories:
        if not steps:
            continue
        terminal = steps[-1]
        family = str(terminal.metadata.get("task_family", "other")) or "other"
        task_family_histogram[family] += 1
        outcome = "success" if bool(terminal.metadata.get("won", False)) else "failure"
        outcome_histogram[outcome] += 1
    total = sum(task_family_histogram.values())
    success = outcome_histogram.get("success", 0)
    return ExplorationDistribution(
        task_family_histogram=dict(task_family_histogram),
        outcome_histogram=dict(outcome_histogram),
        total_trajectories=total,
        success_rate=(success / total) if total else 0.0,
    )


def compute_skill_exploration_distribution(
    trajectories: list[list[AtomicStep]],
    *,
    source_signal: str | None = None,
    signal_cooccurrence: dict[str, int] | None = None,
) -> ExplorationDistribution:
    task_family_histogram: Counter[str] = Counter()
    outcome_histogram: Counter[str] = Counter()
    signal_histogram: Counter[str] = Counter()
    for steps in trajectories:
        if not steps:
            continue
        terminal = steps[-1]
        family = str(terminal.metadata.get("task_family", "other")) or "other"
        task_family_histogram[family] += 1
        outcome = "success" if bool(terminal.metadata.get("won", False)) else "failure"
        outcome_histogram[outcome] += 1
    if source_signal:
        signal_histogram[source_signal] = len(trajectories)
    total = len(trajectories)
    success = outcome_histogram.get("success", 0)
    return ExplorationDistribution(
        task_family_histogram=dict(task_family_histogram),
        outcome_histogram=dict(outcome_histogram),
        signal_histogram=dict(signal_histogram),
        signal_cooccurrence=dict(signal_cooccurrence or {}),
        total_trajectories=total,
        success_rate=(success / total) if total else 0.0,
    )


def compute_signal_cooccurrence(skills: list[Skill]) -> dict[str, int]:
    signals = sorted(
        {
            str(skill.metadata.get("source_signal", "")).strip()
            for skill in skills
            if skill.metadata.get("source_signal")
        }
    )
    cooccurrence: dict[str, int] = defaultdict(int)
    for left_index, left in enumerate(signals):
        for right in signals[left_index + 1 :]:
            key = "|".join(sorted((left, right)))
            cooccurrence[key] += 1
    return dict(cooccurrence)


def aggregate_skill_bank_distribution(skills: list[Skill]) -> ExplorationDistribution:
    task_family_histogram: Counter[str] = Counter()
    outcome_histogram: Counter[str] = Counter()
    signal_histogram: Counter[str] = Counter()
    for skill in skills:
        distribution = skill.exploration_distribution
        if distribution is None:
            signal = str(skill.metadata.get("source_signal", "")).strip()
            if signal:
                signal_histogram[signal] += max(skill.support_count, 1)
            family = str(skill.metadata.get("primary_task_family", "other"))
            task_family_histogram[family] += max(skill.support_count, 1)
            continue
        for family, count in distribution.task_family_histogram.items():
            task_family_histogram[family] += count
        for outcome, count in distribution.outcome_histogram.items():
            outcome_histogram[outcome] += count
        for signal, count in distribution.signal_histogram.items():
            signal_histogram[signal] += count
    total = sum(task_family_histogram.values()) or sum(signal_histogram.values())
    success = outcome_histogram.get("success", 0)
    total_outcomes = sum(outcome_histogram.values())
    return ExplorationDistribution(
        task_family_histogram=dict(task_family_histogram),
        outcome_histogram=dict(outcome_histogram),
        signal_histogram=dict(signal_histogram),
        total_trajectories=total,
        success_rate=(success / total_outcomes) if total_outcomes else 0.0,
    )


def aggregate_skill_cluster_distribution(
    skills: list[Skill],
) -> ExplorationDistribution:
    """Merge per-skill exploration histograms inside one capability cluster.

    Histogram counts are summed across skills (not re-weighted by unique
    evidence). Support for ADD_AGENT gates is computed separately via
    unique ``evidence_ids``.
    """
    return aggregate_skill_bank_distribution(skills)


def shift_for_skill_cluster(
    skills: list[Skill],
    baseline: ExplorationDistribution | None,
    *,
    baseline_skills: list[Skill] | None = None,
) -> DistributionShift:
    """Capability-cluster shift relative to a frozen pre-round baseline.

    When historical skills are supplied, novelty is measured in the fixed
    skill-embedding space. The histogram implementation remains as a fallback
    for old artifacts and callers that do not have skill-level baselines.
    """
    if baseline_skills is not None:
        return embedding_novelty_for_skill_cluster(skills, baseline_skills)
    observed = aggregate_skill_cluster_distribution(skills)
    return estimate_distribution_shift(observed, baseline)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    similarity = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    ) / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _embedding_for_skill(skill: Skill) -> list[float]:
    if (
        not skill.embedding
        or skill.metadata.get("embedding_version")
        != "skill-behavior-hash-v2"
    ):
        skill.embedding = embed_skill(skill)
    return skill.embedding


def build_experience_distribution_snapshot(
    skills: list[Skill],
    historical_skills: list[Skill],
    *,
    round_id: str,
    novelty_threshold: float = 0.15,
    round_distribution: ExplorationDistribution | None = None,
) -> ExperienceDistributionSnapshot:
    """Persistable, selection-aware view of grounded experience prototypes."""
    prototypes = [
        skill
        for skill in skills
        if skill.status.value != "retired"
    ]
    history = [
        skill
        for skill in historical_skills
        if skill.status.value != "retired"
    ]
    capability_histogram: Counter[str] = Counter()
    status_histogram: Counter[str] = Counter()
    novelty_scores: list[float] = []
    prototype_records: list[dict[str, object]] = []
    dimensions = 0
    weighted_centroid: list[float] = []
    total_weight = 0.0
    novel_weight = 0.0
    failure_novel_weight = 0.0
    failure_weight = 0.0

    for skill in prototypes:
        embedding = _embedding_for_skill(skill)
        dimensions = max(dimensions, len(embedding))
        if len(weighted_centroid) < len(embedding):
            weighted_centroid.extend(
                [0.0] * (len(embedding) - len(weighted_centroid))
            )
        weight = float(
            max(skill.support_count, len(skill.evidence_ids), 1)
        )
        capability = (
            skill.capability_key
            or str(skill.metadata.get("source_signal", "unspecified"))
        )
        capability_histogram[capability] += int(weight)
        status_histogram[skill.status.value] += 1
        for index, value in enumerate(embedding):
            weighted_centroid[index] += value * weight
        total_weight += weight

        if history:
            nearest_similarity = max(
                cosine_similarity(
                    embedding,
                    _embedding_for_skill(existing),
                )
                for existing in history
            )
            novelty = (1.0 - nearest_similarity) / 2.0
        else:
            nearest_similarity = None
            novelty = 1.0
        novelty_scores.append(novelty)
        if novelty > novelty_threshold:
            novel_weight += weight

        distribution = skill.exploration_distribution
        failure_count = (
            int(distribution.outcome_histogram.get("failure", 0))
            if distribution is not None
            else 0
        )
        outcome_count = (
            sum(distribution.outcome_histogram.values())
            if distribution is not None
            else 0
        )
        failure_fraction = (
            failure_count / outcome_count if outcome_count else 0.0
        )
        failure_support = weight * failure_fraction
        failure_weight += failure_support
        if novelty > novelty_threshold:
            failure_novel_weight += failure_support
        prototype_records.append(
            {
                "skill_id": skill.skill_id,
                "skill_name": skill.skill_name,
                "capability_key": capability,
                "status": skill.status.value,
                "support": int(weight),
                "novelty": novelty,
                "nearest_historical_similarity": nearest_similarity,
                "failure_fraction": failure_fraction,
                "task_families": list(skill.applicable_task_families),
                "evidence_ids": list(skill.evidence_ids),
                "embedding": list(embedding),
            }
        )

    centroid = [
        value / max(total_weight, 1.0) for value in weighted_centroid
    ]
    centroid_norm = math.sqrt(sum(value * value for value in centroid))
    if centroid_norm > 0.0:
        centroid = [value / centroid_norm for value in centroid]
    raw = round_distribution or aggregate_skill_bank_distribution(prototypes)
    total_outcomes = sum(raw.outcome_histogram.values())
    return ExperienceDistributionSnapshot(
        round_id=round_id,
        representation_version="experience-skill-prototypes-v1",
        prototype_count=len(prototypes),
        total_support=int(total_weight),
        task_family_histogram=dict(raw.task_family_histogram),
        outcome_histogram=dict(raw.outcome_histogram),
        capability_histogram=dict(capability_histogram),
        status_histogram=dict(status_histogram),
        centroid=centroid,
        novelty_scores=novelty_scores,
        mean_novelty=(
            sum(novelty_scores) / len(novelty_scores)
            if novelty_scores
            else 0.0
        ),
        novelty_p50=_percentile(novelty_scores, 0.5),
        novelty_p90=_percentile(novelty_scores, 0.9),
        novel_mass=novel_weight / max(total_weight, 1.0),
        failure_novel_mass=(
            failure_novel_weight / failure_weight
            if failure_weight
            else 0.0
        ),
        novelty_threshold=novelty_threshold,
        historical_prototype_count=len(history),
        success_rate=(
            raw.outcome_histogram.get("success", 0) / total_outcomes
            if total_outcomes
            else raw.success_rate
        ),
        prototypes=prototype_records,
    )


def compare_experience_distribution_snapshots(
    old: ExperienceDistributionSnapshot,
    new: ExperienceDistributionSnapshot,
) -> dict[str, float | int | str]:
    def total_variation(
        left: dict[str, int],
        right: dict[str, int],
    ) -> float:
        left_probabilities = _normalize_histogram(left)
        right_probabilities = _normalize_histogram(right)
        return 0.5 * sum(
            abs(
                left_probabilities.get(key, 0.0)
                - right_probabilities.get(key, 0.0)
            )
            for key in set(left_probabilities) | set(right_probabilities)
        )

    centroid_distance = (
        (1.0 - cosine_similarity(old.centroid, new.centroid)) / 2.0
        if old.centroid and new.centroid
        else 0.0
    )
    return {
        "metric": "paired_experience_distribution_v1",
        "centroid_distance": centroid_distance,
        "task_family_total_variation": total_variation(
            old.task_family_histogram,
            new.task_family_histogram,
        ),
        "outcome_total_variation": total_variation(
            old.outcome_histogram,
            new.outcome_histogram,
        ),
        "capability_total_variation": total_variation(
            old.capability_histogram,
            new.capability_histogram,
        ),
        "novel_mass_delta": new.novel_mass - old.novel_mass,
        "failure_novel_mass_delta": (
            new.failure_novel_mass - old.failure_novel_mass
        ),
        "prototype_count_delta": (
            new.prototype_count - old.prototype_count
        ),
        "success_rate_delta": new.success_rate - old.success_rate,
    }


def _cluster_failure_rate(skills: list[Skill]) -> float:
    failure = 0
    total = 0
    for skill in skills:
        distribution = skill.exploration_distribution
        if distribution is None:
            continue
        failure += int(distribution.outcome_histogram.get("failure", 0))
        total += sum(distribution.outcome_histogram.values())
    return failure / total if total else 0.0


def embedding_novelty_for_skill_cluster(
    skills: list[Skill],
    baseline_skills: list[Skill],
) -> DistributionShift:
    """Distance from each candidate to its nearest historical Skill.

    The baseline list must be captured before adding current-round candidates.
    Distances are support-weighted and normalized to [0, 1] as
    ``(1 - cosine_similarity) / 2``.
    """
    candidates = [
        skill
        for skill in skills
        if skill.status.value not in {"rejected", "retired"}
    ]
    historical = [
        skill
        for skill in baseline_skills
        if skill.status.value not in {"rejected", "retired"}
    ]
    failure_rate = _cluster_failure_rate(candidates)
    if not candidates:
        return DistributionShift(
            baseline_source="embedding_skill_bank",
            metric="cosine_nearest_skill",
            baseline_size=len(historical),
            failure_rate=failure_rate,
        )
    if not historical:
        return DistributionShift(
            kl_divergence=1.0,
            shift_score=1.0,
            baseline_source="empty_skill_bank",
            metric="cosine_nearest_skill",
            nearest_similarity=None,
            novelty_score=1.0,
            baseline_size=0,
            failure_rate=failure_rate,
        )

    weighted_novelty = 0.0
    total_weight = 0.0
    nearest_pair: tuple[float, Skill] | None = None
    for candidate in candidates:
        candidate_embedding = _embedding_for_skill(candidate)
        similarities = [
            (
                cosine_similarity(
                    candidate_embedding,
                    _embedding_for_skill(existing),
                ),
                existing,
            )
            for existing in historical
        ]
        similarity, nearest = max(similarities, key=lambda item: item[0])
        weight = float(max(candidate.support_count, len(candidate.evidence_ids), 1))
        novelty = (1.0 - similarity) / 2.0
        weighted_novelty += novelty * weight
        total_weight += weight
        if nearest_pair is None or similarity > nearest_pair[0]:
            nearest_pair = (similarity, nearest)

    novelty_score = weighted_novelty / max(total_weight, 1.0)
    nearest_similarity, nearest_skill = nearest_pair or (0.0, historical[0])
    return DistributionShift(
        kl_divergence=0.0,
        shift_score=novelty_score,
        baseline_source="embedding_skill_bank",
        metric="cosine_nearest_skill",
        nearest_skill_id=nearest_skill.skill_id,
        nearest_skill_name=nearest_skill.skill_name,
        nearest_similarity=nearest_similarity,
        novelty_score=novelty_score,
        baseline_size=len(historical),
        failure_rate=failure_rate,
    )


def estimate_distribution_shift(
    observed: ExplorationDistribution,
    baseline: ExplorationDistribution | None,
    *,
    baseline_source: str = "skill_bank",
) -> DistributionShift:
    if baseline is None or baseline.total_trajectories <= 0:
        observed_family = _normalize_histogram(observed.task_family_histogram)
        return DistributionShift(
            kl_divergence=1.0 if observed_family else 0.0,
            shift_score=1.0 if observed_family else 0.0,
            task_family_delta=dict(observed.task_family_histogram),
            baseline_source="empty_skill_bank",
        )
    observed_family = _normalize_histogram(observed.task_family_histogram)
    baseline_family = _normalize_histogram(baseline.task_family_histogram)
    divergence = kl_divergence(observed_family, baseline_family)
    task_family_delta = {
        family: observed_family.get(family, 0.0) - baseline_family.get(family, 0.0)
        for family in set(observed_family) | set(baseline_family)
    }
    shift_score = min(
        1.0,
        divergence / math.log(max(len(set(observed_family) | set(baseline_family)), 2) + 1),
    )
    return DistributionShift(
        kl_divergence=divergence,
        shift_score=shift_score,
        task_family_delta=task_family_delta,
        baseline_source=baseline_source,
    )


def attach_distribution_shift(
    skill: Skill,
    *,
    baseline: ExplorationDistribution | None,
    signal_cooccurrence: dict[str, int] | None = None,
) -> Skill:
    trajectories_count = max(skill.support_count, len(skill.evidence_ids))
    if skill.exploration_distribution is None:
        skill.exploration_distribution = ExplorationDistribution(
            task_family_histogram={
                str(skill.metadata.get("primary_task_family", "other")): trajectories_count
            },
            signal_histogram={
                str(skill.metadata.get("source_signal", "")): trajectories_count
            }
            if skill.metadata.get("source_signal")
            else {},
            total_trajectories=trajectories_count,
        )
    if signal_cooccurrence:
        skill.exploration_distribution.signal_cooccurrence = dict(signal_cooccurrence)
    skill.distribution_shift = estimate_distribution_shift(
        skill.exploration_distribution,
        baseline,
    )
    return skill


def evidence_trajectories_for_skill(
    skill: Skill,
    trajectories_by_id: dict[str, list[AtomicStep]],
) -> list[list[AtomicStep]]:
    trajectories: list[list[AtomicStep]] = []
    for evidence_id in skill.evidence_ids:
        steps = trajectories_by_id.get(evidence_id)
        if steps:
            trajectories.append(steps)
    return trajectories


def index_trajectory_steps(
    trajectories: list[list[AtomicStep]],
) -> dict[str, list[AtomicStep]]:
    indexed: dict[str, list[AtomicStep]] = {}
    for steps in trajectories:
        if not steps:
            continue
        indexed[trajectory_id_from_steps(steps)] = steps
    return indexed
