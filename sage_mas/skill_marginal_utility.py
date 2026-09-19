"""Paired with/without probes for skill marginal utility."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from sage_mas.alfworld_evaluator import (
    AlfWorldOrganizationEvaluator,
    EvaluationTrial,
)
from sage_mas.schemas import AgentSpec, Skill, SkillStatus


@dataclass(slots=True)
class MarginalUtilityProbeResult:
    skill_name: str
    n_tasks: int
    with_wins: int
    without_wins: int
    delta_sr: float
    delta_local_effect_rate: float
    ci_low: float
    ci_high: float
    accepted: bool
    reason: str
    gamefiles: list[str] = field(default_factory=list)
    with_won: list[bool] = field(default_factory=list)
    without_won: list[bool] = field(default_factory=list)
    with_trials: list[Any] = field(default_factory=list, repr=False, compare=False)
    without_trials: list[Any] = field(default_factory=list, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("with_trials", None)
        payload.pop("without_trials", None)
        return payload


def select_probe_gamefiles(
    trials: Sequence[EvaluationTrial],
    skill: Skill,
    *,
    max_tasks: int = 6,
) -> list[str]:
    """Prefer failed family-matched tasks from the latest play window."""
    families = set(skill.applicable_task_families or [])
    matched = [
        trial
        for trial in trials
        if not families or trial.task_family in families
    ]
    if not matched:
        matched = list(trials)
    failures = [trial for trial in matched if not trial.won]
    successes = [trial for trial in matched if trial.won]
    ordered = failures + successes
    selected: list[str] = []
    seen: set[str] = set()
    for trial in ordered:
        task_id = str(trial.task_id or "").strip()
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        selected.append(task_id)
        if len(selected) >= max(1, int(max_tasks)):
            break
    return selected


def bootstrap_paired_delta_ci(
    with_won: Sequence[bool],
    without_won: Sequence[bool],
    *,
    bootstrap_samples: int = 500,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean_delta, ci_low, ci_high) for paired win indicators."""
    pairs = list(zip(with_won, without_won))
    if not pairs:
        return 0.0, 0.0, 0.0
    diffs = [float(int(left) - int(right)) for left, right in pairs]
    mean_delta = sum(diffs) / len(diffs)
    if len(diffs) == 1 or bootstrap_samples <= 0:
        return mean_delta, mean_delta, mean_delta
    rng = random.Random(seed)
    samples: list[float] = []
    n = len(diffs)
    for _ in range(bootstrap_samples):
        draw = [diffs[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    alpha = max(0.0, min(1.0, 1.0 - confidence_level))
    low_i = int(alpha / 2.0 * (len(samples) - 1))
    high_i = int((1.0 - alpha / 2.0) * (len(samples) - 1))
    return mean_delta, samples[low_i], samples[high_i]


def trial_has_capability_effect(skill: Skill, trial: EvaluationTrial) -> bool:
    """Local effect proxy reusable by credit relative-baseline checks."""
    from sage_mas.skill_credit import trial_has_local_effect

    return trial_has_local_effect(skill, trial)


def probe_skill_marginal_utility(
    evaluator: AlfWorldOrganizationEvaluator,
    *,
    agents: list[AgentSpec],
    skills: list[Skill],
    skill: Skill,
    gamefiles: list[str],
    min_delta: float = 0.0,
    require_ci_above_zero: bool = False,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> MarginalUtilityProbeResult:
    """Dual-evaluate the same tasks with vs without ``skill`` injected."""
    gamefiles = [str(path) for path in gamefiles if str(path).strip()]
    if not gamefiles:
        return MarginalUtilityProbeResult(
            skill_name=skill.skill_name,
            n_tasks=0,
            with_wins=0,
            without_wins=0,
            delta_sr=0.0,
            delta_local_effect_rate=0.0,
            ci_low=0.0,
            ci_high=0.0,
            accepted=False,
            reason="no probe gamefiles",
        )

    previous_allow = bool(
        getattr(evaluator.config, "allow_executor_skill_injection", False)
    )
    evaluator.config.allow_executor_skill_injection = True
    # Paired probe: force-mount the skill on the "with" arm so the contrast
    # measures the skill itself, not whether the Executor retrieves it.
    previous_force = bool(getattr(evaluator.config, "force_inject_skills", False))
    if hasattr(evaluator.config, "force_inject_skills"):
        evaluator.config.force_inject_skills = True
    try:
        with_trials = evaluator.evaluate(
            agents=agents,
            skills=skills,
            gamefiles=gamefiles,
            condition=f"skill_mu_with:{skill.skill_name}",
            injected_skills=[skill],
        )
        without_trials = evaluator.evaluate(
            agents=agents,
            skills=skills,
            gamefiles=gamefiles,
            condition=f"skill_mu_without:{skill.skill_name}",
            injected_skills=[],
        )
    finally:
        evaluator.config.allow_executor_skill_injection = previous_allow
        if hasattr(evaluator.config, "force_inject_skills"):
            evaluator.config.force_inject_skills = previous_force

    with_by_id = {trial.task_id: trial for trial in with_trials}
    without_by_id = {trial.task_id: trial for trial in without_trials}
    paired_ids = [task_id for task_id in gamefiles if task_id in with_by_id and task_id in without_by_id]
    with_won = [bool(with_by_id[task_id].won) for task_id in paired_ids]
    without_won = [bool(without_by_id[task_id].won) for task_id in paired_ids]
    with_effect = [
        trial_has_capability_effect(skill, with_by_id[task_id])
        for task_id in paired_ids
    ]
    without_effect = [
        trial_has_capability_effect(skill, without_by_id[task_id])
        for task_id in paired_ids
    ]
    n = len(paired_ids)
    with_wins = sum(int(flag) for flag in with_won)
    without_wins = sum(int(flag) for flag in without_won)
    delta_sr, ci_low, ci_high = bootstrap_paired_delta_ci(
        with_won,
        without_won,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    delta_effect = (
        (
            sum(int(flag) for flag in with_effect)
            - sum(int(flag) for flag in without_effect)
        )
        / n
        if n
        else 0.0
    )
    # Non-negative ΔSR is enough; only harmful (negative) paired outcomes fail.
    accepted = delta_sr >= float(min_delta) or delta_effect > float(min_delta)
    if require_ci_above_zero:
        accepted = accepted and ci_low > 0.0
    if n <= 0:
        accepted = False
        reason = "no paired probe trials"
    elif accepted:
        reason = (
            f"paired delta_sr={delta_sr:.3f} "
            f"(with {with_wins}/{n}, without {without_wins}/{n}); "
            f"delta_local_effect={delta_effect:.3f}"
        )
    else:
        reason = (
            f"harmful paired MU: delta_sr={delta_sr:.3f}, "
            f"delta_local_effect={delta_effect:.3f}, "
            f"CI=[{ci_low:.3f}, {ci_high:.3f}]"
        )
    return MarginalUtilityProbeResult(
        skill_name=skill.skill_name,
        n_tasks=n,
        with_wins=with_wins,
        without_wins=without_wins,
        delta_sr=delta_sr,
        delta_local_effect_rate=delta_effect,
        ci_low=ci_low,
        ci_high=ci_high,
        accepted=accepted,
        reason=reason,
        gamefiles=list(paired_ids),
        with_won=with_won,
        without_won=without_won,
        with_trials=list(with_trials),
        without_trials=list(without_trials),
    )


def probe_three_way_skill_gain(
    evaluator: AlfWorldOrganizationEvaluator,
    *,
    executor_agents: list[AgentSpec],
    specialist_agents: list[AgentSpec],
    skills: list[Skill],
    skill: Skill,
    gamefiles: list[str],
) -> dict[str, Any]:
    """Compare Executor / Executor+skill / Specialist+skill on the same tasks.

    Used to separate skill uselessness from specialist prompt/role compile loss.
    Does not change accept gates.
    """
    gamefiles = [str(path) for path in gamefiles if str(path).strip()]
    if not gamefiles:
        return {
            "skill_name": skill.skill_name,
            "n_tasks": 0,
            "executor_sr": 0.0,
            "executor_skill_sr": 0.0,
            "specialist_sr": 0.0,
            "reason": "no probe gamefiles",
        }

    previous_allow = bool(
        getattr(evaluator.config, "allow_executor_skill_injection", False)
    )
    evaluator.config.allow_executor_skill_injection = True
    # Same forced-mount rationale as probe_skill_marginal_utility.
    previous_force = bool(getattr(evaluator.config, "force_inject_skills", False))
    if hasattr(evaluator.config, "force_inject_skills"):
        evaluator.config.force_inject_skills = True
    try:
        bare = evaluator.evaluate(
            agents=executor_agents,
            skills=skills,
            gamefiles=gamefiles,
            condition=f"three_way_executor:{skill.skill_name}",
            injected_skills=[],
        )
        with_skill = evaluator.evaluate(
            agents=executor_agents,
            skills=skills,
            gamefiles=gamefiles,
            condition=f"three_way_executor_skill:{skill.skill_name}",
            injected_skills=[skill],
        )
        specialist = evaluator.evaluate(
            agents=specialist_agents,
            skills=skills,
            gamefiles=gamefiles,
            condition=f"three_way_specialist:{skill.skill_name}",
            injected_skills=[],
        )
    finally:
        evaluator.config.allow_executor_skill_injection = previous_allow
        if hasattr(evaluator.config, "force_inject_skills"):
            evaluator.config.force_inject_skills = previous_force

    def _sr(trials: list[EvaluationTrial]) -> float:
        if not trials:
            return 0.0
        return sum(1 for trial in trials if trial.won) / len(trials)

    executor_sr = _sr(bare)
    executor_skill_sr = _sr(with_skill)
    specialist_sr = _sr(specialist)
    return {
        "skill_name": skill.skill_name,
        "n_tasks": len(gamefiles),
        "executor_sr": executor_sr,
        "executor_skill_sr": executor_skill_sr,
        "specialist_sr": specialist_sr,
        "skill_delta": executor_skill_sr - executor_sr,
        "specialist_delta": specialist_sr - executor_sr,
        "reason": (
            f"executor={executor_sr:.3f}, "
            f"executor+skill={executor_skill_sr:.3f}, "
            f"specialist={specialist_sr:.3f}"
        ),
    }


def apply_marginal_utility_probe(
    skill: Skill,
    result: MarginalUtilityProbeResult,
    *,
    promote_on_accept: bool = True,
) -> Skill:
    """Persist MU on the skill; optionally verify on accepted probes."""
    from sage_mas.skill_student_transfer import apply_student_mu_gate

    gated = apply_student_mu_gate(
        skill,
        delta_sr=float(result.delta_sr),
        min_delta=0.0,
        local_effect_delta=float(result.delta_local_effect_rate),
        promote_on_accept=promote_on_accept,
    )
    # Preserve probe payload on the gated copy; mutate caller's skill in place
    # so existing callers that keep the same object still see updates.
    skill.marginal_utility = gated.marginal_utility
    skill.status = gated.status
    md = dict(skill.metadata or {})
    md.update(gated.metadata or {})
    md["marginal_utility_probe"] = result.as_dict()
    # Honor probe.accepted if the helper and probe disagree on edge cases.
    if not result.accepted:
        md["inject_ready"] = False
        md["add_agent_ready"] = False
        md["mu_rejected"] = True
        md["mu_promoted"] = False
        md["executor_optional_injection"] = False
        md["not_for_add_agent"] = True
        if skill.status == SkillStatus.VERIFIED:
            skill.status = SkillStatus.PROVISIONAL
    skill.metadata = md
    return skill
