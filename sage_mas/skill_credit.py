"""Smoothed online effect-success rates for grounded skills."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any

from sage_mas.alfworld_evaluator import EvaluationTrial
from sage_mas.discovery_fork import CapabilityTransitionDetector
from sage_mas.schemas import AgentSpec, Skill, SkillStatus

# Abstract protocols use "move"; ALFWorld often emits "put ... in/on ...".
_OPERATION_ALIASES: dict[str, frozenset[str]] = {
    "move": frozenset({"move", "put"}),
    "put": frozenset({"move", "put"}),
}
# Navigation alone is too weak to count as "using" a staged protocol.
_WEAK_PROTOCOL_OPS = frozenset({"go", "look", "examine", "inventory"})


@dataclass(slots=True)
class SkillCreditPolicy:
    verify_score: float = 0.60
    prune_score: float = 0.20
    min_uses_for_promotion: int = 1
    min_uses_for_pruning: int = 3
    # When true, credit score alone cannot VERIFIED without positive MU.
    require_positive_mu_for_verify: bool = False
    min_marginal_utility: float = 0.0
    # Fraction of productive protocol ops that must match in order.
    min_protocol_coverage: float = 0.5
    # Count credit success only when control lacked the same local effect.
    relative_credit_to_baseline: bool = False
    # Trajectory-derived executable protocol adherence (0-1).
    min_adherence_for_use: float = 0.34
    min_adherence_for_verify: float = 0.34
    # Prefer episode win as task-benefit signal when available.
    require_task_win_for_success: bool = False
    combine_adherence_with_effect: bool = True


def skill_credit_policy_from_mapping(
    mapping: dict[str, Any],
) -> SkillCreditPolicy:
    return SkillCreditPolicy(
        verify_score=float(mapping.get("verify_score", 0.60)),
        prune_score=float(
            mapping.get("prune_score", mapping.get("demote_score", 0.20))
        ),
        min_uses_for_promotion=int(
            mapping.get("min_uses_for_promotion", 1)
        ),
        min_uses_for_pruning=int(
            mapping.get("min_uses_for_pruning", 3)
        ),
        require_positive_mu_for_verify=bool(
            mapping.get("require_positive_mu_for_verify", False)
        ),
        min_marginal_utility=float(
            mapping.get("min_marginal_utility", 0.0)
        ),
        min_protocol_coverage=float(
            mapping.get("min_protocol_coverage", 0.5)
        ),
        relative_credit_to_baseline=bool(
            mapping.get(
                "relative_credit_to_baseline",
                mapping.get("relative_to_baseline", False),
            )
        ),
        min_adherence_for_use=float(
            mapping.get("min_adherence_for_use", 0.34)
        ),
        min_adherence_for_verify=float(
            mapping.get("min_adherence_for_verify", 0.34)
        ),
        require_task_win_for_success=bool(
            mapping.get("require_task_win_for_success", False)
        ),
        combine_adherence_with_effect=bool(
            mapping.get("combine_adherence_with_effect", True)
        ),
    )


def initialize_skill_credit(
    skill: Skill,
    policy: SkillCreditPolicy,
) -> dict[str, Any]:
    existing = skill.metadata.get("skill_credit")
    if isinstance(existing, dict):
        normalized = _normalized_credit(existing)
        existing.clear()
        existing.update(normalized)
        skill.metadata.setdefault("utility", float(normalized["score"]))
        return existing
    # Prior utility = (0+1)/(0+2) = 0.5 smoothed success rate.
    prior = _smoothed_success_rate(0, 0)
    credit = {
        "uses": 0,
        "successes": 0,
        "score": prior,
        "events": [],
    }
    skill.metadata["skill_credit"] = credit
    skill.metadata["utility"] = float(prior)
    skill.status = SkillStatus.PROVISIONAL
    return credit


def credit_managed_skills(skills: list[Skill]) -> list[Skill]:
    return [
        skill
        for skill in skills
        if skill.status not in {SkillStatus.REJECTED, SkillStatus.RETIRED}
        and isinstance(skill.metadata.get("skill_credit"), dict)
    ]


def update_skill_credits(
    skills: list[Skill],
    trials: list[EvaluationTrial],
    agents: list[AgentSpec],
    policy: SkillCreditPolicy,
    control_trials: list[EvaluationTrial] | None = None,
) -> dict[str, list[Skill]]:
    promoted: list[Skill] = []
    demoted: list[Skill] = []
    retired: list[Skill] = []
    updated: list[Skill] = []
    control_by_task = {
        trial.task_id: trial
        for trial in (control_trials or [])
        if trial.task_id
    }
    owners = {
        skill.skill_name: {
            agent.name
            for agent in agents
            if skill.skill_name in agent.assigned_skills
        }
        for skill in skills
    }
    for skill in credit_managed_skills(skills):
        before = skill.status
        credit = initialize_skill_credit(skill, policy)
        preserved_mu = skill.marginal_utility
        if preserved_mu is None and skill.metadata.get("marginal_utility") is not None:
            try:
                preserved_mu = float(skill.metadata["marginal_utility"])
            except (TypeError, ValueError):
                preserved_mu = None
        for trial in trials:
            family_eligible = (
                not skill.applicable_task_families
                or trial.task_family in skill.applicable_task_families
            )
            if not family_eligible:
                continue
            assigned_owners = owners.get(skill.skill_name, set())
            activated = skill.skill_name in trial.activated_skill_names
            selected = activated and (
                not assigned_owners
                or (
                    trial.assigned_primary_agent in assigned_owners
                    and int(
                        trial.actions_by_agent.get(
                            str(trial.assigned_primary_agent),
                            0,
                        )
                    )
                    > 0
                )
            )
            if not selected:
                continue
            from sage_mas.executable_protocol import (
                ensure_executable_protocol,
                protocol_adherence_score,
                protocol_after_activation,
            )

            ensure_executable_protocol(skill)
            activation_step = max(
                1,
                int(trial.skill_activation_steps.get(skill.skill_name, 1)),
            )
            adherence = protocol_adherence_score(
                skill,
                trial.steps,
                activation_step=activation_step,
            )
            used = _trial_executes_skill(
                skill,
                trial,
                min_coverage=policy.min_protocol_coverage,
            ) and adherence >= float(policy.min_adherence_for_use)
            if used:
                effect_ok = _relative_credit_success(
                    skill,
                    trial,
                    control_by_task=control_by_task,
                    relative=policy.relative_credit_to_baseline,
                )
                win_ok = bool(getattr(trial, "won", False))
                if policy.require_task_win_for_success:
                    succeeded = win_ok and (
                        effect_ok if policy.combine_adherence_with_effect else True
                    )
                elif policy.combine_adherence_with_effect:
                    succeeded = effect_ok or win_ok
                else:
                    succeeded = effect_ok
                # Adherence must stay above verify floor for a credited success.
                if succeeded and adherence < float(policy.min_adherence_for_verify):
                    succeeded = False
                _record_use(
                    credit,
                    succeeded=succeeded,
                    task_id=trial.task_id,
                    extra={
                        "protocol_adherence": adherence,
                        "activation_step": activation_step,
                        "entry_step_index": len(ensure_executable_protocol(skill)) - len(
                            protocol_after_activation(skill, trial.steps, activation_step=activation_step)
                        ),
                        "won": win_ok,
                    },
                )
                credit.setdefault("adherence_scores", []).append(adherence)
                continue
        score = float(credit["score"])
        use_count = int(credit["uses"])
        adherence_scores = credit.get("adherence_scores") or []
        # None means "no adherence evidence yet" (bootstrap-friendly). Do not
        # write 0.0 for unused skills; that falsely blocks ADD_AGENT.
        if adherence_scores:
            mean_adherence = sum(float(value) for value in adherence_scores) / len(
                adherence_scores
            )
            credit["mean_protocol_adherence"] = mean_adherence
        else:
            mean_adherence = None
            credit["mean_protocol_adherence"] = None
        if (
            use_count >= policy.min_uses_for_pruning
            and score < policy.prune_score
        ):
            skill.status = SkillStatus.RETIRED
            skill.metadata["retirement_reason"] = (
                f"smoothed effect success rate {score:.3f} below "
                f"{policy.prune_score:.3f} after {use_count} uses"
            )
        elif before == SkillStatus.VERIFIED:
            skill.status = SkillStatus.VERIFIED
        elif (
            score >= policy.verify_score
            and use_count >= policy.min_uses_for_promotion
            and (
                mean_adherence is None
                or mean_adherence >= float(policy.min_adherence_for_verify)
            )
            and _mu_allows_verification(skill, policy, preserved_mu)
        ):
            skill.status = SkillStatus.VERIFIED
        else:
            skill.status = SkillStatus.PROVISIONAL
        # Online utility is the credit score. Preserve paired MU only when the
        # policy still requires it for verification; otherwise utility drives
        # promote / retire without offline probes.
        from sage_mas.skill_protocol_distill import sync_utility_from_credit

        if policy.require_positive_mu_for_verify and preserved_mu is not None:
            skill.marginal_utility = float(preserved_mu)
            skill.metadata["marginal_utility"] = float(preserved_mu)
        else:
            sync_utility_from_credit(skill)
        updated.append(skill)
        if before != SkillStatus.VERIFIED and skill.status == SkillStatus.VERIFIED:
            skill.metadata["credit_promoted_pending_org"] = True
            promoted.append(skill)
        if before != SkillStatus.RETIRED and skill.status == SkillStatus.RETIRED:
            retired.append(skill)
    return {
        "updated": updated,
        "promoted": promoted,
        "demoted": demoted,
        "retired": retired,
    }


def _mu_allows_verification(
    skill: Skill,
    policy: SkillCreditPolicy,
    preserved_mu: float | None,
) -> bool:
    if not policy.require_positive_mu_for_verify:
        return True
    mu = preserved_mu
    if mu is None:
        mu = skill.marginal_utility
    if mu is None and skill.metadata.get("marginal_utility") is not None:
        try:
            mu = float(skill.metadata["marginal_utility"])
        except (TypeError, ValueError):
            mu = None
    if mu is None:
        return False
    return float(mu) >= float(policy.min_marginal_utility)


def _relative_credit_success(
    skill: Skill,
    trial: EvaluationTrial,
    *,
    control_by_task: dict[str, EvaluationTrial],
    relative: bool,
) -> bool:
    local = _has_local_effect(skill, trial)
    if not relative:
        return local
    if not local:
        return False
    if not control_by_task:
        # Control pass missing this round; fall back to absolute local effect.
        return local
    control = control_by_task.get(trial.task_id)
    if control is None:
        # No paired baseline for this task → do not claim skill credit.
        return False
    return not _has_local_effect(skill, control)


def _trial_executes_skill(
    skill: Skill,
    trial: EvaluationTrial,
    *,
    min_coverage: float = 0.5,
) -> bool:
    """True when post-activation actions cover enough protocol ops in order."""
    activation_step = max(
        1,
        int(trial.skill_activation_steps.get(skill.skill_name, 1)),
    )
    from sage_mas.executable_protocol import protocol_after_activation

    remaining = protocol_after_activation(skill, trial.steps, activation_step=activation_step)
    if not remaining:
        return False
    operations = _skill_execution_operations(
        skill, protocol=[step.action_template for step in remaining]
    )
    if not operations:
        return False
    steps = trial.steps[activation_step - 1 :]
    matched = 0
    cursor = 0
    for operation in operations:
        found = False
        while cursor < len(steps):
            if _action_matches_operation(steps[cursor].get("action", ""), operation):
                matched += 1
                cursor += 1
                found = True
                break
            cursor += 1
        if not found:
            # Continue matching later ops only if we already advanced; keep
            # greedy in-order coverage semantics by stopping this op.
            continue
    coverage = matched / len(operations)
    if coverage < float(min_coverage):
        return False
    # Capability marker (clean/heat/cool/desklamp) must appear when known.
    explicit = _canonical_operation(
        str(skill.metadata.get("capability_operation", "") or "")
    )
    if not explicit and skill.capability_key:
        key = str(skill.capability_key)
        if key.startswith("transform."):
            explicit = key.split(".", 1)[1]
        elif key in {"inspect.with_light", "inspect.light"}:
            explicit = "use"
    if explicit and explicit not in _WEAK_PROTOCOL_OPS:
        return any(
            _action_matches_operation(step.get("action", ""), explicit)
            or (
                explicit == "use"
                and "desklamp" in _environment_action(step.get("action", ""))
            )
            for step in steps
        )
    return True


def trial_has_local_effect(skill: Skill, trial: EvaluationTrial) -> bool:
    """Public wrapper for paired MU / relative-credit checks."""
    return _has_local_effect(skill, trial)


def _skill_execution_operations(skill: Skill, *, protocol: list[str] | None = None) -> list[str]:
    """Verbs that count as executing the skill after activation.

    Prefer an explicit capability operation. Otherwise use productive protocol
    verbs (skip pure navigation / look), so abstracted ``move`` still matches
    environment ``put`` via aliases.
    """
    explicit = _canonical_operation(
        str(skill.metadata.get("capability_operation", "") or "")
    )
    if explicit:
        return [explicit]
    protocol_operations = [
        operation
        for operation in (
            _canonical_operation(instruction)
            for instruction in (skill.action_protocol if protocol is None else protocol)
        )
        if operation and operation not in _WEAK_PROTOCOL_OPS
    ]
    return protocol_operations


def _action_matches_operation(action: Any, operation: str) -> bool:
    text = _environment_action(action)
    if not text or not operation:
        return False
    aliases = _OPERATION_ALIASES.get(operation, frozenset({operation}))
    return any(
        text == alias or text.startswith(f"{alias} ")
        for alias in aliases
    )


def _canonical_operation(text: str) -> str:
    action = _environment_action(text)
    if not action:
        return ""
    # WebShop: search[query] / click[button] → search / click
    if "[" in action:
        return action.split("[", 1)[0].strip()
    return action.split(" ", 1)[0]


def _environment_action(text: Any) -> str:
    value = str(text or "").strip().lower()
    match = re.search(r"<action>(.*?)</action>", value, re.DOTALL)
    return " ".join((match.group(1) if match else value).split())


def _smoothed_success_rate(successes: int, uses: int) -> float:
    return (max(0, successes) + 1) / (max(0, uses) + 2)


def _normalized_credit(credit: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy additive-credit records to success/use counters."""
    uses = max(0, int(credit.get("uses", 0)))
    if "successes" in credit:
        successes = max(0, int(credit.get("successes", 0)))
    else:
        successes = max(
            0,
            int(credit.get("episode_successes", 0))
            + int(credit.get("local_effects", 0)),
        )
    successes = min(successes, uses)
    return {
        "uses": uses,
        "successes": successes,
        "score": _smoothed_success_rate(successes, uses),
        "events": list(credit.get("events") or [])[-50:],
    }


def _record_use(
    credit: dict[str, Any],
    *,
    succeeded: bool,
    task_id: str,
    extra: dict[str, Any] | None = None,
) -> None:
    before = float(credit["score"])
    uses = int(credit["uses"]) + 1
    successes = int(credit["successes"]) + int(succeeded)
    after = _smoothed_success_rate(successes, uses)
    credit["uses"] = uses
    credit["successes"] = successes
    credit["score"] = after
    events = list(credit.get("events") or [])
    event = {
        "task_id": task_id,
        "success": succeeded,
        "score_before": before,
        "score_after": after,
    }
    if extra:
        event.update(extra)
    events.append(event)
    credit["events"] = events[-50:]


def _has_local_effect(skill: Skill, trial: EvaluationTrial) -> bool:
    """Whether the skill's expected local environment effect was observed.

    Episode win alone is not enough — credit must reflect the grounded local
    effect (operation transition, progress delta, or expected observation).
    """
    activation_step = max(
        1,
        int(trial.skill_activation_steps.get(skill.skill_name, 1)),
    )
    post_activation_steps = trial.steps[activation_step - 1 :]
    post_activation_trial = replace(
        trial,
        steps=post_activation_steps,
    )
    operation = str(
        skill.metadata.get("capability_operation", "")
    ).strip().lower()
    if operation:
        try:
            if CapabilityTransitionDetector(operation).detect(
                post_activation_trial,
                attempt=1,
            ):
                return True
        except ValueError:
            pass
        # Abstract place protocols often say "move" while the env emits "put".
        for alias in _OPERATION_ALIASES.get(operation, frozenset()):
            if alias == operation:
                continue
            try:
                if CapabilityTransitionDetector(alias).detect(
                    post_activation_trial,
                    attempt=1,
                ):
                    return True
            except ValueError:
                continue
    elif CapabilityTransitionDetector().detect(
        post_activation_trial,
        attempt=1,
    ):
        return True
    if any(
        float(step.get("goal_progress_delta", 0.0) or 0.0) > 0.0
        for step in post_activation_steps
    ):
        return True
    expected = " ".join(str(skill.expected_effect or "").lower().split())
    return bool(
        expected
        and any(
            expected
            in " ".join(
                str(step.get("observation", "")).lower().split()
            )
            for step in post_activation_steps
        )
    )
