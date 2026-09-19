"""Promote specialists to primary actors only when they beat Executor."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from sage_mas.alfworld_evaluator import AlfWorldOrganizationEvaluator
from sage_mas.onboarding import _agent_scope_families, acting_status
from sage_mas.schemas import AgentSpec, Skill


@dataclass(slots=True)
class ActorPromotionResult:
    agent_name: str
    n_tasks: int
    specialist_wins: int
    executor_wins: int
    specialist_sr: float
    executor_sr: float
    accepted: bool
    reason: str
    gamefiles: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_executor(agent: AgentSpec) -> bool:
    return "executor" in f"{agent.name} {agent.role}".lower()


def _with_status(
    agents: list[AgentSpec],
    *,
    primary_specialist: str | None,
) -> list[AgentSpec]:
    """Build a roster where only Executor or one specialist can be primary."""
    cloned = deepcopy(agents)
    for agent in cloned:
        record = dict(agent.shadow_evaluation_record or {})
        if _is_executor(agent):
            record["acting_status"] = "accepted"
            agent.shadow_evaluation_record = record
            continue
        if primary_specialist and agent.name == primary_specialist:
            record["acting_status"] = "accepted"
            record["dispatch_only"] = False
            record["trial_games_remaining"] = 999
            agent.shadow_evaluation_record = record
            continue
        record["acting_status"] = "demoted"
        record["trial_games_remaining"] = 0
        agent.shadow_evaluation_record = record
    return cloned


def select_promotion_gamefiles(
    pool: Sequence[str],
    agent: AgentSpec,
    *,
    max_tasks: int,
) -> list[str]:
    """Prefer shadow-pool tasks in the specialist's learned families."""
    families = _agent_scope_families(agent)
    if not pool or max_tasks <= 0:
        return []
    if not families:
        return list(pool)[: max(0, int(max_tasks))]
    matched = [
        path
        for path in pool
        if any(family in str(path).lower() for family in families)
    ]
    # Never promote a scoped specialist on unrelated fallback tasks. A clean
    # specialist that beats Executor on look-at-light tasks is not evidence that
    # it should receive clean-task primary control.
    if not matched:
        return []
    return matched[: max(0, int(max_tasks))]


def promotion_beats_executor(
    specialist_sr: float,
    executor_sr: float,
    *,
    min_advantage: float = 0.0,
) -> bool:
    """Promote only when specialist clearly beats Executor.

    Ties never pass. With ``min_advantage=0``, require strict ``>``.
    With ``min_advantage>0``, require ``specialist_sr >= executor_sr + min_advantage``.
    """
    margin = max(0.0, float(min_advantage))
    if margin <= 0.0:
        return float(specialist_sr) > float(executor_sr) + 1e-12
    return float(specialist_sr) + 1e-9 >= float(executor_sr) + margin


def probe_actor_promotion(
    evaluator: AlfWorldOrganizationEvaluator,
    *,
    agents: list[AgentSpec],
    skills: list[Skill],
    specialist: AgentSpec,
    gamefiles: list[str],
    min_advantage: float = 0.0,
    epsilon: float | None = None,
    bare_executor: bool = True,
) -> ActorPromotionResult:
    """Paired held-out: specialist-as-primary vs Executor-as-primary.

    ``epsilon`` is deprecated (old formula allowed ties via ``exec - eps``).
    Prefer ``min_advantage`` so promotion requires a real gain over Executor.

    When ``bare_executor`` is True (default), the Exec arm gets ``skills=[]``
    so the contrast is Spec+skill vs bare mid-model Executor (cold-start style).
    """
    if not gamefiles:
        return ActorPromotionResult(
            agent_name=specialist.name,
            n_tasks=0,
            specialist_wins=0,
            executor_wins=0,
            specialist_sr=0.0,
            executor_sr=0.0,
            accepted=False,
            reason="no promotion gamefiles",
            gamefiles=[],
        )
    specialist_agents = _with_status(
        agents,
        primary_specialist=specialist.name,
    )
    executor_agents = _with_status(agents, primary_specialist=None)

    eval_config = getattr(evaluator, "config", None)
    guard = getattr(evaluator, "specialist_action_guard", None)
    dispatch_config = getattr(getattr(evaluator, "dispatcher", None), "config", None)
    old_controller_enabled = (
        getattr(eval_config, "enable_specialist_controllers", None)
        if eval_config is not None
        else None
    )
    old_guard_enabled = (
        getattr(guard, "enabled", None) if guard is not None else None
    )
    old_require_controller = (
        getattr(dispatch_config, "require_controller_for_eligibility", None)
        if dispatch_config is not None
        else None
    )
    old_auto_assign_single = (
        getattr(dispatch_config, "auto_assign_single", None)
        if dispatch_config is not None
        else None
    )
    old_allow_exec_inject = (
        getattr(eval_config, "allow_executor_skill_injection", None)
        if eval_config is not None
        else None
    )
    try:
        # Actor promotion is the acceptance gate for generated specialists. It
        # must measure LLM+skill execution, never controller-assisted execution.
        if old_controller_enabled is not None:
            eval_config.enable_specialist_controllers = False
        if old_guard_enabled is not None:
            guard.enabled = False
        if old_require_controller is not None:
            dispatch_config.require_controller_for_eligibility = False
        # Promotion is a paired actor test, not an Executor-routing test. Force
        # the single eligible specialist to be primary so the measured wins
        # belong to the candidate actor instead of an LLM keep-Executor choice.
        if old_auto_assign_single is not None:
            dispatch_config.auto_assign_single = True
        if bare_executor and old_allow_exec_inject is not None:
            eval_config.allow_executor_skill_injection = False
        print(
            f"[actor_promo] Spec-as-primary begin agent={specialist.name} "
            f"n={len(gamefiles)} bare_executor={bare_executor}",
            flush=True,
        )
        specialist_trials = evaluator.evaluate(
            agents=specialist_agents,
            skills=skills,
            gamefiles=list(gamefiles),
            condition=f"actor_promo_specialist:{specialist.name}",
        )
        print(
            f"[actor_promo] Spec done wins="
            f"{sum(1 for t in specialist_trials if t.won)}/{len(gamefiles)}; "
            f"Exec-as-primary begin "
            f"(skills={'[]' if bare_executor else 'bank'})",
            flush=True,
        )
        executor_trials = evaluator.evaluate(
            agents=executor_agents,
            skills=[] if bare_executor else skills,
            gamefiles=list(gamefiles),
            condition=f"actor_promo_executor:{specialist.name}",
        )
        print(
            f"[actor_promo] Exec done wins="
            f"{sum(1 for t in executor_trials if t.won)}/{len(gamefiles)}",
            flush=True,
        )
    finally:
        if old_controller_enabled is not None:
            eval_config.enable_specialist_controllers = old_controller_enabled
        if old_guard_enabled is not None:
            guard.enabled = old_guard_enabled
        if old_require_controller is not None:
            dispatch_config.require_controller_for_eligibility = (
                old_require_controller
            )
        if old_auto_assign_single is not None:
            dispatch_config.auto_assign_single = old_auto_assign_single
        if old_allow_exec_inject is not None:
            eval_config.allow_executor_skill_injection = old_allow_exec_inject

    specialist_wins = sum(1 for trial in specialist_trials if trial.won)
    executor_wins = sum(1 for trial in executor_trials if trial.won)
    n_tasks = len(gamefiles)
    specialist_sr = specialist_wins / n_tasks
    executor_sr = executor_wins / n_tasks
    # Ignore legacy epsilon for acceptance; keep it only in the reason string
    # if callers still pass it so old logs stay interpretable.
    advantage = max(0.0, float(min_advantage))
    accepted = promotion_beats_executor(
        specialist_sr,
        executor_sr,
        min_advantage=advantage,
    )
    reason = (
        f"specialist {specialist_wins}/{n_tasks} vs Executor "
        f"{executor_wins}/{n_tasks} (min_advantage={advantage}"
        + (f", deprecated_eps={epsilon}" if epsilon is not None else "")
        + f", bare_executor={bare_executor}"
        + ")"
    )
    if accepted:
        reason = f"promote: {reason}"
    else:
        reason = f"keep Executor primary: {reason}"
    return ActorPromotionResult(
        agent_name=specialist.name,
        n_tasks=n_tasks,
        specialist_wins=specialist_wins,
        executor_wins=executor_wins,
        specialist_sr=specialist_sr,
        executor_sr=executor_sr,
        accepted=accepted,
        reason=reason,
        gamefiles=list(gamefiles),
    )


def apply_actor_promotion_result(
    agent: AgentSpec,
    result: ActorPromotionResult,
    *,
    remove_after_rejected_windows: int = 2,
    online_trial_games: int = 3,
) -> dict[str, Any]:
    """Update acting_status from a promotion probe. Skills stay on Executor.

    A held-out probe is only permission to try a specialist online. It is not
    enough to mark the specialist accepted because small probes can pass by
    chance and then fail immediately on the scored stream.
    """
    record = dict(agent.shadow_evaluation_record or {})
    record["last_actor_promotion"] = result.as_dict()
    decision = {
        "agent": agent.name,
        "accepted": result.accepted,
        "specialist_sr": result.specialist_sr,
        "executor_sr": result.executor_sr,
        "n_tasks": result.n_tasks,
        "reason": result.reason,
    }
    if result.accepted:
        record["acting_status"] = "probation"
        record["dispatch_only"] = False
        record["trial_games_remaining"] = max(1, int(online_trial_games))
        record["promotion_probe_passed"] = True
        record["actor_promoted"] = False
        record["last_onboarding_decision"] = (
            "actor_probe_passed_online_probation"
        )
        decision["new_status"] = "probation"
        decision["online_trial_games"] = record["trial_games_remaining"]
    else:
        rejects = int(record.get("actor_promotion_rejects") or 0) + 1
        record["actor_promotion_rejects"] = rejects
        record["rejected_windows"] = int(record.get("rejected_windows") or 0) + 1
        # Stay on roster as skill owner, but never primary until a later pass.
        if rejects >= max(1, int(remove_after_rejected_windows)):
            record["acting_status"] = "dormant"
            record["last_onboarding_decision"] = "actor_promotion_dormant"
            decision["new_status"] = "dormant"
        else:
            record["acting_status"] = "probation"
            record["last_onboarding_decision"] = "actor_promotion_rejected"
            decision["new_status"] = "probation"
        record["dispatch_only"] = True
        record["trial_games_remaining"] = 0
        record["actor_promoted"] = False
        record["promotion_probe_passed"] = False
    agent.shadow_evaluation_record = record
    return decision


def probation_specialists(agents: list[AgentSpec]) -> list[AgentSpec]:
    specialists: list[AgentSpec] = []
    for agent in agents:
        if _is_executor(agent) or acting_status(agent) != "probation":
            continue
        record = agent.shadow_evaluation_record or {}
        # Probe-passed specialists are already waiting for real online
        # primary-dispatch evidence; do not keep re-probing them.
        if bool(record.get("promotion_probe_passed")):
            continue
        specialists.append(agent)
    return specialists
