"""Onboarding / probation gate for newly added specialists."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from sage_mas.schemas import AgentSpec
from sage_mas.specialist_controllers import specialist_has_controller_for_scopes

# Keep Executor acting prompt aligned with skill-only arm B even when the
# roster temporarily holds specialists. Dispatch is a code path, not a
# "team coordinator" persona that confuses the actor LLM.
SOLE_EXECUTOR_ROLE_SPECIFICATION = (
    "Sole ALFWorld actor with optional injected skill protocols. "
    "Interpret observations and execute admissible environment actions."
)
SOLE_EXECUTOR_RESPONSIBILITY_BOUNDARY = (
    "In scope: observation interpretation, skill-conditioned action selection. "
    "Out of scope: spawning specialists."
)
SOLE_EXECUTOR_RESPONSIBILITIES = [
    "Interpret ALFWorld observations",
    "Select and execute admissible environment actions",
    "Follow injected verified skill protocols when preconditions match",
]
SOLE_EXECUTOR_INPUT_PROTOCOL = (
    "Input: ALFWorld observation template; optional active skill protocols."
)
SOLE_EXECUTOR_OUTPUT_PROTOCOL = (
    "Output exactly one <think>...</think><action>admissible action</action> decision."
)


def acting_status(agent: AgentSpec) -> str:
    record = agent.shadow_evaluation_record or {}
    status = str(record.get("acting_status") or "").strip().lower()
    return status or "accepted"


def _is_executor_name(name: str | None) -> bool:
    return "executor" in str(name or "").lower()


def _executor_agent(agents: Iterable[AgentSpec]) -> AgentSpec | None:
    for agent in agents:
        if _is_executor_name(agent.name) or _is_executor_name(agent.role):
            return agent
    return None


def restore_executor_baseline(
    agents: list[AgentSpec],
    *,
    clear_assigned_skills_when_solo: bool = True,
) -> dict[str, Any]:
    """Reset Executor persona to sole-actor (B) and clear sticky skill ownership.

    When no specialists remain, drop Executor.assigned_skills so verified skills
    flow through bank injection like arm B (not sticky dual-assignment scars).
    """
    executor = _executor_agent(agents)
    if executor is None:
        return {"changed": False, "reason": "no_executor"}

    specialists = [
        agent
        for agent in agents
        if agent is not executor
        and not _is_executor_name(agent.name)
        and not _is_executor_name(agent.role)
    ]
    changed_fields: list[str] = []
    before_skills = list(executor.assigned_skills or [])

    if executor.role_specification != SOLE_EXECUTOR_ROLE_SPECIFICATION:
        executor.role_specification = SOLE_EXECUTOR_ROLE_SPECIFICATION
        changed_fields.append("role_specification")
    if executor.responsibility_boundary != SOLE_EXECUTOR_RESPONSIBILITY_BOUNDARY:
        executor.responsibility_boundary = SOLE_EXECUTOR_RESPONSIBILITY_BOUNDARY
        changed_fields.append("responsibility_boundary")
    if list(executor.responsibilities or []) != SOLE_EXECUTOR_RESPONSIBILITIES:
        executor.responsibilities = list(SOLE_EXECUTOR_RESPONSIBILITIES)
        changed_fields.append("responsibilities")
    if executor.input_protocol != SOLE_EXECUTOR_INPUT_PROTOCOL:
        executor.input_protocol = SOLE_EXECUTOR_INPUT_PROTOCOL
        changed_fields.append("input_protocol")
    if executor.output_protocol != SOLE_EXECUTOR_OUTPUT_PROTOCOL:
        executor.output_protocol = SOLE_EXECUTOR_OUTPUT_PROTOCOL
        changed_fields.append("output_protocol")

    cleared_skills = False
    if clear_assigned_skills_when_solo and not specialists:
        if executor.assigned_skills:
            executor.assigned_skills = []
            cleared_skills = True
            changed_fields.append("assigned_skills")

    return {
        "changed": bool(changed_fields),
        "changed_fields": changed_fields,
        "cleared_assigned_skills": cleared_skills,
        "previous_assigned_skills": before_skills,
        "specialist_count": len(specialists),
        "specialist_names": [agent.name for agent in specialists],
    }


def _trial_won(trial: Any) -> bool:
    if isinstance(trial, dict):
        return bool(trial.get("won"))
    return bool(getattr(trial, "won", False))


def _trial_family(trial: Any) -> str:
    if isinstance(trial, dict):
        return str(trial.get("task_family") or "")
    return str(getattr(trial, "task_family", "") or "")


def _trial_primary(trial: Any) -> str:
    if isinstance(trial, dict):
        return str(trial.get("assigned_primary_agent") or "")
    return str(getattr(trial, "assigned_primary_agent", "") or "")


def _trial_dispatch_layer(trial: Any) -> str:
    if isinstance(trial, dict):
        return str(trial.get("dispatch_layer") or "")
    return str(getattr(trial, "dispatch_layer", "") or "")


def _trial_actions_by_agent(trial: Any) -> dict[str, int]:
    if isinstance(trial, dict):
        payload = trial.get("actions_by_agent") or {}
    else:
        payload = getattr(trial, "actions_by_agent", {}) or {}
    return {str(name): int(count) for name, count in payload.items()}


def _trial_activated_skill_names(trial: Any) -> set[str]:
    if isinstance(trial, dict):
        payload = trial.get("activated_skill_names") or []
    else:
        payload = getattr(trial, "activated_skill_names", []) or []
    return {str(name).strip() for name in payload if str(name).strip()}


def _is_real_dispatch_trial(trial: Any, agent_name: str) -> bool:
    return (
        _trial_primary(trial) == agent_name
        and _trial_dispatch_layer(trial) in {"eligibility_single", "llm"}
        and _trial_actions_by_agent(trial).get(agent_name, 0) > 0
    )


def _is_primary_execution_trial(trial: Any, agent_name: str) -> bool:
    return (
        _trial_primary(trial) == agent_name
        and _trial_actions_by_agent(trial).get(agent_name, 0) > 0
    )


def _agent_scope_families(agent: AgentSpec) -> set[str]:
    record = agent.shadow_evaluation_record or {}
    values = list(record.get("task_families") or [])
    contract = record.get("capability_contract") or {}
    if isinstance(contract, dict):
        values.extend(contract.get("task_families") or [])
    return {
        str(value).strip().lower()
        for value in values
        if str(value).strip() and str(value).strip().lower() != "other"
    }


def _agent_skill_names(agent: AgentSpec) -> set[str]:
    record = agent.shadow_evaluation_record or {}
    contract = record.get("capability_contract") or {}
    names = set(str(name).strip() for name in (agent.assigned_skills or []) if str(name).strip())
    if isinstance(contract, dict):
        names.update(
            str(name).strip()
            for name in (contract.get("skill_names") or [])
            if str(name).strip()
        )
    return names


def _is_contract_applicable_trial(trial: Any, agent: AgentSpec) -> bool:
    scopes = _agent_scope_families(agent)
    if not scopes:
        # Preserve legacy agents that predate CapabilityContract metadata.
        return True
    return _trial_family(trial).strip().lower() in scopes


def _beats_executor_baseline(
    agent_sr: float,
    baseline_sr: float,
    *,
    epsilon: float,
) -> bool:
    """Accept only when specialist is not worse than Executor on-scope."""
    return agent_sr + 1e-9 >= baseline_sr - max(0.0, float(epsilon))


def refresh_acting_statuses(
    agents: list[AgentSpec],
    segment_trials: Iterable[Any],
    *,
    baseline_trials: Iterable[Any] | None = None,
    min_games: int = 3,
    min_wins: int = 2,
    epsilon: float = 0.0,
    remove_after_rejected_windows: int = 2,
    require_controller: bool = False,
    min_new_agent_call_rate: float = 0.5,
) -> dict[str, Any]:
    """Promote only from repeated in-contract full-task wins."""
    segment = list(segment_trials)
    baseline = list(baseline_trials) if baseline_trials is not None else segment
    updates: dict[str, Any] = {"decisions": []}
    budget_changed = False
    min_games = max(1, int(min_games))
    min_wins = max(1, int(min_wins))
    min_call_rate = max(0.0, float(min_new_agent_call_rate))

    agent_trials: dict[str, list[Any]] = defaultdict(list)
    out_of_scope_trials: dict[str, list[Any]] = defaultdict(list)
    for agent in agents:
        for trial in segment:
            if _is_real_dispatch_trial(trial, agent.name):
                if _is_contract_applicable_trial(trial, agent):
                    agent_trials[agent.name].append(trial)
                else:
                    out_of_scope_trials[agent.name].append(trial)

    performance_changed = False
    for agent in agents:
        primary_trials = [
            trial
            for trial in segment
            if _is_primary_execution_trial(trial, agent.name)
        ]
        if not primary_trials:
            continue
        record = dict(agent.shadow_evaluation_record or {})
        performance = dict(record.get("task_performance") or {})
        performance["dispatches"] = int(
            performance.get("dispatches") or 0
        ) + len(primary_trials)
        performance["wins"] = int(performance.get("wins") or 0) + sum(
            int(_trial_won(trial)) for trial in primary_trials
        )
        by_family = {
            str(family): dict(stats)
            for family, stats in dict(
                performance.get("by_family") or {}
            ).items()
        }
        for trial in primary_trials:
            family = _trial_family(trial) or "other"
            stats = by_family.setdefault(
                family,
                {"dispatches": 0, "wins": 0},
            )
            stats["dispatches"] = int(stats.get("dispatches") or 0) + 1
            stats["wins"] = int(stats.get("wins") or 0) + int(
                _trial_won(trial)
            )
        performance["by_family"] = by_family
        record["task_performance"] = performance
        agent.shadow_evaluation_record = record
        performance_changed = True

    executor_by_family: dict[str, list[bool]] = defaultdict(list)
    for trial in baseline:
        family = _trial_family(trial)
        primary = _trial_primary(trial)
        if not family:
            continue
        if not primary or _is_executor_name(primary):
            executor_by_family[family].append(_trial_won(trial))

    # Consume budget and accumulate only genuine dispatch-bound work.
    for agent in agents:
        if acting_status(agent) != "probation":
            continue
        trials = agent_trials.get(agent.name, [])
        rejected_scope = out_of_scope_trials.get(agent.name, [])
        if rejected_scope:
            record = dict(agent.shadow_evaluation_record or {})
            record["out_of_scope_dispatches"] = int(
                record.get("out_of_scope_dispatches") or 0
            ) + len(rejected_scope)
            agent.shadow_evaluation_record = record
            performance_changed = True
        played = len(trials)
        if played <= 0:
            continue
        record = dict(agent.shadow_evaluation_record or {})
        before = int(record.get("trial_games_remaining") or 0)
        remaining = max(0, before - played)
        if remaining != before:
            record["trial_games_remaining"] = remaining
            budget_changed = True
        record["dispatched_games"] = int(
            record.get("dispatched_games") or 0
        ) + played
        record["actions_executed"] = int(
            record.get("actions_executed") or 0
        ) + sum(
            _trial_actions_by_agent(trial).get(agent.name, 0)
            for trial in trials
        )
        record["wins_as_primary"] = int(
            record.get("wins_as_primary") or 0
        ) + sum(int(_trial_won(trial)) for trial in trials)
        record["applicable_dispatched_games"] = int(
            record.get("applicable_dispatched_games") or 0
        ) + played
        record["applicable_wins_as_primary"] = int(
            record.get("applicable_wins_as_primary") or 0
        ) + sum(int(_trial_won(trial)) for trial in trials)
        # Scope-eligible episodes in this segment (family match), whether or
        # not the specialist was chosen — used for call_rate.
        scope_families = set(_agent_scope_families(agent))
        eligible = sum(
            1
            for trial in segment
            if (not scope_families) or _trial_family(trial) in scope_families
        )
        record["scope_eligible_games"] = int(
            record.get("scope_eligible_games") or 0
        ) + int(eligible)
        record["trial_families"] = sorted(
            set(record.get("trial_families") or [])
            | {
                _trial_family(trial)
                for trial in trials
                if _trial_family(trial)
            }
        )
        agent.shadow_evaluation_record = record

    for agent in agents:
        if "executor" in f"{agent.name} {agent.role}".lower():
            continue
        status = acting_status(agent)
        if status != "probation":
            continue
        record = dict(agent.shadow_evaluation_record or {})
        remaining = int(record.get("trial_games_remaining") or 0)
        dispatched_games = int(
            record.get(
                "applicable_dispatched_games",
                record.get("dispatched_games") or 0,
            )
        )
        wins_as_primary = int(
            record.get(
                "applicable_wins_as_primary",
                record.get("wins_as_primary") or 0,
            )
        )
        eligible_games = int(
            record.get("scope_eligible_games") or dispatched_games or 0
        )
        call_rate = (
            float(dispatched_games) / float(eligible_games)
            if eligible_games > 0
            else 0.0
        )
        # Keep probation open until enough in-contract full-task evidence exists.
        # Need enough in-contract full-task trials before any accept/demote.
        if dispatched_games < min_games:
            if remaining <= 0:
                record["trial_games_remaining"] = min_games - dispatched_games
                agent.shadow_evaluation_record = record
                budget_changed = True
            continue

        agent_sr = wins_as_primary / max(dispatched_games, 1)
        contract_families = _agent_scope_families(agent)
        trial_families = set(record.get("trial_families") or [])
        comparison_families = contract_families or trial_families
        specialist_skills = _agent_skill_names(agent)
        # Prefer scoped baseline_trials (recent / segment Executor outcomes)
        # over lifetime Executor task_performance so cold-start specialists are
        # not judged against the entire online history.
        baseline_outcomes: list[bool] = []
        baseline_outcomes_without_skill: list[bool] = []
        for family in comparison_families:
            baseline_outcomes.extend(executor_by_family.get(family, []))
        for trial in baseline:
            family = _trial_family(trial)
            if family not in comparison_families:
                continue
            primary = _trial_primary(trial)
            if primary and not _is_executor_name(primary):
                continue
            activated = _trial_activated_skill_names(trial)
            if specialist_skills and specialist_skills & activated:
                continue
            baseline_outcomes_without_skill.append(_trial_won(trial))
        baseline_wins = 0
        baseline_games = 0
        executor = next(
            (
                candidate
                for candidate in agents
                if _is_executor_name(candidate.name)
            ),
            None,
        )
        cold_start = dispatched_games <= max(min_games, 1)
        if (not cold_start) and executor is not None:
            performance = (
                executor.shadow_evaluation_record or {}
            ).get("task_performance") or {}
            by_family = (
                performance.get("by_family", {})
                if isinstance(performance, dict)
                else {}
            )
            for family in comparison_families:
                stats = (
                    by_family.get(family, {})
                    if isinstance(by_family, dict)
                    else {}
                )
                baseline_wins += int(stats.get("wins") or 0)
                baseline_games += int(stats.get("dispatches") or 0)
        baseline_source = "executor_all_scoped"
        if baseline_outcomes_without_skill:
            baseline_sr = sum(baseline_outcomes_without_skill) / len(
                baseline_outcomes_without_skill
            )
            baseline_source = "executor_without_specialist_skill"
        elif baseline_outcomes:
            baseline_sr = sum(baseline_outcomes) / len(baseline_outcomes)
        elif baseline_games > 0:
            baseline_sr = baseline_wins / baseline_games
            baseline_source = "executor_lifetime_by_family"
        else:
            if remaining <= 0:
                record["trial_games_remaining"] = 1
                agent.shadow_evaluation_record = record
                budget_changed = True
            continue

        verified_mu = record.get("verified_skill_marginal_utility")
        positive_conditional_gain = (
            verified_mu is None or float(verified_mu) > 0.0
        )
        controller_backed = (
            True
            if not require_controller
            else specialist_has_controller_for_scopes(agent, [])
        )
        beats_baseline = _beats_executor_baseline(
            agent_sr,
            baseline_sr,
            epsilon=epsilon,
        )
        call_rate_ok = (
            min_call_rate <= 0.0 or call_rate + 1e-12 >= min_call_rate
        )
        qualifies = (
            wins_as_primary >= min_wins
            and positive_conditional_gain
            and controller_backed
            and beats_baseline
            and call_rate_ok
        )
        decision = {
            "agent": agent.name,
            "previous_status": status,
            "agent_sr": agent_sr,
            "executor_sr": baseline_sr,
            "games": dispatched_games,
            "actions_executed": int(record.get("actions_executed") or 0),
            "wins_as_primary": wins_as_primary,
            "min_games": min_games,
            "min_wins": min_wins,
            "completed_applicable_task": wins_as_primary >= min_wins,
            "applicable_task_families": sorted(comparison_families),
            "epsilon": epsilon,
            "verified_skill_marginal_utility": verified_mu,
            "controller_backed": controller_backed,
            "call_rate": call_rate,
            "min_call_rate": min_call_rate,
            "scope_eligible_games": eligible_games,
            "baseline_source": baseline_source,
        }
        if qualifies:
            record["acting_status"] = "accepted"
            record["trial_games_remaining"] = 0
            record["accepted_windows"] = int(record.get("accepted_windows") or 0) + 1
            record["last_onboarding_decision"] = "accepted"
            decision["new_status"] = "accepted"
        else:
            # Enough in-contract trials exist, but the specialist is not a
            # stable full-task improvement over Executor.
            record["acting_status"] = "demoted"
            record["trial_games_remaining"] = 0
            record["rejected_windows"] = int(record.get("rejected_windows") or 0) + 1
            record["last_onboarding_decision"] = "demoted"
            decision["new_status"] = "demoted"
        agent.shadow_evaluation_record = record
        updates["decisions"].append(decision)

    # Revalidate previously accepted specialists on their learned contract
    # scope with the same full-task stability bar.
    executor = next(
        (agent for agent in agents if _is_executor_name(agent.name)),
        None,
    )
    executor_performance = (
        (executor.shadow_evaluation_record or {}).get("task_performance") or {}
        if executor is not None
        else {}
    )
    executor_by_scope = (
        executor_performance.get("by_family", {})
        if isinstance(executor_performance, dict)
        else {}
    )
    for agent in agents:
        if _is_executor_name(agent.name) or acting_status(agent) != "accepted":
            continue
        scopes = _agent_scope_families(agent)
        if not scopes:
            continue
        if require_controller and not specialist_has_controller_for_scopes(
            agent, []
        ):
            record = dict(agent.shadow_evaluation_record or {})
            record["acting_status"] = "demoted"
            record["trial_games_remaining"] = 0
            record["rejected_windows"] = int(
                record.get("rejected_windows") or 0
            ) + 1
            record["last_onboarding_decision"] = "demoted_no_controller"
            agent.shadow_evaluation_record = record
            updates["decisions"].append(
                {
                    "agent": agent.name,
                    "previous_status": "accepted",
                    "new_status": "demoted",
                    "reason": "no_scoped_specialist_controller",
                    "applicable_task_families": sorted(scopes),
                }
            )
            continue
        record = dict(agent.shadow_evaluation_record or {})
        performance = record.get("task_performance") or {}
        by_family = (
            performance.get("by_family", {})
            if isinstance(performance, dict)
            else {}
        )
        scoped_games = sum(
            int((by_family.get(family) or {}).get("dispatches") or 0)
            for family in scopes
        )
        scoped_wins = sum(
            int((by_family.get(family) or {}).get("wins") or 0)
            for family in scopes
        )
        executor_games = sum(
            int((executor_by_scope.get(family) or {}).get("dispatches") or 0)
            for family in scopes
        )
        executor_wins = sum(
            int((executor_by_scope.get(family) or {}).get("wins") or 0)
            for family in scopes
        )
        if scoped_games < min_games or executor_games <= 0:
            continue
        scoped_sr = scoped_wins / scoped_games
        executor_sr = executor_wins / executor_games
        if (
            scoped_wins >= min_wins
            and _beats_executor_baseline(
                scoped_sr,
                executor_sr,
                epsilon=epsilon,
            )
        ):
            continue
        record["acting_status"] = "demoted"
        record["trial_games_remaining"] = 0
        record["rejected_windows"] = int(
            record.get("rejected_windows") or 0
        ) + 1
        record["last_onboarding_decision"] = "demoted_scope_revalidation"
        agent.shadow_evaluation_record = record
        updates["decisions"].append(
            {
                "agent": agent.name,
                "previous_status": "accepted",
                "new_status": "demoted",
                "reason": "contract_scope_revalidation",
                "agent_sr": scoped_sr,
                "executor_sr": executor_sr,
                "games": scoped_games,
                "wins_as_primary": scoped_wins,
                "min_games": min_games,
                "min_wins": min_wins,
                "applicable_task_families": sorted(scopes),
                "epsilon": epsilon,
            }
        )

    removed_agents: list[str] = []
    retained_agents: list[AgentSpec] = []
    for agent in agents:
        if acting_status(agent) != "demoted":
            retained_agents.append(agent)
            continue
        record = dict(agent.shadow_evaluation_record or {})
        rejected = int(record.get("rejected_windows") or 0)
        # A newly demoted agent already has one rejected window. Each later
        # segment without work is another ineffective window.
        if not any(
            decision.get("agent") == agent.name
            and decision.get("new_status") == "demoted"
            for decision in updates["decisions"]
        ):
            rejected += 1
            record["rejected_windows"] = rejected
            agent.shadow_evaluation_record = record
            budget_changed = True
        if rejected >= max(1, int(remove_after_rejected_windows)):
            removed_agents.append(agent.name)
        else:
            retained_agents.append(agent)
    if removed_agents:
        agents[:] = retained_agents
    updates["removed_agents"] = removed_agents
    executor_restore = restore_executor_baseline(agents)
    updates["executor_restore"] = executor_restore
    updates["changed"] = (
        bool(updates["decisions"])
        or budget_changed
        or performance_changed
        or bool(removed_agents)
        or bool(executor_restore.get("changed"))
    )
    return updates


def revive_cross_model_demotions(
    agents: list[AgentSpec],
    *,
    probation_games: int = 3,
) -> dict[str, Any]:
    """Restore demoted specialists to probation for same-model re-evaluation.

    Use when a prior run demoted Flash specialists against a Pro teacher
    Executor baseline — an invalid cross-model gate for Exp B-style setups.
    """
    revived: list[str] = []
    for agent in agents:
        if _is_executor_name(agent.name) or _is_executor_name(agent.role):
            continue
        if acting_status(agent) != "demoted":
            continue
        record = dict(agent.shadow_evaluation_record or {})
        record["acting_status"] = "probation"
        record["dispatch_only"] = True
        record["trial_games_remaining"] = max(1, int(probation_games))
        # Keep historical counters; only clear the bad reject so one fair
        # same-model window can accept or demote again.
        record["rejected_windows"] = 0
        record["accepted_windows"] = int(record.get("accepted_windows") or 0)
        record["last_onboarding_decision"] = "revived_cross_model_demotion"
        agent.shadow_evaluation_record = record
        revived.append(agent.name)
    return {
        "changed": bool(revived),
        "revived_agents": revived,
        "probation_games": int(probation_games),
    }
