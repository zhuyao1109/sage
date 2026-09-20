"""Probation → accepted onboarding for τ² (aligned with sage_mas.onboarding).

After a segment finishes, consume primary-dispatch trials for each specialist
and promote only when enough in-domain wins accumulate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sage_tau2.organization import EXECUTOR_NAME, Organization
from sage_tau2.schemas import AgentSpec, Tau2Trajectory
from sage_tau2.serialization import write_json


@dataclass(slots=True)
class OnboardingPolicy:
    """Mirrors sage_mas online600: probation_min_games=3, probation_min_wins=2."""

    min_games: int = 3
    min_wins: int = 2
    epsilon: float = 0.0
    remove_after_rejected_windows: int = 2


def acting_status(agent: AgentSpec) -> str:
    record = agent.shadow_evaluation_record or {}
    status = str(record.get("acting_status") or agent.acting_status or "").strip()
    return (status or "accepted").lower()


def append_dispatch_event(
    path: str | Path | None,
    *,
    task_id: str,
    primary: str,
    domain: str,
    layer: str = "",
    **runtime_inject: Any,
) -> None:
    """Append sticky-primary assignment (+ optional runtime inject snapshot).

    Extra kwargs (system_prompt, skills_block, injected_skill_ids, …) are stored
    so PRO dumps can recover the live prompt context that ``results.json``
    messages omit.
    """
    if not path or not task_id or not primary:
        return
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "task_id": str(task_id),
        "primary": str(primary),
        "domain": str(domain),
        "layer": str(layer),
    }
    for key, value in runtime_inject.items():
        if value is None:
            continue
        row[key] = value
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line)


def load_dispatch_index(path: str | Path | None) -> dict[str, str]:
    """task_id → last primary name."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    import json

    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        tid = str(row.get("task_id") or "").strip()
        primary = str(row.get("primary") or "").strip()
        if tid and primary:
            out[tid] = primary
    return out


def _is_executor_name(name: str) -> bool:
    return str(name or "").strip().lower() in {
        EXECUTOR_NAME.lower(),
        "executor",
        "generalist",
    }


def refresh_acting_statuses(
    organization: Organization,
    trajectories: list[Tau2Trajectory],
    *,
    dispatch_by_task: dict[str, str] | None = None,
    policy: OnboardingPolicy | None = None,
) -> dict[str, Any]:
    """Promote probation specialists from repeated primary wins (same domain)."""
    pol = policy or OnboardingPolicy()
    min_games = max(1, int(pol.min_games))
    min_wins = max(1, int(pol.min_wins))
    dispatch_by_task = dict(dispatch_by_task or {})

    # Fallback: trajectory metadata may already carry primary.
    for traj in trajectories:
        tid = str(traj.task_id or "")
        if tid and tid not in dispatch_by_task:
            meta = traj.metadata or {}
            primary = str(meta.get("primary_agent") or meta.get("primary") or "")
            if primary:
                dispatch_by_task[tid] = primary

    trials_by_agent: dict[str, list[Tau2Trajectory]] = defaultdict(list)
    for traj in trajectories:
        primary = dispatch_by_task.get(str(traj.task_id or ""), "")
        if not primary or _is_executor_name(primary):
            continue
        trials_by_agent[primary].append(traj)

    decisions: list[dict[str, Any]] = []
    for agent in list(organization.agents):
        if agent.name == EXECUTOR_NAME:
            continue
        status = acting_status(agent)
        trials = trials_by_agent.get(agent.name, [])
        if not trials:
            continue

        # Domain gate: only count trials matching skill/domain metadata.
        agent_domains = {
            str(d).strip().lower()
            for d in (agent.metadata.get("domains") or [])
            if str(d).strip()
        }
        if not agent_domains:
            # Infer from capability/skill names when missing.
            for key in agent.capability_keys or []:
                # capability keys are tau2.*; domain is separate metadata.
                pass
        scoped = []
        for traj in trials:
            if agent_domains and str(traj.domain).lower() not in agent_domains:
                continue
            scoped.append(traj)
        if not scoped:
            scoped = list(trials)

        record = dict(agent.shadow_evaluation_record or {})
        played = len(scoped)
        wins = sum(1 for t in scoped if t.success)
        before = int(record.get("trial_games_remaining") or 0)
        remaining = max(0, before - played) if "trial_games_remaining" in record else before
        if "trial_games_remaining" in record:
            record["trial_games_remaining"] = remaining
        record["applicable_dispatched_games"] = int(
            record.get("applicable_dispatched_games") or 0
        ) + played
        record["applicable_wins_as_primary"] = int(
            record.get("applicable_wins_as_primary") or 0
        ) + wins
        record["wins_as_primary"] = int(record.get("wins_as_primary") or 0) + wins
        record["dispatched_games"] = int(record.get("dispatched_games") or 0) + played

        perf = dict(record.get("task_performance") or {})
        perf["dispatches"] = int(perf.get("dispatches") or 0) + played
        perf["wins"] = int(perf.get("wins") or 0) + wins
        by_domain = {
            str(k): dict(v)
            for k, v in dict(perf.get("by_domain") or {}).items()
        }
        for traj in scoped:
            dom = str(traj.domain or "other")
            stats = by_domain.setdefault(dom, {"dispatches": 0, "wins": 0})
            stats["dispatches"] = int(stats.get("dispatches") or 0) + 1
            stats["wins"] = int(stats.get("wins") or 0) + int(bool(traj.success))
        perf["by_domain"] = by_domain
        record["task_performance"] = perf

        decision = "continue_probation"
        if status == "probation":
            total_games = int(record.get("applicable_dispatched_games") or 0)
            total_wins = int(record.get("applicable_wins_as_primary") or 0)
            if total_games >= min_games and total_wins >= min_wins:
                agent.acting_status = "accepted"
                record["acting_status"] = "accepted"
                record["dispatch_only"] = False
                agent.metadata["dispatch_only"] = False
                record["last_onboarding_decision"] = "accepted"
                decision = "accepted"
            elif (
                "trial_games_remaining" in record
                and remaining <= 0
                and total_wins < min_wins
            ):
                rejected = int(record.get("rejected_windows") or 0) + 1
                record["rejected_windows"] = rejected
                if rejected >= int(pol.remove_after_rejected_windows):
                    agent.acting_status = "dormant"
                    record["acting_status"] = "dormant"
                    record["dispatch_only"] = True
                    agent.metadata["dispatch_only"] = True
                    record["last_onboarding_decision"] = "dormant"
                    decision = "dormant"
                else:
                    # Reset a short trial window for another chance.
                    record["trial_games_remaining"] = min_games
                    record["last_onboarding_decision"] = "demoted_window"
                    decision = "demoted_window"

        agent.shadow_evaluation_record = record
        decisions.append(
            {
                "agent": agent.name,
                "decision": decision,
                "played": played,
                "wins": wins,
                "status": acting_status(agent),
                "applicable_dispatched_games": record.get(
                    "applicable_dispatched_games"
                ),
                "applicable_wins_as_primary": record.get(
                    "applicable_wins_as_primary"
                ),
            }
        )

    return {
        "decisions": decisions,
        "n_promoted": sum(1 for d in decisions if d["decision"] == "accepted"),
        "n_dormant": sum(1 for d in decisions if d["decision"] == "dormant"),
        "policy": {
            "min_games": min_games,
            "min_wins": min_wins,
            "epsilon": float(pol.epsilon),
        },
    }


def write_onboarding_report(path: str | Path, report: dict[str, Any]) -> None:
    write_json(path, report)


def refresh_delegated_statuses(organization, trajectories, skills, *, policy=None):
    """Track verified local subtask outcomes separately from whole-task wins."""
    from sage_tau2.contracts import local_skill_outcome
    pol = policy or OnboardingPolicy()
    bank = {s.skill_id: s for s in skills}
    decisions = []
    for agent in organization.agents:
        if agent.name == EXECUTOR_NAME:
            continue
        record = dict(agent.shadow_evaluation_record or {})
        played = wins = unknown = 0
        for traj in trajectories:
            if "infrastructure" in str(traj.termination_reason or "").lower():
                continue
            events = [e for e in traj.metadata.get("skill_events", []) or [] if e.get("actor") == agent.name]
            if not events:
                continue
            ids = {sid for e in events for sid in e.get("adopted_skill_ids", [])}
            outcomes = [local_skill_outcome(bank[sid], traj, events)['success'] for sid in ids if sid in bank]
            if not outcomes or any(value is None for value in outcomes) and not any(value is False for value in outcomes):
                unknown += 1
                continue
            played += 1
            wins += int(all(value is True for value in outcomes))
        if not (played or unknown):
            continue
        record['delegated_verified_games'] = int(record.get('delegated_verified_games') or 0) + played
        record['delegated_local_wins'] = int(record.get('delegated_local_wins') or 0) + wins
        record['delegated_unverified_games'] = int(record.get('delegated_unverified_games') or 0) + unknown
        decision = 'continue_probation'
        if acting_status(agent) == 'probation' and record.get('same_skill_probe_passed'):
            if record['delegated_verified_games'] >= pol.min_games and record['delegated_local_wins'] >= pol.min_wins:
                agent.acting_status = 'accepted'
                record['acting_status'] = 'accepted'
                decision = 'accepted'
            elif played and record['delegated_verified_games'] >= pol.min_games and record['delegated_local_wins'] < pol.min_wins:
                record['delegated_rejected_windows'] = int(record.get('delegated_rejected_windows') or 0) + 1
                if record['delegated_rejected_windows'] >= pol.remove_after_rejected_windows:
                    agent.acting_status = 'dormant'
                    record['acting_status'] = 'dormant'
                    decision = 'dormant'
        record['last_delegated_onboarding_decision'] = decision
        agent.shadow_evaluation_record = record
        decisions.append({'agent': agent.name, 'decision': decision, 'played': played,
                          'local_wins': wins, 'unverified': unknown, 'mode': 'delegated_subtasks'})
    return decisions
