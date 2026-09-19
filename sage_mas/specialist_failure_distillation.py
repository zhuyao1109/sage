"""Distill no-controller specialist failures into prompt protocol notes."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from sage_mas.schemas import AgentSpec, Skill
from sage_mas.specialist_state import extract_action_text, infer_specialist_state
from sage_mas.task_parser import (
    TASK_FAMILY_OPERATIONS,
    ParsedTask,
    parse_alfworld_task,
)


def update_specialist_failure_protocols(
    *,
    skills: list[Skill],
    agents: list[AgentSpec],
    trials: Iterable[Any],
    require_no_controller: bool = True,
) -> dict[str, Any]:
    """Attach failure-derived notes to scoped specialist skills.

    This is prompt distillation only: it updates skill metadata rendered to the
    LLM. It does not rewrite actions or install a task-family controller.
    """

    skill_by_name = {skill.skill_name: skill for skill in skills}
    agent_by_name = {agent.name: agent for agent in agents}
    updates: dict[str, dict[str, Any]] = {}
    skipped = Counter()
    for trial in trials:
        agent_name = _trial_value(trial, "assigned_primary_agent", "")
        if not agent_name or "executor" in str(agent_name).lower():
            skipped["executor_or_unassigned"] += 1
            continue
        if bool(_trial_value(trial, "won", False)):
            skipped["won"] += 1
            continue
        steps = list(_trial_value(trial, "steps", []) or [])
        if require_no_controller and _has_controller_rewrite(steps):
            skipped["controller_rewritten"] += 1
            continue
        agent = agent_by_name.get(str(agent_name))
        if agent is None:
            skipped["missing_agent"] += 1
            continue
        family = str(_trial_value(trial, "task_family", "") or "")
        notes = distill_failure_notes(trial)
        if not notes["anti_patterns"] and not notes["failure_branches"]:
            skipped["no_notes"] += 1
            continue
        for skill_name in agent.assigned_skills:
            skill = skill_by_name.get(skill_name)
            if skill is None:
                continue
            if family and skill.applicable_task_families:
                if family not in skill.applicable_task_families:
                    continue
            _merge_metadata_list(skill, "anti_patterns", notes["anti_patterns"], limit=8)
            _merge_metadata_list(
                skill,
                "failure_branches",
                notes["failure_branches"],
                limit=8,
            )
            record = skill.metadata.setdefault(
                "no_controller_failure_distillation",
                {
                    "failure_count": 0,
                    "task_families": defaultdict(int),
                    "updated_from": [],
                },
            )
            if isinstance(record.get("task_families"), defaultdict):
                family_counts = record["task_families"]
            else:
                family_counts = defaultdict(int, record.get("task_families") or {})
            family_counts[family or "unknown"] += 1
            record["failure_count"] = int(record.get("failure_count") or 0) + 1
            record["task_families"] = dict(family_counts)
            _merge_metadata_list(
                skill,
                "updated_from",
                [str(_trial_value(trial, "task_id", ""))],
                limit=20,
                container=record,
            )
            updates.setdefault(
                skill.skill_name,
                {
                    "skill_name": skill.skill_name,
                    "task_families": set(),
                    "anti_patterns_added": [],
                    "failure_branches_added": [],
                    "failure_count": 0,
                },
            )
            item = updates[skill.skill_name]
            item["task_families"].add(family or "unknown")
            item["failure_count"] += 1
            item["anti_patterns_added"].extend(notes["anti_patterns"])
            item["failure_branches_added"].extend(notes["failure_branches"])

    serializable = []
    for item in updates.values():
        serializable.append(
            {
                "skill_name": item["skill_name"],
                "task_families": sorted(item["task_families"]),
                "failure_count": int(item["failure_count"]),
                "anti_patterns_added": _dedupe(item["anti_patterns_added"])[:8],
                "failure_branches_added": _dedupe(item["failure_branches_added"])[:8],
            }
        )
    return {
        "updated": serializable,
        "updated_skill_count": len(serializable),
        "skipped": dict(skipped),
    }


def distill_failure_notes(trial: Any) -> dict[str, list[str]]:
    task = str(_trial_value(trial, "task", "") or "")
    gamefile = str(_trial_value(trial, "task_id", "") or "")
    family = str(_trial_value(trial, "task_family", "") or "")
    steps = [step for step in list(_trial_value(trial, "steps", []) or []) if isinstance(step, dict)]
    parsed = parse_alfworld_task(task, gamefile)
    if not parsed.task_family and family:
        parsed = ParsedTask(
            task=parsed.task,
            gamefile=parsed.gamefile,
            task_family=family,
            operation=parsed.operation or TASK_FAMILY_OPERATIONS.get(family),
            target=parsed.target,
            destination=parsed.destination,
        )
    final_observation = str(steps[-1].get("observation", "") if steps else "")
    state = infer_specialist_state(parsed, steps=steps, observation=final_observation)
    actions = [extract_action_text(step.get("action")) for step in steps]
    invalid_actions = [
        extract_action_text(step.get("action"))
        for step in steps
        if not bool(step.get("is_action_valid", True))
    ]
    anti: list[str] = []
    branches: list[str] = []
    repeated = _repeated_actions(actions)
    if repeated:
        anti.append(
            "Do not repeat the same search/open/navigation action three or "
            "more times without new observation evidence: "
            + ", ".join(repeated[:3])
            + "."
        )
    if invalid_actions:
        anti.append(
            "Avoid retrying recently invalid actions until the observation "
            "changes: "
            + ", ".join(_dedupe(invalid_actions)[-3:])
            + "."
        )
    if state.phase:
        branches.append(
            "Previous no-controller failure progress: "
            f"phase={state.phase}, held_target={state.held_target}, "
            f"operation_done={state.operation_done}, "
            f"placed_target={state.placed_target}. Compare this progress "
            "against the learned skill protocol and resume from the first "
            "unmet protocol step."
        )
    return {
        "anti_patterns": _dedupe(anti),
        "failure_branches": _dedupe(branches),
    }


def _has_controller_rewrite(steps: list[dict[str, Any]]) -> bool:
    return any(
        bool((step.get("action_guard") or {}).get("changed"))
        for step in steps
        if isinstance(step, dict)
    )


def _repeated_actions(actions: list[str]) -> list[str]:
    counts = Counter(action for action in actions if action)
    return [
        action
        for action, count in counts.most_common()
        if count >= 3 and not action.startswith(("take ", "put ", "move "))
    ]


def _merge_metadata_list(
    skill: Skill,
    key: str,
    values: list[str],
    *,
    limit: int,
    container: dict[str, Any] | None = None,
) -> None:
    target = container if container is not None else skill.metadata
    merged = _dedupe([str(item) for item in (target.get(key) or [])] + values)
    target[key] = merged[-limit:]


def _trial_value(trial: Any, key: str, default: Any = None) -> Any:
    if isinstance(trial, dict):
        return trial.get(key, default)
    return getattr(trial, key, default)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = " ".join(str(item or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
