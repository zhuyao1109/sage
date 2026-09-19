"""Prompt-only execution summaries for ALFWorld specialists.

These helpers summarize task state for the LLM actor. They never select,
rewrite, or validate an environment action.
"""

from __future__ import annotations

from collections import Counter

from sage_mas.specialist_action_guard import parse_admissible_actions
from sage_mas.specialist_state import extract_action_text, infer_specialist_state
from sage_mas.task_parser import (
    TASK_FAMILY_OPERATIONS,
    ParsedTask,
    parse_alfworld_task,
)


def build_specialist_execution_summary(
    *,
    task: str | None,
    task_family: str | None,
    observation_prompt: str,
    history_steps: list | None,
) -> str:
    """Return LLM-facing state context for a dispatched specialist."""

    parsed = _parse_task(task, task_family)
    admissible = parse_admissible_actions(observation_prompt)
    steps = [step for step in (history_steps or []) if isinstance(step, dict)]
    state = infer_specialist_state(
        parsed,
        steps=steps,
        observation=observation_prompt,
        admissible_actions=admissible,
    )
    lines = [
        "Specialist execution summary:",
        (
            "- Parsed task: "
            f"family={parsed.task_family or 'unknown'}, "
            f"operation={parsed.operation or 'none'}, "
            f"target={parsed.target or 'unknown'}, "
            f"destination={parsed.destination or 'unknown'}."
        ),
        (
            "- Current progress: "
            f"phase={state.phase}, held_target={state.held_target}, "
            f"operation_done={state.operation_done}, "
            f"placed_target={state.placed_target}, "
            f"target_visible={state.target_visible}, "
            f"destination_visible={state.destination_visible}."
        ),
    ]
    if state.visited_locations:
        lines.append(
            "- Visited locations: "
            + ", ".join(sorted(state.visited_locations)[-8:])
            + "."
        )
    if state.opened_receptacles:
        lines.append(
            "- Opened receptacles: "
            + ", ".join(sorted(state.opened_receptacles)[-8:])
            + "."
        )
    failures = _failure_memory(steps)
    if failures:
        lines.append("- Recent failure memory:")
        lines.extend(f"  - {item}" for item in failures)
    repeats = _repeat_memory(steps)
    if repeats:
        lines.append("- Repetition memory:")
        lines.extend(f"  - {item}" for item in repeats)
    lines.extend(
        [
            "- Use this summary only as context. You must still choose the next",
            "  action yourself from the admissible actions in the observation.",
            "- Do not repeat a recent invalid action unless the observation has",
            "  changed in a way that makes it newly applicable.",
        ]
    )
    return "\n".join(lines)


def _parse_task(task: str | None, task_family: str | None) -> ParsedTask:
    parsed = parse_alfworld_task(str(task or ""))
    family = parsed.task_family or str(task_family or "").strip()
    operation = parsed.operation or TASK_FAMILY_OPERATIONS.get(family)
    return ParsedTask(
        task=parsed.task,
        gamefile=parsed.gamefile,
        task_family=family,
        operation=operation,
        target=parsed.target,
        destination=parsed.destination,
    )


def _failure_memory(steps: list[dict]) -> list[str]:
    rows: list[str] = []
    window = steps[-10:]
    for index, step in enumerate(window):
        action = extract_action_text(step.get("action"))
        if not action:
            continue
        observation = " ".join(str(step.get("observation") or "").split()).lower()
        valid = bool(step.get("is_action_valid", True))
        no_progress = float(step.get("goal_progress_delta") or 0.0) <= 0.0
        if not valid:
            rows.append(f"`{action}` was invalid.")
        elif "nothing happens" in observation:
            rows.append(f"`{action}` produced no useful change.")
        elif no_progress and action in _last_actions(window[:index], 4):
            rows.append(f"`{action}` was recently repeated without progress.")
    return _dedupe(rows)[-4:]


def _repeat_memory(steps: list[dict]) -> list[str]:
    actions = [extract_action_text(step.get("action")) for step in steps[-8:]]
    actions = [action for action in actions if action]
    counts = Counter(actions)
    rows = [
        f"`{action}` occurred {count} times in the recent window."
        for action, count in sorted(counts.items())
        if count >= 3
    ]
    if len(actions) >= 2 and actions[-1] == actions[-2]:
        rows.append(f"The last two actions were both `{actions[-1]}`.")
    return _dedupe(rows)


def _last_actions(steps: list[dict], limit: int) -> set[str]:
    return {
        extract_action_text(step.get("action"))
        for step in steps[-limit:]
        if extract_action_text(step.get("action"))
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
