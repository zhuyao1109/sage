"""Deterministic action guard for dispatched ALFWorld specialists."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sage_mas.schemas import AgentSpec, Skill
from sage_mas.specialist_controllers import (
    DEFAULT_CONTROLLER_REGISTRY,
    SpecialistControllerRegistry,
    parse_admissible_actions,
)
from sage_mas.specialist_state import extract_action_text
from sage_mas.task_parser import parse_alfworld_task


@dataclass(frozen=True, slots=True)
class GuardResult:
    action: str
    changed: bool
    reason: str
    phase: str
    raw_action_text: str
    selected_action_text: str
    state: dict[str, Any]
    controller: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "reason": self.reason,
            "phase": self.phase,
            "raw_action_text": self.raw_action_text,
            "selected_action_text": self.selected_action_text,
            "state": self.state,
            "controller": self.controller,
        }


class SpecialistActionGuard:
    """Route specialist actions through their scoped execution controller."""

    def __init__(
        self,
        registry: SpecialistControllerRegistry | None = None,
        *,
        enabled: bool = True,
    ):
        self.registry = registry or DEFAULT_CONTROLLER_REGISTRY
        self.enabled = bool(enabled)

    def guard(
        self,
        *,
        raw_action: str,
        task: str,
        gamefile: str,
        observation: str,
        prompt: str,
        steps: list[dict[str, Any]],
        primary_agent: str | None,
        agents: list[AgentSpec],
        skills: list[Skill],
    ) -> GuardResult:
        raw_text = extract_action_text(raw_action)
        if not self.enabled:
            return self._unchanged(
                raw_action,
                raw_text,
                "specialist_controllers_disabled",
            )
        parsed = parse_alfworld_task(task, gamefile)
        controller = self.registry.controller_for(
            parsed=parsed,
            primary_agent=primary_agent,
            agents=agents,
            skills=skills,
        )
        if controller is None:
            return self._unchanged(raw_action, raw_text, "no scoped specialist controller")

        admissible = parse_admissible_actions(prompt)
        decision = controller.decide(
            parsed=parsed,
            raw_action_text=raw_text,
            observation=observation,
            admissible_actions=admissible,
            steps=steps,
        )
        selected = decision.selected_action_text
        reason = decision.reason
        if selected is None:
            selected = raw_text if raw_text in admissible else "look"
            reason = "no deterministic controller fallback"
        changed = selected != raw_text
        tagged = _tag_action(
            selected,
            decision.phase,
            reason,
            controller=decision.controller,
        )
        return GuardResult(
            action=tagged if changed or not _is_tagged_action(raw_action) else raw_action,
            changed=changed,
            reason=reason,
            phase=decision.phase,
            raw_action_text=raw_text,
            selected_action_text=selected,
            state=decision.state,
            controller=decision.controller,
        )

    @staticmethod
    def _unchanged(raw_action: str, raw_text: str, reason: str) -> GuardResult:
        return GuardResult(
            action=raw_action,
            changed=False,
            reason=reason,
            phase="UNGUARDED",
            raw_action_text=raw_text,
            selected_action_text=raw_text,
            state={},
            controller=None,
        )


def parse_admissible_actions(prompt: str) -> list[str]:
    match = re.search(
        r"admissible actions of the current situation are:\s*\[(.*?)\]\s*\.",
        str(prompt or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    source = match.group(1) if match else str(prompt or "")
    actions = []
    for quoted in re.finditer(r"'([^']+)'|\"([^\"]+)\"", source):
        action = extract_action_text(quoted.group(1) or quoted.group(2))
        if action and action not in actions:
            actions.append(action)
    return actions


def _tag_action(
    action: str,
    phase: str,
    reason: str,
    *,
    controller: str,
) -> str:
    return f"<think>{controller}: {phase}; {reason}.</think><action>{action}</action>"


def _is_tagged_action(action: str) -> bool:
    return bool(re.search(r"<action>.*?</action>", str(action or ""), re.IGNORECASE | re.DOTALL))
