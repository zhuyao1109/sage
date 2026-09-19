"""Sandboxed capability discovery grounded by environment-confirmed transitions."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from sage_mas.alfworld_evaluator import (
    AlfWorldOrganizationEvaluator,
    EvaluationTrial,
)
from sage_mas.runtime import ChatBackend
from sage_mas.schemas import AgentSpec, AtomicOp, Skill, SkillStatus
from sage_mas.serialization import to_primitive
from sage_mas.trajectory.abstraction import abstract_environment_action


def _environment_action(text: str) -> str:
    match = re.search(r"<action>(.*?)</action>", str(text), re.DOTALL | re.I)
    value = match.group(1) if match else str(text)
    return " ".join(value.strip().lower().split())


@dataclass(slots=True)
class ConfirmedCapabilityTransition:
    task_id: str
    attempt: int
    step: int
    operation: str
    action: str
    observation: str


class CapabilityTransitionDetector:
    """Detect local effects from action/observation evidence, not episode reward."""

    def __init__(self, operation: str | None = None):
        if operation is None:
            self.operation = ""
            return
        operation = operation.strip().lower()
        if not operation or not re.fullmatch(r"[a-z][a-z0-9_-]*", operation):
            raise ValueError(f"Invalid capability operation: {operation!r}")
        self.operation = operation

    def detect(
        self,
        trial: EvaluationTrial,
        *,
        attempt: int,
    ) -> list[ConfirmedCapabilityTransition]:
        transitions = []
        for index, step in enumerate(trial.steps, start=1):
            action = _environment_action(str(step.get("action", "")))
            observation = str(step.get("observation", "")).strip()
            observation_lower = observation.lower()
            operation = self.operation or action.split(" ", 1)[0]
            actual_operation = action.split(" ", 1)[0]
            progress_delta = float(
                step.get("goal_progress_delta", 0.0) or 0.0
            )
            action_confirmed = bool(
                operation
                and (not self.operation or action.startswith(f"{operation} "))
                and re.search(
                    rf"\b{re.escape(operation)}(?:s|ed|ing)?\b",
                    observation_lower,
                )
            )
            confirmed = bool(step.get("is_action_valid", False)) and (
                progress_delta > 0.0
                or float(step.get("reward", 0.0) or 0.0) > 0.0
                or action_confirmed
            )
            if confirmed:
                transitions.append(
                    ConfirmedCapabilityTransition(
                        task_id=trial.task_id,
                        attempt=attempt,
                        step=index,
                        operation=actual_operation or "environment_progress",
                        action=action,
                        observation=observation,
                    )
                )
        return transitions


@dataclass(slots=True)
class DiscoveryForkOutcome:
    experimental_skill: Skill
    grounded_skill: Skill | None
    trials: list[EvaluationTrial]
    confirmed_transitions: list[ConfirmedCapabilityTransition]
    confirmed_task_count: int
    attempts_per_task: int


def merge_discovery_outcomes(
    outcomes: list[DiscoveryForkOutcome],
) -> DiscoveryForkOutcome:
    """Accumulate independent confirmations across discovery hypotheses."""
    if not outcomes:
        raise ValueError("At least one discovery outcome is required")
    transitions = [
        transition
        for outcome in outcomes
        for transition in outcome.confirmed_transitions
    ]
    grounded_source = next(
        (
            outcome.grounded_skill
            for outcome in outcomes
            if outcome.grounded_skill is not None
        ),
        None,
    )
    grounded = deepcopy(grounded_source)
    if grounded is not None:
        evidence_ids = sorted(
            {transition.task_id for transition in transitions}
        )
        grounded.evidence_ids = evidence_ids
        grounded.support_count = len(evidence_ids)
        grounded.metadata["confirmed_transitions"] = to_primitive(
            transitions
        )
        grounded.metadata["discovery_hypothesis_count"] = len(outcomes)
    return DiscoveryForkOutcome(
        experimental_skill=outcomes[-1].experimental_skill,
        grounded_skill=grounded,
        trials=[
            trial
            for outcome in outcomes
            for trial in outcome.trials
        ],
        confirmed_transitions=transitions,
        confirmed_task_count=len(
            {transition.task_id for transition in transitions}
        ),
        attempts_per_task=sum(
            outcome.attempts_per_task for outcome in outcomes
        ),
    )


class ExperimentalProtocolGenerator:
    """Use failed experience to propose a hypothesis; it is never deployable."""

    def __init__(self, backend: ChatBackend):
        self.backend = backend

    def generate(
        self,
        *,
        operation: str | None = None,
        capability_key: str | None = None,
        task_family: str,
        failed_trajectories: list[dict[str, Any]],
    ) -> Skill:
        evidence = [
            {
                "task": trajectory.get("task"),
                "task_family": trajectory.get("task_family"),
                "won": trajectory.get("won"),
                "steps": [
                    {
                        "action": _environment_action(step.get("action", "")),
                        "observation": str(step.get("observation", ""))[:240],
                        "is_action_valid": step.get("is_action_valid"),
                    }
                    for step in (trajectory.get("steps") or [])[-20:]
                ],
            }
            for trajectory in failed_trajectories[:4]
        ]
        result = self.backend.complete(
            (
                "You propose one experimental environment-action protocol from "
                "failed agent experience. The protocol is a hypothesis, not a "
                "verified Skill. Do not claim success and do not add domain "
                "rules beyond actions inferable from the task, observations, "
                "and the environment action syntax present in the evidence. "
                "Return one JSON object."
            ),
            (
                f"Target capability: {capability_key or operation or task_family}\n"
                f"Task scope: {task_family}\n"
                f"Failed experience:\n{json.dumps(evidence, ensure_ascii=False)}\n"
                "Return keys: skill_name, description, precondition, "
                "action_protocol (non-empty list of environment-action "
                "instructions), expected_effect."
            ),
            max_completion_tokens=900,
        )
        payload = self._parse_json(result.content)
        protocol = []
        for step in payload.get("action_protocol", []):
            if isinstance(step, dict):
                value = (
                    step.get("action")
                    or step.get("instruction")
                    or step.get("step")
                    or ""
                )
            else:
                value = step
            normalized = str(value).strip()
            if normalized:
                protocol.append(normalized)
        if not protocol:
            raise ValueError("Experimental protocol contains no actions")
        return Skill(
            skill_name=str(
                payload.get("skill_name")
                or f"Experimental {capability_key or operation or task_family} capability"
            ),
            description=str(payload.get("description") or "").strip(),
            precondition=str(payload.get("precondition") or "").strip(),
            action_protocol=protocol,
            applicable_atomic_ops=[AtomicOp.ACT],
            expected_effect=str(payload.get("expected_effect") or "").strip(),
            status=SkillStatus.CANDIDATE,
            capability_key=(
                capability_key
                or (f"transform.{operation}" if operation else f"experience.{task_family}")
            ),
            applicable_task_families=[task_family],
            metadata={
                "candidate_stage": "discovery_hypothesis",
                "capability_operation": operation or "",
                "capability_key": (
                    capability_key
                    or (
                        f"transform.{operation}"
                        if operation
                        else f"experience.{task_family}"
                    )
                ),
                "primary_task_family": task_family,
                "task_families": [task_family],
                "deployable": False,
            },
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Experimental protocol response is not JSON")
        payload = json.loads(content[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Experimental protocol response must be an object")
        return payload


class DiscoveryForkService:
    def __init__(
        self,
        evaluator: AlfWorldOrganizationEvaluator,
        *,
        operation: str | None = None,
        capability_key: str | None = None,
        task_family: str | None = None,
        attempts_per_task: int = 2,
    ):
        self.evaluator = evaluator
        self.operation = str(operation or "").strip().lower()
        self.capability_key = (
            capability_key
            or (
                f"transform.{self.operation}"
                if self.operation
                else "experience.environment_progress"
            )
        )
        self.task_family = str(task_family or "").strip()
        self.attempts_per_task = max(1, int(attempts_per_task))
        self.detector = CapabilityTransitionDetector(self.operation or None)
        self.evaluator.config.allow_executor_skill_injection = True
        # Discovery tests the hypothesis from the beginning. This is isolated
        # from both the active organization and the held-out probe partition.
        # Forced injection bypasses prompt-based retrieval: the experimental
        # skill is always mounted.
        self.evaluator.config.force_inject_skills = True

    def run(
        self,
        *,
        experimental_skill: Skill,
        agents: list[AgentSpec],
        baseline_skills: list[Skill],
        gamefiles: list[str],
    ) -> DiscoveryForkOutcome:
        trials: list[EvaluationTrial] = []
        transitions: list[ConfirmedCapabilityTransition] = []
        for attempt in range(1, self.attempts_per_task + 1):
            wave = self.evaluator.evaluate(
                agents=agents,
                skills=baseline_skills,
                gamefiles=gamefiles,
                condition=f"discovery_attempt_{attempt}",
                injected_skills=[experimental_skill],
            )
            trials.extend(wave)
            for trial in wave:
                detected = self.detector.detect(trial, attempt=attempt)
                if not self.operation and detected:
                    # Generic discovery grounds the latest observable state
                    # transition, avoiding early navigation/opening effects.
                    detected = [detected[-1]]
                transitions.extend(detected)
        confirmed_tasks = {transition.task_id for transition in transitions}
        grounded = self._ground_confirmed_skill(
            experimental_skill,
            trials,
            transitions,
        )
        return DiscoveryForkOutcome(
            experimental_skill=experimental_skill,
            grounded_skill=grounded,
            trials=trials,
            confirmed_transitions=transitions,
            confirmed_task_count=len(confirmed_tasks),
            attempts_per_task=self.attempts_per_task,
        )

    def _ground_confirmed_skill(
        self,
        hypothesis: Skill,
        trials: list[EvaluationTrial],
        transitions: list[ConfirmedCapabilityTransition],
    ) -> Skill | None:
        if not transitions:
            return None
        transition_by_task_attempt = {
            (transition.task_id, transition.attempt): transition
            for transition in transitions
        }
        indexed_trials: list[tuple[EvaluationTrial, int, ConfirmedCapabilityTransition]] = []
        attempt_counts: dict[str, int] = {}
        for trial in trials:
            attempt_counts[trial.task_id] = attempt_counts.get(trial.task_id, 0) + 1
            attempt = attempt_counts[trial.task_id]
            transition = transition_by_task_attempt.get((trial.task_id, attempt))
            if transition is not None:
                indexed_trials.append((trial, attempt, transition))
        source_trial, _, source_transition = min(
            indexed_trials,
            key=lambda item: (item[2].step, item[0].num_steps, item[0].task_id),
        )
        window = source_trial.steps[
            max(0, source_transition.step - 6) : source_transition.step
        ]
        protocol: list[str] = []
        for step in window:
            action = abstract_environment_action(
                _environment_action(str(step.get("action", "")))
            )
            if action and (not protocol or protocol[-1] != action):
                protocol.append(action)
        evidence_ids = sorted({transition.task_id for transition in transitions})
        return Skill(
            skill_name=(
                f"Environment-confirmed {self.operation or source_trial.task_family} protocol"
            ),
            description=(
                f"Local {self.capability_key} capability compressed only from "
                "environment-confirmed state transitions."
            ),
            precondition=(
                "The current task is inside the empirically confirmed "
                f"applicability scope {source_trial.task_family}."
            ),
            action_protocol=protocol,
            applicable_atomic_ops=[AtomicOp.ACT],
            expected_effect=source_transition.observation,
            status=SkillStatus.CANDIDATE,
            evidence_ids=evidence_ids,
            support_count=len(evidence_ids),
            capability_key=self.capability_key,
            applicable_task_families=[source_trial.task_family],
            metadata={
                "candidate_stage": "environment_confirmed_discovery",
                "capability_operation": (
                    self.operation or source_transition.operation
                ),
                "source_signal": "environment_progress",
                "anchor_state": str(
                    source_trial.steps[
                        source_transition.step - 1
                    ].get("observation_before", "")
                    or source_trial.steps[
                        source_transition.step - 1
                    ].get("observation", "")
                )[:500],
                "primary_task_family": source_trial.task_family,
                "task_families": [source_trial.task_family],
                "deployable": False,
                "confirmed_transitions": to_primitive(transitions),
                "experimental_skill_id": hypothesis.skill_id,
            },
        )
