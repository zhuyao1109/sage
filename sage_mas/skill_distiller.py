"""Candidate-skill extraction from adapted trajectories."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

from sage_mas.schemas import AtomicOp, AtomicStep, Skill


@dataclass(slots=True)
class DistillationConfig:
    high_cost_steps: int = 25
    min_support: int = 1
    operation_min_support: int = 1
    min_family_support: int | None = None
    max_repair_signals_per_trajectory: int = 3


@dataclass(slots=True)
class ExperienceRecord:
    """A reusable piece of evidence extracted from one trajectory."""

    trajectory_id: str
    task_family: str
    task: str
    signal: str
    outcome_class: str
    won: bool
    num_steps: int
    trigger_state: str
    key_transition: str
    anti_pattern: str | None = None
    # When the LLM signal extractor is active, this is set directly.
    # Identity is not filled from a signal table.
    capability_key: str = ""
    observed_actions: list[str] = field(default_factory=list)


class LLMSignalExtractor:
    """LLM-driven signal and capability_key extraction from trajectories.

    Replaces the hardcoded ``_repair_signal_matches`` /
    ``_diagnose_success_trajectory`` rules and the
    ``_CAPABILITY_BY_SIGNAL`` mapping table with an LLM that analyzes
    trajectory content and generates a descriptive signal name plus a
    capability category — no ALFWorld domain knowledge is embedded.
    """

    def __init__(self, backend: DistillationBackend):
        self.backend = backend

    def extract(
        self,
        *,
        task: str,
        task_family: str,
        won: bool,
        num_steps: int,
        action_steps: list[AtomicStep],
    ) -> tuple[str, str] | None:
        """Return ``(signal, capability_key)`` or ``None`` on failure."""
        actions = [
            str(step.action or "").strip()
            for step in action_steps[:25]
            if str(step.action or "").strip()
        ]
        trajectory_text = " → ".join(actions)
        prompt = self._build_prompt(
            task, task_family, won, num_steps, trajectory_text
        )
        try:
            result = self.backend.complete("", prompt)
            return self._parse(result.content)
        except Exception:
            return None

    @staticmethod
    def _build_prompt(
        task: str,
        task_family: str,
        won: bool,
        num_steps: int,
        trajectory_text: str,
    ) -> str:
        outcome = "succeeded" if won else "failed"
        return "\n".join(
            [
                "Analyze this agent trajectory and identify the capability "
                "it demonstrates (or fails to demonstrate).",
                "",
                f"Task: {task}",
                f"Task family: {task_family}",
                f"Outcome: {outcome} in {num_steps} steps",
                f"Action sequence: {trajectory_text}",
                "",
                "Output a JSON object with exactly two fields:",
                '  "signal": a snake_case name describing what happened',
                '  "capability_key": a dotted category (domain.operation)',
                "",
                "Rules:",
                "- For a failed trajectory, describe what capability was "
                "missing or what went wrong (e.g. \"missing_cool_operation\").",
                "- For a successful trajectory, describe what capability was "
                "demonstrated (e.g. \"cool_operation_success\").",
                "- capability_key must be in domain.operation format "
                "(e.g. \"transform.cool\", \"track.place\", "
                "\"inspect.with_light\").",
                "- Be concise and consistent: similar trajectories should "
                "produce the same signal and capability_key.",
                "Output only the JSON object, no explanation.",
            ]
        )

    @staticmethod
    def _parse(content: str) -> tuple[str, str] | None:
        import json

        text = str(content or "").strip()
        if not text:
            return None
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = match.group(1) if match else None
        if candidate is None:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None
        signal = str(parsed.get("signal", "")).strip()
        capability_key = str(parsed.get("capability_key", "")).strip()
        if not signal or not capability_key:
            return None
        # Basic format validation.
        if not re.match(r"^[a-z][a-z0-9_]*$", signal):
            return None
        if not re.match(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$", capability_key):
            return None
        return (signal, capability_key)


@dataclass(slots=True)
class ExperienceCluster:
    """A group of aligned experiences that can seed one candidate skill."""

    signal: str
    task_family: str
    experiences: list[ExperienceRecord]

    @property
    def evidence_ids(self) -> list[str]:
        return sorted({experience.trajectory_id for experience in self.experiences})

    @property
    def support_count(self) -> int:
        return len(self.evidence_ids)


# Bounded local-repair tags aligned with SkillMAS diagnosability.
# A failed trajectory may contribute several ordered experience signals.
OPERATION_SIGNALS = (
    "missing_clean_operation",
    "missing_heat_operation",
    "missing_cool_operation",
    "missing_light_operation",
)
STRUCTURED_REPAIR_SIGNALS = (
    "incomplete_multi_object",
    "search_exhaustion",
)
GENERIC_REPAIR_SIGNALS = (
    "admissible_action_guard",
    "stagnation_recovery",
    "repetition_guard",
    "progress_checkpoint",
    "failure_checkpoint",
)
SUCCESS_OPERATION_SIGNALS = (
    "clean_operation_success",
    "heat_operation_success",
    "cool_operation_success",
    "light_operation_success",
)
SUCCESS_SIGNALS = (
    SUCCESS_OPERATION_SIGNALS
    + (
        "multi_object_success",
        "efficient_execution",
        "task_completion",
    )
)
LOCAL_REPAIR_SIGNALS = (
    OPERATION_SIGNALS + STRUCTURED_REPAIR_SIGNALS + GENERIC_REPAIR_SIGNALS
)
ALL_SIGNALS = LOCAL_REPAIR_SIGNALS + SUCCESS_SIGNALS
SIGNAL_PRIORITY = {signal: index for index, signal in enumerate(ALL_SIGNALS)}


class DistillationBackend(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> Any:
        ...


class HeuristicSkillDistiller:
    """Transparent seed distiller used before an LLM distiller is connected.

    It only proposes candidates. It never marks a skill as verified.

    When ``signal_extractor`` is provided, trajectory signals and
    capability keys are generated by the LLM instead of the hardcoded
    ``_repair_signal_matches`` / ``_diagnose_success_trajectory`` rules
    and the ``_CAPABILITY_BY_SIGNAL`` mapping table.
    """

    def __init__(
        self,
        config: DistillationConfig | None = None,
        *,
        signal_extractor: LLMSignalExtractor | None = None,
    ):
        self.config = config or DistillationConfig()
        self.signal_extractor = signal_extractor

    def distill(self, trajectories: list[list[AtomicStep]]) -> list[Skill]:
        experiences: list[ExperienceRecord] = []
        for steps in trajectories:
            experiences.extend(self.extract_experiences(steps))

        candidates: list[Skill] = []
        for cluster in self.cluster_experiences(experiences):
            required_support = self._required_support(cluster.signal)
            if cluster.support_count < required_support:
                continue
            candidates.append(
                self._build_skill(
                    cluster.signal,
                    cluster.evidence_ids,
                    task_family=cluster.task_family,
                    cluster=cluster,
                )
            )
        return self._deduplicate_candidates(candidates)

    def _required_support(self, signal: str) -> int:
        if signal in OPERATION_SIGNALS or signal in SUCCESS_OPERATION_SIGNALS:
            return self.config.operation_min_support
        family_support = (
            self.config.min_family_support
            if self.config.min_family_support is not None
            else self.config.min_support
        )
        return family_support

    def extract_experiences(
        self,
        steps: list[AtomicStep],
    ) -> list[ExperienceRecord]:
        """Extract reusable trajectory experiences before templating skills."""
        if not steps:
            return []
        terminal = steps[-1]
        action_steps = [
            step
            for step in steps[:-1]
            if step.atomic_op != AtomicOp.COMMUNICATE
        ]
        task = str(terminal.metadata.get("task", "")).lower()
        task_family = str(
            terminal.metadata.get("task_family", "")
        ).lower()
        num_steps = int(terminal.metadata.get("num_steps", len(action_steps)))
        trajectory_id = self._trajectory_id(steps)
        won = bool(terminal.metadata.get("won", False))

        # LLM-driven signal extraction (no hardcoded domain rules).
        if self.signal_extractor is not None:
            result = self.signal_extractor.extract(
                task=task,
                task_family=task_family,
                won=won,
                num_steps=num_steps,
                action_steps=action_steps,
            )
            if result is not None:
                signal, capability_key = result
                return [
                    self._experience_for_signal(
                        signal,
                        trajectory_id=trajectory_id,
                        task_family=task_family,
                        task=task,
                        won=won,
                        num_steps=num_steps,
                        action_steps=action_steps,
                        capability_key=capability_key,
                    )
                ]
            # A failed naming call does not invent a capability from a table.

        if won:
            # Keep the win as evidence. The environment already records the
            # outcome and the task family; do not rename it into a preset
            # operation signal.
            return [
                self._experience_for_signal(
                    "won",
                    trajectory_id=trajectory_id,
                    task_family=task_family,
                    task=task,
                    won=True,
                    num_steps=num_steps,
                    action_steps=action_steps,
                )
            ]

        progress_steps = [
            step
            for step in action_steps
            if self._positive_progress(step.metadata.get("goal_progress_delta"))
        ]
        if progress_steps:
            step = progress_steps[-1]
            action = " ".join(str(step.action or "").lower().split())
            operation = action.split(" ", 1)[0] if action else "action"
            step_index = action_steps.index(step)
            state_before = str(
                step.metadata.get("observation_before", "") or ""
            ).strip()
            if not state_before and step_index > 0:
                state_before = str(
                    action_steps[step_index - 1].observation or ""
                ).strip()
            return [
                ExperienceRecord(
                    trajectory_id=trajectory_id,
                    task_family=task_family or "other",
                    task=task,
                    signal=f"environment_progress_{operation}",
                    outcome_class="local_effect",
                    won=False,
                    num_steps=num_steps,
                    trigger_state=state_before,
                    key_transition=self._step_summary(step),
                )
            ]

        # A failed trajectory without an observed positive environment effect
        # may explain a mistake, but it cannot define an executable Skill.
        return []

    @staticmethod
    def _positive_progress(value: Any) -> bool:
        try:
            return float(value or 0.0) > 0.0
        except (TypeError, ValueError):
            return False

    def _diagnose_trajectory(self, steps: list[AtomicStep]) -> str | None:
        """Return the highest-priority signal for backward compatibility."""
        experiences = self.extract_experiences(steps)
        if not experiences:
            return None
        return experiences[0].signal

    def _repair_signal_matches(
        self,
        action_steps: list[AtomicStep],
        task: str,
        task_family: str,
        num_steps: int,
    ) -> list[str]:
        checks: list[tuple[str, bool]] = [
            (
                "missing_clean_operation",
                self._requires(task_family, "clean")
                and not self._has_operation(action_steps, "clean"),
            ),
            (
                "missing_heat_operation",
                self._requires(task_family, "heat")
                and not self._has_operation(action_steps, "heat"),
            ),
            (
                "missing_cool_operation",
                self._requires(task_family, "cool")
                and not self._has_operation(action_steps, "cool"),
            ),
            (
                "missing_light_operation",
                self._requires(task_family, "light")
                and not self._has_operation(action_steps, "use desklamp"),
            ),
            (
                "admissible_action_guard",
                any(
                    step.metadata.get("is_action_valid") is False
                    for step in action_steps
                ),
            ),
            (
                "search_exhaustion",
                self._has_search_exhaustion(action_steps, num_steps),
            ),
            (
                "incomplete_multi_object",
                self._has_incomplete_multi_object(
                    action_steps,
                    task,
                    task_family,
                    num_steps,
                ),
            ),
            (
                "repetition_guard",
                self._has_unproductive_repetition(action_steps),
            ),
            (
                "stagnation_recovery",
                self._has_material_stagnation(action_steps),
            ),
            (
                "progress_checkpoint",
                num_steps >= self.config.high_cost_steps
                and self._is_multi_stage_task(task, task_family),
            ),
            (
                "failure_checkpoint",
                self._has_incomplete_execution(
                    action_steps,
                    task,
                    task_family,
                    num_steps,
                ),
            ),
        ]
        return [signal for signal, matched in checks if matched]

    def _experience_for_signal(
        self,
        signal: str,
        *,
        trajectory_id: str,
        task_family: str,
        task: str,
        won: bool,
        num_steps: int,
        action_steps: list[AtomicStep],
        capability_key: str = "",
    ) -> ExperienceRecord:
        return ExperienceRecord(
            trajectory_id=trajectory_id,
            task_family=task_family or "other",
            task=task,
            signal=signal,
            outcome_class="success" if won else "failure",
            won=won,
            num_steps=num_steps,
            trigger_state=self._trigger_state_summary(
                signal,
                task=task,
                task_family=task_family,
                num_steps=num_steps,
                action_steps=action_steps,
            ),
            key_transition=self._key_transition_summary(signal, action_steps),
            anti_pattern=None if won else self._anti_pattern_summary(signal),
            capability_key=capability_key,
            observed_actions=self._observed_actions(action_steps),
        )

    @staticmethod
    def _observed_actions(action_steps: list[AtomicStep]) -> list[str]:
        """Environment actions from the trajectory, in order. No verb filter."""
        actions: list[str] = []
        for step in action_steps:
            action = " ".join(str(step.action or "").split())
            if action and action not in actions:
                actions.append(action)
        return actions[-8:]

    @staticmethod
    def cluster_experiences(
        experiences: list[ExperienceRecord],
    ) -> list[ExperienceCluster]:
        grouped: dict[tuple[str, str], list[ExperienceRecord]] = defaultdict(list)
        for experience in experiences:
            grouped[(experience.signal, experience.task_family)].append(experience)
        clusters = [
            ExperienceCluster(
                signal=signal,
                task_family=task_family,
                experiences=items,
            )
            for (signal, task_family), items in grouped.items()
        ]
        return sorted(
            clusters,
            key=lambda cluster: (
                SIGNAL_PRIORITY.get(cluster.signal, len(ALL_SIGNALS)),
                cluster.task_family,
                cluster.signal,
            ),
        )

    @staticmethod
    def _unique_limited(values: list[str], limit: int = 5) -> list[str]:
        output = []
        seen = set()
        for value in values:
            normalized = " ".join(str(value).split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
            if len(output) >= limit:
                break
        return output

    @classmethod
    def _cluster_metadata(cls, cluster: ExperienceCluster) -> dict[str, Any]:
        return {
            "signal": cluster.signal,
            "task_family": cluster.task_family,
            "support_count": cluster.support_count,
            "outcomes": sorted(
                {experience.outcome_class for experience in cluster.experiences}
            ),
            "trigger_states": cls._unique_limited(
                [experience.trigger_state for experience in cluster.experiences]
            ),
            "key_transitions": cls._unique_limited(
                [experience.key_transition for experience in cluster.experiences]
            ),
            "anti_patterns": cls._unique_limited(
                [
                    experience.anti_pattern or ""
                    for experience in cluster.experiences
                ]
            ),
            "evidence_ids": cluster.evidence_ids,
        }

    def _trigger_state_summary(
        self,
        signal: str,
        *,
        task: str,
        task_family: str,
        num_steps: int,
        action_steps: list[AtomicStep],
    ) -> str:
        operation_by_signal = {
            "missing_clean_operation": "clean",
            "missing_heat_operation": "heat",
            "missing_cool_operation": "cool",
            "clean_operation_success": "clean",
            "heat_operation_success": "heat",
            "cool_operation_success": "cool",
        }
        if signal in operation_by_signal:
            operation = operation_by_signal[signal]
            return (
                f"task_family={task_family}; task requires {operation} "
                "transformation before final placement"
            )
        if signal in {"missing_light_operation", "light_operation_success"}:
            return (
                f"task_family={task_family}; task requires object inspection "
                "with a desk lamp"
            )
        if signal == "search_exhaustion":
            go_locations = {
                str(step.action or "").strip().lower()
                for step in action_steps
                if str(step.action or "").strip().lower().startswith("go to ")
            }
            return (
                f"searched {len(go_locations)} distinct locations without "
                "acquiring the target"
            )
        if signal == "incomplete_multi_object":
            required = self._required_instance_count(task, task_family)
            progress = max(
                self._count_take_actions(action_steps),
                self._count_placement_actions(action_steps),
            )
            return f"multi-instance task progress={progress}/{required}"
        if signal == "repetition_guard":
            repeated = self._first_repeated_action(action_steps) or "unknown action"
            return f"action repeated without new evidence: {repeated}"
        if signal == "stagnation_recovery":
            return "latest or repeated observation reports no state change"
        if signal == "admissible_action_guard":
            return "trajectory contains an invalid environment action"
        if signal == "progress_checkpoint":
            return f"long-horizon task reached {num_steps} steps"
        if signal == "failure_checkpoint":
            return "failed trajectory reached a major subgoal or termination checkpoint"
        if signal == "multi_object_success":
            return "successful multi-instance placement trajectory"
        if signal == "efficient_execution":
            return f"successful trajectory completed within {num_steps} steps"
        return f"successful trajectory in {task_family}"

    @classmethod
    def _key_transition_summary(
        cls,
        signal: str,
        action_steps: list[AtomicStep],
    ) -> str:
        step = cls._representative_step(signal, action_steps)
        if step is None:
            return "no action transition available"
        return cls._step_summary(step)

    @staticmethod
    def _anti_pattern_summary(signal: str) -> str:
        anti_patterns = {
            "missing_clean_operation": "placing before cleaning succeeds",
            "missing_heat_operation": "placing before heating succeeds",
            "missing_cool_operation": "placing before cooling succeeds",
            "missing_light_operation": "inspecting before desk-lamp activation",
            "search_exhaustion": "revisiting searched locations instead of expanding search",
            "incomplete_multi_object": "stopping after partial instance completion",
            "repetition_guard": "repeating an action in an equivalent state",
            "stagnation_recovery": "continuing after no observable progress",
            "admissible_action_guard": "issuing invalid or non-admissible actions",
            "progress_checkpoint": "losing track of ordered subgoals",
            "failure_checkpoint": "assuming completion without evidence",
        }
        return anti_patterns.get(signal, "unverified failure pattern")

    @classmethod
    def _representative_step(
        cls,
        signal: str,
        action_steps: list[AtomicStep],
    ) -> AtomicStep | None:
        if not action_steps:
            return None
        if signal == "admissible_action_guard":
            for step in action_steps:
                if step.metadata.get("is_action_valid") is False:
                    return step
        if signal in {"stagnation_recovery", "repetition_guard"}:
            for step in reversed(action_steps):
                if step.metadata.get("stalled") or step.metadata.get("repeated_action"):
                    return step
        operation_markers = {
            "missing_clean_operation": ("take ", "put ", "move "),
            "missing_heat_operation": ("take ", "put ", "move "),
            "missing_cool_operation": ("take ", "put ", "move "),
            "missing_light_operation": ("take ", "use ", "examine "),
            "clean_operation_success": ("clean ",),
            "heat_operation_success": ("heat ",),
            "cool_operation_success": ("cool ",),
            "light_operation_success": ("use desklamp", "use "),
        }
        for marker in operation_markers.get(signal, ()):  # success markers first
            for step in action_steps:
                action = str(step.action or "").strip().lower()
                if action.startswith(marker):
                    return step
        if signal == "search_exhaustion":
            for step in reversed(action_steps):
                if str(step.action or "").strip().lower().startswith("go to "):
                    return step
        if signal in {"incomplete_multi_object", "multi_object_success"}:
            for step in reversed(action_steps):
                action = str(step.action or "").strip().lower()
                if action.startswith(("take ", "put ", "move ")):
                    return step
        return action_steps[-1]

    @staticmethod
    def _step_summary(step: AtomicStep) -> str:
        action = " ".join(str(step.action or "").split())
        observation = " ".join(str(step.observation or "").split())
        if observation:
            return f"{action} -> {observation[:160]}"
        return action or "empty action"

    @staticmethod
    def _first_repeated_action(action_steps: list[AtomicStep]) -> str | None:
        seen = set()
        for step in action_steps:
            action = str(step.action or "").strip().lower()
            if not action:
                continue
            if action in seen:
                return action
            seen.add(action)
        return None

    def _diagnose_success_trajectory(
        self,
        action_steps: list[AtomicStep],
        task: str,
        task_family: str,
        num_steps: int,
    ) -> str:
        # Prefer operation-grounded success signals when the win includes the
        # required transformation — these yield reusable family protocols.
        if self._requires(task_family, "clean") and self._has_operation(
            action_steps, "clean"
        ):
            return "clean_operation_success"
        if self._requires(task_family, "heat") and self._has_operation(
            action_steps, "heat"
        ):
            return "heat_operation_success"
        if self._requires(task_family, "cool") and self._has_operation(
            action_steps, "cool"
        ):
            return "cool_operation_success"
        if self._requires(task_family, "light") and self._has_operation(
            action_steps, "use desklamp"
        ):
            return "light_operation_success"
        required = self._required_instance_count(task, task_family)
        placements = self._count_placement_actions(action_steps)
        if required > 1 and placements >= required:
            return "multi_object_success"
        if num_steps <= max(3, self.config.high_cost_steps // 2):
            return "efficient_execution"
        return "task_completion"

    @staticmethod
    def _task_family(steps: list[AtomicStep]) -> str:
        terminal = steps[-1]
        return str(terminal.metadata.get("task_family", "other")).lower() or "other"

    @staticmethod
    def _trajectory_id(steps: list[AtomicStep]) -> str:
        terminal = steps[-1]
        return str(terminal.metadata.get("trajectory_id", terminal.node_id))

    @staticmethod
    def _required_instance_count(task: str, task_family: str) -> int:
        if "pick_two_obj_and_place" in task_family or re.search(
            r"\btwo\b",
            task,
        ):
            return 2
        return 1

    @classmethod
    def _is_multi_object_task(cls, task: str, task_family: str) -> bool:
        return cls._required_instance_count(task, task_family) > 1

    @staticmethod
    def _is_multi_stage_task(task: str, task_family: str) -> bool:
        if HeuristicSkillDistiller._is_multi_object_task(task, task_family):
            return True
        markers = (
            "pick_clean_then_place",
            "pick_heat_then_place",
            "pick_cool_then_place",
            "look_at_obj_in_light",
            "clean",
            "heat",
            "cool",
            "desklamp",
        )
        return any(marker in task_family or marker in task for marker in markers)

    @classmethod
    def _count_take_actions(cls, action_steps: list[AtomicStep]) -> int:
        return sum(
            1
            for step in action_steps
            if str(step.action or "").strip().lower().startswith("take ")
        )

    @classmethod
    def _count_placement_actions(cls, action_steps: list[AtomicStep]) -> int:
        return sum(
            1
            for step in action_steps
            if str(step.action or "").strip().lower().startswith(("put ", "move "))
        )

    @classmethod
    def _has_incomplete_multi_object(
        cls,
        action_steps: list[AtomicStep],
        task: str,
        task_family: str,
        num_steps: int,
    ) -> bool:
        if not cls._is_multi_object_task(task, task_family):
            return False
        required = cls._required_instance_count(task, task_family)
        placements = cls._count_placement_actions(action_steps)
        takes = cls._count_take_actions(action_steps)
        progress = max(placements, takes)
        if progress >= required or progress == 0:
            return False
        return True

    @classmethod
    def _default_high_cost_steps(cls) -> int:
        return 25

    @classmethod
    def _has_search_exhaustion(
        cls,
        action_steps: list[AtomicStep],
        num_steps: int,
    ) -> bool:
        if num_steps < 12:
            return False
        go_locations = {
            str(step.action or "").strip().lower()
            for step in action_steps
            if str(step.action or "").strip().lower().startswith("go to ")
        }
        if len(go_locations) < 4:
            return False
        if cls._count_take_actions(action_steps) > 0:
            return False
        return cls._count_placement_actions(action_steps) == 0

    @staticmethod
    def _has_material_stagnation(action_steps: list[AtomicStep]) -> bool:
        stalled_steps = [
            step for step in action_steps if step.metadata.get("stalled")
        ]
        if len(stalled_steps) >= 2:
            return True
        return bool(action_steps and action_steps[-1].metadata.get("stalled"))

    @classmethod
    def _has_incomplete_execution(
        cls,
        action_steps: list[AtomicStep],
        task: str,
        task_family: str,
        num_steps: int,
    ) -> bool:
        if not action_steps:
            return True
        if cls._count_placement_actions(action_steps) > 0:
            return True
        if cls._is_multi_stage_task(task, task_family) and num_steps >= 10:
            return True
        return False

    @staticmethod
    def _requires(task_family: str, operation: str) -> bool:
        family_requirements = {
            "clean": ("pick_clean_then_place",),
            "heat": ("pick_heat_then_place",),
            "cool": ("pick_cool_then_place",),
            "light": ("look_at_obj_in_light",),
        }
        return any(
            marker in task_family
            for marker in family_requirements.get(operation, ())
        )

    @staticmethod
    def _has_operation(steps: list[AtomicStep], operation: str) -> bool:
        for step in steps:
            action = str(step.action or "").strip().lower()
            observation = str(step.observation or "").strip().lower()
            if action.startswith(operation) or f"you {operation} " in observation:
                return True
        return False

    @staticmethod
    def _has_unproductive_repetition(steps: list[AtomicStep]) -> bool:
        for previous, current in zip(steps, steps[1:]):
            first = str(previous.action or "").strip().lower()
            second = str(current.action or "").strip().lower()
            if not first or first != second:
                continue
            if previous.metadata.get("stalled") or current.metadata.get("stalled"):
                return True
            first_obs = str(previous.observation or "").strip().lower()
            second_obs = str(current.observation or "").strip().lower()
            if first_obs and first_obs == second_obs:
                return True
        return False

    @staticmethod
    def _deduplicate_candidates(candidates: list[Skill]) -> list[Skill]:
        unique: dict[tuple[str, str, tuple[str, ...]], Skill] = {}
        for skill in candidates:
            key = (
                str(skill.metadata.get("source_signal", "")),
                str(skill.metadata.get("primary_task_family", "")),
                tuple(skill.evidence_ids),
            )
            unique[key] = skill
        ordered = sorted(
            unique.values(),
            key=lambda skill: (
                SIGNAL_PRIORITY.get(
                    str(skill.metadata.get("source_signal", "")),
                    len(ALL_SIGNALS),
                ),
                skill.skill_name,
            ),
        )
        return ordered

    @staticmethod
    def _build_skill(
        signal: str,
        evidence_ids: list[str],
        task_family: str,
        cluster: ExperienceCluster | None = None,
    ) -> Skill:
        observed_transitions = (
            [
                experience.key_transition
                for experience in cluster.experiences
                if experience.key_transition
            ]
            if cluster is not None
            else []
        )
        observed_actions: list[str] = []
        if cluster is not None:
            for experience in cluster.experiences:
                for action in experience.observed_actions:
                    if action and action not in observed_actions:
                        observed_actions.append(action)
            observed_actions = observed_actions[-8:]
        if not observed_actions:
            for transition in observed_transitions:
                action = transition.split(" -> ", 1)[0].strip()
                if action and action not in observed_actions:
                    observed_actions.append(action)
                if len(observed_actions) >= 4:
                    break
        trigger_states = (
            [
                experience.trigger_state
                for experience in cluster.experiences
                if experience.trigger_state
            ]
            if cluster is not None
            else []
        )
        metadata = {
            "distiller": "experience-cluster-v2",
            "source_signal": signal,
            "primary_task_family": task_family,
            "task_families": [task_family],
            "required_tools": [],
            "outcome_class": (
                cluster.experiences[0].outcome_class
                if cluster is not None and cluster.experiences
                else "success" if signal in SUCCESS_SIGNALS else "unknown"
            ),
            "candidate_stage": "experience_cluster",
            "protocol_source": "observed_cluster_actions",
        }
        # Preserve LLM-generated capability_key from the cluster's experiences.
        llm_capability_key = ""
        if cluster is not None and cluster.experiences:
            for experience in cluster.experiences:
                if experience.capability_key:
                    llm_capability_key = experience.capability_key
                    break
        if llm_capability_key:
            metadata["llm_capability_key"] = True
        if cluster is not None:
            metadata["experience_cluster"] = HeuristicSkillDistiller._cluster_metadata(
                cluster
            )
            metadata["trigger_states"] = metadata["experience_cluster"][
                "trigger_states"
            ]
            metadata["key_transitions"] = metadata["experience_cluster"][
                "key_transitions"
            ]
            metadata["anti_patterns"] = metadata["experience_cluster"][
                "anti_patterns"
            ]
            if trigger_states:
                metadata["anchor_state"] = trigger_states[0]
            if signal.startswith("environment_progress_"):
                metadata["confirmed_transitions"] = [
                    {
                        "task_id": experience.trajectory_id,
                        "action": experience.key_transition.split(
                            " -> ", 1
                        )[0],
                        "observation": (
                            experience.key_transition.split(" -> ", 1)[1]
                            if " -> " in experience.key_transition
                            else ""
                        ),
                    }
                    for experience in cluster.experiences
                ]
        label = signal.replace("_", " ")
        expected_effect = (
            observed_transitions[0].split(" -> ", 1)[1]
            if signal.startswith("environment_progress_")
            and observed_transitions
            and " -> " in observed_transitions[0]
            else (
                "Test whether the cited trajectory action pattern improves "
                "held-out task completion."
            )
        )
        return Skill(
            skill_name=f"Observed {label} pattern",
            description=(
                "Candidate behavior derived from environment transitions in "
                f"{task_family} trajectories."
            ),
            precondition=(
                trigger_states[0]
                if trigger_states
                else f"Task context matches {task_family}."
            ),
            action_protocol=observed_actions,
            applicable_atomic_ops=[AtomicOp.ACT],
            expected_effect=expected_effect,
            suggested_role="Executor",
            target_failure_types=[signal],
            evidence_ids=evidence_ids,
            support_count=len(evidence_ids),
            metadata=metadata,
            capability_key=llm_capability_key,
        )

        # Legacy templates below are intentionally unreachable. They are kept
        # temporarily for serialized-run compatibility while all newly
        # distilled candidates use trajectory-derived fields above.
        templates = {
            "missing_clean_operation": {
                "skill_name": "Clean the target object before placement",
                "description": "Execute the required sinkbasin cleaning operation before placing an object when the task requires a clean object.",
                "precondition": "The task explicitly requires a clean object, the target object is held, and no successful clean operation has been observed.",
                "action_protocol": [
                    "Identify and pick up the task's target object",
                    "Navigate to a sinkbasin while holding the target object",
                    "Execute 'clean <object> with <sinkbasin>' and require cleaning feedback",
                    "Only after cleaning succeeds, navigate to the target receptacle and place the object",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Prevent placement of an unclean object and complete clean-then-place tasks.",
                "target_failure_types": ["missing_required_operation", "cleaning_omission"],
                "required_tools": ["alfworld_action"],
            },
            "missing_heat_operation": {
                "skill_name": "Heat the target object before placement",
                "description": "Execute the required microwave heating operation before final placement when the task requires a hot object.",
                "precondition": "The task explicitly requires a hot object, the target object is held, and no successful heat operation has been observed.",
                "action_protocol": [
                    "Identify and pick up the task's target object",
                    "Navigate to a microwave while holding the target object",
                    "Execute 'heat <object> with <microwave>' and require heating feedback",
                    "Only after heating succeeds, place the object in the target receptacle",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Complete the state transformation required by heat-then-place tasks.",
                "target_failure_types": ["missing_required_operation", "heating_omission"],
                "required_tools": ["alfworld_action"],
            },
            "missing_cool_operation": {
                "skill_name": "Cool the target object before placement",
                "description": "Execute the required fridge cooling operation before final placement when the task requires a cool object.",
                "precondition": "The task explicitly requires a cool object, the target object is held, and no successful cool operation has been observed.",
                "action_protocol": [
                    "Identify and pick up the task's target object",
                    "Navigate to a fridge while holding the target object",
                    "Execute 'cool <object> with <fridge>' and require cooling feedback",
                    "Only after cooling succeeds, place the object in the target receptacle",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Complete the state transformation required by cool-then-place tasks.",
                "target_failure_types": ["missing_required_operation", "cooling_omission"],
                "required_tools": ["alfworld_action"],
            },
            "missing_light_operation": {
                "skill_name": "Use the desk lamp to inspect the target object",
                "description": "Bring the target object to the desk lamp and activate the lamp before completing an inspection task.",
                "precondition": "The task requires examining an object with a desk lamp and no successful lamp activation has been observed.",
                "action_protocol": [
                    "Locate and pick up the target object",
                    "Locate the desk containing the desk lamp",
                    "Bring the target object to that desk",
                    "Execute 'use desklamp' before the final object interaction",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Complete the required object-under-light interaction.",
                "target_failure_types": ["missing_required_operation", "lamp_activation_omission"],
                "required_tools": ["alfworld_action"],
            },
            "incomplete_multi_object": {
                "skill_name": "Track multi-instance placement progress",
                "description": "Track how many target instances are collected and placed before switching goals.",
                "precondition": "The task requires multiple instances and at least one required instance is still missing.",
                "action_protocol": [
                    "Parse the required instance count from the task goal",
                    "Record how many target instances have been picked up and placed",
                    "Continue searching or placing until the required count is met",
                    "Do not treat one placed instance as task completion when more are required",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Reduce partial-completion failures on multi-instance tasks.",
                "target_failure_types": ["incomplete_task", "partial_placement"],
            },
            "search_exhaustion": {
                "skill_name": "Expand search across unvisited locations",
                "description": "Search systematically across unexplored receptacles before revisiting already checked locations.",
                "precondition": "The target object has not been acquired and recent actions revisit known empty locations.",
                "action_protocol": [
                    "List locations already inspected without finding the target",
                    "Prioritize unvisited cabinets, drawers, shelves, and surfaces",
                    "Open closed containers before concluding an area is empty",
                    "Avoid looping back to locations already shown to lack the target",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.OBSERVE],
                "expected_effect": "Reduce step-limit failures caused by repeated search of empty areas.",
                "target_failure_types": ["search_failure", "high_cost"],
            },
            "failure_checkpoint": {
                "skill_name": "Verify task progress before termination",
                "description": "Check completed and remaining subgoals before deciding that the task is complete.",
                "precondition": "The agent is near termination or has completed a major subgoal.",
                "action_protocol": [
                    "Restate the task goal as explicit subgoals",
                    "Compare observed state with every subgoal",
                    "Continue acting when any required subgoal lacks evidence",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.VERIFY, AtomicOp.ACT],
                "expected_effect": "Reduce premature termination and incomplete task execution.",
                "suggested_role": "Executor",
                "target_failure_types": ["incomplete_task", "premature_termination"],
            },
            "progress_checkpoint": {
                "skill_name": "Track subgoal progress in long-horizon execution",
                "description": "Maintain completed, current, and remaining subgoals during long trajectories.",
                "precondition": "The task has multiple dependent operations or execution exceeds a step budget.",
                "action_protocol": [
                    "Decompose the task into ordered subgoals",
                    "Record evidence when a subgoal completes",
                    "Select the next action from the earliest unmet subgoal",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Reduce unnecessary steps and forgotten dependencies.",
                "suggested_role": "Executor",
                "target_failure_types": ["high_cost", "lost_progress"],
            },
            "admissible_action_guard": {
                "skill_name": "Validate actions against environment affordances",
                "description": "Check that a proposed action is syntactically and contextually admissible before execution.",
                "precondition": "An environment action is about to be issued.",
                "action_protocol": [
                    "Inspect available objects and admissible action patterns",
                    "Validate object and receptacle references",
                    "Repair the action before sending it when validation fails",
                ],
                "applicable_atomic_ops": [AtomicOp.ACT],
                "expected_effect": "Reduce invalid actions and wasted environment turns.",
                "suggested_role": "Executor",
                "target_failure_types": ["invalid_action"],
            },
            "stagnation_recovery": {
                "skill_name": "Recover from non-progressing actions",
                "description": "Detect observations that show no state change and switch to a recovery action.",
                "precondition": "The latest action produced no observable progress.",
                "action_protocol": [
                    "Compare the current observation with the previous state",
                    "Identify the failed action assumption",
                    "Choose a different admissible action or request a new plan",
                ],
                "applicable_atomic_ops": [AtomicOp.OBSERVE, AtomicOp.PLAN, AtomicOp.ACT],
                "expected_effect": "Reduce stalls and repeated ineffective actions.",
                "suggested_role": "Executor",
                "target_failure_types": ["stagnation", "no_state_change"],
            },
            "repetition_guard": {
                "skill_name": "Prevent unproductive action repetition",
                "description": "Detect repeated actions without new evidence and force an alternative decision.",
                "precondition": "An action has already been attempted in a materially equivalent state.",
                "action_protocol": [
                    "Retrieve recent action-observation pairs",
                    "Check whether the proposed action previously changed state",
                    "Block ineffective repetition and choose an alternative",
                ],
                "applicable_atomic_ops": [AtomicOp.SELECT, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Lower step cost caused by loops.",
                "suggested_role": "Executor",
                "target_failure_types": ["action_loop", "high_cost"],
            },
            "multi_object_success": {
                "skill_name": "Complete all required object instances",
                "description": "Reuse the successful multi-instance placement pattern observed in completed trajectories.",
                "precondition": "The task requires multiple instances of the same object.",
                "action_protocol": [
                    "Track how many instances are already placed",
                    "Collect the next missing instance before switching goals",
                    "Place each instance in the target receptacle",
                    "Verify the required count before stopping",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Maintain reliable completion on multi-instance tasks.",
                "suggested_role": "Executor",
                "target_failure_types": ["multi_instance_success"],
            },
            "clean_operation_success": {
                "skill_name": "Reuse verified clean-then-place routine",
                "description": "Replay the successful clean-then-place sequence observed in winning trajectories.",
                "precondition": "The task requires a clean object and matches the pick_clean family.",
                "action_protocol": [
                    "Identify and pick up the task's target object",
                    "Navigate to a sinkbasin while holding the target object",
                    "Execute 'clean <object> with <sinkbasin>' and require cleaning feedback",
                    "Only after cleaning succeeds, navigate to the target receptacle and place the object",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Reproduce successful clean-then-place completions.",
                "suggested_role": "Executor",
                "target_failure_types": ["cleaning_success"],
                "required_tools": ["alfworld_action"],
            },
            "heat_operation_success": {
                "skill_name": "Reuse verified heat-then-place routine",
                "description": "Replay the successful heat-then-place sequence observed in winning trajectories.",
                "precondition": "The task requires a hot object and matches the pick_heat family.",
                "action_protocol": [
                    "Identify and pick up the task's target object",
                    "Navigate to a microwave while holding the target object",
                    "Execute 'heat <object> with <microwave>' and require heating feedback",
                    "Only after heating succeeds, place the object in the target receptacle",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Reproduce successful heat-then-place completions.",
                "suggested_role": "Executor",
                "target_failure_types": ["heating_success"],
                "required_tools": ["alfworld_action"],
            },
            "cool_operation_success": {
                "skill_name": "Reuse verified cool-then-place routine",
                "description": "Replay the successful cool-then-place sequence observed in winning trajectories.",
                "precondition": "The task requires a cool object and matches the pick_cool family.",
                "action_protocol": [
                    "Identify and pick up the task's target object",
                    "Navigate to a fridge while holding the target object",
                    "Execute 'cool <object> with <fridge>' and require cooling feedback",
                    "Only after cooling succeeds, place the object in the target receptacle",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Reproduce successful cool-then-place completions.",
                "suggested_role": "Executor",
                "target_failure_types": ["cooling_success"],
                "required_tools": ["alfworld_action"],
            },
            "light_operation_success": {
                "skill_name": "Reuse verified look-at-in-light routine",
                "description": "Replay the successful desk-lamp inspection sequence from winning trajectories.",
                "precondition": "The task requires examining an object with a desk lamp.",
                "action_protocol": [
                    "Locate and pick up the target object",
                    "Locate the desk containing the desk lamp",
                    "Bring the target object to that desk",
                    "Execute 'use desklamp' before the final object interaction",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Reproduce successful look-at-in-light completions.",
                "suggested_role": "Executor",
                "target_failure_types": ["lamp_success"],
                "required_tools": ["alfworld_action"],
            },
            "efficient_execution": {
                "skill_name": "Reuse efficient search-to-completion path",
                "description": "Follow the short successful action sequence observed on similar tasks.",
                "precondition": "A similar task family has recently been solved within a small step budget.",
                "action_protocol": [
                    "Recall the shortest successful route from cited trajectories",
                    "Prioritize the same object and receptacle interactions",
                    "Avoid revisiting locations that did not appear in successful traces",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.OBSERVE],
                "expected_effect": "Reduce unnecessary exploration on familiar task families.",
                "suggested_role": "Executor",
                "target_failure_types": ["efficient_success"],
            },
            "task_completion": {
                "skill_name": "Reuse verified completion routine",
                "description": "Apply the successful completion routine extracted from prior winning trajectories.",
                "precondition": "The current task family matches a recently completed trajectory.",
                "action_protocol": [
                    "Identify the subgoals satisfied in the cited successful trace",
                    "Replay the same ordering of object interactions when context matches",
                    "Verify the final goal state before termination",
                ],
                "applicable_atomic_ops": [AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
                "expected_effect": "Improve completion rate on familiar tasks.",
                "suggested_role": "Executor",
                "target_failure_types": ["successful_completion"],
            },
        }
        payload = dict(templates[signal])
        required_tools = payload.pop("required_tools", [])
        metadata = {
            "distiller": "experience-cluster-v1",
            "source_signal": signal,
            "primary_task_family": task_family,
            "task_families": [task_family],
            "required_tools": required_tools,
            "outcome_class": (
                "success"
                if signal in SUCCESS_SIGNALS
                else "failure"
            ),
            "candidate_stage": "experience_cluster",
        }
        if cluster is not None:
            metadata["experience_cluster"] = HeuristicSkillDistiller._cluster_metadata(
                cluster
            )
            metadata["trigger_states"] = metadata["experience_cluster"][
                "trigger_states"
            ]
            metadata["key_transitions"] = metadata["experience_cluster"][
                "key_transitions"
            ]
            metadata["anti_patterns"] = metadata["experience_cluster"][
                "anti_patterns"
            ]
        return Skill(
            **payload,
            evidence_ids=evidence_ids,
            support_count=len(evidence_ids),
            metadata=metadata,
        )


class TrajectoryGroundedSkillDistiller:
    """Backward-compatible wrapper around the enriched distillation pipeline."""

    def __init__(
        self,
        backend: DistillationBackend,
        seed_distiller: HeuristicSkillDistiller,
        max_evidence_trajectories: int = 3,
        max_steps_per_trajectory: int = 15,
        min_wins_for_protocol: int = 3,
    ):
        from sage_mas.enriched_skill_distiller import EnrichedSkillDistiller

        self._enriched = EnrichedSkillDistiller(
            backend=backend,
            seed_distiller=seed_distiller,
            max_evidence_trajectories=max_evidence_trajectories,
            max_steps_per_trajectory=max_steps_per_trajectory,
            use_llm_rewrite=True,
            min_wins_for_protocol=min_wins_for_protocol,
        )

    def distill(self, trajectories: list[list[AtomicStep]]) -> list[Skill]:
        return self._enriched.distill(trajectories)
