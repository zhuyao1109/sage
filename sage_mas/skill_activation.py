"""Task-aware skill precondition matching for online probes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sage_mas.schemas import Skill


@dataclass(slots=True)
class ApplicabilityResult:
    applicable: bool
    reason: str
    matched_step: int | None = None


class SkillPreconditionMatcher:
    """Match operational skill preconditions against an execution history."""

    OPERATION_SIGNALS = {
        "missing_clean_operation": "clean",
        "missing_heat_operation": "heat",
        "missing_cool_operation": "cool",
        "clean_operation_success": "clean",
        "heat_operation_success": "heat",
        "cool_operation_success": "cool",
    }

    OPERATION_TASK_FAMILIES = {
        "clean": "pick_clean_then_place_in_recep",
        "heat": "pick_heat_then_place_in_recep",
        "cool": "pick_cool_then_place_in_recep",
    }

    LIGHT_TASK_FAMILY = "look_at_obj_in_light"

    SIGNAL_TO_TASK_FAMILY = {
        "missing_clean_operation": "pick_clean_then_place_in_recep",
        "missing_heat_operation": "pick_heat_then_place_in_recep",
        "missing_cool_operation": "pick_cool_then_place_in_recep",
        "missing_light_operation": "look_at_obj_in_light",
        "clean_operation_success": "pick_clean_then_place_in_recep",
        "heat_operation_success": "pick_heat_then_place_in_recep",
        "cool_operation_success": "pick_cool_then_place_in_recep",
        "light_operation_success": "look_at_obj_in_light",
        "multi_object_success": "pick_two_obj_and_place",
    }

    # Winning-trajectory / merged success protocols: activate for the whole
    # episode once the family gate passes. Do not drop after the transform
    # step succeeds — place / completion still need the contract.
    FULL_EPISODE_SUCCESS_SIGNALS = {
        "efficient_execution",
        "task_completion",
        "clean_operation_success",
        "heat_operation_success",
        "cool_operation_success",
        "light_operation_success",
        "multi_object_success",
    }

    def evaluate(
        self,
        skill: Skill,
        task: str,
        gamefile: str,
        steps: list[dict[str, Any]],
        current: bool = False,
    ) -> ApplicabilityResult:
        family_gate = self._family_gate(skill, task, gamefile)
        if family_gate is not None and not family_gate.applicable:
            return family_gate

        signal = str(skill.metadata.get("source_signal", ""))
        if signal in self.OPERATION_SIGNALS:
            operation = self.OPERATION_SIGNALS[signal]
            if not self._task_requires_operation(task, gamefile, operation):
                return ApplicabilityResult(
                    False,
                    f"Task is not a {operation} task family.",
                )
            # Merged / success protocols stay active for the full episode so
            # post-transform place steps are still guided.
            if current and self._is_full_episode_success_protocol(skill, signal):
                return ApplicabilityResult(
                    True,
                    (
                        f"Full-episode `{operation}` protocol remains active "
                        "until the task goal completes."
                    ),
                    matched_step=1 if steps else None,
                )
            target = self._target_object(task, gamefile, operation)
            if not target:
                return ApplicabilityResult(
                    False,
                    f"Could not identify target object for {operation} task.",
                )
            first_pickup_step = None
            latest_pickup_step = None
            operation_after_latest_pickup = False
            for index, step in enumerate(steps, start=1):
                action = self._action_text(step)
                observation = str(step.get("observation", "")).lower()
                if (
                    action.startswith("take ")
                    and self._contains_object(action, target)
                ):
                    first_pickup_step = first_pickup_step or index
                    latest_pickup_step = index
                    operation_after_latest_pickup = False
                if (
                    action.startswith(operation)
                    or f"you {operation} " in observation
                ) and latest_pickup_step is not None:
                    operation_after_latest_pickup = True
            matched_step = (
                latest_pickup_step if current else first_pickup_step
            )
            if matched_step is not None and (
                not current or not operation_after_latest_pickup
            ):
                return ApplicabilityResult(
                    True,
                    f"Target {target!r} was picked up before {operation}.",
                    matched_step=matched_step,
                )
            if current and operation_after_latest_pickup:
                return ApplicabilityResult(
                    False,
                    f"Target {target!r} has already completed {operation}.",
                )
            # Trajectory-derived success protocols are full-episode contracts:
            # activate from step 0 so search/pickup also follow the protocol.
            if current and signal.endswith("_success"):
                return ApplicabilityResult(
                    True,
                    f"Full-episode {operation} protocol is active before pickup.",
                    matched_step=1 if steps else None,
                )
            return ApplicabilityResult(
                False,
                f"Target {target!r} was never picked up before {operation}.",
            )

        if signal in {"missing_light_operation", "light_operation_success"}:
            if not self._task_requires_light(task, gamefile):
                return ApplicabilityResult(
                    False,
                    "Task is not a look-at-object-in-light family task.",
                )
            if current and self._is_full_episode_success_protocol(skill, signal):
                return ApplicabilityResult(
                    True,
                    (
                        "Full-episode light protocol remains active until the "
                        "inspection goal completes."
                    ),
                    matched_step=1 if steps else None,
                )
            pickup_step = None
            lamp_used = False
            for index, step in enumerate(steps, start=1):
                action = self._action_text(step)
                if action.startswith("take "):
                    pickup_step = pickup_step or index
                if action.startswith("use desklamp"):
                    lamp_used = True
            if pickup_step is not None and (not current or not lamp_used):
                return ApplicabilityResult(
                    True,
                    "A target object was picked up before lamp activation.",
                    matched_step=pickup_step,
                )
            if current and lamp_used:
                return ApplicabilityResult(
                    False,
                    "The desk lamp has already been activated.",
                )
            return ApplicabilityResult(False, "No target object was picked up.")

        if signal == "repetition_guard":
            actions = [self._action_text(step) for step in steps]
            if current:
                if actions and actions[-1] in actions[:-1]:
                    return ApplicabilityResult(
                        True,
                        "The latest action repeats an earlier action.",
                        matched_step=len(actions),
                    )
                return ApplicabilityResult(
                    False,
                    "The latest action is not a repetition.",
                )
            seen_actions = set()
            for index, action in enumerate(actions, start=1):
                if action and action in seen_actions:
                    return ApplicabilityResult(
                        True,
                        "An action was repeated in the trajectory.",
                        matched_step=index,
                    )
                seen_actions.add(action)
            return ApplicabilityResult(False, "No repeated action was observed.")

        if signal == "stagnation_recovery":
            if current:
                if steps and "nothing happens" in str(
                    steps[-1].get("observation", "")
                ).strip().lower():
                    return ApplicabilityResult(
                        True,
                        "The latest action produced no state change.",
                        matched_step=len(steps),
                    )
                return ApplicabilityResult(
                    False,
                    "The latest observation is not stagnating.",
                )
            for index, step in enumerate(steps, start=1):
                if "nothing happens" in str(
                    step.get("observation", "")
                ).strip().lower():
                    return ApplicabilityResult(
                        True,
                        "The environment reported no state change.",
                        matched_step=index,
                    )
            return ApplicabilityResult(False, "No stagnating observation was found.")

        if signal == "failure_checkpoint":
            for index, step in enumerate(steps, start=1):
                action = self._action_text(step)
                if action.startswith(("put ", "clean ", "heat ", "cool ", "use ")):
                    return ApplicabilityResult(
                        True,
                        "A major execution subgoal was attempted.",
                        matched_step=index,
                    )
            if len(steps) >= 10:
                return ApplicabilityResult(
                    True,
                    "The trajectory is near a long-horizon checkpoint.",
                    matched_step=10,
                )
            return ApplicabilityResult(False, "No termination checkpoint was reached.")

        if signal == "progress_checkpoint":
            multi_stage_markers = (
                "clean",
                "heat",
                "cool",
                "desklamp",
                "two object",
            )
            if any(marker in task.lower() for marker in multi_stage_markers):
                return ApplicabilityResult(
                    True,
                    "The task contains multiple dependent operations.",
                    matched_step=1 if steps else None,
                )
            if len(steps) >= 10:
                return ApplicabilityResult(
                    True,
                    "Execution exceeded the progress-checkpoint budget.",
                    matched_step=10,
                )
            return ApplicabilityResult(False, "No progress checkpoint was reached.")

        if signal == "admissible_action_guard":
            return ApplicabilityResult(
                True,
                "An environment action is about to be issued.",
                matched_step=1 if steps else None,
            )

        if self._is_full_episode_success_protocol(skill, signal):
            if not self._declared_task_families(skill, signal):
                return ApplicabilityResult(
                    False,
                    "Full-episode success protocol has no task-family scope.",
                )
            return ApplicabilityResult(
                True,
                (
                    f"Full-episode `{signal or 'aligned'}` protocol is active "
                    "for this task family."
                ),
                matched_step=1 if steps else None,
            )

        if signal in {"incomplete_multi_object", "multi_object_success"}:
            required = 2 if (
                "pick_two_obj_and_place" in gamefile
                or re.search(r"\btwo\b", task.lower())
            ) else 1
            takes = sum(
                1
                for step in steps
                if self._action_text(step).startswith("take ")
            )
            placements = sum(
                1
                for step in steps
                if self._action_text(step).startswith(("put ", "move "))
            )
            if max(takes, placements) < required:
                return ApplicabilityResult(
                    True,
                    "The task still lacks the required number of instances.",
                    matched_step=len(steps) if steps else None,
                )
            return ApplicabilityResult(
                False,
                "The required instance count appears satisfied.",
            )

        if signal == "search_exhaustion":
            go_locations = {
                self._action_text(step)
                for step in steps
                if self._action_text(step).startswith("go to ")
            }
            takes = sum(
                1
                for step in steps
                if self._action_text(step).startswith("take ")
            )
            if len(go_locations) >= 3 and takes == 0:
                return ApplicabilityResult(
                    True,
                    "The agent has searched multiple locations without acquiring the target.",
                    matched_step=len(steps) if steps else None,
                )
            return ApplicabilityResult(
                False,
                "Search exhaustion conditions are not met.",
            )

        if signal == "environment_progress":
            anchor = str(skill.metadata.get("anchor_state", "") or "")
            if not anchor or not steps:
                return ApplicabilityResult(
                    False,
                    "The learned anchor state has not been observed.",
                )
            anchor_tokens = self._semantic_tokens(anchor)
            current_tokens = self._semantic_tokens(
                str(steps[-1].get("observation", ""))
            )
            overlap = anchor_tokens & current_tokens
            required = max(2, min(5, len(anchor_tokens) // 4))
            if len(overlap) >= required:
                return ApplicabilityResult(
                    True,
                    "The current observation matches the learned anchor state.",
                    matched_step=len(steps),
                )
            return ApplicabilityResult(
                False,
                "The current observation does not match the learned anchor state.",
            )

        return ApplicabilityResult(
            False,
            "No deterministic precondition matcher is available for this skill.",
        )

    def active_skills(
        self,
        skills: list[Skill],
        task: str,
        gamefile: str,
        steps: list[dict[str, Any]],
    ) -> tuple[list[Skill], dict[str, ApplicabilityResult]]:
        active = []
        results = {}
        for skill in skills:
            result = self.evaluate(
                skill,
                task,
                gamefile,
                steps,
                current=True,
            )
            results[skill.skill_name] = result
            if result.applicable:
                active.append(skill)
        return active, results

    def episode_eligible(
        self,
        skill: Skill,
        task: str,
        gamefile: str,
        *,
        task_family: str | None = None,
    ) -> ApplicabilityResult:
        """Layer-1 routing: could this skill apply to the episode at all?

        Unlike ``evaluate`` (step-active), this ignores empty history and only
        checks family / operation grounding so Executor can shortlist specialists
        before the first environment action.
        """
        family_gate = self._family_gate(skill, task, gamefile)
        if family_gate is not None and not family_gate.applicable:
            return family_gate

        signal = str(skill.metadata.get("source_signal", "")).strip()
        current = (task_family or self._task_family(task, gamefile) or "").strip()

        if signal in self.OPERATION_SIGNALS:
            operation = self.OPERATION_SIGNALS[signal]
            if self._task_requires_operation(task, gamefile, operation):
                return ApplicabilityResult(
                    True,
                    f"Task may need {operation} stage.",
                )
            return ApplicabilityResult(
                False,
                f"Task is not a {operation} task family.",
            )

        if signal in {"missing_light_operation", "light_operation_success"}:
            if self._task_requires_light(task, gamefile):
                return ApplicabilityResult(
                    True,
                    "Task may need light-inspection stage.",
                )
            return ApplicabilityResult(False, "Task is not a light family.")

        if signal in {"incomplete_multi_object", "multi_object_success"}:
            if (
                "pick_two_obj_and_place" in gamefile
                or re.search(r"\btwo\b", task.lower())
            ):
                return ApplicabilityResult(
                    True,
                    "Task may need multi-object tracking.",
                )
            return ApplicabilityResult(False, "Task is not multi-object.")

        declared_families = {
            str(item).strip()
            for item in skill.applicable_task_families
            if str(item).strip() and str(item).strip() != "other"
        }
        declared = str(skill.metadata.get("primary_task_family", "") or "").strip()
        if declared and declared != "other":
            declared_families.add(declared)
        if not declared or declared == "other":
            declared = self.SIGNAL_TO_TASK_FAMILY.get(signal, "")
        if declared and declared != "other":
            declared_families.add(declared)
        if current and current in declared_families:
            return ApplicabilityResult(
                True,
                f"Skill family matches {current}.",
            )
        for item in skill.metadata.get("task_families", []) or []:
            if current and str(item).strip().lower() == current.lower():
                return ApplicabilityResult(
                    True,
                    f"Listed in task_families for {current}.",
                )

        if signal in {
            "search_exhaustion",
            "progress_checkpoint",
            "admissible_action_guard",
            "invalid_action_loop",
            "high_cost_trajectory",
            *self.FULL_EPISODE_SUCCESS_SIGNALS,
        } or self._is_full_episode_success_protocol(skill, signal):
            # Generic procedural skills: eligible when family gate did not reject.
            return ApplicabilityResult(
                True,
                f"Generic skill `{signal}` may apply this episode.",
            )

        if family_gate is not None and family_gate.applicable:
            return ApplicabilityResult(
                True,
                family_gate.reason or "Family gate passed.",
            )
        return ApplicabilityResult(
            False,
            "No episode-level family or signal grounding.",
        )

    @classmethod
    def _declared_task_families(
        cls,
        skill: Skill,
        signal: str,
    ) -> set[str]:
        declared_families = {
            str(item).strip()
            for item in skill.applicable_task_families
            if str(item).strip() and str(item).strip() != "other"
        }
        declared = str(skill.metadata.get("primary_task_family", "") or "").strip()
        if declared and declared != "other":
            declared_families.add(declared)
        mapped = cls.SIGNAL_TO_TASK_FAMILY.get(signal, "")
        if mapped:
            declared_families.add(mapped)
        for item in skill.metadata.get("task_families", []) or []:
            value = str(item).strip()
            if value and value != "other":
                declared_families.add(value)
        return declared_families

    @classmethod
    def _is_full_episode_success_protocol(
        cls,
        skill: Skill,
        signal: str,
    ) -> bool:
        if signal in cls.FULL_EPISODE_SUCCESS_SIGNALS:
            return True
        if signal.endswith("_operation_success"):
            return True
        protocol_source = str(skill.metadata.get("protocol_source", "") or "")
        if protocol_source.startswith("aligned_successful_trajectory_stages"):
            return True
        if protocol_source.startswith("merged_success_detour_clean"):
            return True
        capability = str(
            skill.capability_key or skill.metadata.get("capability_key", "") or ""
        )
        return capability.startswith("execution.")

    def _family_gate(
        self,
        skill: Skill,
        task: str,
        gamefile: str,
    ) -> ApplicabilityResult | None:
        """Reject skills whose declared family cannot apply to this task."""
        signal = str(skill.metadata.get("source_signal", "")).strip()
        declared_families = self._declared_task_families(skill, signal)
        if not declared_families:
            return None

        current = self._task_family(task, gamefile)
        if current and current not in declared_families:
            return ApplicabilityResult(
                False,
                f"Skill family scope {sorted(declared_families)!r} "
                f"does not match task family {current!r}.",
            )
        if current not in declared_families and not any(
            family in gamefile for family in declared_families
        ):
            return ApplicabilityResult(
                False,
                f"Skill families {sorted(declared_families)!r} are not "
                "supported by this gamefile.",
            )
        return None

    @classmethod
    def _task_requires_operation(
        cls,
        task: str,
        gamefile: str,
        operation: str,
    ) -> bool:
        family = cls.OPERATION_TASK_FAMILIES[operation]
        if family in gamefile:
            return True
        task_lower = task.lower()
        markers = {
            "clean": (r"\bclean\b", r"\bclean\b"),
            "heat": (r"\bheat\b", r"\bhot\b"),
            "cool": (r"\bcool\b",),
        }
        return any(re.search(pattern, task_lower) for pattern in markers[operation])

    @classmethod
    def _task_requires_light(cls, task: str, gamefile: str) -> bool:
        if cls.LIGHT_TASK_FAMILY in gamefile:
            return True
        task_lower = task.lower()
        return bool(
            re.search(r"\bdesklamp\b|\blamp\b", task_lower)
            or re.search(r"\bexamine\b.+\bwith\b", task_lower)
            or re.search(r"\blook at\b", task_lower)
        )

    @classmethod
    def _task_family(cls, task: str, gamefile: str) -> str | None:
        families = (
            "pick_two_obj_and_place",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
            "look_at_obj_in_light",
            "pick_and_place",
        )
        for family in families:
            if family in gamefile:
                return family
        task_lower = task.lower()
        if cls._task_requires_light(task, gamefile):
            return cls.LIGHT_TASK_FAMILY
        for operation, family in cls.OPERATION_TASK_FAMILIES.items():
            if cls._task_requires_operation(task, gamefile, operation):
                return family
        if re.search(r"\btwo\b", task_lower):
            return "pick_two_obj_and_place"
        return None

    @classmethod
    def _target_object(cls, task: str, gamefile: str, operation: str) -> str | None:
        task_lower = task.lower()
        patterns = {
            "clean": [
                r"clean some ([a-z]+)",
                r"(?:a|the) clean ([a-z]+)",
            ],
            "heat": [
                r"heat some ([a-z]+)",
                r"(?:a|the) hot ([a-z]+)",
            ],
            "cool": [
                r"cool some ([a-z]+)",
                r"(?:a|the) cool ([a-z]+)",
            ],
        }
        for pattern in patterns.get(operation, []):
            match = re.search(pattern, task_lower)
            if match:
                return match.group(1)

        # Only fall back to gamefile object id when the path is the matching family.
        family = cls.OPERATION_TASK_FAMILIES[operation]
        if family not in gamefile:
            return None
        path = Path(gamefile)
        parent_name = path.parents[1].name if len(path.parents) > 1 else path.stem
        parts = parent_name.split("-")
        if len(parts) >= 2:
            return parts[1].lower()
        return None

    @staticmethod
    def _contains_object(action: str, target: str) -> bool:
        return bool(re.search(rf"\b{re.escape(target)}\s+\d+\b", action)) or target in action

    @staticmethod
    def _action_text(step: dict[str, Any]) -> str:
        action = str(step.get("action", "")).strip().lower()
        match = re.search(r"<action>(.*?)</action>", action, re.DOTALL)
        return match.group(1).strip() if match else action

    @staticmethod
    def _semantic_tokens(text: str) -> set[str]:
        stop = {
            "the",
            "and",
            "you",
            "your",
            "are",
            "with",
            "from",
            "this",
            "that",
            "there",
            "here",
            "can",
            "see",
            "some",
            "object",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) >= 3 and token not in stop
        }
