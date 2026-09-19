"""Map raw environment trajectories to replayable AtomicOp nodes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from sage_mas.schemas import AtomicOp, AtomicStep
from sage_mas.serialization import read_jsonl


def build_trajectory_adapter(name: str | None = None):
    """Factory used by the evolution pipeline."""
    normalized = str(name or "alfworld").strip().lower()
    if normalized in {"webshop", "web_shop", "ws"}:
        return WebShopTrajectoryAdapter()
    return AlfWorldTrajectoryAdapter()


class AlfWorldTrajectoryAdapter:
    """A deterministic first-pass adapter for ALFWorld text trajectories."""

    def load(self, path: str | Path) -> list[dict[str, Any]]:
        return read_jsonl(path)

    def adapt_many(self, trajectories: list[dict[str, Any]]) -> list[list[AtomicStep]]:
        return [self.adapt(trajectory, index) for index, trajectory in enumerate(trajectories)]

    def adapt(self, trajectory: dict[str, Any], trajectory_index: int = 0) -> list[AtomicStep]:
        trajectory_id = str(
            trajectory.get("trajectory_id")
            or trajectory.get("gamefile")
            or f"trajectory-{trajectory_index}"
        )
        raw_steps = trajectory.get("steps", [])
        action_counts = Counter(str(step.get("action", "")).strip().lower() for step in raw_steps)
        atomic_steps: list[AtomicStep] = []

        for index, raw_step in enumerate(raw_steps, start=1):
            action = str(raw_step.get("action", "")).strip()
            observation = str(raw_step.get("observation", "")).strip()
            normalized_action = action.lower()
            base_metadata = {
                "trajectory_id": trajectory_id,
                "task": trajectory.get("task", ""),
                "task_family": trajectory.get("task_family") or self._task_family(trajectory),
                "won": bool(trajectory.get("won", False)),
                "step_index": index,
                "is_action_valid": raw_step.get("is_action_valid"),
                "reward": raw_step.get("reward"),
                "observation_before": raw_step.get("observation_before"),
                "goal_progress_before": raw_step.get("goal_progress_before"),
                "goal_progress_after": raw_step.get("goal_progress_after"),
                "goal_progress_delta": raw_step.get("goal_progress_delta"),
                "repeated_action": bool(normalized_action and action_counts[normalized_action] > 1),
                "stalled": observation.lower() == "nothing happens.",
            }
            agent_messages = raw_step.get("agent_messages", [])
            for message_index, message in enumerate(agent_messages, start=1):
                content = str(message.get("content", "")).strip()
                if content == action:
                    continue
                atomic_steps.append(
                    AtomicStep(
                        node_id=f"{trajectory_index}:step_{index}:message_{message_index}",
                        atomic_op=AtomicOp.COMMUNICATE,
                        agent=str(message.get("agent", "Advisor")),
                        observation=None,
                        action=content,
                        metadata={
                            "trajectory_id": trajectory_id,
                            "task": trajectory.get("task", ""),
                            "task_family": base_metadata["task_family"],
                            "won": base_metadata["won"],
                            "step_index": index,
                            "token_cost": message.get("token_cost", 0),
                        },
                    )
                )
            atomic_steps.append(
                AtomicStep(
                    node_id=f"{trajectory_index}:step_{index}",
                    atomic_op=self._infer_atomic_op(normalized_action),
                    agent=str(raw_step.get("agent", "Executor")),
                    observation=observation,
                    action=action,
                    metadata=base_metadata,
                )
            )

        atomic_steps.append(
            AtomicStep(
                node_id=f"{trajectory_index}:terminate",
                atomic_op=AtomicOp.TERMINATE,
                agent="Executor",
                observation=None,
                action=None,
                metadata={
                    "trajectory_id": trajectory_id,
                    "task": trajectory.get("task", ""),
                    "task_family": trajectory.get("task_family") or self._task_family(trajectory),
                    "won": bool(trajectory.get("won", False)),
                    "num_steps": int(trajectory.get("num_steps", len(raw_steps))),
                },
            )
        )
        return atomic_steps

    @staticmethod
    def _infer_atomic_op(action: str) -> AtomicOp:
        if action.startswith(("examine ", "look")):
            return AtomicOp.OBSERVE
        if action.startswith(("take ", "put ", "move ", "go ", "open ", "close ", "use ", "heat ", "cool ", "clean ")):
            return AtomicOp.ACT
        if action.startswith(("choose ", "select ")):
            return AtomicOp.SELECT
        if action.startswith(("verify ", "check ")):
            return AtomicOp.VERIFY
        return AtomicOp.ACT

    @staticmethod
    def _task_family(trajectory: dict[str, Any]) -> str:
        gamefile = str(trajectory.get("gamefile", ""))
        for family in (
            "pick_two_obj_and_place",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
            "pick_and_place",
            "look_at_obj_in_light",
        ):
            if family in gamefile:
                return family
        return "other"


class WebShopTrajectoryAdapter(AlfWorldTrajectoryAdapter):
    """Adapter for WebShop ``search[...]`` / ``click[...]`` trajectories."""

    def adapt(
        self, trajectory: dict[str, Any], trajectory_index: int = 0
    ) -> list[AtomicStep]:
        # Reuse ALFWorld adapt, but override stalled/family inference via hooks.
        steps = super().adapt(trajectory, trajectory_index)
        for step in steps:
            metadata = dict(step.metadata or {})
            if "task_family" in metadata:
                metadata["task_family"] = (
                    trajectory.get("task_family")
                    or self._task_family(trajectory)
                )
            observation = str(step.observation or "")
            observation_before = str(metadata.get("observation_before") or "")
            invalid = metadata.get("is_action_valid") is False
            metadata["stalled"] = bool(
                invalid
                or (
                    observation
                    and observation_before
                    and observation == observation_before
                )
            )
            step.metadata = metadata
        return steps

    @staticmethod
    def _infer_atomic_op(action: str) -> AtomicOp:
        text = str(action or "").strip().lower()
        # Strip optional <action>...</action> wrappers.
        if "<action>" in text and "</action>" in text:
            start = text.find("<action>") + len("<action>")
            end = text.find("</action>")
            text = text[start:end].strip()
        if text.startswith("search"):
            return AtomicOp.ACT
        if text.startswith("click"):
            # Product / option selection is the closest Select analogue.
            return AtomicOp.SELECT
        if text.startswith(("verify ", "check ")):
            return AtomicOp.VERIFY
        return AtomicOp.ACT

    @staticmethod
    def _task_family(trajectory: dict[str, Any]) -> str:
        explicit = str(trajectory.get("task_family") or "").strip()
        if explicit:
            return explicit
        try:
            from examples.prompt_agent.gpt4o_webshop import (
                task_family_from_instruction,
            )

            return task_family_from_instruction(
                str(trajectory.get("task") or "")
            )
        except Exception:
            return "webshop"
