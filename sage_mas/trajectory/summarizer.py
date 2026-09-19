"""LLM/heuristic trajectory summarizer that rewrites skill fields."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from sage_mas.schemas import AtomicStep, Skill, StateActionFragment
from sage_mas.trajectory.abstraction import (
    trajectory_action_steps,
    trajectory_id_from_steps,
)
from sage_mas.trajectory.finalize import (
    finalize_operation_protocol,
    lock_operation_suggested_role,
)
from sage_mas.trajectory.fragments import heuristic_trajectory_summary

class SummarizationBackend(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> Any:
        ...


class TrajectorySummarizer:
    """Generate skill fields from trajectory evidence via LLM or heuristics."""

    def __init__(
        self,
        backend: SummarizationBackend | None = None,
        max_evidence_trajectories: int = 3,
        max_steps_per_trajectory: int = 15,
        max_completion_tokens: int = 4096,
        max_retries: int = 2,
    ):
        self.backend = backend
        self.max_evidence_trajectories = max_evidence_trajectories
        self.max_steps_per_trajectory = max_steps_per_trajectory
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max(1, int(max_retries))

    def summarize(
        self,
        seed: Skill,
        trajectories: list[list[AtomicStep]],
        fragments: list[StateActionFragment],
    ) -> Skill:
        summary = heuristic_trajectory_summary(seed, trajectories, fragments)
        if self.backend is None:
            seed.trajectory_summary = summary
            seed.metadata["summarizer"] = "heuristic-v1"
            return seed
        locked_protocol = [
            str(step).strip()
            for step in (seed.action_protocol or [])
            if str(step).strip()
        ]
        protocol_locked = bool(
            locked_protocol
            and str(seed.metadata.get("protocol_source", "")).startswith(
                "aligned_successful_trajectory_stages"
            )
        )
        system_prompt = (
            "You are the Skill Distiller rewrite layer in SAGE-MAS. "
            "Compress only reusable behavior supported by the supplied "
            "trajectories. Do not invent tools, observations, object states, "
            "or successful outcomes. You rewrite checkable text around an "
            "already-grounded protocol; you do not invent a new protocol. "
            "If locked_action_protocol is present, copy it unchanged into "
            "action_protocol. Every free-form claim must be supported by "
            "cited evidence. Include checkable preconditions/effects and "
            "concrete anti_patterns / failure_branches (what fails when the "
            "operation is skipped or ordered wrong). Keep lamp skills as "
            "`use desklamp` (never `use <entity>`). A positive-effect failed "
            "episode supports only the observed local effect, not full task "
            "completion. Do not use reasoning or communication messages as "
            "executed actions. Return one JSON object with keys: skill_name, "
            "description, precondition, action_protocol (array), "
            "expected_effect, suggested_role, target_failure_types (array), "
            "anti_patterns (array of short strings), failure_branches "
            "(array of short strings), trajectory_summary (one paragraph "
            "citing evidence). Keep suggested_role equal to the seed role "
            "when provided."
        )
        user_prompt = json.dumps(
            {
                "rewrite_mode": "grounded_protocol_locked"
                if protocol_locked
                else "grounded_protocol_preferred",
                "locked_action_protocol": locked_protocol if protocol_locked else None,
                "seed_hypothesis": {
                    "name": seed.skill_name,
                    "description": seed.description,
                    "precondition": seed.precondition,
                    "protocol": seed.action_protocol,
                    "expected_effect": seed.expected_effect,
                    "anti_patterns": seed.metadata.get("anti_patterns") or [],
                    "source_signal": seed.metadata.get("source_signal"),
                    "primary_task_family": seed.metadata.get(
                        "primary_task_family"
                    ),
                    "capability_key": seed.capability_key,
                    "protocol_stages": seed.metadata.get("protocol_stages"),
                },
                "key_fragments": [
                    {
                        "trajectory_id": fragment.trajectory_id,
                        "step_index": fragment.step_index,
                        "action": fragment.action,
                        "observation": fragment.observation,
                        "won": fragment.won,
                        "task_family": fragment.task_family,
                    }
                    for fragment in fragments
                ],
                "trajectory_evidence": [
                    self._trajectory_excerpt(steps)
                    for steps in trajectories[: self.max_evidence_trajectories]
                ],
            },
            ensure_ascii=False,
        )
        last_error: Exception | None = None
        payload: dict[str, Any] | None = None
        for attempt in range(self.max_retries):
            attempt_system = system_prompt
            attempt_user = user_prompt
            if attempt > 0:
                attempt_system = (
                    system_prompt
                    + " Previous response was invalid JSON. Reply with a "
                    "single compact JSON object only — no markdown fences, "
                    "no trailing commas, keep strings short."
                )
                attempt_user = json.dumps(
                    {
                        "rewrite_mode": "grounded_protocol_locked"
                        if protocol_locked
                        else "grounded_protocol_preferred",
                        "locked_action_protocol": (
                            locked_protocol if protocol_locked else None
                        ),
                        "seed_hypothesis": {
                            "name": seed.skill_name,
                            "precondition": seed.precondition,
                            "protocol": seed.action_protocol,
                            "capability_key": seed.capability_key,
                            "source_signal": seed.metadata.get("source_signal"),
                        },
                        "key_fragments": [
                            {
                                "trajectory_id": fragment.trajectory_id,
                                "action": fragment.action,
                                "won": fragment.won,
                            }
                            for fragment in fragments[:3]
                        ],
                    },
                    ensure_ascii=False,
                )
            try:
                try:
                    response = self.backend.complete(
                        attempt_system,
                        attempt_user,
                        max_completion_tokens=self.max_completion_tokens,
                    )
                except TypeError:
                    response = self.backend.complete(attempt_system, attempt_user)
                payload = self._parse_json(str(response.content))
                break
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                continue
        if payload is None:
            seed.trajectory_summary = summary
            seed.metadata["summarizer"] = "heuristic-fallback"
            seed.metadata["summarizer_error"] = (
                f"{type(last_error).__name__}: {last_error}"
                if last_error is not None
                else "empty_summarizer_payload"
            )
            return seed
        try:
            llm_protocol = [
                str(step).strip()
                for step in payload.get("action_protocol", [])
                if str(step).strip()
            ]
            if protocol_locked:
                protocol = list(locked_protocol)
            else:
                protocol = llm_protocol or list(locked_protocol)
            if not protocol:
                seed.trajectory_summary = summary
                seed.metadata["summarizer"] = "heuristic-fallback"
                seed.metadata["summarizer_error"] = "empty_action_protocol"
                return seed
            seed.skill_name = str(
                payload.get("skill_name") or seed.skill_name
            ).strip()
            seed.description = str(
                payload.get("description") or seed.description
            ).strip()
            seed.precondition = str(
                payload.get("precondition") or seed.precondition
            ).strip()
            # Operation markers are mandatory; drop LLM protocol if it omits
            # clean/cool/heat/desklamp. Locked grounded protocols already pass.
            seed.action_protocol = finalize_operation_protocol(seed, protocol)
            seed.expected_effect = str(
                payload.get("expected_effect") or seed.expected_effect or ""
            ).strip()
            seed.suggested_role = lock_operation_suggested_role(
                seed,
                str(
                    payload.get("suggested_role") or seed.suggested_role or "Executor"
                ).strip(),
            )
            seed.target_failure_types = [
                str(item)
                for item in payload.get(
                    "target_failure_types",
                    seed.target_failure_types,
                )
            ]
            anti_patterns = [
                str(item).strip()
                for item in (
                    payload.get("anti_patterns")
                    or seed.metadata.get("anti_patterns")
                    or []
                )
                if str(item).strip()
            ]
            failure_branches = [
                str(item).strip()
                for item in (payload.get("failure_branches") or [])
                if str(item).strip()
            ]
            if anti_patterns:
                seed.metadata["anti_patterns"] = anti_patterns
            if failure_branches:
                seed.metadata["failure_branches"] = failure_branches
            seed.trajectory_summary = str(
                payload.get("trajectory_summary") or summary
            ).strip()
            seed.metadata["summarizer"] = "trajectory-llm-v2-rewrite"
            seed.metadata["trajectory_grounded"] = True
            seed.metadata["protocol_locked"] = protocol_locked
            if hasattr(self.backend, "model"):
                seed.metadata["summarizer_model"] = str(self.backend.model)
            return seed
        except (AttributeError, TypeError, ValueError) as exc:
            seed.trajectory_summary = summary
            seed.metadata["summarizer"] = "heuristic-fallback"
            seed.metadata["summarizer_error"] = f"{type(exc).__name__}: {exc}"
            return seed

    def _trajectory_excerpt(self, steps: list[AtomicStep]) -> dict[str, Any]:
        terminal = steps[-1]
        return {
            "trajectory_id": trajectory_id_from_steps(steps),
            "task": terminal.metadata.get("task"),
            "task_family": terminal.metadata.get("task_family"),
            "won": terminal.metadata.get("won"),
            "steps": [
                {
                    "atomic_op": step.atomic_op.value,
                    "action": step.action,
                    "observation": (
                        str(step.observation)[:240]
                        if step.observation is not None
                        else None
                    ),
                    "is_action_valid": step.metadata.get("is_action_valid"),
                    "stalled": step.metadata.get("stalled"),
                }
                for step in trajectory_action_steps(steps)[
                    -self.max_steps_per_trajectory :
                ]
            ],
        }

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        fenced = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```",
            content,
            re.DOTALL,
        )
        if fenced:
            content = fenced.group(1)
        else:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                content = content[start : end + 1]
        # Common Gemini truncation / style issues.
        content = re.sub(r",\s*}", "}", content)
        content = re.sub(r",\s*]", "]", content)
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Summarizer response must be a JSON object")
        return payload
