"""Trajectory indexing, key fragments, heuristic summaries, embeddings."""

from __future__ import annotations

import hashlib
import re

from sage_mas.schemas import AtomicStep, Skill, StateActionFragment
from sage_mas.trajectory.abstraction import (
    _abstract_environment_action,
    _observation_is_noop,
    is_productive_environment_step,
    stage_for_abstracted_action,
    trajectory_action_steps,
    trajectory_id_from_steps,
)

def index_trajectories(
    trajectories: list[list[AtomicStep]],
) -> dict[str, list[AtomicStep]]:
    indexed: dict[str, list[AtomicStep]] = {}
    for steps in trajectories:
        if not steps:
            continue
        indexed[trajectory_id_from_steps(steps)] = steps
    return indexed


def extract_key_fragments(
    trajectories: list[list[AtomicStep]],
    *,
    max_fragments: int = 5,
) -> list[StateActionFragment]:
    """Prefer productive success steps; keep a few failure no-ops separately."""
    scored: list[tuple[float, StateActionFragment]] = []
    for steps in trajectories:
        if not steps:
            continue
        terminal = steps[-1]
        trajectory_id = trajectory_id_from_steps(steps)
        won = bool(terminal.metadata.get("won", False))
        task_family = str(terminal.metadata.get("task_family", "other"))
        action_steps = trajectory_action_steps(steps)
        if not action_steps:
            continue
        for step in action_steps:
            observation = str(step.observation or "").strip()
            action = str(step.action or "").strip()
            if not observation and not action:
                continue
            productive = is_productive_environment_step(step)
            score = 0.0
            if won and productive:
                abstracted = _abstract_environment_action(action)
                stage = stage_for_abstracted_action(abstracted)
                score += {
                    "transform": 5.0,
                    "pickup": 4.0,
                    "place": 4.0,
                    "access": 2.0,
                    "find": 1.5,
                    "other": 1.0,
                }.get(stage, 1.0)
            elif not won and (
                step.metadata.get("stalled")
                or step.metadata.get("is_action_valid") is False
                or _observation_is_noop(observation)
            ):
                # Keep a little failure evidence for anti-pattern context only.
                score += 1.0
            else:
                continue
            if step is action_steps[-1] and productive:
                score += 0.5
            scored.append(
                (
                    score,
                    StateActionFragment(
                        trajectory_id=trajectory_id,
                        step_index=int(step.metadata.get("step_index", 0)),
                        observation=observation[:240],
                        action=action[:120],
                        won=won,
                        task_family=task_family,
                    ),
                )
            )
        if won and not any(
            fragment.trajectory_id == trajectory_id for _, fragment in scored
        ):
            # Degenerate winning traces still need at least one fragment.
            step = action_steps[-1]
            scored.append(
                (
                    0.25,
                    StateActionFragment(
                        trajectory_id=trajectory_id,
                        step_index=int(step.metadata.get("step_index", 0)),
                        observation=str(step.observation or "").strip()[:240],
                        action=str(step.action or "").strip()[:120],
                        won=True,
                        task_family=task_family,
                    ),
                )
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    unique: list[StateActionFragment] = []
    seen: set[tuple[str, int, str]] = set()
    for _, fragment in scored:
        key = (fragment.trajectory_id, fragment.step_index, fragment.action)
        if key in seen:
            continue
        seen.add(key)
        unique.append(fragment)
        if len(unique) >= max_fragments:
            break
    # Keep chronological order for debugging / reflexion (selection was by score).
    unique.sort(
        key=lambda fragment: (
            str(fragment.trajectory_id or ""),
            int(fragment.step_index),
        )
    )
    return unique


def heuristic_trajectory_summary(
    skill: Skill,
    trajectories: list[list[AtomicStep]],
    fragments: list[StateActionFragment],
) -> str:
    families = sorted(
        {
            str(steps[-1].metadata.get("task_family", "other"))
            for steps in trajectories
            if steps
        }
    )
    outcomes = sorted(
        {
            "success" if bool(steps[-1].metadata.get("won", False)) else "failure"
            for steps in trajectories
            if steps
        }
    )
    signal = str(skill.metadata.get("source_signal", "unknown"))
    stages = skill.metadata.get("protocol_stages") or [
        stage_for_abstracted_action(step) for step in skill.action_protocol
    ]
    anti = skill.metadata.get("anti_patterns") or []
    fragment_lines = [
        (
            f"[{fragment.task_family}|{fragment.trajectory_id}|step {fragment.step_index}] "
            f"{fragment.action} -> {fragment.observation}"
        )
        for fragment in fragments[:3]
    ]
    return (
        f"Compressed from {len(trajectories)} evidence trajectories across "
        f"{', '.join(families) or 'unknown'} with outcomes {', '.join(outcomes)}. "
        f"Dominant signal: {signal}. "
        f"Stages: {' -> '.join(stages) if stages else 'n/a'}. "
        f"Anti-patterns: {'; '.join(anti[:2]) if anti else 'none'}. "
        f"Key transitions: {' | '.join(fragment_lines) if fragment_lines else 'none'}."
    )


def embed_skill_text(text: str, *, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = sum(value * value for value in vector) ** 0.5
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


def embed_skill(skill: Skill) -> list[float]:
    """Embed behavior semantics without task-family or outcome-order leakage.

    The encoder is deterministic and versioned so historical and current
    skills are always compared in the same fixed representation space.
    """
    dimensions = 64
    vector = [0.0] * dimensions
    stopwords = {
        "the",
        "and",
        "then",
        "with",
        "from",
        "into",
        "before",
        "after",
        "object",
        "target",
        "task",
        "skill",
        "agent",
    }
    fields = [
        ("name", skill.skill_name, 1.5),
        ("description", skill.description, 1.0),
        ("precondition", skill.precondition, 2.0),
        ("protocol", " ".join(skill.action_protocol), 3.0),
        ("effect", skill.expected_effect or "", 2.0),
    ]
    fields.extend(
        ("action", fragment.action, 3.0)
        for fragment in skill.key_fragments
    )
    fields.extend(
        ("observation", fragment.observation, 0.5)
        for fragment in skill.key_fragments
    )
    for prefix, text, weight in fields:
        for token in re.findall(r"[a-z0-9_]+", str(text).lower()):
            if token in stopwords:
                continue
            feature = f"{prefix}:{token}"
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign * weight
    norm = sum(value * value for value in vector) ** 0.5
    skill.metadata["embedding_version"] = "skill-behavior-hash-v2"
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]
