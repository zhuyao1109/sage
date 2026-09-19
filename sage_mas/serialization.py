"""JSON serialization helpers for SAGE-MAS dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from sage_mas.schemas import (
    AgentSpec,
    AtomicOp,
    DistributionShift,
    ExplorationDistribution,
    Skill,
    SkillStatus,
    StateActionFragment,
)


def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return to_primitive(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(to_primitive(value), f, ensure_ascii=False, indent=2)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, values: Iterable[Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(json.dumps(to_primitive(value), ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def skill_from_dict(item: dict[str, Any]) -> Skill:
    payload = dict(item)
    payload["status"] = SkillStatus(payload.get("status", SkillStatus.CANDIDATE.value))
    payload["applicable_atomic_ops"] = [
        value if isinstance(value, AtomicOp) else AtomicOp(value)
        for value in payload.get("applicable_atomic_ops", [])
    ]
    if payload.get("key_fragments"):
        payload["key_fragments"] = [
            StateActionFragment(**fragment)
            if isinstance(fragment, dict)
            else fragment
            for fragment in payload["key_fragments"]
        ]
    if payload.get("exploration_distribution"):
        distribution = payload["exploration_distribution"]
        if isinstance(distribution, dict):
            payload["exploration_distribution"] = ExplorationDistribution(
                **distribution
            )
    if payload.get("distribution_shift"):
        shift = payload["distribution_shift"]
        if isinstance(shift, dict):
            payload["distribution_shift"] = DistributionShift(**shift)
    return Skill(**payload)


def agent_from_dict(item: dict[str, Any]) -> AgentSpec:
    allowed = {field.name for field in fields(AgentSpec)}
    payload = {key: value for key, value in dict(item).items() if key in allowed}
    return AgentSpec(**payload)


def load_skills(path: str | Path) -> list[Skill]:
    payload = read_json(path)
    records = payload.get("skills", []) if isinstance(payload, dict) else payload
    return [skill_from_dict(item) for item in records]


def load_agents(path: str | Path) -> list[AgentSpec]:
    payload = read_json(path)
    records = payload.get("agents", []) if isinstance(payload, dict) else payload
    return [agent_from_dict(item) for item in records]
