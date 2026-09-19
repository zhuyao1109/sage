"""Serialize / deserialize sage_tau2 objects."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from sage_tau2.schemas import SkillStatus, Tau2Skill, ToolCallStep, Tau2Trajectory


def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return to_primitive(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): to_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(v) for v in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(to_primitive(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def skill_to_dict(skill: Tau2Skill) -> dict[str, Any]:
    return to_primitive(skill)


def skill_from_dict(item: dict[str, Any]) -> Tau2Skill:
    payload = dict(item)
    status = payload.get("status", SkillStatus.CANDIDATE.value)
    payload["status"] = (
        status if isinstance(status, SkillStatus) else SkillStatus(str(status))
    )
    allowed = {f.name for f in fields(Tau2Skill)}
    return Tau2Skill(**{k: v for k, v in payload.items() if k in allowed})


def trajectory_from_dict(item: dict[str, Any]) -> Tau2Trajectory:
    payload = dict(item)
    steps = []
    for step in payload.get("tool_steps") or []:
        if isinstance(step, ToolCallStep):
            steps.append(step)
        else:
            steps.append(ToolCallStep(**dict(step)))
    payload["tool_steps"] = steps
    allowed = {f.name for f in fields(Tau2Trajectory)}
    return Tau2Trajectory(**{k: v for k, v in payload.items() if k in allowed})
