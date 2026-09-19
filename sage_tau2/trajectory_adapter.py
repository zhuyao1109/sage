"""Adapt PRO trajectories (ALFWorld-style dump) for τ² distill / credit.

Distill reads ``trajectories.jsonl`` (PRO steps) and the slim intermediate
``atomic_trajectories.json`` — not raw ``results.json`` messages or
``evolve_trajectories.jsonl``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from sage_tau2.distill import trajectory_eligible_for_distill
from sage_tau2.schemas import Tau2Trajectory, ToolCallStep
from sage_tau2.serialization import read_json, to_primitive, trajectory_from_dict, write_json
from sage_tau2.tool_sides import is_tool_not_found_result
from sage_tau2.trajectory import protocol_from_tool_steps

_TOOL_RESULT_RE = re.compile(r"^\[tool_result:([^\]]+)\]\s*(.*)", re.DOTALL)
_USER_RE = re.compile(r"^\[USER\]\s*(.*)", re.DOTALL)


def _parse_tool_result_block(text: str) -> tuple[str, str, bool]:
    """Return (tool_name, body, errored) from a PRO observation line."""
    body = str(text or "").strip()
    match = _TOOL_RESULT_RE.match(body)
    if not match:
        return "", body, False
    name = match.group(1).strip()
    payload = match.group(2).strip()
    lowered = payload.lower()
    errored = (
        "error=" in lowered
        or "not found" in lowered
        or is_tool_not_found_result(payload)
    )
    return name, payload, errored


def _user_texts_from_pro_steps(steps: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for step in steps:
        for field in ("observation_before", "observation"):
            raw = str(step.get(field) or "")
            for block in raw.split("\n\n"):
                block = block.strip()
                user_match = _USER_RE.match(block)
                if not user_match:
                    continue
                text = " ".join(user_match.group(1).split())
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
    return texts


def _assistant_texts_from_pro_steps(steps: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for step in steps:
        if step.get("tool_calls"):
            continue
        action = str(step.get("action") or "").strip()
        if action:
            out.append(action)
    return out


def _tool_steps_from_pro_steps(
    steps: list[dict[str, Any]],
) -> list[ToolCallStep]:
    """Pair each step's ``tool_calls`` with tool results in the same step observation."""
    tool_steps: list[ToolCallStep] = []
    for step in steps:
        tool_calls = [
            tc for tc in (step.get("tool_calls") or []) if isinstance(tc, dict)
        ]
        if not tool_calls:
            continue
        obs = str(step.get("observation") or "")
        result_blocks = [
            block.strip() for block in obs.split("\n\n") if block.strip()
        ]
        result_idx = 0
        for tc in tool_calls:
            name = str(tc.get("name") or "").strip()
            if not name:
                continue
            args = tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            call_id = str(tc.get("id") or "") or None
            content: str | None = None
            errored = False
            while result_idx < len(result_blocks):
                block = result_blocks[result_idx]
                result_idx += 1
                if not block.startswith("[tool_result:"):
                    continue
                result_name, payload, block_err = _parse_tool_result_block(block)
                if result_name and result_name != name:
                    # Keep scanning; mismatched ordering is rare in PRO dumps.
                    content = payload
                    errored = block_err
                    break
                content = payload
                errored = block_err
                break
            tool_steps.append(
                ToolCallStep(
                    name=name,
                    arguments=dict(args),
                    tool_call_id=call_id,
                    result_content=content,
                    result_error=errored,
                )
            )
    return tool_steps


def pro_record_to_trajectory(record: dict[str, Any]) -> Tau2Trajectory:
    """Convert one PRO ``trajectories.jsonl`` row into ``Tau2Trajectory``."""
    steps = [s for s in (record.get("steps") or []) if isinstance(s, dict)]
    domain = str(record.get("domain") or "")
    distill_meta = (
        record.get("distill_meta") if isinstance(record.get("distill_meta"), dict) else {}
    )
    tool_steps = _tool_steps_from_pro_steps(steps)
    protocol = protocol_from_tool_steps(tool_steps, domain=domain)
    db_match = record.get("db_match")
    if db_match is not None:
        db_match = bool(db_match)
    metadata = {
        "success_mode": distill_meta.get("success_mode"),
        "action_checks": distill_meta.get("action_checks"),
        "communicate_checks": distill_meta.get("communicate_checks"),
        "write_tools": distill_meta.get("write_tools"),
        "transfer_tools": distill_meta.get("transfer_tools"),
        "gold_actions": distill_meta.get("gold_actions"),
        "tools": distill_meta.get("tools"),
        "pro_format": record.get("format"),
        "eligible_for_distill": distill_meta.get("eligible_for_distill"),
    }
    traj = Tau2Trajectory(
        task_id=str(record.get("task_id") or ""),
        trial=int(record.get("trial") or 0),
        domain=domain,
        reward=float(record.get("reward") or 0.0),
        db_reward=distill_meta.get("db_reward"),
        communicate_reward=distill_meta.get("communicate_reward"),
        db_match=db_match,
        termination_reason=(
            None
            if record.get("termination_reason") is None
            else str(record.get("termination_reason"))
        ),
        tool_protocol=protocol,
        tool_steps=tool_steps,
        assistant_texts=_assistant_texts_from_pro_steps(steps),
        user_texts=_user_texts_from_pro_steps(steps),
        evidence_id=str(record.get("task_id") or uuid4()),
        raw_messages=[],
        metadata={k: v for k, v in metadata.items() if v is not None},
    )
    if metadata.get("eligible_for_distill") is None:
        metadata["eligible_for_distill"] = trajectory_eligible_for_distill(traj)
        traj.metadata["eligible_for_distill"] = metadata["eligible_for_distill"]
    return traj


def load_pro_trajectories(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def adapt_pro_many(records: list[dict[str, Any]]) -> list[Tau2Trajectory]:
    return [pro_record_to_trajectory(record) for record in records]


def trajectories_to_atomic_dicts(trajectories: list[Tau2Trajectory]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for traj in trajectories:
        payload = to_primitive(traj)
        payload["eligible_for_distill"] = trajectory_eligible_for_distill(traj)
        out.append(payload)
    return out


def export_atomic_trajectories(
    trajectories: list[Tau2Trajectory],
    path: str | Path,
) -> Path:
    """Write ALFWorld-style ``atomic_trajectories.json`` for distill."""
    out_path = Path(path).expanduser().resolve()
    write_json(out_path, trajectories_to_atomic_dicts(trajectories))
    return out_path


def load_atomic_trajectories(
    path: str | Path,
    *,
    eligible_only: bool = False,
) -> list[Tau2Trajectory]:
    payload = read_json(path)
    if not isinstance(payload, list):
        return []
    out: list[Tau2Trajectory] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        traj = trajectory_from_dict(item)
        if eligible_only and not trajectory_eligible_for_distill(traj):
            continue
        out.append(traj)
    return out


def pro_trajectories_path(base: str | Path) -> Path:
    return Path(base).expanduser().resolve() / "pro_trajectories" / "trajectories.jsonl"


def atomic_trajectories_path(base: str | Path) -> Path:
    return Path(base).expanduser().resolve() / "pro_trajectories" / "atomic_trajectories.json"
