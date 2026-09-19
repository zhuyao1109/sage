"""Adapt τ² simulation JSON into grounded tool trajectories."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from sage_tau2.schemas import Tau2Trajectory, ToolCallStep
from sage_tau2.tool_sides import (
    is_telecom_user_side_tool,
    is_tool_not_found_result,
    strip_user_side_protocol_steps,
)

# Always abstract these keys (PII / episode-specific handles).
_REDACT_KEYS = {
    "reservation_id",
    "user_id",
    "order_id",
    "item_id",
    "item_ids",
    "payment_id",
    "payment_method_id",
    "payment_methods",
    "certificate_id",
    "customer_id",
    "line_id",
    # Bare ``id`` (e.g. get_details_by_id(id=L1001)) is episode-specific.
    "id",
    "email",
    "phone",
    "phone_number",
    "address",
    "name",
    "passengers",
    "flights",
    "summary",
    "expression",
    "new_item_ids",
    "payment_method_ids",
}

# Keep short categorical / routing literals for distill discrimination.
_LITERAL_KEYS = {
    "cabin",
    "flight_type",
    "insurance",
    "origin",
    "destination",
    "date",
    "status",
    "brand",
    "reason",
    "membership",
}

_USERISH_RE = re.compile(r"^[a-z]+_[a-z]+_\d+$", re.IGNORECASE)
_RESERVATIONISH_RE = re.compile(r"^[A-Z0-9]{5,8}$")


def slot_argument_value(key: str, value: Any) -> str:
    """Map one tool arg to a protocol slot: keep categoricals, redact IDs/PII."""
    k = str(key or "").strip().lower()
    if not k:
        return "?"
    if k in _REDACT_KEYS or k.endswith("_id") or k.endswith("_ids"):
        return "?"
    if isinstance(value, (dict, list)):
        return "?"
    if value is None:
        return "?"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and k in _LITERAL_KEYS:
        return str(value)
    text = str(value).strip()
    if not text:
        return "?"
    if len(text) > 32:
        return "?"
    if _USERISH_RE.match(text):
        return "?"
    if k in {"reservation_id"} or (
        k.endswith("reservation_id") and _RESERVATIONISH_RE.match(text)
    ):
        return "?"
    if k in _LITERAL_KEYS:
        return text
    # Short alnum codes (airport, cabin-like tokens) are safe to keep.
    compact = text.replace("_", "").replace("-", "")
    if 2 <= len(text) <= 16 and compact.isalnum():
        return text
    return "?"


def canonicalize_tool_step(name: str, arguments: dict[str, Any] | None) -> str:
    """Abstract episode IDs into slots; keep short categorical literals.

    Example:
      update_reservation_flights(cabin=economy, flights=?, payment_id=?, reservation_id=?)
    """
    args = arguments or {}
    if not args:
        return f"{name}()"
    parts = []
    for key in sorted(str(k) for k in args.keys()):
        parts.append(f"{key}={slot_argument_value(key, args.get(key))}")
    return f"{name}({', '.join(parts)})"


def _tool_call_name_and_args(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Support both flat and OpenAI-style ``function.{name,arguments}`` shapes."""
    fn = call.get("function")
    if isinstance(fn, dict):
        name = str(fn.get("name") or call.get("name") or "").strip()
        args = fn.get("arguments")
        if args is None:
            args = call.get("arguments") or {}
    else:
        name = str(call.get("name") or "").strip()
        args = call.get("arguments") or {}
    if isinstance(args, str):
        import json

        try:
            parsed = json.loads(args)
            args = parsed if isinstance(parsed, dict) else {}
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, dict(args)


def extract_tool_steps(messages: list[dict[str, Any]]) -> list[ToolCallStep]:
    """Pair assistant tool_calls with subsequent tool results when possible."""
    by_id: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            call_id = str(msg.get("id") or msg.get("tool_call_id") or "")
            if call_id:
                by_id[call_id] = msg

    steps: list[ToolCallStep] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name, args = _tool_call_name_and_args(call)
            if not name:
                continue
            call_id = str(call.get("id") or "") or None
            result = by_id.get(call_id or "")
            content = None if result is None else str(result.get("content") or "")
            errored = bool(result.get("error")) if result else False
            if is_tool_not_found_result(content):
                errored = True
            steps.append(
                ToolCallStep(
                    name=name,
                    arguments=dict(args),
                    tool_call_id=call_id,
                    result_content=content,
                    result_error=errored,
                )
            )
    return steps


def protocol_from_tool_steps(
    tool_steps: list[ToolCallStep],
    *,
    domain: str,
) -> list[str]:
    """Build a distillable agent protocol: drop user-side / not-found calls."""
    kept: list[str] = []
    for step in tool_steps:
        if step.result_error:
            continue
        if str(domain).lower() in {"telecom", "telecom-workflow"} and is_telecom_user_side_tool(
            step.name
        ):
            continue
        kept.append(canonicalize_tool_step(step.name, step.arguments))
    return strip_user_side_protocol_steps(kept, domain=domain)


def simulation_to_trajectory(
    sim: dict[str, Any],
    *,
    domain: str,
    task: dict[str, Any] | None = None,
) -> Tau2Trajectory:
    messages = [m for m in (sim.get("messages") or []) if isinstance(m, dict)]
    reward_info = sim.get("reward_info") or {}
    breakdown = reward_info.get("reward_breakdown") or {}
    db_check = reward_info.get("db_check") or {}
    tool_steps = extract_tool_steps(messages)
    protocol = protocol_from_tool_steps(tool_steps, domain=domain)
    assistant_texts = [
        str(m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "assistant" and m.get("content")
    ]
    user_texts = [
        str(m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "user" and m.get("content")
    ]
    db_match = db_check.get("db_match")
    if db_match is not None:
        db_match = bool(db_match)
    from sage_tau2.success_mode import classify_success_mode

    mode_info = classify_success_mode(sim)
    return Tau2Trajectory(
        task_id=str(sim.get("task_id") or (task or {}).get("id") or ""),
        trial=int(sim.get("trial") or 0),
        domain=str(domain),
        reward=float(reward_info.get("reward") or 0.0),
        db_reward=(
            None
            if breakdown.get("DB") is None and db_check.get("db_reward") is None
            else float(
                breakdown.get("DB")
                if breakdown.get("DB") is not None
                else db_check.get("db_reward")
            )
        ),
        communicate_reward=(
            None
            if breakdown.get("COMMUNICATE") is None
            else float(breakdown.get("COMMUNICATE"))
        ),
        db_match=db_match,
        termination_reason=(
            None
            if sim.get("termination_reason") is None
            else str(sim.get("termination_reason"))
        ),
        tool_protocol=protocol,
        tool_steps=tool_steps,
        assistant_texts=assistant_texts,
        user_texts=user_texts,
        evidence_id=str(sim.get("id") or uuid4()),
        raw_messages=messages,
        metadata={
            "seed": sim.get("seed"),
            "duration": sim.get("duration"),
            "action_checks": reward_info.get("action_checks"),
            "communicate_checks": reward_info.get("communicate_checks"),
            "task_description": (task or {}).get("description"),
            "success_mode": mode_info.get("success_mode"),
            "write_tools": mode_info.get("write_tools"),
            "transfer_tools": mode_info.get("transfer_tools"),
            "gold_actions": mode_info.get("gold_actions"),
            "tools": mode_info.get("tools"),
        },
    )


def results_json_to_trajectories(
    payload: dict[str, Any],
    *,
    domain: str,
) -> list[Tau2Trajectory]:
    tasks = payload.get("tasks") or []
    task_by_id = {
        str(t.get("id")): t for t in tasks if isinstance(t, dict) and t.get("id") is not None
    }
    out: list[Tau2Trajectory] = []
    for sim in payload.get("simulations") or []:
        if not isinstance(sim, dict):
            continue
        task_id = str(sim.get("task_id") or "")
        out.append(
            simulation_to_trajectory(
                sim,
                domain=domain,
                task=task_by_id.get(task_id),
            )
        )
    return out
