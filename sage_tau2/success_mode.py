"""Classify τ² episode success for SAGE / AReaL-style evolution.

Binary ``reward==1`` mixes very different behaviors. Evolution / RL should
distinguish:

- ``solve_write``: DB-mutating tool success (book / update / cancel / …)
- ``transfer_ok``: correct escalation via ``transfer_to_human_agents``
- ``communicate_ok``: pass without write or transfer (explain / read-only)
- ``fail``: reward < 1
"""

from __future__ import annotations

from typing import Any

# Agent-side mutating tools (same spine as distill; no user-device telecom toggles).
WRITE_TOOL_NAMES = frozenset(
    {
        # airline
        "cancel_reservation",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
        "book_reservation",
        "send_certificate",
        # retail
        "cancel_pending_order",
        "modify_pending_order_items",
        "modify_pending_order_address",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
        # telecom agent backend
        "enable_roaming",
        "disable_roaming",
        "refuel_data",
        "resume_line",
        "suspend_line",
        "send_payment_request",
    }
)

TRANSFER_TOOL_NAMES = frozenset({"transfer_to_human_agents"})

SUCCESS_MODES = (
    "fail",
    "solve_write",
    "transfer_ok",
    "communicate_ok",
)


def _tool_name_from_call(call: dict[str, Any]) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = fn.get("name") or call.get("name") or ""
    return str(name).strip()


def tool_names_from_messages(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = _tool_name_from_call(tc)
            if name:
                names.append(name)
    return names


def gold_action_names(reward_info: dict[str, Any] | None) -> list[str]:
    names: list[str] = []
    for check in (reward_info or {}).get("action_checks") or []:
        if not isinstance(check, dict):
            continue
        action = check.get("action")
        if not isinstance(action, dict):
            continue
        name = str(action.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def episode_reward(sim: dict[str, Any]) -> float:
    reward_info = sim.get("reward_info") or {}
    raw = reward_info.get("reward")
    if raw is None:
        raw = sim.get("reward")
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def classify_success_mode(sim: dict[str, Any]) -> dict[str, Any]:
    """Return success_mode plus light evidence for one simulation."""
    reward = episode_reward(sim)
    reward_info = sim.get("reward_info") if isinstance(sim.get("reward_info"), dict) else {}
    tools = tool_names_from_messages(
        [m for m in (sim.get("messages") or []) if isinstance(m, dict)]
    )
    gold = gold_action_names(reward_info)
    tool_set = set(tools)
    gold_set = set(gold)
    combined = tool_set | gold_set
    db_match = (reward_info.get("db_check") or {}).get("db_match")

    if reward < 1.0:
        mode = "fail"
    elif combined & WRITE_TOOL_NAMES:
        mode = "solve_write"
    elif combined & TRANSFER_TOOL_NAMES:
        mode = "transfer_ok"
    else:
        mode = "communicate_ok"

    return {
        "success_mode": mode,
        "reward": reward,
        "db_match": db_match,
        "tools": tools,
        "gold_actions": gold,
        "write_tools": sorted(combined & WRITE_TOOL_NAMES),
        "transfer_tools": sorted(combined & TRANSFER_TOOL_NAMES),
    }


def is_write_gold_episode(sim: dict[str, Any]) -> bool:
    """True when gold actions include a DB-mutating write tool."""
    reward_info = sim.get("reward_info") if isinstance(sim.get("reward_info"), dict) else {}
    gold = set(gold_action_names(reward_info))
    return bool(gold & WRITE_TOOL_NAMES)


def write_gold_fail_task_ids(payload: dict[str, Any]) -> list[str]:
    """Task ids that failed and have write tools in gold (solve_write candidates)."""
    out: list[str] = []
    seen: set[str] = set()
    for sim in payload.get("simulations") or []:
        if not isinstance(sim, dict):
            continue
        if episode_reward(sim) >= 1.0:
            continue
        if not is_write_gold_episode(sim):
            continue
        tid = str(sim.get("task_id") or "").strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def sim_rank_key(sim: dict[str, Any]) -> tuple[float, int, int]:
    """Higher is better: reward, solve_write, db_match."""
    reward = episode_reward(sim)
    mode = classify_success_mode(sim).get("success_mode")
    db = (sim.get("reward_info") or {}).get("db_check") or {}
    return (
        reward,
        1 if mode == "solve_write" else 0,
        1 if db.get("db_match") is True else 0,
    )


def merge_simulations_prefer_best(
    base_payload: dict[str, Any],
    retry_payload: dict[str, Any],
) -> dict[str, Any]:
    """Replace base sims with retry sims when the retry ranks higher (same task_id)."""
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for sim in base_payload.get("simulations") or []:
        if not isinstance(sim, dict):
            continue
        tid = str(sim.get("task_id") or "").strip()
        if not tid:
            continue
        if tid not in by_id:
            order.append(tid)
        by_id[tid] = sim
    replaced: list[str] = []
    for sim in retry_payload.get("simulations") or []:
        if not isinstance(sim, dict):
            continue
        tid = str(sim.get("task_id") or "").strip()
        if not tid:
            continue
        prev = by_id.get(tid)
        if prev is None:
            order.append(tid)
            by_id[tid] = sim
            continue
        if sim_rank_key(sim) > sim_rank_key(prev):
            by_id[tid] = sim
            replaced.append(tid)
    out = dict(base_payload)
    out["simulations"] = [by_id[tid] for tid in order if tid in by_id]
    out["_retry_replaced_task_ids"] = replaced
    return out
