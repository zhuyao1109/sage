"""Executor-source governance: only call tools the agent actually owns.

Telecom policy lists user-device APIs with function-like names. Models then emit
those as agent tool_calls → ``Tool not found``. This module:

1. Adds an explicit toolkit boundary addendum to the system policy.
2. Strips / rewrites illegal tool_calls **before** they hit the environment.

Status — rewrite / guidance path (closed, not scheduled):
    When the boundary prompt works, assistants rarely emit illegal tool_calls, so
    ``guard_assistant_message`` rewrite fires near-zero in clean OOD runs. That is
    expected (belt-and-suspenders), not a defect. Do not add production debug
    counters or further rewrite work unless an ablation explicitly needs them.
    Unit coverage stays in ``tests/sage_tau2/test_executor_tool_guard_smoke.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

from loguru import logger

from sage_tau2.tool_sides import TELECOM_USER_TOOLS, is_telecom_user_side_tool

# Friendly labels for converting blocked calls into user guidance.
_USER_ACTION_HINTS: dict[str, str] = {
    "toggle_airplane_mode": "turn Airplane Mode OFF (or ON) in quick settings",
    "toggle_data": "turn Mobile Data ON in network settings",
    "toggle_roaming": "turn Data Roaming ON in mobile network settings",
    "toggle_data_saver_mode": "turn Data Saver OFF",
    "toggle_wifi_calling": "turn Wi-Fi Calling OFF",
    "toggle_wifi": "toggle Wi-Fi as needed",
    "set_network_mode_preference": "set Preferred Network Mode to `4g_5g_preferred`",
    "reseat_sim_card": "remove and re-insert the SIM card",
    "reboot_device": "reboot the phone",
    "reset_apn_settings": "reset APN settings to default, then reboot",
    "grant_app_permission": "grant SMS and storage permissions to the messaging app",
    "run_speed_test": "run a mobile-data speed test and tell me the result",
    "check_network_status": "check Network Status and read back airplane/data/roaming",
    "check_status_bar": "check the status bar icons and describe what you see",
    "check_sim_status": "check SIM status (active / missing / locked)",
    "check_apn_settings": "check APN / MMSC settings",
    "check_app_permissions": "check messaging app permissions",
    "can_send_mms": "try sending an MMS and tell me if it works",
    "disconnect_vpn": "disconnect any active VPN",
    "make_payment": "open payment requests and complete the payment",
    "check_payment_request": "check pending payment requests on your side",
}


@dataclass(frozen=True)
class ToolGuardDecision:
    """Result of filtering one assistant turn's tool_calls."""

    kept_names: tuple[str, ...]
    blocked_names: tuple[str, ...]
    guidance_content: str | None
    """If set, the turn should become a user-facing message (no tool_calls)."""


def allowed_tool_names(tools: Sequence[Any]) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        name = getattr(tool, "name", None)
        if name:
            names.add(str(name))
    return names


def classify_tool_names(
    call_names: Iterable[str],
    *,
    allowed: set[str],
) -> ToolGuardDecision:
    """Keep names in ``allowed``; block anything else (esp. telecom user tools)."""
    kept: list[str] = []
    blocked: list[str] = []
    for raw in call_names:
        name = str(raw or "").strip()
        if not name:
            continue
        if name in allowed:
            kept.append(name)
        else:
            blocked.append(name)
    guidance = None
    if blocked and not kept:
        guidance = build_user_guidance(blocked)
    return ToolGuardDecision(
        kept_names=tuple(kept),
        blocked_names=tuple(blocked),
        guidance_content=guidance,
    )


def build_user_guidance(blocked_names: Sequence[str]) -> str:
    """Turn illegal device-tool calls into an instruction for the user."""
    # Preserve order, unique.
    seen: set[str] = set()
    ordered: list[str] = []
    for name in blocked_names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    lines = [
        "I can't change phone settings from my side — those are actions on your device.",
        "Please do the following and tell me what you see afterward:",
    ]
    for i, name in enumerate(ordered, start=1):
        hint = _USER_ACTION_HINTS.get(name)
        if hint:
            lines.append(f"{i}. {hint}")
        elif is_telecom_user_side_tool(name):
            lines.append(f"{i}. perform the device action `{name}` and report the result")
        else:
            lines.append(
                f"{i}. I don't have a tool named `{name}`; please share the related "
                "status from your phone settings if you can."
            )
    return "\n".join(lines)


def telecom_tool_boundary_addendum(*, agent_tool_names: Sequence[str] | None = None) -> str:
    """Policy appendix: explicit agent toolkit vs user-device actions."""
    agent_list = sorted(agent_tool_names or [])
    agent_block = (
        ", ".join(f"`{n}`" for n in agent_list)
        if agent_list
        else ", ".join(f"`{n}`" for n in sorted(
            {
                "get_customer_by_phone",
                "get_customer_by_id",
                "get_customer_by_name",
                "get_details_by_id",
                "get_bills_for_customer",
                "get_data_usage",
                "send_payment_request",
                "resume_line",
                "suspend_line",
                "enable_roaming",
                "disable_roaming",
                "refuel_data",
                "transfer_to_human_agents",
            }
        ))
    )
    user_examples = (
        "check_status_bar, check_network_status, check_sim_status, run_speed_test, "
        "toggle_airplane_mode, toggle_data, toggle_roaming, set_network_mode_preference, "
        "reseat_sim_card, reboot_device, reset_apn_settings, grant_app_permission, "
        "can_send_mms, make_payment"
    )
    return f"""
<agent_user_tool_boundary>
CRITICAL — two toolkits exist in telecom. You may ONLY call agent tools.

Agent toolkit (callable by you): {agent_block}

User-device actions (NOT callable by you; names may appear in the policy above):
{user_examples}, and other phone diagnostics/fixes.

If a fix requires a user-device action: ask the user to perform it, wait for their
result, then continue. Never emit those names as your tool calls — that yields
"Tool not found" and wastes the turn.
</agent_user_tool_boundary>
""".strip()


def with_tool_boundary_policy(
    domain_policy: str,
    *,
    tools: Sequence[Any] | None = None,
    domain: str | None = None,
    force: bool = False,
) -> str:
    """Append the boundary addendum for telecom (or when forced)."""
    policy = domain_policy or ""
    if "<agent_user_tool_boundary>" in policy:
        return policy
    dom = (domain or "").lower()
    names = allowed_tool_names(tools or [])
    telecom_markers = {
        "enable_roaming",
        "refuel_data",
        "resume_line",
        "get_customer_by_phone",
    }
    is_telecom = (
        force
        or dom in {"telecom", "telecom-workflow"}
        or bool(names & telecom_markers)
    )
    if not is_telecom:
        return policy
    addendum = telecom_tool_boundary_addendum(
        agent_tool_names=sorted(names) if names else None
    )
    return f"{policy.rstrip()}\n\n{addendum}\n"


def assistant_has_payload(message: Any) -> bool:
    """True if the assistant turn has text and/or tool_calls (tau2 validate())."""
    content = getattr(message, "content", None)
    has_text = content is not None and bool(str(content).strip())
    tool_calls = getattr(message, "tool_calls", None) or []
    return has_text or bool(tool_calls)


def fallback_assistant_message(
    *,
    cost: Any = None,
    usage: Any = None,
    text: str = (
        "Sorry, I didn't catch that. Could you please restate what you need help with?"
    ),
) -> Any:
    """Non-empty recovery message when the LLM returns a blank turn."""
    try:
        from tau2.data_model.message import AssistantMessage

        return AssistantMessage.text(text, cost=cost, usage=usage)
    except Exception:
        return SimpleNamespace(content=text, tool_calls=None, cost=cost, usage=usage)


def ensure_assistant_payload(
    produce: Any,
    *,
    max_attempts: int = 3,
    label: str = "agent",
) -> Any:
    """Call ``produce()`` until the assistant message is non-empty, else fallback.

    ``produce`` is a zero-arg callable returning an AssistantMessage-like object.
    """
    last: Any = None
    for attempt in range(1, max_attempts + 1):
        last = produce()
        if assistant_has_payload(last):
            return last
        logger.warning(
            f"[{label}] empty assistant message "
            f"(attempt {attempt}/{max_attempts}); retrying"
        )
    logger.error(f"[{label}] empty assistant after {max_attempts} attempts; using fallback")
    return fallback_assistant_message(
        cost=getattr(last, "cost", None),
        usage=getattr(last, "usage", None),
    )


# Airline passenger schema uses ``dob``; models often emit ``date_of_birth``.
_PASSENGER_DOB_ALIASES = (
    "date_of_birth",
    "dateOfBirth",
    "birth_date",
    "birthdate",
    "birthday",
)
_PASSENGER_WRITE_TOOLS = frozenset(
    {"book_reservation", "update_reservation_passengers"}
)


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        import json

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def canonicalize_passenger_dob_args(
    tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Rename passenger ``date_of_birth`` (etc.) → schema field ``dob``."""
    if tool_name not in _PASSENGER_WRITE_TOOLS:
        return arguments
    passengers = arguments.get("passengers")
    if not isinstance(passengers, list):
        return arguments
    out = dict(arguments)
    fixed: list[Any] = []
    changed = False
    for item in passengers:
        if not isinstance(item, dict):
            fixed.append(item)
            continue
        pax = dict(item)
        dob = pax.get("dob")
        if dob in (None, ""):
            for alias in _PASSENGER_DOB_ALIASES:
                if pax.get(alias) not in (None, ""):
                    pax["dob"] = pax.pop(alias)
                    changed = True
                    break
        for alias in _PASSENGER_DOB_ALIASES:
            if alias in pax:
                pax.pop(alias, None)
                changed = True
        fixed.append(pax)
    if changed:
        out["passengers"] = fixed
    return out


def _set_tool_call_arguments(tc: Any, arguments: dict[str, Any]) -> Any:
    """Return a tool_call-like object with updated arguments (dict, not JSON str)."""
    import json

    # Prefer structured dict; tau2 / litellm accept both.
    payload = arguments
    try:
        # OpenAI-style nested function
        fn = getattr(tc, "function", None)
        if fn is not None and hasattr(fn, "arguments"):
            raw = getattr(fn, "arguments", None)
            new_args = (
                json.dumps(payload, ensure_ascii=False)
                if isinstance(raw, str)
                else payload
            )
            new_fn = (
                fn.model_copy(update={"arguments": new_args})
                if hasattr(fn, "model_copy")
                else SimpleNamespace(
                    **{
                        **(
                            fn.model_dump()
                            if hasattr(fn, "model_dump")
                            else {
                                "name": getattr(fn, "name", None),
                                "arguments": new_args,
                            }
                        ),
                        "arguments": new_args,
                    }
                )
            )
            if hasattr(tc, "model_copy"):
                return tc.model_copy(update={"function": new_fn})
            tc.function = new_fn
            return tc
    except Exception:
        pass

    raw = getattr(tc, "arguments", None)
    new_args = json.dumps(payload, ensure_ascii=False) if isinstance(raw, str) else payload
    if hasattr(tc, "model_copy"):
        try:
            return tc.model_copy(update={"arguments": new_args})
        except Exception:
            pass
    try:
        tc.arguments = new_args
    except Exception:
        pass
    return tc


def canonicalize_assistant_tool_arguments(message: Any) -> Any:
    """Apply schema aliases on tool_call arguments before env execution."""
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return message
    rewritten: list[Any] = []
    any_change = False
    for tc in tool_calls:
        name = str(getattr(tc, "name", None) or "")
        if not name:
            fn = getattr(tc, "function", None)
            name = str(getattr(fn, "name", None) or "")
        raw_args = getattr(tc, "arguments", None)
        if raw_args is None:
            fn = getattr(tc, "function", None)
            raw_args = getattr(fn, "arguments", None) if fn is not None else None
        args = _parse_tool_arguments(raw_args)
        new_args = canonicalize_passenger_dob_args(name, args)
        if new_args != args:
            any_change = True
            rewritten.append(_set_tool_call_arguments(tc, new_args))
        else:
            rewritten.append(tc)
    if not any_change:
        return message
    try:
        return message.model_copy(update={"tool_calls": rewritten})
    except Exception:
        try:
            message.tool_calls = rewritten
        except Exception:
            pass
        return message


def guard_assistant_message(message: Any, *, tools: Sequence[Any]) -> Any:
    """Canonicalize args; strip illegal tool_calls; else user guidance.

    Works with tau2 ``AssistantMessage`` (duck-typed on ``tool_calls`` / ``content``).
    """
    message = canonicalize_assistant_tool_arguments(message)
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return message

    allowed = allowed_tool_names(tools)
    names = []
    for tc in tool_calls:
        name = str(getattr(tc, "name", "") or "")
        if not name:
            fn = getattr(tc, "function", None)
            name = str(getattr(fn, "name", "") or "")
        names.append(name)
    decision = classify_tool_names(names, allowed=allowed)
    if not decision.blocked_names:
        return message

    if decision.guidance_content:
        # All calls illegal → talk to the user instead.
        try:
            from tau2.data_model.message import AssistantMessage

            return AssistantMessage.text(
                decision.guidance_content,
                cost=getattr(message, "cost", None),
                usage=getattr(message, "usage", None),
            )
        except Exception:
            # Fallback duck patch for tests without tau2.
            message.tool_calls = None
            message.content = decision.guidance_content
            return message

    # Mixed: keep only legal calls.
    kept = []
    for tc in tool_calls:
        name = str(getattr(tc, "name", "") or "")
        if not name:
            fn = getattr(tc, "function", None)
            name = str(getattr(fn, "name", "") or "")
        if name in allowed:
            kept.append(tc)
    try:
        return message.model_copy(update={"tool_calls": kept or None})
    except Exception:
        message.tool_calls = kept or None
        return message
