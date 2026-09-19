"""Separate τ² agent-callable tools from user-device actions (esp. telecom).

Telecom policy documents phone diagnostics/fixes with function-like names
(``toggle_airplane_mode``, ``check_network_status``, …). Those APIs belong to
the **user simulator**, not the agent toolkit. Agents that call them get
``Tool '…' not found``; distill must not promote those calls into skill
protocols.
"""

from __future__ import annotations

import re

# Backend tools registered on the telecom agent environment.
TELECOM_AGENT_TOOLS: frozenset[str] = frozenset(
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
)

# User-simulator device / payment actions (not agent-callable).
TELECOM_USER_TOOLS: frozenset[str] = frozenset(
    {
        "can_send_mms",
        "check_apn_settings",
        "check_app_permissions",
        "check_app_status",
        "check_data_restriction_status",
        "check_installed_apps",
        "check_network_mode_preference",
        "check_network_status",
        "check_payment_request",
        "check_sim_status",
        "check_status_bar",
        "check_vpn_status",
        "check_wifi_calling_status",
        "check_wifi_status",
        "connect_vpn",
        "disconnect_vpn",
        "grant_app_permission",
        "make_payment",
        "reboot_device",
        "reseat_sim_card",
        "reset_apn_settings",
        "run_speed_test",
        "set_apn_settings",
        "set_network_mode_preference",
        "toggle_airplane_mode",
        "toggle_data",
        "toggle_data_saver_mode",
        "toggle_roaming",
        "toggle_wifi",
        "toggle_wifi_calling",
    }
)

_TOOL_NOT_FOUND_RE = re.compile(
    r"Tool\s+'[^']+'\s+not found",
    re.IGNORECASE,
)


def tool_name_from_step(step: str) -> str:
    return str(step or "").split("(", 1)[0].strip()


def is_telecom_user_side_tool(name: str) -> bool:
    return str(name or "").strip() in TELECOM_USER_TOOLS


def is_tool_not_found_result(content: str | None) -> bool:
    if not content:
        return False
    return bool(_TOOL_NOT_FOUND_RE.search(str(content)))


def strip_user_side_protocol_steps(
    protocol: list[str],
    *,
    domain: str | None = None,
) -> list[str]:
    """Drop user-device steps from an agent tool protocol.

    When ``domain`` is unset or ``telecom``, always strip the known telecom
    user-tool set (safe for mixed banks). Other domains currently have no
    parallel dual-toolkit trap, so the list is left unchanged.

    Dialogue / conditional branch rows (confirm, if-bug, instruct user, …)
    are kept — they are not agent tools and must remain visible to injectors.
    """
    if domain is not None and str(domain).lower() not in {"", "telecom", "telecom-workflow"}:
        return list(protocol or [])
    out: list[str] = []
    for step in protocol or []:
        text = str(step or "").strip()
        if not text:
            continue
        low = text.lower()
        if low.startswith(
            (
                "if ",
                "confirm ",
                "resolve ",
                "extract ",
                "communicate ",
                "guide ",
                "instruct ",
                "ask ",
                "diagnose ",
                "mark ",
            )
        ):
            out.append(step)
            continue
        name = tool_name_from_step(step)
        if not name or is_telecom_user_side_tool(name):
            continue
        out.append(step)
    return out
