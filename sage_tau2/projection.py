"""GiGPO-style think/action validity checks for τ² assistant turns.

Mirrors ``agent_system/.../alfworld/projection.py``: require ``<think>`` and a
concrete action (native tool call(s) and/or visible text after think).
"""

from __future__ import annotations

from typing import Any

from sage_tau2.pro_steps import action_from_assistant_message, parse_think_and_visible


def _message_to_dict(msg: Any) -> dict[str, Any]:
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "dict"):
        return msg.dict()
    return {
        "role": getattr(msg, "role", None),
        "content": getattr(msg, "content", None),
        "tool_calls": getattr(msg, "tool_calls", None),
    }


def validate_think_action(message: Any) -> tuple[bool, str]:
    """Return ``(ok, reason)``. ``reason`` is empty when ok.

    Valid iff:
      - content contains non-empty ``<think>...</think>``
      - and there is either ≥1 tool_call OR non-empty visible text after think
    """
    msg = _message_to_dict(message)
    content = msg.get("content")
    think, visible = parse_think_and_visible(content)
    if not think:
        return False, "missing_think"
    tool_calls = msg.get("tool_calls") or []
    has_tools = bool(tool_calls)
    has_text = bool(str(visible or "").strip())
    # Also accept structured action extraction (tool string / visible).
    action = action_from_assistant_message(msg)
    if not has_tools and not has_text and not action:
        return False, "missing_action"
    if has_tools and has_text:
        # Soft warning only — τ² policy prefers XOR, but both can be valid
        # enough for trajectory logging; do not fail validity.
        pass
    return True, ""


def tau2_projection(message: Any) -> tuple[bool, str | None, str | None]:
    """Projection-style helper: ``(valid, think, action_str)``."""
    msg = _message_to_dict(message)
    think, _visible = parse_think_and_visible(msg.get("content"))
    action = action_from_assistant_message(msg)
    ok, _reason = validate_think_action(msg)
    return ok, think, action
