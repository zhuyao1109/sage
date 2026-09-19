"""Operation-protocol finalize helpers and seed-role locking."""

from __future__ import annotations

from sage_mas.schemas import AtomicStep, Skill
from sage_mas.skill_distiller import (
    OPERATION_SIGNALS,
    SUCCESS_OPERATION_SIGNALS,
)

OPERATION_PROTOCOL_MARKERS = {
    "missing_clean_operation": ("clean",),
    "missing_heat_operation": ("heat",),
    "missing_cool_operation": ("cool",),
    "missing_light_operation": ("desklamp", "use desklamp"),
    "clean_operation_success": ("clean",),
    "heat_operation_success": ("heat",),
    "cool_operation_success": ("cool",),
    "light_operation_success": ("desklamp", "use desklamp"),
}


def finalize_operation_protocol(seed: Skill, protocol: list[str]) -> list[str]:
    """Ensure operation skills retain the required transformation step.

    If the LLM drops clean/heat/cool/desklamp, discard its protocol and keep the
    seed template which already encodes the required operation order.
    """
    signal = str(seed.metadata.get("source_signal", "")).strip()
    markers = OPERATION_PROTOCOL_MARKERS.get(signal)
    if not markers:
        return [step.strip() for step in protocol if str(step).strip()]

    merged = [step.strip() for step in protocol if str(step).strip()]
    has_operation = any(
        any(marker in step.lower() for marker in markers)
        for step in merged
    )
    if has_operation:
        return merged
    # Never accept an operation skill protocol that omits the transformation.
    return list(seed.action_protocol) or merged


def prefer_success_trajectories(
    trajectories: list[list[AtomicStep]],
) -> list[list[AtomicStep]]:
    """Sort evidence so winning trajectories are summarized first."""
    return sorted(
        trajectories,
        key=lambda steps: (
            0 if steps and bool(steps[-1].metadata.get("won", False)) else 1,
            str(steps[-1].metadata.get("trajectory_id", "")) if steps else "",
        ),
    )


def lock_operation_suggested_role(seed: Skill, suggested_role: str) -> str:
    signal = str(seed.metadata.get("source_signal", "")).strip()
    if (
        signal in OPERATION_SIGNALS or signal in SUCCESS_OPERATION_SIGNALS
    ) and seed.suggested_role:
        return seed.suggested_role
    return suggested_role or seed.suggested_role or "Executor"
