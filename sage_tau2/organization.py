"""τ² organization state: Executor + specialists from nominate/admit."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any
from uuid import uuid4

from sage_tau2.schemas import AgentSpec, OrganizationEdit, OrganizationEditType
from sage_tau2.serialization import read_json, to_primitive, write_json

EXECUTOR_NAME = "Executor"


def default_executor() -> AgentSpec:
    return AgentSpec(
        name=EXECUTOR_NAME,
        role="generalist",
        responsibilities=[
            "Handle any τ² episode end-to-end",
            "Fall back when no specialist matches",
        ],
        assigned_skills=[],
        capability_keys=[],
        tool_permissions=["env_action"],
        activation_condition="default",
        role_specification=(
            "General customer-service executor for the domain policy. "
            "Use tools; obey policy."
        ),
        acting_status="accepted",
        agent_id=str(uuid4()),
        metadata={"builtin": True},
    )


def agent_from_dict(item: dict[str, Any]) -> AgentSpec:
    allowed = {f.name for f in fields(AgentSpec)}
    return AgentSpec(**{k: v for k, v in item.items() if k in allowed})


class Organization:
    def __init__(self, agents: list[AgentSpec] | None = None):
        self.agents: list[AgentSpec] = list(agents or [default_executor()])
        if not any(a.name == EXECUTOR_NAME for a in self.agents):
            self.agents.insert(0, default_executor())

    def executor(self) -> AgentSpec:
        for agent in self.agents:
            if agent.name == EXECUTOR_NAME:
                return agent
        return default_executor()

    def specialists(self) -> list[AgentSpec]:
        return [
            a
            for a in self.agents
            if a.name != EXECUTOR_NAME and a.acting_status != "dormant"
        ]

    def has_carrier(self, capability_key: str) -> bool:
        key = str(capability_key or "").strip().lower()
        if not key:
            return False
        for agent in self.specialists():
            caps = {str(c).strip().lower() for c in (agent.capability_keys or [])}
            if key in caps:
                return True
        return False

    def apply_edit(self, edit: OrganizationEdit) -> AgentSpec | None:
        if edit.edit_type == OrganizationEditType.DO_NOTHING:
            return None
        if edit.edit_type == OrganizationEditType.ADD_AGENT:
            if edit.new_agent is None:
                raise ValueError("ADD_AGENT requires new_agent")
            # Replace same-name specialist if re-admitted.
            self.agents = [
                a for a in self.agents if a.name != edit.new_agent.name
            ]
            self.agents.append(edit.new_agent)
            return edit.new_agent
        if edit.edit_type == OrganizationEditType.ASSIGN_SKILL:
            target = edit.target_agent
            for agent in self.agents:
                if agent.name == target:
                    for name in edit.assigned_skill_names:
                        if name not in agent.assigned_skills:
                            agent.assigned_skills.append(name)
                    if edit.capability_key and edit.capability_key not in (
                        agent.capability_keys or []
                    ):
                        agent.capability_keys.append(edit.capability_key)
                    return agent
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"agents": [to_primitive(a) for a in self.agents]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | list[Any]) -> "Organization":
        if isinstance(payload, list):
            agents = [agent_from_dict(x) for x in payload if isinstance(x, dict)]
            return cls(agents)
        agents = [
            agent_from_dict(x)
            for x in (payload.get("agents") or [])
            if isinstance(x, dict)
        ]
        return cls(agents)

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "Organization":
        p = Path(path)
        if not p.exists():
            org = cls()
            org.save(p)
            return org
        return cls.from_dict(read_json(p))


def format_organization_for_prompt(org: Organization) -> str:
    """Always emit a non-empty org roster (Executor + specialists or none)."""
    specialists = org.specialists()
    lines = ["<organization>"]
    exe = org.executor()
    lines.append(
        f"Executor: {exe.name} (default generalist; handles unmatched episodes)"
    )
    if specialists:
        lines.append(
            "Active specialists (prefer matching one when the user goal fits):"
        )
        for agent in specialists:
            caps = ", ".join(agent.capability_keys) or "n/a"
            skills = ", ".join(agent.assigned_skills[:6]) or "n/a"
            lines.append(
                f"- {agent.name} [{agent.acting_status}] caps=[{caps}] "
                f"skills=[{skills}]"
            )
            if agent.role_specification:
                lines.append(f"  role: {agent.role_specification[:240]}")
            if agent.activation_condition:
                lines.append(f"  activate_when: {agent.activation_condition}")
        lines.append(
            "If a specialist matches, follow its assigned skill protocols. "
            "Otherwise act as Executor."
        )
    else:
        lines.append("Specialists: (none yet)")
    lines.append("</organization>")
    return "\n".join(lines)
