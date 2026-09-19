"""τ² skill / trajectory schemas (independent of sage_mas.schemas)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class SkillStatus(str, Enum):
    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    VERIFIED = "verified"
    # Birth/credit score would verify, but support_count is below the strong bar.
    VERIFIED_LOW_SUPPORT = "verified_low_support"
    REJECTED = "rejected"
    RETIRED = "retired"


@dataclass(slots=True)
class ToolCallStep:
    """One agent tool invocation extracted from a τ² message."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None
    result_content: str | None = None
    result_error: bool = False


@dataclass(slots=True)
class Tau2Trajectory:
    """One simulated episode, normalized for distill / credit."""

    task_id: str
    trial: int
    domain: str
    reward: float
    db_reward: float | None
    communicate_reward: float | None
    db_match: bool | None
    termination_reason: str | None
    tool_protocol: list[str]
    tool_steps: list[ToolCallStep]
    assistant_texts: list[str]
    user_texts: list[str]
    evidence_id: str = field(default_factory=lambda: str(uuid4()))
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        if self.db_match is True:
            return True
        return float(self.reward or 0.0) >= 1.0

    @property
    def has_positive_db_effect(self) -> bool:
        return self.db_match is True or (
            self.db_reward is not None and float(self.db_reward) >= 1.0
        )


@dataclass(slots=True)
class Tau2Skill:
    """Tool-protocol skill for τ² domains."""

    skill_name: str
    description: str
    precondition: str
    action_protocol: list[str]
    expected_effect: str
    capability_key: str = ""
    skill_id: str = field(default_factory=lambda: str(uuid4()))
    status: SkillStatus = SkillStatus.CANDIDATE
    support_count: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    domain: str = "airline"
    metadata: dict[str, Any] = field(default_factory=dict)


class OrganizationEditType(str, Enum):
    DO_NOTHING = "do_nothing"
    ASSIGN_SKILL = "assign_skill"
    ADD_AGENT = "add_agent"


@dataclass(slots=True)
class AgentSpec:
    """Specialist carrier produced by τ² nominate/admit."""

    name: str
    role: str
    responsibilities: list[str]
    assigned_skills: list[str] = field(default_factory=list)
    capability_keys: list[str] = field(default_factory=list)
    tool_permissions: list[str] = field(default_factory=list)
    activation_condition: str | None = None
    role_specification: str = ""
    acting_status: str = "probation"  # probation | accepted | dormant
    shadow_evaluation_record: dict[str, Any] = field(default_factory=dict)
    agent_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrganizationEdit:
    edit_type: OrganizationEditType
    rationale: str
    target_agent: str | None = None
    new_agent: AgentSpec | None = None
    assigned_skill_names: list[str] = field(default_factory=list)
    status: str = "candidate"
    cluster_id: str | None = None
    capability_key: str | None = None
