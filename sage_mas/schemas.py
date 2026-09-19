"""Shared data structures for the lightweight SAGE-MAS prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class AtomicOp(str, Enum):
    PLAN = "Plan"
    ACT = "Act"
    OBSERVE = "Observe"
    COMMUNICATE = "Communicate"
    SELECT = "Select"
    VERIFY = "Verify"
    TERMINATE = "Terminate"


class SkillStatus(str, Enum):
    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    VERIFIED = "verified"
    REJECTED = "rejected"
    RETIRED = "retired"


class OrganizationEditType(str, Enum):
    DO_NOTHING = "do_nothing"
    ASSIGN_SKILL = "assign_skill"
    ADD_AGENT = "add_agent"


@dataclass(slots=True)
class AtomicStep:
    node_id: str
    atomic_op: AtomicOp
    agent: str
    observation: str | None = None
    action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StateActionFragment:
    trajectory_id: str
    step_index: int
    observation: str
    action: str
    won: bool | None = None
    task_family: str | None = None


@dataclass(slots=True)
class ExplorationDistribution:
    task_family_histogram: dict[str, int] = field(default_factory=dict)
    outcome_histogram: dict[str, int] = field(default_factory=dict)
    signal_histogram: dict[str, int] = field(default_factory=dict)
    signal_cooccurrence: dict[str, int] = field(default_factory=dict)
    total_trajectories: int = 0
    success_rate: float = 0.0


@dataclass(slots=True)
class DistributionShift:
    kl_divergence: float = 0.0
    shift_score: float = 0.0
    task_family_delta: dict[str, float] = field(default_factory=dict)
    baseline_source: str = "skill_bank"
    # Embedding-space novelty diagnostics. ``shift_score`` is the nearest
    # historical-skill distance when metric == "cosine_nearest_skill".
    metric: str = "family_histogram_kl"
    nearest_skill_id: str | None = None
    nearest_skill_name: str | None = None
    nearest_similarity: float | None = None
    novelty_score: float = 0.0
    baseline_size: int = 0
    failure_rate: float = 0.0


@dataclass(slots=True)
class ExperienceDistributionSnapshot:
    round_id: str
    representation_version: str
    prototype_count: int
    total_support: int
    task_family_histogram: dict[str, int] = field(default_factory=dict)
    outcome_histogram: dict[str, int] = field(default_factory=dict)
    capability_histogram: dict[str, int] = field(default_factory=dict)
    status_histogram: dict[str, int] = field(default_factory=dict)
    centroid: list[float] = field(default_factory=list)
    novelty_scores: list[float] = field(default_factory=list)
    mean_novelty: float = 0.0
    novelty_p50: float = 0.0
    novelty_p90: float = 0.0
    novel_mass: float = 0.0
    failure_novel_mass: float = 0.0
    novelty_threshold: float = 0.15
    historical_prototype_count: int = 0
    success_rate: float = 0.0
    prototypes: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class CapabilityContract:
    capability_id: str
    name: str
    description: str
    preconditions: list[str]
    action_protocols: list[list[str]]
    expected_effects: list[str]
    atomic_ops: list[AtomicOp]
    required_tools: list[str]
    skill_ids: list[str]
    skill_names: list[str]
    evidence_ids: list[str]
    task_families: list[str]
    marginal_utility: float | None = None
    reliability: float = 1.0


@dataclass(slots=True)
class Skill:
    skill_name: str
    description: str
    precondition: str
    action_protocol: list[str]
    applicable_atomic_ops: list[AtomicOp]
    expected_effect: str | None = None
    suggested_role: str | None = None
    target_failure_types: list[str] = field(default_factory=list)
    skill_id: str = field(default_factory=lambda: str(uuid4()))
    status: SkillStatus = SkillStatus.CANDIDATE
    marginal_utility: float | None = None
    evidence_ids: list[str] = field(default_factory=list)
    support_count: int = 0
    trajectory_summary: str | None = None
    key_fragments: list[StateActionFragment] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    exploration_distribution: ExplorationDistribution | None = None
    distribution_shift: DistributionShift | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Stable semantic identity. Task families are applicability scope rather
    # than part of the identity so one reusable capability is not duplicated
    # for every ALFWorld family.
    capability_key: str = ""
    applicable_task_families: list[str] = field(default_factory=list)
    # Verification history drives reliability and eventual retirement.
    utility_history: list[float] = field(default_factory=list)
    reliability_weight: float = 1.0
    consecutive_negative_evaluations: int = 0


@dataclass(slots=True)
class AgentSpec:
    name: str
    role: str
    responsibilities: list[str]
    assigned_skills: list[str] = field(default_factory=list)
    tool_permissions: list[str] = field(default_factory=list)
    activation_condition: str | None = None
    token_budget: int | None = None
    turn_budget: int | None = None
    # Fuller role instance attributes (filled for every ADD_AGENT).
    role_specification: str = ""
    responsibility_boundary: str = ""
    input_protocol: str = ""
    output_protocol: str = ""
    communication_edges: list[str] = field(default_factory=list)
    retirement_condition: str | None = None
    shadow_evaluation_record: dict[str, Any] = field(default_factory=dict)
    agent_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        # YAML `assigned_skills:` with no value parses as None.
        if self.assigned_skills is None:
            self.assigned_skills = []
        if self.tool_permissions is None:
            self.tool_permissions = []
        if self.responsibilities is None:
            self.responsibilities = []
        if self.communication_edges is None:
            self.communication_edges = []
        if self.shadow_evaluation_record is None:
            self.shadow_evaluation_record = {}


@dataclass(slots=True)
class OrganizationEdit:
    edit_type: OrganizationEditType
    rationale: str
    target_agent: str | None = None
    new_agent: AgentSpec | None = None
    assigned_skill_names: list[str] = field(default_factory=list)
    assignment_gap: float | None = None
    status: str = "candidate"


@dataclass(slots=True)
class ShadowMetrics:
    success_rate: float
    token_cost: float
    active_agent_count: float = 1.0
    protocol_adherence: float = 1.0
    # Fraction of in-contract shadow tasks where the new agent was primary.
    new_agent_call_rate: float | None = None


@dataclass(slots=True)
class ShadowDecision:
    accepted: bool
    old_utility: float
    new_utility: float
    relative_gain: float
    reason: str
    paired_delta_ci_low: float | None = None
    paired_delta_ci_high: float | None = None
    paired_task_count: int = 0
    new_agent_call_rate: float | None = None
