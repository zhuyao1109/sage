"""Skill-Agent Assignment Gap estimation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from sage_mas.schemas import AgentSpec, Skill


@dataclass(slots=True)
class AssignmentGapResult:
    skill_ids: list[str]
    assignment_gap: float
    best_agent_id: str | None
    best_agent_name: str | None
    best_capacity: float
    capacities: dict[str, float]
    empirical_coverage: float = 0.0


class AssignmentGapEstimator:
    """Estimate ΔA(N, A) = 1 - max Cap(ai, N).

    The default role scorer is lexical and auditable. A semantic scorer can be
    injected later without changing the organization-evolution API.
    """

    def __init__(
        self,
        role_weight: float = 0.7,
        tool_weight: float = 0.3,
        role_scorer: Callable[[AgentSpec, Skill], float] | None = None,
        empirical_weight: float = 0.7,
    ):
        total = role_weight + tool_weight
        if total <= 0:
            raise ValueError("role_weight + tool_weight must be positive")
        self.role_weight = role_weight / total
        self.tool_weight = tool_weight / total
        self.role_scorer = role_scorer or self._lexical_role_match
        self.empirical_weight = max(
            0.0,
            min(1.0, empirical_weight),
        )

    def estimate(
        self,
        skills: list[Skill],
        agents: list[AgentSpec],
        empirical_capacities: dict[tuple[str, str], float] | None = None,
    ) -> AssignmentGapResult:
        if not skills:
            raise ValueError("At least one skill is required for gap estimation")
        if not agents:
            return AssignmentGapResult(
                skill_ids=[skill.skill_id for skill in skills],
                assignment_gap=1.0,
                best_agent_id=None,
                best_agent_name=None,
                best_capacity=0.0,
                capacities={},
                empirical_coverage=0.0,
            )

        empirical_capacities = empirical_capacities or {}
        empirical_count = 0
        total_pairs = len(agents) * len(skills)

        def capacity(agent: AgentSpec, skill: Skill) -> float:
            nonlocal empirical_count
            lexical_capacity = self._capacity(agent, skill)
            empirical = empirical_capacities.get(
                (agent.agent_id, skill.skill_id)
            )
            if empirical is None:
                return lexical_capacity
            empirical_count += 1
            empirical = max(0.0, min(1.0, empirical))
            return (
                self.empirical_weight * empirical
                + (1.0 - self.empirical_weight) * lexical_capacity
            )

        capacities = {
            agent.agent_id: sum(
                capacity(agent, skill) for skill in skills
            )
            / len(skills)
            for agent in agents
        }
        best_agent = max(agents, key=lambda agent: capacities[agent.agent_id])
        best_capacity = capacities[best_agent.agent_id]
        return AssignmentGapResult(
            skill_ids=[skill.skill_id for skill in skills],
            assignment_gap=max(0.0, min(1.0, 1.0 - best_capacity)),
            best_agent_id=best_agent.agent_id,
            best_agent_name=best_agent.name,
            best_capacity=best_capacity,
            capacities=capacities,
            empirical_coverage=(
                empirical_count / total_pairs if total_pairs else 0.0
            ),
        )

    def _capacity(self, agent: AgentSpec, skill: Skill) -> float:
        role_match = self.role_scorer(agent, skill)
        tool_match = self._tool_match(agent, skill)
        return self.role_weight * role_match + self.tool_weight * tool_match

    @classmethod
    def _lexical_role_match(cls, agent: AgentSpec, skill: Skill) -> float:
        role_tokens = cls._tokens(
            " ".join([agent.name, agent.role, *agent.responsibilities])
        )
        skill_tokens = cls._tokens(
            " ".join(
                filter(
                    None,
                    [
                        skill.skill_name,
                        skill.description,
                        skill.suggested_role,
                        *skill.target_failure_types,
                    ],
                )
            )
        )
        if not skill_tokens:
            return 0.0
        overlap = len(role_tokens & skill_tokens) / len(skill_tokens)
        suggested_match = (
            1.0
            if skill.suggested_role
            and skill.suggested_role.lower() in f"{agent.name} {agent.role}".lower()
            else 0.0
        )
        assigned_match = 1.0 if skill.skill_name in agent.assigned_skills else 0.0
        return min(1.0, 0.45 * overlap + 0.4 * suggested_match + 0.15 * assigned_match)

    @staticmethod
    def _tool_match(agent: AgentSpec, skill: Skill) -> float:
        required_tools = set(skill.metadata.get("required_tools", []))
        if not required_tools:
            return 1.0
        available_tools = set(agent.tool_permissions)
        return len(required_tools & available_tools) / len(required_tools)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", text.lower()))
