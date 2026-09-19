"""Enhanced meta-agent dispatcher with profile-based agent selection.

Current issue: Runtime selection is based on skill name matching, not semantic
understanding of agent capabilities vs task requirements.

Improvement: Build agent profiles from their skills and match against task semantics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sage_mas.schemas import AgentSpec, Skill


@dataclass(slots=True)
class AgentProfile:
    """Rich profile for agent capability description."""
    agent_name: str
    role: str

    # Capability inventory
    task_families: set[str] = field(default_factory=set)
    operations: set[str] = field(default_factory=set)  # cool, heat, clean, etc.
    object_types: set[str] = field(default_factory=set)  # object categories they handle

    # Performance metrics
    success_rate: float = 0.0
    total_dispatches: int = 0
    total_wins: int = 0

    # Skill coverage
    skill_count: int = 0
    verified_skill_count: int = 0
    skill_names: list[str] = field(default_factory=list)

    # Capability description (for LLM-based dispatch)
    capability_summary: str = ""

    # Dispatch eligibility
    can_dispatch: bool = True
    dispatch_only: bool = False  # Can only be explicitly dispatched, not auto-selected


@dataclass(slots=True)
class TaskContext:
    """Parsed task context for agent matching."""
    task_text: str
    task_family: str | None = None
    operation: str | None = None  # cool/heat/clean/use
    target_object: str | None = None
    destination: str | None = None

    # Semantic keywords
    keywords: set[str] = field(default_factory=set)
    verbs: set[str] = field(default_factory=set)


class MetaAgentDispatcher:
    """
    Enhanced meta-agent that selects specialists based on semantic matching
    between agent profiles and task requirements.
    """

    def __init__(
        self,
        agents: list[AgentSpec],
        skills: list[Skill],
        *,
        enable_llm_dispatch: bool = False,
        llm_backend: Any = None,
        operation_keywords: list[str] | None = None,
        verb_patterns: list[str] | None = None,
        stop_words: set[str] | None = None,
    ):
        self.agents = agents
        self.skills = skills
        self.skill_by_name = {skill.skill_name: skill for skill in skills}

        # Configurable keyword patterns
        self.operation_keywords = operation_keywords or ["cool", "heat", "clean", "use"]
        self.verb_patterns = verb_patterns or [
            r"\b(put|place|move)\b",
            r"\b(cool|heat|clean|wash)\b",
            r"\b(pick|take|get|grab)\b",
            r"\b(use|toggle|turn)\b",
            r"\b(look|examine|find)\b",
        ]
        self.stop_words = stop_words or {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "some",
        }

        # Build agent profiles
        self.profiles = self._build_agent_profiles(agents, skills)

        # Find executor (fallback agent)
        self.executor = self._find_executor(agents)

        self.enable_llm_dispatch = enable_llm_dispatch
        self.llm_backend = llm_backend

    def select_agent(
        self,
        *,
        task: str,
        task_family: str | None = None,
        gamefile: str | None = None,
        observation: str | None = None,
        history_steps: list[Any] | None = None,
        active_skill_names: set[str] | None = None,
    ) -> tuple[AgentSpec, str]:
        """
        Select the best agent for the given task.

        Returns:
            (selected_agent, reason)
        """
        # Parse task context
        task_ctx = self._parse_task_context(
            task=task,
            task_family=task_family,
            gamefile=gamefile,
        )

        # Strategy 1: Active skill-based dispatch (current mechanism)
        if active_skill_names:
            agent, reason = self._select_by_active_skills(
                active_skill_names,
                task_ctx,
            )
            if agent is not None:
                return agent, reason

        # Strategy 2: Profile-based semantic matching
        agent, reason = self._select_by_profile_match(task_ctx)
        if agent is not None:
            return agent, reason

        # Strategy 3: LLM-based dispatch (optional, expensive)
        if self.enable_llm_dispatch and self.llm_backend:
            agent, reason = self._select_by_llm(
                task_ctx,
                observation=observation,
                history_steps=history_steps,
            )
            if agent is not None:
                return agent, reason

        # Fallback: executor
        return self.executor, "No specialist matched; using executor fallback"

    def _build_agent_profiles(
        self,
        agents: list[AgentSpec],
        skills: list[Skill],
    ) -> dict[str, AgentProfile]:
        """Build rich profiles for each agent from their assigned skills."""
        profiles: dict[str, AgentProfile] = {}

        for agent in agents:
            profile = AgentProfile(
                agent_name=agent.name,
                role=agent.role,
            )

            # Gather skills
            agent_skills = [
                self.skill_by_name[skill_name]
                for skill_name in agent.assigned_skills
                if skill_name in self.skill_by_name
            ]

            profile.skill_count = len(agent_skills)
            profile.skill_names = [s.skill_name for s in agent_skills]

            # Extract capabilities from skills
            for skill in agent_skills:
                # Task families
                if skill.applicable_task_families:
                    profile.task_families.update(skill.applicable_task_families)

                # Operations from protocol
                protocol_text = " ".join(skill.action_protocol or []).lower()
                for operation in self.operation_keywords:
                    if operation in protocol_text:
                        profile.operations.add(operation)

                # Count verified skills
                if skill.status.value == "verified":
                    profile.verified_skill_count += 1

            # Performance metrics from shadow evaluation
            shadow_record = agent.shadow_evaluation_record or {}
            if "success_rate" in shadow_record:
                profile.success_rate = float(shadow_record["success_rate"])
            if "total_dispatches" in shadow_record:
                profile.total_dispatches = int(shadow_record["total_dispatches"])
            if "total_wins" in shadow_record:
                profile.total_wins = int(shadow_record["total_wins"])

            # Dispatch eligibility
            acting_status = str(shadow_record.get("acting_status", "")).lower()
            profile.can_dispatch = acting_status not in {"demoted", "dormant"}
            profile.dispatch_only = bool(shadow_record.get("dispatch_only", False))

            # Build capability summary
            profile.capability_summary = self._build_capability_summary(
                agent,
                profile,
                agent_skills,
            )

            profiles[agent.name] = profile

        return profiles

    def _build_capability_summary(
        self,
        agent: AgentSpec,
        profile: AgentProfile,
        skills: list[Skill],
    ) -> str:
        """Generate human-readable capability summary for LLM dispatch."""
        lines = [f"{agent.name} ({agent.role}):"]

        if profile.task_families:
            families_str = ", ".join(sorted(profile.task_families))
            lines.append(f"  Task families: {families_str}")

        if profile.operations:
            ops_str = ", ".join(sorted(profile.operations))
            lines.append(f"  Operations: {ops_str}")

        if profile.verified_skill_count > 0:
            lines.append(
                f"  Skills: {profile.verified_skill_count} verified / "
                f"{profile.skill_count} total"
            )

        if profile.total_dispatches > 0:
            lines.append(
                f"  Track record: {profile.total_wins}/{profile.total_dispatches} "
                f"wins ({profile.success_rate:.0%})"
            )

        # Add top skills
        if skills:
            top_skills = skills[:3]
            skill_bullets = [f"    - {s.skill_name}" for s in top_skills]
            lines.append("  Key skills:")
            lines.extend(skill_bullets)

        return "\n".join(lines)

    def _parse_task_context(
        self,
        *,
        task: str,
        task_family: str | None,
        gamefile: str | None,
    ) -> TaskContext:
        """Parse task into structured context for matching."""
        from sage_mas.task_parser import parse_alfworld_task

        parsed = parse_alfworld_task(task, gamefile or "")

        # Extract keywords and verbs
        task_lower = task.lower()
        keywords = set(task_lower.split())

        # Remove stop words
        keywords -= self.stop_words

        # Extract verbs using configured patterns
        verbs = set()
        for pattern in self.verb_patterns:
            matches = re.findall(pattern, task_lower)
            verbs.update(matches)

        return TaskContext(
            task_text=task,
            task_family=task_family or parsed.task_family,
            operation=parsed.operation,
            target_object=parsed.target,
            destination=parsed.destination,
            keywords=keywords,
            verbs=verbs,
        )

    def _select_by_active_skills(
        self,
        active_skill_names: set[str],
        task_ctx: TaskContext,
    ) -> tuple[AgentSpec | None, str]:
        """Select agent by active skill names (current mechanism)."""
        if not active_skill_names:
            return None, ""

        # Find agents with most overlapping active skills
        candidates: list[tuple[AgentSpec, int, AgentProfile]] = []

        for agent in self.agents:
            if agent.agent_id == self.executor.agent_id:
                continue  # Skip executor in specialist selection

            profile = self.profiles.get(agent.name)
            if not profile or not profile.can_dispatch:
                continue

            overlap = len(set(agent.assigned_skills) & active_skill_names)
            if overlap > 0:
                candidates.append((agent, overlap, profile))

        if not candidates:
            return None, ""

        # Sort by: overlap count (desc), success rate (desc), name (asc)
        candidates.sort(
            key=lambda item: (-item[1], -item[2].success_rate, item[0].name)
        )

        selected, overlap, profile = candidates[0]
        reason = (
            f"{selected.name} selected: {overlap} active skills matched, "
            f"SR={profile.success_rate:.0%}"
        )

        return selected, reason

    def _select_by_profile_match(
        self,
        task_ctx: TaskContext,
    ) -> tuple[AgentSpec | None, str]:
        """Select agent by semantic profile matching."""
        # Score each agent by profile-task alignment
        candidates: list[tuple[AgentSpec, float, str]] = []

        for agent in self.agents:
            if agent.agent_id == self.executor.agent_id:
                continue

            profile = self.profiles.get(agent.name)
            if not profile or not profile.can_dispatch:
                continue

            # Cannot auto-select dispatch-only agents
            if profile.dispatch_only:
                continue

            score, reason = self._score_agent_match(profile, task_ctx)
            if score > 0:
                candidates.append((agent, score, reason))

        if not candidates:
            return None, ""

        # Sort by score (desc), success rate (desc)
        candidates.sort(
            key=lambda item: (-item[1], -self.profiles[item[0].name].success_rate)
        )

        selected, score, reason = candidates[0]
        return selected, f"{selected.name} (match_score={score:.2f}): {reason}"

    def _score_agent_match(
        self,
        profile: AgentProfile,
        task_ctx: TaskContext,
    ) -> tuple[float, str]:
        """
        Score how well an agent profile matches task requirements.

        Returns:
            (score, reason_string)
        """
        score = 0.0
        reasons: list[str] = []

        # Task family match (strongest signal)
        if task_ctx.task_family and task_ctx.task_family in profile.task_families:
            score += 10.0
            reasons.append(f"family={task_ctx.task_family}")

        # Operation match (strong signal for transform tasks)
        if task_ctx.operation and task_ctx.operation in profile.operations:
            score += 5.0
            reasons.append(f"op={task_ctx.operation}")

        # Keyword overlap (weak signal)
        keyword_overlap = task_ctx.keywords & profile.task_families
        if keyword_overlap:
            score += len(keyword_overlap) * 0.5

        # Performance bonus (verified skills)
        if profile.verified_skill_count > 0:
            score += profile.verified_skill_count * 0.2

        # Success rate bonus
        if profile.total_dispatches >= 3:
            score += profile.success_rate * 2.0
            reasons.append(f"SR={profile.success_rate:.0%}")

        reason_str = ", ".join(reasons) if reasons else "general match"
        return score, reason_str

    def _select_by_llm(
        self,
        task_ctx: TaskContext,
        *,
        observation: str | None,
        history_steps: list[Any] | None,
    ) -> tuple[AgentSpec | None, str]:
        """Use LLM to select agent (expensive, only when enabled)."""
        if not self.llm_backend:
            return None, ""

        # Build roster prompt with agent profiles
        roster_lines = ["Available specialist agents:"]
        eligible_agents = [
            (agent, self.profiles[agent.name])
            for agent in self.agents
            if agent.agent_id != self.executor.agent_id
            and self.profiles.get(agent.name)
            and self.profiles[agent.name].can_dispatch
        ]

        if not eligible_agents:
            return None, ""

        for agent, profile in eligible_agents:
            roster_lines.append(f"\n{profile.capability_summary}")

        roster = "\n".join(roster_lines)

        # Ask LLM to select
        system_prompt = (
            "You are a meta-agent dispatcher. Select the best specialist agent "
            "for the given task based on their capabilities and track record. "
            "Respond with ONLY the agent name, nothing else."
        )

        user_prompt = (
            f"{roster}\n\n"
            f"Task: {task_ctx.task_text}\n"
            f"Task family: {task_ctx.task_family or 'unknown'}\n\n"
            "Which agent should handle this task? Reply with agent name only."
        )

        try:
            result = self.llm_backend.complete(system_prompt, user_prompt)
            selected_name = result.content.strip()

            # Find the agent by name
            for agent, _ in eligible_agents:
                if agent.name.lower() in selected_name.lower():
                    return agent, f"LLM selected {agent.name}"
        except Exception as e:
            import logging
            logging.warning(f"LLM dispatch failed: {e}")

        return None, ""

    def _find_executor(self, agents: list[AgentSpec]) -> AgentSpec:
        """Find the executor (fallback) agent."""
        for agent in agents:
            if "executor" in f"{agent.name} {agent.role}".lower():
                return agent
        return agents[0] if agents else None

    def get_roster_summary(self) -> str:
        """Get a human-readable summary of all agent profiles."""
        lines = ["Agent Roster Summary:", "=" * 60]

        for profile in sorted(
            self.profiles.values(),
            key=lambda p: (-p.verified_skill_count, p.agent_name),
        ):
            lines.append(f"\n{profile.capability_summary}")
            if not profile.can_dispatch:
                lines.append("  Status: INACTIVE (demoted/dormant)")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
