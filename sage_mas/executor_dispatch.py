"""Executor-mediated task assignment: eligibility → LLM dispatch → evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sage_mas.capability_contract import sync_agent_skill_scopes
from sage_mas.runtime import ChatBackend
from sage_mas.schemas import AgentSpec, Skill, SkillStatus
from sage_mas.specialist_controllers import specialist_has_controller_for_family


_ASSIGN_RE = re.compile(
    r"<\s*assign\s*>\s*([^<>]+?)\s*<\s*/\s*assign\s*>",
    re.IGNORECASE,
)
_ASSIGN_LINE_RE = re.compile(
    r"(?im)^\s*assign(?:ed)?\s*(?:to)?\s*[:\-]\s*(.+?)\s*$",
)


@dataclass(slots=True)
class AgentAssignment:
    """Episode-level ownership for one task."""

    primary_agent: str
    support_agents: list[str] = field(default_factory=list)
    score_by_agent: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    # Layer-1 / Layer-4 evidence for org / onboarding feedback.
    eligible_agents: list[str] = field(default_factory=list)
    eligible_skills_by_agent: dict[str, list[str]] = field(default_factory=dict)
    dispatch_layer: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutorDispatchConfig:
    """How the Executor chooses a primary agent for a new task."""

    enabled: bool = True
    allow_dormant: bool = True
    # Keep False so Executor compares itself with every candidate specialist.
    auto_assign_single: bool = True
    # Only accepted specialists may take primary control. Probation agents stay
    # on the roster (skills still inject via Executor) until an actor-promotion
    # probe proves they are not worse than Executor.
    require_accepted_for_primary: bool = False
    # Main SAGE eligibility uses learned skill/contract scope. Hard controllers
    # may be required only in explicit controller ablations.
    require_controller_for_eligibility: bool = False
    # When a specialist's learned contract matches the episode, assign that
    # specialist (or LLM among matching specialists). Do not keep Executor.
    prefer_matching_specialist: bool = True
    # Probation specialists remain primary-eligible on in-contract episodes
    # until they accumulate this many applicable primary dispatches, even if
    # require_accepted_for_primary or trial_games_remaining would otherwise
    # lock them out. Set 0 to disable the quota override.
    probation_primary_quota: int = 8
    # Optional empirical family gates (from held-out specialist vs Executor ΔSR).
    # Empty enabled list means "no allowlist". Disabled always wins.
    enabled_task_families: list[str] = field(default_factory=list)
    disabled_task_families: list[str] = field(default_factory=list)
    # When True: open roster (all dispatchable specialists are candidates),
    # disable prefer/auto shortcuts, and always ask the Executor LLM to pick
    # among specialists + itself. Used to probe learned LLM routing without
    # contract-match direct assign.
    llm_only_dispatch: bool = False


class ExecutorDispatcher:
    """Three-layer routing: skill eligibility → Executor LLM → sticky primary."""

    def __init__(
        self,
        config: ExecutorDispatchConfig | None = None,
        *,
        backend: ChatBackend | None = None,
    ):
        self.config = config or ExecutorDispatchConfig()
        self.backend = backend

    def assign(
        self,
        *,
        task: str,
        task_family: str,
        agents: list[AgentSpec],
        skills: list[Skill] | None = None,
        executor_name: str | None = None,
        gamefile: str | None = None,
    ) -> AgentAssignment:
        if not agents:
            raise ValueError("ExecutorDispatcher requires at least one agent")
        executor = self._executor(agents, executor_name)
        gamefile = gamefile or ""
        skills = skills or []

        # Keep agent contracts aligned with assigned verified skill scopes.
        for agent in agents:
            if agent.agent_id != executor.agent_id:
                sync_agent_skill_scopes(agent, skills)

        if not self.config.enabled:
            return self._keep_executor(
                executor,
                dispatch_layer="disabled",
                rationale="executor-dispatch/disabled; keep Executor",
                task_family=task_family,
            )

        # Layer 1: enforce the scope learned into each verified Skill contract.
        # The policy model ranks only specialists whose learned contract can
        # apply; it must never repair a cross-capability candidate list.
        # Exception: llm_only_dispatch opens the full specialist roster.
        eligible = self._eligible_specialists(
            agents=agents,
            skills=skills,
            task=task,
            task_family=task_family,
            gamefile=gamefile,
            executor=executor,
        )
        eligible_names = [agent.name for agent, _ in eligible]
        skills_by_agent = {
            agent.name: [skill.skill_name for skill in skill_list]
            for agent, skill_list in eligible
        }

        if not eligible:
            return self._keep_executor(
                executor,
                dispatch_layer="eligibility_empty",
                rationale=(
                    "executor-dispatch/eligibility_empty; "
                    "no specialist skill matches this episode"
                    if not self.config.llm_only_dispatch
                    else "executor-dispatch/eligibility_empty; no dispatchable specialists"
                ),
                task_family=task_family,
                eligible_agents=[],
                eligible_skills_by_agent={},
            )

        prefer = bool(self.config.prefer_matching_specialist)
        if self.config.llm_only_dispatch:
            prefer = False
        if (
            not self.config.llm_only_dispatch
            and len(eligible) == 1
            and (self.config.auto_assign_single or prefer)
        ):
            agent, skill_list = eligible[0]
            layer = (
                "eligibility_single"
                if self.config.auto_assign_single
                else "prefer_matching_specialist"
            )
            return AgentAssignment(
                primary_agent=agent.name,
                support_agents=[],
                score_by_agent={agent.name: 1.0},
                rationale=(
                    f"executor-dispatch/{layer}; "
                    f"only {agent.name} eligible "
                    f"({', '.join(s.skill_name for s in skill_list) or 'family'})"
                ),
                eligible_agents=eligible_names,
                eligible_skills_by_agent=skills_by_agent,
                dispatch_layer=layer,
                evidence={
                    "task_family": task_family,
                    "eligible_agents": eligible_names,
                    "eligible_skills_by_agent": skills_by_agent,
                    "assigned_primary": agent.name,
                    "dispatch_layer": layer,
                },
            )

        # Layer 2: Executor LLM among eligible specialists (+ self unless prefer).
        if self.backend is None:
            if prefer:
                return self._assign_specialist(
                    eligible[0][0],
                    eligible=eligible,
                    dispatch_layer="prefer_matching_no_backend",
                    rationale=(
                        "executor-dispatch/prefer_matching_specialist; "
                        f"no chat backend, defaulted to {eligible[0][0].name}"
                    ),
                    task_family=task_family,
                    eligible_agents=eligible_names,
                    eligible_skills_by_agent=skills_by_agent,
                )
            return self._keep_executor(
                executor,
                dispatch_layer="llm_no_backend",
                rationale=(
                    "executor-dispatch/llm-keep-executor; "
                    "no chat backend for multi-candidate dispatch"
                ),
                task_family=task_family,
                eligible_agents=eligible_names,
                eligible_skills_by_agent=skills_by_agent,
            )

        llm_pick = self._llm_assign(
            task=task,
            task_family=task_family,
            executor=executor,
            eligible=eligible,
            include_executor=not prefer,
        )
        if llm_pick is None:
            if prefer:
                return self._assign_specialist(
                    eligible[0][0],
                    eligible=eligible,
                    dispatch_layer="prefer_matching_llm_fallback",
                    rationale=(
                        "executor-dispatch/prefer_matching_specialist; "
                        f"LLM unparseable, defaulted to {eligible[0][0].name}"
                    ),
                    task_family=task_family,
                    eligible_agents=eligible_names,
                    eligible_skills_by_agent=skills_by_agent,
                )
            return self._keep_executor(
                executor,
                dispatch_layer="llm_unparseable",
                rationale=(
                    "executor-dispatch/llm-keep-executor; "
                    "LLM assign failed or unparseable"
                ),
                task_family=task_family,
                eligible_agents=eligible_names,
                eligible_skills_by_agent=skills_by_agent,
            )

        chosen_name = llm_pick["name"]
        if chosen_name == executor.name:
            if prefer:
                return self._assign_specialist(
                    eligible[0][0],
                    eligible=eligible,
                    dispatch_layer="prefer_matching_override_executor",
                    rationale=(
                        "executor-dispatch/prefer_matching_specialist; "
                        "overrode Executor keep; "
                        f"assigned {eligible[0][0].name} "
                        f"({llm_pick['reason']})"
                    ),
                    task_family=task_family,
                    eligible_agents=eligible_names,
                    eligible_skills_by_agent=skills_by_agent,
                )
            return self._keep_executor(
                executor,
                dispatch_layer="llm_keep_executor",
                rationale=(
                    f"executor-dispatch/llm; Executor kept control "
                    f"({llm_pick['reason']})"
                ),
                task_family=task_family,
                eligible_agents=eligible_names,
                eligible_skills_by_agent=skills_by_agent,
            )

        if chosen_name not in eligible_names:
            if prefer:
                return self._assign_specialist(
                    eligible[0][0],
                    eligible=eligible,
                    dispatch_layer="prefer_matching_rejected_pick",
                    rationale=(
                        "executor-dispatch/prefer_matching_specialist; "
                        f"rejected non-eligible `{chosen_name}`, "
                        f"defaulted to {eligible[0][0].name}"
                    ),
                    task_family=task_family,
                    eligible_agents=eligible_names,
                    eligible_skills_by_agent=skills_by_agent,
                )
            return self._keep_executor(
                executor,
                dispatch_layer="llm_rejected",
                rationale=(
                    "executor-dispatch/llm-keep-executor; "
                    f"rejected non-eligible pick `{chosen_name}`"
                ),
                task_family=task_family,
                eligible_agents=eligible_names,
                eligible_skills_by_agent=skills_by_agent,
            )

        return AgentAssignment(
            primary_agent=chosen_name,
            support_agents=[],
            score_by_agent={chosen_name: 1.0},
            rationale=(
                f"executor-dispatch/llm; selected {chosen_name} "
                f"({llm_pick['reason']})"
            ),
            eligible_agents=eligible_names,
            eligible_skills_by_agent=skills_by_agent,
            dispatch_layer="llm",
            evidence={
                "task_family": task_family,
                "eligible_agents": eligible_names,
                "eligible_skills_by_agent": skills_by_agent,
                "assigned_primary": chosen_name,
                "dispatch_layer": "llm",
                "llm_reason": llm_pick["reason"],
            },
        )

    def _eligible_specialists(
        self,
        *,
        agents: list[AgentSpec],
        skills: list[Skill],
        task: str,
        task_family: str,
        gamefile: str,
        executor: AgentSpec,
    ) -> list[tuple[AgentSpec, list[Skill]]]:
        if not self._family_dispatch_allowed(task_family):
            return []

        skill_by_name = {skill.skill_name: skill for skill in skills}
        eligible: list[tuple[AgentSpec, list[Skill]]] = []
        for agent in agents:
            if agent.agent_id == executor.agent_id:
                continue
            if not self._is_dispatchable(agent, executor):
                continue
            if self.config.llm_only_dispatch:
                # Open roster: contract scope is advisory for the LLM prompt,
                # not a hard eligibility gate.
                matched = [
                    skill
                    for skill_name in agent.assigned_skills
                    if (skill := skill_by_name.get(skill_name)) is not None
                    and skill.status == SkillStatus.VERIFIED
                ]
                eligible.append((agent, matched))
                continue
            matched: list[Skill] = []
            for skill_name in agent.assigned_skills:
                skill = skill_by_name.get(skill_name)
                if skill is None:
                    continue
                if skill.status != SkillStatus.VERIFIED:
                    continue
                if not self._contract_matches_episode(
                    skill,
                    agent=agent,
                    task=task,
                    task_family=task_family,
                    gamefile=gamefile,
                ):
                    continue
                if (
                    self.config.require_controller_for_eligibility
                    and not specialist_has_controller_for_family(
                        agent,
                        skills,
                        task_family,
                    )
                ):
                    continue
                matched.append(skill)
            if matched:
                eligible.append((agent, matched))
        return eligible

    def _family_dispatch_allowed(self, task_family: str) -> bool:
        """Apply optional allow/deny lists without hard-coded family policies."""
        current = self._normalized_scope(task_family)
        if not current:
            return True
        disabled = {
            self._normalized_scope(value)
            for value in (self.config.disabled_task_families or [])
            if self._normalized_scope(value)
        }
        if current in disabled:
            return False
        enabled = {
            self._normalized_scope(value)
            for value in (self.config.enabled_task_families or [])
            if self._normalized_scope(value)
        }
        if enabled and current not in enabled:
            return False
        return True

    @classmethod
    def _contract_matches_episode(
        cls,
        skill: Skill,
        *,
        agent: AgentSpec,
        task: str,
        task_family: str,
        gamefile: str,
    ) -> bool:
        """Gate routing with learned contract scope, without family policies."""
        scopes = cls._skill_scopes(skill)
        if not scopes:
            record = agent.shadow_evaluation_record or {}
            contract = record.get("capability_contract") or {}
            if isinstance(contract, dict):
                scopes.update(
                    cls._normalized_scope(value)
                    for value in contract.get("task_families", []) or []
                    if cls._normalized_scope(value)
                )

        current_scope = cls._normalized_scope(task_family)
        if current_scope and scopes:
            return current_scope in scopes
        if scopes and gamefile:
            normalized_gamefile = cls._normalized_scope(gamefile)
            return any(scope in normalized_gamefile for scope in scopes)

        # Legacy contracts may lack a learned scope. In that case require
        # discriminative semantic overlap rather than admitting every Skill.
        contract_text = " ".join(
            [
                skill.capability_key,
                skill.skill_name,
                skill.description,
                skill.precondition,
                str(skill.expected_effect or ""),
            ]
        )
        contract_tokens = cls._semantic_tokens(contract_text)
        task_tokens = cls._semantic_tokens(f"{task_family} {task} {gamefile}")
        return bool(contract_tokens & task_tokens)

    @classmethod
    def _skill_scopes(cls, skill: Skill) -> set[str]:
        values = list(skill.applicable_task_families)
        values.extend(skill.metadata.get("task_families", []) or [])
        values.append(skill.metadata.get("primary_task_family", ""))
        return {
            cls._normalized_scope(value)
            for value in values
            if cls._normalized_scope(value)
            and cls._normalized_scope(value) != "other"
        }

    @staticmethod
    def _normalized_scope(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")

    @staticmethod
    def _semantic_tokens(value: str) -> set[str]:
        generic = {
            "agent",
            "capability",
            "complete",
            "current",
            "effect",
            "environment",
            "execute",
            "learned",
            "object",
            "protocol",
            "reproduce",
            "skill",
            "specialist",
            "target",
            "task",
            "trajectory",
            "transform",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
            if len(token) >= 3 and token not in generic
        }

    def _keep_executor(
        self,
        executor: AgentSpec,
        *,
        dispatch_layer: str,
        rationale: str,
        task_family: str,
        eligible_agents: list[str] | None = None,
        eligible_skills_by_agent: dict[str, list[str]] | None = None,
    ) -> AgentAssignment:
        eligible_agents = list(eligible_agents or [])
        eligible_skills_by_agent = dict(eligible_skills_by_agent or {})
        return AgentAssignment(
            primary_agent=executor.name,
            support_agents=[],
            score_by_agent={executor.name: 0.0},
            rationale=rationale,
            eligible_agents=eligible_agents,
            eligible_skills_by_agent=eligible_skills_by_agent,
            dispatch_layer=dispatch_layer,
            evidence={
                "task_family": task_family,
                "eligible_agents": eligible_agents,
                "eligible_skills_by_agent": eligible_skills_by_agent,
                "assigned_primary": executor.name,
                "dispatch_layer": dispatch_layer,
            },
        )

    def _assign_specialist(
        self,
        agent: AgentSpec,
        *,
        eligible: list[tuple[AgentSpec, list[Skill]]],
        dispatch_layer: str,
        rationale: str,
        task_family: str,
        eligible_agents: list[str],
        eligible_skills_by_agent: dict[str, list[str]],
    ) -> AgentAssignment:
        del eligible  # roster already flattened into evidence fields
        return AgentAssignment(
            primary_agent=agent.name,
            support_agents=[],
            score_by_agent={agent.name: 1.0},
            rationale=rationale,
            eligible_agents=eligible_agents,
            eligible_skills_by_agent=eligible_skills_by_agent,
            dispatch_layer=dispatch_layer,
            evidence={
                "task_family": task_family,
                "eligible_agents": eligible_agents,
                "eligible_skills_by_agent": eligible_skills_by_agent,
                "assigned_primary": agent.name,
                "dispatch_layer": dispatch_layer,
            },
        )

    def _probation_quota_remaining(self, agent: AgentSpec) -> int:
        quota = int(self.config.probation_primary_quota or 0)
        if quota <= 0:
            return 0
        record = agent.shadow_evaluation_record or {}
        used = int(record.get("applicable_dispatched_games") or 0)
        return max(0, quota - used)

    def _is_dispatchable(self, agent: AgentSpec, executor: AgentSpec) -> bool:
        if agent.agent_id == executor.agent_id:
            return True
        status = _acting_status(agent)
        if status == "demoted":
            return False
        if status == "dormant" and not self.config.allow_dormant:
            return False
        record = agent.shadow_evaluation_record or {}
        quota_remaining = self._probation_quota_remaining(agent)
        if self.config.require_accepted_for_primary and status != "accepted":
            probe_ok = status == "probation" and bool(
                record.get("promotion_probe_passed")
            )
            quota_ok = status == "probation" and quota_remaining > 0
            if not probe_ok and not quota_ok:
                return False
        if status == "probation":
            # Missing budget means probation has not been consumed yet; only an
            # explicit remaining<=0 blocks dispatch (avoids cold-start lockout).
            # In-contract primary quota keeps Heat/Cool-style specialists
            # eligible until they accumulate enough matching evidence.
            if "trial_games_remaining" in record:
                remaining = int(record.get("trial_games_remaining") or 0)
                if remaining <= 0 and quota_remaining <= 0:
                    return False
        permissions = {
            permission.lower() for permission in agent.tool_permissions
        }
        return bool(
            permissions
            & {"alfworld_action", "webshop_action", "env_action"}
        )

    def _llm_assign(
        self,
        *,
        task: str,
        task_family: str,
        executor: AgentSpec,
        eligible: list[tuple[AgentSpec, list[Skill]]],
        include_executor: bool = True,
    ) -> dict[str, str] | None:
        assert self.backend is not None
        name_lookup = {executor.name.lower(): executor.name}
        roster_lines: list[str] = []
        if include_executor:
            roster_lines.append(
                f"- {executor.name} | role=Executor (fallback) | "
                f"status=accepted | {self._performance_text(executor, task_family)} | "
                "keep control when no specialist capability directly fits"
            )
        for agent, skill_list in eligible:
            name_lookup[agent.name.lower()] = agent.name
            status = _acting_status(agent)
            skill_bits = []
            for skill in skill_list[:4]:
                precondition = " ".join(skill.precondition.split())
                effect = " ".join(
                    str(skill.expected_effect or "").split()
                )
                if len(precondition) > 100:
                    precondition = precondition[:97] + "..."
                if len(effect) > 100:
                    effect = effect[:97] + "..."
                skill_bits.append(
                    f"{skill.skill_name}"
                    f" [description={' '.join(skill.description.split()) or 'n/a'}; "
                    f"precondition={precondition or 'n/a'}; "
                    f"effect={effect or 'n/a'}]"
                )
            contract = (agent.shadow_evaluation_record or {}).get(
                "capability_contract",
                {},
            )
            contract_description = " ".join(
                str(
                    contract.get("description", "")
                    if isinstance(contract, dict)
                    else ""
                ).split()
            )
            roster_lines.append(
                f"- {agent.name} | role={agent.role or agent.name} | "
                f"status={status} | "
                f"{self._performance_text(agent, task_family)} | "
                f"capability_contract={contract_description or 'n/a'} | "
                f"verified_skills={'; '.join(skill_bits)}"
            )
        if include_executor:
            system_prompt = (
                "You are the Executor of a multi-agent ALFWorld team. "
                "Choose the single agent best suited to complete the new task. "
                "Compare the task semantically with each learned CapabilityContract, "
                "Skill precondition, and expected effect. Use full-task wins and "
                "dispatch history as soft evidence, not as a hard veto. "
                "Prefer a specialist when their contract clearly covers the task; "
                "keep Executor only when no specialist fit is clear. "
                "Reply with exactly one line: <assign>AgentName</assign>."
            )
        else:
            system_prompt = (
                "You are the Executor of a multi-agent ALFWorld team. "
                "A specialist already matches this episode via a learned "
                "capability contract. Choose the single best matching "
                "specialist from the eligible roster. Do not choose Executor. "
                "Reply with exactly one line: <assign>AgentName</assign>."
            )
        user_prompt = (
            f"Task family: {task_family}\n"
            f"Task: {task}\n\n"
            f"Eligible roster:\n" + "\n".join(roster_lines)
        )
        try:
            result = self.backend.complete(system_prompt, user_prompt)
        except Exception:
            return None
        content = str(
            getattr(result, "content", None)
            or getattr(result, "text", None)
            or result
            or ""
        )
        chosen = self._parse_assign_name(content, name_lookup)
        if chosen is None:
            return None
        return {"name": chosen, "reason": content.strip()[:240]}

    @staticmethod
    def _performance_text(agent: AgentSpec, task_family: str) -> str:
        record = agent.shadow_evaluation_record or {}
        performance = record.get("task_performance") or {}
        by_family = (
            performance.get("by_family", {})
            if isinstance(performance, dict)
            else {}
        )
        scoped = (
            by_family.get(task_family, {})
            if isinstance(by_family, dict)
            else {}
        )
        source = scoped if scoped else performance
        dispatches = int(
            source.get("dispatches", 0)
            if isinstance(source, dict)
            else 0
        )
        wins = int(
            source.get("wins", 0)
            if isinstance(source, dict)
            else 0
        )
        rate = (wins + 1) / (dispatches + 2)
        scope = f"family={task_family}" if scoped else "overall"
        return (
            f"full_task_history({scope}): wins={wins}, "
            f"dispatches={dispatches}, smoothed_success={rate:.3f}"
        )

    @staticmethod
    def _parse_assign_name(
        content: str,
        name_lookup: dict[str, str],
    ) -> str | None:
        match = _ASSIGN_RE.search(content or "")
        raw = match.group(1).strip() if match else ""
        if not raw:
            line = _ASSIGN_LINE_RE.search(content or "")
            if line:
                raw = line.group(1).strip().strip("\"'`")
        if not raw:
            return None
        if raw in {name for name in name_lookup.values()}:
            return raw
        lowered = raw.lower()
        if lowered in name_lookup:
            return name_lookup[lowered]
        prefix_hits = [
            name
            for key, name in name_lookup.items()
            if key.startswith(lowered) or lowered.startswith(key)
        ]
        if len(prefix_hits) == 1:
            return prefix_hits[0]
        return None

    @staticmethod
    def _is_executor(agent: AgentSpec) -> bool:
        return "executor" in f"{agent.name} {agent.role}".lower()

    def _executor(
        self,
        agents: list[AgentSpec],
        executor_name: str | None,
    ) -> AgentSpec:
        if executor_name:
            for agent in agents:
                if agent.name == executor_name:
                    return agent
        for agent in agents:
            if self._is_executor(agent):
                return agent
        return agents[0]


def _acting_status(agent: AgentSpec) -> str:
    record = agent.shadow_evaluation_record or {}
    status = str(record.get("acting_status") or "").strip().lower()
    return status or "accepted"


def dispatch_config_from_mapping(raw: dict[str, Any] | None) -> ExecutorDispatchConfig:
    raw = raw or {}
    return ExecutorDispatchConfig(
        enabled=bool(raw.get("enabled", True)),
        allow_dormant=bool(raw.get("allow_dormant", True)),
        auto_assign_single=bool(raw.get("auto_assign_single", True)),
        require_accepted_for_primary=bool(
            raw.get("require_accepted_for_primary", False)
        ),
        require_controller_for_eligibility=bool(
            raw.get("require_controller_for_eligibility", False)
        ),
        prefer_matching_specialist=bool(
            raw.get("prefer_matching_specialist", True)
        ),
        probation_primary_quota=int(raw.get("probation_primary_quota", 8)),
        enabled_task_families=[
            str(item)
            for item in (raw.get("enabled_task_families") or [])
            if str(item).strip()
        ],
        disabled_task_families=[
            str(item)
            for item in (raw.get("disabled_task_families") or [])
            if str(item).strip()
        ],
        llm_only_dispatch=bool(raw.get("llm_only_dispatch", False)),
    )
