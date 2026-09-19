"""Executor-mediated task assignment for τ² (ported from sage_mas).

eligibility → prefer/auto assign → optional LLM dispatch → sticky primary.
Benchmark-specific pieces (ALFWorld controllers / action guards) are omitted;
matching uses capability_key identity, gated by ``skill.domain`` == episode domain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from sage_tau2.schemas import AgentSpec, SkillStatus, Tau2Skill
from sage_tau2.task_context import domains_match, episode_scope_from_task_id, organizational_capability_key, skill_matches_episode

_ASSIGN_RE = re.compile(
    r"<\s*assign\s*>\s*([^<>]+?)\s*<\s*/\s*assign\s*>",
    re.IGNORECASE,
)
_ASSIGN_LINE_RE = re.compile(
    r"(?im)^\s*assign(?:ed)?\s*(?:to)?\s*[:\-]\s*(.+?)\s*$",
)

# Permissions that make a specialist eligible to own a τ² episode.
_DISPATCH_PERMISSIONS = {"env_action", "tau2_tools", "alfworld_action"}


class ChatBackend(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> Any: ...


@dataclass(slots=True)
class AgentAssignment:
    """Episode-level ownership for one τ² task."""

    primary_agent: str
    support_agents: list[str] = field(default_factory=list)
    score_by_agent: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    eligible_agents: list[str] = field(default_factory=list)
    eligible_skills_by_agent: dict[str, list[str]] = field(default_factory=dict)
    dispatch_layer: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutorDispatchConfig:
    enabled: bool = True
    allow_dormant: bool = False
    auto_assign_single: bool = True
    # Match sage_mas online600: only accepted (or quota probation) take primary.
    require_accepted_for_primary: bool = True
    prefer_matching_specialist: bool = True
    probation_primary_quota: int = 8
    allow_provisional_eligibility: bool = True
    llm_only_dispatch: bool = False
    enabled_domains: list[str] = field(default_factory=list)
    disabled_domains: list[str] = field(default_factory=list)


def sync_agent_skill_scopes(
    agent: AgentSpec, skills: list[Tau2Skill]
) -> None:
    """Keep agent.capability_keys aligned with assigned skills (write-first)."""
    from sage_tau2.skill_resolve import resolve_assigned_skills

    caps = [str(c) for c in (agent.capability_keys or []) if str(c).strip()]
    seen = {c.lower() for c in caps}
    for skill in resolve_assigned_skills(agent.assigned_skills, skills):
        key = organizational_capability_key(skill) or str(
            skill.capability_key or ""
        ).strip()
        if key and key.lower() not in seen:
            caps.append(key)
            seen.add(key.lower())
    agent.capability_keys = caps


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
        domain: str,
        agents: list[AgentSpec],
        skills: list[Tau2Skill] | None = None,
        executor_name: str | None = None,
        task_id: str | None = None,
        episode_scope: str | None = None,
    ) -> AgentAssignment:
        if not agents:
            raise ValueError("ExecutorDispatcher requires at least one agent")
        executor = self._executor(agents, executor_name)
        skills = skills or []
        task_id = task_id or ""

        for agent in agents:
            if agent.agent_id != executor.agent_id:
                sync_agent_skill_scopes(agent, skills)

        if not self.config.enabled:
            return self._keep_executor(
                executor,
                dispatch_layer="disabled",
                rationale="executor-dispatch/disabled; keep Executor",
                domain=domain,
            )

        eligible = self._eligible_specialists(
            agents=agents,
            skills=skills,
            task=task,
            domain=domain,
            task_id=task_id,
            episode_scope=episode_scope or "",
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
                domain=domain,
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
                    f"({', '.join(s.skill_name for s in skill_list) or 'capability'})"
                ),
                eligible_agents=eligible_names,
                eligible_skills_by_agent=skills_by_agent,
                dispatch_layer=layer,
                evidence={
                    "domain": domain,
                    "task_id": task_id,
                    "eligible_agents": eligible_names,
                    "eligible_skills_by_agent": skills_by_agent,
                    "assigned_primary": agent.name,
                    "dispatch_layer": layer,
                },
            )

        if self.backend is None:
            if prefer:
                return self._assign_specialist(
                    eligible[0][0],
                    dispatch_layer="prefer_matching_no_backend",
                    rationale=(
                        "executor-dispatch/prefer_matching_specialist; "
                        f"no chat backend, defaulted to {eligible[0][0].name}"
                    ),
                    domain=domain,
                    task_id=task_id,
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
                domain=domain,
                eligible_agents=eligible_names,
                eligible_skills_by_agent=skills_by_agent,
            )

        llm_pick = self._llm_assign(
            task=task,
            domain=domain,
            executor=executor,
            eligible=eligible,
            include_executor=not prefer,
        )
        if llm_pick is None:
            if prefer:
                return self._assign_specialist(
                    eligible[0][0],
                    dispatch_layer="prefer_matching_llm_fallback",
                    rationale=(
                        "executor-dispatch/prefer_matching_specialist; "
                        f"LLM unparseable, defaulted to {eligible[0][0].name}"
                    ),
                    domain=domain,
                    task_id=task_id,
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
                domain=domain,
                eligible_agents=eligible_names,
                eligible_skills_by_agent=skills_by_agent,
            )

        chosen_name = llm_pick["name"]
        if chosen_name == executor.name:
            if prefer:
                return self._assign_specialist(
                    eligible[0][0],
                    dispatch_layer="prefer_matching_override_executor",
                    rationale=(
                        "executor-dispatch/prefer_matching_specialist; "
                        "overrode Executor keep; "
                        f"assigned {eligible[0][0].name} "
                        f"({llm_pick['reason']})"
                    ),
                    domain=domain,
                    task_id=task_id,
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
                domain=domain,
                eligible_agents=eligible_names,
                eligible_skills_by_agent=skills_by_agent,
            )

        if chosen_name not in eligible_names:
            if prefer:
                return self._assign_specialist(
                    eligible[0][0],
                    dispatch_layer="prefer_matching_rejected_pick",
                    rationale=(
                        "executor-dispatch/prefer_matching_specialist; "
                        f"rejected non-eligible `{chosen_name}`, "
                        f"assigned {eligible[0][0].name}"
                    ),
                    domain=domain,
                    task_id=task_id,
                    eligible_agents=eligible_names,
                    eligible_skills_by_agent=skills_by_agent,
                )
            return self._keep_executor(
                executor,
                dispatch_layer="llm_rejected",
                rationale=(
                    "executor-dispatch/llm-keep-executor; "
                    f"rejected non-eligible `{chosen_name}`"
                ),
                domain=domain,
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
                "domain": domain,
                "task_id": task_id,
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
        skills: list[Tau2Skill],
        task: str,
        domain: str,
        task_id: str,
        episode_scope: str,
        executor: AgentSpec,
    ) -> list[tuple[AgentSpec, list[Tau2Skill]]]:
        if not self._domain_dispatch_allowed(domain):
            return []

        skill_by_name = {skill.skill_name: skill for skill in skills}
        allowed_status = {SkillStatus.VERIFIED}
        if self.config.allow_provisional_eligibility:
            allowed_status.add(SkillStatus.PROVISIONAL)

        from sage_tau2.skill_resolve import resolve_assigned_skills

        eligible: list[tuple[AgentSpec, list[Tau2Skill]]] = []
        for agent in agents:
            if agent.agent_id == executor.agent_id or agent.name == executor.name:
                continue
            if not self._is_dispatchable(agent, executor):
                continue
            if self.config.llm_only_dispatch:
                matched = resolve_assigned_skills(
                    agent.assigned_skills,
                    skills,
                    allowed_status=allowed_status,
                )
                eligible.append((agent, matched))
                continue
            matched: list[Tau2Skill] = []
            for skill in resolve_assigned_skills(
                agent.assigned_skills,
                skills,
                allowed_status=allowed_status,
            ):
                if not self._contract_matches_episode(
                    skill,
                    agent=agent,
                    task=task,
                    domain=domain,
                    task_id=task_id,
                    episode_scope=episode_scope,
                ):
                    continue
                matched.append(skill)
            if not matched:
                # Capability fallback: assigned_skills refs may dangle when a
                # skill was retired/replaced before the org was healed. Keep the
                # specialist dispatchable by matching active skills that carry
                # one of the agent's declared capability keys.
                agent_caps = {
                    str(c or "").strip().lower()
                    for c in (agent.capability_keys or [])
                    if str(c or "").strip()
                }
                for skill in skills:
                    if skill.status not in allowed_status:
                        continue
                    key = str(
                        organizational_capability_key(skill) or skill.capability_key or ""
                    ).strip().lower()
                    if not key or key not in agent_caps:
                        continue
                    if not self._contract_matches_episode(
                        skill,
                        agent=agent,
                        task=task,
                        domain=domain,
                        task_id=task_id,
                        episode_scope=episode_scope,
                    ):
                        continue
                    matched.append(skill)
            if matched:
                eligible.append((agent, matched))
        # Prefer accepted specialists, then higher nomination utility — mirrors
        # ALFWorld preference for proven carriers when several scopes match.
        eligible.sort(
            key=lambda item: (
                0 if _acting_status(item[0]) == "accepted" else 1,
                -float((item[0].metadata or {}).get("nominated_utility") or 0.0),
                item[0].name,
            )
        )
        return eligible

    def _domain_dispatch_allowed(self, domain: str) -> bool:
        current = self._normalized_scope(domain)
        if not current:
            return True
        disabled = {
            self._normalized_scope(value)
            for value in (self.config.disabled_domains or [])
            if self._normalized_scope(value)
        }
        if current in disabled:
            return False
        enabled = {
            self._normalized_scope(value)
            for value in (self.config.enabled_domains or [])
            if self._normalized_scope(value)
        }
        if enabled and current not in enabled:
            return False
        return True

    @classmethod
    def _contract_matches_episode(
        cls,
        skill: Tau2Skill,
        *,
        agent: AgentSpec,
        task: str,
        domain: str,
        task_id: str,
        episode_scope: str,
    ) -> bool:
        """Gate routing with the same scope rules as skill injection."""
        del agent  # retained for call-site compatibility
        return skill_matches_episode(
            skill,
            domain=domain,
            task_id=task_id,
            episode_scope=episode_scope,
            task_text=task,
        )

    @classmethod
    def _skill_scopes(cls, skill: Tau2Skill, agent: AgentSpec) -> set[str]:
        values = list(skill.metadata.get("task_families") or [])
        primary = skill.metadata.get("primary_task_family")
        if primary:
            values.append(str(primary))
        if not values:
            record = agent.shadow_evaluation_record or {}
            contract = record.get("capability_contract") or {}
            if isinstance(contract, dict):
                values.extend(contract.get("task_families") or [])
        key = str(skill.capability_key or "")
        if "." in key:
            values.append(key.split(".", 1)[-1])
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
            "customer",
            "effect",
            "environment",
            "execute",
            "learned",
            "policy",
            "protocol",
            "skill",
            "specialist",
            "target",
            "task",
            "trajectory",
            "user",
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
        domain: str,
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
                "domain": domain,
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
        dispatch_layer: str,
        rationale: str,
        domain: str,
        task_id: str,
        eligible_agents: list[str],
        eligible_skills_by_agent: dict[str, list[str]],
    ) -> AgentAssignment:
        return AgentAssignment(
            primary_agent=agent.name,
            support_agents=[],
            score_by_agent={agent.name: 1.0},
            rationale=rationale,
            eligible_agents=eligible_agents,
            eligible_skills_by_agent=eligible_skills_by_agent,
            dispatch_layer=dispatch_layer,
            evidence={
                "domain": domain,
                "task_id": task_id,
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
        if agent.agent_id == executor.agent_id or agent.name == executor.name:
            return False
        status = _acting_status(agent)
        if status == "demoted":
            return False
        if status == "dormant" and not self.config.allow_dormant:
            return False
        record = agent.shadow_evaluation_record or {}
        probe_passed = bool(record.get("promotion_probe_passed"))
        # Align with sage_mas: only block nominees still awaiting Spec-vs-Exec.
        # Editor-committed probation agents may take primary via quota.
        awaiting_spec = bool(record.get("nominate_admit_awaiting_admission"))
        if awaiting_spec and not probe_passed:
            return False
        quota_remaining = self._probation_quota_remaining(agent)
        if self.config.require_accepted_for_primary and status != "accepted":
            probe_ok = status == "probation" and probe_passed
            quota_ok = status == "probation" and quota_remaining > 0
            if not probe_ok and not quota_ok:
                return False
        if status == "probation" and "trial_games_remaining" in record:
            remaining = int(record.get("trial_games_remaining") or 0)
            if remaining <= 0 and quota_remaining <= 0 and not probe_passed:
                return False
        permissions = {
            permission.lower() for permission in (agent.tool_permissions or [])
        }
        if not permissions:
            return True
        return bool(permissions & _DISPATCH_PERMISSIONS)

    def _llm_assign(
        self,
        *,
        task: str,
        domain: str,
        executor: AgentSpec,
        eligible: list[tuple[AgentSpec, list[Tau2Skill]]],
        include_executor: bool = True,
    ) -> dict[str, str] | None:
        assert self.backend is not None
        name_lookup = {executor.name.lower(): executor.name}
        roster_lines: list[str] = []
        if include_executor:
            roster_lines.append(
                f"- {executor.name} | role=Executor (fallback) | "
                "status=accepted | keep control when no specialist capability fits"
            )
        for agent, skill_list in eligible:
            name_lookup[agent.name.lower()] = agent.name
            status = _acting_status(agent)
            skill_bits = []
            for skill in skill_list[:4]:
                precondition = " ".join(skill.precondition.split())
                if len(precondition) > 100:
                    precondition = precondition[:97] + "..."
                skill_bits.append(
                    f"{skill.skill_name} [cap={skill.capability_key}; "
                    f"precondition={precondition or 'n/a'}]"
                )
            roster_lines.append(
                f"- {agent.name} | role=specialist | status={status} | "
                f"caps={agent.capability_keys} | skills={'; '.join(skill_bits) or 'n/a'}"
            )
        if include_executor:
            system_prompt = (
                "You are the Executor of a multi-agent τ² customer-service team. "
                "Choose the single agent best suited to complete the new task. "
                "Prefer a specialist when their capability clearly covers the task; "
                "keep Executor only when no specialist fit is clear. "
                "Reply with exactly one line: <assign>AgentName</assign>."
            )
        else:
            system_prompt = (
                "You are the Executor of a multi-agent τ² customer-service team. "
                "A specialist already matches this episode via a learned capability. "
                "Choose the single best matching specialist. Do not choose Executor. "
                "Reply with exactly one line: <assign>AgentName</assign>."
            )
        user_prompt = (
            f"Domain: {domain}\n"
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
    status = str(agent.acting_status or "").strip().lower()
    if status:
        return status
    record = agent.shadow_evaluation_record or {}
    return str(record.get("acting_status") or "accepted").strip().lower()


def dispatch_config_from_mapping(raw: dict[str, Any] | None) -> ExecutorDispatchConfig:
    raw = raw or {}
    return ExecutorDispatchConfig(
        enabled=bool(raw.get("enabled", True)),
        allow_dormant=bool(raw.get("allow_dormant", False)),
        auto_assign_single=bool(raw.get("auto_assign_single", True)),
        require_accepted_for_primary=bool(
            raw.get("require_accepted_for_primary", True)
        ),
        prefer_matching_specialist=bool(
            raw.get("prefer_matching_specialist", True)
        ),
        probation_primary_quota=int(raw.get("probation_primary_quota", 8)),
        allow_provisional_eligibility=bool(
            raw.get("allow_provisional_eligibility", True)
        ),
        llm_only_dispatch=bool(raw.get("llm_only_dispatch", False)),
        enabled_domains=[
            str(item)
            for item in (raw.get("enabled_domains") or [])
            if str(item).strip()
        ],
        disabled_domains=[
            str(item)
            for item in (raw.get("disabled_domains") or [])
            if str(item).strip()
        ],
    )


def record_primary_dispatch(agent: AgentSpec) -> None:
    """Bump probation dispatch counters after sticky primary assignment."""
    record = dict(agent.shadow_evaluation_record or {})
    record["applicable_dispatched_games"] = int(
        record.get("applicable_dispatched_games") or 0
    ) + 1
    if "trial_games_remaining" in record:
        record["trial_games_remaining"] = max(
            0, int(record.get("trial_games_remaining") or 0) - 1
        )
    agent.shadow_evaluation_record = record
