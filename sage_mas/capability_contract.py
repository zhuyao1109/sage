"""Compile verified, trajectory-grounded skills into executable agent roles."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict

from sage_mas.schemas import (
    AgentSpec,
    CapabilityContract,
    Skill,
    SkillStatus,
)


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _skill_families(skill: Skill) -> list[str]:
    values = list(skill.applicable_task_families)
    primary = str(skill.metadata.get("primary_task_family", "")).strip()
    if primary:
        values.append(primary)
    values.extend(
        str(value)
        for value in (skill.metadata.get("task_families", []) or [])
        if value
    )
    return _ordered_unique(values)


def sync_agent_skill_scopes(
    agent: AgentSpec,
    skills: list[Skill],
) -> bool:
    """Align agent contract task_families with assigned verified skill scopes.

    Learned skill scopes are the source of truth for dispatch eligibility.
    ADD_AGENT / ASSIGN_SKILL can leave the agent contract stale; call this
    before routing so specialists become eligible when their skills match.
    """
    if "executor" in f"{agent.name} {agent.role}".lower():
        return False
    skill_by_name = {skill.skill_name: skill for skill in skills}
    families: list[str] = []
    for skill_name in agent.assigned_skills:
        skill = skill_by_name.get(skill_name)
        if skill is None or skill.status != SkillStatus.VERIFIED:
            continue
        families.extend(_skill_families(skill))
    merged = _ordered_unique(families)
    if not merged:
        return False

    record = dict(agent.shadow_evaluation_record or {})
    changed = False
    existing = [
        str(value)
        for value in (record.get("task_families") or [])
        if str(value).strip()
    ]
    combined = _ordered_unique(existing + merged)
    if combined != existing:
        record["task_families"] = combined
        changed = True

    contract = record.get("capability_contract")
    if isinstance(contract, dict):
        contract = dict(contract)
        contract_families = [
            str(value)
            for value in (contract.get("task_families") or [])
            if str(value).strip()
        ]
        contract_combined = _ordered_unique(contract_families + merged)
        if contract_combined != contract_families:
            contract["task_families"] = contract_combined
            record["capability_contract"] = contract
            changed = True
    elif merged:
        # Minimal contract scope so dispatcher can fall back to agent record.
        record["capability_contract"] = {
            "task_families": list(merged),
            "skill_names": list(agent.assigned_skills),
        }
        changed = True

    if changed:
        agent.shadow_evaluation_record = record
    return changed


def compile_capability_contract(
    skills: list[Skill],
) -> CapabilityContract:
    if not skills:
        raise ValueError("CapabilityContract requires at least one skill")
    unverified = [
        skill.skill_name
        for skill in skills
        if skill.status != SkillStatus.VERIFIED
    ]
    if unverified:
        raise ValueError(
            "CapabilityContract accepts verified skills only: "
            + ", ".join(unverified)
        )
    capability_weights: Counter[str] = Counter()
    for skill in skills:
        key = (
            skill.capability_key.strip()
            or str(skill.metadata.get("source_signal", "")).strip()
            or skill.skill_name
        )
        capability_weights[key] += max(
            skill.support_count,
            len(skill.evidence_ids),
            1,
        )
    name = capability_weights.most_common(1)[0][0]
    skill_ids = sorted(skill.skill_id for skill in skills)
    digest = hashlib.sha256("|".join(skill_ids).encode("utf-8")).hexdigest()[:12]
    utilities = [
        float(skill.marginal_utility)
        for skill in skills
        if skill.marginal_utility is not None
    ]
    reliability_weight = sum(
        max(skill.support_count, len(skill.evidence_ids), 1)
        for skill in skills
    )
    reliability = sum(
        skill.reliability_weight
        * max(skill.support_count, len(skill.evidence_ids), 1)
        for skill in skills
    ) / max(reliability_weight, 1)
    required_tools = _ordered_unique(
        [
            str(tool)
            for skill in skills
            for tool in (skill.metadata.get("required_tools", []) or [])
        ]
    )
    env_tools = {"alfworld_action", "webshop_action", "env_action"}
    if not (set(required_tools) & env_tools):
        blob = " ".join(
            [
                *(skill.description for skill in skills),
                *(skill.precondition for skill in skills),
                *(
                    step
                    for skill in skills
                    for step in (skill.action_protocol or [])
                ),
            ]
        ).lower()
        if "search[" in blob or "click[" in blob or "webshop" in blob:
            required_tools.append("webshop_action")
        else:
            required_tools.append("alfworld_action")
    return CapabilityContract(
        capability_id=f"capability-{digest}",
        name=name,
        description=" ".join(
            _ordered_unique([skill.description.strip() for skill in skills])
        ),
        preconditions=_ordered_unique(
            [skill.precondition.strip() for skill in skills]
        ),
        action_protocols=[
            list(skill.action_protocol)
            for skill in skills
            if skill.action_protocol
        ],
        expected_effects=_ordered_unique(
            [
                str(skill.expected_effect or "").strip()
                for skill in skills
            ]
        ),
        atomic_ops=sorted(
            {
                operation
                for skill in skills
                for operation in skill.applicable_atomic_ops
            },
            key=lambda operation: operation.value,
        ),
        required_tools=required_tools,
        skill_ids=skill_ids,
        skill_names=[skill.skill_name for skill in skills],
        evidence_ids=sorted(
            {
                evidence_id
                for skill in skills
                for evidence_id in skill.evidence_ids
            }
        ),
        task_families=sorted(
            {
                family
                for skill in skills
                for family in _skill_families(skill)
            }
        ),
        marginal_utility=min(utilities) if utilities else None,
        reliability=reliability,
    )


class AgentRoleCompiler:
    """Deterministically compile a role from a verified capability contract."""

    @staticmethod
    def _role_name(contract: CapabilityContract) -> str:
        words = re.findall(r"[A-Za-z0-9]+", contract.name)
        stem = "".join(word.capitalize() for word in words[:6])
        if not stem or stem.lower() in {
            "executor",
            "agent",
            "specialist",
            "capability",
        }:
            stem = f"Capability{contract.capability_id[-6:].upper()}"
        return f"{stem}Specialist"

    @staticmethod
    def exclusive_skill_context(skills: list[Skill]) -> str:
        """Build specialist-only context from assigned Executable skill protocols.

        No hand-written domain checklists: context is derived only from the
        learned skill cluster the specialist carries.
        """
        blocks: list[str] = []
        for skill in skills:
            protocol = [
                str(step).strip()
                for step in (skill.action_protocol or [])
                if str(step).strip()
            ]
            if not protocol:
                continue
            preview = " -> ".join(protocol[:8])
            blocks.append(f"[{skill.skill_name}] {preview}")
        return " || ".join(blocks)

    def compile(
        self,
        contract: CapabilityContract,
        existing_agent_names: set[str],
        *,
        executor_name: str = "Executor",
        dispatch_only: bool = True,
        trial_games: int = 3,
        skills: list[Skill] | None = None,
        teacher_records: list | None = None,
        student_records: list | None = None,
        max_demos: int = 2,
        max_demo_actions: int = 10,
    ) -> AgentSpec:
        from sage_mas.specialist_context import build_specialist_acting_brief

        role = self._role_name(contract)
        name = role
        suffix = 2
        while name in existing_agent_names:
            name = f"{role}-{suffix}"
            suffix += 1
        skill_list = list(skills or [])
        brief = build_specialist_acting_brief(
            agent_name=name,
            skills=skill_list,
            executor_name=executor_name,
            dispatch_only=dispatch_only,
            teacher_records=teacher_records,
            student_records=student_records,
            max_demos=max_demos,
            max_actions=max_demo_actions,
        )
        # Fallback if no skills/demos: keep a short exclusive protocol sketch.
        role_specification = str(brief.get("role_specification") or "").strip()
        if not role_specification:
            exclusive = self.exclusive_skill_context(skill_list)
            protocol_preview = exclusive or " | ".join(
                " -> ".join(protocol)
                for protocol in contract.action_protocols[:2]
            )
            role_specification = (
                f"You are {name}, a capability specialist. When dispatched, "
                "complete the full environment task using only admissible "
                "actions and the exclusive skill protocols below."
            )
            if protocol_preview:
                role_specification += f" Exclusive protocols: {protocol_preview}."
            if dispatch_only:
                role_specification += (
                    f" Act only when {executor_name} assigns this episode."
                )
        responsibilities = _ordered_unique(
            [
                "Complete the assigned environment task end-to-end",
                "Follow exclusive trajectory-derived demos and skill protocols "
                "for this specialist only",
                f"Return control to {executor_name} when inactive",
            ]
        )
        families = ", ".join(contract.task_families)
        activation_condition = (
            f"Activate only when {executor_name} dispatches this agent."
        )
        if families:
            activation_condition += f" Learned applicability: {families}."
        # Richer context needs a slightly larger budget than the Executor shell.
        has_traj_context = int(brief.get("n_demos") or 0) > 0 or int(
            brief.get("n_cues") or 0
        ) > 0
        return AgentSpec(
            name=name,
            role=role,
            responsibilities=responsibilities,
            assigned_skills=list(contract.skill_names),
            tool_permissions=list(contract.required_tools),
            activation_condition=activation_condition,
            token_budget=None,
            turn_budget=None,
            role_specification=role_specification,
            responsibility_boundary=(
                f"In scope: exclusive skills [{', '.join(contract.skill_names)}] "
                "and their trajectory demos after dispatch. Out of scope: "
                "acting without dispatch or using skills outside this cluster."
            ),
            input_protocol=(
                "Input: observation, admissible actions, task goal, exclusive "
                "skill protocol and success demos for this specialist."
            ),
            output_protocol=(
                "Output exactly one "
                "<think>brief reasoning grounded in demos/protocol</think>"
                "<action>one admissible environment action</action>."
            ),
            communication_edges=[executor_name],
            retirement_condition=(
                "Retire after repeated negative conditional utility or "
                "persistent non-use across evaluation windows."
            ),
            shadow_evaluation_record={
                "status": "pending",
                "acting_status": "probation",
                "dispatch_only": bool(dispatch_only),
                "trial_games_remaining": max(1, int(trial_games)),
                "dispatched_games": 0,
                "actions_executed": 0,
                "wins_as_primary": 0,
                "applicable_dispatched_games": 0,
                "applicable_wins_as_primary": 0,
                "out_of_scope_dispatches": 0,
                "scope_eligible_games": 0,
                "accepted_windows": 0,
                "rejected_windows": 0,
                "last_shadow_decision": None,
                "capability_contract_id": contract.capability_id,
                "capability_contract": asdict(contract),
                "capability_name": contract.name,
                "task_families": list(contract.task_families),
                "verified_skill_marginal_utility": contract.marginal_utility,
                "contract_reliability": contract.reliability,
                "exclusive_skill_context": True,
                "specialist_context_source": brief.get("context_source")
                or "executable_cluster",
                "trajectory_context": {
                    "n_demos": brief.get("n_demos"),
                    "n_cues": brief.get("n_cues"),
                    "n_protocol_blocks": brief.get("n_protocol_blocks"),
                },
            },
        )
