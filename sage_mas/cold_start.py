"""Cold-start bundle: filter specialists/skills by empirical family ΔSR."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from sage_mas.schemas import AgentSpec, Skill
from sage_mas.serialization import (
    load_agents,
    load_skills,
    to_primitive,
    write_json,
)

# Default from eval_family_specialists_gpt4omini (valid_unseen, n=8).
DEFAULT_FAMILY_ENABLE: dict[str, dict[str, Any]] = {
    "pick_and_place": {
        "enabled": False,
        "delta_sr": -0.25,
        "reason": "negative_transfer_vs_executor",
    },
    "pick_two_obj_and_place": {
        "enabled": False,
        "delta_sr": -0.125,
        "reason": "negative_transfer_and_actor_ceiling",
    },
    "look_at_obj_in_light": {
        "enabled": True,
        "delta_sr": 0.375,
        "reason": "positive_delta",
    },
    "pick_clean_then_place_in_recep": {
        "enabled": True,
        "delta_sr": 0.0,
        "reason": "non_negative_delta",
    },
    "pick_heat_then_place_in_recep": {
        "enabled": True,
        "delta_sr": 0.125,
        "reason": "positive_delta",
    },
    "pick_cool_then_place_in_recep": {
        "enabled": True,
        "delta_sr": 0.25,
        "reason": "positive_delta",
    },
}


def family_of_agent(agent: AgentSpec) -> str | None:
    if agent.name == "Executor" or "executor" in f"{agent.name} {agent.role}".lower():
        return None
    record = dict(agent.shadow_evaluation_record or {})
    families = [str(x) for x in (record.get("task_families") or []) if str(x).strip()]
    if families:
        return families[0]
    contract = record.get("capability_contract") or {}
    if isinstance(contract, dict):
        for value in contract.get("task_families") or []:
            text = str(value).strip()
            if text:
                return text
    return None


def family_of_skill(skill: Skill) -> str | None:
    families = [str(x) for x in (skill.applicable_task_families or []) if str(x).strip()]
    if families:
        return families[0]
    meta = skill.metadata or {}
    primary = str(meta.get("primary_task_family") or "").strip()
    return primary or None


def enable_table_from_eval_summary(
    summary: dict[str, Any],
    *,
    min_delta_sr: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Build enable/disable table from eval_family_specialists summary.json."""
    table: dict[str, dict[str, Any]] = {}
    for row in summary.get("families") or []:
        family = str(row.get("family") or "").strip()
        if not family:
            continue
        delta = row.get("delta_sr")
        if delta is None:
            spec = float(row.get("specialist_sr") or 0.0)
            exe = row.get("executor_sr")
            delta = None if exe is None else float(spec) - float(exe)
        enabled = delta is not None and float(delta) >= float(min_delta_sr)
        table[family] = {
            "enabled": bool(enabled),
            "delta_sr": delta,
            "specialist_sr": row.get("specialist_sr"),
            "executor_sr": row.get("executor_sr"),
            "agent_name": row.get("agent_name"),
            "reason": (
                "positive_or_zero_delta"
                if enabled
                else "negative_delta_or_missing"
            ),
        }
    return table


def resolve_enable_table(
    *,
    eval_summary: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    min_delta_sr: float = 0.0,
) -> dict[str, dict[str, Any]]:
    if eval_summary:
        table = enable_table_from_eval_summary(
            eval_summary,
            min_delta_sr=min_delta_sr,
        )
    else:
        table = deepcopy(DEFAULT_FAMILY_ENABLE)
    for family, payload in (overrides or {}).items():
        row = dict(table.get(family) or {})
        if isinstance(payload, bool):
            row["enabled"] = payload
            row["reason"] = "manual_override"
        elif isinstance(payload, dict):
            row.update(payload)
        table[str(family)] = row
    return table


def enabled_families(table: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        family
        for family, row in table.items()
        if bool((row or {}).get("enabled"))
    )


def disabled_families(table: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        family
        for family, row in table.items()
        if not bool((row or {}).get("enabled"))
    )


def family_enable_from_dispatch_lists(
    *,
    enabled: list[str] | tuple[str, ...] | None = None,
    disabled: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Seed a runtime enable table from executor_dispatch allow/deny lists."""
    table: dict[str, dict[str, Any]] = {}
    for family in enabled or []:
        key = str(family).strip()
        if not key:
            continue
        table[key] = {
            "enabled": True,
            "reason": "seed_dispatch_enabled",
            "source": "dispatch_config",
        }
    for family in disabled or []:
        key = str(family).strip()
        if not key:
            continue
        table[key] = {
            "enabled": False,
            "reason": "seed_dispatch_disabled",
            "source": "dispatch_config",
        }
    return table


def apply_admit_decision_to_family_enable(
    table: dict[str, dict[str, Any]] | None,
    *,
    family: str,
    accepted: bool,
    specialist_sr: float | None = None,
    executor_sr: float | None = None,
    agent_name: str | None = None,
    source: str = "admit",
    n_tasks: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Update one family's enable row from an Admit / actor-promotion decision.

    Pass → enabled=True (family may receive specialist dispatch).
    Fail → enabled=False (family stays Executor-only for specialists).
    """
    key = str(family or "").strip()
    out: dict[str, dict[str, Any]] = {
        str(name): dict(row or {}) for name, row in (table or {}).items()
    }
    if not key:
        return out
    row = dict(out.get(key) or {})
    delta = None
    if specialist_sr is not None and executor_sr is not None:
        delta = float(specialist_sr) - float(executor_sr)
    row.update(
        {
            "enabled": bool(accepted),
            "delta_sr": delta,
            "specialist_sr": specialist_sr,
            "executor_sr": executor_sr,
            "agent_name": agent_name,
            "n_tasks": n_tasks,
            "reason": "admit_pass" if accepted else "admit_fail",
            "source": source,
        }
    )
    out[key] = row
    return out


def dispatch_lists_from_family_enable(
    table: dict[str, dict[str, Any]] | None,
) -> tuple[list[str], list[str]]:
    """Map enable table → (enabled_task_families, disabled_task_families)."""
    table = table or {}
    return enabled_families(table), disabled_families(table)


def apply_admit_decisions_to_family_enable(
    table: dict[str, dict[str, Any]] | None,
    *,
    agents: list[AgentSpec],
    decisions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    source: str = "admit",
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Apply a batch of Admit / actor-promotion decisions to the enable table.

    Decision dicts may use ``agent`` (actor_promotion) or ``agent_name``
    (nominate_admit). Returns ``(updated_table, update_rows)``.
    """
    by_name = {agent.name: agent for agent in agents}
    out: dict[str, dict[str, Any]] = {
        str(name): dict(row or {}) for name, row in (table or {}).items()
    }
    updates: list[dict[str, Any]] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        name = str(
            decision.get("agent") or decision.get("agent_name") or ""
        ).strip()
        if not name:
            continue
        agent = by_name.get(name)
        family = family_of_agent(agent) if agent is not None else None
        if not family:
            family = str(
                decision.get("family") or decision.get("task_family") or ""
            ).strip() or None
        if not family:
            updates.append(
                {
                    "agent": name,
                    "skipped": True,
                    "reason": "no_family",
                }
            )
            continue
        accepted = bool(decision.get("accepted"))
        out = apply_admit_decision_to_family_enable(
            out,
            family=family,
            accepted=accepted,
            specialist_sr=decision.get("specialist_sr"),
            executor_sr=decision.get("executor_sr"),
            agent_name=name,
            source=source,
            n_tasks=decision.get("n_tasks"),
        )
        updates.append(
            {
                "agent": name,
                "family": family,
                "accepted": accepted,
                "enabled": accepted,
                "specialist_sr": decision.get("specialist_sr"),
                "executor_sr": decision.get("executor_sr"),
            }
        )
    return out, updates


def filter_cold_start_organization(
    agents: list[AgentSpec],
    *,
    enable_table: dict[str, dict[str, Any]],
) -> list[AgentSpec]:
    allowed = set(enabled_families(enable_table))
    kept: list[AgentSpec] = []
    enabled_skill_names: set[str] = set()
    for agent in agents:
        family = family_of_agent(agent)
        if family is None:
            kept.append(deepcopy(agent))
            continue
        if family not in allowed:
            continue
        clone = deepcopy(agent)
        kept.append(clone)
        enabled_skill_names.update(clone.assigned_skills or [])
    for agent in kept:
        if agent.name == "Executor" or "executor" in f"{agent.name} {agent.role}".lower():
            agent.assigned_skills = sorted(enabled_skill_names)
    return kept


def filter_cold_start_skills(
    skills: list[Skill],
    *,
    enable_table: dict[str, dict[str, Any]],
    organization: list[AgentSpec] | None = None,
) -> list[Skill]:
    allowed = set(enabled_families(enable_table))
    if organization is not None:
        assigned = {
            name
            for agent in organization
            for name in (agent.assigned_skills or [])
        }
        return [
            deepcopy(skill)
            for skill in skills
            if skill.skill_name in assigned
            or family_of_skill(skill) in allowed
        ]
    return [
        deepcopy(skill)
        for skill in skills
        if family_of_skill(skill) in allowed
    ]


def build_cold_start_bundle(
    *,
    organization_path: str | Path,
    skill_bank_path: str | Path,
    output_dir: str | Path,
    eval_summary: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    min_delta_sr: float = 0.0,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    enable_table = resolve_enable_table(
        eval_summary=eval_summary,
        overrides=overrides,
        min_delta_sr=min_delta_sr,
    )
    agents = load_agents(organization_path)
    skills = load_skills(skill_bank_path)
    filtered_agents = filter_cold_start_organization(
        agents,
        enable_table=enable_table,
    )
    filtered_skills = filter_cold_start_skills(
        skills,
        enable_table=enable_table,
        organization=filtered_agents,
    )
    org_path = output / "organization.json"
    bank_path = output / "skill_bank.json"
    enable_path = output / "family_enable.json"
    write_json(org_path, {"agents": to_primitive(filtered_agents)})
    write_json(bank_path, {"skills": to_primitive(filtered_skills)})
    manifest = {
        "probe": "cold_start_bundle",
        "source_organization": str(organization_path),
        "source_skill_bank": str(skill_bank_path),
        "min_delta_sr": min_delta_sr,
        "family_enable": enable_table,
        "enabled_task_families": enabled_families(enable_table),
        "disabled_task_families": disabled_families(enable_table),
        "n_agents": len(filtered_agents),
        "n_skills": len(filtered_skills),
        "agent_names": [agent.name for agent in filtered_agents],
        "skill_names": [skill.skill_name for skill in filtered_skills],
        "organization": str(org_path),
        "skill_bank": str(bank_path),
        "family_enable_path": str(enable_path),
    }
    write_json(enable_path, enable_table)
    write_json(output / "manifest.json", manifest)
    return manifest
