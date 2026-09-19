#!/usr/bin/env python3
"""Build multi-agent seed org + verified skill bank from Pro cold-start v6.

Dispatch eligibility requires SkillStatus.VERIFIED. Cold-start skills are
provisional, so a frozen multi-agent eval must promote a verified copy and
attach capability_contract.task_families for routing.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from uuid import uuid4

FAMILY_TO_AGENT: dict[str, str] = {
    "pick_clean_then_place_in_recep": "TransformCleanSpecialist",
    "pick_heat_then_place_in_recep": "TransformHeatSpecialist",
    "pick_cool_then_place_in_recep": "TransformCoolSpecialist",
    "look_at_obj_in_light": "InspectWithLightSpecialist",
    "pick_two_obj_and_place": "TrackMultipleObjectsSpecialist",
    "pick_and_place": "TrackPlaceSpecialist",
}

FAMILY_TO_CAPABILITY: dict[str, str] = {
    "pick_clean_then_place_in_recep": "transform.clean",
    "pick_heat_then_place_in_recep": "transform.heat",
    "pick_cool_then_place_in_recep": "transform.cool",
    "look_at_obj_in_light": "inspect.with_light",
    "pick_two_obj_and_place": "track.multiple_objects",
    "pick_and_place": "track.place",
}

AGENT_ORDER = [
    "TransformCleanSpecialist",
    "TransformHeatSpecialist",
    "TransformCoolSpecialist",
    "InspectWithLightSpecialist",
    "TrackMultipleObjectsSpecialist",
    "TrackPlaceSpecialist",
]


def _protocol_chain(action_protocol: list[str]) -> str:
    return " -> ".join(action_protocol)


def _capability_contract(skill: dict) -> dict:
    families = list(skill.get("applicable_task_families") or [])
    family = families[0]
    capability = FAMILY_TO_CAPABILITY[family]
    skill_id = str(skill.get("skill_id") or uuid4())
    return {
        "capability_id": f"capability-{skill_id[:12]}",
        "name": capability,
        "description": skill.get("description") or "",
        "preconditions": [skill.get("precondition") or ""],
        "action_protocols": [list(skill.get("action_protocol") or [])],
        "expected_effects": [str(skill.get("expected_effect") or "")],
        "atomic_ops": list(skill.get("applicable_atomic_ops") or ["Act"]),
        "required_tools": ["alfworld_action"],
        "skill_ids": [skill_id],
        "skill_names": [skill["skill_name"]],
        "evidence_ids": list(skill.get("evidence_ids") or []),
        "task_families": families,
    }


def _specialist_agent(skill: dict) -> dict:
    families = skill.get("applicable_task_families") or []
    if not families:
        raise ValueError(f"skill missing applicable_task_families: {skill.get('skill_name')}")
    family = families[0]
    name = FAMILY_TO_AGENT[family]
    skill_name = skill["skill_name"]
    protocol = _protocol_chain(skill.get("action_protocol") or [])
    contract = _capability_contract(skill)
    return {
        "name": name,
        "role": name,
        "responsibilities": [
            "Complete the assigned environment task end-to-end",
            "Follow exclusive trajectory-derived demos and skill protocols for this specialist only",
            "Return control to Executor when inactive",
        ],
        "assigned_skills": [skill_name],
        "tool_permissions": ["alfworld_action"],
        "activation_condition": f"Activated by Executor dispatch for {family} tasks.",
        "token_budget": None,
        "turn_budget": None,
        "role_specification": (
            f"You are {name}, a capability specialist. When dispatched, complete the full "
            f"environment task end-to-end using only admissible actions. Bind placeholders "
            f"from the current observation. Use the exclusive trajectory-derived context below; "
            f"do not invent off-scope strategies. Exclusive protocols: [{skill_name}] "
            f"{protocol}. Act only when Executor assigns this episode."
        ),
        "responsibility_boundary": (
            f"In scope: exclusive skills [{skill_name}] and their trajectory demos after dispatch. "
            "Out of scope: acting without dispatch or using skills outside this cluster."
        ),
        "input_protocol": "Input: ALFWorld observation, active skill protocols for this specialist.",
        "output_protocol": (
            "Output exactly one <think>brief reasoning</think>"
            "<action>admissible action</action> decision."
        ),
        "communication_edges": ["Executor"],
        "retirement_condition": None,
        "shadow_evaluation_record": {
            "status": "pending",
            "acting_status": "accepted",
            "dispatch_only": True,
            "capability_contract_id": contract["capability_id"],
            "capability_contract": contract,
        },
        "agent_id": str(uuid4()),
    }


def build_verified_bank(skill_bank_path: Path) -> dict:
    payload = json.loads(skill_bank_path.read_text())
    skills = []
    seen: set[str] = set()
    for skill in payload.get("skills") or []:
        name = skill.get("skill_name")
        if not name or name in seen:
            continue
        seen.add(name)
        item = copy.deepcopy(skill)
        item["status"] = "verified"
        # Ensure dispatch scope fields exist.
        families = list(item.get("applicable_task_families") or [])
        if families:
            meta = dict(item.get("metadata") or {})
            meta.setdefault("task_families", families)
            meta.setdefault("primary_task_family", families[0])
            item["metadata"] = meta
            item["capability_key"] = FAMILY_TO_CAPABILITY.get(
                families[0],
                item.get("capability_key") or name,
            )
        skills.append(item)
    return {"version": int(payload.get("version") or 1), "skills": skills}


def build_seed_organization(verified_bank: dict) -> dict:
    by_name = {skill["skill_name"]: skill for skill in verified_bank["skills"]}
    executor = {
        "name": "Executor",
        "role": "Environment executor",
        "responsibilities": [
            "Interpret ALFWorld observations",
            "Select and execute admissible environment actions",
            "Dispatch to matching specialist when task family matches a specialist capability",
        ],
        "assigned_skills": [],
        "tool_permissions": ["alfworld_action"],
        "activation_condition": "Always active during environment interaction.",
        "token_budget": None,
        "turn_budget": None,
        "role_specification": (
            "ALFWorld dispatcher/executor. Prefer matching specialists by task family; "
            "otherwise execute admissible actions directly."
        ),
        "responsibility_boundary": (
            "In scope: observation interpretation, specialist dispatch, fallback execution. "
            "Out of scope: monopolizing tasks that a matching specialist should handle."
        ),
        "input_protocol": "Input: ALFWorld observation template; optional active skill protocols.",
        "output_protocol": (
            "Output exactly one <think>brief reasoning</think>"
            "<action>admissible action</action> decision."
        ),
        "communication_edges": [],
        "retirement_condition": None,
        "shadow_evaluation_record": {
            "status": "baseline",
            "acting_status": "accepted",
        },
        "agent_id": str(uuid4()),
    }
    specialists = [_specialist_agent(skill) for skill in by_name.values()]
    specialists.sort(key=lambda item: AGENT_ORDER.index(item["name"]))
    return {"agents": [executor, *specialists]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skill-bank",
        default="logs/sage_mas/cold_start_60_gemini-2.5-pro_n10/skill_bank_v6_llm.json",
    )
    parser.add_argument(
        "--org-output",
        default="logs/sage_mas/cold_start_60_gemini-2.5-pro_n10/seed_organization.json",
    )
    parser.add_argument(
        "--bank-output",
        default=(
            "logs/sage_mas/cold_start_60_gemini-2.5-pro_n10/"
            "skill_bank_v6_verified_for_agents.json"
        ),
    )
    args = parser.parse_args()

    source = Path(args.skill_bank)
    org_path = Path(args.org_output)
    bank_path = Path(args.bank_output)

    verified = build_verified_bank(source)
    org = build_seed_organization(verified)

    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(verified, ensure_ascii=False, indent=2) + "\n")
    org_path.write_text(json.dumps(org, ensure_ascii=False, indent=2) + "\n")

    print(f"Wrote {bank_path} ({len(verified['skills'])} verified skills)")
    names = [agent["name"] for agent in org["agents"]]
    print(f"Wrote {org_path} ({len(names)} agents): {names}")
    for agent in org["agents"][1:]:
        families = (
            (agent.get("shadow_evaluation_record") or {})
            .get("capability_contract", {})
            .get("task_families")
        )
        print(f"  {agent['name']}: {agent['assigned_skills']} families={families}")


if __name__ == "__main__":
    main()
