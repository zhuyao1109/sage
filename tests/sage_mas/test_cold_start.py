"""Unit tests for cold-start family enable filtering and dispatch gates."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sage_mas.cold_start import (
    DEFAULT_FAMILY_ENABLE,
    apply_admit_decision_to_family_enable,
    apply_admit_decisions_to_family_enable,
    build_cold_start_bundle,
    dispatch_lists_from_family_enable,
    enable_table_from_eval_summary,
    enabled_families,
    family_enable_from_dispatch_lists,
    filter_cold_start_organization,
)
from sage_mas.executor_dispatch import ExecutorDispatchConfig, ExecutorDispatcher
from sage_mas.schemas import AgentSpec, Skill, SkillStatus
from sage_mas.serialization import write_json


def _agent(name: str, family: str | None, skills: list[str]) -> AgentSpec:
    record = {"acting_status": "accepted", "status": "accepted"}
    if family:
        record["task_families"] = [family]
        record["dispatch_only"] = True
    return AgentSpec(
        name=name,
        role="specialist" if family else "Environment executor",
        responsibilities=["act"],
        assigned_skills=skills,
        tool_permissions=["alfworld_action"],
        activation_condition="always",
        shadow_evaluation_record=record,
    )


def _skill(name: str, family: str) -> Skill:
    return Skill(
        skill_name=name,
        capability_key=f"test.{family}",
        description=name,
        precondition="task active",
        action_protocol=["go to <location>"],
        applicable_atomic_ops=[],
        expected_effect="progress",
        status=SkillStatus.VERIFIED,
        applicable_task_families=[family],
        metadata={"primary_task_family": family},
    )


class ColdStartBundleTests(unittest.TestCase):
    def test_default_enable_keeps_positive_families(self) -> None:
        enabled = enabled_families(DEFAULT_FAMILY_ENABLE)
        self.assertIn("look_at_obj_in_light", enabled)
        self.assertIn("pick_cool_then_place_in_recep", enabled)
        self.assertNotIn("pick_and_place", enabled)
        self.assertNotIn("pick_two_obj_and_place", enabled)

    def test_enable_table_from_eval_summary(self) -> None:
        summary = {
            "families": [
                {"family": "pick_and_place", "delta_sr": -0.25},
                {"family": "look_at_obj_in_light", "delta_sr": 0.375},
                {"family": "pick_clean_then_place_in_recep", "delta_sr": 0.0},
            ]
        }
        table = enable_table_from_eval_summary(summary, min_delta_sr=0.0)
        self.assertFalse(table["pick_and_place"]["enabled"])
        self.assertTrue(table["look_at_obj_in_light"]["enabled"])
        self.assertTrue(table["pick_clean_then_place_in_recep"]["enabled"])

    def test_filter_organization_drops_negative_specialists(self) -> None:
        agents = [
            _agent("Executor", None, ["light", "place"]),
            _agent("LightSpec", "look_at_obj_in_light", ["light"]),
            _agent("PlaceSpec", "pick_and_place", ["place"]),
        ]
        filtered = filter_cold_start_organization(
            agents,
            enable_table=DEFAULT_FAMILY_ENABLE,
        )
        names = [agent.name for agent in filtered]
        self.assertEqual(names, ["Executor", "LightSpec"])
        executor = filtered[0]
        self.assertEqual(executor.assigned_skills, ["light"])

    def test_build_bundle_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            org = [
                _agent("Executor", None, ["light", "place"]),
                _agent("LightSpec", "look_at_obj_in_light", ["light"]),
                _agent("PlaceSpec", "pick_and_place", ["place"]),
            ]
            skills = [
                _skill("light", "look_at_obj_in_light"),
                _skill("place", "pick_and_place"),
            ]
            write_json(root / "organization.json", {"agents": org})
            write_json(root / "skill_bank.json", {"skills": skills})
            out = root / "bundle"
            manifest = build_cold_start_bundle(
                organization_path=root / "organization.json",
                skill_bank_path=root / "skill_bank.json",
                output_dir=out,
                eval_summary=None,
                min_delta_sr=0.0,
            )
            self.assertEqual(manifest["n_agents"], 2)
            self.assertEqual(manifest["n_skills"], 1)
            self.assertTrue((out / "organization.json").exists())
            self.assertTrue((out / "skill_bank.json").exists())
            self.assertTrue((out / "family_enable.json").exists())


class FamilyDispatchGateTests(unittest.TestCase):
    def test_disabled_family_keeps_executor(self) -> None:
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(
                prefer_matching_specialist=True,
                disabled_task_families=["pick_and_place"],
                enabled_task_families=[
                    "look_at_obj_in_light",
                    "pick_clean_then_place_in_recep",
                ],
            )
        )
        agents = [
            _agent("Executor", None, []),
            _agent("PlaceSpec", "pick_and_place", ["place"]),
        ]
        skills = [_skill("place", "pick_and_place")]
        assignment = dispatcher.assign(
            task="put soap in toilet",
            task_family="pick_and_place",
            agents=agents,
            skills=skills,
        )
        self.assertEqual(assignment.primary_agent, "Executor")
        self.assertEqual(assignment.dispatch_layer, "eligibility_empty")

    def test_enabled_family_routes_specialist(self) -> None:
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(
                prefer_matching_specialist=True,
                enabled_task_families=["look_at_obj_in_light"],
                disabled_task_families=["pick_and_place"],
            )
        )
        agents = [
            _agent("Executor", None, []),
            _agent("LightSpec", "look_at_obj_in_light", ["light"]),
        ]
        skills = [_skill("light", "look_at_obj_in_light")]
        assignment = dispatcher.assign(
            task="look at book under desklamp",
            task_family="look_at_obj_in_light",
            agents=agents,
            skills=skills,
        )
        self.assertEqual(assignment.primary_agent, "LightSpec")
        self.assertIn(assignment.dispatch_layer, {"eligibility_single", "prefer_matching_specialist"})


class AdmitFamilyEnableTests(unittest.TestCase):
    def test_admit_pass_enables_family(self) -> None:
        table = family_enable_from_dispatch_lists(
            enabled=["look_at_obj_in_light"],
            disabled=["pick_two_obj_and_place"],
        )
        table = apply_admit_decision_to_family_enable(
            table,
            family="pick_heat_then_place_in_recep",
            accepted=True,
            specialist_sr=0.5,
            executor_sr=0.25,
            agent_name="HeatSpec",
            source="actor_promotion",
        )
        enabled, disabled = dispatch_lists_from_family_enable(table)
        self.assertIn("pick_heat_then_place_in_recep", enabled)
        self.assertIn("look_at_obj_in_light", enabled)
        self.assertIn("pick_two_obj_and_place", disabled)
        self.assertTrue(table["pick_heat_then_place_in_recep"]["enabled"])
        self.assertEqual(table["pick_heat_then_place_in_recep"]["delta_sr"], 0.25)

    def test_admit_fail_disables_family(self) -> None:
        table = family_enable_from_dispatch_lists(
            enabled=["pick_two_obj_and_place", "look_at_obj_in_light"],
        )
        table = apply_admit_decision_to_family_enable(
            table,
            family="pick_two_obj_and_place",
            accepted=False,
            specialist_sr=0.5,
            executor_sr=0.5,
            agent_name="PickTwoSpec",
        )
        enabled, disabled = dispatch_lists_from_family_enable(table)
        self.assertNotIn("pick_two_obj_and_place", enabled)
        self.assertIn("pick_two_obj_and_place", disabled)
        self.assertIn("look_at_obj_in_light", enabled)

    def test_batch_decisions_from_actor_promotion(self) -> None:
        agents = [
            _agent("Executor", None, []),
            _agent("LightSpec", "look_at_obj_in_light", ["light"]),
            _agent("PickTwoSpec", "pick_two_obj_and_place", ["picktwo"]),
        ]
        table, updates = apply_admit_decisions_to_family_enable(
            {},
            agents=agents,
            decisions=[
                {
                    "agent": "LightSpec",
                    "accepted": True,
                    "specialist_sr": 0.75,
                    "executor_sr": 0.5,
                    "n_tasks": 8,
                },
                {
                    "agent": "PickTwoSpec",
                    "accepted": False,
                    "specialist_sr": 0.5,
                    "executor_sr": 0.5,
                    "n_tasks": 8,
                },
            ],
            source="actor_promotion",
        )
        self.assertEqual(len(updates), 2)
        enabled, disabled = dispatch_lists_from_family_enable(table)
        self.assertEqual(enabled, ["look_at_obj_in_light"])
        self.assertEqual(disabled, ["pick_two_obj_and_place"])

        # Allowlist: only admitted families can receive specialists.
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(
                enabled_task_families=enabled,
                disabled_task_families=disabled,
            )
        )
        light = dispatcher.assign(
            task="look at book under desklamp",
            task_family="look_at_obj_in_light",
            agents=agents,
            skills=[_skill("light", "look_at_obj_in_light")],
        )
        self.assertEqual(light.primary_agent, "LightSpec")
        picktwo = dispatcher.assign(
            task="put two pillows in sofa",
            task_family="pick_two_obj_and_place",
            agents=agents,
            skills=[_skill("picktwo", "pick_two_obj_and_place")],
        )
        self.assertEqual(picktwo.primary_agent, "Executor")


if __name__ == "__main__":
    unittest.main()
