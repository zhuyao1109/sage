"""Unit tests for llm_only vs contract Executor dispatch routing."""

from __future__ import annotations

import unittest

from sage_mas.executor_dispatch import (
    ExecutorDispatchConfig,
    ExecutorDispatcher,
    dispatch_config_from_mapping,
)
from sage_mas.schemas import AgentSpec, AtomicOp, Skill, SkillStatus


def _agents_and_skills():
    heat = Skill(
        skill_name="Heat protocol",
        description="Heat objects",
        precondition="Need heat",
        action_protocol=["heat"],
        applicable_atomic_ops=[AtomicOp.ACT],
        status=SkillStatus.VERIFIED,
        metadata={
            "primary_task_family": "pick_heat_then_place_in_recep",
            "task_families": ["pick_heat_then_place_in_recep"],
        },
    )
    clean = Skill(
        skill_name="Clean protocol",
        description="Clean objects",
        precondition="Need clean",
        action_protocol=["clean"],
        applicable_atomic_ops=[AtomicOp.ACT],
        status=SkillStatus.VERIFIED,
        metadata={
            "primary_task_family": "pick_clean_then_place_in_recep",
            "task_families": ["pick_clean_then_place_in_recep"],
        },
    )
    executor = AgentSpec(
        name="Executor",
        role="Environment executor",
        responsibilities=["Act"],
        tool_permissions=["alfworld_action"],
    )
    heater = AgentSpec(
        name="HeatSpecialist",
        role="HeatSpecialist",
        responsibilities=["Heat"],
        assigned_skills=[heat.skill_name],
        tool_permissions=["alfworld_action"],
        shadow_evaluation_record={
            "acting_status": "accepted",
            "task_families": ["pick_heat_then_place_in_recep"],
        },
    )
    cleaner = AgentSpec(
        name="CleanSpecialist",
        role="CleanSpecialist",
        responsibilities=["Clean"],
        assigned_skills=[clean.skill_name],
        tool_permissions=["alfworld_action"],
        shadow_evaluation_record={
            "acting_status": "accepted",
            "task_families": ["pick_clean_then_place_in_recep"],
        },
    )
    return executor, heater, cleaner, heat, clean


class FakePickBackend:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def complete(self, system_prompt, user_prompt):
        from sage_mas.runtime import LLMResult

        self.calls += 1
        return LLMResult(
            f"<assign>{self.name}</assign>",
            prompt_tokens=1,
            completion_tokens=1,
        )


class TestLlmOnlyDispatch(unittest.TestCase):
    def test_contract_uses_eligibility_single(self):
        executor, heater, cleaner, heat, clean = _agents_and_skills()
        assignment = ExecutorDispatcher(
            ExecutorDispatchConfig(
                prefer_matching_specialist=True,
                auto_assign_single=True,
                llm_only_dispatch=False,
            ),
            backend=None,
        ).assign(
            task="heat an apple and put it on the table",
            task_family="pick_heat_then_place_in_recep",
            agents=[executor, heater, cleaner],
            skills=[heat, clean],
            gamefile="/fake/pick_heat_then_place_in_recep-0/game.tw-pddl",
        )
        self.assertEqual(assignment.primary_agent, "HeatSpecialist")
        self.assertEqual(assignment.dispatch_layer, "eligibility_single")
        self.assertEqual(assignment.eligible_agents, ["HeatSpecialist"])

    def test_llm_only_opens_roster_and_calls_llm(self):
        executor, heater, cleaner, heat, clean = _agents_and_skills()
        backend = FakePickBackend("CleanSpecialist")
        assignment = ExecutorDispatcher(
            ExecutorDispatchConfig(
                prefer_matching_specialist=False,
                auto_assign_single=False,
                llm_only_dispatch=True,
            ),
            backend=backend,
        ).assign(
            task="heat an apple and put it on the table",
            task_family="pick_heat_then_place_in_recep",
            agents=[executor, heater, cleaner],
            skills=[heat, clean],
            gamefile="/fake/pick_heat_then_place_in_recep-0/game.tw-pddl",
        )
        self.assertEqual(backend.calls, 1)
        self.assertEqual(assignment.primary_agent, "CleanSpecialist")
        self.assertEqual(assignment.dispatch_layer, "llm")
        self.assertEqual(
            set(assignment.eligible_agents),
            {"HeatSpecialist", "CleanSpecialist"},
        )

    def test_llm_only_can_keep_executor(self):
        executor, heater, cleaner, heat, clean = _agents_and_skills()
        backend = FakePickBackend("Executor")
        assignment = ExecutorDispatcher(
            ExecutorDispatchConfig(llm_only_dispatch=True),
            backend=backend,
        ).assign(
            task="heat an apple and put it on the table",
            task_family="pick_heat_then_place_in_recep",
            agents=[executor, heater, cleaner],
            skills=[heat, clean],
            gamefile="/fake/pick_heat_then_place_in_recep-0/game.tw-pddl",
        )
        self.assertEqual(assignment.primary_agent, "Executor")
        self.assertEqual(assignment.dispatch_layer, "llm_keep_executor")

    def test_dispatch_config_from_mapping_reads_flag(self):
        cfg = dispatch_config_from_mapping({"llm_only_dispatch": True})
        self.assertTrue(cfg.llm_only_dispatch)


if __name__ == "__main__":
    unittest.main()
