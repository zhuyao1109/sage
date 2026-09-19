"""Tests for dual-bank roles, idle filters, compiler, and ρ_mini coverage."""

from __future__ import annotations

import unittest

from sage_mas.executable_coverage import (
    cluster_ready_for_agent_nomination,
    executable_coverage_for_cluster,
)
from sage_mas.idle_trajectory import filter_informative_failures, is_idle_failure
from sage_mas.organization import OrganizationEditor, OrganizationPolicy
from sage_mas.schemas import Skill, SkillStatus
from sage_mas.skill_bank_roles import (
    ROLE_CANONICAL,
    ROLE_EXECUTABLE,
    mark_canonical,
    mark_executable,
    skill_bank_role,
)


def _skill(**kwargs) -> Skill:
    base = dict(
        skill_name="Merged transform heat protocol",
        description="canonical draft",
        precondition="Task family matches heat.",
        action_protocol=[
            "go to <location>",
            "open <entity>",
            "take <object> from <receptacle>",
            "heat <object> with <tool>",
            "move <object> to <receptacle>",
        ],
        applicable_atomic_ops=["Act"],
        expected_effect="won",
        suggested_role="Executor",
        target_failure_types=[],
        applicable_task_families=["pick_heat_then_place_in_recep"],
        capability_key="transform.heat",
        status=SkillStatus.PROVISIONAL,
        metadata={
            "primary_task_family": "pick_heat_then_place_in_recep",
            "capability_key": "transform.heat",
            "protocol_form_ok": True,
            "protocol_structure_ok": True,
            "protocol_alignment_ok": True,
            "protocol_alignment_support": 3,
        },
    )
    base.update(kwargs)
    return Skill(**base)


def _traj(family: str, won: bool, actions: list[str], gamefile: str) -> dict:
    return {
        "task_family": family,
        "won": won,
        "gamefile": gamefile,
        "steps": [{"action": a, "observation": "ok"} for a in actions],
    }


class DualBankFlowTests(unittest.TestCase):
    def test_mark_canonical_blocks_injection_role(self) -> None:
        skill = mark_canonical(_skill(), teacher_model="gemini-2.5-pro")
        self.assertEqual(skill_bank_role(skill), ROLE_CANONICAL)
        self.assertTrue(skill.metadata["not_for_injection"])

    def test_idle_failure_filtered(self) -> None:
        idle = _traj(
            "pick_heat_then_place_in_recep",
            False,
            ["go to fridge 1"] * 10,
            "g_idle",
        )
        useful = _traj(
            "pick_heat_then_place_in_recep",
            False,
            [
                "go to fridge 1",
                "open fridge 1",
                "take apple 1 from fridge 1",
                "go to cabinet 1",
                "go to microwave 1",
            ],
            "g_useful",
        )
        self.assertTrue(is_idle_failure(idle))
        self.assertFalse(is_idle_failure(useful))
        kept = filter_informative_failures([idle, useful])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["gamefile"], "g_useful")

    def test_shadow_rejects_low_call_rate(self) -> None:
        from sage_mas.schemas import ShadowMetrics
        from sage_mas.shadow_evaluation import ShadowEvaluator

        evaluator = ShadowEvaluator(
            significance_threshold=0.0,
            min_new_agent_call_rate=0.5,
        )
        old = ShadowMetrics(success_rate=0.2, token_cost=10.0)
        new = ShadowMetrics(
            success_rate=0.5,
            token_cost=10.0,
            new_agent_call_rate=0.1,
        )
        decision = evaluator.decide(old, new, paired_deltas=[0.3])
        self.assertFalse(decision.accepted)
        self.assertIn("call_rate", decision.reason)

        new_ok = ShadowMetrics(
            success_rate=0.5,
            token_cost=10.0,
            new_agent_call_rate=0.8,
        )
        decision_ok = evaluator.decide(old, new_ok, paired_deltas=[0.3])
        self.assertTrue(decision_ok.accepted)

    def test_specialist_context_is_exclusive_skill_protocols(self) -> None:
        from sage_mas.capability_contract import (
            AgentRoleCompiler,
            compile_capability_contract,
        )

        skill = mark_executable(
            _skill(
                skill_name="exec heat",
                status=SkillStatus.VERIFIED,
                marginal_utility=0.25,
                applicable_atomic_ops=[],
            ),
            delta_sr=0.25,
        )
        skill.status = SkillStatus.VERIFIED
        context = AgentRoleCompiler.exclusive_skill_context([skill])
        self.assertIn("exec heat", context)
        self.assertNotIn("Check visibility", context)
        agent = AgentRoleCompiler().compile(
            compile_capability_contract([skill]),
            set(),
            skills=[skill],
        )
        self.assertTrue(
            (agent.shadow_evaluation_record or {}).get("exclusive_skill_context")
        )
        self.assertIn("Exclusive protocols", agent.role_specification)

    def test_specialist_brief_includes_trajectory_demos(self) -> None:
        from sage_mas.capability_contract import (
            AgentRoleCompiler,
            compile_capability_contract,
        )
        from sage_mas.specialist_context import attach_trajectory_context_to_skills

        skill = mark_executable(
            _skill(
                skill_name="exec heat",
                status=SkillStatus.VERIFIED,
                marginal_utility=0.25,
                applicable_atomic_ops=[],
            ),
            delta_sr=0.25,
        )
        skill.status = SkillStatus.VERIFIED
        teacher = _traj(
            "pick_heat_then_place_in_recep",
            True,
            [
                "go to fridge 1",
                "open fridge 1",
                "take apple 1 from fridge 1",
                "go to microwave 1",
                "heat apple 1 with microwave 1",
                "move apple 1 to countertop 1",
            ],
            "g_demo",
        )
        student = _traj(
            "pick_heat_then_place_in_recep",
            False,
            [
                "go to fridge 1",
                "open fridge 1",
                "go to fridge 1",
                "go to fridge 1",
            ],
            "g_demo",
        )
        enriched = attach_trajectory_context_to_skills(
            [skill],
            teacher_records=[teacher],
            student_records=[student],
        )[0]
        self.assertTrue(enriched.metadata.get("specialist_demos"))
        agent = AgentRoleCompiler().compile(
            compile_capability_contract([enriched]),
            set(),
            skills=[enriched],
            teacher_records=[teacher],
            student_records=[student],
        )
        spec = agent.role_specification.lower()
        self.assertIn("success demos", spec)
        self.assertIn("heat apple", spec)
        self.assertNotIn("check visibility", spec)
        self.assertNotIn("inventory space", spec)
        traj_ctx = (agent.shadow_evaluation_record or {}).get("trajectory_context") or {}
        self.assertGreaterEqual(int(traj_ctx.get("n_demos") or 0), 1)
        self.assertEqual(
            (agent.shadow_evaluation_record or {}).get("specialist_context_source"),
            "trajectory_demos",
        )

    def test_rho_mini_and_org_gate(self) -> None:
        canonical = mark_canonical(_skill())
        executable = mark_executable(
            _skill(skill_name="exec heat", marginal_utility=0.25),
            delta_sr=0.25,
        )
        executable.metadata["inject_ready"] = True
        executable.marginal_utility = 0.25
        executable.metadata["marginal_utility"] = 0.25
        executable.metadata["skill_bank_role"] = ROLE_EXECUTABLE

        cov = executable_coverage_for_cluster([canonical, executable])
        self.assertEqual(cov["n_cluster"], 2)
        self.assertEqual(cov["n_executable"], 1)
        self.assertAlmostEqual(cov["rho_mini"], 0.5)

        ready, stats = cluster_ready_for_agent_nomination(
            [canonical, executable], min_rho=0.5
        )
        self.assertTrue(ready)
        self.assertTrue(stats["ready"])

        editor = OrganizationEditor(
            OrganizationPolicy(
                require_positive_mu_for_add_agent=True,
                require_executable_bank_for_add_agent=True,
                min_executable_coverage_for_add_agent=0.5,
                require_executable_protocol=False,
                min_protocol_adherence_for_add_agent=0.0,
            )
        )
        # Canonical-only cluster blocked.
        reason = editor._skills_ready_for_add_agent([canonical])
        self.assertIsNotNone(reason)
        # Mixed cluster with ρ>=0.5 and executable MU ok.
        executable.status = SkillStatus.VERIFIED
        reason2 = editor._skills_ready_for_add_agent([canonical, executable])
        self.assertIsNone(reason2)


if __name__ == "__main__":
    unittest.main()
