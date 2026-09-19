"""Tests for trajectory-derived executable protocols and soft step alignment."""

from __future__ import annotations

import unittest

from sage_mas.executable_protocol import (
    ensure_executable_protocol,
    infer_current_step_index,
    parse_admissible_actions,
    protocol_adherence_score,
    render_current_step_guidance,
    soft_rank_admissible,
    steps_from_trajectory,
)
from sage_mas.organization import OrganizationEditor, OrganizationPolicy
from sage_mas.schemas import (
    AgentSpec,
    AtomicOp,
    OrganizationEditType,
    Skill,
    SkillStatus,
)
from sage_mas.shadow_evaluation import ShadowEvaluator
from sage_mas.schemas import ShadowMetrics
from sage_mas.skill_credit import (
    SkillCreditPolicy,
    initialize_skill_credit,
    update_skill_credits,
)
from sage_mas.alfworld_evaluator import EvaluationTrial
from sage_mas.assignment_gap import AssignmentGapResult


class ExecutableProtocolTests(unittest.TestCase):
    def test_steps_from_trajectory_and_current_guidance(self) -> None:
        records = [
            {"action": "take mug from countertop", "observation": "You pick up the mug"},
            {"action": "go to sinkbasin 1", "observation": "You arrive at sinkbasin"},
            {
                "action": "clean mug with sinkbasin",
                "observation": "You clean the mug with the sinkbasin",
            },
            {"action": "put mug in cabinet", "observation": "You put the mug"},
        ]
        steps = steps_from_trajectory(records)
        self.assertGreaterEqual(len(steps), 3)
        self.assertEqual(steps[0].verb, "take")
        self.assertEqual(steps[0].action_template, "take <target> from <source>")
        self.assertTrue(all("<" in step.action_template for step in steps))
        # Duplicate consecutive concrete actions collapse after slotting.
        duped = steps_from_trajectory(
            records
            + [{"action": "put mug in cabinet", "observation": "You put the mug again"}]
        )
        self.assertEqual(len(duped), len(steps))

        skill = Skill(
            skill_name="clean_proto",
            description="d",
            precondition="p",
            action_protocol=[],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
        )
        ensure_executable_protocol(skill, trajectory_steps=records)
        self.assertTrue(skill.metadata.get("executable_protocol"))
        self.assertFalse(
            any(
                "mug" in (row.get("action_template") or "")
                for row in skill.metadata["executable_protocol"]
            )
        )

        obs = (
            "You are holding a mug.\n"
            "Your admissible actions of the current situation are: "
            "['clean mug with sinkbasin', 'put mug in cabinet', 'look']."
        )
        history = [
            {"action": "take mug from countertop"},
            {"action": "go to sinkbasin 1"},
        ]
        index = infer_current_step_index(
            skill, history_steps=history, observation=obs
        )
        self.assertGreaterEqual(index, 1)
        guidance = render_current_step_guidance(
            [skill],
            observation=obs,
            history_steps=history,
            task="clean some mug and put it in cabinet.",
        )
        self.assertIn("Executable protocol guidance", guidance)
        self.assertIn("Bound for this task", guidance)
        self.assertIn("Closest admissible actions", guidance)
        admissible = parse_admissible_actions(obs)
        ranked = soft_rank_admissible("clean <target> with <tool>", admissible)
        self.assertEqual(ranked[0], "clean mug with sinkbasin")

    def test_mismatch_when_take_differs_from_target(self) -> None:
        from sage_mas.executable_protocol import protocol_mismatch_lines

        lines = protocol_mismatch_lines(
            target="apple",
            history_steps=[{"action": "take cup 1 from cabinet 1"}],
        )
        self.assertTrue(any("Mismatch" in line for line in lines))

    def test_role_slot_rejects_instance_overfit_cache(self) -> None:
        skill = Skill(
            skill_name="overfit",
            description="d",
            precondition="p",
            action_protocol=["take <object> from <receptacle>"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={
                "executable_protocol": [
                    {
                        "index": 0,
                        "action_template": "take cup 1 from cabinet 1",
                        "expected_obs_hint": "",
                        "verb": "take",
                    }
                ],
                "executable_protocol_source": "trajectory",
            },
        )
        steps = ensure_executable_protocol(skill)
        self.assertEqual(steps[0].action_template, "take <target> from <source>")

    def test_protocol_adherence_in_order(self) -> None:
        skill = Skill(
            skill_name="heat_proto",
            description="d",
            precondition="p",
            action_protocol=[
                "take apple from fridge",
                "heat apple with microwave",
                "put apple in fridge",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
        )
        ensure_executable_protocol(skill)
        trial_steps = [
            {"action": "take apple from fridge"},
            {"action": "go to microwave"},
            {"action": "heat apple with microwave"},
            {"action": "put apple in fridge"},
        ]
        score = protocol_adherence_score(skill, trial_steps)
        self.assertGreaterEqual(score, 0.99)

    def test_credit_requires_adherence_and_benefit(self) -> None:
        skill = Skill(
            skill_name="clean_proto",
            description="d",
            precondition="p",
            action_protocol=[
                "take mug from countertop",
                "clean mug with sinkbasin",
                "put mug in cabinet",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.PROVISIONAL,
            applicable_task_families=["pick_clean_then_place_in_recep"],
            capability_key="transform.clean",
            metadata={
                "capability_operation": "clean",
                "primary_task_family": "pick_clean_then_place_in_recep",
            },
        )
        ensure_executable_protocol(skill)
        initialize_skill_credit(skill, SkillCreditPolicy())
        agents = [
            AgentSpec(
                name="Executor",
                role="executor",
                responsibilities=["act"],
                assigned_skills=[skill.skill_name],
            )
        ]
        good = EvaluationTrial(
            task_id="t1",
            task="put a clean mug in cabinet",
            task_family="pick_clean_then_place_in_recep",
            condition="train",
            won=True,
            reward=1.0,
            cost=10.0,
            num_steps=3,
            steps=[
                {"action": "take mug from countertop"},
                {"action": "clean mug with sinkbasin"},
                {"action": "put mug in cabinet"},
            ],
            activated_skill_names=[skill.skill_name],
            skill_activation_steps={skill.skill_name: 1},
            assigned_primary_agent="Executor",
            actions_by_agent={"Executor": 3},
        )
        bad = EvaluationTrial(
            task_id="t2",
            task="put a clean mug in cabinet",
            task_family="pick_clean_then_place_in_recep",
            condition="train",
            won=False,
            reward=0.0,
            cost=10.0,
            num_steps=2,
            steps=[
                {"action": "look"},
                {"action": "inventory"},
            ],
            activated_skill_names=[skill.skill_name],
            skill_activation_steps={skill.skill_name: 1},
            assigned_primary_agent="Executor",
            actions_by_agent={"Executor": 2},
        )
        result = update_skill_credits(
            [skill],
            [good, bad],
            agents,
            SkillCreditPolicy(
                min_adherence_for_use=0.3,
                min_adherence_for_verify=0.3,
                min_uses_for_promotion=1,
                verify_score=0.5,
            ),
        )
        credit = skill.metadata["skill_credit"]
        self.assertGreaterEqual(int(credit["uses"]), 1)
        self.assertIn("mean_protocol_adherence", credit)
        self.assertEqual(skill.status, SkillStatus.VERIFIED)
        self.assertTrue(result["promoted"])

    def test_add_agent_blocked_without_executable_protocol(self) -> None:
        skill = Skill(
            skill_name="empty_proto",
            description="d",
            precondition="p",
            action_protocol=[],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            capability_key="transform.clean",
            support_count=5,
            evidence_ids=["e1", "e2", "e3"],
            metadata={"primary_task_family": "pick_clean_then_place_in_recep"},
        )
        editor = OrganizationEditor(
            OrganizationPolicy(
                require_executable_protocol=True,
                min_assignment_gap=0.0,
                distribution_shift_threshold=0.0,
                min_cluster_support=1,
                allow_bootstrap_add=True,
            )
        )
        from sage_mas.distribution_stats import DistributionShift

        edit = editor.propose(
            [skill],
            AssignmentGapResult(
                skill_ids=[skill.skill_id],
                assignment_gap=1.0,
                best_agent_id=None,
                best_agent_name="Executor",
                best_capacity=0.0,
                capacities={"Executor": 0.0},
            ),
            existing_agent_names={"Executor"},
            existing_agents=[
                AgentSpec(
                    name="Executor",
                    role="executor",
                    responsibilities=["act"],
                )
            ],
            cluster_shift=DistributionShift(
                shift_score=1.0,
                novelty_score=1.0,
                metric="cosine_nearest_skill",
                baseline_source="empty_skill_bank",
            ),
            bootstrap_slots_remaining=1,
            add_slots_remaining=1,
        )
        self.assertNotEqual(edit.edit_type, OrganizationEditType.ADD_AGENT)
        self.assertIn("executable protocol", edit.rationale.lower())

    def test_ground_skill_builds_trajectory_executable_protocol(self) -> None:
        from sage_mas.schemas import AtomicOp, Skill, SkillStatus
        from sage_mas.trajectory_adapter import AlfWorldTrajectoryAdapter
        from sage_mas.trajectory.grounding import (
            ground_skill_in_successful_trajectory,
        )
        from sage_mas.executable_protocol import get_executable_steps

        adapter = AlfWorldTrajectoryAdapter()
        trajectories = adapter.adapt_many(
            [
                {
                    "gamefile": "/tmp/pick_clean_then_place_in_recep-full/game.tw-pddl",
                    "task": "put a clean mug on the desk",
                    "won": True,
                    "num_steps": 7,
                    "steps": [
                        {
                            "observation": "You arrive at countertop 1.",
                            "action": "go to countertop 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "Taken.",
                            "action": "take mug 1 from countertop 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "You arrive at sinkbasin 1.",
                            "action": "go to sinkbasin 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "You clean the mug 1 using the sinkbasin 1.",
                            "action": "clean mug 1 with sinkbasin 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "You arrive at desk 1.",
                            "action": "go to desk 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "You move the mug 1 to the desk 1.",
                            "action": "move mug 1 to desk 1",
                            "is_action_valid": True,
                        },
                    ],
                }
            ]
        )
        seed = Skill(
            skill_name="seed",
            description="seed",
            precondition="seed",
            action_protocol=["clean <object> with <tool>"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.CANDIDATE,
            capability_key="transform.clean",
            metadata={"source_signal": "clean_operation_success"},
        )
        grounded = ground_skill_in_successful_trajectory(seed, trajectories)
        steps = get_executable_steps(grounded)
        self.assertGreaterEqual(len(steps), 4)
        self.assertEqual(
            grounded.metadata.get("executable_protocol_source"),
            "trajectory",
        )
        verbs = [step.verb for step in steps]
        self.assertIn("go", verbs)
        self.assertIn("take", verbs)
        self.assertIn("clean", verbs)
        self.assertIn("move", verbs)
        self.assertTrue(
            grounded.metadata.get("executable_protocol_includes_find_prefix")
        )
        self.assertIn("find", grounded.metadata.get("protocol_stages") or [])

    def test_add_agent_blocked_without_positive_mu(self) -> None:
        skill = Skill(
            skill_name="clean_proto",
            description="d",
            precondition="p",
            action_protocol=[
                "go to countertop 1",
                "take mug 1 from countertop 1",
                "clean mug 1 with sinkbasin 1",
                "move mug 1 to cabinet 1",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            capability_key="transform.clean",
            support_count=5,
            evidence_ids=["e1", "e2", "e3"],
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "executable_protocol": [
                    {
                        "index": 0,
                        "action_template": "take mug 1 from countertop 1",
                        "expected_obs_hint": "",
                        "verb": "take",
                    },
                    {
                        "index": 1,
                        "action_template": "clean mug 1 with sinkbasin 1",
                        "expected_obs_hint": "",
                        "verb": "clean",
                    },
                    {
                        "index": 2,
                        "action_template": "move mug 1 to cabinet 1",
                        "expected_obs_hint": "",
                        "verb": "move",
                    },
                ],
                "executable_protocol_source": "trajectory",
            },
        )
        editor = OrganizationEditor(
            OrganizationPolicy(
                require_executable_protocol=True,
                require_trajectory_executable_protocol=False,
                require_positive_mu_for_add_agent=True,
                min_assignment_gap=0.0,
                distribution_shift_threshold=0.0,
                min_cluster_support=1,
                allow_bootstrap_add=True,
            )
        )
        from sage_mas.distribution_stats import DistributionShift

        edit = editor.propose(
            [skill],
            AssignmentGapResult(
                skill_ids=[skill.skill_id],
                assignment_gap=1.0,
                best_agent_id=None,
                best_agent_name="Executor",
                best_capacity=0.0,
                capacities={"Executor": 0.0},
            ),
            existing_agent_names={"Executor"},
            existing_agents=[
                AgentSpec(
                    name="Executor",
                    role="executor",
                    responsibilities=["act"],
                )
            ],
            cluster_shift=DistributionShift(
                shift_score=1.0,
                novelty_score=1.0,
                metric="cosine_nearest_skill",
                baseline_source="embedding_skill_bank",
            ),
            bootstrap_slots_remaining=1,
            add_slots_remaining=1,
        )
        self.assertNotEqual(edit.edit_type, OrganizationEditType.ADD_AGENT)
        self.assertIn("positive paired mu", edit.rationale.lower())

    def test_shadow_utility_blends_adherence(self) -> None:
        evaluator = ShadowEvaluator(cost_weight=0.0, significance_threshold=0.0)
        high = evaluator.utility(
            ShadowMetrics(success_rate=1.0, token_cost=0.0, protocol_adherence=1.0)
        )
        low = evaluator.utility(
            ShadowMetrics(success_rate=1.0, token_cost=0.0, protocol_adherence=0.0)
        )
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()
