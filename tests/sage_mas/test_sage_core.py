import tempfile
import unittest
import json
import yaml
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sage_mas.assignment_gap import (
    AssignmentGapEstimator,
    AssignmentGapResult,
)
from sage_mas.alfworld_evaluator import (
    AlfWorldEvaluatorConfig,
    AlfWorldOrganizationEvaluator,
    EvaluationTrial,
    select_prompt_agent_gamefiles,
)
from sage_mas.capability_contract import (
    AgentRoleCompiler,
    compile_capability_contract,
)
from sage_mas.distribution_stats import (
    aggregate_skill_bank_distribution,
    build_experience_distribution_snapshot,
    compare_experience_distribution_snapshots,
    compute_round_exploration_distribution,
    embedding_novelty_for_skill_cluster,
    estimate_distribution_shift,
    kl_divergence,
    shift_for_skill_cluster,
)
from sage_mas.discovery_fork import (
    CapabilityTransitionDetector,
    ConfirmedCapabilityTransition,
    DiscoveryForkOutcome,
    DiscoveryForkService,
)
from sage_mas.enriched_skill_distiller import EnrichedSkillDistiller
from sage_mas.evaluation_services import OrganizationReportingService
from sage_mas.organization import (
    OrganizationEditor,
    OrganizationPolicy,
    OrganizationStateManager,
    cluster_key_for_skill,
)
from sage_mas.runtime import LLMResult, MASRuntime, OpenAIChatBackend
from sage_mas.serialization import load_agents, load_skills, read_json, write_json, write_jsonl
from sage_mas.schemas import (
    AgentSpec,
    AtomicOp,
    DistributionShift,
    ExplorationDistribution,
    OrganizationEdit,
    OrganizationEditType,
    ShadowDecision,
    ShadowMetrics,
    Skill,
    SkillStatus,
    StateActionFragment,
)
from sage_mas.shadow_evaluation import ShadowEvaluator
from sage_mas.skill_bank import SkillBank
from sage_mas.skill_credit import (
    SkillCreditPolicy,
    initialize_skill_credit,
    update_skill_credits,
)
from sage_mas.skill_quality import (
    generic_skill_contract_reasons,
    trajectory_has_capability,
    validate_environment_confirmed_skill,
    validate_skill_evidence,
)
from sage_mas.skill_activation import SkillPreconditionMatcher
from sage_mas.skill_distiller import (
    DistillationConfig,
    HeuristicSkillDistiller,
    TrajectoryGroundedSkillDistiller,
)
from sage_mas.trajectory_adapter import AlfWorldTrajectoryAdapter
from sage_mas.trajectory.abstraction import stage_for_abstracted_action
from sage_mas.trajectory.causal import protocol_is_org_ready
from sage_mas.trajectory.finalize import (
    finalize_operation_protocol,
    lock_operation_suggested_role,
    prefer_success_trajectories,
)
from sage_mas.trajectory.fragments import embed_skill, extract_key_fragments
from sage_mas.trajectory.grounding import (
    extract_anti_patterns,
    ground_skill_in_successful_trajectory,
)
from sage_mas.trajectory.protocol_canon import (
    canonicalize_protocol_stages,
    protocol_structure_issues,
)
def _trajectory(
    game_id: str,
    won: bool = False,
    task: str = "put a mug on the desk",
):
    return {
        "gamefile": f"/tmp/{game_id}/game.tw-pddl",
        "task": task,
        "won": won,
        "num_steps": 3,
        "steps": [
            {"observation": "Nothing happens.", "action": "look", "is_action_valid": True},
            {"observation": "Nothing happens.", "action": "look", "is_action_valid": True},
            {"observation": "Nothing happens.", "action": "look", "is_action_valid": True},
        ],
    }


def _operation_pair(game_id: str, operation: str, task: str):
    failed = _trajectory(game_id + "-failed", task=task)
    successful = _trajectory(game_id + "-successful", won=True, task=task)
    successful["steps"] = [
        {
            "observation": "You arrive at the countertop.",
            "action": "go to countertop 1",
            "is_action_valid": True,
        },
        {
            "observation": "You pick up the target.",
            "action": "take mug 1 from countertop 1",
            "is_action_valid": True,
        },
        {
            "observation": "You arrive at the cabinet.",
            "action": "go to cabinet 1",
            "is_action_valid": True,
        },
        {
            "observation": "You arrive at the sinkbasin.",
            "action": "go to sinkbasin 1",
            "is_action_valid": True,
        },
        {
            "observation": f"The target is now {operation}ed.",
            "action": f"{operation} mug 1 with sinkbasin 1",
            "is_action_valid": True,
        },
        {
            "observation": "You arrive at the desk.",
            "action": "go to desk 1",
            "is_action_valid": True,
        },
        {
            "observation": "Task completed.",
            "action": "put mug 1 in/on desk 1",
            "is_action_valid": True,
        },
    ]
    return [failed, successful]


class SageCoreTest(unittest.TestCase):
    def test_adapter_and_distiller_produce_provisional_candidates(self):
        adapter = AlfWorldTrajectoryAdapter()
        adapted = adapter.adapt_many(
            [_trajectory("a", won=True), _trajectory("b", won=True)]
        )
        skills = HeuristicSkillDistiller(
            DistillationConfig(high_cost_steps=20, min_support=2)
        ).distill(adapted)

        self.assertEqual(adapted[0][-1].atomic_op, AtomicOp.TERMINATE)
        self.assertTrue(skills)
        self.assertTrue(all(skill.status == SkillStatus.CANDIDATE for skill in skills))
        self.assertTrue(
            any(skill.metadata["source_signal"] == "won" for skill in skills)
        )

    def test_reasoning_messages_are_not_environment_action_evidence(self):
        trajectory = _trajectory(
            "pick_heat_then_place_in_recep-test",
            won=True,
            task="put a hot mug on the desk",
        )
        trajectory["steps"][0]["agent_messages"] = [
            {
                "agent": "Executor",
                "content": (
                    "<think>I should heat the mug with the microwave.</think>"
                    "<action>look</action>"
                ),
            }
        ]
        adapted = AlfWorldTrajectoryAdapter().adapt(trajectory)
        fragments = extract_key_fragments([adapted], max_fragments=10)

        self.assertFalse(
            trajectory_has_capability(adapted, "transform.heat")
        )
        self.assertFalse(
            any("<think>" in fragment.action for fragment in fragments)
        )
        communicate = [
            step for step in adapted if step.atomic_op == AtomicOp.COMMUNICATE
        ]
        self.assertTrue(communicate)
        self.assertNotIn("stalled", communicate[0].metadata)
        self.assertNotIn("repeated_action", communicate[0].metadata)

    def test_distiller_accepts_failed_trajectory_with_positive_progress(self):
        trajectory = _trajectory(
            "pick_clean_then_place_in_recep-test",
            task="put a clean mug on the desk",
        )
        trajectory["steps"] = [
            {
                "observation_before": "You are holding mug 1 by sinkbasin 1.",
                "observation": "You clean the mug 1.",
                "action": "clean mug 1 with sinkbasin 1",
                "is_action_valid": True,
                "goal_progress_delta": 0.5,
            }
        ]
        adapted = AlfWorldTrajectoryAdapter().adapt_many([trajectory])
        skills = HeuristicSkillDistiller(
            DistillationConfig(
                high_cost_steps=20,
                min_support=1,
                operation_min_support=1,
            )
        ).distill(adapted)

        self.assertTrue(skills)
        self.assertEqual(
            skills[0].skill_name,
            "Observed environment progress clean pattern",
        )
        self.assertTrue(any("clean" in step for step in skills[0].action_protocol))
        self.assertEqual(
            skills[0].metadata["anchor_state"],
            "You are holding mug 1 by sinkbasin 1.",
        )
        self.assertEqual(
            skills[0].metadata["protocol_source"],
            "observed_cluster_actions",
        )
        # Simplified enriched path requires successful wins for protocol merge.
        enriched = EnrichedSkillDistiller(
            DistillationConfig(min_support=1, operation_min_support=1)
        ).distill(adapted)
        self.assertEqual(enriched, [])

    def test_skill_bank_upgrades_duplicate_provisional_skill(self):
        provisional = Skill(
            skill_name="Check progress",
            description="Check task progress",
            precondition="Before termination",
            action_protocol=["Check"],
            applicable_atomic_ops=[AtomicOp.VERIFY],
            status=SkillStatus.PROVISIONAL,
            evidence_ids=["a"],
        )
        verified = Skill(
            skill_name="Check progress",
            description="Check task progress",
            precondition="Before termination",
            action_protocol=["Check"],
            applicable_atomic_ops=[AtomicOp.VERIFY],
            status=SkillStatus.VERIFIED,
            marginal_utility=0.3,
            evidence_ids=["b"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            bank = SkillBank(Path(temp_dir) / "skill_bank.json")
            bank.add(provisional, allow_provisional=True)
            bank.add(verified)

        self.assertEqual(bank.skills[0].status, SkillStatus.VERIFIED)
        self.assertEqual(bank.skills[0].evidence_ids, ["a", "b"])
        self.assertEqual(bank.skills[0].marginal_utility, 0.3)

    def test_skill_bank_keeps_cross_signal_routines_separate(self):
        clean = Skill(
            skill_name="Reuse verified clean-then-place routine",
            description="Replay the successful clean-then-place sequence observed in winning trajectories.",
            precondition="clean task",
            action_protocol=["clean"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={
                "source_signal": "clean_operation_success",
                "primary_task_family": "pick_clean_then_place_in_recep",
            },
        )
        heat = Skill(
            skill_name="Reuse verified heat-then-place routine",
            description="Replay the successful heat-then-place sequence observed in winning trajectories.",
            precondition="heat task",
            action_protocol=["heat"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={
                "source_signal": "heat_operation_success",
                "primary_task_family": "pick_heat_then_place_in_recep",
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            bank = SkillBank(Path(temp_dir) / "skill_bank.json")
            self.assertTrue(bank.add(clean))
            self.assertTrue(bank.add(heat))

        self.assertEqual(len(bank.skills), 2)

    def test_skill_bank_merges_capability_across_family_scope(self):
        first = Skill(
            skill_name="Recover search coverage",
            description="Search unexplored receptacles",
            precondition="Search has stalled",
            action_protocol=["Choose a different unexplored location"],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="search.coverage",
            applicable_task_families=["pick_clean_then_place_in_recep"],
            metadata={
                "source_signal": "search_exhaustion",
                "primary_task_family": "pick_clean_then_place_in_recep",
            },
        )
        second = Skill(
            skill_name="Expand object search",
            description="Continue systematic search",
            precondition="Target is not found",
            action_protocol=["Choose another unexplored location"],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="search.coverage",
            applicable_task_families=["pick_heat_then_place_in_recep"],
            metadata={
                "source_signal": "search_exhaustion",
                "primary_task_family": "pick_heat_then_place_in_recep",
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            bank = SkillBank(Path(temp_dir) / "skill_bank.json")
            self.assertTrue(bank.add(first))
            self.assertFalse(bank.add(second))
            canonical = bank.canonical_skill(second)

        self.assertEqual(len(bank.skills), 1)
        self.assertIsNotNone(canonical)
        self.assertEqual(canonical.skill_name, first.skill_name)
        self.assertEqual(
            bank.skills[0].applicable_task_families,
            [
                "pick_clean_then_place_in_recep",
                "pick_heat_then_place_in_recep",
            ],
        )

    def test_skill_bank_retires_after_repeated_negative_utility(self):
        def evaluated_skill(utility):
            return Skill(
                skill_name="Recover non-progress",
                description="Change action after no progress",
                precondition="Observation is unchanged",
                action_protocol=["Choose a different valid action"],
                applicable_atomic_ops=[AtomicOp.ACT],
                status=SkillStatus.VERIFIED,
                marginal_utility=utility,
                capability_key="recovery.non_progress",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            bank = SkillBank(
                Path(temp_dir) / "skill_bank.json",
                retire_after_negative=2,
            )
            bank.add(evaluated_skill(-0.1))
            bank.add(evaluated_skill(-0.2))

        self.assertEqual(bank.skills[0].status, SkillStatus.RETIRED)
        self.assertEqual(bank.skills[0].utility_history, [-0.1, -0.2])
        self.assertEqual(bank.stable_skills(), [])

    def test_skill_bank_disables_heldout_mu_retirement_by_default(self):
        def evaluated_skill(utility):
            return Skill(
                skill_name="Recover non-progress",
                description="Change action after no progress",
                precondition="Observation is unchanged",
                action_protocol=["Choose a different valid action"],
                applicable_atomic_ops=[AtomicOp.ACT],
                status=SkillStatus.VERIFIED,
                marginal_utility=utility,
                capability_key="recovery.non_progress",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            bank = SkillBank(Path(temp_dir) / "skill_bank.json")
            self.assertEqual(bank.retire_after_negative, 0)
            bank.add(evaluated_skill(-0.1))
            bank.add(evaluated_skill(-0.2))
            bank.add(evaluated_skill(-0.3))

        # Held-out MU history is kept, but retirement is disabled by default.
        self.assertNotEqual(bank.skills[0].status, SkillStatus.RETIRED)
        self.assertIsNone(
            (bank.skills[0].metadata or {}).get("retirement_reason")
        )
        self.assertEqual(bank.skills[0].utility_history, [-0.1, -0.2, -0.3])
        self.assertTrue(bank.skills)

    def test_paired_operation_evidence_passes_grounding_validation(self):
        trajectories = AlfWorldTrajectoryAdapter().adapt_many(
            _operation_pair(
                "pick_clean_then_place_in_recep-pair",
                "clean",
                task="put a clean mug on the desk",
            )
        )
        skills = EnrichedSkillDistiller(
            DistillationConfig(operation_min_support=1),
            max_evidence_trajectories=3,
            min_wins_for_protocol=1,
        ).distill(trajectories)

        clean_skill = skills[0]
        validation = validate_skill_evidence(
            clean_skill,
            {
                str(steps[-1].metadata["trajectory_id"]): steps
                for steps in trajectories
            },
        )
        self.assertTrue(validation.accepted)
        self.assertTrue(validation.success_evidence_ids)
        self.assertFalse(validation.failure_evidence_ids)
        self.assertGreater(validation.protocol_coverage_rate, 0.0)

    def test_local_transition_validation_requires_three_independent_tasks(self):
        records = []
        transitions = []
        for index in range(3):
            task_id = f"/valid_seen/clean-{index}/game.tw-pddl"
            action = f"clean mug {index + 1} with sinkbasin 1"
            observation = (
                f"You clean the mug {index + 1} using the sinkbasin 1."
            )
            transitions.append(
                {
                    "task_id": task_id,
                    "action": action,
                    "observation": observation,
                }
            )
            records.append(
                {
                    "gamefile": task_id,
                    "task": "put a clean mug on the desk",
                    "task_family": "pick_clean_then_place_in_recep",
                    "won": False,
                    "steps": [
                        {
                            "observation": observation,
                            "action": action,
                            "is_action_valid": True,
                        }
                    ],
                }
            )
        trajectories = AlfWorldTrajectoryAdapter().adapt_many(records)
        indexed = {
            str(steps[-1].metadata["trajectory_id"]): steps
            for steps in trajectories
        }
        skill = Skill(
            skill_name="Environment-confirmed clean protocol",
            description="Grounded local transition",
            precondition="Clean task",
            action_protocol=["clean <object> with <tool>"],
            applicable_atomic_ops=[AtomicOp.ACT],
            evidence_ids=[record["gamefile"] for record in records],
            support_count=3,
            capability_key="transform.clean",
            metadata={"confirmed_transitions": transitions},
        )

        accepted = validate_environment_confirmed_skill(skill, indexed)
        self.assertTrue(accepted.accepted)
        self.assertEqual(len(accepted.success_evidence_ids), 3)
        skill.evidence_ids = skill.evidence_ids[:2]
        rejected = validate_environment_confirmed_skill(skill, indexed)
        self.assertFalse(rejected.accepted)

    def test_generic_skill_requires_anchor_effect_and_three_transitions(self):
        skill = Skill(
            skill_name="Generic recovery",
            description="d",
            precondition="p",
            action_protocol=["look"],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="recovery.non_progress",
            expected_effect="Improve task execution.",
        )
        reasons = generic_skill_contract_reasons(skill)
        self.assertEqual(len(reasons), 3)
        skill.metadata.update(
            {
                "anchor_state": "The current observation shows no progress.",
                "confirmed_transitions": [
                    {
                        "task_id": f"task-{index}",
                        "action": "open drawer 1",
                        "observation": "You open the drawer 1.",
                    }
                    for index in range(3)
                ],
            }
        )
        skill.expected_effect = "The next valid action changes the observation."
        self.assertEqual(generic_skill_contract_reasons(skill), [])

    def test_generic_skill_accepts_key_fragment_environment_effects(self):
        skill = Skill(
            skill_name="Trajectory-derived execution efficiency protocol",
            description="Aligned pick_and_place protocol",
            precondition="Task family matches pick_and_place",
            action_protocol=[
                "go to <location>",
                "take <object> from <receptacle>",
                "move <object> to <receptacle>",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="execution.efficiency",
            expected_effect=(
                "Episode reaches a won terminal state on pick_and_place."
            ),
            key_fragments=[
                StateActionFragment(
                    trajectory_id="task-candle",
                    step_index=6,
                    observation="You move the candle 1 to the drawer 1.",
                    action="move candle 1 to drawer 1",
                    won=True,
                    task_family="pick_and_place",
                )
            ],
            metadata={
                "anchor_state": "successful trajectory completed within 6 steps",
                "source_signal": "efficient_execution",
            },
        )
        self.assertEqual(generic_skill_contract_reasons(skill), [])

    def test_online_skill_credit_promotes_after_one_effect_success(self):
        policy = SkillCreditPolicy()
        skill = Skill(
            skill_name="Learned clean",
            description="d",
            precondition="p",
            action_protocol=["clean object"],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="transform.clean",
            applicable_task_families=[
                "pick_clean_then_place_in_recep"
            ],
            metadata={"capability_operation": "clean"},
        )
        initialize_skill_credit(skill, policy)
        trials = [
            EvaluationTrial(
                task_id=f"/clean-{index}",
                task="clean mug",
                task_family="pick_clean_then_place_in_recep",
                condition="online",
                reward=0.0,
                cost=1.0,
                won=False,
                num_steps=1,
                steps=[
                    {
                        "action": "clean mug 1 with sinkbasin 1",
                        "observation": "You clean the mug 1.",
                        "is_action_valid": True,
                    }
                ],
                activated_skill_names=[skill.skill_name],
                skill_activation_steps={skill.skill_name: 1},
                assigned_primary_agent="Executor",
                actions_by_agent={"Executor": 1},
            )
            for index in range(1)
        ]

        result = update_skill_credits(
            [skill],
            trials,
            [
                AgentSpec(
                    name="Executor",
                    role="executor",
                    responsibilities=["act"],
                )
            ],
            policy,
        )

        self.assertEqual(skill.status, SkillStatus.VERIFIED)
        self.assertAlmostEqual(skill.metadata["skill_credit"]["score"], 2 / 3)
        self.assertEqual(skill.metadata["skill_credit"]["uses"], 1)
        self.assertEqual(skill.metadata["skill_credit"]["successes"], 1)
        self.assertTrue(skill.metadata["credit_promoted_pending_org"])
        self.assertEqual(result["promoted"], [skill])

    def test_place_protocol_credit_counts_put_as_move(self):
        """Abstract protocols say move; ALFWorld often emits put."""
        policy = SkillCreditPolicy(
            verify_score=0.60,
            min_uses_for_promotion=2,
        )
        skill = Skill(
            skill_name="Trajectory-derived execution efficiency protocol",
            description="d",
            precondition="pick_and_place",
            action_protocol=[
                "go to <location>",
                "take <object> from <receptacle>",
                "move <object> to <receptacle>",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="execution.efficiency",
            applicable_task_families=["pick_and_place"],
            metadata={
                "source_signal": "efficient_execution",
                "primary_task_family": "pick_and_place",
            },
        )
        initialize_skill_credit(skill, policy)
        trials = [
            EvaluationTrial(
                task_id=f"/place-{index}",
                task="put some candle on drawer",
                task_family="pick_and_place",
                condition="online",
                reward=1.0,
                cost=1.0,
                won=True,
                num_steps=3,
                steps=[
                    {
                        "action": "go to desk 1",
                        "observation": "You arrive at desk 1.",
                        "is_action_valid": True,
                    },
                    {
                        "action": "take candle 1 from desk 1",
                        "observation": "You pick up the candle 1.",
                        "is_action_valid": True,
                    },
                    {
                        "action": "put candle 1 in/on drawer 1",
                        "observation": "You put the candle 1 in/on the drawer 1.",
                        "is_action_valid": True,
                        "reward": 1.0,
                    },
                ],
                activated_skill_names=[skill.skill_name],
                skill_activation_steps={skill.skill_name: 1},
                assigned_primary_agent="Executor",
                actions_by_agent={"Executor": 3},
            )
            for index in range(2)
        ]
        result = update_skill_credits(
            [skill],
            trials,
            [
                AgentSpec(
                    name="Executor",
                    role="executor",
                    responsibilities=["act"],
                )
            ],
            policy,
        )
        self.assertEqual(skill.metadata["skill_credit"]["uses"], 2)
        self.assertEqual(skill.metadata["skill_credit"]["successes"], 2)
        self.assertEqual(skill.status, SkillStatus.VERIFIED)
        self.assertEqual(result["promoted"], [skill])

    def test_credit_does_not_count_win_without_local_effect(self):
        policy = SkillCreditPolicy(
            verify_score=0.60,
            min_uses_for_promotion=1,
            min_adherence_for_use=0.0,
        )
        skill = Skill(
            skill_name="Trajectory-derived execution efficiency protocol",
            description="d",
            precondition="pick_and_place",
            action_protocol=["take <object>", "move <object> to <receptacle>"],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="execution.efficiency",
            applicable_task_families=["pick_and_place"],
            expected_effect="object is placed in the target receptacle",
            metadata={
                "source_signal": "efficient_execution",
                "primary_task_family": "pick_and_place",
                "capability_operation": "move",
            },
        )
        initialize_skill_credit(skill, policy)
        trial = EvaluationTrial(
            task_id="/place-no-effect",
            task="put candle on drawer",
            task_family="pick_and_place",
            condition="online",
            reward=1.0,
            cost=1.0,
            won=True,
            num_steps=1,
            steps=[
                {
                    # Skill ran (put/move), episode won, but no grounded local
                    # effect — credit must not treat win as success.
                    "action": "put candle 1 in drawer 1",
                    "observation": "Nothing happens.",
                    "is_action_valid": True,
                }
            ],
            activated_skill_names=[skill.skill_name],
            skill_activation_steps={skill.skill_name: 1},
            assigned_primary_agent="Executor",
            actions_by_agent={"Executor": 1},
        )
        update_skill_credits(
            [skill],
            [trial],
            [
                AgentSpec(
                    name="Executor",
                    role="executor",
                    responsibilities=["act"],
                )
            ],
            policy,
        )
        self.assertEqual(skill.metadata["skill_credit"]["uses"], 1)
        self.assertEqual(skill.metadata["skill_credit"]["successes"], 0)

    def test_pipeline_keeps_credit_promoted_skill_without_segment_evidence(self):
        from sage_mas.pipeline import SageEvolutionPipeline

        promoted = Skill(
            skill_name="Trajectory-derived execution efficiency protocol",
            description="Aligned pick_and_place protocol",
            precondition="Task family matches pick_and_place",
            action_protocol=[
                "go to <location>",
                "take <object> from <receptacle>",
                "move <object> to <receptacle>",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            capability_key="execution.efficiency",
            applicable_task_families=["pick_and_place"],
            evidence_ids=["/historical/task-a", "/historical/task-b"],
            support_count=2,
            metadata={
                "source_signal": "efficient_execution",
                "primary_task_family": "pick_and_place",
                "anchor_state": "successful trajectory completed within 6 steps",
                "protocol_source": "aligned_successful_trajectory_stages_v2",
                "protocol_alignment_ok": True,
                "protocol_structure_ok": True,
                "protocol_alignment_support": 2,
                "confirmed_transitions": [
                    {
                        "task_id": "/historical/task-a",
                        "action": "move candle 1 to drawer 1",
                        "observation": "You move the candle 1 to the drawer 1.",
                    }
                ],
                "credit_promoted_pending_org": True,
                "skill_credit": {
                    "uses": 2,
                    "successes": 2,
                    "score": 0.75,
                    "events": [],
                },
                "evidence_validation": {
                    "accepted": True,
                    "reasons": [],
                    "success_evidence_ids": [
                        "/historical/task-a",
                        "/historical/task-b",
                    ],
                    "failure_evidence_ids": [],
                    "protocol_coverage_rate": 1.0,
                    "protocol_grounding_rate": 1.0,
                    "fragment_alignment_rate": 1.0,
                },
            },
            expected_effect="Episode reaches a won terminal state.",
        )
        accepted, rejected = SageEvolutionPipeline._validate_candidates(
            [promoted],
            [],
        )
        self.assertEqual(accepted, [promoted])
        self.assertEqual(rejected, [])
        self.assertEqual(promoted.status, SkillStatus.VERIFIED)

    def test_credit_promotion_does_not_force_organization_shift(self):
        """Credit proves usability; ADD_AGENT still needs independent novelty."""
        from sage_mas.pipeline import SageEvolutionPipeline
        from sage_mas.skill_bank import SkillBank

        historical = Skill(
            skill_name="Historical heat protocol",
            description="Heat then place from prior trajectories",
            precondition="Task requires heat transformation",
            action_protocol=[
                "go to microwave",
                "heat object with microwave",
                "move object to receptacle",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            evidence_ids=["hist-1"],
            support_count=2,
            capability_key="transform.heat",
            applicable_task_families=["pick_heat_then_place_in_recep"],
            metadata={
                "source_signal": "heat_operation_success",
                "primary_task_family": "pick_heat_then_place_in_recep",
                "skill_credit": {
                    "uses": 3,
                    "successes": 3,
                    "score": 0.8,
                    "events": [],
                },
            },
        )
        promoted = Skill(
            skill_name="Promoted heat protocol",
            description="Heat then place from prior trajectories",
            precondition="Task requires heat transformation",
            action_protocol=[
                "go to microwave",
                "heat object with microwave",
                "move object to receptacle",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            evidence_ids=["promo-1"],
            support_count=2,
            capability_key="transform.heat",
            applicable_task_families=["pick_heat_then_place_in_recep"],
            metadata={
                "source_signal": "heat_operation_success",
                "primary_task_family": "pick_heat_then_place_in_recep",
                "credit_promoted_pending_org": True,
                "skill_credit": {
                    "uses": 2,
                    "successes": 2,
                    "score": 0.75,
                    "events": [],
                },
                "evidence_validation": {
                    "accepted": True,
                    "reasons": [],
                    "success_evidence_ids": ["promo-1"],
                    "failure_evidence_ids": [],
                    "protocol_coverage_rate": 1.0,
                    "protocol_grounding_rate": 1.0,
                    "fragment_alignment_rate": 1.0,
                },
            },
        )
        novel = Skill(
            skill_name="Promoted cool protocol",
            description="Cool then place from prior trajectories",
            precondition="Task requires cool transformation",
            action_protocol=[
                "go to fridge",
                "cool object with fridge",
                "move object to receptacle",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            evidence_ids=["promo-cool-1"],
            support_count=2,
            capability_key="transform.cool",
            applicable_task_families=["pick_cool_then_place_in_recep"],
            metadata={
                "source_signal": "cool_operation_success",
                "primary_task_family": "pick_cool_then_place_in_recep",
                "credit_promoted_pending_org": True,
                "skill_credit": {
                    "uses": 2,
                    "successes": 2,
                    "score": 0.75,
                    "events": [],
                },
                "evidence_validation": {
                    "accepted": True,
                    "reasons": [],
                    "success_evidence_ids": ["promo-cool-1"],
                    "failure_evidence_ids": [],
                    "protocol_coverage_rate": 1.0,
                    "protocol_grounding_rate": 1.0,
                    "fragment_alignment_rate": 1.0,
                },
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bank_path = temp_path / "skill_bank.json"
            bank = SkillBank(bank_path)
            bank.add(historical)
            bank.save()
            trajectory_path = temp_path / "trajectories.jsonl"
            write_jsonl(
                trajectory_path,
                [
                    {
                        "trajectory_id": "promo-1",
                        "task": "heat some potato and put it in garbagecan",
                        "task_family": "pick_heat_then_place_in_recep",
                        "won": True,
                        "steps": [
                            {
                                "observation": "You heat the potato.",
                                "action": "heat potato 1 with microwave 1",
                                "is_action_valid": True,
                            }
                        ],
                    },
                    {
                        "trajectory_id": "promo-cool-1",
                        "task": "cool some mug and put it in cabinet",
                        "task_family": "pick_cool_then_place_in_recep",
                        "won": True,
                        "steps": [
                            {
                                "observation": "You cool the mug.",
                                "action": "cool mug 1 with fridge 1",
                                "is_action_valid": True,
                            }
                        ],
                    },
                ],
            )
            org_path = temp_path / "active_organization.json"
            write_json(
                org_path,
                {
                    "agents": [
                        {
                            "name": "Executor",
                            "role": "Environment executor",
                            "responsibilities": ["Act"],
                            "tool_permissions": ["alfworld_action"],
                        }
                    ]
                },
            )
            pipeline = SageEvolutionPipeline(
                {
                    "sage": {
                        "skill_bank_path": str(bank_path),
                        "organization": {
                            "distribution_shift_threshold": 0.15,
                            "min_cluster_support": 1,
                            "min_assignment_gap": 0.1,
                            "max_bootstrap_agents_per_round": 2,
                        },
                        "agents": [
                            {
                                "name": "Executor",
                                "role": "Environment executor",
                                "responsibilities": ["Act"],
                            }
                        ],
                    }
                }
            )
            artifacts = pipeline.run(
                trajectory_path=trajectory_path,
                output_root=temp_path / "distill",
                active_organization_path=org_path,
                additional_candidates=[promoted, novel],
            )
            edits = read_json(artifacts.organization_edits)
            bank_after = SkillBank(bank_path).skills
            for skill in bank_after:
                self.assertNotIn(
                    "credit_promoted_pending_org",
                    skill.metadata,
                )
                if skill.distribution_shift is not None:
                    self.assertNotEqual(
                        skill.distribution_shift.metric,
                        "credit_promotion",
                    )
                    self.assertNotEqual(
                        skill.distribution_shift.baseline_source,
                        "online_skill_credit_promotion",
                    )

        heat_edits = [
            edit
            for edit in edits
            if any(
                "heat" in str(name).lower()
                for name in (edit.get("assigned_skill_names") or [])
            )
        ]
        cool_edits = [
            edit
            for edit in edits
            if any(
                "cool" in str(name).lower()
                for name in (edit.get("assigned_skill_names") or [])
            )
        ]
        self.assertTrue(edits)
        # Near-duplicate heat Skill after credit promotion must not invent a
        # forced novelty path.
        self.assertTrue(heat_edits)
        self.assertTrue(
            all(edit.get("edit_type") != "add_agent" for edit in heat_edits)
        )
        self.assertTrue(
            all(
                "credit_promotion" not in str(edit.get("rationale", ""))
                for edit in edits
            )
        )
        # Novel cool capability may still expand through independent novelty.
        self.assertTrue(
            any(edit.get("edit_type") == "add_agent" for edit in cool_edits)
        )
        self.assertTrue(
            any(
                "independent cluster novelty" in str(edit.get("rationale", ""))
                for edit in cool_edits
            )
        )

    def test_online_skill_credit_does_not_count_family_match_as_use(self):
        policy = SkillCreditPolicy()
        skill = Skill(
            skill_name="Learned clean",
            description="d",
            precondition="p",
            action_protocol=["clean object"],
            applicable_atomic_ops=[AtomicOp.ACT],
            applicable_task_families=[
                "pick_clean_then_place_in_recep"
            ],
        )
        initialize_skill_credit(skill, policy)
        irrelevant = EvaluationTrial(
            task_id="/heat",
            task="heat mug",
            task_family="pick_heat_then_place_in_recep",
            condition="online",
            reward=0.0,
            cost=1.0,
            won=False,
            num_steps=1,
        )
        eligible = [
            EvaluationTrial(
                task_id=f"/clean-{index}",
                task="clean mug",
                task_family="pick_clean_then_place_in_recep",
                condition="online",
                reward=0.0,
                cost=1.0,
                won=False,
                num_steps=1,
            )
            for index in range(3)
        ]

        update_skill_credits(
            [skill],
            [irrelevant, *eligible],
            [],
            policy,
        )

        credit = skill.metadata["skill_credit"]
        self.assertAlmostEqual(credit["score"], 0.5)
        self.assertEqual(credit["uses"], 0)
        self.assertEqual(credit["successes"], 0)

    def test_selected_skill_counts_use_only_after_protocol_action(self):
        policy = SkillCreditPolicy()
        skill = Skill(
            skill_name="Learned clean",
            description="d",
            precondition="p",
            action_protocol=["clean object"],
            applicable_atomic_ops=[AtomicOp.ACT],
            applicable_task_families=["pick_clean_then_place_in_recep"],
            metadata={"capability_operation": "clean"},
        )
        initialize_skill_credit(skill, policy)
        selected_but_ignored = EvaluationTrial(
            task_id="/clean",
            task="clean mug",
            task_family="pick_clean_then_place_in_recep",
            condition="online",
            reward=0.0,
            cost=1.0,
            won=False,
            num_steps=1,
            steps=[
                {
                    "action": "go to sinkbasin 1",
                    "observation": "You arrive at sinkbasin 1.",
                    "is_action_valid": True,
                }
            ],
            activated_skill_names=[skill.skill_name],
            skill_activation_steps={skill.skill_name: 1},
        )
        update_skill_credits(
            [skill],
            [selected_but_ignored],
            [],
            policy,
        )
        credit = skill.metadata["skill_credit"]
        self.assertEqual(credit["uses"], 0)
        self.assertEqual(credit["successes"], 0)
        self.assertEqual(credit["score"], 0.5)

    def test_online_skill_credit_migrates_legacy_additive_record(self):
        skill = Skill(
            skill_name="Legacy learned skill",
            description="d",
            precondition="p",
            action_protocol=["act"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={
                "skill_credit": {
                    "score": 0.7,
                    "opportunities": 4,
                    "uses": 4,
                    "episode_successes": 1,
                    "local_effects": 2,
                    "failures": 1,
                    "events": [],
                }
            },
        )

        credit = initialize_skill_credit(skill, SkillCreditPolicy())

        self.assertEqual(credit["uses"], 4)
        self.assertEqual(credit["successes"], 3)
        self.assertAlmostEqual(credit["score"], 4 / 6)
        self.assertNotIn("opportunities", credit)
        self.assertNotIn("episode_successes", credit)

    def test_online_skill_credit_prunes_after_five_low_effect_uses(self):
        policy = SkillCreditPolicy()
        skill = Skill(
            skill_name="Stable learned skill",
            description="d",
            precondition="p",
            action_protocol=["act"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            applicable_task_families=["pick_and_place"],
            metadata={
                "skill_credit": {
                    "score": 0.1,
                    "uses": 5,
                    "successes": 0,
                    "events": [],
                }
            },
        )

        result = update_skill_credits([skill], [], [], policy)
        self.assertEqual(skill.status, SkillStatus.RETIRED)
        self.assertAlmostEqual(
            skill.metadata["skill_credit"]["score"],
            1 / 7,
        )
        self.assertEqual(result["demoted"], [])
        self.assertEqual(result["retired"], [skill])

    def test_trajectory_grounded_distiller_uses_evidence_protocol(self):
        class FakeBackend:
            def complete(self, system_prompt, user_prompt):
                return LLMResult(
                    content=(
                        '{"skill_name":"Wash held target",'
                        '"description":"Grounded clean operation",'
                        '"precondition":"Target is held",'
                        '"action_protocol":["go to sinkbasin",'
                        '"clean target with sinkbasin"],'
                        '"expected_effect":"Target becomes clean",'
                        '"suggested_role":"Executor",'
                        '"target_failure_types":["cleaning_omission"],'
                        '"trajectory_summary":"Grounded from one failed clean trajectory."}'
                    )
                )

        trajectories = AlfWorldTrajectoryAdapter().adapt_many(
            _operation_pair(
                "pick_clean_then_place_in_recep-test",
                "clean",
                task="put a clean mug on the desk",
            )
        )
        skills = TrajectoryGroundedSkillDistiller(
            backend=FakeBackend(),
            seed_distiller=HeuristicSkillDistiller(
                DistillationConfig(operation_min_support=1)
            ),
            min_wins_for_protocol=1,
        ).distill(trajectories)

        self.assertEqual(skills[0].skill_name, "Wash held target")
        self.assertTrue(skills[0].metadata["trajectory_grounded"])
        self.assertTrue(skills[0].trajectory_summary)
        self.assertTrue(skills[0].key_fragments)
        self.assertTrue(skills[0].embedding)
        self.assertIsNotNone(skills[0].exploration_distribution)
        self.assertEqual(
            skills[0].metadata["source_signal"],
            "won",
        )

    def test_finalize_operation_protocol_restores_missing_operation_step(self):
        seed = Skill(
            skill_name="Clean the target object before placement",
            description="Clean before placement",
            precondition="Held dirty object",
            action_protocol=[
                "Navigate to a sinkbasin while holding the target object",
                "Execute 'clean <object> with <sinkbasin>' and require cleaning feedback",
                "Place the cleaned object in the target receptacle",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={"source_signal": "missing_clean_operation"},
            suggested_role="Executor",
        )
        finalized = finalize_operation_protocol(
            seed,
            [
                "take soapbar 1 from countertop 1",
                "move soapbar 1 to countertop 1",
            ],
        )

        self.assertTrue(
            any("clean" in step.lower() for step in finalized)
        )
        self.assertEqual(finalized, seed.action_protocol)

    def test_distiller_learns_clean_from_successful_trajectory(self):
        trajectory = {
            "gamefile": "/tmp/pick_clean_then_place_in_recep-Mug/game.tw-pddl",
            "task": "put a clean mug on the desk",
            "won": True,
            "num_steps": 4,
            "steps": [
                {
                    "observation": "Taken.",
                    "action": "take mug 1 from desk 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You clean the mug.",
                    "action": "clean mug 1 with sinkbasin 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You put the mug.",
                    "action": "move mug 1 to desk 1",
                    "is_action_valid": True,
                },
            ],
        }
        adapted = AlfWorldTrajectoryAdapter().adapt_many([trajectory])
        skills = HeuristicSkillDistiller(
            DistillationConfig(operation_min_support=1, min_support=1)
        ).distill(adapted)

        self.assertEqual(len(skills), 1)
        self.assertEqual(
            skills[0].metadata["source_signal"],
            "won",
        )
        self.assertTrue(
            any("clean" in step.lower() for step in skills[0].action_protocol)
        )
        self.assertEqual(skills[0].suggested_role, "Executor")

    def test_prefer_success_trajectories_orders_wins_first(self):
        adapter = AlfWorldTrajectoryAdapter()
        failed = adapter.adapt_many(
            [_trajectory("pick_clean_then_place_in_recep-a", won=False)]
        )[0]
        won = adapter.adapt_many(
            [
                {
                    "gamefile": "/tmp/pick_clean_then_place_in_recep-b/game.tw-pddl",
                    "task": "put a clean mug on the desk",
                    "won": True,
                    "num_steps": 2,
                    "steps": [
                        {
                            "observation": "You clean the mug.",
                            "action": "clean mug 1 with sinkbasin 1",
                            "is_action_valid": True,
                        }
                    ],
                }
            ]
        )[0]
        ordered = prefer_success_trajectories([failed, won])
        self.assertTrue(ordered[0][-1].metadata["won"])
        self.assertFalse(ordered[1][-1].metadata["won"])

    def test_ground_skill_filters_noop_and_builds_stages(self):
        adapter = AlfWorldTrajectoryAdapter()
        trajectories = adapter.adapt_many(
            [
                {
                    "gamefile": "/tmp/pick_clean_then_place_in_recep-noise/game.tw-pddl",
                    "task": "put a clean mug on the desk",
                    "won": True,
                    "num_steps": 8,
                    "steps": [
                        {
                            "observation": "Nothing happens.",
                            "action": "go to cabinet 1",
                            "is_action_valid": True,
                            "stalled": True,
                        },
                        {
                            "observation": "Nothing happens.",
                            "action": "examine mug 1",
                            "is_action_valid": True,
                        },
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
                            "observation": "Inventory.",
                            "action": "inventory",
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
            action_protocol=["clean"],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="transform.clean",
            metadata={"source_signal": "clean_operation_success"},
        )
        grounded = ground_skill_in_successful_trajectory(seed, trajectories)
        protocol_text = " | ".join(grounded.action_protocol)
        self.assertNotIn("examine <entity>", grounded.action_protocol)
        self.assertNotIn("inventory", grounded.action_protocol)
        self.assertTrue(any("clean" in step for step in grounded.action_protocol))
        self.assertTrue(any("take" in step for step in grounded.action_protocol))
        self.assertIn("pickup", grounded.metadata["protocol_stages"])
        self.assertIn("transform", grounded.metadata["protocol_stages"])
        self.assertIn("place", grounded.metadata["protocol_stages"])
        self.assertIn("confirms", grounded.expected_effect.lower())
        self.assertIn("instantiate", grounded.precondition.lower())
        self.assertEqual(
            grounded.metadata["protocol_source"],
            "aligned_successful_trajectory_stages_v2",
        )
        # Single-win grounding is alignment-ok at min_alignment_wins=1;
        # org edits still require structure + injectable capability gates.
        self.assertTrue(grounded.metadata.get("protocol_alignment_ok"))
        self.assertGreaterEqual(
            int(grounded.metadata.get("protocol_alignment_support") or 0),
            1,
        )
        self.assertIn("clean", protocol_text)

    def test_canonicalize_protocol_enforces_pickup_transform_place(self):
        scrambled = [
            "move <object> to <receptacle>",
            "open <entity>",
            "take <object> from <receptacle>",
            "heat <object> with <tool>",
            "go to <location>",
        ]
        ordered = canonicalize_protocol_stages(
            scrambled,
            capability="transform.heat",
        )
        self.assertEqual(
            protocol_structure_issues(ordered, capability="transform.heat"),
            [],
        )
        take_i = ordered.index("take <object> from <receptacle>")
        heat_i = ordered.index("heat <object> with <tool>")
        place_i = ordered.index("move <object> to <receptacle>")
        self.assertLess(take_i, heat_i)
        self.assertLess(heat_i, place_i)

    def test_ground_skill_org_ready_requires_two_wins(self):
        adapter = AlfWorldTrajectoryAdapter()
        trajectories = adapter.adapt_many(
            [
                {
                    "gamefile": f"/tmp/pick_cool_then_place_in_recep-{i}/game.tw-pddl",
                    "task": "put a cool tomato in microwave",
                    "won": True,
                    "num_steps": 4,
                    "steps": [
                        {
                            "observation": "Arrived.",
                            "action": "go to fridge 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "Taken.",
                            "action": "take tomato 1 from fridge 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "Cooled.",
                            "action": "cool tomato 1 with fridge 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "Placed.",
                            "action": "move tomato 1 to microwave 1",
                            "is_action_valid": True,
                        },
                    ],
                }
                for i in (1, 2)
            ]
        )
        seed = Skill(
            skill_name="seed",
            description="seed",
            precondition="seed",
            action_protocol=["cool"],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="transform.cool",
            metadata={"source_signal": "cool_operation_success"},
        )
        grounded = ground_skill_in_successful_trajectory(seed, trajectories)
        self.assertTrue(grounded.metadata.get("protocol_alignment_ok"))
        self.assertTrue(grounded.metadata.get("protocol_structure_ok"))
        self.assertTrue(protocol_is_org_ready(grounded))
        self.assertEqual(
            grounded.metadata["protocol_stages"],
            ["find", "pickup", "transform", "place"],
        )

    def test_ground_skill_aligns_shared_steps_across_wins(self):
        adapter = AlfWorldTrajectoryAdapter()
        trajectories = adapter.adapt_many(
            [
                {
                    "gamefile": "/tmp/pick_heat_then_place_in_recep-a/game.tw-pddl",
                    "task": "put a hot apple in garbagecan",
                    "won": True,
                    "num_steps": 5,
                    "steps": [
                        {
                            "observation": "Arrived.",
                            "action": "go to fridge 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "Taken.",
                            "action": "take apple 1 from fridge 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "Heated.",
                            "action": "heat apple 1 with microwave 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "Placed.",
                            "action": "move apple 1 to garbagecan 1",
                            "is_action_valid": True,
                        },
                    ],
                },
                {
                    "gamefile": "/tmp/pick_heat_then_place_in_recep-b/game.tw-pddl",
                    "task": "put a hot apple in garbagecan",
                    "won": True,
                    "num_steps": 6,
                    "steps": [
                        {
                            "observation": "Nothing happens.",
                            "action": "go to sofa 1",
                            "is_action_valid": True,
                            "stalled": True,
                        },
                        {
                            "observation": "Arrived.",
                            "action": "go to fridge 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "Taken.",
                            "action": "take apple 1 from fridge 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "Heated.",
                            "action": "heat apple 1 with microwave 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "Placed.",
                            "action": "move apple 1 to garbagecan 1",
                            "is_action_valid": True,
                        },
                    ],
                },
            ]
        )
        seed = Skill(
            skill_name="seed",
            description="seed",
            precondition="seed",
            action_protocol=["heat"],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="transform.heat",
            metadata={"source_signal": "heat_operation_success"},
        )
        grounded = ground_skill_in_successful_trajectory(seed, trajectories)
        self.assertGreaterEqual(grounded.metadata["protocol_alignment_support"], 2)
        self.assertIn("take <object> from <receptacle>", grounded.action_protocol)
        self.assertIn("heat <object> with <tool>", grounded.action_protocol)
        self.assertIn("move <object> to <receptacle>", grounded.action_protocol)
        # Unique dead-end sofa navigation from one trace should not dominate.
        self.assertTrue(grounded.metadata.get("anti_patterns"))

    def test_extract_key_fragments_prefers_productive_success_steps(self):
        adapter = AlfWorldTrajectoryAdapter()
        trajectories = adapter.adapt_many(
            [
                {
                    "gamefile": "/tmp/pick_clean_then_place_in_recep-frag/game.tw-pddl",
                    "task": "put a clean mug on the desk",
                    "won": True,
                    "num_steps": 3,
                    "steps": [
                        {
                            "observation": "Nothing happens.",
                            "action": "go to cabinet 9",
                            "is_action_valid": True,
                            "stalled": True,
                        },
                        {
                            "observation": "You clean the mug 1 using the sinkbasin 1.",
                            "action": "clean mug 1 with sinkbasin 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "Placed.",
                            "action": "move mug 1 to desk 1",
                            "is_action_valid": True,
                        },
                    ],
                }
            ]
        )
        fragments = extract_key_fragments(trajectories, max_fragments=3)
        actions = [fragment.action for fragment in fragments]
        self.assertTrue(any("clean" in action for action in actions))
        self.assertFalse(any("cabinet 9" in action for action in actions))
        # Selected by score, then stored chronologically for debugging.
        indices = [fragment.step_index for fragment in fragments]
        self.assertEqual(indices, sorted(indices))

    def test_extract_anti_patterns_from_noop_failures(self):
        adapter = AlfWorldTrajectoryAdapter()
        trajectories = adapter.adapt_many(
            [
                {
                    "gamefile": "/tmp/pick_cool_then_place_in_recep-anti/game.tw-pddl",
                    "task": "put a cool tomato in microwave",
                    "won": False,
                    "num_steps": 3,
                    "steps": [
                        {
                            "observation": "Nothing happens.",
                            "action": "go to sofa 1",
                            "is_action_valid": True,
                            "stalled": True,
                            "repeated_action": True,
                        },
                        {
                            "observation": "Nothing happens.",
                            "action": "go to sofa 1",
                            "is_action_valid": True,
                            "stalled": True,
                            "repeated_action": True,
                        },
                        {
                            "observation": "Nothing happens.",
                            "action": "go to sofa 1",
                            "is_action_valid": False,
                            "stalled": True,
                        },
                    ],
                }
            ]
        )
        patterns = extract_anti_patterns(trajectories)
        self.assertTrue(patterns)
        self.assertTrue(any("go to <location>" in pattern for pattern in patterns))

    def test_lock_operation_role_covers_success_signals(self):
        seed = Skill(
            skill_name="Reuse clean",
            description="x",
            precondition="y",
            action_protocol=["clean"],
            applicable_atomic_ops=[AtomicOp.ACT],
            suggested_role="Executor",
            metadata={"source_signal": "clean_operation_success"},
        )
        self.assertEqual(
            lock_operation_suggested_role(seed, "CleanupSpecialistAgent"),
            "Executor",
        )

    def test_evaluator_ignore_assigned_skills_skips_bank_skills(self):
        class FakeBackend:
            def complete(self, system_prompt, user_prompt):
                return LLMResult(
                    content="<think>ok</think><action>look</action>",
                    prompt_tokens=1,
                    completion_tokens=1,
                )

        captured: dict[str, object] = {}

        class FakeRuntime:
            def __init__(self, **kwargs):
                captured["runtime_kwargs"] = kwargs
                agents = kwargs.get("agents") or []
                self.executor = next(
                    (
                        agent
                        for agent in agents
                        if "executor" in f"{agent.name} {agent.role}".lower()
                    ),
                    agents[0] if agents else SimpleNamespace(name="Executor"),
                )

            def act(
                self,
                observation,
                injected_skills=None,
                active_assigned_skill_names=None,
                preferred_actor_name=None,
                task=None,
                task_family=None,
                history_steps=None,
                gamefile=None,
            ):
                captured["injected"] = list(injected_skills or [])
                captured["assigned"] = set(active_assigned_skill_names or set())
                captured["preferred_actor"] = preferred_actor_name
                captured["task_family"] = task_family
                return SimpleNamespace(
                    action="<think>ok</think><action>look</action>",
                    token_cost=2,
                    messages=[],
                )

        class FakeEnv:
            def __init__(self):
                self.config = SimpleNamespace(
                    env=SimpleNamespace(history_length=0)
                )

            def reset(self, _):
                return (
                    {
                        "text": ["TASK: put a clean mug on the desk"],
                        "anchor": ["put a clean mug on the desk"],
                    },
                    [{"won": False}],
                )

            def step(self, actions):
                return (
                    {
                        "text": ["done"],
                        "anchor": ["Nothing happens."],
                    },
                    [0.0],
                    [True],
                    [{"won": False, "is_action_valid": True}],
                )

            def close(self):
                return None

        with patch(
            "sage_mas.alfworld_evaluator.build_alfworld_env_manager",
            return_value=FakeEnv(),
        ), patch(
            "sage_mas.alfworld_evaluator.EnhancedMASRuntime",
            FakeRuntime,
        ):
            evaluator = AlfWorldOrganizationEvaluator(
                FakeBackend(),
                AlfWorldEvaluatorConfig(
                    max_steps=1,
                    parallel_envs=1,
                    api_concurrency=1,
                    ignore_assigned_skills=True,
                    max_advisors=0,
                    save_steps=False,
                ),
            )
            trials = evaluator.evaluate(
                agents=[
                    AgentSpec(
                        name="Executor",
                        role="Environment executor",
                        responsibilities=["act"],
                        assigned_skills=["broken_skill"],
                        tool_permissions=["alfworld_action"],
                    ),
                    AgentSpec(
                        name="Advisor",
                        role="advisor",
                        responsibilities=["advise"],
                        assigned_skills=["broken_skill"],
                        tool_permissions=["alfworld_action"],
                    ),
                ],
                skills=[
                    Skill(
                        skill_name="broken_skill",
                        description="bad",
                        precondition="x",
                        action_protocol=["place without clean"],
                        applicable_atomic_ops=[AtomicOp.ACT],
                    )
                ],
                gamefiles=[
                    "/tmp/pick_clean_then_place_in_recep-Mug/game.tw-pddl"
                ],
                condition="baseline",
            )

        self.assertEqual(len(trials), 1)
        self.assertEqual(captured["assigned"], set())
        self.assertEqual(captured["injected"], [])
        self.assertEqual(captured["runtime_kwargs"]["max_advisors"], 0)

    def test_trajectory_grounded_distiller_preserves_required_operation_step(self):
        class FakeBackend:
            def complete(self, system_prompt, user_prompt):
                return LLMResult(
                    content=(
                        '{"skill_name":"pick_clean_then_place_in_recep",'
                        '"description":"Grounded clean operation",'
                        '"precondition":"Target is held",'
                        '"action_protocol":["take soapbar 1 from countertop 1",'
                        '"move soapbar 1 to countertop 1"],'
                        '"expected_effect":"Object placed",'
                        '"suggested_role":"execution_agent",'
                        '"target_failure_types":["cleaning_omission"],'
                        '"trajectory_summary":"Grounded from one failed clean trajectory."}'
                    )
                )

        trajectories = AlfWorldTrajectoryAdapter().adapt_many(
            _operation_pair(
                "pick_clean_then_place_in_recep-test",
                "clean",
                task="put a clean mug on the desk",
            )
        )
        skills = TrajectoryGroundedSkillDistiller(
            backend=FakeBackend(),
            seed_distiller=HeuristicSkillDistiller(
                DistillationConfig(operation_min_support=1)
            ),
            min_wins_for_protocol=1,
        ).distill(trajectories)

        self.assertTrue(
            any("clean" in step.lower() for step in skills[0].action_protocol)
        )

    def test_clean_skill_requires_target_pickup_to_be_applicable(self):
        skill = Skill(
            skill_name="Clean the target object before placement",
            description="Clean the target object",
            precondition="The target object is held.",
            action_protocol=["Clean it"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={"source_signal": "missing_clean_operation"},
        )
        matcher = SkillPreconditionMatcher()
        task = "put a clean mug on the desk"
        gamefile = "/tmp/pick_clean_then_place_in_recep-Mug-None-Desk/x/game.tw-pddl"

        missing = matcher.evaluate(
            skill,
            task,
            gamefile,
            [{"action": "<action>go to cabinet 1</action>", "observation": "Open."}],
        )
        reached = matcher.evaluate(
            skill,
            task,
            gamefile,
            [
                {"action": "<action>go to cabinet 1</action>", "observation": "Open."},
                {"action": "<action>take mug 1 from cabinet 1</action>", "observation": "Taken."},
            ],
        )

        self.assertFalse(missing.applicable)
        self.assertTrue(reached.applicable)
        self.assertEqual(reached.matched_step, 2)

    def test_light_skill_does_not_fire_on_non_look_tasks(self):
        skill = Skill(
            skill_name="inspect_obj_with_desk_lamp_fallback",
            description="Inspect under lamp",
            precondition="Object held, lamp available",
            action_protocol=["use desklamp"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={
                "source_signal": "missing_light_operation",
                "primary_task_family": "look_at_obj_in_light",
            },
        )
        matcher = SkillPreconditionMatcher()
        pickup = [
            {"action": "<action>take bowl 1 from cabinet 1</action>", "observation": "Taken."}
        ]
        clean = matcher.evaluate(
            skill,
            "clean some bowl and put it in cabinet.",
            "/tmp/pick_clean_then_place_in_recep-Bowl-None-Cabinet/x/game.tw-pddl",
            pickup,
            current=True,
        )
        look = matcher.evaluate(
            skill,
            "examine the cd with the desklamp.",
            "/tmp/look_at_obj_in_light-CD-None-DeskLamp/x/game.tw-pddl",
            pickup,
            current=True,
        )

        self.assertFalse(clean.applicable)
        self.assertTrue(
            "look-at-object-in-light" in clean.reason
            or "does not match task family" in clean.reason
        )
        self.assertTrue(look.applicable)

    def test_heat_skill_does_not_fire_on_clean_task_after_pickup(self):
        skill = Skill(
            skill_name="heat_target_before_placement",
            description="Heat target",
            precondition="Held target needs heating",
            action_protocol=["heat"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={
                "source_signal": "missing_heat_operation",
                "primary_task_family": "pick_heat_then_place_in_recep",
            },
        )
        matcher = SkillPreconditionMatcher()
        pickup = [
            {"action": "<action>take bowl 1 from cabinet 1</action>", "observation": "Taken."}
        ]
        misfire = matcher.evaluate(
            skill,
            "clean some bowl and put it in cabinet.",
            "/tmp/pick_clean_then_place_in_recep-Bowl-None-Cabinet/x/game.tw-pddl",
            pickup,
            current=True,
        )
        heat = matcher.evaluate(
            skill,
            "heat some mug and put it in coffeemachine.",
            "/tmp/pick_heat_then_place_in_recep-Mug-None-CoffeeMachine/x/game.tw-pddl",
            [
                {
                    "action": "<action>take mug 1 from cabinet 1</action>",
                    "observation": "Taken.",
                }
            ],
            current=True,
        )

        self.assertFalse(misfire.applicable)
        self.assertTrue(heat.applicable)

    def test_primary_task_family_gate_blocks_cross_family_skills(self):
        skill = Skill(
            skill_name="pick_clean_then_place_in_recep",
            description="Clean then place",
            precondition="Held dirty object",
            action_protocol=["clean"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={
                "source_signal": "missing_clean_operation",
                "primary_task_family": "pick_clean_then_place_in_recep",
            },
        )
        matcher = SkillPreconditionMatcher()
        result = matcher.evaluate(
            skill,
            "put two soapbar in garbagecan.",
            "/tmp/pick_two_obj_and_place-SoapBar-None-GarbageCan/x/game.tw-pddl",
            [
                {
                    "action": "<action>take soapbar 1 from countertop 1</action>",
                    "observation": "Taken.",
                }
            ],
            current=True,
        )

        self.assertFalse(result.applicable)
        self.assertIn("does not match task family", result.reason)

    def test_family_gate_infers_declared_family_from_source_signal(self):
        skill = Skill(
            skill_name="inspect_obj_with_desk_lamp_fallback",
            description="Inspect under lamp",
            precondition="Object held",
            action_protocol=["use desklamp"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={"source_signal": "missing_light_operation"},
        )
        matcher = SkillPreconditionMatcher()
        result = matcher.evaluate(
            skill,
            "heat some apple and put it in fridge.",
            "/tmp/pick_heat_then_place_in_recep-Apple-None-Fridge/x/game.tw-pddl",
            [
                {
                    "action": "<action>take apple 1 from countertop 1</action>",
                    "observation": "Taken.",
                }
            ],
            current=True,
        )

        self.assertFalse(result.applicable)
        self.assertIn("does not match task family", result.reason)

    def test_skill_distribution_shift_proposes_new_agent(self):
        skill = Skill(
            skill_name="Verify completion",
            description="Verify every subgoal before termination",
            precondition="Before termination",
            action_protocol=["Verify"],
            applicable_atomic_ops=[AtomicOp.VERIFY],
            suggested_role="Verifier",
            status=SkillStatus.VERIFIED,
            marginal_utility=0.2,
            evidence_ids=["t1", "t2"],
            support_count=2,
            distribution_shift=DistributionShift(
                kl_divergence=0.8,
                shift_score=0.6,
                baseline_source="skill_bank",
            ),
        )
        agents = [
            AgentSpec(
                name="Executor",
                role="Environment executor",
                responsibilities=["Execute actions"],
                tool_permissions=[],
            )
        ]
        gap = AssignmentGapEstimator().estimate([skill], agents)
        edit = OrganizationEditor(
            OrganizationPolicy(
                distribution_shift_threshold=0.15,
                min_cluster_support=2,
                min_assignment_gap=0.0,
            )
        ).propose([skill], gap)

        self.assertEqual(edit.edit_type, OrganizationEditType.ADD_AGENT)
        self.assertIsNotNone(edit.new_agent)
        self.assertEqual(
            edit.new_agent.role,
            "VerifyCompletionSpecialist",
        )

    def test_add_agent_bootstraps_when_adherence_unobserved(self):
        skill = Skill(
            skill_name="Clean Object and Place in Receptacle",
            description="clean then place",
            precondition="need clean object",
            action_protocol=[
                "go to <location>",
                "take <object> from <receptacle>",
                "clean <object> with <tool>",
                "put <object> in/on <receptacle>",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            suggested_role="TransformCleanSpecialist",
            status=SkillStatus.VERIFIED,
            capability_key="transform.clean",
            applicable_task_families=["pick_clean_then_place_in_recep"],
            evidence_ids=["t1"],
            support_count=1,
            distribution_shift=DistributionShift(
                kl_divergence=0.5,
                shift_score=0.5,
                baseline_source="skill_bank",
            ),
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "required_tools": ["alfworld_action"],
                # Placeholder 0.0 with uses=0 must NOT block ADD_AGENT.
                "skill_credit": {
                    "uses": 0,
                    "successes": 0,
                    "score": 0.5,
                    "mean_protocol_adherence": 0.0,
                    "events": [],
                },
            },
        )
        agents = [
            AgentSpec(
                name="Executor",
                role="Environment executor",
                responsibilities=["Execute actions"],
                tool_permissions=["alfworld_action"],
            )
        ]
        editor = OrganizationEditor(
            OrganizationPolicy(
                distribution_shift_threshold=0.15,
                min_cluster_support=1,
                min_assignment_gap=0.0,
                require_executable_protocol=True,
                min_protocol_adherence_for_add_agent=0.34,
            )
        )
        self.assertIsNone(editor._skills_ready_for_add_agent([skill]))
        gap = AssignmentGapEstimator().estimate([skill], agents)
        edit = editor.propose([skill], gap, existing_agents=agents)
        self.assertEqual(edit.edit_type, OrganizationEditType.ADD_AGENT)

        # Once a skill has real low adherence measurements, ADD_AGENT blocks.
        skill.metadata["skill_credit"] = {
            "uses": 2,
            "successes": 0,
            "score": 0.25,
            "mean_protocol_adherence": 0.0,
            "adherence_scores": [0.0, 0.0],
            "events": [],
        }
        reason = editor._skills_ready_for_add_agent([skill])
        self.assertIsNotNone(reason)
        self.assertIn("mean protocol adherence", reason)

    def test_add_agent_fills_role_instance_attributes(self):
        skill = Skill(
            skill_name="Clean the target object before placement",
            description="Execute the required sinkbasin cleaning operation",
            precondition="Task requires a clean object and cleaning not yet observed",
            action_protocol=[
                "Navigate to a sinkbasin",
                "Execute 'clean <object> with <sinkbasin>'",
            ],
            applicable_atomic_ops=[AtomicOp.PLAN, AtomicOp.ACT, AtomicOp.VERIFY],
            suggested_role="Executor",
            status=SkillStatus.VERIFIED,
            marginal_utility=0.2,
            evidence_ids=["t1", "t2"],
            support_count=2,
            distribution_shift=DistributionShift(
                kl_divergence=0.8,
                shift_score=0.6,
                baseline_source="skill_bank",
            ),
            metadata={
                "source_signal": "missing_clean_operation",
                "primary_task_family": "pick_clean_then_place_in_recep",
                "required_tools": ["alfworld_action"],
            },
        )
        agents = [
            AgentSpec(
                name="Executor",
                role="Environment executor",
                responsibilities=["Execute actions"],
                tool_permissions=["alfworld_action"],
            )
        ]
        gap = AssignmentGapEstimator().estimate([skill], agents)
        edit = OrganizationEditor(
            OrganizationPolicy(
                distribution_shift_threshold=0.15,
                min_cluster_support=2,
                min_assignment_gap=0.0,
                probation_games=3,
            )
        ).propose(
            [skill],
            gap,
            existing_agent_names={"Executor"},
            executor_name="Executor",
        )

        self.assertEqual(edit.edit_type, OrganizationEditType.ADD_AGENT)
        agent = edit.new_agent
        self.assertIsNotNone(agent)
        assert agent is not None
        self.assertEqual(agent.name, "MissingCleanOperationSpecialist")
        self.assertEqual(agent.role, "MissingCleanOperationSpecialist")
        self.assertTrue(agent.role_specification)
        self.assertIn("complete the full environment task", agent.role_specification.lower())
        self.assertIn("after dispatch", agent.responsibility_boundary.lower())
        self.assertIn("<action>", agent.output_protocol)
        self.assertEqual(
            agent.assigned_skills,
            ["Clean the target object before placement"],
        )
        self.assertTrue(agent.input_protocol)
        self.assertTrue(agent.output_protocol)
        self.assertEqual(agent.communication_edges, ["Executor"])
        self.assertIn("alfworld_action", agent.tool_permissions)
        self.assertTrue(agent.activation_condition)
        self.assertEqual(agent.token_budget, 512)
        self.assertIsNone(agent.turn_budget)
        self.assertTrue(agent.retirement_condition)
        self.assertEqual(agent.shadow_evaluation_record.get("status"), "pending")
        self.assertEqual(
            agent.shadow_evaluation_record.get("acting_status"),
            "probation",
        )
        self.assertEqual(
            agent.shadow_evaluation_record.get("trial_games_remaining"),
            3,
        )
        self.assertTrue(
            agent.shadow_evaluation_record.get("dispatch_only")
        )
        self.assertEqual(
            agent.shadow_evaluation_record.get("capability_name"),
            "missing_clean_operation",
        )

    def test_add_agent_transfers_skill_off_executor(self):
        skill_name = "Trajectory-derived transform clean protocol"
        specialist = AgentSpec(
            name="TransformCleanSpecialist",
            role="TransformCleanSpecialist",
            responsibilities=["Clean then place"],
            assigned_skills=[skill_name],
            tool_permissions=["alfworld_action"],
        )
        edit = OrganizationEdit(
            edit_type=OrganizationEditType.ADD_AGENT,
            rationale="test add_agent keeps executor copy",
            new_agent=specialist,
            assigned_skill_names=[skill_name],
            assignment_gap=0.5,
        )
        agents = [
            AgentSpec(
                name="Executor",
                role="Environment executor",
                responsibilities=["Execute actions"],
                tool_permissions=["alfworld_action"],
            )
        ]
        candidate = OrganizationStateManager.apply_candidate(agents, [edit])
        by_name = {agent.name: agent for agent in candidate}
        self.assertIn(skill_name, by_name["TransformCleanSpecialist"].assigned_skills)
        self.assertNotIn(skill_name, by_name["Executor"].assigned_skills)

    def test_assign_skill_to_specialist_transfers_off_executor(self):
        skill_name = "Trajectory-derived transform heat protocol"
        agents = [
            AgentSpec(
                name="Executor",
                role="Environment executor",
                responsibilities=["Execute actions"],
                tool_permissions=["alfworld_action"],
            ),
            AgentSpec(
                name="TransformHeatSpecialist",
                role="TransformHeatSpecialist",
                responsibilities=["Heat then place"],
                tool_permissions=["alfworld_action"],
            ),
        ]
        edit = OrganizationEdit(
            edit_type=OrganizationEditType.ASSIGN_SKILL,
            rationale="test assign keeps executor copy",
            target_agent="TransformHeatSpecialist",
            assigned_skill_names=[skill_name],
            assignment_gap=0.2,
        )
        candidate = OrganizationStateManager.apply_candidate(agents, [edit])
        by_name = {agent.name: agent for agent in candidate}
        self.assertIn(skill_name, by_name["TransformHeatSpecialist"].assigned_skills)
        self.assertNotIn(skill_name, by_name["Executor"].assigned_skills)

    def test_success_protocol_activates_before_pickup(self):
        matcher = SkillPreconditionMatcher()
        skill = Skill(
            skill_name="Heat protocol",
            description="Heat then place",
            precondition="pick_heat_then_place_in_recep",
            action_protocol=["heat <object> with <tool>"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            metadata={
                "source_signal": "heat_operation_success",
                "primary_task_family": "pick_heat_then_place_in_recep",
                "protocol_source": "merged_success_detour_clean_v1",
            },
            applicable_task_families=["pick_heat_then_place_in_recep"],
        )
        result = matcher.evaluate(
            skill,
            task="heat some apple and put it in fridge",
            gamefile=(
                "/data/pick_heat_then_place_in_recep-Apple-None-Fridge-10/"
                "trial/game.tw-pddl"
            ),
            steps=[],
            current=True,
        )
        self.assertTrue(result.applicable)
        self.assertIn("Full-episode", result.reason)

    def test_success_protocol_stays_active_after_transform(self):
        matcher = SkillPreconditionMatcher()
        skill = Skill(
            skill_name="Clean protocol",
            description="Clean then place",
            precondition="Task is not yet won",
            action_protocol=[
                "clean <object> with <tool>",
                "move <object> to <receptacle>",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.PROVISIONAL,
            capability_key="transform.clean",
            metadata={
                "source_signal": "clean_operation_success",
                "primary_task_family": "pick_clean_then_place_in_recep",
                "protocol_source": "merged_success_detour_clean_v1",
            },
            applicable_task_families=["pick_clean_then_place_in_recep"],
        )
        after_clean = matcher.evaluate(
            skill,
            task="put a clean mug on the desk",
            gamefile=(
                "/data/pick_clean_then_place_in_recep-Mug-None-Desk-1/"
                "trial/game.tw-pddl"
            ),
            steps=[
                {
                    "action": "take mug 1 from countertop 1",
                    "observation": "You pick up the mug 1.",
                },
                {
                    "action": "clean mug 1 with sinkbasin 1",
                    "observation": "You clean the mug 1 using the sinkbasin 1.",
                },
            ],
            current=True,
        )
        self.assertTrue(after_clean.applicable)
        self.assertIn("Full-episode", after_clean.reason)

    def test_efficient_execution_protocol_activates_on_family_match(self):
        matcher = SkillPreconditionMatcher()
        skill = Skill(
            skill_name="Trajectory-derived execution efficiency protocol",
            description="Aligned pick_and_place protocol",
            precondition="Task family matches pick_and_place",
            action_protocol=[
                "go to <location>",
                "take <object> from <receptacle>",
                "move <object> to <receptacle>",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.PROVISIONAL,
            capability_key="execution.efficiency",
            applicable_task_families=["pick_and_place"],
            metadata={
                "source_signal": "efficient_execution",
                "primary_task_family": "pick_and_place",
                "protocol_source": "aligned_successful_trajectory_stages_v2",
                "anchor_state": "successful trajectory completed within 6 steps",
            },
        )
        place = matcher.evaluate(
            skill,
            task="put some candle on drawer",
            gamefile=(
                "/data/pick_and_place_simple-Candle-None-Drawer-411/"
                "trial/game.tw-pddl"
            ),
            steps=[],
            current=True,
        )
        cool = matcher.evaluate(
            skill,
            task="cool some apple and put it in fridge",
            gamefile=(
                "/data/pick_cool_then_place_in_recep-Apple-None-Fridge-10/"
                "trial/game.tw-pddl"
            ),
            steps=[],
            current=True,
        )
        active, _ = matcher.active_skills(
            [skill],
            task="put some candle on drawer",
            gamefile=(
                "/data/pick_and_place_simple-Candle-None-Drawer-411/"
                "trial/game.tw-pddl"
            ),
            steps=[{"action": "look", "observation": "You are in the middle."}],
        )

        self.assertTrue(place.applicable)
        self.assertIn("Full-episode", place.reason)
        self.assertFalse(cool.applicable)
        self.assertEqual([item.skill_name for item in active], [skill.skill_name])

    def test_executor_alfworld_prompt_includes_assigned_verified_skill(self):
        class FakeBackend:
            def __init__(self):
                self.user_prompt = ""

            def complete(self, system_prompt, user_prompt):
                self.user_prompt = user_prompt
                return LLMResult("<action>look</action>")

        skill = Skill(
            skill_name="Assigned clean",
            description="Clean protocol",
            precondition="clean task",
            action_protocol=["clean <object> with <tool>"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
        )
        backend = FakeBackend()
        runtime = MASRuntime(
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Act"],
                    assigned_skills=[skill.skill_name],
                )
            ],
            skills=[skill],
            backend=backend,
            allow_executor_skill_injection=True,
        )
        runtime.act("Observation text")
        self.assertIn(skill.skill_name, backend.user_prompt)
        self.assertIn("Executor fallback", backend.user_prompt)

    def test_executor_dispatch_picks_clean_operator_for_clean_task(self):
        from sage_mas.executor_dispatch import ExecutorDispatcher

        class PickCleanerBackend:
            def __init__(self):
                self.user_prompt = ""

            def complete(self, system_prompt, user_prompt):
                self.user_prompt = user_prompt
                return LLMResult(
                    "<assign>CleanOperator</assign>",
                    prompt_tokens=1,
                    completion_tokens=1,
                )

        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Fallback"],
            tool_permissions=["alfworld_action"],
        )
        cleaner = AgentSpec(
            name="CleanOperator",
            role="CleanOperator",
            responsibilities=["Clean stage"],
            assigned_skills=["Clean the target object before placement"],
            tool_permissions=["clean", "alfworld_action"],
            shadow_evaluation_record={
                "task_families": ["pick_clean_then_place_in_recep"],
                "task_performance": {
                    "dispatches": 3,
                    "wins": 2,
                },
            },
        )
        skill = Skill(
            skill_name="Clean the target object before placement",
            description="Clean then place",
            precondition="Held",
            action_protocol=["clean"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            metadata={
                "source_signal": "missing_clean_operation",
                "primary_task_family": "pick_clean_then_place_in_recep",
            },
        )

        backend = PickCleanerBackend()
        assignment = ExecutorDispatcher(backend=backend).assign(
            task="put a clean mug in cabinet",
            task_family="pick_clean_then_place_in_recep",
            agents=[executor, cleaner],
            skills=[skill],
            gamefile="/fake/pick_clean_then_place_in_recep-0/game.tw-pddl",
        )
        self.assertEqual(assignment.primary_agent, "CleanOperator")
        self.assertEqual(assignment.dispatch_layer, "eligibility_single")
        self.assertEqual(assignment.eligible_agents, ["CleanOperator"])
        # Single eligible specialist is auto-assigned; LLM is not required.

    def test_add_agent_reuses_existing_same_role(self):
        skill = Skill(
            skill_name="Clean the target object before placement",
            description="Clean then place",
            precondition="Held",
            action_protocol=["clean"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            evidence_ids=["a", "b"],
            support_count=2,
            distribution_shift=DistributionShift(
                kl_divergence=0.5,
                shift_score=0.4,
                baseline_source="skill_bank",
            ),
            metadata={
                "source_signal": "missing_clean_operation",
                "primary_task_family": "pick_clean_then_place_in_recep",
            },
        )
        existing = [
            AgentSpec(
                name="Executor",
                role="Environment executor",
                responsibilities=["Act"],
                tool_permissions=["alfworld_action"],
            ),
            AgentSpec(
                name="MissingCleanOperationSpecialist",
                role="MissingCleanOperationSpecialist",
                responsibilities=["Clean"],
                assigned_skills=[],
                tool_permissions=["alfworld_action"],
            ),
        ]
        gap = AssignmentGapEstimator().estimate([skill], existing)
        edit = OrganizationEditor(
            OrganizationPolicy(distribution_shift_threshold=0.15, min_cluster_support=2)
        ).propose(
            [skill],
            gap,
            existing_agent_names={agent.name for agent in existing},
            existing_agents=existing,
        )
        self.assertEqual(edit.edit_type, OrganizationEditType.ASSIGN_SKILL)
        self.assertEqual(
            edit.target_agent,
            "MissingCleanOperationSpecialist",
        )

    def test_no_skill_change_assigns_to_existing_agent(self):
        skill = Skill(
            skill_name="Prevent unproductive action repetition",
            description="Avoid loops",
            precondition="Before acting",
            action_protocol=["Check history"],
            applicable_atomic_ops=[AtomicOp.ACT],
            suggested_role="Executor",
            status=SkillStatus.VERIFIED,
            evidence_ids=["t1", "t2", "t3"],
        )
        agents = [
            AgentSpec(
                name="Executor",
                role="Environment executor",
                responsibilities=["Execute actions"],
                tool_permissions=["alfworld_action"],
            )
        ]
        gap = AssignmentGapResult(
            skill_ids=[skill.skill_id],
            assignment_gap=0.9,
            best_agent_id=agents[0].agent_id,
            best_agent_name=agents[0].name,
            best_capacity=0.1,
            capacities={agents[0].agent_id: 0.1},
        )
        edit = OrganizationEditor().propose([skill], gap)

        self.assertEqual(edit.edit_type, OrganizationEditType.ASSIGN_SKILL)
        self.assertEqual(edit.target_agent, "Executor")

    def test_distiller_ignores_failed_trajectory_without_positive_effect(self):
        adapter = AlfWorldTrajectoryAdapter()
        failed = _trajectory("pick_two_obj_and_place-Safe-1", won=False)
        succeeded = _trajectory("pick_two_obj_and_place-Safe-2", won=True)
        adapted = adapter.adapt_many([failed, succeeded])
        skills = HeuristicSkillDistiller(
            DistillationConfig(high_cost_steps=20, min_support=1)
        ).distill(adapted)

        self.assertEqual(len(skills), 1)
        signals = {skill.metadata["source_signal"] for skill in skills}
        self.assertNotIn("repetition_guard", signals)
        self.assertEqual(signals, {"won"})

    def test_distiller_does_not_create_skills_from_failed_trajectories(self):
        adapter = AlfWorldTrajectoryAdapter()
        trajectories = [
            {
                "gamefile": "/tmp/pick_two_obj_and_place-Mug-None-Desk/game.tw-pddl",
                "task": "put two mug in desk",
                "won": False,
                "num_steps": 25,
                "steps": [
                    {
                        "observation": "Nothing happens.",
                        "action": "look",
                        "is_action_valid": True,
                    },
                    {
                        "observation": "Nothing happens.",
                        "action": "look",
                        "is_action_valid": False,
                    },
                ],
            },
            {
                "gamefile": "/tmp/pick_two_obj_and_place-Key-None-Safe/game.tw-pddl",
                "task": "put two key in safe",
                "won": False,
                "num_steps": 25,
                "steps": [
                    {
                        "observation": "Nothing happens.",
                        "action": "go to safe 1",
                        "is_action_valid": True,
                    },
                    {
                        "observation": "Nothing happens.",
                        "action": "go to safe 1",
                        "is_action_valid": True,
                    },
                ],
            },
        ]
        skills = HeuristicSkillDistiller(
            DistillationConfig(high_cost_steps=20, min_support=1)
        ).distill(adapter.adapt_many(trajectories))
        self.assertEqual(skills, [])

    def test_distiller_clusters_evidence_by_task_family(self):
        adapter = AlfWorldTrajectoryAdapter()
        trajectories = [
            {
                "gamefile": "/tmp/look_at_obj_in_light-Book-None-DeskLamp-308/game.tw-pddl",
                "task_family": "look_at_obj_in_light",
                "task": "look at book under the desklamp.",
                "won": True,
                "num_steps": 3,
                "steps": [
                    {
                        "observation": "Nothing happens.",
                        "action": "look",
                        "is_action_valid": True,
                    },
                    {
                        "observation": "Nothing happens.",
                        "action": "look",
                        "is_action_valid": True,
                    },
                ],
            },
            {
                "gamefile": "/tmp/pick_two_obj_and_place-CD-None-Safe-308/game.tw-pddl",
                "task_family": "pick_two_obj_and_place",
                "task": "put two cd in safe.",
                "won": True,
                "num_steps": 20,
                "steps": [
                    {"observation": "You are in the middle of a room.", "action": "look", "is_action_valid": True},
                    {"observation": "You arrive at shelf 1.", "action": "go to shelf 1", "is_action_valid": True},
                    {"observation": "You arrive at shelf 2.", "action": "go to shelf 2", "is_action_valid": True},
                    {"observation": "You arrive at shelf 3.", "action": "go to shelf 3", "is_action_valid": True},
                    {"observation": "You arrive at drawer 1.", "action": "go to drawer 1", "is_action_valid": True},
                ],
            },
            {
                "gamefile": "/tmp/pick_two_obj_and_place-Pillow-None-Sofa-219/game.tw-pddl",
                "task_family": "pick_two_obj_and_place",
                "task": "put two pillow in sofa.",
                "won": True,
                "num_steps": 20,
                "steps": [
                    {"observation": "You are in the middle of a room.", "action": "look", "is_action_valid": True},
                    {"observation": "You arrive at shelf 1.", "action": "go to shelf 1", "is_action_valid": True},
                    {"observation": "You arrive at shelf 2.", "action": "go to shelf 2", "is_action_valid": True},
                    {"observation": "You arrive at shelf 3.", "action": "go to shelf 3", "is_action_valid": True},
                    {"observation": "You arrive at drawer 1.", "action": "go to drawer 1", "is_action_valid": True},
                ],
            },
        ]
        skills = HeuristicSkillDistiller(
            DistillationConfig(
                high_cost_steps=20,
                min_support=1,
                operation_min_support=1,
            )
        ).distill(adapter.adapt_many(trajectories))

        families = {
            skill.metadata["primary_task_family"] for skill in skills
        }
        self.assertIn("look_at_obj_in_light", families)
        self.assertIn("pick_two_obj_and_place", families)
        self.assertTrue(
            all(skill.metadata["distiller"] == "experience-cluster-v2" for skill in skills)
        )
        self.assertTrue(
            all("trigger_states" in skill.metadata for skill in skills)
        )
        pick_two_skill = next(
            skill
            for skill in skills
            if skill.metadata["primary_task_family"] == "pick_two_obj_and_place"
        )
        self.assertEqual(
            pick_two_skill.metadata["source_signal"],
            "won",
        )

    def test_empirical_capacity_can_override_lexical_assignment(self):
        skill = Skill(
            skill_name="Verify completion",
            description="Verify completion",
            precondition="Before termination",
            action_protocol=["Verify"],
            applicable_atomic_ops=[AtomicOp.VERIFY],
            suggested_role="Verifier",
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
        )
        verifier = AgentSpec(
            name="Verifier",
            role="Verifier",
            responsibilities=["Verify"],
        )
        result = AssignmentGapEstimator(
            empirical_weight=1.0
        ).estimate(
            [skill],
            [executor, verifier],
            empirical_capacities={
                (executor.agent_id, skill.skill_id): 0.9,
                (verifier.agent_id, skill.skill_id): 0.2,
            },
        )

        self.assertEqual(result.best_agent_name, "Executor")
        self.assertAlmostEqual(result.assignment_gap, 0.1)
        self.assertEqual(result.empirical_coverage, 1.0)

    def test_new_agent_name_is_unique_across_rounds(self):
        skill = Skill(
            skill_name="Verify completion",
            description="Verify completion",
            precondition="Before termination",
            action_protocol=["Verify"],
            applicable_atomic_ops=[AtomicOp.VERIFY],
            suggested_role="Verifier",
            status=SkillStatus.VERIFIED,
            evidence_ids=["a", "b", "c"],
            distribution_shift=DistributionShift(
                kl_divergence=0.5,
                shift_score=0.4,
                baseline_source="skill_bank",
            ),
        )
        gap = AssignmentGapResult(
            skill_ids=[skill.skill_id],
            assignment_gap=0.9,
            best_agent_id=None,
            best_agent_name=None,
            best_capacity=0.1,
            capacities={},
        )
        edit = OrganizationEditor(
            OrganizationPolicy(
                distribution_shift_threshold=0.15,
                min_cluster_support=2,
            )
        ).propose(
            [skill],
            gap,
            existing_agent_names={"VerifyCompletionSpecialist"},
        )

        self.assertEqual(
            edit.new_agent.name,
            "VerifyCompletionSpecialist-2",
        )

    def test_new_agent_starts_dispatch_only_probation(self):
        skill = Skill(
            skill_name="Clean protocol",
            description="d",
            precondition="p",
            action_protocol=["clean object"],
            applicable_atomic_ops=[AtomicOp.ACT],
            suggested_role="Cleaner",
            status=SkillStatus.VERIFIED,
            evidence_ids=["a", "b", "c"],
            distribution_shift=DistributionShift(
                kl_divergence=0.5,
                shift_score=0.4,
                baseline_source="skill_bank",
            ),
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "source_signal": "missing_clean_operation",
            },
        )
        gap = AssignmentGapResult(
            skill_ids=[skill.skill_id],
            assignment_gap=0.9,
            best_agent_id=None,
            best_agent_name=None,
            best_capacity=0.1,
            capacities={},
        )
        edit = OrganizationEditor(
            OrganizationPolicy(
                distribution_shift_threshold=0.15,
                min_cluster_support=2,
                dispatch_only_new_agents=True,
                probation_games=3,
            )
        ).propose([skill], gap)

        self.assertEqual(edit.edit_type, OrganizationEditType.ADD_AGENT)
        agent = edit.new_agent
        self.assertIsNotNone(agent)
        record = agent.shadow_evaluation_record or {}
        self.assertEqual(record.get("acting_status"), "probation")
        self.assertTrue(record.get("dispatch_only"))
        self.assertEqual(record.get("trial_games_remaining"), 3)
        self.assertIn("alfworld_action", agent.tool_permissions)
        self.assertFalse(MASRuntime._is_stage_actor(agent))
        self.assertFalse(MASRuntime._is_dormant(agent))
        self.assertTrue(MASRuntime._may_take_dispatched_episode(agent))

    def test_executor_can_dispatch_dormant_specialist(self):
        from sage_mas.executor_dispatch import (
            ExecutorDispatchConfig,
            ExecutorDispatcher,
        )

        class PickCleanerBackend:
            def complete(self, system_prompt, user_prompt):
                return LLMResult(
                    "<assign>CleanOperator</assign>",
                    prompt_tokens=1,
                    completion_tokens=1,
                )

        skill = Skill(
            skill_name="Clean protocol",
            description="Clean objects at sink",
            precondition="Need a cleaned object",
            action_protocol=["clean object with sinkbasin"],
            applicable_atomic_ops=[AtomicOp.ACT],
            suggested_role="CleanOperator",
            status=SkillStatus.VERIFIED,
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "source_signal": "missing_clean_operation",
                "task_families": ["pick_clean_then_place_in_recep"],
            },
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
            tool_permissions=["alfworld_action"],
        )
        specialist = AgentSpec(
            name="CleanOperator",
            role="CleanOperator",
            responsibilities=["Clean"],
            assigned_skills=[skill.skill_name],
            tool_permissions=["clean", "alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "dormant",
                "task_families": ["pick_clean_then_place_in_recep"],
            },
        )
        gamefile = "/fake/pick_clean_then_place_in_recep-0/game.tw-pddl"

        blocked = ExecutorDispatcher(
            ExecutorDispatchConfig(allow_dormant=False),
        ).assign(
            task="put a clean mug on the desk",
            task_family="pick_clean_then_place_in_recep",
            agents=[executor, specialist],
            skills=[skill],
            gamefile=gamefile,
        )
        self.assertEqual(blocked.primary_agent, "Executor")
        self.assertEqual(blocked.dispatch_layer, "eligibility_empty")

        dispatched = ExecutorDispatcher(
            ExecutorDispatchConfig(allow_dormant=True),
            backend=PickCleanerBackend(),
        ).assign(
            task="put a clean mug on the desk",
            task_family="pick_clean_then_place_in_recep",
            agents=[executor, specialist],
            skills=[skill],
            gamefile=gamefile,
        )
        self.assertEqual(dispatched.primary_agent, "CleanOperator")
        self.assertEqual(dispatched.dispatch_layer, "eligibility_single")
        self.assertIn("executor-dispatch", dispatched.rationale)

        runtime = MASRuntime(
            agents=[executor, specialist],
            skills=[skill],
            backend=type(
                "B",
                (),
                {
                    "complete": staticmethod(
                        lambda system, user: LLMResult(
                            "<think>x</think><action>look</action>",
                            prompt_tokens=1,
                            completion_tokens=1,
                        )
                    )
                },
            )(),
        )
        actor = runtime._select_step_actor(
            {skill.skill_name},
            preferred_actor_name="CleanOperator",
        )
        self.assertEqual(actor.name, "CleanOperator")
        # Without Executor dispatch, dormant must not auto-steal the stage.
        auto = runtime._select_step_actor({skill.skill_name})
        self.assertEqual(auto.name, "Executor")

    def test_llm_dispatch_keeps_executor_when_assign_fails(self):
        from sage_mas.executor_dispatch import (
            ExecutorDispatchConfig,
            ExecutorDispatcher,
        )

        skill_a = Skill(
            skill_name="Clean protocol",
            description="Clean objects at sink",
            precondition="Need a cleaned object",
            action_protocol=["clean object with sinkbasin"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            suggested_role="CleanOperator",
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "source_signal": "missing_clean_operation",
                "task_families": ["pick_clean_then_place_in_recep"],
            },
        )
        skill_b = Skill(
            skill_name="Search clean rooms",
            description="Search for clean targets",
            precondition="Need target",
            action_protocol=["go to"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            suggested_role="Searcher",
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "source_signal": "search_exhaustion",
                "task_families": ["pick_clean_then_place_in_recep"],
            },
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
            tool_permissions=["alfworld_action"],
        )
        cleaner = AgentSpec(
            name="CleanOperator",
            role="CleanOperator",
            responsibilities=["Clean"],
            assigned_skills=[skill_a.skill_name],
            tool_permissions=["clean", "alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "dormant",
                "task_families": ["pick_clean_then_place_in_recep"],
            },
        )
        searcher = AgentSpec(
            name="Searcher",
            role="Searcher",
            responsibilities=["Search"],
            assigned_skills=[skill_b.skill_name],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "dormant",
                "task_families": ["pick_clean_then_place_in_recep"],
            },
        )

        class BadBackend:
            def complete(self, system_prompt, user_prompt):
                return LLMResult("I am unsure.", prompt_tokens=1, completion_tokens=1)

        assignment = ExecutorDispatcher(
            ExecutorDispatchConfig(
                allow_dormant=True,
                auto_assign_single=False,
                prefer_matching_specialist=False,
            ),
            backend=BadBackend(),
        ).assign(
            task="put a clean mug on the desk",
            task_family="pick_clean_then_place_in_recep",
            agents=[executor, cleaner, searcher],
            skills=[skill_a, skill_b],
            gamefile="/fake/pick_clean_then_place_in_recep-0/game.tw-pddl",
        )
        self.assertEqual(assignment.primary_agent, "Executor")
        self.assertIn("llm-keep-executor", assignment.rationale)
        self.assertGreaterEqual(len(assignment.eligible_agents), 2)

    def test_dispatch_eligibility_empty_keeps_executor(self):
        from sage_mas.executor_dispatch import ExecutorDispatcher

        heat = Skill(
            skill_name="Heat protocol",
            description="Heat objects",
            precondition="Need heat",
            action_protocol=["heat"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={
                "primary_task_family": "pick_heat_then_place_in_recep",
                "source_signal": "missing_heat_operation",
            },
        )
        assignment = ExecutorDispatcher().assign(
            task="put a clean mug on the desk",
            task_family="pick_clean_then_place_in_recep",
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Act"],
                    tool_permissions=["alfworld_action"],
                ),
                AgentSpec(
                    name="HeatOperator",
                    role="HeatOperator",
                    responsibilities=["Heat"],
                    assigned_skills=[heat.skill_name],
                    tool_permissions=["heat", "alfworld_action"],
                    shadow_evaluation_record={
                        "acting_status": "dormant",
                        "task_families": ["pick_heat_then_place_in_recep"],
                    },
                ),
            ],
            skills=[heat],
            gamefile="/fake/pick_clean_then_place_in_recep-0/game.tw-pddl",
        )
        self.assertEqual(assignment.primary_agent, "Executor")
        self.assertEqual(assignment.dispatch_layer, "eligibility_empty")
        self.assertEqual(assignment.eligible_agents, [])

    def test_dispatch_llm_among_multiple_eligible(self):
        from sage_mas.executor_dispatch import (
            ExecutorDispatchConfig,
            ExecutorDispatcher,
        )

        skill_a = Skill(
            skill_name="Clean protocol",
            description="Clean",
            precondition="held",
            action_protocol=["clean"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "source_signal": "missing_clean_operation",
            },
        )
        skill_b = Skill(
            skill_name="Search clean",
            description="Search",
            precondition="lost",
            action_protocol=["go to"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "source_signal": "search_exhaustion",
            },
        )

        class PickSearcherBackend:
            def complete(self, system_prompt, user_prompt):
                assert "Eligible roster" in user_prompt
                assert "CleanOperator" in user_prompt
                assert "Searcher" in user_prompt
                return LLMResult(
                    "<assign>Searcher</assign>",
                    prompt_tokens=1,
                    completion_tokens=1,
                )

        assignment = ExecutorDispatcher(
            ExecutorDispatchConfig(allow_dormant=True),
            backend=PickSearcherBackend(),
        ).assign(
            task="put a clean egg in fridge",
            task_family="pick_clean_then_place_in_recep",
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Act"],
                    tool_permissions=["alfworld_action"],
                ),
                AgentSpec(
                    name="CleanOperator",
                    role="CleanOperator",
                    responsibilities=["Clean"],
                    assigned_skills=[skill_a.skill_name],
                    tool_permissions=["alfworld_action"],
                    shadow_evaluation_record={
                        "acting_status": "dormant",
                        "task_families": ["pick_clean_then_place_in_recep"],
                    },
                ),
                AgentSpec(
                    name="Searcher",
                    role="Searcher",
                    responsibilities=["Search"],
                    assigned_skills=[skill_b.skill_name],
                    tool_permissions=["alfworld_action"],
                    shadow_evaluation_record={
                        "acting_status": "dormant",
                        "task_families": ["pick_clean_then_place_in_recep"],
                    },
                ),
            ],
            skills=[skill_a, skill_b],
            gamefile="/fake/pick_clean_then_place_in_recep-0/game.tw-pddl",
        )
        self.assertEqual(assignment.primary_agent, "Searcher")
        self.assertEqual(assignment.dispatch_layer, "llm")
        self.assertEqual(
            set(assignment.eligible_agents),
            {"CleanOperator", "Searcher"},
        )

    def test_novel_skill_with_low_assignment_gap_reuses_existing_agent(self):
        skill = Skill(
            skill_name="Novel verified capability",
            description="Learned capability",
            precondition="Learned precondition",
            action_protocol=["learned action"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            marginal_utility=0.2,
            evidence_ids=["a", "b", "c"],
        )
        shift = DistributionShift(
            shift_score=0.6,
            novelty_score=0.6,
            metric="cosine_nearest_skill",
            baseline_source="embedding_skill_bank",
            failure_rate=1.0,
        )
        skill.distribution_shift = shift
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
        )
        gap = AssignmentGapResult(
            skill_ids=[skill.skill_id],
            assignment_gap=0.2,
            best_agent_id=executor.agent_id,
            best_agent_name=executor.name,
            best_capacity=0.8,
            capacities={executor.agent_id: 0.8},
        )

        edit = OrganizationEditor(
            OrganizationPolicy(
                min_cluster_support=3,
                min_assignment_gap=0.4,
            )
        ).propose(
            [skill],
            gap,
            existing_agents=[executor],
        )

        self.assertEqual(edit.edit_type, OrganizationEditType.ASSIGN_SKILL)
        self.assertEqual(edit.target_agent, "Executor")

    def test_dispatch_contract_gate_rejects_wrong_capability_before_llm(self):
        from sage_mas.executor_dispatch import ExecutorDispatcher

        class KeepExecutorBackend:
            called = False

            def complete(self, system_prompt, user_prompt):
                self.called = True
                return LLMResult(
                    "<assign>Executor</assign>",
                    prompt_tokens=1,
                    completion_tokens=1,
                )

        skill = Skill(
            skill_name="Heat capability",
            description="Heat a held target",
            precondition="Target requires heating",
            action_protocol=["heat target"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            metadata={
                "source_signal": "missing_heat_operation",
                "primary_task_family": "pick_heat_then_place_in_recep",
            },
        )
        backend = KeepExecutorBackend()
        assignment = ExecutorDispatcher(backend=backend).assign(
            task="put a clean mug in cabinet",
            task_family="pick_clean_then_place_in_recep",
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Act"],
                    tool_permissions=["alfworld_action"],
                ),
                AgentSpec(
                    name="HeatOperator",
                    role="HeatOperator",
                    responsibilities=["Heat"],
                    assigned_skills=[skill.skill_name],
                    tool_permissions=["alfworld_action"],
                    shadow_evaluation_record={
                        "acting_status": "accepted",
                        # This used to bypass Skill applicability.
                        "task_families": [
                            "pick_clean_then_place_in_recep",
                        ],
                    },
                ),
            ],
            skills=[skill],
            gamefile=(
                "/fake/pick_clean_then_place_in_recep-0/game.tw-pddl"
            ),
        )

        self.assertEqual(assignment.primary_agent, "Executor")
        self.assertEqual(assignment.dispatch_layer, "eligibility_empty")
        self.assertEqual(assignment.eligible_agents, [])
        self.assertFalse(backend.called)

    def test_shadow_evaluator_accepts_material_gain(self):
        evaluator = ShadowEvaluator(cost_weight=0.0, significance_threshold=0.05)
        decision = evaluator.decide(
            ShadowMetrics(success_rate=0.4, token_cost=100),
            ShadowMetrics(success_rate=0.5, token_cost=120),
        )
        self.assertTrue(decision.accepted)
        self.assertGreater(decision.relative_gain, 0.05)

    def test_shadow_evaluator_accepts_utility_tie(self):
        evaluator = ShadowEvaluator(cost_weight=0.0, significance_threshold=0.0)
        decision = evaluator.decide(
            ShadowMetrics(success_rate=0.0, token_cost=100),
            ShadowMetrics(success_rate=0.0, token_cost=120),
        )
        self.assertTrue(decision.accepted)
        self.assertIn("meets or exceeds", decision.reason)

        tied = evaluator.decide(
            ShadowMetrics(success_rate=0.5, token_cost=100),
            ShadowMetrics(success_rate=0.5, token_cost=100),
        )
        self.assertTrue(tied.accepted)

        worse = evaluator.decide(
            ShadowMetrics(success_rate=0.5, token_cost=100),
            ShadowMetrics(success_rate=0.0, token_cost=100),
        )
        self.assertFalse(worse.accepted)
        self.assertIn("does not meet", worse.reason)

    def test_shadow_requires_confident_paired_gain_when_enabled(self):
        evaluator = ShadowEvaluator(
            cost_weight=0.0,
            significance_threshold=0.0,
            bootstrap_samples=200,
            require_confident_gain=True,
            min_paired_tasks=3,
        )
        decision = evaluator.decide(
            ShadowMetrics(success_rate=0.4, token_cost=100),
            ShadowMetrics(success_rate=0.5, token_cost=100),
            paired_deltas=[0.2, -0.1, 0.2],
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.paired_task_count, 3)
        self.assertLessEqual(decision.paired_delta_ci_low, 0.0)

    def test_alfworld_runtime_keeps_executor_prompt_gigpo_pure(self):
        class FakeBackend:
            def __init__(self):
                self.calls = []

            def complete(self, system_prompt, user_prompt):
                self.calls.append((system_prompt, user_prompt))
                # AlfWorld-style executor: empty system, template in user.
                if system_prompt == "":
                    return LLMResult(
                        "<think>act</think><action>look</action>",
                        prompt_tokens=5,
                        completion_tokens=2,
                    )
                return LLMResult(
                    "Avoid repeating the previous action.",
                    prompt_tokens=3,
                    completion_tokens=2,
                )

        skill = Skill(
            skill_name="Prevent repetition",
            description="Avoid loops",
            precondition="The last action did not change state.",
            action_protocol=["Choose another action"],
            applicable_atomic_ops=[AtomicOp.ACT],
            suggested_role="Monitor",
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Issue actions"],
        )
        monitor = AgentSpec(
            name="MonitorAgent",
            role="Monitor",
            responsibilities=["Detect loops"],
            assigned_skills=[skill.skill_name],
        )
        backend = FakeBackend()
        result = MASRuntime(
            agents=[executor, monitor],
            skills=[skill],
            backend=backend,
        ).act("Current observation")

        self.assertEqual(result.action, "<think>act</think><action>look</action>")
        self.assertEqual(result.token_cost, 7)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][0], "")
        self.assertEqual(backend.calls[0][1], "Current observation")
        self.assertNotIn(skill.skill_name, backend.calls[0][1])

    def test_runtime_stage_specialist_emits_environment_action(self):
        class FakeBackend:
            def __init__(self):
                self.calls = []

            def complete(self, system_prompt, user_prompt, max_completion_tokens=None):
                self.calls.append((system_prompt, user_prompt))
                return LLMResult(
                    "<think>clean now</think><action>clean mug 1 with sinkbasin 1</action>",
                    prompt_tokens=2,
                    completion_tokens=2,
                )

        skill = Skill(
            skill_name="Clean the target object before placement",
            description="Clean held object",
            precondition="Object held",
            action_protocol=["clean object with sinkbasin"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Fallback actor"],
            tool_permissions=["alfworld_action"],
        )
        cleaner = AgentSpec(
            name="CleanOperator",
            role="CleanOperator",
            responsibilities=["Execute clean stage"],
            assigned_skills=[skill.skill_name],
            tool_permissions=["clean", "alfworld_action"],
            role_specification="You are CleanOperator. Owns clean-then-place execution.",
            output_protocol=(
                "Emit <think>...</think><action>...</action>."
            ),
            shadow_evaluation_record={
                "acting_status": "accepted",
                "trial_games_remaining": 0,
            },
        )
        backend = FakeBackend()
        result = MASRuntime(
            agents=[executor, cleaner],
            skills=[skill],
            backend=backend,
            enable_action_guards=False,
            compact_alfworld_prompts=True,
        ).act(
            "obs",
            active_assigned_skill_names={skill.skill_name},
            preferred_actor_name="CleanOperator",
            task_family="pick_clean_then_place_in_recep",
            task="put a clean mug in cabinet",
        )

        self.assertIn("clean mug 1 with sinkbasin 1", result.action)
        self.assertEqual(result.messages[-1].agent_name, "CleanOperator")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][0], "")
        # Specialist receives only the learned verified Skill contract.
        self.assertIn("obs", backend.calls[0][1])
        self.assertNotIn("Hard rules", backend.calls[0][1])
        self.assertIn(skill.skill_name, backend.calls[0][1])
        self.assertIn(skill.precondition, backend.calls[0][1])
        self.assertIn(skill.action_protocol[0], backend.calls[0][1])

    def test_router_blocks_probation_without_trial_budget(self):
        from sage_mas.executor_dispatch import ExecutorDispatcher

        class PickSpecialistBackend:
            def complete(self, system_prompt, user_prompt):
                return LLMResult(
                    "<assign>LookAtObjInLightSpecialist</assign>",
                    prompt_tokens=1,
                    completion_tokens=1,
                )

        assignment = ExecutorDispatcher(backend=PickSpecialistBackend()).assign(
            task="examine the book with the desklamp",
            task_family="look_at_obj_in_light",
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["fallback"],
                    tool_permissions=["alfworld_action"],
                ),
                AgentSpec(
                    name="LookAtObjInLightSpecialist",
                    role="LookAtObjInLightSpecialist",
                    responsibilities=["light"],
                    assigned_skills=["Reuse look at light"],
                    tool_permissions=["alfworld_action"],
                    shadow_evaluation_record={
                        "acting_status": "probation",
                        "trial_games_remaining": 0,
                        "task_families": ["look_at_obj_in_light"],
                    },
                ),
            ],
            skills=[
                Skill(
                    skill_name="Reuse look at light",
                    description="light",
                    precondition="held",
                    action_protocol=["take", "use"],
                    applicable_atomic_ops=[AtomicOp.ACT],
                    metadata={
                        "primary_task_family": "look_at_obj_in_light",
                        "source_signal": "missing_light_operation",
                    },
                )
            ],
            gamefile="/fake/look_at_obj_in_light-0/game.tw-pddl",
        )
        self.assertEqual(assignment.primary_agent, "Executor")
        self.assertEqual(assignment.dispatch_layer, "eligibility_empty")

    def test_onboarding_demotes_worse_specialist(self):
        from sage_mas.onboarding import refresh_acting_statuses

        specialist = AgentSpec(
            name="LookAtObjInLightSpecialist",
            role="LookAtObjInLightSpecialist",
            responsibilities=["light"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "probation",
                "trial_games_remaining": 3,
                "verified_skill_marginal_utility": 0.2,
            },
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["fallback"],
            tool_permissions=["alfworld_action"],
        )
        segment = [
            {
                "assigned_primary_agent": "LookAtObjInLightSpecialist",
                "dispatch_layer": "eligibility_single",
                "actions_by_agent": {
                    "LookAtObjInLightSpecialist": 2,
                },
                "task_family": "look_at_obj_in_light",
                "won": False,
            },
            {
                "assigned_primary_agent": "LookAtObjInLightSpecialist",
                "dispatch_layer": "eligibility_single",
                "actions_by_agent": {
                    "LookAtObjInLightSpecialist": 2,
                },
                "task_family": "look_at_obj_in_light",
                "won": False,
            },
        ]
        baseline = [
            {
                "assigned_primary_agent": "Executor",
                "task_family": "look_at_obj_in_light",
                "won": True,
            },
            {
                "assigned_primary_agent": "Executor",
                "task_family": "look_at_obj_in_light",
                "won": True,
            },
        ]
        updates = refresh_acting_statuses(
            [executor, specialist],
            segment,
            baseline_trials=baseline,
            min_games=2,
            epsilon=0.05,
        )
        self.assertTrue(updates["changed"])
        self.assertEqual(
            specialist.shadow_evaluation_record["acting_status"],
            "demoted",
        )

    def test_onboarding_prefers_executor_baseline_without_specialist_skill(self):
        from sage_mas.onboarding import refresh_acting_statuses

        skill_name = "Merged inspect with light protocol"
        specialist = AgentSpec(
            name="LookAtObjInLightSpecialist",
            role="LookAtObjInLightSpecialist",
            responsibilities=["light"],
            assigned_skills=[skill_name],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "probation",
                "trial_games_remaining": 2,
                "verified_skill_marginal_utility": 0.2,
                "task_families": ["look_at_obj_in_light"],
                "capability_contract": {
                    "task_families": ["look_at_obj_in_light"],
                    "skill_names": [skill_name],
                },
            },
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["fallback"],
            tool_permissions=["alfworld_action"],
        )
        segment = [
            {
                "assigned_primary_agent": specialist.name,
                "dispatch_layer": "eligibility_single",
                "actions_by_agent": {specialist.name: 2},
                "task_family": "look_at_obj_in_light",
                "won": True,
            },
            {
                "assigned_primary_agent": specialist.name,
                "dispatch_layer": "eligibility_single",
                "actions_by_agent": {specialist.name: 2},
                "task_family": "look_at_obj_in_light",
                "won": True,
            },
        ]
        baseline = [
            {
                "assigned_primary_agent": "Executor",
                "task_family": "look_at_obj_in_light",
                "activated_skill_names": [skill_name],
                "won": True,
            },
            {
                "assigned_primary_agent": "Executor",
                "task_family": "look_at_obj_in_light",
                "activated_skill_names": [skill_name],
                "won": True,
            },
            {
                "assigned_primary_agent": "Executor",
                "task_family": "look_at_obj_in_light",
                "activated_skill_names": [],
                "won": False,
            },
            {
                "assigned_primary_agent": "Executor",
                "task_family": "look_at_obj_in_light",
                "activated_skill_names": [],
                "won": False,
            },
        ]
        updates = refresh_acting_statuses(
            [executor, specialist],
            segment,
            baseline_trials=baseline,
            min_games=2,
            min_wins=2,
            epsilon=0.0,
        )
        self.assertTrue(updates["changed"])
        self.assertEqual(
            specialist.shadow_evaluation_record["acting_status"],
            "accepted",
        )
        self.assertEqual(
            updates["decisions"][0].get("baseline_source"),
            "executor_without_specialist_skill",
        )

    def test_revive_cross_model_demotions_restores_probation(self):
        from sage_mas.onboarding import revive_cross_model_demotions

        specialist = AgentSpec(
            name="TrackMultipleObjectsSpecialist",
            role="TrackMultipleObjectsSpecialist",
            responsibilities=["pick_two"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "demoted",
                "dispatch_only": True,
                "trial_games_remaining": 0,
                "rejected_windows": 1,
                "dispatched_games": 10,
                "wins_as_primary": 7,
                "last_onboarding_decision": "demoted",
            },
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["fallback"],
            tool_permissions=["alfworld_action"],
        )
        result = revive_cross_model_demotions(
            [executor, specialist],
            probation_games=3,
        )
        self.assertTrue(result["changed"])
        self.assertEqual(result["revived_agents"], [specialist.name])
        record = specialist.shadow_evaluation_record
        self.assertEqual(record["acting_status"], "probation")
        self.assertTrue(record["dispatch_only"])
        self.assertEqual(record["trial_games_remaining"], 3)
        self.assertEqual(record["rejected_windows"], 0)
        self.assertEqual(
            record["last_onboarding_decision"],
            "revived_cross_model_demotion",
        )

    def test_onboarding_defers_when_same_model_baseline_empty(self):
        """Empty baseline must not demote (used when Pro train baseline is withheld)."""
        from sage_mas.onboarding import refresh_acting_statuses

        specialist = AgentSpec(
            name="TrackMultipleObjectsSpecialist",
            role="TrackMultipleObjectsSpecialist",
            responsibilities=["pick_two"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "probation",
                "dispatch_only": True,
                "trial_games_remaining": 0,
                "verified_skill_marginal_utility": 0.3,
                "task_families": ["pick_two_obj_and_place"],
            },
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["fallback"],
            tool_permissions=["alfworld_action"],
        )
        segment = [
            {
                "assigned_primary_agent": specialist.name,
                "dispatch_layer": "llm",
                "actions_by_agent": {specialist.name: 5},
                "task_family": "pick_two_obj_and_place",
                "won": True,
            },
            {
                "assigned_primary_agent": specialist.name,
                "dispatch_layer": "llm",
                "actions_by_agent": {specialist.name: 5},
                "task_family": "pick_two_obj_and_place",
                "won": True,
            },
        ]
        updates = refresh_acting_statuses(
            [executor, specialist],
            segment,
            baseline_trials=[],  # same_model unavailable → defer
            min_games=2,
            min_wins=1,
            epsilon=0.0,
        )
        self.assertEqual(
            specialist.shadow_evaluation_record["acting_status"],
            "probation",
        )
        self.assertGreaterEqual(
            int(specialist.shadow_evaluation_record["trial_games_remaining"]),
            1,
        )
        self.assertFalse(
            any(
                d.get("new_status") == "demoted"
                for d in updates.get("decisions") or []
            )
        )

    def test_mid_model_gates_forced_when_teacher_differs(self):
        """Teacher!=gate forces same-model onboarding; skills can auto-verify."""
        from sage_mas.online_evolution import OnlineEvolutionConfig
        from sage_mas.skill_credit import SkillCreditPolicy

        class _Backend:
            def __init__(self, model: str):
                self.model = model

        # Build a bare instance without full __init__.
        evo = object.__new__(
            __import__(
                "sage_mas.online_evolution", fromlist=["OnlineAlfWorldEvolution"]
            ).OnlineAlfWorldEvolution
        )
        evo.backend = _Backend("gemini-2.5-pro")
        evo.specialist_backend = _Backend("gemini-2.5-flash")
        evo.acceptor_backend = _Backend("gemini-2.5-flash")
        evo.evaluator = type("E", (), {"backend": evo.backend})()
        evo.acceptor_evaluator = type(
            "E", (), {"backend": evo.acceptor_backend}
        )()
        evo.config = OnlineEvolutionConfig(
            onboarding_baseline="train",
            paired_mu_probe=False,
            revive_cross_model_demotions=False,
            auto_verify_form_ok_skills=False,
            require_positive_mu_for_injection=False,
        )
        evo.skill_credit_policy = SkillCreditPolicy(
            require_positive_mu_for_verify=False,
        )
        meta = evo._enforce_mid_model_promotion_gates()
        self.assertTrue(meta["models_differ"])
        self.assertEqual(evo.config.onboarding_baseline, "same_model")
        self.assertFalse(evo.config.paired_mu_probe)
        self.assertTrue(evo.config.revive_cross_model_demotions)
        self.assertTrue(evo.config.auto_verify_form_ok_skills)
        self.assertFalse(evo.skill_credit_policy.require_positive_mu_for_verify)
        self.assertIn("onboarding_baseline=same_model", meta["forced"])
        self.assertIn("auto_verify_form_ok_skills=true", meta["forced"])
        self.assertNotIn("paired_mu_probe=true", meta["forced"])

    def test_onboarding_never_accepts_specialist_without_a_task_win(self):
        from sage_mas.onboarding import refresh_acting_statuses

        specialist = AgentSpec(
            name="LearnedSpecialist",
            role="LearnedSpecialist",
            responsibilities=["Execute learned Skill"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "probation",
                "dispatch_only": True,
                "trial_games_remaining": 1,
            },
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
        )
        failed_dispatch = {
            "assigned_primary_agent": specialist.name,
            "dispatch_layer": "eligibility_single",
            "actions_by_agent": {specialist.name: 2},
            "task_family": "other",
            "won": False,
        }
        failed_baseline = {
            "assigned_primary_agent": executor.name,
            "task_family": "other",
            "won": False,
        }

        updates = refresh_acting_statuses(
            [executor, specialist],
            [failed_dispatch],
            baseline_trials=[failed_baseline],
            min_games=1,
        )

        record = specialist.shadow_evaluation_record
        self.assertEqual(record["acting_status"], "demoted")
        self.assertEqual(record["wins_as_primary"], 0)
        self.assertEqual(
            updates["decisions"][0]["completed_applicable_task"],
            False,
        )

    def test_onboarding_promotes_only_real_dispatched_work(self):
        from sage_mas.onboarding import refresh_acting_statuses

        specialist = AgentSpec(
            name="LearnedSpecialist",
            role="LearnedSpecialist",
            responsibilities=["Execute learned Skill"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "probation",
                "dispatch_only": True,
                "trial_games_remaining": 3,
                "verified_skill_marginal_utility": 0.2,
                "task_families": ["pick_clean_then_place_in_recep"],
            },
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
        )
        real_trials = [
            {
                "assigned_primary_agent": specialist.name,
                "dispatch_layer": "eligibility_single",
                "actions_by_agent": {specialist.name: 2},
                "task_family": "pick_clean_then_place_in_recep",
                "won": won,
            }
            for won in (True, True, False)
        ]
        fake_primary_without_actions = {
            "assigned_primary_agent": specialist.name,
            "dispatch_layer": "eligibility_single",
            "actions_by_agent": {},
            "task_family": "pick_clean_then_place_in_recep",
            "won": True,
        }
        baseline = [
            {
                "assigned_primary_agent": "Executor",
                "task_family": "pick_clean_then_place_in_recep",
                "won": won,
            }
            for won in (True, False, False)
        ]

        updates = refresh_acting_statuses(
            [executor, specialist],
            real_trials + [fake_primary_without_actions],
            baseline_trials=baseline,
            min_games=3,
            min_wins=2,
            epsilon=0.0,
        )

        record = specialist.shadow_evaluation_record
        self.assertEqual(record["acting_status"], "accepted")
        self.assertEqual(record["dispatched_games"], 3)
        self.assertEqual(record["actions_executed"], 6)
        self.assertEqual(record["wins_as_primary"], 2)
        self.assertEqual(
            record["task_performance"],
            {
                "dispatches": 3,
                "wins": 2,
                "by_family": {
                    "pick_clean_then_place_in_recep": {
                        "dispatches": 3,
                        "wins": 2,
                    }
                },
            },
        )
        self.assertEqual(updates["removed_agents"], [])

    def test_onboarding_requires_repeated_in_contract_wins(self):
        from sage_mas.onboarding import refresh_acting_statuses

        specialist = AgentSpec(
            name="CoolSpecialist",
            role="CoolSpecialist",
            responsibilities=["Execute learned cool capability"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "probation",
                "dispatch_only": True,
                "trial_games_remaining": 3,
                "task_families": ["pick_cool_then_place_in_recep"],
                "capability_contract": {
                    "task_families": ["pick_cool_then_place_in_recep"],
                },
            },
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
            shadow_evaluation_record={
                "task_performance": {
                    "by_family": {
                        "pick_cool_then_place_in_recep": {
                            "dispatches": 6,
                            "wins": 2,
                        }
                    }
                }
            },
        )
        one_lucky_win = {
            "assigned_primary_agent": specialist.name,
            "dispatch_layer": "llm",
            "actions_by_agent": {specialist.name: 8},
            "task_family": "pick_cool_then_place_in_recep",
            "won": True,
        }

        updates = refresh_acting_statuses(
            [executor, specialist],
            [one_lucky_win],
            min_games=3,
            min_wins=2,
            epsilon=0.0,
        )

        record = specialist.shadow_evaluation_record
        self.assertEqual(record["acting_status"], "probation")
        self.assertEqual(record["applicable_wins_as_primary"], 1)
        self.assertEqual(record["applicable_dispatched_games"], 1)
        self.assertEqual(updates["decisions"], [])

    def test_onboarding_ignores_out_of_contract_win(self):
        from sage_mas.onboarding import refresh_acting_statuses

        specialist = AgentSpec(
            name="CoolSpecialist",
            role="CoolSpecialist",
            responsibilities=["Execute learned cool capability"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "probation",
                "dispatch_only": True,
                "trial_games_remaining": 3,
                "task_families": ["pick_cool_then_place_in_recep"],
                "capability_contract": {
                    "task_families": ["pick_cool_then_place_in_recep"],
                },
            },
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
        )
        wrong_scope_win = {
            "assigned_primary_agent": specialist.name,
            "dispatch_layer": "llm",
            "actions_by_agent": {specialist.name: 4},
            "task_family": "pick_clean_then_place_in_recep",
            "won": True,
        }

        updates = refresh_acting_statuses(
            [executor, specialist],
            [wrong_scope_win],
            min_games=3,
            min_wins=2,
        )

        record = specialist.shadow_evaluation_record
        self.assertEqual(record["acting_status"], "probation")
        self.assertEqual(record["trial_games_remaining"], 3)
        self.assertEqual(record.get("wins_as_primary", 0), 0)
        self.assertEqual(record["out_of_scope_dispatches"], 1)
        self.assertEqual(updates["decisions"], [])

    def test_onboarding_revalidates_accepted_agent_on_contract_scope(self):
        from sage_mas.onboarding import refresh_acting_statuses

        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
            shadow_evaluation_record={
                "task_performance": {
                    "dispatches": 7,
                    "wins": 2,
                    "by_family": {
                        "pick_cool_then_place_in_recep": {
                            "dispatches": 7,
                            "wins": 2,
                        },
                    },
                },
            },
        )
        specialist = AgentSpec(
            name="CoolSpecialist",
            role="CoolSpecialist",
            responsibilities=["Cool"],
            shadow_evaluation_record={
                "acting_status": "accepted",
                "task_families": ["pick_cool_then_place_in_recep"],
                "task_performance": {
                    "dispatches": 14,
                    "wins": 2,
                    "by_family": {
                        "pick_cool_then_place_in_recep": {
                            "dispatches": 7,
                            "wins": 1,
                        },
                        "pick_clean_then_place_in_recep": {
                            "dispatches": 7,
                            "wins": 1,
                        },
                    },
                },
            },
        )

        updates = refresh_acting_statuses(
            [executor, specialist],
            [],
            min_games=3,
            min_wins=2,
            epsilon=0.0,
        )

        self.assertEqual(
            specialist.shadow_evaluation_record["acting_status"],
            "demoted",
        )
        self.assertEqual(
            specialist.shadow_evaluation_record["last_onboarding_decision"],
            "demoted_scope_revalidation",
        )
        self.assertEqual(
            updates["decisions"][0]["reason"],
            "contract_scope_revalidation",
        )

    def test_onboarding_tracks_executor_full_task_history(self):
        from sage_mas.onboarding import refresh_acting_statuses

        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
        )
        trials = [
            {
                "assigned_primary_agent": "Executor",
                "dispatch_layer": "eligibility_empty",
                "actions_by_agent": {"Executor": 3},
                "task_family": "pick_and_place",
                "won": won,
            }
            for won in (True, False)
        ]

        updates = refresh_acting_statuses([executor], trials)

        self.assertTrue(updates["changed"])
        self.assertEqual(
            executor.shadow_evaluation_record["task_performance"],
            {
                "dispatches": 2,
                "wins": 1,
                "by_family": {
                    "pick_and_place": {
                        "dispatches": 2,
                        "wins": 1,
                    }
                },
            },
        )

    def test_onboarding_removes_persistently_demoted_agent(self):
        from sage_mas.onboarding import refresh_acting_statuses

        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
        )
        demoted = AgentSpec(
            name="UnusedSpecialist",
            role="Specialist",
            responsibilities=["Unused"],
            shadow_evaluation_record={
                "acting_status": "demoted",
                "rejected_windows": 1,
            },
        )
        agents = [executor, demoted]

        updates = refresh_acting_statuses(
            agents,
            [],
            remove_after_rejected_windows=2,
        )

        self.assertEqual([agent.name for agent in agents], ["Executor"])
        self.assertEqual(updates["removed_agents"], ["UnusedSpecialist"])

    def test_executor_dispatch_skips_demoted_specialist(self):
        from sage_mas.executor_dispatch import ExecutorDispatcher

        class PickSpecialistBackend:
            def complete(self, system_prompt, user_prompt):
                return LLMResult(
                    "<assign>LookAtObjInLightSpecialist</assign>",
                    prompt_tokens=1,
                    completion_tokens=1,
                )

        assignment = ExecutorDispatcher(backend=PickSpecialistBackend()).assign(
            task="examine the book with the desklamp",
            task_family="look_at_obj_in_light",
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["fallback"],
                    tool_permissions=["alfworld_action"],
                ),
                AgentSpec(
                    name="LookAtObjInLightSpecialist",
                    role="LookAtObjInLightSpecialist",
                    responsibilities=["light"],
                    assigned_skills=["Reuse look at light"],
                    tool_permissions=["alfworld_action"],
                    shadow_evaluation_record={
                        "acting_status": "demoted",
                        "task_families": ["look_at_obj_in_light"],
                    },
                ),
            ],
            skills=[
                Skill(
                    skill_name="Reuse look at light",
                    description="light",
                    precondition="held",
                    action_protocol=["take", "use"],
                    applicable_atomic_ops=[AtomicOp.ACT],
                    metadata={
                        "primary_task_family": "look_at_obj_in_light",
                        "source_signal": "missing_light_operation",
                    },
                )
            ],
            gamefile="/fake/look_at_obj_in_light-0/game.tw-pddl",
        )
        self.assertEqual(assignment.primary_agent, "Executor")
        self.assertEqual(assignment.dispatch_layer, "eligibility_empty")

    def test_executor_prompt_ignores_injected_skills_on_gigpo_path(self):
        class FakeBackend:
            def __init__(self):
                self.user_prompts = []

            def complete(self, system_prompt, user_prompt):
                self.user_prompts.append(user_prompt)
                return LLMResult(
                    "<action>look</action>",
                    prompt_tokens=1,
                    completion_tokens=1,
                )

        skill = Skill(
            skill_name="Clean target",
            description="Clean target",
            precondition="Target is held",
            action_protocol=["Clean"],
            applicable_atomic_ops=[AtomicOp.ACT],
        )
        backend = FakeBackend()
        runtime = MASRuntime(
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Issue actions"],
                )
            ],
            skills=[],
            backend=backend,
            injected_skills=[skill],
        )

        runtime.act("Before pickup", injected_skills=[])
        runtime.act("After pickup", injected_skills=[skill])

        self.assertNotIn(skill.skill_name, backend.user_prompts[0])
        self.assertNotIn(skill.skill_name, backend.user_prompts[1])

    def test_counterfactual_executor_prompt_can_receive_experimental_skill(self):
        class FakeBackend:
            def __init__(self):
                self.user_prompt = ""

            def complete(self, system_prompt, user_prompt):
                self.user_prompt = user_prompt
                return LLMResult("<action>look</action>")

        skill = Skill(
            skill_name="Experimental clean",
            description="Hypothesis",
            precondition="Task eligible",
            action_protocol=["clean <object> with <tool>"],
            applicable_atomic_ops=[AtomicOp.ACT],
        )
        backend = FakeBackend()
        runtime = MASRuntime(
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Act"],
                )
            ],
            skills=[],
            backend=backend,
            allow_executor_skill_injection=True,
        )
        runtime.act("Observation", injected_skills=[skill])
        self.assertIn(skill.skill_name, backend.user_prompt)
        self.assertIn("executor fallback", backend.user_prompt.lower())
        self.assertIn("learned skill contract", backend.user_prompt.lower())

    def test_capability_transition_requires_valid_action_and_environment_feedback(self):
        trial = EvaluationTrial(
            task_id="clean-task",
            task="clean a mug",
            task_family="pick_clean_then_place_in_recep",
            condition="discovery",
            reward=0.0,
            cost=1.0,
            won=False,
            num_steps=2,
            steps=[
                {
                    "action": "<action>clean mug 1 with sinkbasin 1</action>",
                    "observation": "Nothing happens.",
                    "is_action_valid": False,
                },
                {
                    "action": "<action>clean mug 1 with sinkbasin 1</action>",
                    "observation": "You clean the mug 1.",
                    "is_action_valid": True,
                },
            ],
        )
        transitions = CapabilityTransitionDetector("clean").detect(
            trial,
            attempt=1,
        )
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].step, 2)

    def test_generic_capability_transition_uses_progress_not_navigation(self):
        trial = EvaluationTrial(
            task_id="task",
            task="complete the task",
            task_family="family",
            condition="discovery",
            reward=0.0,
            cost=1.0,
            won=False,
            num_steps=3,
            steps=[
                {
                    "action": "<action>go to fridge 1</action>",
                    "observation": "You arrive at fridge 1.",
                    "is_action_valid": True,
                },
                {
                    "action": "<action>cool apple 1 with fridge 1</action>",
                    "observation": "You cool the apple 1.",
                    "is_action_valid": True,
                },
                {
                    "action": "<action>move apple 1 to table 1</action>",
                    "observation": "The table now contains apple 1.",
                    "is_action_valid": True,
                    "goal_progress_delta": 0.5,
                },
            ],
        )
        transitions = CapabilityTransitionDetector().detect(trial, attempt=1)
        self.assertEqual([item.step for item in transitions], [2, 3])

    def test_online_discovery_selects_highest_support_uncovered_scope(self):
        from sage_mas.online_discovery import (
            OnlineDiscoveryConfig,
            OnlineDiscoveryCoordinator,
        )

        failures = [
            {
                "task_id": f"/two-{index}",
                "task": "place two objects",
                "task_family": "pick_two_obj_and_place",
                "won": False,
                "steps": [],
            }
            for index in range(5)
        ] + [
            {
                "task_id": f"/cool-{index}",
                "task": "place a cooled object",
                "task_family": "pick_cool_then_place_in_recep",
                "won": False,
                "steps": [],
            }
            for index in range(4)
        ]
        coordinator = OnlineDiscoveryCoordinator(
            backend=SimpleNamespace(),
            evaluator=SimpleNamespace(),
            config=OnlineDiscoveryConfig(
                enabled=True,
                auto_capability_queue=True,
                min_failures=3,
                min_confirmed_tasks=3,
            ),
        )
        outcome = coordinator.run(
            collection_trials=failures,
            agents=[],
            known_skills=[],
            discovery_pool=[],
        )
        self.assertEqual(outcome.status, "insufficient_discovery_pool")
        self.assertEqual(
            outcome.summary["task_family"],
            "pick_two_obj_and_place",
        )

    def test_generic_discovery_grounds_the_hypothesis_final_operation(self):
        class FakeEvaluator:
            def __init__(self):
                self.config = SimpleNamespace(
                    allow_executor_skill_injection=False,
                    gate_injected_skills=True,
                )

            def evaluate(
                self,
                agents,
                skills,
                gamefiles,
                condition,
                injected_skills=None,
            ):
                return [
                    EvaluationTrial(
                        task_id=gamefile,
                        task="place a cooled apple",
                        task_family="pick_cool_then_place_in_recep",
                        condition=condition,
                        reward=0.0,
                        cost=1.0,
                        won=False,
                        num_steps=1,
                        steps=[
                            {
                                "action": "cool apple 1 with fridge 1",
                                "observation_before": "You hold apple 1.",
                                "observation": "You cool the apple 1.",
                                "is_action_valid": True,
                            }
                        ],
                    )
                    for gamefile in gamefiles
                ]

        hypothesis = Skill(
            skill_name="Experimental cooling",
            description="d",
            precondition="p",
            action_protocol=[
                "go to fridge 1",
                "place apple 1 in fridge 1",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            expected_effect="The held object becomes cool.",
            capability_key="experience.pick_cool_then_place_in_recep",
        )
        outcome = DiscoveryForkService(
            FakeEvaluator(),
            capability_key=hypothesis.capability_key,
            task_family="pick_cool_then_place_in_recep",
            attempts_per_task=1,
        ).run(
            experimental_skill=hypothesis,
            agents=[],
            baseline_skills=[],
            gamefiles=["/cool-1", "/cool-2", "/cool-3"],
        )
        self.assertEqual(outcome.confirmed_task_count, 3)
        self.assertIsNotNone(outcome.grounded_skill)
        self.assertEqual(
            outcome.grounded_skill.metadata["capability_operation"],
            "cool",
        )
        self.assertEqual(
            outcome.grounded_skill.metadata["anchor_state"],
            "You hold apple 1.",
        )

    def test_discovery_ledger_accumulates_independent_rounds(self):
        from sage_mas.online_discovery import (
            OnlineDiscoveryConfig,
            OnlineDiscoveryCoordinator,
        )

        def outcome(task_id):
            skill = Skill(
                skill_name="Grounded placement",
                description="d",
                precondition="p",
                action_protocol=["move object to receptacle"],
                applicable_atomic_ops=[AtomicOp.ACT],
                expected_effect="Object is moved to the receptacle.",
                capability_key="experience.pick_two_obj_and_place",
                applicable_task_families=["pick_two_obj_and_place"],
                metadata={
                    "candidate_stage": "environment_confirmed_discovery",
                    "anchor_state": "The target object is held.",
                },
            )
            transition = ConfirmedCapabilityTransition(
                task_id=task_id,
                attempt=1,
                step=1,
                operation="move",
                action="move object 1 to drawer 1",
                observation="You move the object 1 to the drawer 1.",
            )
            trial = EvaluationTrial(
                task_id=task_id,
                task="put two objects in drawer",
                task_family="pick_two_obj_and_place",
                condition="discovery",
                reward=0.0,
                cost=1.0,
                won=False,
                num_steps=1,
                steps=[
                    {
                        "action": transition.action,
                        "observation": transition.observation,
                        "is_action_valid": True,
                    }
                ],
            )
            return DiscoveryForkOutcome(
                experimental_skill=skill,
                grounded_skill=skill,
                trials=[trial],
                confirmed_transitions=[transition],
                confirmed_task_count=1,
                attempts_per_task=1,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            ledger_path = Path(temp_dir) / "discovery_ledger.json"
            coordinator = OnlineDiscoveryCoordinator(
                backend=SimpleNamespace(),
                evaluator=SimpleNamespace(),
                config=OnlineDiscoveryConfig(enabled=True),
                ledger_path=ledger_path,
            )
            entry = {
                "confirmed_transitions": [],
                "evidence_trajectories": [],
                "grounded_skill": None,
            }
            _, first_count, _ = coordinator._accumulate_discovery(
                entry,
                outcome("task-1"),
            )
            skill, second_count, evidence = (
                coordinator._accumulate_discovery(
                    entry,
                    outcome("task-2"),
                )
            )
            skill, third_count, evidence = (
                coordinator._accumulate_discovery(
                    entry,
                    outcome("task-3"),
                )
            )
            ledger = {
                "version": 1,
                "capabilities": {"experience.test": entry},
            }
            coordinator._save_ledger(ledger)
            reloaded = coordinator._load_ledger()

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 2)
        self.assertEqual(third_count, 3)
        self.assertEqual(skill.support_count, 3)
        self.assertEqual(len(evidence), 3)
        self.assertEqual(
            len(
                reloaded["capabilities"]["experience.test"][
                    "confirmed_transitions"
                ]
            ),
            3,
        )
        self.assertEqual(
            OnlineDiscoveryCoordinator._select_rotated_games(
                ["task-a", "task-b", "task-c"],
                {"task-a": 2, "task-b": 1},
                2,
            ),
            ["task-c", "task-b"],
        )

    def test_runtime_gates_assigned_advisor_and_skill(self):
        class FakeBackend:
            def __init__(self):
                self.system_prompts = []
                self.user_prompts = []

            def complete(self, system_prompt, user_prompt):
                self.system_prompts.append(system_prompt)
                self.user_prompts.append(user_prompt)
                if "Verifier" in system_prompt:
                    return LLMResult("Advice")
                return LLMResult("<action>look</action>")

        skill = Skill(
            skill_name="Verify target",
            description="Verify target",
            precondition="Target is held",
            action_protocol=["Verify"],
            applicable_atomic_ops=[AtomicOp.VERIFY],
        )
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
        )
        advisor = AgentSpec(
            name="VerifierAgent",
            role="Verifier",
            responsibilities=["Verify"],
            assigned_skills=[skill.skill_name],
        )
        backend = FakeBackend()
        runtime = MASRuntime(
            agents=[executor, advisor],
            skills=[skill],
            backend=backend,
            prompt_style="mas",
        )

        runtime.act(
            "Before precondition",
            active_assigned_skill_names=set(),
        )
        self.assertEqual(len(backend.system_prompts), 1)
        runtime.act(
            "After precondition",
            active_assigned_skill_names={skill.skill_name},
        )
        self.assertEqual(len(backend.system_prompts), 3)
        self.assertIn(skill.skill_name, backend.system_prompts[1])
        self.assertIn("Executor", backend.system_prompts[2])
        self.assertIn("VerifierAgent advice", backend.user_prompts[2])

    def test_runtime_selects_matching_advisor_with_max_advisors(self):
        class FakeBackend:
            def __init__(self):
                self.system_prompts = []

            def complete(self, system_prompt, user_prompt):
                self.system_prompts.append(system_prompt)
                if "Advisor" in system_prompt:
                    return LLMResult("Advice")
                return LLMResult("<action>look</action>")

        clean_skill = Skill(
            skill_name="pick_clean_then_place_in_recep",
            description="Clean",
            precondition="Held dirty object",
            action_protocol=["clean"],
            applicable_atomic_ops=[AtomicOp.ACT],
        )
        heat_skill = Skill(
            skill_name="heat_target_before_placement",
            description="Heat",
            precondition="Held target needs heating",
            action_protocol=["heat"],
            applicable_atomic_ops=[AtomicOp.ACT],
        )
        backend = FakeBackend()
        MASRuntime(
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Act"],
                ),
                AgentSpec(
                    name="CleanAdvisor",
                    role="Cleaner",
                    responsibilities=["Clean"],
                    assigned_skills=[clean_skill.skill_name],
                ),
                AgentSpec(
                    name="HeatAdvisor",
                    role="Heater",
                    responsibilities=["Heat"],
                    assigned_skills=[heat_skill.skill_name],
                ),
            ],
            skills=[clean_skill, heat_skill],
            backend=backend,
            max_advisors=1,
            prompt_style="mas",
        ).act(
            "Heat task observation",
            active_assigned_skill_names={heat_skill.skill_name},
        )

        self.assertEqual(len(backend.system_prompts), 2)
        self.assertIn("HeatAdvisor", backend.system_prompts[0])
        self.assertNotIn("CleanAdvisor", backend.system_prompts[0])

    def test_alfworld_executor_does_not_receive_assigned_skill_prompt(self):
        class FakeBackend:
            def __init__(self):
                self.user_prompts = []

            def complete(self, system_prompt, user_prompt):
                self.user_prompts.append(user_prompt)
                return LLMResult("<action>look</action>")

        heat_skill = Skill(
            skill_name="heat_target_before_placement",
            description="Heat",
            precondition="Held target needs heating",
            action_protocol=["heat target with microwave"],
            applicable_atomic_ops=[AtomicOp.ACT],
        )
        backend = FakeBackend()
        MASRuntime(
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Act"],
                ),
                AgentSpec(
                    name="HeatAdvisor",
                    role="Heater",
                    responsibilities=["Heat"],
                    assigned_skills=[heat_skill.skill_name],
                ),
            ],
            skills=[heat_skill],
            backend=backend,
        ).act(
            "Observation",
            active_assigned_skill_names={heat_skill.skill_name},
        )

        self.assertNotIn(heat_skill.skill_name, backend.user_prompts[-1])
        self.assertNotIn("heat target with microwave", backend.user_prompts[-1])

    def test_runtime_mas_prompt_style_keeps_custom_executor_system(self):
        class FakeBackend:
            def complete(self, system_prompt, user_prompt):
                return LLMResult("<action>look</action>")

        calls = []

        class RecordingBackend(FakeBackend):
            def complete(self, system_prompt, user_prompt):
                calls.append((system_prompt, user_prompt))
                return super().complete(system_prompt, user_prompt)

        result = MASRuntime(
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Act"],
                )
            ],
            skills=[],
            backend=RecordingBackend(),
            prompt_style="mas",
        ).act("obs")

        self.assertEqual(result.action, "<action>look</action>")
        self.assertIn("only agent allowed to act", calls[0][0])
        self.assertEqual(calls[0][1], "obs")

    def test_openai_backend_retries_unsupported_token_parameter(self):
        class FakeCompletions:
            def __init__(self):
                self.requests = []

            def create(self, **request):
                self.requests.append(request)
                if "max_completion_tokens" in request:
                    raise RuntimeError(
                        "Unsupported parameter: max_completion_tokens"
                    )
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="ok")
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=2,
                        completion_tokens=1,
                    ),
                )

        backend = object.__new__(OpenAIChatBackend)
        backend.model = "gpt-5-mini"
        backend.temperature = 0.0
        completions = FakeCompletions()
        backend.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        result = backend.complete(
            "system",
            "user",
            max_completion_tokens=10,
        )

        self.assertEqual(result.content, "ok")
        self.assertEqual(len(completions.requests), 2)
        self.assertNotIn(
            "max_completion_tokens",
            completions.requests[1],
        )

    def test_evaluator_aborts_when_all_llm_calls_fail(self):
        class FailingBackend:
            def complete(self, system_prompt, user_prompt):
                raise RuntimeError("provider unavailable")

        class FakeEnvironmentManager:
            def __init__(self):
                self.config = SimpleNamespace(
                    env=SimpleNamespace(history_length=0)
                )
                self.step_called = False

            def reset(self, payload):
                return (
                    {
                        "anchor": ["Your task is to: look around."],
                        "text": ["Observation"],
                    },
                    [{}],
                )

            def step(self, actions):
                self.step_called = True
                raise AssertionError("Fallback actions must not be submitted")

            def close(self):
                pass

        manager = FakeEnvironmentManager()
        evaluator = AlfWorldOrganizationEvaluator(
            backend=FailingBackend(),
            config=AlfWorldEvaluatorConfig(
                max_steps=2,
                parallel_envs=1,
                api_concurrency=1,
            ),
        )
        agents = [
            AgentSpec(
                name="Executor",
                role="Environment executor",
                responsibilities=["Act"],
            )
        ]
        with patch(
            "sage_mas.alfworld_evaluator.build_alfworld_env_manager",
            return_value=manager,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "All active SAGE-MAS LLM calls failed",
            ):
                evaluator.evaluate(
                    agents=agents,
                    skills=[],
                    gamefiles=["/fake/task/game.tw-pddl"],
                    condition="test",
                )

        self.assertFalse(manager.step_called)

    def test_select_prompt_agent_gamefiles_matches_first_selection(self):
        gamefiles = [f"/tmp/game-{index}/game.tw-pddl" for index in range(10)]
        with patch(
            "sage_mas.alfworld_evaluator._list_unseen_gamefiles",
            return_value=gamefiles,
        ):
            selected = select_prompt_agent_gamefiles(
                "/tmp/alfworld",
                num_games=4,
                game_selection="first",
                excluded={"/tmp/game-1/game.tw-pddl"},
            )
        self.assertEqual(
            selected,
            [
                "/tmp/game-0/game.tw-pddl",
                "/tmp/game-2/game.tw-pddl",
                "/tmp/game-3/game.tw-pddl",
                "/tmp/game-4/game.tw-pddl",
            ],
        )

    def test_reporting_service_summarizes_by_family(self):
        trials = [
            EvaluationTrial(
                task_id="a",
                task="t1",
                task_family="look_at_obj_in_light",
                condition="baseline",
                reward=1.0,
                cost=10.0,
                won=True,
                num_steps=1,
            ),
            EvaluationTrial(
                task_id="b",
                task="t2",
                task_family="pick_two_obj_and_place",
                condition="baseline",
                reward=0.0,
                cost=12.0,
                won=False,
                num_steps=2,
            ),
        ]
        summary = OrganizationReportingService._summarize("baseline", trials)
        self.assertEqual(summary.wins, 1)
        self.assertEqual(summary.num_games, 2)
        self.assertAlmostEqual(summary.success_rate, 0.5)
        self.assertEqual(
            summary.by_family["look_at_obj_in_light"]["success_rate"],
            1.0,
        )

    def test_enriched_distiller_attaches_compression_and_distribution(self):
        adapter = AlfWorldTrajectoryAdapter()
        trajectories = adapter.adapt_many(
            [
                {
                    "gamefile": f"/tmp/pick_clean_then_place_in_recep-Safe-{i}/game.tw-pddl",
                    "task": "put a clean mug on the desk",
                    "won": True,
                    "num_steps": 4,
                    "steps": [
                        {
                            "observation": "Taken.",
                            "action": "take mug 1 from desk 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "You clean the mug.",
                            "action": "clean mug 1 with sinkbasin 1",
                            "is_action_valid": True,
                        },
                        {
                            "observation": "You put the mug.",
                            "action": "move mug 1 to desk 1",
                            "is_action_valid": True,
                        },
                    ],
                }
                for i in (1, 2, 3)
            ]
        )
        skills = EnrichedSkillDistiller(
            DistillationConfig(
                high_cost_steps=20,
                min_support=2,
                operation_min_support=2,
            ),
            min_wins_for_protocol=3,
        ).distill(trajectories)

        self.assertTrue(skills)
        skill = skills[0]
        self.assertTrue(skill.trajectory_summary)
        self.assertTrue(skill.key_fragments)
        self.assertTrue(skill.metadata.get("confirmed_transitions"))
        self.assertEqual(len(skill.embedding), 64)
        self.assertIsNotNone(skill.exploration_distribution)
        self.assertGreater(skill.exploration_distribution.total_trajectories, 0)
        self.assertTrue(protocol_is_org_ready(skill))

    def test_distribution_shift_against_skill_bank(self):
        baseline = aggregate_skill_bank_distribution(
            [
                Skill(
                    skill_name="Old skill",
                    description="old",
                    precondition="old",
                    action_protocol=["act"],
                    applicable_atomic_ops=[AtomicOp.ACT],
                    exploration_distribution=ExplorationDistribution(
                        task_family_histogram={"pick_and_place": 5},
                        total_trajectories=5,
                    ),
                )
            ]
        )
        observed = ExplorationDistribution(
            task_family_histogram={
                "pick_two_obj_and_place": 3,
                "look_at_obj_in_light": 1,
            },
            total_trajectories=4,
        )
        shift = estimate_distribution_shift(observed, baseline)
        self.assertGreater(shift.kl_divergence, 0.0)
        self.assertGreater(shift.shift_score, 0.0)
        self.assertGreater(
            kl_divergence(
                {"a": 1.0},
                {"b": 1.0},
            ),
            0.0,
        )

    def test_embedding_novelty_ignores_family_histogram_order(self):
        historical = Skill(
            skill_name="Systematic search",
            description="Search unexplored locations",
            precondition="Target has not been found",
            action_protocol=["go to another unexplored location"],
            applicable_atomic_ops=[AtomicOp.ACT],
            embedding=[1.0, 0.0],
            metadata={
                "primary_task_family": "pick_and_place",
                "embedding_version": "skill-behavior-hash-v2",
            },
        )
        current = Skill(
            skill_name="Systematic search",
            description="Search unexplored locations",
            precondition="Target has not been found",
            action_protocol=["go to another unexplored location"],
            applicable_atomic_ops=[AtomicOp.ACT],
            embedding=[1.0, 0.0],
            evidence_ids=["new-a", "new-b"],
            support_count=2,
            metadata={
                "primary_task_family": "pick_heat_then_place_in_recep",
                "embedding_version": "skill-behavior-hash-v2",
            },
            exploration_distribution=ExplorationDistribution(
                outcome_histogram={"failure": 2},
                total_trajectories=2,
            ),
        )

        shift = embedding_novelty_for_skill_cluster(
            [current],
            [historical],
        )

        self.assertEqual(shift.metric, "cosine_nearest_skill")
        self.assertAlmostEqual(shift.nearest_similarity, 1.0)
        self.assertAlmostEqual(shift.novelty_score, 0.0)
        self.assertAlmostEqual(shift.failure_rate, 1.0)

    def test_embedding_novelty_detects_unseen_behavior(self):
        historical = Skill(
            skill_name="Old",
            description="Old",
            precondition="Old",
            action_protocol=["old"],
            applicable_atomic_ops=[AtomicOp.ACT],
            embedding=[1.0, 0.0],
            metadata={"embedding_version": "skill-behavior-hash-v2"},
        )
        current = Skill(
            skill_name="New",
            description="New",
            precondition="New",
            action_protocol=["new"],
            applicable_atomic_ops=[AtomicOp.ACT],
            embedding=[0.0, 1.0],
            evidence_ids=["a"],
            metadata={"embedding_version": "skill-behavior-hash-v2"},
        )

        shift = embedding_novelty_for_skill_cluster(
            [current],
            [historical],
        )

        self.assertAlmostEqual(shift.nearest_similarity, 0.0)
        self.assertAlmostEqual(shift.novelty_score, 0.5)
        self.assertAlmostEqual(shift.shift_score, 0.5)

    def test_behavior_embedding_excludes_task_family_metadata(self):
        left = Skill(
            skill_name="Search",
            description="Search unexplored locations",
            precondition="Target missing",
            action_protocol=["go to another location"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={"primary_task_family": "pick_and_place"},
        )
        right = Skill(
            skill_name="Search",
            description="Search unexplored locations",
            precondition="Target missing",
            action_protocol=["go to another location"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={"primary_task_family": "pick_heat_then_place_in_recep"},
        )

        self.assertEqual(embed_skill(left), embed_skill(right))

    def test_round_exploration_distribution_counts_success_and_failure(self):
        adapter = AlfWorldTrajectoryAdapter()
        trajectories = adapter.adapt_many(
            [
                _trajectory("pick_two_obj_and_place-Safe-1", won=False),
                _trajectory("pick_two_obj_and_place-Safe-2", won=True),
            ]
        )
        distribution = compute_round_exploration_distribution(trajectories)
        self.assertEqual(distribution.total_trajectories, 2)
        self.assertEqual(distribution.outcome_histogram["success"], 1)
        self.assertEqual(distribution.outcome_histogram["failure"], 1)
        self.assertAlmostEqual(distribution.success_rate, 0.5)

    def test_experience_distribution_snapshot_persists_prototypes_and_shift(self):
        historical = Skill(
            skill_name="Historical search",
            description="Search",
            precondition="Missing",
            action_protocol=["look"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            embedding=[1.0, 0.0],
            metadata={"embedding_version": "skill-behavior-hash-v2"},
        )
        current = Skill(
            skill_name="Observed transform",
            description="Transform",
            precondition="Held",
            action_protocol=["transform object"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.PROVISIONAL,
            capability_key="observed-transform",
            support_count=3,
            evidence_ids=["a", "b", "c"],
            embedding=[0.0, 1.0],
            exploration_distribution=ExplorationDistribution(
                task_family_histogram={"transform": 3},
                outcome_histogram={"failure": 2, "success": 1},
                total_trajectories=3,
                success_rate=1 / 3,
            ),
            metadata={"embedding_version": "skill-behavior-hash-v2"},
        )
        snapshot = build_experience_distribution_snapshot(
            [current],
            [historical],
            round_id="round-1",
            novelty_threshold=0.4,
        )
        self.assertEqual(snapshot.prototype_count, 1)
        self.assertEqual(snapshot.total_support, 3)
        self.assertEqual(snapshot.capability_histogram["observed-transform"], 3)
        self.assertAlmostEqual(snapshot.mean_novelty, 0.5)
        self.assertAlmostEqual(snapshot.novel_mass, 1.0)
        self.assertEqual(snapshot.prototypes[0]["evidence_ids"], ["a", "b", "c"])

        comparison = compare_experience_distribution_snapshots(
            snapshot,
            build_experience_distribution_snapshot(
                [],
                [historical],
                round_id="round-1-new",
                round_distribution=ExplorationDistribution(
                    outcome_histogram={"success": 1},
                    total_trajectories=1,
                    success_rate=1.0,
                ),
            ),
        )
        self.assertAlmostEqual(comparison["success_rate_delta"], 2 / 3)
        self.assertEqual(comparison["prototype_count_delta"], -1)

    def test_capability_contract_compiles_role_without_signal_role_map(self):
        skill = Skill(
            skill_name="Observed cleaning protocol",
            description="Execute an observed protocol",
            precondition="The learned precondition matches",
            action_protocol=["clean mug 1 with sinkbasin 1"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            capability_key="observed-cleaning",
            marginal_utility=0.2,
            evidence_ids=["trajectory-1"],
            support_count=1,
            metadata={
                "source_signal": "missing_clean_operation",
                "required_tools": ["alfworld_action"],
            },
        )
        contract = compile_capability_contract([skill])
        agent = AgentRoleCompiler().compile(
            contract,
            {"Executor"},
            executor_name="Executor",
        )
        self.assertEqual(contract.name, "observed-cleaning")
        self.assertEqual(agent.role, "ObservedCleaningSpecialist")
        self.assertNotEqual(agent.role, "CleanOperator")
        self.assertEqual(
            agent.shadow_evaluation_record["capability_contract_id"],
            contract.capability_id,
        )

    def test_online_segment_gamefiles_are_contiguous_and_non_overlapping(self):
        from sage_mas.online_evolution import segment_gamefiles, summarize_trials

        games = [f"g{i}" for i in range(134)]
        segments = segment_gamefiles(games, 20)
        self.assertEqual(len(segments), 7)
        self.assertEqual(len(segments[0]), 20)
        self.assertEqual(len(segments[-1]), 14)
        self.assertEqual([g for seg in segments for g in seg], games)

        trials = [
            EvaluationTrial(
                task_id="a",
                task="t",
                task_family="pick_clean_then_place_in_recep",
                condition="online",
                reward=1.0,
                cost=1.0,
                won=True,
                num_steps=2,
                assigned_primary_agent="CleanSpecialist",
                eligible_agents=["CleanSpecialist"],
                dispatch_layer="eligibility_single",
                actions_by_agent={"CleanSpecialist": 2},
            ),
            EvaluationTrial(
                task_id="b",
                task="t",
                task_family="pick_two_obj_and_place",
                condition="online",
                reward=0.0,
                cost=1.0,
                won=False,
                num_steps=3,
                assigned_primary_agent="Executor",
                eligible_agents=["TwoObjectSpecialist"],
                dispatch_layer="llm_keep_executor",
                actions_by_agent={"Executor": 3},
            ),
        ]
        summary = summarize_trials(trials, name="seg")
        self.assertEqual(summary["wins"], 1)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertEqual(
            summary["by_family"]["pick_clean_then_place_in_recep"]["wins"],
            1,
        )
        dispatch = summary["dispatch"]
        self.assertEqual(
            dispatch["dispatch_count_by_agent"],
            {"CleanSpecialist": 1, "Executor": 1},
        )
        self.assertAlmostEqual(dispatch["specialist_primary_rate"], 0.5)
        self.assertAlmostEqual(dispatch["executor_fallback_rate"], 0.5)
        self.assertEqual(
            dispatch["dispatch_count_by_layer"],
            {"eligibility_single": 1, "llm_keep_executor": 1},
        )
        self.assertEqual(
            dispatch["conditional_success_by_agent"]["CleanSpecialist"][
                "success_rate"
            ],
            1.0,
        )

    def test_skill_recall_bm25_ranks_matching_capability(self):
        from sage_mas.skill_recall import build_recall_index, tokenize

        def mk(name, cap, pre, proto, effect, family, status=SkillStatus.VERIFIED):
            return Skill(
                skill_name=name,
                description=f"{name} desc",
                precondition=pre,
                action_protocol=proto,
                applicable_atomic_ops=[AtomicOp.ACT],
                expected_effect=effect,
                status=status,
                applicable_task_families=[family],
                metadata={"source_signal": "x", "primary_task_family": family},
                capability_key=cap,
            )

        bank = [
            mk("clean proto", "transform.clean", "holding a dirty object",
               ["clean <object> with <tool>"], "object is clean",
               "pick_clean_then_place_in_recep"),
            mk("heat proto", "transform.heat", "holding an object that needs heating",
               ["heat <object> with <appliance>"], "object is hot",
               "pick_heat_then_place_in_recep"),
            mk("cool proto", "transform.cool", "holding an object that needs cooling",
               ["cool <object> with <appliance>"], "object is cold",
               "pick_cool_then_place_in_recep"),
            mk("inspect proto", "inspect.with_light", "need to examine an object in the dark",
               ["use <lamp>"], "object examined under light", "look_at_obj_in_light"),
            mk("place proto", "track.place", "object held",
               ["take <object> from <receptacle>"], "object placed", "pick_and_place"),
            # Rejected skills are not recallable; duplicate names dedup to verified.
            mk("bad proto", "transform.clean", "dirty", ["clean"], "x",
               "pick_clean_then_place_in_recep", status=SkillStatus.REJECTED),
            mk("clean proto", "transform.clean", "dup", ["clean"], "x",
               "pick_clean_then_place_in_recep", status=SkillStatus.PROVISIONAL),
        ]
        index, skills = build_recall_index(bank)
        self.assertEqual(len(skills), 5)

        def top1(query):
            ranked = index.search(tokenize(query), 3)
            return skills[ranked[0][0]].skill_name

        self.assertEqual(
            top1("examine the alarmclock with the desklamp light"),
            "inspect proto",
        )
        self.assertEqual(top1("clean a dirty mug before placing it"), "clean proto")
        self.assertEqual(top1("cool the potato using the fridge"), "cool proto")
        self.assertEqual(top1("heat the apple in the microwave"), "heat proto")

    def test_skill_recall_ignores_shared_protocol_boilerplate(self):
        from sage_mas.skill_recall import build_recall_index, tokenize

        def mk(name, cap, proto):
            return Skill(
                skill_name=name,
                description=(
                    "Slotted-shape consensus protocol. Task family matches. "
                    "find -> access -> pickup. Search prior."
                ),
                precondition=(
                    "Instantiate every placeholder. Follow stage order: find. "
                    "The target object was found at fridge."
                ),
                action_protocol=proto,
                applicable_atomic_ops=[AtomicOp.ACT],
                expected_effect="Episode reaches a won terminal state.",
                status=SkillStatus.VERIFIED,
                capability_key=cap,
                metadata={"trajectory_summary": "put the mug in coffeemachine"},
            )

        bank = [
            mk(
                "loc",
                "perception.object_localization",
                [
                    "go to <source>",
                    "open <source>",
                    "take <object> from <source>",
                    "go to microwave",
                    "heat <object> with microwave",
                    "move <object> to <destination>",
                ],
            ),
            mk(
                "heat",
                "transform.heat",
                [
                    "go to <source>",
                    "take <object> from <source>",
                    "go to microwave",
                    "open microwave",
                    "heat <object> with microwave",
                    "move <object> to <destination>",
                ],
            ),
            mk(
                "cool",
                "transform.cool",
                [
                    "go to <source>",
                    "take <object> from <source>",
                    "go to fridge",
                    "open fridge",
                    "cool <object> with fridge",
                    "move <object> to <destination>",
                ],
            ),
            mk(
                "light",
                "inspect.with_light",
                ["go to desk", "take <object> from desk", "use desklamp"],
            ),
        ]
        index, skills = build_recall_index(bank)

        def scores(query):
            ranked = {
                skills[i].skill_name: score
                for i, score in index.search(tokenize(query), len(skills))
            }
            return ranked

        generic = scores("find remotecontrol")
        self.assertTrue(all(score == 0.0 for score in generic.values()))
        generic = scores("open cabinet")
        self.assertTrue(all(score == 0.0 for score in generic.values()))
        generic = scores("pick up an object")
        self.assertTrue(all(score == 0.0 for score in generic.values()))

        cool_q = scores("cool the plate with the fridge")
        self.assertGreater(cool_q["cool"], 0.0)
        self.assertEqual(cool_q["heat"], 0.0)
        self.assertGreater(cool_q["cool"], cool_q["loc"])

        heat_q = scores("heat the mug with the microwave")
        self.assertGreater(heat_q["heat"], 0.0)
        self.assertEqual(heat_q["cool"], 0.0)
        self.assertGreater(heat_q["heat"], 0.0)

        light_q = scores("examine the pen with the desklamp")
        self.assertGreater(light_q["light"], light_q["cool"])
        self.assertEqual(light_q["cool"], 0.0)

    def test_skill_recall_query_parse(self):
        from sage_mas.skill_recall import parse_recall_query

        self.assertEqual(
            parse_recall_query("<query>cool an object with the fridge</query>"),
            "cool an object with the fridge",
        )
        self.assertEqual(parse_recall_query("cool an object"), "cool an object")
        self.assertIsNone(parse_recall_query("<query>none</query>"))
        self.assertIsNone(parse_recall_query("none"))
        self.assertIsNone(parse_recall_query("line1\nline2 long " + "x" * 200))
        self.assertIsNone(parse_recall_query(""))

    def test_recall_query_prompt_drops_action_template(self):
        from sage_mas.skill_recall import (
            build_recall_query_prompt,
            retrieval_situation,
        )

        template = (
            "You are an expert agent operating in the ALFRED Embodied Environment.\n"
            "Your current observation is: You arrive at sinkbasin 1. "
            "You are holding butterknife 1.\n"
            "Your admissible actions of the current situation are: "
            "['clean butterknife 1 with sinkbasin 1'].\n"
            "Now it's your turn to take an action.\n"
            "You should first reason step-by-step. This reasoning process "
            "MUST be enclosed within <think> </think> tags.\n"
            "Present it within <action> </action> tags.\n"
        )
        situation = retrieval_situation(template)
        self.assertIn("butterknife 1", situation)
        self.assertNotIn("admissible", situation.lower())
        self.assertNotIn("<action>", situation)
        prompt = build_recall_query_prompt("put a clean knife in countertop.", template)
        self.assertIn("butterknife 1", prompt)
        self.assertNotIn("admissible", prompt.lower())
        self.assertNotIn("take an action", prompt)
        self.assertNotIn("MUST be enclosed", prompt)
        self.assertIn("Do not write <think> or <action>.", prompt)
        self.assertIn("<query>none</query>", prompt)
        self.assertIn('find <object>', prompt)
        self.assertIn("operation verb and the tool", prompt)
        self.assertNotIn("Use words from the task and the observation", prompt)
        self.assertNotIn("opening and closing", prompt)
        self.assertNotIn("still searching", prompt)
        raw = "You arrive at sinkbasin 1."
        self.assertEqual(retrieval_situation(raw), raw)

    def test_evaluator_on_demand_recall_mounts_bm25_hits(self):
        from sage_mas.skill_recall import build_recall_index

        class FakeBackend:
            def complete(self, system_prompt, user_prompt):
                if "<query>" in user_prompt:
                    return LLMResult("<query>cool the potato</query>", 1, 1)
                return LLMResult("<action>look</action>", 1, 1)

        class FakeRuntime:
            def __init__(self):
                self.executor = AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["act"],
                )
                self.captured = None

            def act(self, observation, injected_skills, active_assigned_skill_names,
                    preferred_actor_name, task, task_family, history_steps, gamefile):
                self.captured = (list(injected_skills), set(active_assigned_skill_names))
                return SimpleNamespace(action="<action>look</action>", token_cost=1, messages=[])

        cool = Skill(
            skill_name="cool proto",
            description="d",
            precondition="holding an object that needs cooling",
            action_protocol=["cool <object> with <appliance>"],
            applicable_atomic_ops=[AtomicOp.ACT],
            expected_effect="object is cold",
            status=SkillStatus.VERIFIED,
            capability_key="transform.cool",
        )
        place = Skill(
            skill_name="place proto",
            description="d",
            precondition="object held",
            action_protocol=["move <object> to <receptacle>"],
            applicable_atomic_ops=[AtomicOp.ACT],
            expected_effect="object placed",
            status=SkillStatus.VERIFIED,
            capability_key="track.place",
        )
        evaluator = AlfWorldOrganizationEvaluator(
            FakeBackend(),
            AlfWorldEvaluatorConfig(max_injected_skills=2),
        )
        runtime = FakeRuntime()
        index, skills = build_recall_index([cool, place])
        _result, retrieval_info, shown = evaluator._act_with_skill_retrieval(
            runtime,
            observation="You are holding a potato.",
            history_steps=[],
            gamefile="g",
            primary_agent="Executor",
            task="put a cool potato on the table",
            task_family="pick_cool_then_place_in_recep",
            assigned_skills=[],
            assigned_skill_names=set(),
            injected_skills=None,
            ignore_assigned=False,
            agents=[runtime.executor],
            recall_index=index,
            recall_skills=skills,
        )
        self.assertEqual(retrieval_info["query"], "cool the potato")
        self.assertFalse(retrieval_info["skipped"])
        self.assertEqual(retrieval_info["selected"], ["cool proto"])
        self.assertEqual([s.skill_name for s in runtime.captured[0]], ["cool proto"])
        self.assertEqual(runtime.captured[1], set())
        self.assertEqual(shown, ["cool proto"])

        class DeclineBackend(FakeBackend):
            def complete(self, system_prompt, user_prompt):
                return LLMResult("<query>none</query>", 1, 1)

        evaluator.backend = DeclineBackend()
        runtime.captured = None
        _result, retrieval_info, shown = evaluator._act_with_skill_retrieval(
            runtime,
            observation="You are in the kitchen.",
            history_steps=[],
            gamefile="g",
            primary_agent="Executor",
            task="put a cool potato on the table",
            task_family="pick_cool_then_place_in_recep",
            assigned_skills=[],
            assigned_skill_names=set(),
            injected_skills=None,
            ignore_assigned=False,
            agents=[runtime.executor],
            recall_index=index,
            recall_skills=skills,
        )
        self.assertTrue(retrieval_info["skipped"])
        self.assertEqual(runtime.captured[0], [])
        self.assertEqual(shown, [])

    def test_probation_does_not_reorder_scored_task_stream(self):
        from sage_mas.online_evolution import OnlineAlfWorldEvolution

        skill = Skill(
            skill_name="Learned cool",
            description="d",
            precondition="p",
            action_protocol=["act"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.PROVISIONAL,
            applicable_task_families=["pick_cool_then_place_in_recep"],
            metadata={"skill_credit": {"uses": 0, "score": 0.5}},
        )
        original = [
            "/x/pick_and_place/a/game.tw-pddl",
            "/x/pick_two_obj_and_place/b/game.tw-pddl",
            "/x/pick_cool_then_place_in_recep/c/game.tw-pddl",
            "/x/pick_cool_then_place_in_recep/d/game.tw-pddl",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = object.__new__(OnlineAlfWorldEvolution)
            runner.state_path = Path(temp_dir) / "state.json"
            runner.skill_credit_policy = SkillCreditPolicy()
            state = {"segments": [original[:2], original[2:]]}
            selected, summary = runner._prioritize_probation_tasks(
                state,
                1,
                [skill],
            )

        self.assertEqual(selected, original[:2])
        self.assertEqual(
            sorted(item for segment in state["segments"] for item in segment),
            sorted(original),
        )
        self.assertFalse(summary["enabled"])
        self.assertEqual(summary["prioritized_task_count"], 0)

    def test_pipeline_org_eligible_skills_requires_verification(self):
        from sage_mas.pipeline import SageEvolutionPipeline, SKILL_VERIFICATION_ACTIVE

        self.assertTrue(SKILL_VERIFICATION_ACTIVE)
        verified = Skill(
            skill_name="Verified skill",
            description="d",
            precondition="p",
            action_protocol=["take <object>", "move <object> to <receptacle>"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            metadata={
                "protocol_alignment_ok": True,
                "protocol_structure_ok": True,
                "protocol_alignment_support": 2,
            },
        )
        provisional = Skill(
            skill_name="Provisional skill",
            description="d",
            precondition="p",
            action_protocol=["take <object>", "move <object> to <receptacle>"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.PROVISIONAL,
            metadata={
                "protocol_alignment_ok": True,
                "protocol_structure_ok": True,
                "protocol_alignment_support": 2,
            },
        )
        strict_pipeline = SageEvolutionPipeline(
            {"sage": {"allow_provisional_org_edits": False}}
        )
        eligible = strict_pipeline._org_eligible_skills([verified, provisional])
        self.assertEqual(
            {skill.skill_name for skill in eligible},
            {"Verified skill"},
        )

    def test_pipeline_credit_only_org_eligible_without_inject_ready(self):
        from sage_mas.pipeline import SageEvolutionPipeline

        skill = Skill(
            skill_name="Merged transform clean protocol",
            description="Clean target",
            precondition="Target is held",
            action_protocol=[
                "take <object> from <receptacle>",
                "clean <object> with <tool>",
                "move <object> to <receptacle>",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            capability_key="transform.clean",
            applicable_task_families=["pick_clean_then_place_in_recep"],
            metadata={
                "protocol_alignment_ok": True,
                "protocol_structure_ok": True,
                "protocol_form_ok": True,
                "protocol_alignment_support": 3,
                "credit_promoted_pending_org": True,
            },
        )
        credit_only = SageEvolutionPipeline(
            {
                "sage": {
                    "online": {
                        "require_positive_mu_for_injection": False,
                        "paired_mu_probe": False,
                    },
                    "skill_credit": {
                        "require_positive_mu_for_verify": False,
                    },
                }
            }
        )
        self.assertFalse(credit_only.require_inject_ready_for_org)
        self.assertFalse(
            credit_only.org_editor.policy.require_executable_bank_for_add_agent
        )
        eligible = credit_only._org_eligible_skills([skill])
        self.assertEqual([skill.skill_name], [item.skill_name for item in eligible])

    def test_pipeline_initializes_grounded_candidate_credit(self):
        from sage_mas.pipeline import SageEvolutionPipeline

        skill = Skill(
            skill_name="Clean the target object before placement",
            description="Clean target",
            precondition="Target is held",
            action_protocol=[
                "take <object> from <receptacle>",
                "clean <object> with <tool>",
                "move <object> to <receptacle>",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            evidence_ids=["t1", "t2"],
            support_count=2,
            metadata={"source_signal": "missing_clean_operation"},
            capability_key="transform.clean",
            applicable_task_families=["pick_clean_then_place_in_recep"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trajectory_path = temp_path / "trajectories.jsonl"
            write_jsonl(
                trajectory_path,
                [
                    {
                        "trajectory_id": "t1",
                        "task": "put a clean mug on the desk",
                        "task_family": "pick_clean_then_place_in_recep",
                        "won": False,
                        "steps": [],
                    },
                    {
                        "trajectory_id": "t2",
                        "task": "put a clean mug on the desk",
                        "task_family": "pick_clean_then_place_in_recep",
                        "won": True,
                        "steps": [
                            {
                                "observation": "Target held.",
                                "action": "take mug 1 from desk 1",
                                "is_action_valid": True,
                            },
                            {
                                "observation": "Target cleaned.",
                                "action": "clean mug 1 with sinkbasin 1",
                                "is_action_valid": True,
                            },
                        ],
                    },
                ],
            )
            candidate_path = temp_path / "candidate_skills.json"
            write_json(candidate_path, [skill])
            pipeline = SageEvolutionPipeline(
                {
                    "sage": {
                        "organization": {
                            "distribution_shift_threshold": 0.0,
                            "min_cluster_support": 1,
                        },
                        "agents": [
                            {
                                "name": "Executor",
                                "role": "Environment executor",
                                "responsibilities": ["Act"],
                            }
                        ],
                    }
                }
            )
            artifacts = pipeline.run(
                trajectory_path=trajectory_path,
                output_root=temp_path / "distill",
                candidate_skills_path=candidate_path,
            )
            loaded = load_skills(artifacts.candidate_skills)
            edits = read_json(artifacts.organization_edits)
            snapshot = read_json(
                artifacts.experience_distribution_snapshot
            )
            contracts = read_json(artifacts.capability_contracts)

        self.assertEqual(loaded[0].status, SkillStatus.PROVISIONAL)
        self.assertEqual(loaded[0].metadata["skill_credit"]["score"], 0.5)
        self.assertEqual(snapshot["round_id"], Path(artifacts.run_dir).name)
        self.assertEqual(snapshot["prototype_count"], 1)
        self.assertEqual(contracts, [])
        self.assertTrue(all(edit.get("edit_type") == "do_nothing" for edit in edits))

    def test_online_partition_carves_shadow_from_budget(self):
        from sage_mas.online_evolution import OnlineAlfWorldEvolution

        selected = [f"/fake/game-{index}/game.tw-pddl" for index in range(134)]
        llm_config = {
            "openai": {"model": "fake"},
            "alfworld": {"data_path": "/unused"},
        }
        sage_config = {
            "sage": {
                "online": {
                    "num_games": 134,
                    "shadow_pool_size": 14,
                    "segment_size": 20,
                    "seed": 1,
                },
                "agents": [
                    {
                        "name": "Executor",
                        "role": "Environment executor",
                        "responsibilities": ["Act"],
                    }
                ],
            }
        }
        with patch(
            "sage_mas.online_evolution._list_unseen_gamefiles",
            return_value=selected,
        ):
            with patch.object(
                OnlineAlfWorldEvolution,
                "_select_gamefiles",
                return_value=selected,
            ):
                online = OnlineAlfWorldEvolution(
                    llm_config=llm_config,
                    sage_config=sage_config,
                    output_root="/tmp/unused-online-partition-test",
                    evaluator=object(),
                )
                partition = online._select_and_partition_online_gamefiles()

        self.assertTrue(partition["mechanism_carved_from_budget"])
        self.assertEqual(len(partition["shadow_pool"]), 14)
        self.assertEqual(len(partition["collection"]), 120)

    def test_online_partition_uses_external_mechanism_dataset(self):
        from sage_mas.online_evolution import OnlineAlfWorldEvolution

        collection = [
            f"/fake/valid_unseen/game-{index}/game.tw-pddl"
            for index in range(134)
        ]
        shadow_tasks = [
            f"/fake/valid_seen/shadow-{index}/game.tw-pddl"
            for index in range(30)
        ]
        llm_config = {
            "openai": {"model": "fake"},
            "alfworld": {"data_path": "/unused"},
        }
        sage_config = {
            "sage": {
                "online": {
                    "num_games": 134,
                    "shadow_pool_size": 30,
                    "mechanism_dataset": "valid_seen",
                    "segment_size": 20,
                    "seed": 1,
                },
                "agents": [
                    {
                        "name": "Executor",
                        "role": "Environment executor",
                        "responsibilities": ["Act"],
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                OnlineAlfWorldEvolution,
                "_select_gamefiles",
                return_value=collection,
            ), patch.object(
                OnlineAlfWorldEvolution,
                "_select_mechanism_gamefiles",
                return_value=shadow_tasks,
            ):
                online = OnlineAlfWorldEvolution(
                    llm_config=llm_config,
                    sage_config=sage_config,
                    output_root=temp_dir,
                    evaluator=object(),
                )
                state = read_json(online.state_path)

        self.assertEqual(len(state["gamefiles"]), 134)
        self.assertNotIn("discovery_pool", state)
        self.assertEqual(state["shadow_pool"], shadow_tasks)
        self.assertFalse(state["config"]["mechanism_carved_from_budget"])
        self.assertEqual(state["config"]["mechanism_dataset"], "valid_seen")
        self.assertTrue(
            set(state["gamefiles"]).isdisjoint(state["shadow_pool"])
        )

    def test_online_segment_commits_without_shadow_and_dormant_agents(self):
        from sage_mas.online_evolution import OnlineAlfWorldEvolution

        class FakeEvaluator:
            backend = None

            def evaluate(
                self,
                agents,
                skills,
                gamefiles,
                condition,
                injected_skills=None,
            ):
                if condition == "with_skill":
                    reward = 1.0
                elif condition == "new_organization":
                    reward = (
                        1.0
                        if any(agent.assigned_skills for agent in agents)
                        else float(len(agents) > 1)
                    )
                else:
                    reward = 0.0
                return [
                    EvaluationTrial(
                        task_id=gamefile,
                        task="put a clean mug on the desk",
                        task_family="pick_clean_then_place_in_recep",
                        condition=condition,
                        reward=reward,
                        cost=10.0,
                        won=bool(reward),
                        num_steps=2,
                        steps=[
                            {
                                "observation": "You pick up the mug.",
                                "action": "<think>x</think><action>take mug 1 from desk 1</action>",
                                "is_action_valid": True,
                            },
                            {
                                "observation": "Nothing happens.",
                                "action": "<think>x</think><action>look</action>",
                                "is_action_valid": False,
                            },
                        ],
                        activated_skill_names=[
                            injected_skills[0].skill_name
                        ] if injected_skills else [],
                    )
                    for gamefile in gamefiles
                ]

        llm_config = {
            "openai": {"model": "fake"},
            "alfworld": {"data_path": "/unused"},
        }
        collection = [f"/fake/collect-{index}/game.tw-pddl" for index in range(4)]
        sage_config = {
            "sage": {
                "allow_provisional_org_edits": True,
                "distillation": {"high_cost_steps": 10, "min_support": 1},
                "verification": {
                    "enabled": False,
                    "utility_threshold": 0.1,
                    "cost_weight": 0.0,
                    "min_probe_count": 1,
                },
                "probe": {"num_tasks": 1, "max_attempts_per_skill": 2},
                "shadow": {
                    "enabled": False,
                    "num_tasks": 2,
                    "cost_weight": 0.0,
                    "min_relative_gain": 0.0,
                    "min_paired_tasks": 1,
                },
                "online": {
                    "num_games": 4,
                    "segment_size": 2,
                    "shadow_pool_size": 0,
                    "skip_shadow": True,
                    "commit_org_on_last_segment": True,
                    "max_skills_per_round": 1,
                    "distill_after_last_segment": False,
                    "seed": 1,
                },
                "organization": {
                    "distribution_shift_threshold": 0.0,
                    "min_cluster_support": 1,
                    "allow_bootstrap_add": True,
                    "max_bootstrap_agents_per_round": 2,
                    "cluster_by": "family_signal",
                    "dispatch_only_new_agents": True,
                },
                "agents": [
                    {
                        "name": "Executor",
                        "role": "Environment executor",
                        "responsibilities": ["Issue actions"],
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            partition = {
                "shadow_pool": [],
                "collection": collection,
                "mechanism_carved_from_budget": False,
            }
            with patch.object(
                OnlineAlfWorldEvolution,
                "_select_and_partition_online_gamefiles",
                return_value=partition,
            ):
                online = OnlineAlfWorldEvolution(
                    llm_config=llm_config,
                    sage_config=sage_config,
                    output_root=temp_dir,
                    evaluator=FakeEvaluator(),
                )
                state = read_json(online.state_path)
                outcome = online.run_segment(1, state)
                state = online._commit_segment(state, outcome)
                active_agents = load_agents(
                    Path(temp_dir) / "active_organization.json"
                )

        self.assertEqual(outcome["status"], "no_candidates")
        self.assertEqual(state["completed_segments"], 1)
        self.assertEqual(len(active_agents), 1)
        self.assertIsNone(outcome.get("shadow_result"))

    def test_shadow_sampling_targets_new_specialist_capability(self):
        from sage_mas.online_evolution import OnlineAlfWorldEvolution

        online = OnlineAlfWorldEvolution.__new__(OnlineAlfWorldEvolution)
        online.sage_config = {"sage": {"shadow": {"num_tasks": 2}}}
        active = [
            AgentSpec(
                name="Executor",
                role="executor",
                responsibilities=["act"],
            )
        ]
        candidate = active + [
            AgentSpec(
                name="CleanSpecialist",
                role="specialist",
                responsibilities=["clean"],
                assigned_skills=["Verified clean"],
            )
        ]
        skill = Skill(
            skill_name="Verified clean",
            description="d",
            precondition="p",
            action_protocol=["clean object"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            applicable_task_families=[
                "pick_clean_then_place_in_recep"
            ],
        )
        families = online._shadow_capability_families(
            active,
            candidate,
            [skill],
        )
        pool = [
            "/valid_seen/pick_heat_then_place_in_recep-A/game.tw-pddl",
            "/valid_seen/pick_clean_then_place_in_recep-A/game.tw-pddl",
            "/valid_seen/pick_two_obj_and_place-A/game.tw-pddl",
            "/valid_seen/pick_clean_then_place_in_recep-B/game.tw-pddl",
        ]

        selected = online._round_shadow_tasks(
            pool,
            1,
            required_families=families,
        )

        self.assertEqual(
            families,
            ["pick_clean_then_place_in_recep"],
        )
        self.assertEqual(len(selected), 2)
        self.assertTrue(
            all("pick_clean_then_place_in_recep" in item for item in selected)
        )

    def test_shadow_families_ignore_executor_skill_copies(self):
        from sage_mas.online_evolution import OnlineAlfWorldEvolution

        online = OnlineAlfWorldEvolution.__new__(OnlineAlfWorldEvolution)
        active = [
            AgentSpec(
                name="Executor",
                role="executor",
                responsibilities=["act"],
            )
        ]
        candidate = [
            AgentSpec(
                name="Executor",
                role="executor",
                responsibilities=["act"],
                assigned_skills=[
                    "Verified clean",
                    "Verified inspect",
                ],
            ),
            AgentSpec(
                name="CleanSpecialist",
                role="specialist",
                responsibilities=["clean"],
                assigned_skills=["Verified clean"],
            ),
        ]
        skills = [
            Skill(
                skill_name="Verified clean",
                description="d",
                precondition="p",
                action_protocol=["clean object"],
                applicable_atomic_ops=[AtomicOp.ACT],
                status=SkillStatus.VERIFIED,
                applicable_task_families=[
                    "pick_clean_then_place_in_recep"
                ],
            ),
            Skill(
                skill_name="Verified inspect",
                description="d",
                precondition="p",
                action_protocol=["use desklamp"],
                applicable_atomic_ops=[AtomicOp.ACT],
                status=SkillStatus.VERIFIED,
                applicable_task_families=["look_at_obj_in_light"],
            ),
        ]

        families = online._shadow_capability_families(
            active,
            candidate,
            skills,
        )

        self.assertEqual(families, ["pick_clean_then_place_in_recep"])

    def test_specialist_dispatch_correctness_requires_in_contract_routing(self):
        from sage_mas.online_evolution import OnlineAlfWorldEvolution

        active = [
            AgentSpec(
                name="Executor",
                role="executor",
                responsibilities=["act"],
            )
        ]
        candidate = active + [
            AgentSpec(
                name="CleanSpecialist",
                role="specialist",
                responsibilities=["clean"],
            )
        ]
        good = EvaluationTrial(
            task_id="/clean/game.tw-pddl",
            task="clean mug",
            task_family="pick_clean_then_place_in_recep",
            condition="new_organization",
            reward=1.0,
            cost=1.0,
            won=True,
            num_steps=2,
            assigned_primary_agent="CleanSpecialist",
            actions_by_agent={"CleanSpecialist": 2},
        )
        leaked = EvaluationTrial(
            task_id="/light/game.tw-pddl",
            task="look",
            task_family="look_at_obj_in_light",
            condition="new_organization",
            reward=1.0,
            cost=1.0,
            won=True,
            num_steps=2,
            assigned_primary_agent="CleanSpecialist",
            actions_by_agent={"CleanSpecialist": 2},
        )

        utilization = OnlineAlfWorldEvolution._specialist_shadow_utilization(
            active,
            candidate,
            [good, leaked],
            required_families=["pick_clean_then_place_in_recep"],
        )
        self.assertFalse(utilization["dispatch_correct"])
        self.assertEqual(utilization["out_of_contract_primary_count"], 1)

        utilization_ok = OnlineAlfWorldEvolution._specialist_shadow_utilization(
            active,
            candidate,
            [
                good,
                EvaluationTrial(
                    task_id="/light/game.tw-pddl",
                    task="look",
                    task_family="look_at_obj_in_light",
                    condition="new_organization",
                    reward=1.0,
                    cost=1.0,
                    won=True,
                    num_steps=2,
                    assigned_primary_agent="Executor",
                    actions_by_agent={"Executor": 2},
                ),
            ],
            required_families=["pick_clean_then_place_in_recep"],
        )
        self.assertTrue(utilization_ok["dispatch_correct"])

    def test_add_agent_shadow_rejects_ineffective_specialist(self):
        from sage_mas.online_evolution import OnlineAlfWorldEvolution

        online = OnlineAlfWorldEvolution.__new__(OnlineAlfWorldEvolution)
        online.sage_config = {
            "sage": {
                "shadow": {
                    "min_relative_gain": 0.125,
                    "cost_weight": 0.0,
                    "agent_count_weight": 0.05,
                    "require_confident_gain": False,
                    "min_paired_tasks": 1,
                    "bootstrap_samples": 0,
                    "seed": 0,
                }
            }
        }
        active = [
            AgentSpec(
                name="Executor",
                role="executor",
                responsibilities=["act"],
            )
        ]
        candidate = active + [
            AgentSpec(
                name="CleanSpecialist",
                role="specialist",
                responsibilities=["clean"],
                assigned_skills=["Verified clean"],
            )
        ]
        skill = Skill(
            skill_name="Verified clean",
            description="d",
            precondition="p",
            action_protocol=["clean object"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            applicable_task_families=[
                "pick_clean_then_place_in_recep"
            ],
        )
        old_trials = [
            EvaluationTrial(
                task_id="/clean/a.tw-pddl",
                task="clean",
                task_family="pick_clean_then_place_in_recep",
                condition="old_organization",
                reward=0.0,
                cost=1.0,
                won=False,
                num_steps=2,
                assigned_primary_agent="Executor",
                actions_by_agent={"Executor": 2},
            )
        ]
        new_trials = [
            EvaluationTrial(
                task_id="/clean/a.tw-pddl",
                task="clean",
                task_family="pick_clean_then_place_in_recep",
                condition="new_organization",
                reward=0.0,
                cost=1.0,
                won=False,
                num_steps=2,
                assigned_primary_agent="CleanSpecialist",
                actions_by_agent={"CleanSpecialist": 2},
                activated_skill_names=["Verified clean"],
            )
        ]
        evidence = OnlineAlfWorldEvolution._specialist_shadow_utilization(
            active,
            candidate,
            new_trials,
            required_families=["pick_clean_then_place_in_recep"],
        )
        decision = online._add_agent_shadow_decision(
            old_trials=old_trials,
            new_trials=new_trials,
            required_families=["pick_clean_then_place_in_recep"],
            specialist_evidence=evidence,
            fallback_decision=ShadowDecision(
                accepted=True,
                old_utility=1.0,
                new_utility=1.0,
                relative_gain=0.0,
                reason="fallback",
            ),
            skills=[skill],
            old_agents=active,
            new_agents=candidate,
        )
        self.assertTrue(evidence["dispatch_correct"])
        self.assertFalse(decision.accepted)
        self.assertIn("did not improve in-contract success", decision.reason)

    def test_shadow_zero_dispatch_is_insufficient_evidence(self):
        from sage_mas.online_evolution import OnlineAlfWorldEvolution

        active = [
            AgentSpec(
                name="Executor",
                role="executor",
                responsibilities=["act"],
            )
        ]
        candidate = active + [
            AgentSpec(
                name="CleanSpecialist",
                role="specialist",
                responsibilities=["clean"],
            )
        ]
        trial = EvaluationTrial(
            task_id="/clean/game.tw-pddl",
            task="clean mug",
            task_family="pick_clean_then_place_in_recep",
            condition="new_organization",
            reward=0.0,
            cost=1.0,
            won=False,
            num_steps=2,
            assigned_primary_agent="Executor",
            actions_by_agent={"Executor": 2},
        )

        utilization = (
            OnlineAlfWorldEvolution._specialist_shadow_utilization(
                active,
                candidate,
                [trial],
                required_families=[
                    "pick_clean_then_place_in_recep"
                ],
            )
        )

        self.assertEqual(utilization["primary_dispatch_count"], 0)
        self.assertEqual(utilization["actions_executed"], 0)
        self.assertFalse(utilization["evidence_sufficient"])

    def test_deferred_probation_commits_candidate_organization(self):
        from sage_mas.online_evolution import (
            OnlineAlfWorldEvolution,
            OnlineEvolutionConfig,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            online = OnlineAlfWorldEvolution.__new__(
                OnlineAlfWorldEvolution
            )
            online.active_organization_path = root / "active.json"
            online.state_path = root / "state.json"
            online.skill_bank_path = root / "bank.json"
            online.config = OnlineEvolutionConfig(test_eval_enabled=False)
            selected = root / "selected.json"
            write_json(
                online.active_organization_path,
                {
                    "agents": [
                        AgentSpec(
                            name="Executor",
                            role="executor",
                            responsibilities=["act"],
                        )
                    ]
                },
            )
            write_json(
                selected,
                {
                    "agents": [
                        AgentSpec(
                            name="Executor",
                            role="executor",
                            responsibilities=["act"],
                        ),
                        AgentSpec(
                            name="CleanSpecialist",
                            role="specialist",
                            responsibilities=["clean"],
                        ),
                    ]
                },
            )
            state = {
                "history": [],
                "all_trials": [],
                "trajectory_paths": [],
                "completed_segments": 0,
            }

            online._commit_segment(
                state,
                {
                    "round": 1,
                    "status": "deferred_probation",
                    "selected_organization": str(selected),
                    "trials": [],
                },
            )

            self.assertEqual(
                len(load_agents(online.active_organization_path)),
                2,
            )

    def test_capability_cluster_key_and_family_signal_grouping(self):
        from sage_mas.pipeline import SageEvolutionPipeline

        clean_a = Skill(
            skill_name="Clean A",
            description="d",
            precondition="p",
            action_protocol=["a"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "source_signal": "missing_clean_operation",
            },
        )
        clean_b = Skill(
            skill_name="Clean B",
            description="d",
            precondition="p",
            action_protocol=["a"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "source_signal": "missing_clean_operation",
            },
        )
        search = Skill(
            skill_name="Search",
            description="d",
            precondition="p",
            action_protocol=["a"],
            applicable_atomic_ops=[AtomicOp.ACT],
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "source_signal": "search_exhaustion",
            },
        )
        self.assertEqual(
            cluster_key_for_skill(clean_a),
            "pick_clean_then_place_in_recep|missing_clean_operation",
        )
        clusters = SageEvolutionPipeline._cluster_skills(
            [clean_a, clean_b, search],
            cluster_by="family_signal",
        )
        self.assertEqual(len(clusters), 2)
        sizes = sorted(len(cluster) for cluster in clusters)
        self.assertEqual(sizes, [1, 2])

    def test_low_support_cluster_does_not_add_agent(self):
        skill = Skill(
            skill_name="Sparse",
            description="d",
            precondition="p",
            action_protocol=["a"],
            applicable_atomic_ops=[AtomicOp.ACT],
            evidence_ids=["only-one"],
            distribution_shift=DistributionShift(
                kl_divergence=1.0,
                shift_score=1.0,
                baseline_source="empty_skill_bank",
            ),
            metadata={
                "primary_task_family": "pick_and_place",
                "source_signal": "search_exhaustion",
            },
        )
        gap = AssignmentGapResult(
            skill_ids=[skill.skill_id],
            assignment_gap=1.0,
            best_agent_id=None,
            best_agent_name=None,
            best_capacity=0.0,
            capacities={},
        )
        edit = OrganizationEditor(
            OrganizationPolicy(min_cluster_support=3)
        ).propose([skill], gap)
        self.assertEqual(edit.edit_type, OrganizationEditType.DO_NOTHING)

    def test_no_deviation_assigns_instead_of_add(self):
        skill = Skill(
            skill_name="Stable",
            description="d",
            precondition="p",
            action_protocol=["a"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            evidence_ids=["a", "b", "c"],
            distribution_shift=DistributionShift(
                kl_divergence=0.01,
                shift_score=0.01,
                baseline_source="skill_bank",
            ),
            metadata={
                "primary_task_family": "pick_and_place",
                "source_signal": "search_exhaustion",
            },
        )
        agents = [
            AgentSpec(
                name="Executor",
                role="Environment executor",
                responsibilities=["Act"],
            )
        ]
        gap = AssignmentGapResult(
            skill_ids=[skill.skill_id],
            assignment_gap=0.5,
            best_agent_id=agents[0].agent_id,
            best_agent_name="Executor",
            best_capacity=0.5,
            capacities={agents[0].agent_id: 0.5},
        )
        edit = OrganizationEditor(
            OrganizationPolicy(
                distribution_shift_threshold=0.15,
                min_cluster_support=3,
            )
        ).propose([skill], gap)
        self.assertEqual(edit.edit_type, OrganizationEditType.ASSIGN_SKILL)
        self.assertEqual(edit.target_agent, "Executor")

    def test_uncovered_capability_adds_agent_from_executor_only(self):
        """Executor-only org can ADD when capability has no specialist carrier."""
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
            tool_permissions=["alfworld_action"],
        )
        skill = Skill(
            skill_name="Merged transform heat protocol",
            description="d",
            precondition="p",
            action_protocol=["heat"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            evidence_ids=["e1", "e2"],
            capability_key="transform.heat",
            marginal_utility=0.25,
            metadata={
                "primary_task_family": "pick_heat_then_place_in_recep",
                "marginal_utility": 0.25,
                "mu_promoted": True,
                "executable_protocol": [{"action": "heat object"}],
                "executable_protocol_source": "action_protocol",
            },
            distribution_shift=DistributionShift(
                kl_divergence=0.0,
                shift_score=0.05,
                baseline_source="skill_bank",
                metric="cosine_nearest_skill",
                nearest_skill_name="other",
            ),
        )
        gap = AssignmentGapResult(
            skill_ids=[skill.skill_id],
            assignment_gap=0.2,
            best_agent_id=executor.agent_id,
            best_agent_name="Executor",
            best_capacity=0.8,
            capacities={executor.agent_id: 0.8},
        )
        edit = OrganizationEditor(
            OrganizationPolicy(
                distribution_shift_threshold=0.15,
                add_when_uncovered_capability=True,
                require_positive_mu_for_add_agent=True,
                min_protocol_adherence_for_add_agent=0.0,
                require_trajectory_executable_protocol=False,
                require_executable_protocol=True,
            )
        ).propose([skill], gap, existing_agents=[executor])
        self.assertEqual(edit.edit_type, OrganizationEditType.ADD_AGENT)
        self.assertIsNotNone(edit.new_agent)

    def test_high_shift_uncovered_still_adds_from_executor_only(self):
        """SAGE: high cluster shift must not skip uncovered → ASSIGN-to-Executor.

        Seg1 Exp B previously parked MU-proven skills on Executor because
        distribution deviation short-circuited uncovered marking while
        prefer_assign_before_add treated Executor as a carrier.
        """
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Act"],
            tool_permissions=["alfworld_action"],
        )
        skill = Skill(
            skill_name="Merged track multiple objects protocol",
            description="d",
            precondition="p",
            action_protocol=["take object"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            evidence_ids=[f"e{i}" for i in range(10)],
            capability_key="track.multiple_objects",
            marginal_utility=0.33,
            metadata={
                "primary_task_family": "pick_two_obj_and_place",
                "marginal_utility": 0.33,
                "mu_promoted": True,
                "executable_protocol": [{"action": "take object"}],
                "executable_protocol_source": "action_protocol",
            },
            distribution_shift=DistributionShift(
                kl_divergence=0.0,
                shift_score=0.133,
                baseline_source="skill_bank",
                metric="cosine_nearest_skill",
                nearest_skill_name="Merged transform cool protocol",
            ),
        )
        gap = AssignmentGapResult(
            skill_ids=[skill.skill_id],
            assignment_gap=0.40,
            best_agent_id=executor.agent_id,
            best_agent_name="Executor",
            best_capacity=0.60,
            capacities={executor.agent_id: 0.60},
        )
        edit = OrganizationEditor(
            OrganizationPolicy(
                distribution_shift_threshold=0.08,
                add_when_uncovered_capability=True,
                prefer_assign_before_add=True,
                require_positive_mu_for_add_agent=True,
                min_protocol_adherence_for_add_agent=0.0,
                require_trajectory_executable_protocol=False,
                require_executable_protocol=True,
                min_assignment_gap=0.05,
            )
        ).propose([skill], gap, existing_agents=[executor])
        self.assertEqual(edit.edit_type, OrganizationEditType.ADD_AGENT)
        self.assertIsNotNone(edit.new_agent)
        self.assertIn("uncovered_capability=True", edit.rationale)

    def test_empty_bank_bootstrap_quota_limits_add_agent(self):
        policy = OrganizationPolicy(
            distribution_shift_threshold=0.15,
            min_cluster_support=2,
            allow_bootstrap_add=True,
            max_bootstrap_agents_per_round=1,
        )
        editor = OrganizationEditor(policy)
        agents = [
            AgentSpec(
                name="Executor",
                role="Environment executor",
                responsibilities=["Act"],
            )
        ]
        clean = Skill(
            skill_name="Clean",
            description="d",
            precondition="p",
            action_protocol=["a"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            evidence_ids=["c1", "c2"],
            distribution_shift=DistributionShift(
                kl_divergence=1.0,
                shift_score=1.0,
                baseline_source="empty_skill_bank",
            ),
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "source_signal": "missing_clean_operation",
            },
        )
        heat = Skill(
            skill_name="Heat",
            description="d",
            precondition="p",
            action_protocol=["a"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            evidence_ids=["h1", "h2"],
            distribution_shift=DistributionShift(
                kl_divergence=1.0,
                shift_score=1.0,
                baseline_source="empty_skill_bank",
            ),
            metadata={
                "primary_task_family": "pick_heat_then_place_in_recep",
                "source_signal": "missing_heat_operation",
            },
        )
        gap_clean = AssignmentGapEstimator().estimate([clean], agents)
        first = editor.propose(
            [clean],
            gap_clean,
            existing_agents=agents,
            bootstrap_slots_remaining=1,
        )
        self.assertEqual(first.edit_type, OrganizationEditType.ADD_AGENT)
        agents_after = agents + [first.new_agent]
        gap_heat = AssignmentGapEstimator().estimate([heat], agents_after)
        second = editor.propose(
            [heat],
            gap_heat,
            existing_agents=agents_after,
            existing_agent_names={agent.name for agent in agents_after},
            bootstrap_slots_remaining=0,
        )
        self.assertNotEqual(second.edit_type, OrganizationEditType.ADD_AGENT)

    def test_sole_capability_cluster_bootstraps_instead_of_zero_shift(self):
        """First/only bank capability should get empty-bank bootstrap, not shift=0."""
        from sage_mas.distribution_stats import shift_for_skill_cluster

        only = Skill(
            skill_name="Trajectory-derived execution efficiency protocol",
            description="d",
            precondition="p",
            action_protocol=["take", "move"],
            applicable_atomic_ops=[AtomicOp.ACT],
            skill_id="eff-1",
            support_count=4,
            evidence_ids=["e1", "e2", "e3", "e4"],
            status=SkillStatus.VERIFIED,
            capability_key="execution.efficiency",
            applicable_task_families=["pick_and_place"],
            metadata={"primary_task_family": "pick_and_place"},
        )
        # Bank only contains this cluster → historical baseline empty.
        historical = []
        shift = shift_for_skill_cluster(
            [only],
            None,
            baseline_skills=historical,
        )
        self.assertEqual(shift.baseline_source, "empty_skill_bank")
        self.assertGreater(shift.shift_score, 0.0)
        editor = OrganizationEditor(
            OrganizationPolicy(
                distribution_shift_threshold=0.05,
                allow_bootstrap_add=True,
                max_bootstrap_agents_per_round=1,
                min_assignment_gap=0.1,
            )
        )
        only.distribution_shift = shift
        gap = AssignmentGapEstimator().estimate(
            [only],
            [
                AgentSpec(
                    name="Executor",
                    role="executor",
                    responsibilities=["act"],
                )
            ],
        )
        edit = editor.propose(
            [only],
            gap,
            existing_agents=[
                AgentSpec(
                    name="Executor",
                    role="executor",
                    responsibilities=["act"],
                )
            ],
            existing_agent_names={"Executor"},
            cluster_shift=shift,
            bootstrap_slots_remaining=1,
        )
        self.assertEqual(edit.edit_type, OrganizationEditType.ADD_AGENT)

    def test_multi_object_structure_accepts_expanded_take_place_slots(self):
        protocol = canonicalize_protocol_stages(
            [
                "go to <location>",
                "take <object> from <receptacle>",
                "move <object> to <receptacle>",
            ],
            capability="track.multiple_objects",
        )
        issues = protocol_structure_issues(
            protocol,
            capability="track.multiple_objects",
        )
        self.assertEqual(issues, [])
        self.assertGreaterEqual(
            sum(1 for step in protocol if stage_for_abstracted_action(step) == "pickup"),
            1,
        )

    def test_dispatch_auto_assigns_single_eligible_specialist(self):
        from sage_mas.executor_dispatch import (
            ExecutorDispatchConfig,
            ExecutorDispatcher,
        )

        skill = Skill(
            skill_name="Clean protocol",
            description="Clean objects",
            precondition="Need clean",
            action_protocol=["clean"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
            },
            applicable_task_families=["pick_clean_then_place_in_recep"],
        )
        executor = AgentSpec(
            name="Executor",
            role="executor",
            responsibilities=["act"],
        )
        cleaner = AgentSpec(
            name="TransformCleanSpecialist",
            role="Clean specialist",
            responsibilities=["clean"],
            assigned_skills=[skill.skill_name],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "probation",
                "task_families": ["pick_clean_then_place_in_recep"],
                "dispatch_only": True,
                "trial_games_remaining": 8,
            },
        )
        assignment = ExecutorDispatcher(
            ExecutorDispatchConfig(auto_assign_single=True),
        ).assign(
            task="put a clean mug on the desk",
            task_family="pick_clean_then_place_in_recep",
            agents=[executor, cleaner],
            skills=[skill],
            gamefile="/fake/pick_clean_then_place_in_recep-0/game.tw-pddl",
        )
        self.assertEqual(assignment.primary_agent, "TransformCleanSpecialist")
        self.assertEqual(assignment.dispatch_layer, "eligibility_single")

    def test_scope_sync_enables_stale_contract_specialist(self):
        from sage_mas.capability_contract import sync_agent_skill_scopes
        from sage_mas.executor_dispatch import (
            ExecutorDispatchConfig,
            ExecutorDispatcher,
        )

        skill = Skill(
            skill_name="heat_and_place_object",
            description="Heat then place",
            precondition="Need heat",
            action_protocol=["heat"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            applicable_task_families=["pick_heat_then_place_in_recep"],
            metadata={"primary_task_family": "pick_heat_then_place_in_recep"},
        )
        executor = AgentSpec(
            name="Executor",
            role="executor",
            responsibilities=["act"],
            tool_permissions=["alfworld_action"],
        )
        # Stale contract: skill assigned but agent scope empty / wrong.
        heater = AgentSpec(
            name="TransformHeatSpecialist",
            role="Heat specialist",
            responsibilities=["heat"],
            assigned_skills=[skill.skill_name],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "probation",
                "trial_games_remaining": 0,
                "applicable_dispatched_games": 0,
                "task_families": [],
                "capability_contract": {"task_families": []},
            },
        )
        self.assertTrue(sync_agent_skill_scopes(heater, [skill]))
        self.assertIn(
            "pick_heat_then_place_in_recep",
            heater.shadow_evaluation_record["task_families"],
        )
        assignment = ExecutorDispatcher(
            ExecutorDispatchConfig(
                prefer_matching_specialist=True,
                require_accepted_for_primary=True,
                probation_primary_quota=8,
                auto_assign_single=False,
            )
        ).assign(
            task="heat some apple and put it in fridge",
            task_family="pick_heat_then_place_in_recep",
            agents=[executor, heater],
            skills=[skill],
            gamefile="/fake/pick_heat_then_place_in_recep-0/game.tw-pddl",
        )
        self.assertEqual(assignment.primary_agent, "TransformHeatSpecialist")
        self.assertIn("prefer_matching", assignment.dispatch_layer)

    def test_prefer_matching_overrides_llm_keep_executor(self):
        from sage_mas.executor_dispatch import (
            ExecutorDispatchConfig,
            ExecutorDispatcher,
        )

        skill = Skill(
            skill_name="cool_and_place",
            description="Cool then place",
            precondition="Need cool",
            action_protocol=["cool"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            applicable_task_families=["pick_cool_then_place_in_recep"],
            metadata={"primary_task_family": "pick_cool_then_place_in_recep"},
        )
        executor = AgentSpec(
            name="Executor",
            role="executor",
            responsibilities=["act"],
            tool_permissions=["alfworld_action"],
        )
        cooler = AgentSpec(
            name="TransformCoolSpecialist",
            role="Cool specialist",
            responsibilities=["cool"],
            assigned_skills=[skill.skill_name],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "accepted",
                "task_families": ["pick_cool_then_place_in_recep"],
            },
        )

        class KeepExecutorBackend:
            def complete(self, system_prompt, user_prompt):
                return LLMResult(
                    "<assign>Executor</assign>",
                    prompt_tokens=1,
                    completion_tokens=1,
                )

        assignment = ExecutorDispatcher(
            ExecutorDispatchConfig(
                prefer_matching_specialist=True,
                auto_assign_single=False,
            ),
            backend=KeepExecutorBackend(),
        ).assign(
            task="cool some lettuce and put it in countertop",
            task_family="pick_cool_then_place_in_recep",
            agents=[executor, cooler],
            skills=[skill],
            gamefile="/fake/pick_cool_then_place_in_recep-0/game.tw-pddl",
        )
        self.assertEqual(assignment.primary_agent, "TransformCoolSpecialist")
        self.assertIn("prefer_matching", assignment.dispatch_layer)

        # With multiple matching specialists, LLM keep-Executor is overridden.
        skill_b = Skill(
            skill_name="cool_search",
            description="Find cool targets",
            precondition="Need target",
            action_protocol=["go to"],
            applicable_atomic_ops=[AtomicOp.ACT],
            status=SkillStatus.VERIFIED,
            applicable_task_families=["pick_cool_then_place_in_recep"],
            metadata={"primary_task_family": "pick_cool_then_place_in_recep"},
        )
        searcher = AgentSpec(
            name="CoolSearcher",
            role="Search specialist",
            responsibilities=["search"],
            assigned_skills=[skill_b.skill_name],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "accepted",
                "task_families": ["pick_cool_then_place_in_recep"],
            },
        )
        multi = ExecutorDispatcher(
            ExecutorDispatchConfig(
                prefer_matching_specialist=True,
                auto_assign_single=False,
            ),
            backend=KeepExecutorBackend(),
        ).assign(
            task="cool some lettuce and put it in countertop",
            task_family="pick_cool_then_place_in_recep",
            agents=[executor, cooler, searcher],
            skills=[skill, skill_b],
            gamefile="/fake/pick_cool_then_place_in_recep-0/game.tw-pddl",
        )
        self.assertIn(
            multi.primary_agent,
            {"TransformCoolSpecialist", "CoolSearcher"},
        )
        self.assertEqual(
            multi.dispatch_layer, "prefer_matching_override_executor"
        )

    def test_shift_for_skill_cluster_uses_pre_add_baseline(self):
        baseline_skill = Skill(
            skill_name="Old",
            description="d",
            precondition="p",
            action_protocol=["a"],
            applicable_atomic_ops=[AtomicOp.ACT],
            support_count=5,
            exploration_distribution=ExplorationDistribution(
                task_family_histogram={"pick_and_place": 5},
                total_trajectories=5,
            ),
            metadata={"primary_task_family": "pick_and_place"},
        )
        baseline = aggregate_skill_bank_distribution([baseline_skill])
        new_skill = Skill(
            skill_name="New Clean",
            description="d",
            precondition="p",
            action_protocol=["a"],
            applicable_atomic_ops=[AtomicOp.ACT],
            support_count=4,
            exploration_distribution=ExplorationDistribution(
                task_family_histogram={"pick_clean_then_place_in_recep": 4},
                total_trajectories=4,
            ),
            metadata={
                "primary_task_family": "pick_clean_then_place_in_recep",
                "source_signal": "missing_clean_operation",
            },
        )
        contaminated = aggregate_skill_bank_distribution([baseline_skill, new_skill])
        honest = shift_for_skill_cluster([new_skill], baseline)
        polluted = shift_for_skill_cluster([new_skill], contaminated)
        self.assertGreater(honest.shift_score, polluted.shift_score)


class LearningCurveAndTrainSplitTests(unittest.TestCase):
    def test_build_learning_curve_series_tasks_and_steps(self) -> None:
        from sage_mas.runners.plot_learning_curve import (
            build_learning_curve_series,
        )

        rows = build_learning_curve_series(
            [
                {"won": True, "num_steps": 10},
                {"won": False, "num_steps": 20},
                {"won": True, "num_steps": 5},
            ],
            rolling_window=2,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["cumulative_tasks"], 1)
        self.assertEqual(rows[0]["cumulative_env_steps"], 10)
        self.assertAlmostEqual(rows[0]["cumulative_success_rate"], 1.0)
        self.assertEqual(rows[2]["cumulative_tasks"], 3)
        self.assertEqual(rows[2]["cumulative_env_steps"], 35)
        self.assertAlmostEqual(rows[2]["cumulative_success_rate"], 2 / 3)
        self.assertAlmostEqual(rows[2]["rolling_success_rate"], 0.5)

    def test_infer_dataset_split(self) -> None:
        from sage_mas.online_evolution import infer_dataset_split

        self.assertEqual(infer_dataset_split("/data/json_2.1.1/train/foo/game.tw-pddl"), "train")
        self.assertEqual(infer_dataset_split("/data/json_2.1.1/valid_seen/foo/game.tw-pddl"), "val")
        self.assertEqual(infer_dataset_split("/data/json_2.1.1/valid_unseen/foo/game.tw-pddl"), "test")
        self.assertEqual(infer_dataset_split("x", "valid_seen"), "val")
        self.assertEqual(infer_dataset_split("x", "valid_unseen"), "test")

    def test_build_eval_checkpoint_series(self) -> None:
        from sage_mas.runners.plot_learning_curve import build_eval_checkpoint_series

        rows = build_eval_checkpoint_series(
            [
                {"won": True, "train_progress_at_eval": 10},
                {"won": False, "train_progress_at_eval": 10},
                {"won": True, "train_progress_at_eval": 20},
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["train_progress"], 10)
        self.assertAlmostEqual(rows[0]["batch_success_rate"], 0.5)
        self.assertAlmostEqual(rows[0]["cumulative_success_rate"], 0.5)
        self.assertAlmostEqual(rows[1]["cumulative_success_rate"], 2 / 3)

    def test_plot_anchor_offsets_series_and_sets_nonzero_baseline(self) -> None:
        from sage_mas.runners.plot_learning_curve import (
            build_split_series,
        )

        splits = {
            "train": [{"won": i % 4 == 0} for i in range(30)],
            "val": [
                {"won": False, "train_progress_at_eval": 20},
                {"won": True, "train_progress_at_eval": 30},
            ],
            "test": [],
        }
        state = {
            "all_trials": splits["train"],
            "history": [
                {
                    "round": 2,
                    "status": "accepted",
                    "agent_names": ["Executor"],
                    "next_agent_names": ["Executor", "SearchSpecialist"],
                    "trial_count": 10,
                }
            ],
        }
        series, anchor = build_split_series(
            splits,
            rolling_window=2,
            state=state,
        )
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.progress, 10)
        self.assertGreater(anchor.train_sr, 0.0)
        self.assertEqual(series["train"][0]["train_progress"], 0)
        self.assertAlmostEqual(
            float(series["train"][0]["cumulative_success_rate"]),
            anchor.train_sr,
        )
        # Val/test keep real checkpoint SRs only (no synthetic train-SR point at x=0).
        self.assertEqual(series["val"][0]["train_progress"], 10)
        self.assertAlmostEqual(
            float(series["val"][0]["cumulative_success_rate"]),
            0.0,
        )
        self.assertGreater(float(series["val"][-1]["train_progress"]), 0)

    def test_plot_waits_without_add_agent(self) -> None:
        from sage_mas.runners.plot_learning_curve import build_split_series

        splits = {
            "train": [{"won": True}, {"won": False}],
            "val": [{"won": True, "train_progress_at_eval": 2}],
            "test": [],
        }
        state = {
            "all_trials": splits["train"],
            "history": [
                {
                    "round": 1,
                    "status": "accepted",
                    "agent_names": ["Executor"],
                    "next_agent_names": ["Executor"],
                    "trial_count": 2,
                    "shadow_result": "shadow.json",
                }
            ],
        }
        series, anchor = build_split_series(
            splits,
            rolling_window=2,
            state=state,
        )
        self.assertIsNone(anchor)
        self.assertEqual(series["train"], [])
        self.assertEqual(series["val"], [])
        self.assertEqual(series["test"], [])

    def test_series_vs_env_steps_aligns_val_to_train_steps(self) -> None:
        from sage_mas.runners.plot_learning_curve import series_vs_env_steps

        series = {
            "train": [
                {
                    "train_progress": 10,
                    "cumulative_tasks": 10,
                    "cumulative_env_steps": 400,
                    "cumulative_success_rate": 0.4,
                    "rolling_success_rate": 0.4,
                },
                {
                    "train_progress": 20,
                    "cumulative_tasks": 20,
                    "cumulative_env_steps": 900,
                    "cumulative_success_rate": 0.3,
                    "rolling_success_rate": 0.3,
                },
            ],
            "val": [
                {
                    "train_progress": 10,
                    "cumulative_tasks": 10,
                    "cumulative_success_rate": 0.25,
                },
                {
                    "train_progress": 20,
                    "cumulative_tasks": 20,
                    "cumulative_success_rate": 0.35,
                },
            ],
            "test": [],
        }
        remapped = series_vs_env_steps(series)
        self.assertEqual(remapped["train"][0]["x"], 400)
        self.assertEqual(remapped["train"][1]["x"], 900)
        self.assertEqual(remapped["val"][0]["x"], 400)
        self.assertEqual(remapped["val"][1]["x"], 900)

    def test_segment_added_agents_and_test_eval_gate(self) -> None:
        from sage_mas.online_evolution import OnlineAlfWorldEvolution

        outcome = {
            "status": "accepted",
            "agent_names": ["Executor"],
            "next_agent_names": ["Executor", "SearchSpecialist"],
        }
        self.assertEqual(
            OnlineAlfWorldEvolution._segment_added_agents(outcome),
            ["SearchSpecialist"],
        )
        runner = object.__new__(OnlineAlfWorldEvolution)
        runner.config = type(
            "Cfg",
            (),
            {
                "test_eval_enabled": True,
                "test_eval_every_segment": False,
                "test_eval_every_n_segments": 0,
                "val_eval_enabled": True,
                "val_eval_every_segment": True,
                "val_eval_every_n_segments": 0,
                "num_games": 60,
                "segment_size": 10,
            },
        )()
        self.assertTrue(
            OnlineAlfWorldEvolution._should_run_test_eval(
                runner, outcome, round_index=2
            )
        )
        outcome["status"] = "no_edit"
        self.assertFalse(
            OnlineAlfWorldEvolution._should_run_test_eval(
                runner, outcome, round_index=2
            )
        )
        self.assertTrue(
            OnlineAlfWorldEvolution._should_run_val_eval(
                runner, outcome, round_index=1
            )
        )
        outcome["status"] = "accepted"
        outcome["next_agent_names"] = ["Executor"]
        self.assertFalse(
            OnlineAlfWorldEvolution._should_run_test_eval(
                runner, outcome, round_index=2
            )
        )
        runner.config.test_eval_every_segment = True
        self.assertTrue(
            OnlineAlfWorldEvolution._should_run_test_eval(
                runner, outcome, round_index=2
            )
        )
        runner.config.test_eval_every_segment = False
        runner.config.test_eval_every_n_segments = 3
        self.assertFalse(
            OnlineAlfWorldEvolution._should_run_test_eval(
                runner, outcome, round_index=2
            )
        )
        self.assertTrue(
            OnlineAlfWorldEvolution._should_run_test_eval(
                runner, outcome, round_index=3
            )
        )
        self.assertTrue(
            OnlineAlfWorldEvolution._should_run_test_eval(
                runner, outcome, round_index=6
            )
        )
    def test_periodic_eval_checkpoint_uses_batch_sr(self) -> None:
        from sage_mas.runners.plot_learning_curve import (
            build_eval_checkpoint_series,
            build_split_series,
            state_uses_periodic_curve,
        )

        trials = [
            {
                "won": True,
                "train_progress_at_eval": 20,
                "segment": 1,
                "fixed_eval_pool": True,
                "phase": "periodic_val",
            },
            {
                "won": False,
                "train_progress_at_eval": 20,
                "segment": 1,
                "fixed_eval_pool": True,
                "phase": "periodic_val",
            },
            {
                "won": True,
                "train_progress_at_eval": 40,
                "segment": 2,
                "fixed_eval_pool": True,
                "phase": "periodic_val",
            },
            {
                "won": True,
                "train_progress_at_eval": 40,
                "segment": 2,
                "fixed_eval_pool": True,
                "phase": "periodic_val",
            },
        ]
        rows = build_eval_checkpoint_series(trials)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[0]["cumulative_success_rate"]), 0.5)
        self.assertAlmostEqual(float(rows[1]["cumulative_success_rate"]), 1.0)
        state = {
            "config": {
                "val_eval_every_segment": True,
                "test_eval_every_segment": True,
                "freeze_organization": True,
            },
            "all_trials": [
                {"won": True, "dataset_split": "train", "num_steps": 10},
                {"won": False, "dataset_split": "train", "num_steps": 12},
            ],
            "val_trials": trials,
            "test_trials": [],
            "history": [],
        }
        self.assertTrue(state_uses_periodic_curve(state))
        series, anchor = build_split_series(
            {
                "train": state["all_trials"],
                "val": trials,
                "test": [],
            },
            rolling_window=50,
            state=state,
        )
        self.assertIsNone(anchor)
        self.assertEqual(len(series["train"]), 2)
        self.assertEqual(len(series["val"]), 2)
    def test_load_split_trials_reconstructs_val_from_shadow(self) -> None:
        from sage_mas.runners.plot_learning_curve import load_split_trials

        state_path = Path(
            "/home/zhuyao/verl-agent/logs/sage_mas/online1000_shadow_distill_v4/online_state.json"
        )
        if not state_path.exists():
            self.skipTest("online1000_shadow_distill_v4 state not available")
        splits = load_split_trials(state_path)
        self.assertGreater(len(splits["train"]), 0)
        self.assertGreater(len(splits["val"]), 0)

    def test_sample_train_split_gamefiles(self) -> None:
        from sage_mas.alfworld_evaluator import (
            list_alfworld_split_gamefiles,
            sample_alfworld_gamefiles,
        )

        data_path = "/home/zhuyao/verl-agent/data/alfworld"
        train_root = Path(data_path) / "json_2.1.1" / "train"
        if not train_root.exists():
            self.skipTest("ALFWorld train split not available")
        train_games = list_alfworld_split_gamefiles(data_path, "train")
        self.assertGreaterEqual(len(train_games), 1000)
        sampled = sample_alfworld_gamefiles(
            data_path,
            num_tasks=1000,
            seed=1,
            split="train",
        )
        self.assertEqual(len(sampled), 1000)
        self.assertEqual(len(set(sampled)), 1000)
        sampled_again = sample_alfworld_gamefiles(
            data_path,
            num_tasks=1000,
            seed=1,
            split="train",
        )
        self.assertEqual(sampled, sampled_again)


if __name__ == "__main__":
    unittest.main()
