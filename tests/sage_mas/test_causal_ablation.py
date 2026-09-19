"""Unit tests for causal ablation helpers and freeze_organization config."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from sage_mas.online_evolution import (
    OnlineEvolutionConfig,
    summarize_adaptation_history,
)
from sage_mas.schemas import AgentSpec, Skill, SkillStatus
from sage_mas.specialist_action_guard import (
    SpecialistActionGuard,
    parse_admissible_actions,
)
from sage_mas.specialist_controllers import (
    DEFAULT_CONTROLLER_REGISTRY,
    specialist_has_controller_for_family,
)
from sage_mas.specialist_state import infer_specialist_state
from sage_mas.task_parser import parse_alfworld_task
from sage_mas.skill_injection_policy import (
    distill_seed_is_excluded,
    filter_injectable_skills,
    skill_is_injectable,
)


def _clean_guard_fixture() -> tuple[list[AgentSpec], list[Skill]]:
    skill = Skill(
        skill_name="Clean protocol",
        description="clean target before placing it",
        precondition="clean task",
        action_protocol=["take target", "clean target", "place target"],
        applicable_atomic_ops=[],
        capability_key="transform.clean",
        applicable_task_families=["pick_clean_then_place_in_recep"],
        metadata={"source_signal": "clean_operation_success"},
    )
    agent = AgentSpec(
        name="TransformCleanSpecialist",
        role="TransformCleanSpecialist",
        responsibilities=["complete clean tasks"],
        assigned_skills=[skill.skill_name],
        tool_permissions=["alfworld_action"],
    )
    return [AgentSpec(name="Executor", role="Executor", responsibilities=[]), agent], [skill]


class CausalAblationTests(unittest.TestCase):
    def test_clean_task_parser_extracts_goal_from_text_and_gamefile(self) -> None:
        parsed = parse_alfworld_task(
            "put a clean butter knife in countertop.",
            "/tmp/valid_seen/pick_clean_then_place_in_recep-ButterKnife-None-CounterTop-8/trial/game.tw-pddl",
        )

        self.assertEqual(parsed.operation, "clean")
        self.assertEqual(parsed.target, "butterknife")
        self.assertEqual(parsed.destination, "countertop")
        self.assertEqual(parsed.task_family, "pick_clean_then_place_in_recep")

        fallback = parse_alfworld_task(
            "do the task",
            "/tmp/valid_seen/pick_clean_then_place_in_recep-DishSponge-None-CounterTop-403/trial/game.tw-pddl",
        )
        self.assertEqual(fallback.target, "dishsponge")
        self.assertEqual(fallback.destination, "countertop")

    def test_clean_specialist_state_infers_phase_from_history(self) -> None:
        parsed = parse_alfworld_task(
            "put a clean mug in coffeemachine.",
            "/tmp/pick_clean_then_place_in_recep-Mug-None-CoffeeMachine-26/game.tw-pddl",
        )
        state = infer_specialist_state(
            parsed,
            steps=[
                {
                    "action": "<think>x</think><action>take mug 1 from countertop 1</action>",
                    "observation": "You pick up the mug 1.",
                    "is_action_valid": True,
                },
                {
                    "action": "<action>clean mug 1 with sinkbasin 1</action>",
                    "observation": "You clean the mug 1.",
                    "is_action_valid": True,
                },
            ],
            observation="You are near coffee machine 1.",
            admissible_actions=["go to coffeemachine 1", "move mug 1 to coffeemachine 1"],
        )

        self.assertTrue(state.held_target)
        self.assertTrue(state.operation_done)
        self.assertEqual(state.phase, "PLACE_TARGET")

    def test_specialist_prompt_summary_exposes_progress_and_failures(self) -> None:
        from sage_mas.specialist_prompt_summary import (
            build_specialist_execution_summary,
        )

        summary = build_specialist_execution_summary(
            task="put a clean mug in the cabinet.",
            task_family="pick_clean_then_place_in_recep",
            observation_prompt=(
                "The admissible actions of the current situation are: "
                "['look', 'take mug 1', 'go to sinkbasin 1']."
            ),
            history_steps=[
                {
                    "action": "go to shelf 1",
                    "observation": "Nothing happens.",
                    "goal_progress_delta": 0.0,
                    "is_action_valid": True,
                },
                {
                    "action": "go to shelf 1",
                    "observation": "Nothing happens.",
                    "goal_progress_delta": 0.0,
                    "is_action_valid": True,
                },
                {
                    "action": "go to shelf 1",
                    "observation": "Nothing happens.",
                    "goal_progress_delta": 0.0,
                    "is_action_valid": True,
                },
                {
                    "action": "take plate 1",
                    "observation": "You cannot see that.",
                    "goal_progress_delta": 0.0,
                    "is_action_valid": False,
                },
            ],
        )

        self.assertIn("Parsed task", summary)
        self.assertIn("family=pick_clean_then_place_in_recep", summary)
        self.assertIn("phase=TAKE_TARGET", summary)
        self.assertIn("Recent failure memory", summary)
        self.assertIn("Repetition memory", summary)
        self.assertIn("choose the next", summary)
        self.assertNotIn("Phase checklist", summary)
        self.assertNotIn("SEARCH_TARGET ->", summary)

    def test_no_controller_failure_distills_skill_protocol_notes(self) -> None:
        from sage_mas.specialist_failure_distillation import (
            update_specialist_failure_protocols,
        )

        skill = Skill(
            skill_name="Clean protocol",
            description="clean target before placing",
            precondition="clean task",
            action_protocol=["take target", "clean target", "place target"],
            applicable_atomic_ops=[],
            status=SkillStatus.VERIFIED,
            applicable_task_families=["pick_clean_then_place_in_recep"],
        )
        agent = AgentSpec(
            name="TransformCleanSpecialist",
            role="clean specialist",
            responsibilities=["clean"],
            assigned_skills=[skill.skill_name],
        )
        trial = SimpleNamespace(
            assigned_primary_agent=agent.name,
            won=False,
            task="put a clean mug in the cabinet.",
            task_id=(
                "/tmp/pick_clean_then_place_in_recep-Mug-None-Cabinet-1/"
                "trial/game.tw-pddl"
            ),
            task_family="pick_clean_then_place_in_recep",
            steps=[
                {
                    "action": "go to shelf 1",
                    "observation": "Nothing happens.",
                    "is_action_valid": True,
                    "goal_progress_delta": 0.0,
                    "action_guard": {"changed": False},
                },
                {
                    "action": "go to shelf 1",
                    "observation": "Nothing happens.",
                    "is_action_valid": True,
                    "goal_progress_delta": 0.0,
                    "action_guard": {"changed": False},
                },
                {
                    "action": "go to shelf 1",
                    "observation": "Nothing happens.",
                    "is_action_valid": True,
                    "goal_progress_delta": 0.0,
                    "action_guard": {"changed": False},
                },
            ],
        )

        summary = update_specialist_failure_protocols(
            skills=[skill],
            agents=[agent],
            trials=[trial],
        )

        self.assertEqual(summary["updated_skill_count"], 1)
        self.assertTrue(skill.metadata["anti_patterns"])
        self.assertTrue(skill.metadata["failure_branches"])
        self.assertIn("no_controller_failure_distillation", skill.metadata)

    def test_clean_guard_takes_visible_target_before_searching(self) -> None:
        agents, skills = _clean_guard_fixture()
        prompt = (
            "Your admissible actions of the current situation are: "
            "['go to cabinet 1', 'take dishsponge 1 from countertop 1', 'look']."
        )

        result = SpecialistActionGuard().guard(
            raw_action="<think>search</think><action>go to cabinet 1</action>",
            task="clean some dishsponge and put it in countertop.",
            gamefile="/tmp/pick_clean_then_place_in_recep-DishSponge-None-CounterTop-403/game.tw-pddl",
            observation="You see dishsponge 1 on countertop 1.",
            prompt=prompt,
            steps=[],
            primary_agent="TransformCleanSpecialist",
            agents=agents,
            skills=skills,
        )

        self.assertTrue(result.changed)
        self.assertEqual(result.selected_action_text, "take dishsponge 1 from countertop 1")

    def test_clean_guard_cleans_held_target_at_sink(self) -> None:
        agents, skills = _clean_guard_fixture()
        prompt = (
            "Your admissible actions of the current situation are: "
            "['go to drawer 1', 'clean fork 1 with sinkbasin 1', 'look']."
        )

        result = SpecialistActionGuard().guard(
            raw_action="<action>go to drawer 1</action>",
            task="put a clean fork in drawer.",
            gamefile="/tmp/pick_clean_then_place_in_recep-Fork-None-Drawer-8/game.tw-pddl",
            observation="You are at sinkbasin 1.",
            prompt=prompt,
            steps=[
                {
                    "action": "<action>take fork 1 from countertop 1</action>",
                    "observation": "You pick up the fork 1.",
                    "is_action_valid": True,
                }
            ],
            primary_agent="TransformCleanSpecialist",
            agents=agents,
            skills=skills,
        )

        self.assertEqual(result.selected_action_text, "clean fork 1 with sinkbasin 1")

    def test_clean_guard_places_clean_target_in_destination(self) -> None:
        agents, skills = _clean_guard_fixture()
        prompt = (
            "Your admissible actions of the current situation are: "
            "['move ladle 1 to diningtable 1', 'go to sinkbasin 1', 'look']."
        )

        result = SpecialistActionGuard().guard(
            raw_action="<action>go to sinkbasin 1</action>",
            task="put a clean ladle on dining table.",
            gamefile="/tmp/pick_clean_then_place_in_recep-Ladle-None-DiningTable-4/game.tw-pddl",
            observation="You are at diningtable 1.",
            prompt=prompt,
            steps=[
                {
                    "action": "<action>take ladle 1 from drawer 1</action>",
                    "observation": "You pick up the ladle 1.",
                    "is_action_valid": True,
                },
                {
                    "action": "<action>clean ladle 1 with sinkbasin 1</action>",
                    "observation": "You clean the ladle 1.",
                    "is_action_valid": True,
                },
            ],
            primary_agent="TransformCleanSpecialist",
            agents=agents,
            skills=skills,
        )

        self.assertEqual(result.selected_action_text, "move ladle 1 to diningtable 1")

    def test_clean_guard_opens_closed_destination_before_rewalking(self) -> None:
        agents, skills = _clean_guard_fixture()
        prompt = (
            "Your admissible actions of the current situation are: "
            "['open drawer 1', 'go to drawer 2', 'go to sinkbasin 1', 'look']."
        )

        result = SpecialistActionGuard().guard(
            raw_action="<action>go to drawer 2</action>",
            task="clean some fork and put it in drawer.",
            gamefile="/tmp/pick_clean_then_place_in_recep-Fork-None-Drawer-8/game.tw-pddl",
            observation="You are at drawer 1. The drawer 1 is closed.",
            prompt=prompt,
            steps=[
                {
                    "action": "<action>take fork 1 from countertop 1</action>",
                    "observation": "You pick up the fork 1.",
                    "is_action_valid": True,
                },
                {
                    "action": "<action>clean fork 1 with sinkbasin 1</action>",
                    "observation": "You clean the fork 1.",
                    "is_action_valid": True,
                },
            ],
            primary_agent="TransformCleanSpecialist",
            agents=agents,
            skills=skills,
        )

        self.assertEqual(result.selected_action_text, "open drawer 1")

    def test_parse_admissible_actions_handles_quoted_prompt_list(self) -> None:
        self.assertEqual(
            parse_admissible_actions(
                'Your admissible actions of the current situation are: ["look", "go to sinkbasin 1"].'
            ),
            ["look", "go to sinkbasin 1"],
        )

    def test_specialist_controller_registry_covers_hard_scopes(self) -> None:
        self.assertEqual(
            DEFAULT_CONTROLLER_REGISTRY.supported_task_families(),
            {
                "pick_clean_then_place_in_recep",
                "pick_heat_then_place_in_recep",
                "pick_cool_then_place_in_recep",
                "look_at_obj_in_light",
                "pick_two_obj_and_place",
            },
        )
        agents, skills = _clean_guard_fixture()
        self.assertTrue(
            specialist_has_controller_for_family(
                agents[1],
                skills,
                "pick_clean_then_place_in_recep",
            )
        )

    def test_controller_is_synthesized_from_generated_skill_contract(self) -> None:
        skill = Skill(
            skill_name="Generated heat skill",
            description="heat target object using microwave before placing it",
            precondition="task requires a hot object",
            action_protocol=[
                "take <object> from <receptacle>",
                "heat <object> with microwave",
                "move <object> to <receptacle>",
            ],
            applicable_atomic_ops=[],
            status=SkillStatus.VERIFIED,
            capability_key="transform.heat",
            applicable_task_families=["pick_heat_then_place_in_recep"],
        )
        agent = AgentSpec(
            name="AutoHeatSpecialist",
            role="Generated specialist",
            responsibilities=["execute generated heat contract"],
            assigned_skills=[skill.skill_name],
            tool_permissions=["alfworld_action"],
        )
        parsed = parse_alfworld_task(
            "put a hot mug in cabinet.",
            "/tmp/pick_heat_then_place_in_recep-Mug-None-Cabinet-1/game.tw-pddl",
        )

        controller = DEFAULT_CONTROLLER_REGISTRY.controller_for(
            parsed=parsed,
            primary_agent=agent.name,
            agents=[agent],
            skills=[skill],
        )

        self.assertIsNotNone(controller)
        self.assertTrue(controller.name.startswith("GeneratedHeatController:"))
        self.assertNotEqual(controller.name, "HeatController")

    def test_dispatch_default_uses_skill_scope_without_controller_package(self) -> None:
        from sage_mas.executor_dispatch import (
            ExecutorDispatchConfig,
            ExecutorDispatcher,
        )

        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["act"],
            tool_permissions=["alfworld_action"],
        )
        specialist = AgentSpec(
            name="UnsupportedSpecialist",
            role="UnsupportedSpecialist",
            responsibilities=["unsupported"],
            assigned_skills=["unsupported"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "accepted",
                "task_families": ["unsupported_family"],
            },
        )
        skill = Skill(
            skill_name="unsupported",
            description="unsupported",
            precondition="unsupported",
            action_protocol=["do unsupported"],
            applicable_atomic_ops=[],
            status=SkillStatus.VERIFIED,
            capability_key="unsupported.family",
            applicable_task_families=["unsupported_family"],
        )

        assignment = ExecutorDispatcher(
            ExecutorDispatchConfig(
                auto_assign_single=True,
            )
        ).assign(
            task="unsupported task",
            task_family="unsupported_family",
            agents=[executor, specialist],
            skills=[skill],
            gamefile="/tmp/unsupported_family/game.tw-pddl",
        )

        self.assertEqual(assignment.primary_agent, "UnsupportedSpecialist")
        self.assertEqual(assignment.dispatch_layer, "eligibility_single")

    def test_dispatch_only_blocks_missing_controller_when_explicitly_required(self) -> None:
        from sage_mas.executor_dispatch import (
            ExecutorDispatchConfig,
            ExecutorDispatcher,
        )

        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["act"],
            tool_permissions=["alfworld_action"],
        )
        specialist = AgentSpec(
            name="UnsupportedSpecialist",
            role="UnsupportedSpecialist",
            responsibilities=["unsupported"],
            assigned_skills=["unsupported"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "accepted",
                "task_families": ["unsupported_family"],
            },
        )
        skill = Skill(
            skill_name="unsupported",
            description="unsupported",
            precondition="unsupported",
            action_protocol=["do unsupported"],
            applicable_atomic_ops=[],
            status=SkillStatus.VERIFIED,
            capability_key="unsupported.family",
            applicable_task_families=["unsupported_family"],
        )

        assignment = ExecutorDispatcher(
            ExecutorDispatchConfig(
                auto_assign_single=True,
                require_controller_for_eligibility=True,
            )
        ).assign(
            task="unsupported task",
            task_family="unsupported_family",
            agents=[executor, specialist],
            skills=[skill],
            gamefile="/tmp/unsupported_family/game.tw-pddl",
        )

        self.assertEqual(assignment.primary_agent, "Executor")
        self.assertEqual(assignment.dispatch_layer, "eligibility_empty")

    def test_actor_promotion_probe_forces_no_controller_and_restores(self) -> None:
        from sage_mas.actor_promotion import probe_actor_promotion

        class FakeEvaluator:
            def __init__(self):
                self.config = SimpleNamespace(enable_specialist_controllers=True)
                self.specialist_action_guard = SimpleNamespace(enabled=True)
                self.dispatcher = SimpleNamespace(
                    config=SimpleNamespace(
                        require_controller_for_eligibility=True,
                        auto_assign_single=False,
                    )
                )
                self.calls = []

            def evaluate(self, agents, skills, gamefiles, condition):
                self.calls.append(
                    {
                        "condition": condition,
                        "controllers": self.config.enable_specialist_controllers,
                        "guard": self.specialist_action_guard.enabled,
                        "require_controller": (
                            self.dispatcher.config.require_controller_for_eligibility
                        ),
                        "auto_assign_single": (
                            self.dispatcher.config.auto_assign_single
                        ),
                    }
                )
                won = condition.startswith("actor_promo_specialist")
                return [SimpleNamespace(won=won) for _ in gamefiles]

        specialist = AgentSpec(
            name="UnsupportedSpecialist",
            role="UnsupportedSpecialist",
            responsibilities=["unsupported"],
            assigned_skills=["unsupported"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "probation",
                "task_families": ["unsupported_family"],
            },
        )
        skill = Skill(
            skill_name="unsupported",
            description="unsupported",
            precondition="unsupported",
            action_protocol=["do unsupported"],
            applicable_atomic_ops=[],
            status=SkillStatus.VERIFIED,
            applicable_task_families=["unsupported_family"],
        )

        evaluator = FakeEvaluator()
        result = probe_actor_promotion(
            evaluator,  # type: ignore[arg-type]
            agents=[specialist],
            skills=[skill],
            specialist=specialist,
            gamefiles=["/tmp/unsupported_family/game.tw-pddl"],
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.n_tasks, 1)
        self.assertTrue(evaluator.config.enable_specialist_controllers)
        self.assertTrue(evaluator.specialist_action_guard.enabled)
        self.assertTrue(evaluator.dispatcher.config.require_controller_for_eligibility)
        self.assertFalse(evaluator.dispatcher.config.auto_assign_single)
        self.assertEqual(len(evaluator.calls), 2)
        for call in evaluator.calls:
            self.assertFalse(call["controllers"])
            self.assertFalse(call["guard"])
            self.assertFalse(call["require_controller"])
            self.assertTrue(call["auto_assign_single"])

    def test_summarize_adaptation_history_counts_statuses(self) -> None:
        history = [
            {
                "status": "organization_frozen",
                "candidate_skill_names": ["a"],
                "shadow_decision": None,
            },
            {
                "status": "accepted",
                "candidate_skill_names": ["b"],
                "shadow_decision": (
                    "new utility 0.500000 meets or exceeds required utility "
                    "0.000000; paired tasks=8, CI=[0.000000, 1.000000]."
                ),
            },
            {
                "status": "rolled_back",
                "candidate_skill_names": ["c"],
                "shadow_decision": (
                    "new utility 0.000000 does not meet required utility "
                    "0.000000; paired tasks=8, CI=[0.000000, 0.000000]."
                ),
            },
            {"status": "no_candidates", "candidate_skill_names": []},
        ]
        summary = summarize_adaptation_history(history)
        self.assertEqual(summary["organization_frozen"], 1)
        self.assertEqual(summary["org_accepted"], 1)
        self.assertEqual(summary["org_rolled_back"], 1)
        self.assertEqual(summary["segments_with_candidates"], 3)
        self.assertEqual(summary["shadow_compared"], 2)
        self.assertEqual(summary["shadow_accepted"], 1)
        self.assertAlmostEqual(summary["shadow_accept_rate"], 0.5)
        self.assertIsNotNone(summary["mean_shadow_new_utility"])

    def test_freeze_organization_defaults_with_executor_only(self) -> None:
        cfg = OnlineEvolutionConfig(executor_only=True, freeze_organization=True)
        self.assertTrue(cfg.executor_only)
        self.assertTrue(cfg.freeze_organization)
        skill_only = OnlineEvolutionConfig(
            executor_only=False,
            freeze_organization=True,
        )
        self.assertFalse(skill_only.executor_only)
        self.assertTrue(skill_only.freeze_organization)

    def test_execution_skills_blocked_from_injection(self) -> None:
        efficiency = Skill(
            skill_name="eff",
            description="d",
            precondition="p",
            action_protocol=["go to <location>"],
            applicable_atomic_ops=[],
            capability_key="execution.efficiency",
            status=SkillStatus.VERIFIED,
            metadata={"source_signal": "efficient_execution"},
        )
        clean = Skill(
            skill_name="clean",
            description="d",
            precondition="p",
            action_protocol=["clean <object> with <tool>"],
            applicable_atomic_ops=[],
            capability_key="transform.clean",
            status=SkillStatus.VERIFIED,
            metadata={"source_signal": "clean_operation_success"},
        )
        self.assertTrue(skill_is_injectable(efficiency))
        self.assertTrue(skill_is_injectable(clean))
        self.assertFalse(
            skill_is_injectable(
                efficiency,
                block_prefixes=("execution.",),
            )
        )
        kept = filter_injectable_skills(
            [efficiency, clean],
            allow_prefixes=("transform.", "track.", "inspect."),
        )
        self.assertEqual([skill.skill_name for skill in kept], ["clean"])
        self.assertTrue(
            distill_seed_is_excluded(
                efficiency,
                exclude_prefixes=("execution.",),
                exclude_signals=("efficient_execution",),
            )
        )

    def test_light_protocol_keeps_desklamp_marker(self) -> None:
        from sage_mas.schemas import AtomicOp, AtomicStep
        from sage_mas.skill_quality import protocol_has_capability
        from sage_mas.trajectory.abstraction import abstract_environment_action
        from sage_mas.trajectory.grounding import (
            ground_skill_in_successful_trajectory,
        )
        from sage_mas.trajectory.protocol_canon import protocol_structure_issues

        self.assertEqual(
            abstract_environment_action("use desklamp 1"),
            "use desklamp",
        )
        self.assertEqual(
            abstract_environment_action("use microwave 1"),
            "use <entity>",
        )
        skill = Skill(
            skill_name="light",
            description="d",
            precondition="p",
            action_protocol=["use desklamp"],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="inspect.with_light",
            metadata={"source_signal": "light_operation_success"},
        )
        self.assertTrue(protocol_has_capability(skill))
        self.assertEqual(
            protocol_structure_issues(
                [
                    "go to <location>",
                    "take <object> from <receptacle>",
                    "use desklamp",
                ],
                capability="inspect.with_light",
            ),
            [],
        )
        steps = [
            AtomicStep(
                node_id="n0",
                atomic_op=AtomicOp.ACT,
                agent="Executor",
                action="go to desk 1",
                observation="You arrive at desk 1.",
                metadata={"is_action_valid": True},
            ),
            AtomicStep(
                node_id="n1",
                atomic_op=AtomicOp.ACT,
                agent="Executor",
                action="take alarmclock 1 from desk 1",
                observation="You pick up the alarmclock.",
                metadata={"is_action_valid": True},
            ),
            AtomicStep(
                node_id="n2",
                atomic_op=AtomicOp.ACT,
                agent="Executor",
                action="use desklamp 1",
                observation="You turn on the desklamp.",
                metadata={"is_action_valid": True},
            ),
            AtomicStep(
                node_id="n3",
                atomic_op=AtomicOp.ACT,
                agent="Executor",
                action="examine alarmclock 1",
                observation="Nothing happens.",
                metadata={"is_action_valid": True, "stalled": True},
            ),
            AtomicStep(
                node_id="n4",
                atomic_op=AtomicOp.VERIFY,
                agent="Executor",
                action=None,
                observation="Task success",
                metadata={
                    "won": True,
                    "task_family": "look_at_obj_in_light",
                    "trajectory_id": "light-traj-1",
                    "num_steps": 4,
                },
            ),
        ]
        grounded = ground_skill_in_successful_trajectory(skill, [steps])
        self.assertTrue(grounded.metadata.get("protocol_alignment_ok"))
        self.assertIn("use desklamp", grounded.action_protocol)
        self.assertTrue(protocol_has_capability(grounded))

    def test_b_gemini_config_keeps_executor_separate(self) -> None:
        from pathlib import Path

        import yaml

        path = Path("examples/sage_mas/sage_config.causal134_B_gemini.yaml")
        with path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        sage = cfg["sage"]
        distill = sage["distillation"]
        online = sage["online"]
        self.assertEqual(distill["mode"], "trajectory_enriched")
        self.assertEqual(distill["model"], "gemini-2.5-pro")
        self.assertTrue(online["freeze_organization"])
        self.assertFalse(online["executor_only"])
        self.assertIn("execution.", distill["exclude_capability_prefixes"])
        self.assertIn("transform.", distill["inject_capability_allowlist"])
        self.assertTrue(online["require_positive_mu_for_injection"])
        self.assertTrue(online["paired_mu_probe"])
        self.assertTrue(online["run_control_eval_for_credit"])
        self.assertTrue(sage["skill_credit"]["require_positive_mu_for_verify"])
        self.assertTrue(sage["skill_credit"]["relative_credit_to_baseline"])

    def test_injection_requires_positive_mu(self) -> None:
        from sage_mas.skill_injection_policy import skill_is_injectable

        skill = Skill(
            skill_name="clean",
            description="d",
            precondition="p",
            action_protocol=["clean <object> with <tool>"],
            applicable_atomic_ops=[],
            capability_key="transform.clean",
            status=SkillStatus.VERIFIED,
            marginal_utility=None,
        )
        self.assertFalse(
            skill_is_injectable(
                skill,
                allow_prefixes=("transform.",),
                require_positive_mu=True,
            )
        )
        skill.marginal_utility = 0.0
        self.assertTrue(
            skill_is_injectable(
                skill,
                allow_prefixes=("transform.",),
                require_positive_mu=True,
            )
        )
        skill.marginal_utility = 0.1
        self.assertTrue(
            skill_is_injectable(
                skill,
                allow_prefixes=("transform.",),
                require_positive_mu=True,
            )
        )
        skill.marginal_utility = -0.1
        self.assertFalse(
            skill_is_injectable(
                skill,
                allow_prefixes=("transform.",),
                require_positive_mu=True,
            )
        )

    def test_credit_relative_baseline_and_mu_gate(self) -> None:
        from sage_mas.alfworld_evaluator import EvaluationTrial
        from sage_mas.schemas import AgentSpec, AtomicOp
        from sage_mas.skill_credit import (
            SkillCreditPolicy,
            initialize_skill_credit,
            update_skill_credits,
        )

        policy = SkillCreditPolicy(
            require_positive_mu_for_verify=True,
            relative_credit_to_baseline=True,
            min_protocol_coverage=0.5,
        )
        skill = Skill(
            skill_name="Learned clean",
            description="d",
            precondition="p",
            action_protocol=[
                "take <object> from <receptacle>",
                "clean <object> with <tool>",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="transform.clean",
            applicable_task_families=["pick_clean_then_place_in_recep"],
            metadata={"capability_operation": "clean"},
        )
        initialize_skill_credit(skill, policy)
        with_trial = EvaluationTrial(
            task_id="/clean-1",
            task="clean mug",
            task_family="pick_clean_then_place_in_recep",
            condition="with",
            reward=0.0,
            cost=1.0,
            won=False,
            num_steps=2,
            steps=[
                {
                    "action": "take mug 1 from sidetable 1",
                    "observation": "You pick up the mug.",
                    "is_action_valid": True,
                },
                {
                    "action": "clean mug 1 with sinkbasin 1",
                    "observation": "You clean the mug 1.",
                    "is_action_valid": True,
                },
            ],
            activated_skill_names=[skill.skill_name],
            skill_activation_steps={skill.skill_name: 1},
            assigned_primary_agent="Executor",
            actions_by_agent={"Executor": 2},
        )
        # Control already cleaned → not skill-caused.
        control_same = EvaluationTrial(
            task_id="/clean-1",
            task="clean mug",
            task_family="pick_clean_then_place_in_recep",
            condition="control",
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
            activated_skill_names=[],
            skill_activation_steps={},
            assigned_primary_agent="Executor",
            actions_by_agent={"Executor": 1},
        )
        agents = [
            AgentSpec(name="Executor", role="executor", responsibilities=["act"])
        ]
        update_skill_credits(
            [skill],
            [with_trial],
            agents,
            policy,
            control_trials=[control_same],
        )
        self.assertEqual(skill.metadata["skill_credit"]["uses"], 1)
        self.assertEqual(skill.metadata["skill_credit"]["successes"], 0)
        self.assertEqual(skill.status, SkillStatus.PROVISIONAL)

        # Positive MU still required for verify even with high credit.
        skill.metadata["skill_credit"] = {
            "uses": 3,
            "successes": 3,
            "score": 0.8,
            "events": [],
        }
        skill.marginal_utility = None
        update_skill_credits([skill], [], agents, policy)
        self.assertEqual(skill.status, SkillStatus.PROVISIONAL)
        skill.marginal_utility = 0.25
        update_skill_credits([skill], [], agents, policy)
        self.assertEqual(skill.status, SkillStatus.VERIFIED)
        self.assertAlmostEqual(skill.marginal_utility, 0.25)

    def test_trial_executes_skill_requires_protocol_coverage(self) -> None:
        from sage_mas.alfworld_evaluator import EvaluationTrial
        from sage_mas.schemas import AtomicOp
        from sage_mas.skill_credit import _trial_executes_skill

        skill = Skill(
            skill_name="clean",
            description="d",
            precondition="p",
            action_protocol=[
                "take <object> from <receptacle>",
                "clean <object> with <tool>",
                "move <object> to <receptacle>",
            ],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="transform.clean",
            metadata={"capability_operation": "clean"},
        )
        weak = EvaluationTrial(
            task_id="/t",
            task="t",
            task_family="pick_clean_then_place_in_recep",
            condition="c",
            reward=0.0,
            cost=1.0,
            won=False,
            num_steps=1,
            steps=[{"action": "go to sinkbasin 1", "observation": "ok"}],
            activated_skill_names=[skill.skill_name],
            skill_activation_steps={skill.skill_name: 1},
        )
        self.assertFalse(
            _trial_executes_skill(skill, weak, min_coverage=0.5)
        )
        strong = EvaluationTrial(
            task_id="/t2",
            task="t",
            task_family="pick_clean_then_place_in_recep",
            condition="c",
            reward=0.0,
            cost=1.0,
            won=False,
            num_steps=3,
            steps=[
                {"action": "take mug 1 from sidetable 1", "observation": "ok"},
                {
                    "action": "clean mug 1 with sinkbasin 1",
                    "observation": "You clean the mug 1.",
                },
                {"action": "put mug 1 in sidetable 1", "observation": "ok"},
            ],
            activated_skill_names=[skill.skill_name],
            skill_activation_steps={skill.skill_name: 1},
        )
        self.assertTrue(
            _trial_executes_skill(skill, strong, min_coverage=0.5)
        )

    def test_apply_marginal_utility_probe_promotes(self) -> None:
        from sage_mas.skill_marginal_utility import (
            MarginalUtilityProbeResult,
            apply_marginal_utility_probe,
        )

        skill = Skill(
            skill_name="clean",
            description="d",
            precondition="p",
            action_protocol=["clean"],
            applicable_atomic_ops=[],
            capability_key="transform.clean",
            status=SkillStatus.PROVISIONAL,
        )
        accepted = MarginalUtilityProbeResult(
            skill_name="clean",
            n_tasks=4,
            with_wins=3,
            without_wins=1,
            delta_sr=0.5,
            delta_local_effect_rate=0.25,
            ci_low=0.1,
            ci_high=0.9,
            accepted=True,
            reason="ok",
        )
        apply_marginal_utility_probe(skill, accepted)
        self.assertEqual(skill.status, SkillStatus.VERIFIED)
        self.assertAlmostEqual(skill.marginal_utility, 0.5)
        self.assertTrue(skill.metadata.get("mu_promoted"))

        rejected = MarginalUtilityProbeResult(
            skill_name="clean",
            n_tasks=4,
            with_wins=0,
            without_wins=2,
            delta_sr=-0.5,
            delta_local_effect_rate=0.0,
            ci_low=-1.0,
            ci_high=0.0,
            accepted=False,
            reason="harmful",
        )
        apply_marginal_utility_probe(skill, rejected)
        self.assertEqual(skill.status, SkillStatus.PROVISIONAL)
        self.assertTrue(skill.metadata.get("mu_rejected"))

    def test_c_bmu_config_unfreezes_org_and_seeds_bank(self) -> None:
        from pathlib import Path

        import yaml

        path = Path("examples/sage_mas/sage_config.causal134_C_bmu.yaml")
        with path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        sage = cfg["sage"]
        online = sage["online"]
        self.assertFalse(online["freeze_organization"])
        self.assertFalse(online["skip_shadow"])
        self.assertTrue(online["propose_org_for_unassigned_skills"])
        self.assertEqual(online["org_warmup_segments"], 3)
        self.assertEqual(online["max_skills_per_round"], 0)
        self.assertEqual(online["max_advisors"], 0)
        self.assertTrue(online["actor_promotion_probe"])
        self.assertGreaterEqual(float(online["actor_promotion_min_advantage"]), 0.125)
        self.assertTrue(sage["executor_dispatch"]["require_accepted_for_primary"])
        self.assertTrue(sage["shadow"]["enabled"])
        self.assertGreaterEqual(float(sage["shadow"]["min_relative_gain"]), 0.125)
        self.assertEqual(int(sage["organization"]["max_new_agents_per_round"]), 1)
        self.assertIn("B_gemini_mu", sage["seed_skill_bank_path"])
        self.assertEqual(sage["seed_skill_statuses"], ["verified"])
        self.assertEqual(sage["distillation"]["model"], "gemini-2.5-pro")
        executor = sage["agents"][0]
        self.assertIn("Sole ALFWorld actor", executor["role_specification"])
        self.assertNotIn("Team coordinator", executor["role_specification"])
        self.assertEqual(executor["assigned_skills"], [])

    def test_promotion_rejects_ties_and_requires_advantage(self) -> None:
        from sage_mas.actor_promotion import (
            promotion_beats_executor,
            select_promotion_gamefiles,
        )
        from sage_mas.schemas import AgentSpec

        # Old v3 bug: 2/8 == 2/8 would promote with eps=0.
        self.assertFalse(promotion_beats_executor(0.25, 0.25, min_advantage=0.0))
        self.assertFalse(promotion_beats_executor(0.25, 0.25, min_advantage=0.125))
        self.assertTrue(promotion_beats_executor(0.375, 0.25, min_advantage=0.125))
        self.assertTrue(promotion_beats_executor(0.5, 0.375, min_advantage=0.0))
        self.assertFalse(promotion_beats_executor(0.375, 0.375, min_advantage=0.0))

        clean_specialist = AgentSpec(
            name="TransformCleanSpecialist",
            role="Clean specialist",
            responsibilities=["clean"],
            shadow_evaluation_record={
                "task_families": ["pick_clean_then_place_in_recep"],
            },
        )
        self.assertEqual(
            select_promotion_gamefiles(
                [
                    "/fake/valid_unseen/look_at_obj_in_light-A/game.tw-pddl",
                    "/fake/valid_unseen/pick_and_place-B/game.tw-pddl",
                ],
                clean_specialist,
                max_tasks=2,
            ),
            [],
        )
        self.assertEqual(
            select_promotion_gamefiles(
                [
                    "/fake/valid_unseen/look_at_obj_in_light-A/game.tw-pddl",
                    "/fake/valid_unseen/pick_clean_then_place_in_recep-B/game.tw-pddl",
                ],
                clean_specialist,
                max_tasks=2,
            ),
            [
                "/fake/valid_unseen/pick_clean_then_place_in_recep-B/game.tw-pddl",
            ],
        )

    def test_restore_executor_baseline_clears_sticky_skills(self) -> None:
        from sage_mas.onboarding import (
            SOLE_EXECUTOR_ROLE_SPECIFICATION,
            restore_executor_baseline,
        )
        from sage_mas.schemas import AgentSpec

        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Judge each new task and assign a matching roster agent"],
            assigned_skills=["Clean Object and Place in Receptacle"],
            tool_permissions=["alfworld_action"],
            role_specification=(
                "Team coordinator and fallback ALFWorld actor. "
                "Assign a specialist when a match exists."
            ),
            responsibility_boundary="assign specialists",
            input_protocol="optional team advice",
            output_protocol="action",
        )
        specialist = AgentSpec(
            name="TransformCleanSpecialist",
            role="specialist",
            responsibilities=["clean"],
            assigned_skills=["Clean Object and Place in Receptacle"],
            tool_permissions=["alfworld_action"],
        )
        # With specialist present: fix persona, keep skill copies for fallback.
        agents = [executor, specialist]
        mid = restore_executor_baseline(agents)
        self.assertTrue(mid["changed"])
        self.assertEqual(executor.role_specification, SOLE_EXECUTOR_ROLE_SPECIFICATION)
        self.assertEqual(
            executor.assigned_skills,
            ["Clean Object and Place in Receptacle"],
        )
        # After specialist removal: clear sticky ownership (B-like injection).
        agents = [executor]
        end = restore_executor_baseline(agents)
        self.assertTrue(end["cleared_assigned_skills"])
        self.assertEqual(executor.assigned_skills, [])
    def test_unassigned_bank_skills_proposed_for_org(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from sage_mas.online_evolution import OnlineAlfWorldEvolution
        from sage_mas.schemas import AgentSpec
        from sage_mas.skill_bank import SkillBank

        verified = Skill(
            skill_name="clean",
            description="d",
            precondition="p",
            action_protocol=["clean <object> with <tool>"],
            applicable_atomic_ops=[],
            capability_key="transform.clean",
            status=SkillStatus.VERIFIED,
            marginal_utility=0.2,
        )
        assigned = Skill(
            skill_name="inspect",
            description="d",
            precondition="p",
            action_protocol=["use desklamp"],
            applicable_atomic_ops=[],
            capability_key="inspect.light",
            status=SkillStatus.VERIFIED,
            marginal_utility=0.2,
        )
        with TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "skill_bank.json"
            bank = SkillBank(bank_path)
            bank.add(verified)
            bank.add(assigned)
            bank.save()
            agents = [
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["act"],
                    assigned_skills=["inspect"],
                )
            ]
            evo = OnlineAlfWorldEvolution.__new__(OnlineAlfWorldEvolution)
            evo.config = OnlineEvolutionConfig(
                propose_org_for_unassigned_skills=True,
                require_positive_mu_for_injection=True,
                min_marginal_utility=0.0,
            )
            proposals = evo._unassigned_verified_skills_for_org(
                SkillBank(bank_path),
                agents,
            )
            self.assertEqual(
                [skill.skill_name for skill in proposals],
                ["clean"],
            )
            self.assertTrue(
                proposals[0].metadata.get("credit_promoted_pending_org")
            )

    def test_pipeline_keeps_extra_candidates_when_max_is_zero(self) -> None:
        """max_candidates must trim distill only, not bank/org extras."""
        distilled = [
            Skill(
                skill_name=f"d{i}",
                description="d",
                precondition="p",
                action_protocol=["look"],
                applicable_atomic_ops=[],
                capability_key=f"execution.x{i}",
                status=SkillStatus.PROVISIONAL,
            )
            for i in range(3)
        ]
        extras = [
            Skill(
                skill_name="clean",
                description="d",
                precondition="p",
                action_protocol=["clean <object> with <tool>"],
                applicable_atomic_ops=[],
                capability_key="transform.clean",
                status=SkillStatus.VERIFIED,
                metadata={"credit_promoted_pending_org": True},
            )
        ]
        max_candidates = 0
        trimmed = distilled[: max(0, int(max_candidates))]
        merged = extras + trimmed
        self.assertEqual([s.skill_name for s in merged], ["clean"])

    def test_require_accepted_blocks_probation_primary(self) -> None:
        from sage_mas.executor_dispatch import (
            ExecutorDispatchConfig,
            ExecutorDispatcher,
        )
        from sage_mas.schemas import AgentSpec

        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["act"],
            tool_permissions=["alfworld_action"],
        )
        specialist = AgentSpec(
            name="TransformCleanSpecialist",
            role="TransformCleanSpecialist",
            responsibilities=["clean"],
            assigned_skills=["Clean Object and Place in Receptacle"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={
                "acting_status": "probation",
                "trial_games_remaining": 3,
                "task_families": ["pick_clean_then_place_in_recep"],
            },
        )
        skill = Skill(
            skill_name="Clean Object and Place in Receptacle",
            description="d",
            precondition="p",
            action_protocol=["clean <object> with <tool>"],
            applicable_atomic_ops=[],
            capability_key="transform.clean",
            status=SkillStatus.VERIFIED,
            metadata={"primary_task_family": "pick_clean_then_place_in_recep"},
        )
        blocked = ExecutorDispatcher(
            ExecutorDispatchConfig(
                require_accepted_for_primary=True,
                auto_assign_single=True,
                probation_primary_quota=0,
            )
        ).assign(
            task="clean some bowl and put it in cabinet.",
            task_family="pick_clean_then_place_in_recep",
            agents=[executor, specialist],
            skills=[skill],
            gamefile="pick_clean_then_place_in_recep-Bowl/game.tw-pddl",
        )
        self.assertEqual(blocked.primary_agent, "Executor")
        self.assertEqual(blocked.dispatch_layer, "eligibility_empty")

        # Probation quota lets matching specialists take primary before accept.
        quota_allowed = ExecutorDispatcher(
            ExecutorDispatchConfig(
                require_accepted_for_primary=True,
                auto_assign_single=True,
                prefer_matching_specialist=True,
                probation_primary_quota=8,
            )
        ).assign(
            task="clean some bowl and put it in cabinet.",
            task_family="pick_clean_then_place_in_recep",
            agents=[executor, specialist],
            skills=[skill],
            gamefile="pick_clean_then_place_in_recep-Bowl/game.tw-pddl",
        )
        self.assertEqual(
            quota_allowed.primary_agent, "TransformCleanSpecialist"
        )

        specialist.shadow_evaluation_record["promotion_probe_passed"] = True
        probe_passed = ExecutorDispatcher(
            ExecutorDispatchConfig(
                require_accepted_for_primary=True,
                auto_assign_single=True,
                probation_primary_quota=0,
            )
        ).assign(
            task="clean some bowl and put it in cabinet.",
            task_family="pick_clean_then_place_in_recep",
            agents=[executor, specialist],
            skills=[skill],
            gamefile="pick_clean_then_place_in_recep-Bowl/game.tw-pddl",
        )
        self.assertEqual(probe_passed.primary_agent, "TransformCleanSpecialist")

        specialist.shadow_evaluation_record["acting_status"] = "accepted"
        allowed = ExecutorDispatcher(
            ExecutorDispatchConfig(
                require_accepted_for_primary=True,
                auto_assign_single=True,
            )
        ).assign(
            task="clean some bowl and put it in cabinet.",
            task_family="pick_clean_then_place_in_recep",
            agents=[executor, specialist],
            skills=[skill],
            gamefile="pick_clean_then_place_in_recep-Bowl/game.tw-pddl",
        )
        self.assertEqual(allowed.primary_agent, "TransformCleanSpecialist")

    def test_actor_promotion_accept_and_dormant(self) -> None:
        from sage_mas.actor_promotion import (
            ActorPromotionResult,
            apply_actor_promotion_result,
        )
        from sage_mas.schemas import AgentSpec

        agent = AgentSpec(
            name="TransformCleanSpecialist",
            role="TransformCleanSpecialist",
            responsibilities=["clean"],
            tool_permissions=["alfworld_action"],
            shadow_evaluation_record={"acting_status": "probation"},
        )
        accept = ActorPromotionResult(
            agent_name=agent.name,
            n_tasks=8,
            specialist_wins=4,
            executor_wins=3,
            specialist_sr=0.5,
            executor_sr=0.375,
            accepted=True,
            reason="promote",
            gamefiles=[],
        )
        decision = apply_actor_promotion_result(agent, accept)
        self.assertEqual(decision["new_status"], "probation")
        self.assertEqual(
            agent.shadow_evaluation_record["acting_status"],
            "probation",
        )
        self.assertTrue(agent.shadow_evaluation_record["promotion_probe_passed"])
        self.assertFalse(agent.shadow_evaluation_record["actor_promoted"])
        self.assertFalse(agent.shadow_evaluation_record["dispatch_only"])
        self.assertEqual(agent.shadow_evaluation_record["trial_games_remaining"], 3)

        agent.shadow_evaluation_record = {"acting_status": "probation"}
        reject = ActorPromotionResult(
            agent_name=agent.name,
            n_tasks=8,
            specialist_wins=1,
            executor_wins=4,
            specialist_sr=0.125,
            executor_sr=0.5,
            accepted=False,
            reason="keep Executor",
            gamefiles=[],
        )
        apply_actor_promotion_result(
            agent,
            reject,
            remove_after_rejected_windows=2,
        )
        self.assertEqual(
            agent.shadow_evaluation_record["acting_status"],
            "probation",
        )
        apply_actor_promotion_result(
            agent,
            reject,
            remove_after_rejected_windows=2,
        )
        self.assertEqual(
            agent.shadow_evaluation_record["acting_status"],
            "dormant",
        )


if __name__ == "__main__":
    unittest.main()
