"""Tests for simplified multi-win detour-clean skill distillation."""

from __future__ import annotations

import unittest

from sage_mas.enriched_skill_distiller import EnrichedSkillDistiller
from sage_mas.schemas import SkillStatus
from sage_mas.skill_credit import SkillCreditPolicy, update_skill_credits
from sage_mas.skill_distiller import DistillationConfig
from sage_mas.skill_protocol_distill import (
    build_protocol_from_successes,
    initialize_provisional_utility,
    remove_detours_from_trajectory,
)
from sage_mas.trajectory_adapter import AlfWorldTrajectoryAdapter


def _heat_win(*, game_id: str, with_stove_detour: bool) -> dict:
    steps = [
        {
            "observation": "You open the fridge.",
            "action": "open fridge 1",
            "is_action_valid": True,
        },
        {
            "observation": "You pick up the potato.",
            "action": "take potato 1 from fridge 1",
            "is_action_valid": True,
        },
    ]
    if with_stove_detour:
        steps.extend(
            [
                {
                    "observation": "Nothing happens.",
                    "action": "examine stoveburner 1",
                    "is_action_valid": True,
                    "stalled": True,
                },
                {
                    "observation": "Nothing happens.",
                    "action": "go to stoveburner 1",
                    "is_action_valid": True,
                    "stalled": True,
                },
            ]
        )
    steps.extend(
        [
            {
                "observation": "You arrive at the microwave.",
                "action": "go to microwave 1",
                "is_action_valid": True,
            },
            {
                "observation": "You open the microwave.",
                "action": "open microwave 1",
                "is_action_valid": True,
            },
            {
                "observation": "You heat the potato using the microwave.",
                "action": "heat potato 1 with microwave 1",
                "is_action_valid": True,
            },
            {
                "observation": "You arrive at the garbagecan.",
                "action": "go to garbagecan 1",
                "is_action_valid": True,
            },
            {
                "observation": "You move the potato.",
                "action": "move potato 1 to garbagecan 1",
                "is_action_valid": True,
            },
        ]
    )
    return {
        "gamefile": (
            f"/tmp/pick_heat_then_place_in_recep-Potato-None-GarbageCan-"
            f"{game_id}/game.tw-pddl"
        ),
        "task": "heat some potato and put it in garbagecan",
        "won": True,
        "num_steps": len(steps),
        "steps": steps,
    }


class MergedProtocolDistillTests(unittest.TestCase):
    def test_remove_detours_drops_examine_and_noop(self) -> None:
        adapted = AlfWorldTrajectoryAdapter().adapt_many(
            [_heat_win(game_id="1", with_stove_detour=True)]
        )
        protocol = remove_detours_from_trajectory(
            adapted[0],
            capability="transform.heat",
        )
        joined = " | ".join(protocol)
        self.assertNotIn("examine", joined)
        self.assertTrue(
            any(step.startswith("heat ") and " with " in step for step in protocol),
            msg=joined,
        )
        self.assertTrue(
            any(step.startswith("move ") and " to " in step for step in protocol),
            msg=joined,
        )

    def test_merge_ignores_single_trace_detour(self) -> None:
        adapted = AlfWorldTrajectoryAdapter().adapt_many(
            [
                _heat_win(game_id="1", with_stove_detour=True),
                _heat_win(game_id="2", with_stove_detour=False),
            ]
        )
        from sage_mas.schemas import AtomicOp, Skill

        seed = Skill(
            skill_name="Observed heat",
            description="heat",
            precondition="heat",
            action_protocol=[],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="transform.heat",
            metadata={
                "source_signal": "heat_operation_success",
                "primary_task_family": "pick_heat_then_place_in_recep",
            },
        )
        skill = build_protocol_from_successes(seed, adapted, min_wins=1)
        self.assertTrue(skill.metadata.get("protocol_alignment_ok"))
        self.assertNotEqual(skill.status, SkillStatus.REJECTED)
        self.assertEqual(
            skill.metadata.get("executable_protocol_source"),
            "trajectory",
        )
        templates = [
            item.get("action_template")
            for item in (skill.metadata.get("executable_protocol") or [])
        ]
        self.assertTrue(any("heat" in str(t) for t in templates))
        self.assertFalse(any("examine" in str(t) for t in templates))
        # Merged protocol should match executable templates as single truth.
        for step in skill.action_protocol:
            self.assertTrue(
                any(step.split()[0] in str(t) for t in templates),
                msg=f"missing executable mirror for {step}",
            )

    def test_single_win_rejected_when_min_wins_three(self) -> None:
        adapted = AlfWorldTrajectoryAdapter().adapt_many(
            [_heat_win(game_id="1", with_stove_detour=False)]
        )
        from sage_mas.schemas import AtomicOp, Skill

        seed = Skill(
            skill_name="Observed heat",
            description="heat",
            precondition="heat",
            action_protocol=[],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="transform.heat",
            metadata={
                "source_signal": "heat_operation_success",
                "primary_task_family": "pick_heat_then_place_in_recep",
            },
        )
        skill = build_protocol_from_successes(seed, adapted, min_wins=3)
        self.assertFalse(skill.metadata.get("protocol_alignment_ok"))
        self.assertEqual(skill.status, SkillStatus.REJECTED)
        self.assertIn("need>=3", str(skill.metadata.get("protocol_quality_reject")))

    def test_enriched_distiller_sets_provisional_utility(self) -> None:
        adapted = AlfWorldTrajectoryAdapter().adapt_many(
            [
                _heat_win(game_id="1", with_stove_detour=False),
                _heat_win(game_id="2", with_stove_detour=False),
                _heat_win(game_id="3", with_stove_detour=False),
            ]
        )
        skills = EnrichedSkillDistiller(
            DistillationConfig(operation_min_support=1, min_support=1),
            use_llm_rewrite=False,
            min_wins_for_protocol=3,
        ).distill(adapted)
        self.assertTrue(skills)
        skill = skills[0]
        self.assertEqual(skill.status, SkillStatus.PROVISIONAL)
        self.assertEqual(skill.metadata.get("utility"), 0.5)
        self.assertIsNone(skill.marginal_utility)
        self.assertNotIn("marginal_utility", skill.metadata)
        self.assertEqual(skill.metadata.get("skill_credit", {}).get("score"), 0.5)
        self.assertEqual(
            skill.metadata.get("distiller"),
            "merged-success-detour-clean-v2",
        )
        self.assertNotIn("failure_branches", skill.metadata)

    def test_online_utility_promotes_without_mu(self) -> None:
        adapted = AlfWorldTrajectoryAdapter().adapt_many(
            [
                _heat_win(game_id="1", with_stove_detour=False),
                _heat_win(game_id="2", with_stove_detour=False),
                _heat_win(game_id="3", with_stove_detour=False),
            ]
        )
        skills = EnrichedSkillDistiller(
            DistillationConfig(operation_min_support=1, min_support=1),
            use_llm_rewrite=False,
            min_wins_for_protocol=3,
            credit_policy=SkillCreditPolicy(
                verify_score=0.6,
                min_uses_for_promotion=1,
                require_positive_mu_for_verify=False,
                min_adherence_for_use=0.0,
                min_adherence_for_verify=0.0,
                require_task_win_for_success=True,
            ),
        ).distill(adapted)
        skill = skills[0]
        initialize_provisional_utility(
            skill,
            policy=SkillCreditPolicy(require_positive_mu_for_verify=False),
        )
        self.assertIsNone(skill.marginal_utility)
        from sage_mas.alfworld_evaluator import EvaluationTrial
        from sage_mas.schemas import AgentSpec

        agents = [
            AgentSpec(
                name="Executor",
                role="Executor",
                responsibilities=["act"],
                assigned_skills=[skill.skill_name],
                tool_permissions=["alfworld_action"],
            )
        ]
        trial = EvaluationTrial(
            task_id="/tmp/heat-1/game.tw-pddl",
            task="heat potato",
            task_family="pick_heat_then_place_in_recep",
            condition="unit",
            reward=1.0,
            cost=2.0,
            won=True,
            num_steps=2,
            steps=[
                {
                    "observation": "You heat the potato using the microwave.",
                    "action": "heat potato 1 with microwave 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You move the potato.",
                    "action": "move potato 1 to garbagecan 1",
                    "is_action_valid": True,
                },
            ],
            assigned_primary_agent="Executor",
            activated_skill_names=[skill.skill_name],
            skill_activation_steps={skill.skill_name: 1},
            actions_by_agent={"Executor": 2},
        )
        # Mark capability operation for local-effect fallback.
        skill.metadata["capability_operation"] = "heat"
        update_skill_credits(
            [skill],
            [trial],
            agents,
            SkillCreditPolicy(
                verify_score=0.55,
                min_uses_for_promotion=1,
                require_positive_mu_for_verify=False,
                min_adherence_for_use=0.0,
                min_adherence_for_verify=0.0,
                require_task_win_for_success=True,
                combine_adherence_with_effect=False,
                min_protocol_coverage=0.0,
            ),
        )
        self.assertGreaterEqual(skill.metadata["utility"], 0.5)
        self.assertIn(
            skill.status,
            {SkillStatus.PROVISIONAL, SkillStatus.VERIFIED},
        )


def _light_win(
    *,
    game_id: str,
    object_name: str = "bowl",
    with_search: bool = True,
) -> dict:
    steps = []
    if with_search:
        steps.extend(
            [
                {
                    "observation": "You arrive at the desk.",
                    "action": "go to desk 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You open the desk.",
                    "action": "open desk 1",
                    "is_action_valid": True,
                },
                {
                    "observation": f"You pick up the {object_name}.",
                    "action": f"take {object_name} 1 from desk 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You arrive at the desklamp.",
                    "action": "go to desklamp 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You turn on the desklamp.",
                    "action": "use desklamp 1",
                    "is_action_valid": True,
                },
            ]
        )
    else:
        # Over-collapsed skeleton: go → take → use desklamp only.
        steps.extend(
            [
                {
                    "observation": "You arrive at the desk.",
                    "action": "go to desk 1",
                    "is_action_valid": True,
                },
                {
                    "observation": f"You pick up the {object_name}.",
                    "action": f"take {object_name} 1 from desk 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You turn on the desklamp.",
                    "action": "use desklamp 1",
                    "is_action_valid": True,
                },
            ]
        )
    return {
        "gamefile": (
            f"/tmp/look_at_obj_in_light-{object_name.capitalize()}-None-"
            f"DeskLamp-{game_id}/game.tw-pddl"
        ),
        "task": f"look at {object_name} under the desklamp",
        "won": True,
        "num_steps": len(steps),
        "steps": steps,
    }


class LightProtocolNoiseGateTests(unittest.TestCase):
    def _seed(self):
        from sage_mas.schemas import AtomicOp, Skill

        return Skill(
            skill_name="Observed light",
            description="light",
            precondition="light",
            action_protocol=[],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="inspect.with_light",
            metadata={
                "source_signal": "light_operation_success",
                "primary_task_family": "look_at_obj_in_light",
            },
        )

    def test_shallow_light_skeleton_accepted_when_causal_clean(self) -> None:
        """go→take→use desklamp is the complete causal chain for this family.

        The old length floor rejected it as an "over-collapsed skeleton"; the
        causal gate supersedes that proxy — a short coherent backbone is a
        valid minimal recipe.
        """
        adapted = AlfWorldTrajectoryAdapter().adapt_many(
            [
                _light_win(game_id="1", object_name="bowl", with_search=False),
                _light_win(game_id="2", object_name="mug", with_search=False),
                _light_win(game_id="3", object_name="book", with_search=False),
            ]
        )
        skill = build_protocol_from_successes(self._seed(), adapted, min_wins=3)
        self.assertNotEqual(skill.status, SkillStatus.REJECTED, msg=skill.metadata)
        self.assertTrue(skill.metadata.get("protocol_form_ok"), msg=skill.metadata)
        joined = " | ".join(skill.action_protocol).lower()
        self.assertIn("desklamp", joined)
        self.assertIn("take", joined)

    def test_rich_light_protocol_accepted(self) -> None:
        adapted = AlfWorldTrajectoryAdapter().adapt_many(
            [
                _light_win(game_id="1", object_name="bowl", with_search=True),
                _light_win(game_id="2", object_name="mug", with_search=True),
                _light_win(game_id="3", object_name="book", with_search=True),
            ]
        )
        skill = build_protocol_from_successes(self._seed(), adapted, min_wins=3)
        self.assertNotEqual(skill.status, SkillStatus.REJECTED)
        self.assertTrue(skill.metadata.get("protocol_form_ok"))
        self.assertGreaterEqual(len(skill.action_protocol), 4)
        joined = " | ".join(skill.action_protocol).lower()
        self.assertIn("desklamp", joined)
        self.assertIn("take", joined)

    def test_mixed_light_prefers_short_clean_backbones(self) -> None:
        """Shortest-first selection: clean shallow wins outrank detour-rich ones."""
        adapted = AlfWorldTrajectoryAdapter().adapt_many(
            [
                _light_win(game_id="1", object_name="bowl", with_search=False),
                _light_win(game_id="2", object_name="mug", with_search=False),
                _light_win(game_id="3", object_name="book", with_search=True),
                _light_win(game_id="4", object_name="cd", with_search=True),
            ]
        )
        skill = build_protocol_from_successes(
            self._seed(),
            adapted,
            min_wins=3,
            max_trajectories=3,
        )
        self.assertNotEqual(skill.status, SkillStatus.REJECTED, msg=skill.metadata)
        joined = " | ".join(skill.action_protocol).lower()
        self.assertIn("desklamp", joined)
        self.assertIn("take", joined)
        self.assertTrue(skill.metadata.get("protocol_form_ok"))


def _cool_win(*, game_id: str, extra_search: int = 2) -> dict:
    """Pro-like cool win: search hops, return to fridge, then place."""
    steps = [
        {
            "observation": "You arrive at the fridge.",
            "action": "go to fridge 1",
            "is_action_valid": True,
        },
        {
            "observation": "You open the fridge.",
            "action": "open fridge 1",
            "is_action_valid": True,
        },
        {
            "observation": "You arrive at the countertop.",
            "action": "go to countertop 1",
            "is_action_valid": True,
        },
    ]
    search_targets = ["diningtable 1", "countertop 2", "cabinet 1"]
    for index in range(max(0, int(extra_search))):
        loc = search_targets[index % len(search_targets)]
        steps.append(
            {
                "observation": f"You arrive at the {loc}.",
                "action": f"go to {loc}",
                "is_action_valid": True,
            }
        )
    steps.extend(
        [
            {
                "observation": "You pick up the apple.",
                "action": f"take apple {game_id} from diningtable 1",
                "is_action_valid": True,
            },
            {
                "observation": "You arrive at the fridge.",
                "action": "go to fridge 1",
                "is_action_valid": True,
            },
            {
                "observation": "You cool the apple using the fridge.",
                "action": f"cool apple {game_id} with fridge 1",
                "is_action_valid": True,
            },
            {
                "observation": "You arrive at the countertop.",
                "action": "go to countertop 1",
                "is_action_valid": True,
            },
            {
                "observation": "You move the apple.",
                "action": f"move apple {game_id} to countertop 1",
                "is_action_valid": True,
            },
        ]
    )
    return {
        "gamefile": (
            f"/tmp/pick_cool_then_place_in_recep-Apple-None-CounterTop-"
            f"{game_id}/game.tw-pddl"
        ),
        "task": "cool some apple and put it in/on countertop",
        "won": True,
        "num_steps": len(steps),
        "steps": steps,
    }


class CoolProtocolKeepsSearchHopsTests(unittest.TestCase):
    def _seed(self):
        from sage_mas.schemas import AtomicOp, Skill

        return Skill(
            skill_name="Observed cool",
            description="cool",
            precondition="cool",
            action_protocol=[],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="transform.cool",
            metadata={
                "source_signal": "cool_operation_success",
                "primary_task_family": "pick_cool_then_place_in_recep",
            },
        )

    def test_filtered_cool_trace_keeps_repeated_go_to(self) -> None:
        from sage_mas.trajectory.abstraction import stage_for_abstracted_action
        from sage_mas.trajectory.protocol_canon import (
            _filtered_protocol_from_trajectory,
            canonicalize_protocol_stages,
        )

        adapted = AlfWorldTrajectoryAdapter().adapt_many(
            [_cool_win(game_id="1", extra_search=2)]
        )
        protocol = _filtered_protocol_from_trajectory(
            adapted[0],
            capability="transform.cool",
        )
        n_find = sum(
            1 for step in protocol if stage_for_abstracted_action(step) == "find"
        )
        self.assertGreaterEqual(n_find, 3, msg=protocol)
        ordered = canonicalize_protocol_stages(
            protocol,
            capability="transform.cool",
        )
        self.assertGreaterEqual(
            sum(
                1
                for step in ordered
                if stage_for_abstracted_action(step) == "find"
            ),
            3,
            msg=ordered,
        )

    def test_merged_cool_protocol_is_causal_backbone(self) -> None:
        adapted = AlfWorldTrajectoryAdapter().adapt_many(
            [
                _cool_win(game_id="1", extra_search=2),
                _cool_win(game_id="2", extra_search=3),
                _cool_win(game_id="3", extra_search=2),
            ]
        )
        skill = build_protocol_from_successes(self._seed(), adapted, min_wins=3)
        self.assertNotEqual(skill.status, SkillStatus.REJECTED, msg=skill.metadata)
        protocol = list(skill.action_protocol or [])
        from sage_mas.trajectory.abstraction import stage_for_abstracted_action
        from sage_mas.trajectory.causal import protocol_causal_issues

        # Search detours never enter the backbone: exactly one nav hop per
        # core op (source / tool / target).
        n_find = sum(
            1 for step in protocol if stage_for_abstracted_action(step) == "find"
        )
        self.assertEqual(n_find, 3, msg=protocol)
        self.assertGreaterEqual(len(protocol), 6, msg=protocol)
        self.assertTrue(protocol[0].startswith("go to "), msg=protocol)
        take_i = next(i for i, s in enumerate(protocol) if s.startswith("take "))
        cool_i = next(i for i, s in enumerate(protocol) if s.startswith("cool "))
        place_i = next(i for i, s in enumerate(protocol) if s.startswith("move "))
        self.assertLess(take_i, cool_i)
        self.assertLess(cool_i, place_i)
        # Return-to-tool hop after pickup must survive; unique-set merge
        # used to dump every ``go to`` at the front.
        self.assertTrue(
            any(
                stage_for_abstracted_action(step) == "find"
                for step in protocol[take_i + 1 : cool_i]
            ),
            msg=protocol,
        )
        self.assertEqual(
            protocol_causal_issues(protocol, capability="transform.cool"),
            [],
            msg=protocol,
        )
        self.assertGreaterEqual(
            int(skill.metadata.get("protocol_consensus_support") or 0),
            2,
        )
        joined = " | ".join(protocol)
        self.assertTrue(
            any(step.startswith("cool ") and " with " in step for step in protocol),
            msg=joined,
        )

    def test_five_step_transform_skeleton_is_too_short(self) -> None:
        from sage_mas.skill_protocol_distill import assess_protocol_form_quality

        ok, reasons, _metrics = assess_protocol_form_quality(
            [
                "go to <location>",
                "open <entity>",
                "take <object> from <receptacle>",
                "cool <object> with <tool>",
                "move <object> to <receptacle>",
            ],
            [["go to <location>"] * 4 + ["open <entity>", "take <object> from <receptacle>", "cool <object> with <tool>", "move <object> to <receptacle>"]],
            capability="transform.cool",
            min_protocol_steps=4,
        )
        self.assertFalse(ok)
        self.assertTrue(any("too short" in reason for reason in reasons), msg=reasons)


class LongEpisodeKeepsCoreStagesTests(unittest.TestCase):
    def test_long_search_clean_trace_keeps_clean_and_place(self) -> None:
        from sage_mas.trajectory.protocol_canon import (
            _filtered_protocol_from_trajectory,
            protocol_structure_issues,
        )

        steps = []
        for index in range(12):
            loc = f"cabinet {index + 1}"
            steps.append(
                {
                    "observation": f"You arrive at the {loc}.",
                    "action": f"go to {loc}",
                    "is_action_valid": True,
                }
            )
            steps.append(
                {
                    "observation": f"You open the {loc}.",
                    "action": f"open {loc}",
                    "is_action_valid": True,
                }
            )
        steps.extend(
            [
                {
                    "observation": "You pick up the mug.",
                    "action": "take mug 1 from cabinet 12",
                    "is_action_valid": True,
                },
                {
                    "observation": "You arrive at the sinkbasin.",
                    "action": "go to sinkbasin 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You clean the mug using the sinkbasin.",
                    "action": "clean mug 1 with sinkbasin 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You arrive at the desk.",
                    "action": "go to desk 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You move the mug.",
                    "action": "move mug 1 to desk 1",
                    "is_action_valid": True,
                },
            ]
        )
        record = {
            "gamefile": "/tmp/pick_clean_then_place_in_recep-Mug-None-Desk-1/game.tw-pddl",
            "task": "put a clean mug on the desk",
            "won": True,
            "num_steps": len(steps),
            "steps": steps,
        }
        adapted = AlfWorldTrajectoryAdapter().adapt_many([record])[0]
        protocol = _filtered_protocol_from_trajectory(
            adapted,
            capability="transform.clean",
        )
        self.assertEqual(
            protocol_structure_issues(protocol, capability="transform.clean"),
            [],
            msg=protocol,
        )
        joined = " | ".join(protocol)
        self.assertTrue(
            any(step.startswith("clean ") and " with " in step for step in protocol),
            msg=joined,
        )
        self.assertTrue(
            any(step.startswith("move ") and " to " in step for step in protocol),
            msg=joined,
        )

    def test_merge_prefers_complete_protocol_over_search_only(self) -> None:
        from sage_mas.trajectory.consensus import (
            _align_protocols_across_trajectories,
        )
        from sage_mas.trajectory.protocol_canon import protocol_structure_issues

        search_only = ["go to <location>"] * 12 + [
            "open <entity>",
            "take <object> from <receptacle>",
        ]
        complete = [
            "go to <location>",
            "open <entity>",
            "take <object> from <receptacle>",
            "go to <location>",
            "clean <object> with <tool>",
            "go to <location>",
            "move <object> to <receptacle>",
        ]
        merged = _align_protocols_across_trajectories(
            [search_only, complete, search_only],
            capability="transform.clean",
        )
        self.assertEqual(
            protocol_structure_issues(merged, capability="transform.clean"),
            [],
            msg=merged,
        )
        self.assertIn("clean <object> with <tool>", merged)
        self.assertIn("move <object> to <receptacle>", merged)


class CausalBackboneTests(unittest.TestCase):
    def _adapt(self, records: list[dict]) -> list:
        return AlfWorldTrajectoryAdapter().adapt_many(records)

    def test_backbone_drops_search_detours_and_keeps_chain(self) -> None:
        from sage_mas.trajectory.causal import extract_causal_backbone

        steps = []
        for index in range(6):
            loc = f"sidetable {index + 1}"
            steps.append(
                {
                    "observation": f"You arrive at the {loc}.",
                    "action": f"go to {loc}",
                    "is_action_valid": True,
                }
            )
        steps.extend(
            [
                {
                    "observation": "You arrive at the dresser.",
                    "action": "go to dresser 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You pick up the book.",
                    "action": "take book 1 from dresser 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You arrive at the coffeetable.",
                    "action": "go to coffeetable 1",
                    "is_action_valid": True,
                },
                {
                    "observation": "You move the book.",
                    "action": "move book 1 to coffeetable 1",
                    "is_action_valid": True,
                },
            ]
        )
        record = {
            "gamefile": "/tmp/pick_and_place-Book-None-CoffeeTable-1/game.tw-pddl",
            "task": "put a book on the coffeetable",
            "won": True,
            "num_steps": len(steps),
            "steps": steps,
        }
        adapted = self._adapt([record])[0]
        backbone = extract_causal_backbone(adapted, capability="track.place")
        self.assertEqual(
            backbone,
            [
                "go to dresser",
                "take book from dresser",
                "go to coffeetable",
                "move book to coffeetable",
            ],
            msg=backbone,
        )

    def test_backbone_uses_first_acquisition_not_reacquisition(self) -> None:
        """Heat trace with put-in/take-out confusion: chain starts at cabinet."""
        from sage_mas.trajectory.causal import extract_causal_backbone

        steps = [
            ("go to cabinet 9", "You arrive at the cabinet 9."),
            ("open cabinet 9", "You open the cabinet 9."),
            ("take plate 2 from cabinet 9", "You pick up the plate 2 from the cabinet 9."),
            ("go to microwave 1", "You arrive at the microwave 1."),
            ("open microwave 1", "You open the microwave 1."),
            ("move plate 2 to microwave 1", "You move the plate 2 to the microwave 1."),
            ("take plate 2 from microwave 1", "You pick up the plate 2 from the microwave 1."),
            ("heat plate 2 with microwave 1", "You heat the plate 2 using the microwave 1."),
            ("go to countertop 1", "You arrive at the countertop 1."),
            ("move plate 2 to countertop 1", "You move the plate 2 to the countertop 1."),
        ]
        record = {
            "gamefile": "/tmp/pick_heat_then_place_in_recep-Plate-None-CounterTop-1/game.tw-pddl",
            "task": "put a hot plate in countertop",
            "won": True,
            "num_steps": len(steps),
            "steps": [
                {"action": a, "observation": o, "is_action_valid": True}
                for a, o in steps
            ],
        }
        adapted = self._adapt([record])[0]
        backbone = extract_causal_backbone(adapted, capability="transform.heat")
        joined = " | ".join(backbone)
        self.assertIn("take plate from cabinet", joined, msg=joined)
        self.assertNotIn("take plate from microwave", joined, msg=joined)
        self.assertIn("heat plate with microwave", joined, msg=joined)
        self.assertEqual(backbone[-1], "move plate to countertop", msg=joined)

    def test_causal_gate_flags_closed_receptacle_use(self) -> None:
        from sage_mas.trajectory.causal import protocol_causal_issues

        protocol = [
            "go to cabinet",
            "open cabinet",
            "close cabinet",
            "take <object> from cabinet",
        ]
        issues = protocol_causal_issues(protocol, capability="track.place")
        self.assertTrue(any("closed" in issue for issue in issues), msg=issues)

    def test_causal_gate_flags_location_mismatch_and_stutter(self) -> None:
        from sage_mas.trajectory.causal import protocol_causal_issues

        protocol = [
            "go to sidetable",
            "go to sidetable",
            "go to dresser",
            "take book from coffeetable",
        ]
        issues = protocol_causal_issues(protocol, capability="track.place")
        self.assertTrue(
            any("consecutive duplicate navigation" in issue for issue in issues),
            msg=issues,
        )
        self.assertTrue(
            any("current location" in issue for issue in issues),
            msg=issues,
        )

    def test_causal_gate_skips_slotted_and_use_steps(self) -> None:
        from sage_mas.trajectory.causal import protocol_causal_issues

        protocol = [
            "go to <source>",
            "take <object> from <source>",
            "use desklamp",
        ]
        self.assertEqual(
            protocol_causal_issues(protocol, capability="inspect.with_light"),
            [],
        )

    def test_causal_gate_requires_concrete_transform_tool(self) -> None:
        from sage_mas.trajectory.causal import protocol_causal_issues

        protocol = [
            "go to <source>",
            "take <object> from <source>",
            "cool <object> with <tool>",
            "go to <destination>",
            "move <object> to <destination>",
        ]
        issues = protocol_causal_issues(protocol, capability="transform.cool")
        self.assertTrue(any("concrete tool" in issue for issue in issues), msg=issues)

    def test_picktwo_backbone_keeps_two_cycles(self) -> None:
        from sage_mas.trajectory.causal import extract_causal_backbone

        steps = [
            ("go to countertop 1", "You arrive at the countertop 1."),
            ("take peppershaker 1 from countertop 1", "You pick up the peppershaker 1."),
            ("go to drawer 1", "You arrive at the drawer 1."),
            ("open drawer 1", "You open the drawer 1."),
            ("move peppershaker 1 to drawer 1", "You move the peppershaker 1 to the drawer 1."),
            ("go to shelf 1", "You arrive at the shelf 1."),
            ("take peppershaker 2 from shelf 1", "You pick up the peppershaker 2."),
            ("go to drawer 1", "You arrive at the drawer 1."),
            ("move peppershaker 2 to drawer 1", "You move the peppershaker 2 to the drawer 1."),
        ]
        record = {
            "gamefile": "/tmp/pick_two_obj_and_place-PepperShaker-None-Drawer-1/game.tw-pddl",
            "task": "put two peppershaker in drawer",
            "won": True,
            "num_steps": len(steps),
            "steps": [
                {"action": a, "observation": o, "is_action_valid": True}
                for a, o in steps
            ],
        }
        adapted = self._adapt([record])[0]
        backbone = extract_causal_backbone(
            adapted, capability="track.multiple_objects"
        )
        takes = [s for s in backbone if s.startswith("take ")]
        places = [s for s in backbone if s.startswith("move ")]
        self.assertEqual(len(takes), 2, msg=backbone)
        self.assertEqual(len(places), 2, msg=backbone)

    def test_heterogeneous_wins_fall_back_to_slotted_shape(self) -> None:
        """Wins with disjoint receptacles must not merge into a chimera."""
        from sage_mas.schemas import AtomicOp, Skill

        def place_win(game_id: str, source: str, target: str) -> dict:
            return {
                "gamefile": (
                    f"/tmp/pick_and_place-Book-None-Target-{game_id}/game.tw-pddl"
                ),
                "task": "put a book somewhere",
                "won": True,
                "num_steps": 4,
                "steps": [
                    {
                        "action": f"go to {source} 1",
                        "observation": f"You arrive at the {source} 1.",
                        "is_action_valid": True,
                    },
                    {
                        "action": f"take book 1 from {source} 1",
                        "observation": "You pick up the book 1.",
                        "is_action_valid": True,
                    },
                    {
                        "action": f"go to {target} 1",
                        "observation": f"You arrive at the {target} 1.",
                        "is_action_valid": True,
                    },
                    {
                        "action": f"move book 1 to {target} 1",
                        "observation": "You move the book 1.",
                        "is_action_valid": True,
                    },
                ],
            }

        adapted = self._adapt(
            [
                place_win("1", "dresser", "coffeetable"),
                place_win("2", "sofa", "diningtable"),
                place_win("3", "countertop", "cabinet"),
            ]
        )
        seed = Skill(
            skill_name="Observed place",
            description="place",
            precondition="place",
            action_protocol=[],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="track.place",
            metadata={
                "source_signal": "place_task_success",
                "primary_task_family": "pick_and_place",
            },
        )
        skill = build_protocol_from_successes(seed, adapted, min_wins=3)
        self.assertNotEqual(skill.status, SkillStatus.REJECTED, msg=skill.metadata)
        self.assertEqual(
            skill.metadata.get("protocol_merge_mode"),
            "slotted_shape_consensus",
        )
        self.assertEqual(
            skill.action_protocol,
            [
                "go to <source>",
                "take <object> from <source>",
                "go to <destination>",
                "move <object> to <destination>",
            ],
            msg=skill.action_protocol,
        )


class SearchPriorDistillTests(unittest.TestCase):
    """Search prior: where wins first acquired the target object."""

    @staticmethod
    def _place_win(game_id: str, source: str, target: str) -> dict:
        return {
            "gamefile": (
                f"/tmp/pick_and_place-Book-None-Target-{game_id}/game.tw-pddl"
            ),
            "task": "put a book somewhere",
            "won": True,
            "num_steps": 4,
            "steps": [
                {
                    "action": f"go to {source} 1",
                    "observation": f"You arrive at the {source} 1.",
                    "is_action_valid": True,
                },
                {
                    "action": f"take book 1 from {source} 1",
                    "observation": "You pick up the book 1.",
                    "is_action_valid": True,
                },
                {
                    "action": f"go to {target} 1",
                    "observation": f"You arrive at the {target} 1.",
                    "is_action_valid": True,
                },
                {
                    "action": f"move book 1 to {target} 1",
                    "observation": "You move the book 1.",
                    "is_action_valid": True,
                },
            ],
        }

    def test_prior_distilled_into_metadata_and_precondition(self) -> None:
        from sage_mas.schemas import AtomicOp, Skill

        adapted = AlfWorldTrajectoryAdapter().adapt_many(
            [
                self._place_win("1", "countertop", "cabinet"),
                self._place_win("2", "countertop", "shelf"),
                self._place_win("3", "drawer", "cabinet"),
            ]
        )
        seed = Skill(
            skill_name="Observed place",
            description="place",
            precondition="place",
            action_protocol=[],
            applicable_atomic_ops=[AtomicOp.ACT],
            capability_key="track.place",
            metadata={
                "source_signal": "place_task_success",
                "primary_task_family": "pick_and_place",
            },
        )
        skill = build_protocol_from_successes(seed, adapted, min_wins=3)
        self.assertNotEqual(skill.status, SkillStatus.REJECTED, msg=skill.metadata)
        prior = skill.metadata.get("search_prior") or {}
        self.assertEqual(prior.get("episodes"), 3, msg=prior)
        self.assertEqual(
            prior.get("sources"),
            [
                {"source": "countertop", "count": 2},
                {"source": "drawer", "count": 1},
            ],
            msg=prior,
        )
        self.assertIn("Search prior", skill.precondition)
        self.assertIn("countertop (2)", skill.precondition)
        self.assertIn("drawer (1)", skill.precondition)

    def test_prior_ignores_post_transform_regrab(self) -> None:
        from sage_mas.skill_protocol_distill import extract_search_prior

        win = {
            "gamefile": "/tmp/pick_heat_then_place_in_recep-Potato-None-GarbageCan-1/game.tw-pddl",
            "task": "heat some potato and put it in garbagecan",
            "won": True,
            "num_steps": 7,
            "steps": [
                {"action": "go to fridge 1", "observation": "Arrive.", "is_action_valid": True},
                {"action": "take potato 1 from fridge 1", "observation": "Pick up.", "is_action_valid": True},
                {"action": "go to microwave 1", "observation": "Arrive.", "is_action_valid": True},
                {"action": "heat potato 1 with microwave 1", "observation": "Heat.", "is_action_valid": True},
                {"action": "move potato 1 to microwave 1", "observation": "Put in.", "is_action_valid": True},
                # Re-grab of the same instance after the transform: not a
                # new acquisition, must not count as a microwave source.
                {"action": "take potato 1 from microwave 1", "observation": "Pick up.", "is_action_valid": True},
                {"action": "go to garbagecan 1", "observation": "Arrive.", "is_action_valid": True},
                {"action": "move potato 1 to garbagecan 1", "observation": "Done.", "is_action_valid": True},
            ],
        }
        adapted = AlfWorldTrajectoryAdapter().adapt_many([win])
        prior = extract_search_prior(adapted)
        self.assertEqual(prior["sources"], [{"source": "fridge", "count": 1}])
        self.assertEqual(prior["episodes"], 1)

    def test_prior_empty_when_take_has_no_source(self) -> None:
        from sage_mas.skill_protocol_distill import (
            extract_search_prior,
            search_prior_sentence,
        )

        win = {
            "gamefile": "/tmp/pick_and_place-Mug-None-Cabinet-1/game.tw-pddl",
            "task": "put a mug in cabinet",
            "won": True,
            "num_steps": 2,
            "steps": [
                {"action": "take mug 1", "observation": "Pick up.", "is_action_valid": True},
                {"action": "move mug 1 to cabinet 1", "observation": "Done.", "is_action_valid": True},
            ],
        }
        adapted = AlfWorldTrajectoryAdapter().adapt_many([win])
        prior = extract_search_prior(adapted)
        self.assertEqual(prior["sources"], [])
        self.assertEqual(prior["episodes"], 0)
        self.assertEqual(search_prior_sentence(prior), "")


class ProductiveStepFilterTests(unittest.TestCase):
    def test_confirmed_effect_overrides_false_is_action_valid(self) -> None:
        from sage_mas.schemas import AtomicOp, AtomicStep
        from sage_mas.trajectory.abstraction import (
            is_productive_environment_step,
        )

        cases = [
            ("take apple 1 from fridge 1", "You pick up the apple 1 from the fridge 1."),
            ("clean knife 1 with sinkbasin 1", "You clean the knife 1 using the sinkbasin 1."),
            ("heat potato 1 with microwave 1", "You heat the potato 1 using the microwave 1."),
            ("cool apple 1 with fridge 1", "You cool the apple 1 using the fridge 1."),
            ("use desklamp 1", "You turn on the desklamp 1."),
        ]
        for action, observation in cases:
            with self.subTest(action=action):
                step = AtomicStep(
                    node_id="s1",
                    atomic_op=AtomicOp.ACT,
                    agent="Executor",
                    observation=observation,
                    action=action,
                    metadata={"is_action_valid": False, "stalled": False},
                )
                self.assertTrue(is_productive_environment_step(step))

    def test_false_valid_without_effect_still_dropped(self) -> None:
        from sage_mas.schemas import AtomicOp, AtomicStep
        from sage_mas.trajectory.abstraction import (
            is_productive_environment_step,
        )

        step = AtomicStep(
            node_id="s1",
            atomic_op=AtomicOp.ACT,
            agent="Executor",
            observation="You are facing the table.",
            action="go to table 1",
            metadata={"is_action_valid": False, "stalled": False},
        )
        self.assertFalse(is_productive_environment_step(step))


if __name__ == "__main__":
    unittest.main()
