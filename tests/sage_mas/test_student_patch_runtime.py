"""Tests for student_patch_v2 binding: trigger / slots / concrete inject."""

from __future__ import annotations

import unittest

from sage_mas.schemas import Skill, SkillStatus
from sage_mas.student_patch_runtime import (
    action_hits_forbidden,
    bind_student_patch,
    render_skills_bound,
)
from sage_mas.student_skill_enrich import enrich_skill_for_student


def _cool_skill() -> Skill:
    return Skill(
        skill_name="Merged transform cool protocol",
        description="cool",
        precondition="Task family matches pick_cool_then_place_in_recep.",
        action_protocol=[
            "go to <location>",
            "take <object> from <receptacle>",
            "cool <object> with <tool>",
            "move <object> to <receptacle>",
        ],
        applicable_atomic_ops=["Act"],
        expected_effect="cool then place",
        suggested_role="Executor",
        target_failure_types=[],
        status=SkillStatus.VERIFIED,
        applicable_task_families=["pick_cool_then_place_in_recep"],
        capability_key="transform.cool",
        metadata={"primary_task_family": "pick_cool_then_place_in_recep"},
    )


OBS = """
You are in the middle of a room. Looking quickly around you, you see a fridge 1,
a countertop 1, and an apple 1.
Inventory: nothing
Admissible actions: ['go to fridge 1', 'go to countertop 1', 'take apple 1 from countertop 1', 'cool apple 1 with fridge 1', 'move apple 1 to countertop 1', 'open fridge 1']
""".strip()

TASK = "cool some apple and put it in/on countertop."
GAME = (
    "/data/alfworld/json_2.1.1/valid_unseen/"
    "pick_cool_then_place_in_recep-Apple-None-CounterTop-1/trial/game.tw-pddl"
)


class StudentPatchRuntimeTests(unittest.TestCase):
    def test_enrich_emits_v2_structure(self) -> None:
        enriched = enrich_skill_for_student(_cool_skill())
        patch = enriched.metadata["student_patch"]
        self.assertEqual(patch["schema_version"], "student_patch_v2")
        self.assertIn("trigger_signature", patch)
        self.assertIn("slot_bindings", patch)
        self.assertIn("action_templates", patch)
        self.assertTrue(patch["action_templates"])

    def test_bound_inject_has_no_placeholders(self) -> None:
        enriched = enrich_skill_for_student(_cool_skill())
        bound = bind_student_patch(
            enriched,
            observation=OBS,
            task=TASK,
            gamefile=GAME,
            task_family="pick_cool_then_place_in_recep",
        )
        self.assertTrue(bound.matched, bound.reason)
        self.assertTrue(bound.inject_text)
        self.assertNotIn("<", bound.inject_text)
        self.assertNotIn("family `", bound.inject_text)
        self.assertIn("[Skill:", bound.inject_text)
        self.assertIn("cool apple 1 with fridge 1", bound.inject_text.lower())
        self.assertTrue(any("apple" in step for step in bound.concrete_steps))

    def test_forbidden_after_progress(self) -> None:
        enriched = enrich_skill_for_student(_cool_skill())
        history = [
            {"action": "go to fridge 1"},
            {"action": "take apple 1 from countertop 1"},
        ]
        bound = bind_student_patch(
            enriched,
            observation=OBS,
            task=TASK,
            gamefile=GAME,
            task_family="pick_cool_then_place_in_recep",
            history_steps=history,
        )
        self.assertTrue(bound.matched)
        self.assertGreaterEqual(bound.current_step_index, 1)
        self.assertTrue(bound.forbidden_actions)
        self.assertTrue(
            action_hits_forbidden(bound.forbidden_actions[0], bound.forbidden_actions)
        )
        self.assertIn("Do not output", bound.inject_text)

    def test_family_mismatch_does_not_inject(self) -> None:
        enriched = enrich_skill_for_student(_cool_skill())
        bound = bind_student_patch(
            enriched,
            observation=OBS,
            task="put a pillow on the sofa",
            gamefile="pick_and_place-Pillow-None-Sofa-1/game.tw-pddl",
            task_family="pick_and_place",
        )
        self.assertFalse(bound.matched)
        self.assertEqual(bound.inject_text, "")

    def test_bound_inject_includes_success_demos(self) -> None:
        skill = _cool_skill()
        skill.metadata = dict(skill.metadata or {})
        skill.metadata["specialist_demos"] = [
            {
                "task_family": "pick_cool_then_place_in_recep",
                "won": True,
                "actions": [
                    "go to fridge 1",
                    "open fridge 1",
                    "take apple 3 from diningtable 1",
                    "cool apple 3 with fridge 1",
                    "move apple 3 to diningtable 1",
                ],
            }
        ]
        skill.metadata["specialist_failure_cues"] = [
            "observed student stall near `go to <location>` (n=2); "
            "do not repeat that prefix"
        ]
        skill.metadata["anti_patterns"] = [
            "Avoid repeating `go to <location>` after no-op / invalid feedback."
        ]
        enriched = enrich_skill_for_student(skill)
        self.assertEqual(len(enriched.metadata["student_patch"]["success_demos"]), 1)
        self.assertTrue(enriched.metadata["student_patch"]["failure_cues"])
        bound = bind_student_patch(
            enriched,
            observation=OBS,
            task=TASK,
            gamefile=GAME,
            task_family="pick_cool_then_place_in_recep",
        )
        self.assertTrue(bound.matched)
        self.assertIn("Success demo 1", bound.inject_text)
        self.assertIn("cool apple 3 with fridge 1", bound.inject_text)
        self.assertIn("Failure cues:", bound.inject_text)
        self.assertIn("Anti-patterns:", bound.inject_text)

    def test_render_skills_bound(self) -> None:
        enriched = enrich_skill_for_student(_cool_skill())
        text = render_skills_bound(
            [enriched],
            observation=OBS,
            task=TASK,
            gamefile=GAME,
            task_family="pick_cool_then_place_in_recep",
        )
        self.assertIn("Follow these steps in order:", text)
        self.assertNotIn("<object>", text)


if __name__ == "__main__":
    unittest.main()
