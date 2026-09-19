"""Tests for extracted rich skill context formatting."""

from __future__ import annotations

import unittest

from sage_mas.rich_skill_context import (
    render_rich_skill_block,
    render_rich_supplement_for_student_patch,
)
from sage_mas.runtime import MASRuntime
from sage_mas.schemas import Skill, SkillStatus


def _cool() -> Skill:
    return Skill(
        skill_name="Merged transform cool protocol",
        description="cool",
        precondition="Task family matches pick_cool_then_place_in_recep.",
        action_protocol=[
            "go to <location>",
            "cool <object> with <tool>",
            "move <object> to <receptacle>",
        ],
        applicable_atomic_ops=["Act"],
        expected_effect="cool then place",
        suggested_role="Executor",
        target_failure_types=[],
        status=SkillStatus.VERIFIED,
        applicable_task_families=["pick_cool_then_place_in_recep"],
        metadata={
            "primary_task_family": "pick_cool_then_place_in_recep",
            "protocol_stages": ["find", "transform", "place"],
            "specialist_demos": [
                {
                    "task_family": "pick_cool_then_place_in_recep",
                    "won": True,
                    "actions": [
                        "go to fridge 1",
                        "cool apple 1 with fridge 1",
                        "move apple 1 to countertop 1",
                    ],
                }
            ],
            "specialist_failure_cues": [
                "observed student stall near `go to <location>`"
            ],
            "anti_patterns": [
                "Avoid repeating `go to <location>` after no-op."
            ],
        },
    )


class RichSkillContextTests(unittest.TestCase):
    def test_full_block_matches_runtime_render(self) -> None:
        skill = _cool()
        extracted = render_rich_skill_block(skill)
        via_runtime = MASRuntime._render_skills([skill])
        self.assertEqual(extracted, via_runtime)
        self.assertIn("Success demo 1", extracted)
        self.assertIn("Failure cues:", extracted)
        self.assertIn("Anti-patterns:", extracted)

    def test_student_supplement_omits_long_contract(self) -> None:
        skill = _cool()
        text = render_rich_supplement_for_student_patch(skill)
        self.assertIn("Success demo 1", text)
        self.assertIn("Failure cues:", text)
        self.assertNotIn("Precondition:", text)
        self.assertNotIn("Placeholder note:", text)


if __name__ == "__main__":
    unittest.main()
