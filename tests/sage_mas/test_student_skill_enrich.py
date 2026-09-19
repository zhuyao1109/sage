"""Tests for strong→mid-strong student skill enrichment."""

from __future__ import annotations

import unittest

from sage_mas.runtime import MASRuntime
from sage_mas.schemas import Skill, SkillStatus
from sage_mas.skill_bank_roles import skill_bank_role
from sage_mas.student_skill_enrich import (
    enrich_skill_for_student,
    enrich_skills_for_student,
    render_student_inject_text,
)


def _skill() -> Skill:
    return Skill(
        skill_name="Merged transform cool protocol",
        description="Shared cool protocol.",
        precondition=(
            "Task family matches `pick_cool_then_place_in_recep`. "
            "Instantiate placeholders from observation."
        ),
        action_protocol=[
            "go to <location>",
            "open <entity>",
            "take <object> from <receptacle>",
            "cool <object> with <tool>",
            "move <object> to <receptacle>",
        ],
        applicable_atomic_ops=["Act"],
        expected_effect="Cool confirmed then placed.",
        suggested_role="Executor",
        target_failure_types=["cool_operation_success"],
        status=SkillStatus.VERIFIED,
        applicable_task_families=["pick_cool_then_place_in_recep"],
        capability_key="transform.cool",
        metadata={
            "primary_task_family": "pick_cool_then_place_in_recep",
            "anti_patterns": [
                "Avoid repeating go to after Nothing happens.",
                "Avoid placing before cool.",
            ],
        },
    )


def _traj(family: str, won: bool, actions: list[str], gamefile: str) -> dict:
    return {
        "task_family": family,
        "won": won,
        "gamefile": gamefile,
        "steps": [{"action": action, "observation": "ok"} for action in actions],
    }


class StudentSkillEnrichTests(unittest.TestCase):
    def test_protocol_fallback_builds_v2_patch(self) -> None:
        enriched = enrich_skill_for_student(_skill())
        patch = enriched.metadata["student_patch"]
        self.assertEqual(patch["schema_version"], "student_patch_v2")
        self.assertEqual(patch["source"], "canonical_protocol")
        self.assertGreaterEqual(len(patch["action_templates"]), 3)
        self.assertIn("cool <object> with <tool>", patch["action_templates"])
        self.assertIn("trigger_signature", patch)
        self.assertIn("slot_bindings", patch)
        self.assertEqual(skill_bank_role(enriched), "compiled_candidate")
        self.assertFalse(enriched.metadata.get("inject_ready"))

    def test_contrastive_remainder_preferred(self) -> None:
        family = "pick_cool_then_place_in_recep"
        game = "game/cool-1"
        teacher = _traj(
            family,
            True,
            [
                "go to table 1",
                "take apple 1 from table 1",
                "go to fridge 1",
                "cool apple 1 with fridge 1",
                "move apple 1 to countertop 1",
            ],
            game,
        )
        student = _traj(
            family,
            False,
            [
                "go to table 1",
                "take apple 1 from table 1",
                "move apple 1 to countertop 1",
                "go to table 1",
            ],
            game,
        )
        enriched = enrich_skill_for_student(
            _skill(),
            teacher_records=[teacher],
            student_records=[student],
        )
        patch = enriched.metadata["student_patch"]
        self.assertEqual(patch["source"], "contrastive_remainder")
        self.assertTrue(patch["contrastive"]["used_contrastive"])
        joined = " | ".join(patch["action_templates"])
        self.assertIn("cool", joined)

    def test_static_render_does_not_leak_placeholders(self) -> None:
        enriched = enrich_skill_for_student(_skill())
        rendered = MASRuntime._render_skills([enriched])
        self.assertIn("student_patch_v2", rendered)
        self.assertNotIn("go to <location>", rendered)

    def test_bank_enrich_marks_canonical_inputs(self) -> None:
        result = enrich_skills_for_student([_skill()], teacher_model="pro")
        self.assertEqual(len(result["canonical"]), 1)
        self.assertEqual(len(result["enriched"]), 1)
        self.assertEqual(skill_bank_role(result["canonical"][0]), "canonical")
        self.assertFalse(result["canonical"][0].metadata.get("inject_ready"))
        self.assertEqual(skill_bank_role(result["enriched"][0]), "compiled_candidate")

    def test_render_helper_is_compact(self) -> None:
        text = render_student_inject_text(
            skill_name="cool",
            when="family cool",
            do_steps=["take <object>", "cool <object> with <tool>", "move <object>"],
            dont_steps=["place before cool"],
        )
        self.assertIn("Do (in order", text)
        self.assertIn("Don't:", text)
        self.assertLess(len(text), 400)


if __name__ == "__main__":
    unittest.main()
