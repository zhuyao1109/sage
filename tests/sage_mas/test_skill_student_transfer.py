"""Tests for student-aware transfer, contrastive patches, and MU gates."""

from __future__ import annotations

import unittest

from sage_mas.schemas import Skill, SkillStatus
from sage_mas.skill_injection_policy import skill_is_injectable
from sage_mas.skill_student_transfer import (
    adapt_skill_for_student,
    apply_student_mu_gate,
    distill_contrastive_patches,
    pair_teacher_wins_student_fails,
    revise_skill_from_student_failures,
    skill_is_transfer_ready,
)
from sage_mas.trajectory.causal import protocol_is_org_ready


def _traj(family: str, won: bool, actions: list[str], gamefile: str) -> dict:
    return {
        "task_family": family,
        "won": won,
        "gamefile": gamefile,
        "steps": [{"action": action, "observation": "ok"} for action in actions],
    }


class StudentTransferTests(unittest.TestCase):
    def test_mu_gate_zero_delta_is_bank_ready(self) -> None:
        skill = Skill(
            skill_name="draft",
            description="d",
            precondition="p",
            action_protocol=["go to <location>", "take <object> from <receptacle>"],
            applicable_atomic_ops=["Act"],
            expected_effect="e",
            suggested_role="Executor",
            target_failure_types=[],
            status=SkillStatus.PROVISIONAL,
            metadata={"capability_key": "track.place"},
        )
        gated = apply_student_mu_gate(skill, delta_sr=0.0)
        self.assertTrue(gated.metadata["inject_ready"])
        self.assertTrue(gated.metadata["add_agent_ready"])
        self.assertFalse(gated.metadata["mu_rejected"])
        self.assertEqual(gated.status, SkillStatus.VERIFIED)
        self.assertTrue(skill_is_transfer_ready(gated))

    def test_mu_gate_positive_delta_inject_ready(self) -> None:
        skill = Skill(
            skill_name="draft",
            description="d",
            precondition="p",
            action_protocol=["go to <location>", "heat <object> with <tool>"],
            applicable_atomic_ops=["Act"],
            expected_effect="e",
            suggested_role="Executor",
            target_failure_types=[],
            status=SkillStatus.PROVISIONAL,
            metadata={"capability_key": "transform.heat"},
        )
        gated = apply_student_mu_gate(skill, delta_sr=0.25)
        self.assertTrue(gated.metadata["inject_ready"])
        self.assertTrue(gated.metadata["add_agent_ready"])
        self.assertEqual(gated.status, SkillStatus.VERIFIED)
        self.assertTrue(
            skill_is_injectable(
                gated,
                require_positive_mu=True,
                require_inject_ready=True,
            )
        )

    def test_explicit_inject_ready_false_blocks_injection(self) -> None:
        skill = Skill(
            skill_name="draft",
            description="d",
            precondition="p",
            action_protocol=["go to <location>"],
            applicable_atomic_ops=["Act"],
            expected_effect="e",
            suggested_role="Executor",
            target_failure_types=[],
            capability_key="transform.clean",
            marginal_utility=0.5,
            metadata={
                "capability_key": "transform.clean",
                "inject_ready": False,
                "marginal_utility": 0.5,
            },
        )
        self.assertFalse(
            skill_is_injectable(
                skill,
                require_positive_mu=False,
                require_inject_ready=False,
            )
        )

    def test_student_aware_adapt_shortens_and_conditions(self) -> None:
        skill = Skill(
            skill_name="Merged transform heat protocol",
            description="teacher draft",
            precondition="family heat",
            action_protocol=[
                "go to <location>",
                "open <entity>",
                "close <entity>",
                "take <object> from <receptacle>",
                "heat <object> with <tool>",
                "move <object> to <receptacle>",
            ],
            applicable_atomic_ops=["Act"],
            expected_effect="win",
            suggested_role="Executor",
            target_failure_types=[],
            applicable_task_families=["pick_heat_then_place_in_recep"],
            capability_key="transform.heat",
            metadata={
                "primary_task_family": "pick_heat_then_place_in_recep",
                "capability_key": "transform.heat",
                "protocol_form_ok": True,
            },
        )
        adapted = adapt_skill_for_student(
            skill,
            student_failure_notes=["observed frequent terminal step `go to <location>` (n=3)"],
        )
        self.assertTrue(adapted.metadata["student_aware_adapted"])
        self.assertFalse(adapted.metadata["inject_ready"])
        self.assertTrue(any(step.startswith("then ") for step in adapted.action_protocol))
        self.assertIn("heat <object> with <tool>", " ".join(adapted.action_protocol))
        self.assertTrue(
            any("observed frequent terminal step" in step for step in adapted.action_protocol)
        )

    def test_contrastive_patch_from_paired_trajectories(self) -> None:
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
            "game_a",
        )
        student = _traj(
            "pick_heat_then_place_in_recep",
            False,
            [
                "go to fridge 1",
                "open fridge 1",
                "go to cabinet 1",
                "go to fridge 1",
            ],
            "game_a",
        )
        pairs = pair_teacher_wins_student_fails([teacher], [student])
        self.assertEqual(len(pairs), 1)
        patches = distill_contrastive_patches(
            [teacher],
            [student],
            draft_skills=[
                Skill(
                    skill_name="Merged transform heat protocol",
                    description="d",
                    precondition="p",
                    action_protocol=["heat <object> with <tool>"],
                    applicable_atomic_ops=["Act"],
                    expected_effect="e",
                    suggested_role="Executor",
                    target_failure_types=[],
                    applicable_task_families=["pick_heat_then_place_in_recep"],
                    capability_key="transform.heat",
                    metadata={
                        "primary_task_family": "pick_heat_then_place_in_recep",
                        "capability_key": "transform.heat",
                    },
                )
            ],
        )
        self.assertEqual(len(patches), 1)
        patch = patches[0]
        self.assertEqual(patch.metadata["source_signal"], "contrastive_student_patch")
        self.assertFalse(patch.metadata["inject_ready"])
        self.assertTrue(any("then " in step for step in patch.action_protocol))

    def test_revise_appends_stall_cues(self) -> None:
        skill = Skill(
            skill_name="Student-adapted: heat",
            description="d",
            precondition="p",
            action_protocol=["then heat <object> with <tool>"],
            applicable_atomic_ops=["Act"],
            expected_effect="e",
            suggested_role="Executor",
            target_failure_types=[],
            applicable_task_families=["pick_heat_then_place_in_recep"],
            metadata={"primary_task_family": "pick_heat_then_place_in_recep"},
        )
        fails = [
            _traj(
                "pick_heat_then_place_in_recep",
                False,
                ["go to fridge 1", "go to cabinet 1", "go to fridge 1"],
                "g1",
            )
        ]
        revised = revise_skill_from_student_failures(skill, fails)
        self.assertTrue(revised.metadata["skill_revised"])
        self.assertTrue(any(step.startswith("if stalled:") for step in revised.action_protocol))
        self.assertFalse(revised.metadata["inject_ready"])

    def test_org_ready_blocked_when_mu_rejected(self) -> None:
        skill = Skill(
            skill_name="draft",
            description="d",
            precondition="p",
            action_protocol=[
                "go to <location>",
                "open <entity>",
                "take <object> from <receptacle>",
                "heat <object> with <tool>",
                "move <object> to <receptacle>",
            ],
            applicable_atomic_ops=["Act"],
            expected_effect="e",
            suggested_role="Executor",
            target_failure_types=[],
            capability_key="transform.heat",
            metadata={
                "capability_key": "transform.heat",
                "protocol_alignment_ok": True,
                "protocol_structure_ok": True,
                "protocol_form_ok": True,
                "protocol_alignment_support": 3,
                "inject_ready": False,
                "mu_rejected": True,
            },
        )
        self.assertFalse(protocol_is_org_ready(skill))


if __name__ == "__main__":
    unittest.main()
