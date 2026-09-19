"""Birth seed / online prune / inject gates (ALFWorld-aligned spirit)."""

from __future__ import annotations

import unittest

from sage_tau2.credit import (
    CreditPolicy,
    apply_credit_for_episode,
    initialize_credit,
    injectable_skills,
    seed_credit_from_birth_support,
    skill_allowed_for_inject,
)
from sage_tau2.schemas import SkillStatus, Tau2Skill, Tau2Trajectory
from sage_tau2.task_context import normalize_bug_tag


def _roam_skill(*, support: int = 4) -> Tau2Skill:
    skill = Tau2Skill(
        skill_name="enable roaming skill",
        description="r",
        precondition="p",
        action_protocol=[
            "get_customer_by_phone(phone_number=?)",
            "enable_roaming(customer_id=?, line_id=?)",
        ],
        expected_effect="e",
        capability_key="tau2.enable_roaming",
        status=SkillStatus.CANDIDATE,
        support_count=support,
        evidence_ids=[f"e{i}" for i in range(support)],
        domain="telecom",
        metadata={
            "primary_write": "enable_roaming",
            "agent_tool_protocol": [
                "get_customer_by_phone(phone_number=?)",
                "enable_roaming(customer_id=?, line_id=?)",
            ],
            "success_mode": "solve_write",
        },
    )
    initialize_credit(skill)
    return skill


def _fail_traj(task_id: str) -> Tau2Trajectory:
    return Tau2Trajectory(
        task_id=task_id,
        trial=0,
        domain="telecom",
        reward=0.0,
        db_reward=0.0,
        communicate_reward=None,
        db_match=False,
        termination_reason="user_stop",
        tool_protocol=[
            "get_customer_by_phone(phone_number=?)",
            "enable_roaming(customer_id=?, line_id=?)",
        ],
        tool_steps=[],
        assistant_texts=[],
        user_texts=[],
        metadata={
            "success_mode": "solve_write",
            "action_checks": [
                {
                    "action": {
                        "name": "enable_roaming",
                        "requestor": "assistant",
                        "arguments": {"customer_id": "C1001", "line_id": "L1001"},
                    },
                    "action_match": False,
                }
            ],
        },
    )


class BirthAndOnlineCreditTests(unittest.TestCase):
    def test_birth_low_support_stays_provisional(self) -> None:
        skill = _roam_skill(support=4)
        seed_credit_from_birth_support(skill)
        self.assertEqual(skill.status, SkillStatus.PROVISIONAL)
        self.assertTrue(skill.metadata.get("credit_seeded_from_birth"))
        self.assertEqual(skill.metadata["skill_credit"]["birth_uses"], 4)

    def test_birth_high_support_can_verify(self) -> None:
        skill = _roam_skill(support=5)
        seed_credit_from_birth_support(
            skill, policy=CreditPolicy(min_support_for_verified=5)
        )
        self.assertEqual(skill.status, SkillStatus.VERIFIED)

    def test_online_fails_retire_despite_birth_score(self) -> None:
        skill = _roam_skill(support=4)
        seed_credit_from_birth_support(skill)
        # Birth score stays high (~0.83) but online Laplace collapses.
        tid = "[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:Easy]"
        policy = CreditPolicy(
            min_online_uses_for_pruning=2,
            prune_score=0.35,
            min_online_score_for_inject=0.35,
        )
        for _ in range(2):
            apply_credit_for_episode(
                [skill],
                _fail_traj(tid),
                injected_skill_ids=[skill.skill_id],
                policy=policy,
            )
        credit = skill.metadata["skill_credit"]
        self.assertEqual(credit["online_uses"], 2)
        self.assertEqual(credit["online_successes"], 0)
        self.assertEqual(skill.status, SkillStatus.RETIRED)
        self.assertFalse(
            skill_allowed_for_inject(skill, allow_provisional=True, policy=policy)
        )

    def test_inject_blocks_after_online_fail_streak_before_retire_bar(self) -> None:
        skill = _roam_skill(support=4)
        seed_credit_from_birth_support(skill)
        tid = "[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:Easy]"
        # prune_score very low so we don't retire yet; inject gate still fires.
        policy = CreditPolicy(
            min_online_uses_for_pruning=2,
            prune_score=0.05,
            min_online_score_for_inject=0.40,
        )
        for _ in range(2):
            apply_credit_for_episode(
                [skill],
                _fail_traj(tid),
                injected_skill_ids=[skill.skill_id],
                policy=policy,
            )
        self.assertNotEqual(skill.status, SkillStatus.RETIRED)
        offered = injectable_skills(
            [skill], max_skills=2, allow_provisional=True, policy=policy
        )
        self.assertEqual(offered, [])

    def test_roaming_tags_not_collapsed(self) -> None:
        self.assertEqual(
            normalize_bug_tag("user_abroad_roaming_enabled_off"),
            "user_abroad_roaming_enabled_off",
        )
        self.assertEqual(
            normalize_bug_tag("user_abroad_roaming_disabled_on"),
            "user_abroad_roaming_disabled_on",
        )
        self.assertNotEqual(
            normalize_bug_tag("user_abroad_roaming_enabled_off"),
            normalize_bug_tag("user_abroad_roaming_disabled_on"),
        )


if __name__ == "__main__":
    unittest.main()
