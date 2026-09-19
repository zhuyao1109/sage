"""Arg-level credit: hard write action_match gates success."""

from __future__ import annotations

import unittest

from sage_tau2.credit import (
    apply_credit_for_episode,
    episode_counts_as_credit_success,
    hard_write_arg_match_score,
    initialize_credit,
)
from sage_tau2.schemas import SkillStatus, Tau2Skill, Tau2Trajectory


def _skill() -> Tau2Skill:
    skill = Tau2Skill(
        skill_name="refuel",
        description="r",
        precondition="p",
        action_protocol=[
            "get_data_usage(customer_id=?, line_id=?)",
            "refuel_data(customer_id=?, gb_amount=2, line_id=?)",
        ],
        expected_effect="e",
        capability_key="tau2.refuel_data",
        status=SkillStatus.PROVISIONAL,
        support_count=1,
        evidence_ids=["e1"],
        domain="telecom-workflow",
        metadata={
            "primary_write": "refuel_data",
            "write_capability_key": "tau2.refuel_data",
            "success_mode": "solve_write",
            "agent_tool_protocol": [
                "get_data_usage(customer_id=?, line_id=?)",
                "refuel_data(customer_id=?, gb_amount=2, line_id=?)",
            ],
        },
    )
    initialize_credit(skill)
    return skill


def _traj(*, reward: float, refuel_match: bool) -> Tau2Trajectory:
    return Tau2Trajectory(
        task_id="[mobile_data_issue]data_usage_exceeded[PERSONA:None]",
        trial=0,
        domain="telecom-workflow",
        reward=reward,
        db_reward=1.0 if reward >= 1 else 0.0,
        communicate_reward=None,
        db_match=reward >= 1,
        termination_reason="user_stop",
        tool_protocol=[
            "get_data_usage(customer_id=?, line_id=?)",
            "refuel_data(customer_id=?, gb_amount=?, line_id=?)",
        ],
        tool_steps=[],
        assistant_texts=[],
        user_texts=["help"],
        metadata={
            "success_mode": "solve_write",
            "action_checks": [
                {
                    "action": {
                        "name": "refuel_data",
                        "requestor": "assistant",
                        "arguments": {
                            "customer_id": "C1001",
                            "line_id": "L1002",
                            "gb_amount": 2.0,
                        },
                    },
                    "action_match": refuel_match,
                }
            ],
        },
    )


class ArgCreditTests(unittest.TestCase):
    def test_arg_match_score(self) -> None:
        skill = _skill()
        ok = hard_write_arg_match_score(skill, _traj(reward=1.0, refuel_match=True))
        bad = hard_write_arg_match_score(skill, _traj(reward=0.0, refuel_match=False))
        self.assertEqual(ok, 1.0)
        self.assertEqual(bad, 0.0)

    def test_success_requires_arg_match(self) -> None:
        skill = _skill()
        # Env green but wrong line_id → not credit success
        self.assertFalse(
            episode_counts_as_credit_success(
                skill, _traj(reward=1.0, refuel_match=False)
            )
        )
        self.assertTrue(
            episode_counts_as_credit_success(
                skill, _traj(reward=1.0, refuel_match=True)
            )
        )

    def test_apply_credit_records_arg_match(self) -> None:
        skill = _skill()
        events = apply_credit_for_episode(
            [skill],
            _traj(reward=0.0, refuel_match=False),
            injected_skill_ids=[skill.skill_id],
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["arg_match"], 0.0)
        self.assertFalse(events[0]["success"])


if __name__ == "__main__":
    unittest.main()
