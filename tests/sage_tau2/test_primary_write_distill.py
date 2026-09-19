"""Protocol helpers retained after sharp primary-write distill removal."""

from __future__ import annotations

import unittest

from sage_tau2.credit import initialize_credit
from sage_tau2.distill import (
    bind_protocol_slots,
    bind_write_args_from_evidence,
    distill_skills_from_trajectories,
    reorder_protocol_hard_writes_first,
    stable_guide_names_from_trajs,
)
from sage_tau2.schemas import SkillStatus, ToolCallStep, Tau2Skill, Tau2Trajectory
from sage_tau2.task_context import episode_skill_coverage_score


def _traj(
    task_id: str,
    *,
    protocol: list[str],
    guides: list[str] | None = None,
    tool_steps: list[ToolCallStep] | None = None,
) -> Tau2Trajectory:
    checks = []
    for name in protocol:
        n = name.split("(", 1)[0]
        if n.startswith("get_"):
            continue
        checks.append(
            {
                "action": {"name": n, "requestor": "assistant"},
                "action_match": True,
            }
        )
    for g in guides or []:
        checks.append(
            {
                "action": {"name": g, "requestor": "user"},
                "action_match": True,
            }
        )
    return Tau2Trajectory(
        task_id=task_id,
        trial=0,
        domain="telecom-workflow",
        reward=1.0,
        db_reward=1.0,
        communicate_reward=None,
        db_match=True,
        termination_reason="user_stop",
        tool_protocol=protocol,
        tool_steps=tool_steps or [],
        assistant_texts=[],
        user_texts=["help"],
        metadata={"action_checks": checks, "success_mode": "solve_write"},
    )


class ProtocolHelperTests(unittest.TestCase):
    def test_stable_guides_threshold(self) -> None:
        trajs = [
            _traj("a", protocol=["refuel_data()"], guides=["toggle_data", "x"]),
            _traj("b", protocol=["refuel_data()"], guides=["toggle_data"]),
            _traj("c", protocol=["refuel_data()"], guides=["toggle_data"]),
        ]
        names = stable_guide_names_from_trajs(trajs, min_fraction=0.67)
        self.assertEqual(names, ["toggle_data"])

    def test_bind_write_args(self) -> None:
        traj = _traj(
            "t",
            protocol=["refuel_data(customer_id=?, gb_amount=?, line_id=?)"],
            tool_steps=[
                ToolCallStep(
                    name="refuel_data",
                    arguments={"gb_amount": 8, "customer_id": "C", "line_id": "L"},
                )
            ],
        )
        out = bind_write_args_from_evidence(
            ["refuel_data(customer_id=?, gb_amount=?, line_id=?)"],
            [traj],
        )
        self.assertIn("gb_amount=8", out[0])
        self.assertIn("never_default_first_listed_line", out[0])
        self.assertNotIn("line_id=L", out[0])

    def test_bind_protocol_slots_line_id(self) -> None:
        proto = bind_protocol_slots(
            [
                "get_customer_by_phone(phone_number=?)",
                "get_details_by_id(id=?)",
                "get_data_usage(customer_id=?, line_id=?)",
                "refuel_data(customer_id=?, gb_amount=2, line_id=?)",
            ],
            resolved_line_id=True,
        )
        joined = " | ".join(proto)
        self.assertIn("do_not_pass_guessed_line_id", joined)
        self.assertIn("never_default_first_listed_line", joined)
        self.assertNotIn("line_id=<resolved_line_id>", joined)

    def test_payment_order_make_payment_before_resume(self) -> None:
        ordered = reorder_protocol_hard_writes_first(
            [
                "get_bills_for_customer(customer_id=?)",
                "resume_line(customer_id=?, line_id=?)",
                "send_payment_request(bill_id=?, customer_id=?)",
                "guide user: make_payment",
                "guide user: reseat_sim_card",
            ]
        )
        names = []
        for step in ordered:
            low = step.lower()
            if low.startswith("guide user:"):
                names.append(low.split(":", 1)[1].strip())
            else:
                names.append(step.split("(", 1)[0])
        self.assertEqual(
            names,
            [
                "get_bills_for_customer",
                "send_payment_request",
                "make_payment",
                "resume_line",
                "reseat_sim_card",
            ],
        )

    def test_primary_write_mode_aliases_to_union_gate(self) -> None:
        """Deprecated primary_write mode must not sharpen / drop sibling writes."""
        t1 = _traj(
            "[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]",
            protocol=[
                "get_customer_by_phone(phone_number=?)",
                "enable_roaming(customer_id=?, line_id=?)",
                "get_data_usage(customer_id=?, line_id=?)",
                "refuel_data(customer_id=?, gb_amount=?, line_id=?)",
            ],
            guides=["toggle_data"],
            tool_steps=[
                ToolCallStep(
                    name="refuel_data",
                    arguments={"customer_id": "C1", "line_id": "L1", "gb_amount": 5},
                )
            ],
        )
        skills = distill_skills_from_trajectories(
            [t1],
            min_support=1,
            bug_bucket_mode="primary_write",
            require_action_checks_ok=True,
            max_skills=8,
        )
        self.assertTrue(skills)
        skill = skills[0]
        self.assertFalse(skill.metadata.get("sharp_card"))
        self.assertEqual(
            skill.metadata.get("bug_bucket_mode"), "union_with_signature_gate"
        )
        joined = " | ".join(skill.action_protocol)
        # Keep the observed protocol spine; do not strip enable_roaming.
        self.assertIn("refuel_data", joined)
        self.assertIn("enable_roaming", joined)

    def test_coverage_prefers_matching_bug_signature(self) -> None:
        narrow = Tau2Skill(
            skill_name="refuel narrow",
            description="r",
            precondition="p",
            action_protocol=["refuel_data(customer_id=?, line_id=?)"],
            expected_effect="e",
            capability_key="tau2.refuel_data",
            status=SkillStatus.PROVISIONAL,
            support_count=1,
            evidence_ids=[],
            domain="telecom-workflow",
            metadata={
                "primary_write": "refuel_data",
                "write_capability_key": "tau2.refuel_data",
                "bug_signature": ["data_usage_exceeded"],
            },
        )
        rich = Tau2Skill(
            skill_name="refuel rich",
            description="r",
            precondition="p",
            action_protocol=["refuel_data(customer_id=?, line_id=?)"],
            expected_effect="e",
            capability_key="tau2.refuel_data",
            status=SkillStatus.PROVISIONAL,
            support_count=1,
            evidence_ids=[],
            domain="telecom-workflow",
            metadata={
                "primary_write": "refuel_data",
                "write_capability_key": "tau2.refuel_data",
                "bug_signature": [
                    "data_mode_off",
                    "data_usage_exceeded",
                ],
            },
        )
        initialize_credit(narrow)
        initialize_credit(rich)
        tid = "[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]"
        self.assertGreater(
            episode_skill_coverage_score(rich, task_id=tid),
            episode_skill_coverage_score(narrow, task_id=tid),
        )


if __name__ == "__main__":
    unittest.main()
