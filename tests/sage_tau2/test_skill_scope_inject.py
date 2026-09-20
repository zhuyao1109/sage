"""Tests for skill-episode matching, scoped inject, and credit gating."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sage_tau2.credit import (
    CreditPolicy,
    apply_credit_for_episode,
    initialize_credit,
)
from sage_tau2.injection import offer_skills_for_domain
from sage_tau2.schemas import SkillStatus, Tau2Skill, Tau2Trajectory
from sage_tau2.seed_skills import (
    TELECOM_ENABLE_ROAMING_SKILL_ID,
    TELECOM_REFUEL_DATA_SKILL_ID,
    default_telecom_seed_skills,
    resolve_seed_skills,
)
from sage_tau2.skill_bank import Tau2SkillBank
from sage_tau2.task_context import (
    skill_credit_should_apply,
    skill_matches_episode,
)


def _roaming_skill() -> Tau2Skill:
    skill = Tau2Skill(
        skill_name="enable roaming skill",
        description="enable roaming",
        precondition="abroad",
        action_protocol=[
            "get_customer_by_phone(phone_number=?)",
            "get_details_by_id(id=?)",
            "enable_roaming(customer_id=?, line_id=?)",
        ],
        expected_effect="roaming on",
        capability_key="tau2.enable_roaming",
        domain="telecom",
        status=SkillStatus.VERIFIED,
        support_count=2,
        metadata={
            "primary_task_family": "enable_roaming",
            "task_families": ["enable_roaming", "mobile_data_issue"],
        },
    )
    initialize_credit(skill)
    credit = skill.metadata["skill_credit"]
    credit["uses"] = 2
    credit["successes"] = 2
    credit["score"] = 0.75
    skill.metadata["utility"] = 0.75
    return skill


def _traj(
    task_id: str,
    *,
    reward: float,
    protocol: list[str] | None = None,
) -> Tau2Trajectory:
    return Tau2Trajectory(
        task_id=task_id,
        trial=0,
        domain="telecom",
        reward=reward,
        db_reward=reward,
        communicate_reward=None,
        db_match=reward >= 1.0,
        termination_reason="user_stop",
        tool_protocol=protocol
        or [
            "get_customer_by_phone(phone_number=?)",
            "get_details_by_id(id=?)",
            "enable_roaming(customer_id=?, line_id=?)",
        ],
        tool_steps=[],
        assistant_texts=[],
        user_texts=[],
    )


class SkillScopeMatchTests(unittest.TestCase):
    def test_roaming_matches_roaming_bug_not_plain_mms(self) -> None:
        skill = _roaming_skill()
        roaming_tid = (
            "[mobile_data_issue]airplane_mode_on|user_abroad_roaming_disabled_on"
            "[PERSONA:Easy]"
        )
        plain_mms = (
            "[mms_issue]airplane_mode_on|break_apn_mms_setting|bad_wifi_calling"
            "[PERSONA:None]"
        )
        self.assertTrue(
            skill_matches_episode(
                skill, domain="telecom", task_id=roaming_tid, episode_scope="mobile_data_issue"
            )
        )
        self.assertFalse(
            skill_matches_episode(
                skill, domain="telecom", task_id=plain_mms, episode_scope="mms_issue"
            )
        )

    def test_roaming_variant_gate_does_not_merge_different_conditions(self) -> None:
        """Evidence used enabled_off; episode uses disabled_on + sibling bugs."""
        skill = Tau2Skill(
            skill_name="mms issue abroad skill",
            description="abroad mms",
            precondition="abroad",
            action_protocol=[
                "get_customer_by_phone(phone_number=?)",
                "get_details_by_id(id=?)",
                "enable_roaming(customer_id=?, line_id=?)",
            ],
            expected_effect="env ok",
            capability_key="tau2.mms_issue_abroad",
            domain="telecom",
            status=SkillStatus.VERIFIED,
            support_count=7,
            metadata={
                "primary_task_family": "mms_issue",
                "task_families": ["mms_issue", "mms_issue_abroad", "enable_roaming"],
                "write_capability_key": "tau2.enable_roaming",
                "parent_capability_key": "tau2.mms_issue_abroad",
                "bug_bucket_mode": "union_with_signature_gate",
                "bug_intersection": ["user_abroad_roaming_enabled_off"],
                "bug_union": [
                    "break_apn_mms_setting",
                    "break_app_storage_permission",
                    "unseat_sim_card",
                    "user_abroad_roaming_enabled_off",
                ],
                "activation_signatures": [
                    [
                        "break_apn_mms_setting",
                        "unseat_sim_card",
                        "user_abroad_roaming_enabled_off",
                    ]
                ],
            },
        )
        initialize_credit(skill)
        fail_like = (
            "[mms_issue]break_apn_mms_setting|break_app_storage_permission|"
            "unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]"
        )
        self.assertFalse(
            skill_matches_episode(
                skill,
                domain="telecom",
                task_id=fail_like,
                episode_scope="mms_issue",
            )
        )
        # Airplane-only must still be rejected (no shared roaming core).
        self.assertFalse(
            skill_matches_episode(
                skill,
                domain="telecom",
                task_id="[mms_issue]airplane_mode_on[PERSONA:None]",
                episode_scope="mms_issue",
            )
        )

    def test_offer_skills_filters_by_task_id(self) -> None:
        with TemporaryDirectory() as tmp:
            bank = Tau2SkillBank(Path(tmp) / "bank.json")
            bank.extend([_roaming_skill()])
            plain_mms = (
                "[mms_issue]airplane_mode_on|break_apn_mms_setting[PERSONA:None]"
            )
            roaming = (
                "[mms_issue]data_mode_off|user_abroad_roaming_disabled_on[PERSONA:Easy]"
            )
            none = offer_skills_for_domain(
                bank,
                domain="telecom",
                max_skills=2,
                allow_provisional=True,
                same_domain_only=True,
                task_id=plain_mms,
                episode_scope="mms_issue",
            )
            some = offer_skills_for_domain(
                bank,
                domain="telecom",
                max_skills=2,
                allow_provisional=True,
                same_domain_only=True,
                task_id=roaming,
                episode_scope="mms_issue",
            )
            self.assertEqual(none, [])
            self.assertEqual(len(some), 1)

    def test_credit_skips_multi_bug_failure(self) -> None:
        skill = _roaming_skill()
        hard = (
            "[mms_issue]data_usage_exceeded|user_abroad_roaming_disabled_on"
            "[PERSONA:Hard]"
        )
        self.assertFalse(
            skill_credit_should_apply(
                skill, domain="telecom", task_id=hard, success=False
            )
        )
        sole = "[mobile_data_issue]user_abroad_roaming_disabled_on[PERSONA:Easy]"
        self.assertTrue(
            skill_credit_should_apply(
                skill, domain="telecom", task_id=sole, success=False
            )
        )

        events = apply_credit_for_episode(
            [skill],
            _traj(hard, reward=0.0),
            injected_skill_ids=[skill.skill_id],
            policy=CreditPolicy(gate_irrelevant=True, prune_score=0.2),
        )
        self.assertEqual(events, [])
        self.assertEqual(skill.metadata["skill_credit"]["uses"], 2)

        events_ok = apply_credit_for_episode(
            [skill],
            _traj(sole, reward=0.0),
            injected_skill_ids=[skill.skill_id],
            policy=CreditPolicy(gate_irrelevant=True),
        )
        self.assertEqual(len(events_ok), 1)
        self.assertEqual(skill.metadata["skill_credit"]["uses"], 3)


class TelecomSeedTests(unittest.TestCase):
    def test_resolve_telecom_seeds(self) -> None:
        skills = resolve_seed_skills("telecom", domain="telecom")
        ids = {s.skill_id for s in skills}
        self.assertEqual(
            ids, {TELECOM_ENABLE_ROAMING_SKILL_ID, TELECOM_REFUEL_DATA_SKILL_ID}
        )
        for s in default_telecom_seed_skills():
            self.assertEqual(s.status, SkillStatus.VERIFIED)
            self.assertIn("id=?", " ".join(s.action_protocol))


if __name__ == "__main__":
    unittest.main()
