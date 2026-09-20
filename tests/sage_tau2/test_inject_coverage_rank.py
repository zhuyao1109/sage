"""Matching v1: coverage ranking + write dedupe; Required card format."""

from __future__ import annotations

import unittest

from sage_tau2.credit import initialize_credit, injectable_skills
from sage_tau2.injection import format_skills_for_prompt, offer_skills_for_domain
from sage_tau2.schemas import SkillStatus, Tau2Skill
from sage_tau2.skill_bank import Tau2SkillBank
from sage_tau2.task_context import (
    episode_skill_coverage_score,
    hinted_agent_writes_from_task_id,
)


def _skill(
    name: str,
    protocol: list[str],
    *,
    domain: str = "telecom",
    score: float = 0.5,
    primary: str = "",
) -> Tau2Skill:
    skill = Tau2Skill(
        skill_name=name,
        description=name,
        precondition="test",
        action_protocol=protocol,
        expected_effect="ok",
        capability_key=f"tau2.{name}",
        domain=domain,
        status=SkillStatus.VERIFIED_LOW_SUPPORT,
        support_count=1,
        metadata={"primary_task_family": primary or "mobile_data_issue"},
    )
    initialize_credit(skill)
    skill.metadata["skill_credit"]["score"] = score
    skill.metadata["utility"] = score
    return skill


class CoverageRankTests(unittest.TestCase):
    def test_bill_hint_and_payment_outranks_lookup(self) -> None:
        tid = (
            "[service_issue]airplane_mode_on|overdue_bill_suspension|unseat_sim_card"
            "[PERSONA:Easy]"
        )
        self.assertIn("send_payment_request", hinted_agent_writes_from_task_id(tid))
        pay = _skill(
            "send payment request skill",
            [
                "get_customer_by_phone(phone_number=?)",
                "get_bills_for_customer(customer_id=?)",
                "send_payment_request(bill_id=?)",
            ],
            score=0.4,
            primary="service_issue",
        )
        lookup = _skill(
            "get customer by phone get details by id skill",
            [
                "get_customer_by_phone(phone_number=?)",
                "get_details_by_id(id=?)",
                "guide user: toggle_airplane_mode",
                "guide user: reseat_sim_card",
            ],
            score=0.9,
            primary="service_issue",
        )
        self.assertGreater(
            episode_skill_coverage_score(pay, task_id=tid),
            episode_skill_coverage_score(lookup, task_id=tid),
        )
        picked = injectable_skills(
            [lookup, pay], max_skills=1, allow_provisional=True, task_id=tid
        )
        self.assertEqual(len(picked), 1)
        self.assertIn("payment", picked[0].skill_name)

    def test_refuel_outranks_duplicate_roaming_when_usage_hinted(self) -> None:
        tid = (
            "[mobile_data_issue]data_mode_off|data_usage_exceeded"
            "[PERSONA:None]"
        )
        roam_a = _skill(
            "enable roaming skill (n=3)",
            [
                "get_customer_by_phone(phone_number=?)",
                "enable_roaming(customer_id=?, line_id=?)",
                "refuel_data(customer_id=?, gb_amount=?, line_id=?)",
            ],
            score=0.9,
        )
        roam_b = _skill(
            "enable roaming skill (n=2)",
            [
                "get_customer_by_phone(phone_number=?)",
                "enable_roaming(customer_id=?, line_id=?)",
            ],
            score=0.85,
        )
        refuel = _skill(
            "refuel data skill (n=1)",
            [
                "get_customer_by_phone(phone_number=?)",
                "get_data_usage(customer_id=?, line_id=?)",
                "refuel_data(customer_id=?, gb_amount=?, line_id=?)",
            ],
            score=0.4,
        )
        # Both roam_a and refuel cover refuel_data; dedupe keeps one enable_roaming.
        picked = injectable_skills(
            [roam_a, roam_b, refuel],
            max_skills=2,
            allow_provisional=True,
            task_id=tid,
        )
        names = [s.skill_name for s in picked]
        self.assertEqual(len(picked), 2)
        self.assertEqual(len([n for n in names if "roaming" in n]), 1)
        self.assertTrue(any("refuel" in n or "roaming" in n for n in names))
        # With max_skills=1 and only usage hint, prefer sharp refuel over roam+refuel.
        top = injectable_skills(
            [roam_a, roam_b, refuel],
            max_skills=1,
            allow_provisional=True,
            task_id=tid,
        )
        self.assertEqual(len(top), 1)
        self.assertIn("refuel", top[0].skill_name)

    def test_offer_passes_task_id_ranking(self) -> None:
        from tempfile import TemporaryDirectory

        tid = (
            "[service_issue]overdue_bill_suspension|unseat_sim_card[PERSONA:Easy]"
        )
        with TemporaryDirectory() as tmp:
            bank = Tau2SkillBank(tmp + "/bank.json")
            bank.add(
                _skill(
                    "lookup",
                    [
                        "get_customer_by_phone(phone_number=?)",
                        "guide user: reseat_sim_card",
                    ],
                    score=0.99,
                    primary="service_issue",
                )
            )
            bank.add(
                _skill(
                    "pay",
                    [
                        "get_customer_by_phone(phone_number=?)",
                        "send_payment_request(bill_id=?)",
                    ],
                    score=0.2,
                    primary="service_issue",
                )
            )
            offered = offer_skills_for_domain(
                bank,
                domain="telecom",
                max_skills=1,
                allow_provisional=True,
                same_domain_only=True,
                task_id=tid,
                episode_scope="service_issue",
            )
            self.assertEqual(len(offered), 1)
            self.assertEqual(offered[0].skill_name, "pay")


class RequiredCardFormatTests(unittest.TestCase):
    def test_ordered_required_lists(self) -> None:
        skill = _skill(
            "refuel data skill",
            [
                "get_customer_by_phone(phone_number=?)",
                "refuel_data(customer_id=?, gb_amount=?, line_id=?)",
                "guide user: toggle_data",
            ],
        )
        text = format_skills_for_prompt([skill])
        self.assertIn("Protocol (preserve this order, including user actions):", text)
        self.assertIn("1. get_customer_by_phone", text)
        self.assertIn("2. refuel_data", text)
        self.assertIn("candidates, not mandatory actions", text)
        self.assertIn("3. guide user: toggle_data", text)
        self.assertIn("guide user: toggle_data", text)
        self.assertIn("Verify:", text)
        self.assertNotIn("Protocol (follow this full card):", text)


if __name__ == "__main__":
    unittest.main()
