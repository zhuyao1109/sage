"""Seed skill bank helpers for τ² ablations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sage_tau2.injection import format_skills_for_prompt, offer_skills_for_domain
from sage_tau2.schemas import SkillStatus
from sage_tau2.seed_skills import (
    AIRLINE_BALANCE_SPLIT_SKILL_ID,
    AIRLINE_PASSENGER_DOB_SKILL_ID,
    TELECOM_ENABLE_ROAMING_SKILL_ID,
    TELECOM_REFUEL_DATA_SKILL_ID,
    airline_balance_split_skill,
    airline_passenger_dob_skill,
    default_telecom_seed_skills,
    ensure_seed_skills_in_bank,
    resolve_seed_skills,
    write_seed_bank,
)
from sage_tau2.skill_bank import Tau2SkillBank


class SeedSkillsTests(unittest.TestCase):
    def test_balance_split_skill_is_verified_airline(self) -> None:
        skill = airline_balance_split_skill()
        self.assertEqual(skill.skill_id, AIRLINE_BALANCE_SPLIT_SKILL_ID)
        self.assertEqual(skill.domain, "airline")
        self.assertEqual(skill.status, SkillStatus.VERIFIED)
        self.assertIn("gift_card", " ".join(skill.action_protocol))
        self.assertIn("certificate", " ".join(skill.action_protocol))
        self.assertIn("both category totals", skill.expected_effect.lower())

    def test_write_seed_bank_injectable_without_provisional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill_bank.json"
            write_seed_bank(path, skills=[airline_balance_split_skill(), airline_passenger_dob_skill()])
            bank = Tau2SkillBank(path)
            offered = offer_skills_for_domain(
                bank,
                domain="airline",
                max_skills=2,
                allow_provisional=False,
                same_domain_only=True,
            )
            self.assertEqual(len(offered), 2)
            ids = {s.skill_id for s in offered}
            self.assertEqual(
                ids,
                {AIRLINE_BALANCE_SPLIT_SKILL_ID, AIRLINE_PASSENGER_DOB_SKILL_ID},
            )
            text = format_skills_for_prompt(offered)
            self.assertIn("report_gift_card_and_certificate_balances_separately", text)
            self.assertIn("use_dob_field_on_passenger_writes", text)
            self.assertIn("two separate", text)
            self.assertIn("Protocol (follow this full card):", text)

    def test_passenger_dob_skill_mentions_schema_field(self) -> None:
        skill = airline_passenger_dob_skill()
        self.assertEqual(skill.skill_id, AIRLINE_PASSENGER_DOB_SKILL_ID)
        self.assertIn("dob", skill.description)
        self.assertNotIn("date_of_birth", " ".join(skill.action_protocol))

    def test_ensure_seed_merges_into_empty_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill_bank.json"
            Tau2SkillBank(path).save()
            bank = ensure_seed_skills_in_bank(
                path,
                skills=[airline_balance_split_skill(), airline_passenger_dob_skill()],
            )
            self.assertEqual(len(bank.active()), 2)
            self.assertTrue(all(s.status == SkillStatus.VERIFIED for s in bank.active()))

    def test_telecom_seed_pack(self) -> None:
        skills = resolve_seed_skills(True, domain="telecom")
        self.assertEqual(len(skills), 2)
        ids = {s.skill_id for s in skills}
        self.assertEqual(
            ids, {TELECOM_ENABLE_ROAMING_SKILL_ID, TELECOM_REFUEL_DATA_SKILL_ID}
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill_bank.json"
            write_seed_bank(path, skills=default_telecom_seed_skills())
            bank = Tau2SkillBank(path)
            offered = offer_skills_for_domain(
                bank,
                domain="telecom",
                max_skills=2,
                allow_provisional=False,
                same_domain_only=True,
                task_id="[mobile_data_issue]user_abroad_roaming_disabled_on[PERSONA:None]",
                episode_scope="mobile_data_issue",
            )
            self.assertTrue(any(s.skill_id == TELECOM_ENABLE_ROAMING_SKILL_ID for s in offered))


if __name__ == "__main__":
    unittest.main()
