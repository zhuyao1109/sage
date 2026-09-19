"""Tests for multi-domain skill injection isolation."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sage_tau2.credit import initialize_credit
from sage_tau2.injection import offer_skills_for_domain
from sage_tau2.schemas import SkillStatus, Tau2Skill
from sage_tau2.skill_bank import Tau2SkillBank


def _skill(*, name: str, domain: str, status: SkillStatus = SkillStatus.VERIFIED) -> Tau2Skill:
    skill = Tau2Skill(
        skill_name=name,
        description=name,
        precondition="p",
        action_protocol=[f"{name}_tool(x=?)"],
        expected_effect="e",
        capability_key=f"tau2.{name}",
        status=status,
        support_count=1,
        domain=domain,
    )
    initialize_credit(skill)
    credit = skill.metadata["skill_credit"]
    credit["uses"] = 2
    credit["successes"] = 2
    credit["score"] = 0.75
    skill.metadata["utility"] = 0.75
    return skill


class SameDomainInjectTests(unittest.TestCase):
    def test_does_not_fall_back_to_other_domains(self) -> None:
        with TemporaryDirectory() as tmp:
            bank = Tau2SkillBank(Path(tmp) / "bank.json")
            bank.extend(
                [
                    _skill(name="return_items", domain="retail"),
                    _skill(name="cancel_order", domain="retail"),
                ]
            )
            offered = offer_skills_for_domain(
                bank,
                domain="telecom",
                max_skills=2,
                allow_provisional=True,
                same_domain_only=True,
            )
            self.assertEqual(offered, [])

    def test_offers_same_domain_skills(self) -> None:
        with TemporaryDirectory() as tmp:
            bank = Tau2SkillBank(Path(tmp) / "bank.json")
            bank.extend(
                [
                    _skill(name="return_items", domain="retail"),
                    _skill(name="enable_roaming", domain="telecom"),
                ]
            )
            offered = offer_skills_for_domain(
                bank,
                domain="telecom",
                max_skills=2,
                allow_provisional=True,
                same_domain_only=True,
            )
            self.assertEqual(len(offered), 1)
            self.assertEqual(offered[0].domain, "telecom")

    def test_prior_segment_snapshot_filters_new_skills(self) -> None:
        with TemporaryDirectory() as tmp:
            bank = Tau2SkillBank(Path(tmp) / "bank.json")
            old = _skill(name="enable_roaming", domain="telecom")
            bank.extend([old])
            # Simulate a skill distilled later in the same segment.
            new = _skill(name="reset_apn", domain="telecom")
            bank.extend([new])
            offered = offer_skills_for_domain(
                bank,
                domain="telecom",
                max_skills=4,
                allow_provisional=True,
                same_domain_only=True,
                allowed_skill_ids={old.skill_id},
            )
            self.assertEqual(len(offered), 1)
            self.assertEqual(offered[0].skill_id, old.skill_id)

    def test_runtime_executor_skills_respect_same_domain(self) -> None:
        """Agent Executor path must use offer_skills_for_domain semantics.

        Importing sage_tau2.agent pulls τ² deps; assert wiring via source and
        exercise the shared offer helper that agent now calls.
        """
        agent_src = Path(__file__).resolve().parents[2] / "sage_tau2" / "agent.py"
        text = agent_src.read_text(encoding="utf-8")
        self.assertIn("offer_skills_for_domain", text)
        self.assertIn("inject_same_domain_only", text)
        self.assertIn("same_domain_only=same_domain_only", text)

        with TemporaryDirectory() as tmp:
            bank = Tau2SkillBank(Path(tmp) / "bank.json")
            bank.extend(
                [
                    _skill(name="return_items", domain="retail"),
                    _skill(name="cancel_order", domain="retail"),
                    _skill(name="enable_roaming", domain="telecom"),
                ]
            )
            # Mirrors _skills_for_primary(Executor, domain=airline, same_domain_only=True)
            airline = offer_skills_for_domain(
                bank,
                domain="airline",
                max_skills=2,
                allow_provisional=True,
                same_domain_only=True,
            )
            self.assertEqual(airline, [])
            telecom = offer_skills_for_domain(
                bank,
                domain="telecom",
                max_skills=2,
                allow_provisional=True,
                same_domain_only=True,
            )
            self.assertEqual(len(telecom), 1)
            self.assertEqual(telecom[0].domain, "telecom")
            cross = offer_skills_for_domain(
                bank,
                domain="airline",
                max_skills=2,
                allow_provisional=True,
                same_domain_only=False,
            )
            self.assertEqual(len(cross), 2)


if __name__ == "__main__":
    unittest.main()

