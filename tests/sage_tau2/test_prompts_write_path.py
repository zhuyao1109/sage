"""Prompt soft-rule checks: thin instructions + GiGPO/ALFWorld window format."""

from __future__ import annotations

import unittest

from sage_tau2.injection import (
    append_skills_to_user_prompt,
    format_skills_for_prompt,
)
from sage_tau2.prompts import (
    EXECUTOR_INSTRUCTION,
    SPECIALIST_INSTRUCTION,
    SYSTEM_PROMPT,
    TAU2_TEMPLATE,
    TAU2_TEMPLATE_NO_HIS,
    instruction_for_agent,
)
from sage_tau2.schemas import SkillStatus, Tau2Skill


class PromptWritePathTests(unittest.TestCase):
    def test_executor_keeps_env_contract_only(self) -> None:
        text = EXECUTOR_INSTRUCTION.lower()
        self.assertIn("only call tools from your agent toolkit", text)
        self.assertIn("active skill patch", text)
        self.assertIn("domain <policy> is binding", text)
        # No hand-written escalate / confirm-before-write recipes.
        self.assertNotIn("genuine attempt to resolve", text)
        self.assertNotIn("looking up the customer", text)
        self.assertNotIn("confirm before", text)
        self.assertNotIn("read before write", text)
        self.assertNotIn("product-level", text)

    def test_specialist_mirrors_thin_contract(self) -> None:
        text = SPECIALIST_INSTRUCTION.lower()
        self.assertIn("only call tools from your agent toolkit", text)
        self.assertNotIn("do not escalate after only a lookup", text)
        self.assertNotIn("confirm before writes", text)

    def test_single_instruction_family_all_domains(self) -> None:
        self.assertEqual(
            instruction_for_agent(domain="telecom-workflow", specialist=False),
            EXECUTOR_INSTRUCTION,
        )
        self.assertEqual(
            instruction_for_agent(domain="retail", specialist=False),
            EXECUTOR_INSTRUCTION,
        )
        self.assertEqual(
            instruction_for_agent(domain="telecom", specialist=True),
            SPECIALIST_INSTRUCTION,
        )

    def test_system_has_no_skills_slot(self) -> None:
        self.assertNotIn("{skills_block}", SYSTEM_PROMPT)
        self.assertIn("{domain_policy}", SYSTEM_PROMPT)
        self.assertIn("<instructions>", SYSTEM_PROMPT)

    def test_window_templates_match_gigpo_shell(self) -> None:
        for tmpl in (TAU2_TEMPLATE, TAU2_TEMPLATE_NO_HIS):
            self.assertIn("You are an expert agent operating in the τ²", tmpl)
            self.assertIn("{task_description}", tmpl)
            self.assertIn("{current_observation}", tmpl)
            self.assertIn("<think>", tmpl)
            self.assertIn("MUST", tmpl)
            self.assertNotIn("{available_tools}", tmpl)
            self.assertNotIn("{admissible_actions}", tmpl)
        self.assertIn(
            "Prior to this step, you have already taken {step_count}", TAU2_TEMPLATE
        )
        self.assertIn("You are now at step {current_step}", TAU2_TEMPLATE)

    def test_skills_format_is_alfworld_patch(self) -> None:
        skill = Tau2Skill(
            skill_name="enable roaming skill (n=2)",
            description="test",
            precondition="user abroad",
            action_protocol=["get_customer_by_phone(phone_number=?)", "enable_roaming(customer_id=?, line_id=?)"],
            expected_effect="roaming on",
            capability_key="tau2.enable_roaming",
            status=SkillStatus.PROVISIONAL,
            domain="telecom",
        )
        text = format_skills_for_prompt([skill])
        self.assertIn("Active skill patch", text)
        self.assertIn("- enable roaming skill (n=2)", text)
        self.assertIn("Required agent tools (in order):", text)
        self.assertIn("enable_roaming", text)
        self.assertNotIn("<available_skills>", text)
        self.assertNotIn("confirm before writes", text)
        self.assertEqual(format_skills_for_prompt([]), "")
        joined = append_skills_to_user_prompt("WINDOW", text)
        self.assertTrue(joined.startswith("WINDOW\n\nActive skill patch"))

        skill.metadata["dialogue_gates"] = ["confirm before enable_roaming"]
        skill.metadata["user_side_hints"] = ["user: turn airplane mode OFF"]
        skill.metadata["inline_dialogue_in_protocol"] = False
        layered = format_skills_for_prompt([skill])
        self.assertIn("Dialogue gates:", layered)
        self.assertIn("User-side hints", layered)
        legacy = format_skills_for_prompt([skill], layered=False)
        self.assertIn("Protocol:", legacy)
        self.assertNotIn("Required agent tools (in order):", legacy)

if __name__ == "__main__":
    unittest.main()
