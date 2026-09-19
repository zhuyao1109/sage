"""GiGPO/ALFWorld-shell TAU2_TEMPLATE + think/action projection checks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from sage_tau2.pro_steps import (
    build_live_window_prompt,
    compose_window_prompt,
    format_available_tools,
    lean_window_from_prior_steps,
    messages_to_rich_pro_steps,
)
from sage_tau2.projection import tau2_projection, validate_think_action
from sage_tau2.prompts import TAU2_TEMPLATE, TAU2_TEMPLATE_NO_HIS


class Tau2TemplateTests(unittest.TestCase):
    def test_template_has_gigpo_slots(self) -> None:
        for key in (
            "{task_description}",
            "{step_count}",
            "{history_length}",
            "{action_history}",
            "{current_step}",
            "{current_observation}",
            "{booking_scratchpad_block}",
        ):
            self.assertIn(key, TAU2_TEMPLATE)
        self.assertIn("<think>", TAU2_TEMPLATE)
        self.assertIn("MUST", TAU2_TEMPLATE)
        self.assertIn("Prior to this step", TAU2_TEMPLATE)
        self.assertNotIn("{available_tools}", TAU2_TEMPLATE)
        self.assertNotIn("{admissible_actions}", TAU2_TEMPLATE)
        self.assertIn("{task_description}", TAU2_TEMPLATE_NO_HIS)
        self.assertIn("{current_observation}", TAU2_TEMPLATE_NO_HIS)

    def test_compose_window_is_gigpo_shell(self) -> None:
        prompt = compose_window_prompt(
            history_summary="[Observation 1: '(none)', Action 1: 'Hi!']",
            observation_before="[USER] Cancel my flight.",
            step_count=1,
            current_step=2,
            history_length=1,
            task_description="Cancel two reservations.",
            available_tools="'get_user_details'",  # ignored
        )
        self.assertIn("Cancel two reservations.", prompt)
        self.assertNotIn("get_user_details", prompt)
        self.assertIn("Cancel my flight", prompt)
        self.assertIn("<think>", prompt)
        self.assertIn("Prior to this step", prompt)
        self.assertIn("You are now at step 2", prompt)
        self.assertIn("your current observation is:", prompt)

    def test_compose_first_step_uses_no_his(self) -> None:
        prompt = compose_window_prompt(
            history_summary="(none)",
            observation_before="(none)",
            step_count=0,
            current_step=1,
            history_length=0,
            task_description="Help the customer.",
        )
        self.assertNotIn("Prior to this step", prompt)
        self.assertIn("Your current observation is:", prompt)
        self.assertIn("Help the customer.", prompt)

    def test_format_available_tools(self) -> None:
        tools = [
            SimpleNamespace(name="get_user_details", short_desc="Fetch user profile"),
            SimpleNamespace(name="cancel_reservation", long_desc="Cancel a booking"),
        ]
        text = format_available_tools(tools)
        self.assertIn("'get_user_details': Fetch user profile", text)
        self.assertIn("'cancel_reservation': Cancel a booking", text)

    def test_live_window_uses_gigpo_template(self) -> None:
        msgs = [
            {"role": "assistant", "content": "<think>greet</think>\nHi!"},
            {"role": "user", "content": "Change my flight."},
        ]
        live, meta = build_live_window_prompt(
            msgs, history_window=4, task_description="Change a flight."
        )
        self.assertIn("Your task is to:", live)
        self.assertIn("Change a flight.", live)
        self.assertNotIn("available tools", live.lower())
        self.assertIn("Observation 1", live)
        self.assertIn("Change my flight", live)
        self.assertEqual(meta["current_step"], 2)

    def test_lean_window_from_prior_steps(self) -> None:
        prior = [
            {
                "step": 1,
                "observation_before": "(none)",
                "action": "Hi!",
                "observation": "[USER] Help",
            }
        ]
        text = lean_window_from_prior_steps(
            prior,
            observation_before="[USER] Help",
            history_window=4,
            task_description="Help the user.",
        )
        self.assertIn("Prior to this step", text)
        self.assertIn("Hi!", text)
        self.assertIn("[USER] Help", text)


class Tau2ProjectionTests(unittest.TestCase):
    def test_valid_text_turn(self) -> None:
        msg = {"role": "assistant", "content": "<think>ask id</think>\nWhat is your user id?"}
        ok, reason = validate_think_action(msg)
        self.assertTrue(ok)
        self.assertEqual(reason, "")
        valid, think, action = tau2_projection(msg)
        self.assertTrue(valid)
        self.assertEqual(think, "ask id")
        self.assertIn("user id", action or "")

    def test_valid_tool_turn(self) -> None:
        msg = {
            "role": "assistant",
            "content": "<think>need profile</think>",
            "tool_calls": [
                {"name": "get_user_details", "arguments": {"user_id": "u1"}}
            ],
        }
        ok, reason = validate_think_action(msg)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_missing_think_invalid(self) -> None:
        msg = {
            "role": "assistant",
            "content": "Hi!",
            "tool_calls": [],
        }
        ok, reason = validate_think_action(msg)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_think")

    def test_missing_action_invalid(self) -> None:
        msg = {"role": "assistant", "content": "<think>stuck</think>"}
        ok, reason = validate_think_action(msg)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_action")

    def test_dump_does_not_synthesize_think(self) -> None:
        msgs = [
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Help"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "get_user_details", "arguments": {"user_id": "u"}}
                ],
            },
            {"role": "tool", "id": "c1", "content": '{"user_id":"u"}'},
        ]
        steps = messages_to_rich_pro_steps(msgs, allow_synthesize_think=False)
        self.assertNotIn("think", steps[0])
        self.assertNotIn("think_source", steps[0])
        self.assertNotIn("<think>", steps[0]["agent_messages"][0]["content"])
        self.assertNotIn("<think>", steps[1]["agent_messages"][0]["content"])
        self.assertNotIn("prompt", steps[0])
        self.assertNotIn("history_summary", steps[0])
        # Opt-in synthesis still embeds fallback think into the reply.
        synth = messages_to_rich_pro_steps(msgs, allow_synthesize_think=True)
        self.assertTrue(all("<think>" in s["agent_messages"][0]["content"] for s in synth))


if __name__ == "__main__":
    unittest.main()
