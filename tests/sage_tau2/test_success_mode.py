"""Tests for τ² success_mode classification."""

from __future__ import annotations

import unittest

from sage_tau2.success_mode import classify_success_mode


class SuccessModeTests(unittest.TestCase):
    def test_fail(self) -> None:
        sim = {
            "task_id": "1",
            "reward_info": {"reward": 0.0, "db_check": {"db_match": False}},
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "book_reservation", "arguments": {}}],
                }
            ],
        }
        self.assertEqual(classify_success_mode(sim)["success_mode"], "fail")

    def test_solve_write(self) -> None:
        sim = {
            "task_id": "16",
            "reward_info": {
                "reward": 1.0,
                "db_check": {"db_match": True},
                "action_checks": [
                    {"action": {"name": "update_reservation_flights"}, "action_match": True}
                ],
            },
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"name": "get_reservation_details", "arguments": {}},
                        {"name": "update_reservation_flights", "arguments": {}},
                    ],
                }
            ],
        }
        out = classify_success_mode(sim)
        self.assertEqual(out["success_mode"], "solve_write")
        self.assertIn("update_reservation_flights", out["write_tools"])

    def test_transfer_ok_from_gold(self) -> None:
        sim = {
            "task_id": "13",
            "reward_info": {
                "reward": 1.0,
                "db_check": {"db_match": True},
                "action_checks": [
                    {
                        "action": {"name": "transfer_to_human_agents"},
                        "action_match": True,
                    }
                ],
            },
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "get_reservation_details", "arguments": {}}],
                }
            ],
        }
        self.assertEqual(classify_success_mode(sim)["success_mode"], "transfer_ok")

    def test_communicate_ok(self) -> None:
        sim = {
            "task_id": "2",
            "reward_info": {
                "reward": 1.0,
                "db_check": {"db_match": True},
                "action_checks": [
                    {"action": {"name": "get_user_details"}, "action_match": True}
                ],
            },
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "get_user_details", "arguments": {}}],
                }
            ],
        }
        self.assertEqual(classify_success_mode(sim)["success_mode"], "communicate_ok")

    def test_write_beats_transfer(self) -> None:
        sim = {
            "task_id": "x",
            "reward_info": {
                "reward": 1.0,
                "action_checks": [
                    {"action": {"name": "transfer_to_human_agents"}},
                    {"action": {"name": "book_reservation"}},
                ],
            },
            "messages": [],
        }
        self.assertEqual(classify_success_mode(sim)["success_mode"], "solve_write")


if __name__ == "__main__":
    unittest.main()
