"""Tests for write-gold fail retry helpers."""

from __future__ import annotations

import unittest

from sage_tau2.success_mode import (
    is_write_gold_episode,
    merge_simulations_prefer_best,
    write_gold_fail_task_ids,
)


def _sim(
    task_id: str,
    *,
    reward: float,
    gold: list[str] | None = None,
    tool: str | None = None,
    db_match: bool | None = False,
) -> dict:
    messages = []
    if tool:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"name": tool, "arguments": {}}],
            }
        ]
    return {
        "task_id": task_id,
        "reward_info": {
            "reward": reward,
            "db_check": {"db_match": db_match},
            "action_checks": [
                {"action": {"name": name}, "action_match": False} for name in (gold or [])
            ],
        },
        "messages": messages,
    }


class WriteRetryTests(unittest.TestCase):
    def test_write_gold_fail_ids(self) -> None:
        payload = {
            "simulations": [
                _sim("1", reward=0.0, gold=["update_reservation_flights"]),
                _sim("2", reward=1.0, gold=["update_reservation_flights"], db_match=True),
                _sim("3", reward=0.0, gold=["transfer_to_human_agents"]),
                _sim("4", reward=0.0, gold=["cancel_reservation"]),
            ]
        }
        self.assertEqual(write_gold_fail_task_ids(payload), ["1", "4"])
        self.assertTrue(is_write_gold_episode(payload["simulations"][0]))
        self.assertFalse(is_write_gold_episode(payload["simulations"][2]))

    def test_merge_prefers_solve_write(self) -> None:
        base = {
            "simulations": [
                _sim("1", reward=0.0, gold=["update_reservation_flights"]),
                _sim("2", reward=0.0, gold=["cancel_reservation"]),
            ]
        }
        retry = {
            "simulations": [
                _sim(
                    "1",
                    reward=1.0,
                    gold=["update_reservation_flights"],
                    tool="update_reservation_flights",
                    db_match=True,
                ),
                _sim("2", reward=0.0, gold=["cancel_reservation"]),
            ]
        }
        merged = merge_simulations_prefer_best(base, retry)
        by_id = {s["task_id"]: s for s in merged["simulations"]}
        self.assertEqual(by_id["1"]["reward_info"]["reward"], 1.0)
        self.assertEqual(by_id["2"]["reward_info"]["reward"], 0.0)
        self.assertIn("1", merged["_retry_replaced_task_ids"])


if __name__ == "__main__":
    unittest.main()
