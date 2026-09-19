"""Align SAGE-τ² episodes to AReaL RL dump schema."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sage_tau2.runners.align_areal_trajectories import (
    AREAL_DUMP_KEYS,
    export_areal_trajectories,
    format_completion,
    pro_trajectory_to_areal_turns,
    reconstruct_system_prompt,
    simulation_to_areal_turns,
)
from sage_tau2.runners.dump_pro_trajectories import export_pro_trajectories


def _sim() -> dict:
    return {
        "task_id": "13",
        "trial": 2,
        "termination_reason": "user_stop",
        "duration": 1.5,
        "policy": "# Airline Agent Policy\nYou can book and cancel flights.",
        "reward_info": {
            "reward": 1.0,
            "db_check": {"db_match": True, "db_reward": 1.0},
        },
        "messages": [
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Change my flight."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "get_user_details",
                        "arguments": {"user_id": "u1"},
                    }
                ],
            },
            {
                "role": "tool",
                "id": "call_1",
                "name": "get_user_details",
                "content": '{"user_id":"u1"}',
            },
            {"role": "assistant", "content": "I found your account."},
        ],
    }


class AlignArealTrajectoriesTests(unittest.TestCase):
    def test_dump_schema_and_system_memory(self) -> None:
        turns = simulation_to_areal_turns(_sim())
        self.assertEqual(len(turns), 3)
        self.assertEqual(tuple(turns[0].keys()), AREAL_DUMP_KEYS)
        for key in (
            "format",
            "source_framework",
            "messages",
            "answer",
            "metadata",
            "turn_index",
        ):
            self.assertNotIn(key, turns[0])

        first = turns[0]
        self.assertEqual(first["task_id"], "13")
        self.assertEqual(first["sample_idx"], 0)
        self.assertEqual(first["reward"], 1.0)
        self.assertEqual(first["original_reward"], 1.0)
        self.assertIn("[SYSTEM]", first["prompt"])
        self.assertIn("<instructions>", first["prompt"])
        self.assertIn("<policy>", first["prompt"])
        self.assertIn("# Airline Agent Policy", first["prompt"])
        self.assertIn("Hi! How can I help?", first["completion"])
        self.assertIsNone(first["seqlen"])

        # System prompt stays in later turns; history is a bounded window.
        mid = turns[1]
        self.assertEqual(mid["sample_idx"], 1)
        self.assertTrue(mid["prompt"].startswith("[SYSTEM]"))
        self.assertIn("Change my flight.", mid["prompt"])
        self.assertIn("current observation is:", mid["prompt"].lower())
        self.assertIn("[TOOL_CALLS]", mid["completion"])

        last = turns[2]
        self.assertTrue(last["prompt"].startswith("[SYSTEM]"))
        self.assertIn("get_user_details", last["prompt"])
        self.assertIn("Prior to this step", last["prompt"])
        self.assertEqual(last["completion"], "I found your account.")

    def test_reconstruct_falls_back_to_results_info(self) -> None:
        sim = {"task_id": "1", "messages": [], "policy": ""}
        info = {
            "environment_info": {
                "policy": "# From info\nRetail policy.",
            }
        }
        text = reconstruct_system_prompt(sim, results_info=info)
        self.assertIn("# From info", text)
        self.assertIn("<policy>", text)

    def test_format_completion_empty_without_content_or_tools(self) -> None:
        self.assertEqual(format_completion(None, []), "")

    def test_pro_fallback_can_inject_system(self) -> None:
        traj = {
            "task_id": "2",
            "trial": 0,
            "reward": 0.0,
            "won": False,
            "domain": "airline",
            "steps": [
                {
                    "step": 1,
                    "prompt": "(episode start)",
                    "response": {"content": "Hello", "tool_calls": []},
                },
                {
                    "step": 2,
                    "prompt": "[ASSISTANT] Hello",
                    "response": {
                        "content": None,
                        "tool_calls": [
                            {
                                "name": "get_reservation_details",
                                "arguments": {"reservation_id": "A"},
                            }
                        ],
                    },
                },
            ],
        }
        turns = pro_trajectory_to_areal_turns(
            traj, system_prompt="<instructions>x</instructions>"
        )
        self.assertEqual(len(turns), 2)
        self.assertEqual(tuple(turns[0].keys()), AREAL_DUMP_KEYS)
        self.assertIn("[SYSTEM]", turns[0]["prompt"])
        self.assertIn("<instructions>x</instructions>", turns[0]["prompt"])
        self.assertTrue(turns[1]["prompt"].startswith("[SYSTEM]"))
        self.assertIn("[ASSISTANT] Hello", turns[1]["prompt"])

    def test_export_from_results_and_pro_hook(self) -> None:
        payload = {"simulations": [_sim()]}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            areal = export_areal_trajectories(
                results_payload=payload,
                output_dir=out,
                domain="airline",
            )
            rows = [json.loads(line) for line in areal.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(tuple(rows[0].keys()), AREAL_DUMP_KEYS)
            self.assertIn("[SYSTEM]", rows[0]["prompt"])

            export_pro_trajectories(
                results_payload=payload,
                output_dir=out / "pro",
                domain="airline",
            )
            # PRO dump no longer auto-writes AReaL (use align_areal CLI if needed).
            pro_areal = out / "pro" / "areal_trajectories.jsonl"
            self.assertFalse(pro_areal.exists())
            self.assertTrue((out / "pro" / "trajectories.jsonl").exists())

    def test_export_from_pro_jsonl_fallback(self) -> None:
        traj = {
            "framework": "sage_tau2",
            "task_id": "7",
            "trial": 1,
            "reward": 1.0,
            "won": True,
            "steps": [
                {
                    "step": 1,
                    "prompt": "(episode start)",
                    "response": {"content": "Hi", "tool_calls": []},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "trajectories.jsonl"
            src.write_text(json.dumps(traj) + "\n", encoding="utf-8")
            out = export_areal_trajectories(
                pro_jsonl_path=src,
                output_dir=tmp,
                domain="retail",
            )
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_idx"], 0)
            self.assertEqual(rows[0]["task_id"], "7")
            # PRO-only path has no policy → empty first prompt is ok.
            self.assertEqual(rows[0]["prompt"], "")


if __name__ == "__main__":
    unittest.main()
