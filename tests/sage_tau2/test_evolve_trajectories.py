"""Tests for evolve trajectory dump + solve_write distill gate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sage_tau2.distill import distill_skills_from_trajectories, trajectory_eligible_for_distill
from sage_tau2.runners.dump_evolve_trajectories import (
    EVOLVE_FORMAT,
    export_evolve_trajectories,
    load_evolve_trajectories,
    trajectory_to_evolve_row,
)
from sage_tau2.schemas import Tau2Trajectory
from sage_tau2.trajectory import simulation_to_trajectory


def _write_sim(*, task_id: str = "16") -> dict:
    return {
        "id": "ev-1",
        "task_id": task_id,
        "trial": 0,
        "termination_reason": "user_stop",
        "policy": "# airline policy\nAuthenticate before writes.",
        "reward_info": {
            "reward": 1.0,
            "db_check": {"db_match": True, "db_reward": 1.0},
            "action_checks": [
                {"action": {"name": "update_reservation_flights"}, "action_match": True}
            ],
        },
        "messages": [
            {"role": "user", "content": "Please change my flight date."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "get_user_details",
                        "arguments": {"user_id": "u1"},
                    }
                ],
            },
            {
                "role": "tool",
                "id": "c1",
                "name": "get_user_details",
                "content": "{}",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "update_reservation_flights",
                        "arguments": {
                            "reservation_id": "R1",
                            "cabin": "economy",
                            "flights": [],
                            "payment_id": "p1",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "id": "c2",
                "name": "update_reservation_flights",
                "content": "{}",
            },
        ],
    }


def _transfer_sim() -> dict:
    return {
        "id": "ev-t",
        "task_id": "13",
        "trial": 0,
        "reward_info": {
            "reward": 1.0,
            "db_check": {"db_match": True},
            "action_checks": [
                {"action": {"name": "transfer_to_human_agents"}, "action_match": True}
            ],
        },
        "messages": [
            {"role": "user", "content": "Change destination please."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "name": "transfer_to_human_agents",
                        "arguments": {"summary": "cannot change destination"},
                    }
                ],
            },
            {
                "role": "tool",
                "id": "t1",
                "name": "transfer_to_human_agents",
                "content": "Transfer successful",
            },
        ],
    }


class EvolveDumpTests(unittest.TestCase):
    def test_simulation_metadata_has_success_mode(self) -> None:
        traj = simulation_to_trajectory(_write_sim(), domain="airline")
        self.assertEqual(traj.metadata.get("success_mode"), "solve_write")
        row = trajectory_to_evolve_row(traj)
        self.assertEqual(row["format"], EVOLVE_FORMAT)
        self.assertTrue(row["eligible_for_distill"])
        self.assertTrue(row["write_spine"])
        self.assertIn("update_reservation_flights", " ".join(row["write_spine"]))
        self.assertTrue(row["dialogue"])
        self.assertNotIn("system_prompt", row)
        self.assertNotIn("raw_messages", row)
        self.assertTrue(row["tool_steps"])

    def test_transfer_not_eligible_by_default(self) -> None:
        traj = simulation_to_trajectory(_transfer_sim(), domain="airline")
        self.assertEqual(traj.metadata.get("success_mode"), "transfer_ok")
        self.assertFalse(trajectory_eligible_for_distill(traj))
        self.assertFalse(
            trajectory_eligible_for_distill(traj, require_solve_write=True)
        )
        row = trajectory_to_evolve_row(traj)
        self.assertFalse(row["eligible_for_distill"])

    def test_distill_skips_transfer_mode(self) -> None:
        write = simulation_to_trajectory(_write_sim(), domain="airline")
        transfer = simulation_to_trajectory(_transfer_sim(), domain="airline")
        skills = distill_skills_from_trajectories([write, transfer], min_support=1)
        self.assertEqual(len(skills), 1)
        self.assertIn("update_reservation_flights", skills[0].capability_key)

    def test_export_roundtrip_distill(self) -> None:
        payload = {"simulations": [_write_sim(), _transfer_sim()], "tasks": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = export_evolve_trajectories(
                results_payload=payload,
                output_dir=tmp,
                domain="airline",
            )
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "evolve_trajectories.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 2)
            write_row = next(r for r in rows if r["task_id"] == "16")
            # Concrete short args kept in tool_steps (not fully wiped to ?).
            user_step = next(
                s for s in write_row["tool_steps"] if s["name"] == "get_user_details"
            )
            self.assertEqual(user_step["arguments"].get("user_id"), "u1")
            update = next(
                s
                for s in write_row["tool_steps"]
                if s["name"] == "update_reservation_flights"
            )
            self.assertEqual(update["arguments"].get("cabin"), "economy")
            # Protocol still abstracts IDs but keeps cabin literal.
            proto = " ".join(write_row["tool_protocol"])
            self.assertIn("cabin=economy", proto)
            self.assertIn("reservation_id=?", proto)
            loaded = load_evolve_trajectories(path)
            skills = distill_skills_from_trajectories(
                loaded, min_support=1, require_solve_write=True
            )
            self.assertEqual(len(skills), 1)
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["format"], EVOLVE_FORMAT)
            self.assertTrue(rows[0].get("dialogue"))
            self.assertNotIn("system_prompt", rows[0])
            for msg in rows[0]["dialogue"]:
                self.assertNotIn("usage", msg)
                self.assertNotIn("raw_data", msg)
            readable = Path(tmp) / "evolve_readable" / "task_16.txt"
            self.assertTrue(readable.exists())
            text = readable.read_text(encoding="utf-8")
            self.assertIn("[USER]", text)
            self.assertIn("[MODEL]", text)
            self.assertIn("change my flight", text.lower())

            loaded = load_evolve_trajectories(path)
            self.assertEqual(len(loaded), 2)
            eligible = load_evolve_trajectories(path, eligible_only=True)
            self.assertEqual(len(eligible), 1)
            skills = distill_skills_from_trajectories(
                eligible, min_support=1, require_solve_write=True
            )
            self.assertEqual(len(skills), 1)

    def test_manual_traj_without_mode_falls_back_to_write_spine(self) -> None:
        traj = Tau2Trajectory(
            task_id="x",
            trial=0,
            domain="airline",
            reward=1.0,
            db_reward=1.0,
            communicate_reward=None,
            db_match=True,
            termination_reason="user_stop",
            tool_protocol=[
                "get_user_details(user_id=?)",
                "book_reservation(user_id=?)",
            ],
            tool_steps=[],
            assistant_texts=[],
            user_texts=["book"],
            metadata={},
        )
        self.assertTrue(
            trajectory_eligible_for_distill(traj, require_solve_write=True)
        )


if __name__ == "__main__":
    unittest.main()
