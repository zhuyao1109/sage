"""PRO trajectories → atomic_trajectories.json (ALFWorld-aligned distill input)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sage_tau2.pipeline import _load_segment_trajectories, load_prior_distill_trajectories
from sage_tau2.trajectory_adapter import (
    adapt_pro_many,
    export_atomic_trajectories,
    load_atomic_trajectories,
    pro_record_to_trajectory,
)


def _pro_record(*, task_id: str = "t1", reward: float = 1.0) -> dict:
    return {
        "format": "sage_tau2_pro_v5",
        "domain": "telecom",
        "task_id": task_id,
        "trial": 0,
        "reward": reward,
        "db_match": reward >= 1.0,
        "distill_meta": {
            "success_mode": "solve_write",
            "action_checks": [
                {"action": {"name": "enable_roaming"}, "action_match": True}
            ],
        },
        "steps": [
            {
                "step": 1,
                "observation_before": "[USER] My data is slow in France.",
                "action": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "get_customer_by_phone",
                        "arguments": {"phone_number": "555-123-2002"},
                    }
                ],
                "observation": (
                    "[tool_result:get_customer_by_phone] "
                    '{"customer_id": "C1001", "line_ids": ["L1001"]}'
                ),
            },
            {
                "step": 2,
                "observation_before": "[tool_result:get_customer_by_phone] {...}",
                "action": None,
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "enable_roaming",
                        "arguments": {"customer_id": "C1001", "line_id": "L1001"},
                    }
                ],
                "observation": "[tool_result:enable_roaming] ok",
            },
        ],
    }


class TrajectoryAdapterTests(unittest.TestCase):
    def test_pro_record_to_trajectory_builds_protocol(self) -> None:
        traj = pro_record_to_trajectory(_pro_record())
        self.assertEqual(traj.task_id, "t1")
        self.assertIn("get_customer_by_phone(phone_number=?)", traj.tool_protocol)
        self.assertIn("enable_roaming(customer_id=?, line_id=?)", traj.tool_protocol)
        self.assertEqual(traj.metadata.get("success_mode"), "solve_write")
        self.assertIn("France", " ".join(traj.user_texts))

    def test_atomic_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "atomic_trajectories.json"
            atomic = adapt_pro_many([_pro_record()])
            export_atomic_trajectories(atomic, path)
            loaded = load_atomic_trajectories(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].tool_protocol, atomic[0].tool_protocol)

    def test_pipeline_prefers_atomic_over_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seg = Path(tmp) / "segment_001"
            pro = seg / "pro_trajectories"
            pro.mkdir(parents=True)
            atomic = adapt_pro_many([_pro_record(task_id="from_pro")])
            export_atomic_trajectories(atomic, pro / "atomic_trajectories.json")
            trajectories = _load_segment_trajectories(
                results_payload={"simulations": []},
                domain="telecom",
                segment_dir=seg,
            )
            self.assertEqual(trajectories[0].task_id, "from_pro")

    def test_prior_window_reads_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, tid in [(1, "a"), (2, "b")]:
                pro = root / f"segment_{i:03d}" / "pro_trajectories"
                pro.mkdir(parents=True)
                export_atomic_trajectories(
                    adapt_pro_many([_pro_record(task_id=tid)]),
                    pro / "atomic_trajectories.json",
                )
            cur = root / "segment_003"
            cur.mkdir()
            prior = load_prior_distill_trajectories(
                segment_dir=cur, domain="telecom", prior_segments=2
            )
            self.assertEqual([t.task_id for t in prior], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
