"""Cross-segment distill window (sage_mas distill_prior_segments style)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sage_tau2.pipeline import (
    _merge_trajectories_for_distill,
    load_prior_distill_trajectories,
)
from sage_tau2.schemas import Tau2Trajectory
from sage_tau2.trajectory_adapter import adapt_pro_many, export_atomic_trajectories
from tests.sage_tau2.test_trajectory_adapter import _pro_record  # noqa: PLC2701


class DistillPriorWindowTests(unittest.TestCase):
    def test_loads_last_two_domain_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, tid in [(1, "a"), (2, "b")]:
                d = root / f"segment_{i:03d}" / "domain_telecom" / "pro_trajectories"
                d.mkdir(parents=True)
                export_atomic_trajectories(
                    adapt_pro_many([_pro_record(task_id=tid)]),
                    d / "atomic_trajectories.json",
                )
            cur = root / "segment_003" / "domain_telecom"
            cur.mkdir(parents=True)
            prior = load_prior_distill_trajectories(
                segment_dir=cur, domain="telecom", prior_segments=2
            )
            self.assertEqual([t.task_id for t in prior], ["a", "b"])

    def test_merge_dedupes_task_ids(self) -> None:
        a = Tau2Trajectory(
            task_id="a",
            trial=0,
            domain="telecom",
            reward=1.0,
            db_reward=None,
            communicate_reward=None,
            db_match=True,
            termination_reason=None,
            tool_protocol=["x"],
            tool_steps=[],
            assistant_texts=[],
            user_texts=[],
        )
        prior = [a]
        current = [a]
        merged, meta = _merge_trajectories_for_distill(current, prior)
        self.assertEqual(meta["n_merged"], 1)
        self.assertEqual(len(merged), 1)


if __name__ == "__main__":
    unittest.main()
