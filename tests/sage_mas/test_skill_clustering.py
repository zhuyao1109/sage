"""Tests for the persistent skill-cluster archive (leader algorithm)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from sage_mas.schemas import AtomicOp, ExplorationDistribution, Skill, SkillStatus
from sage_mas.skill_clustering import (
    BIRTH_CAPABILITY,
    BIRTH_VARIANT,
    DEFAULT_NOVELTY_THRESHOLD,
    EVENT_ASSIGNED,
    SkillClusterArchive,
    calibrate_novelty_threshold,
    novelty_distance,
)


def _skill(
    name: str,
    capability: str,
    *,
    vector: list[float] | None = None,
    status: SkillStatus = SkillStatus.VERIFIED,
    support: int = 5,
    utility: float | None = None,
    protocol: list[str] | None = None,
    search_prior: dict | None = None,
    outcome_histogram: dict[str, int] | None = None,
) -> Skill:
    skill = Skill(
        skill_name=name,
        description=f"skill for {name}",
        precondition="at start location",
        action_protocol=protocol or ["go to <source>", "take <object> from <source>"],
        applicable_atomic_ops=[AtomicOp.ACT],
        status=status,
        support_count=support,
        marginal_utility=utility,
        capability_key=capability,
        applicable_task_families=[capability],
    )
    if outcome_histogram is not None:
        total = sum(outcome_histogram.values())
        skill.exploration_distribution = ExplorationDistribution(
            task_family_histogram={capability: total},
            outcome_histogram=dict(outcome_histogram),
            total_trajectories=total,
            success_rate=outcome_histogram.get("success", 0) / max(total, 1),
        )
    if search_prior is not None:
        skill.metadata["search_prior"] = search_prior
    if vector is not None:
        skill.embedding = list(vector)
        skill.metadata["embedding_version"] = "skill-behavior-hash-v2"
    return skill


def _normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


class ArchiveUpdateTests(unittest.TestCase):
    def test_first_skill_births_capability_cluster(self) -> None:
        archive = SkillClusterArchive()
        skill = _skill("heat mug", "transform.heat", vector=[1.0, 0.0, 0.0])
        report = archive.update([skill], segment=0)

        self.assertEqual(len(archive.clusters), 1)
        cluster = archive.clusters[0]
        self.assertEqual(cluster.birth_event, BIRTH_CAPABILITY)
        self.assertEqual(cluster.capability_key, "transform.heat")
        self.assertEqual(cluster.leader_skill_id, skill.skill_id)
        self.assertEqual(cluster.member_skill_ids, [skill.skill_id])
        self.assertEqual(cluster.born_segment, 0)
        self.assertEqual(report.events[0].event, BIRTH_CAPABILITY)

    def test_near_skill_assigns_to_existing_cluster(self) -> None:
        archive = SkillClusterArchive()
        leader = _skill("heat mug", "transform.heat", vector=[1.0, 0.0, 0.0])
        variant = _skill(
            "heat potato",
            "transform.heat",
            vector=_normalized([0.99, 0.14, 0.0]),
        )
        archive.update([leader], segment=0)
        report = archive.update([variant], segment=1)

        self.assertEqual(len(archive.clusters), 1)
        event = report.events[0]
        self.assertEqual(event.event, EVENT_ASSIGNED)
        self.assertLessEqual(event.novelty, DEFAULT_NOVELTY_THRESHOLD)
        self.assertIn(variant.skill_id, archive.clusters[0].member_skill_ids)

    def test_distant_same_capability_skill_births_variant_cluster(self) -> None:
        archive = SkillClusterArchive()
        leader = _skill("heat mug", "transform.heat", vector=[1.0, 0.0, 0.0])
        distant = _skill(
            "heat with pre-open",
            "transform.heat",
            vector=[0.0, 1.0, 0.0],
        )
        archive.update([leader], segment=0)
        report = archive.update([distant], segment=1)

        self.assertEqual(len(archive.clusters), 2)
        event = report.events[0]
        self.assertEqual(event.event, BIRTH_VARIANT)
        self.assertEqual(
            archive.cluster_of(distant.skill_id).birth_event, BIRTH_VARIANT
        )

    def test_new_capability_always_births_even_if_embedding_close(self) -> None:
        """capability_key is a hard pre-partition: embedding proximity across
        capabilities must never merge clusters."""
        archive = SkillClusterArchive()
        heat = _skill("heat mug", "transform.heat", vector=[1.0, 0.0, 0.0])
        clean = _skill("clean mug", "transform.clean", vector=[1.0, 0.0, 0.0])
        report = archive.update([heat, clean], segment=0)

        self.assertEqual(len(archive.clusters), 2)
        births = {event.event for event in report.events}
        self.assertEqual(births, {BIRTH_CAPABILITY, BIRTH_CAPABILITY})
        self.assertNotEqual(
            archive.cluster_of(heat.skill_id).cluster_id,
            archive.cluster_of(clean.skill_id).cluster_id,
        )

    def test_rejected_and_retired_skills_are_not_clustered(self) -> None:
        archive = SkillClusterArchive()
        live = _skill("heat mug", "transform.heat", vector=[1.0, 0.0, 0.0])
        rejected = _skill(
            "junk", "transform.heat",
            vector=[0.0, 1.0, 0.0], status=SkillStatus.REJECTED,
        )
        retired = _skill(
            "old", "transform.cool",
            vector=[0.0, 0.0, 1.0], status=SkillStatus.RETIRED,
        )
        report = archive.update([live, rejected, retired], segment=0)

        self.assertEqual(len(archive.clusters), 1)
        self.assertEqual(len(report.events), 1)
        self.assertIsNone(archive.cluster_of(rejected.skill_id))
        self.assertIsNone(archive.cluster_of(retired.skill_id))

    def test_input_order_does_not_change_archive(self) -> None:
        def build() -> SkillClusterArchive:
            archive = SkillClusterArchive()
            leader = _skill(
                "canonical", "transform.heat",
                vector=[1.0, 0.0, 0.0], support=10,
            )
            follower = _skill(
                "variant", "transform.heat",
                vector=_normalized([0.99, 0.14, 0.0]), support=2,
            )
            far = _skill(
                "other cap", "track.place",
                vector=[0.0, 1.0, 0.0], support=7,
            )
            return archive, [leader, follower, far]

        archive_a, skills_a = build()
        archive_a.update(skills_a, segment=0)
        archive_b, skills_b = build()
        archive_b.update(list(reversed(skills_b)), segment=0)

        def membership(archive: SkillClusterArchive) -> dict[str, str]:
            return {
                skill.skill_id: archive.cluster_of(skill.skill_id).cluster_id
                for skill in (skills_a if archive is archive_a else skills_b)
            }

        # Rebuilt skills have fresh ids; compare structural invariants instead.
        self.assertEqual(len(archive_a.clusters), len(archive_b.clusters))
        self.assertEqual(
            sorted(len(c.member_skill_ids) for c in archive_a.clusters),
            sorted(len(c.member_skill_ids) for c in archive_b.clusters),
        )
        self.assertEqual(
            sorted(c.capability_key for c in archive_a.clusters),
            sorted(c.capability_key for c in archive_b.clusters),
        )
        self.assertEqual(
            sorted(c.birth_event for c in archive_a.clusters),
            sorted(c.birth_event for c in archive_b.clusters),
        )

    def test_frozen_leader_prevents_chain_merging(self) -> None:
        """A skill close to a *member* but far from the frozen leader must
        birth its own cluster; leader clustering is not single-linkage."""
        archive = SkillClusterArchive(novelty_threshold=0.30)
        leader = _skill("A", "transform.heat", vector=[1.0, 0.0, 0.0], support=10)
        near_leader = _skill(
            "B", "transform.heat", vector=_normalized([0.95, 0.31, 0.0]),
            support=9,
        )
        near_member_far_leader = _skill(
            "C", "transform.heat", vector=_normalized([0.30, 0.95, 0.0]),
            support=8,
        )
        archive.update([leader, near_leader, near_member_far_leader], segment=0)

        cluster_of_c = archive.cluster_of(near_member_far_leader.skill_id)
        cluster_of_leader = archive.cluster_of(leader.skill_id)
        self.assertIsNotNone(cluster_of_c)
        self.assertNotEqual(cluster_of_c.cluster_id, cluster_of_leader.cluster_id)
        self.assertEqual(cluster_of_c.birth_event, BIRTH_VARIANT)
        # sanity: C really was close to member B
        self.assertLess(
            novelty_distance(
                _normalized([0.30, 0.95, 0.0]), _normalized([0.95, 0.31, 0.0])
            ),
            0.30,
        )

    def test_cluster_identity_persists_across_segments(self) -> None:
        archive = SkillClusterArchive()
        first = _skill("heat mug", "transform.heat", vector=[1.0, 0.0, 0.0])
        report0 = archive.update([first], segment=0)
        self.assertEqual(len(report0.events), 1)

        # Same bank next segment: no re-assignment events, only a snapshot.
        report1 = archive.update([first], segment=1)
        self.assertEqual(report1.events, [])
        cluster = archive.clusters[0]
        self.assertEqual(len(cluster.segment_stats), 2)
        self.assertEqual(cluster.stats_for_segment(0)["member_count"], 1)
        self.assertEqual(cluster.stats_for_segment(1)["member_count"], 1)

    def test_segment_stats_are_idempotent_per_segment(self) -> None:
        archive = SkillClusterArchive()
        skill = _skill("heat mug", "transform.heat", vector=[1.0, 0.0, 0.0])
        archive.update([skill], segment=0)
        archive.update([skill], segment=0)
        self.assertEqual(len(archive.clusters[0].segment_stats), 1)

    def test_segment_stats_aggregate_support_utility_and_priors(self) -> None:
        archive = SkillClusterArchive()
        leader = _skill(
            "heat mug", "transform.heat",
            vector=[1.0, 0.0, 0.0], support=6, utility=0.8,
            search_prior={
                "sources": [
                    {"source": "countertop", "count": 4},
                    {"source": "shelf", "count": 1},
                ],
                "episodes": 5,
                "acquisitions": 5,
            },
            outcome_histogram={"success": 5, "failure": 1},
        )
        variant = _skill(
            "heat potato", "transform.heat",
            vector=_normalized([0.99, 0.14, 0.0]), support=4, utility=0.4,
            search_prior={
                "sources": [{"source": "countertop", "count": 2}],
                "episodes": 2,
                "acquisitions": 2,
            },
            outcome_histogram={"success": 4},
        )
        archive.update([leader, variant], segment=0)

        stats = archive.clusters[0].stats_for_segment(0)
        self.assertEqual(stats["member_count"], 2)
        self.assertEqual(stats["support_total"], 10)
        self.assertAlmostEqual(stats["utility_mean"], 0.6)
        self.assertEqual(stats["outcome_histogram"], {"success": 9, "failure": 1})
        self.assertEqual(
            stats["search_prior_histogram"], {"countertop": 6, "shelf": 1}
        )
        self.assertEqual(stats["status_histogram"], {"verified": 2})


class PersistenceTests(unittest.TestCase):
    def test_serialization_round_trip(self) -> None:
        archive = SkillClusterArchive(novelty_threshold=0.25)
        archive.update(
            [
                _skill("heat mug", "transform.heat", vector=[1.0, 0.0]),
                _skill("place mug", "track.place", vector=[0.0, 1.0]),
            ],
            segment=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill_clusters.json"
            archive.save(path)
            loaded = SkillClusterArchive.load(path)

        self.assertEqual(loaded.novelty_threshold, 0.25)
        self.assertEqual(loaded.last_segment, 3)
        self.assertEqual(len(loaded.clusters), 2)
        original = {c.cluster_id: c for c in archive.clusters}
        for cluster in loaded.clusters:
            twin = original[cluster.cluster_id]
            self.assertEqual(cluster.leader_skill_id, twin.leader_skill_id)
            self.assertEqual(cluster.leader_embedding, twin.leader_embedding)
            self.assertEqual(cluster.protocol_template, twin.protocol_template)
            self.assertEqual(cluster.segment_stats, twin.segment_stats)

    def test_load_or_create_respects_existing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill_clusters.json"
            archive = SkillClusterArchive(novelty_threshold=0.2)
            archive.update(
                [_skill("heat mug", "transform.heat", vector=[1.0, 0.0])],
                segment=0,
            )
            archive.save(path)

            resumed = SkillClusterArchive.load_or_create(path, novelty_threshold=0.9)
            self.assertEqual(resumed.novelty_threshold, 0.2)
            self.assertEqual(len(resumed.clusters), 1)

            fresh = SkillClusterArchive.load_or_create(
                Path(tmp) / "missing.json", novelty_threshold=0.9
            )
            self.assertEqual(fresh.novelty_threshold, 0.9)
            self.assertEqual(fresh.clusters, [])


class CalibrationTests(unittest.TestCase):
    def test_separable_ranges_yield_midpoint_tau(self) -> None:
        skills = [
            _skill("heat a", "transform.heat", vector=[1.0, 0.0, 0.0]),
            _skill(
                "heat b", "transform.heat", vector=_normalized([0.99, 0.14, 0.0])
            ),
            _skill("place a", "track.place", vector=[0.0, 1.0, 0.0]),
            _skill(
                "place b", "track.place", vector=_normalized([0.14, 0.99, 0.0])
            ),
        ]
        report = calibrate_novelty_threshold(skills)

        self.assertTrue(report["separable"])
        self.assertEqual(report["within_capability"]["n"], 2)
        self.assertEqual(report["cross_capability"]["n"], 4)
        within_p90 = report["within_capability"]["p90"]
        cross_min = report["cross_capability"]["min"]
        self.assertGreater(cross_min, within_p90)
        self.assertAlmostEqual(
            report["suggested_novelty_threshold"],
            round((within_p90 + cross_min) / 2.0, 4),
        )

    def test_empty_bank_falls_back_to_default(self) -> None:
        report = calibrate_novelty_threshold([])
        self.assertFalse(report["separable"])
        self.assertEqual(
            report["suggested_novelty_threshold"], DEFAULT_NOVELTY_THRESHOLD
        )


class PipelineWiringTests(unittest.TestCase):
    def test_pipeline_writes_cluster_archive_and_events(self) -> None:
        from sage_mas.pipeline import SageEvolutionPipeline
        from sage_mas.skill_bank import SkillBank

        pipeline = SageEvolutionPipeline(
            {"sage": {"distillation": {"mode": "heuristic"}}}
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            bank_path = Path(tmp) / "skill_bank.json"
            bank = SkillBank(bank_path, redundancy_threshold=0.9)
            bank.add(_skill("heat mug", "transform.heat", vector=[1.0, 0.0]))
            bank.add(_skill("place mug", "track.place", vector=[0.0, 1.0]))

            pipeline._update_cluster_archive(
                bank=bank, bank_path=bank_path, run_dir=run_dir
            )

            archive_path = Path(tmp) / "skill_clusters.json"
            self.assertTrue(archive_path.exists())
            archive = SkillClusterArchive.load(archive_path)
            self.assertEqual(len(archive.clusters), 2)
            events_path = run_dir / "cluster_events.json"
            self.assertTrue(events_path.exists())

    def test_clustering_disabled_via_config(self) -> None:
        from sage_mas.pipeline import SageEvolutionPipeline
        from sage_mas.skill_bank import SkillBank

        pipeline = SageEvolutionPipeline(
            {
                "sage": {
                    "distillation": {"mode": "heuristic"},
                    "clustering": {"enabled": False},
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            bank_path = Path(tmp) / "skill_bank.json"
            bank = SkillBank(bank_path, redundancy_threshold=0.9)
            bank.add(_skill("heat mug", "transform.heat", vector=[1.0, 0.0]))

            pipeline._update_cluster_archive(
                bank=bank, bank_path=bank_path, run_dir=run_dir
            )
            self.assertFalse((Path(tmp) / "skill_clusters.json").exists())


if __name__ == "__main__":
    unittest.main()
