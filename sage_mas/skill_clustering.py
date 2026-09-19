"""Persistent skill-cluster archive: leader algorithm with frozen leaders.

Role in SAGE: this module is the *measurement instrument* of the evolution
loop, not a decision maker. It maintains a persistent archive of skill
clusters so that downstream stages (drift detection, specialist promotion)
can ask time-series questions like "has cluster C3 moved over the last k
segments?" or "when did this capability first appear?".

Design rules (fixed by the SAGE methodology):

- **Two-tier partitioning.** ``capability_key`` (action-grammar level, from
  distillation) is a hard pre-partition: a skill only ever joins clusters of
  its own capability. Within a capability, clustering is by continuous
  behavioral novelty. A new ``capability_key`` therefore always produces a
  capability-level birth event.
- **Leader algorithm with frozen leaders.** Each cluster is represented by
  the frozen embedding of its founding member. The reference point never
  moves, so "cluster drift" is well-defined: it is the changing distance
  distribution of *new members* against a *fixed* reference. A running
  centroid would conflate "the cluster grew" with "the cluster moved".
- **Everything admitted to the bank is clustered.** Only ``rejected`` /
  ``retired`` skills are excluded (mirroring ``distribution_stats``). The
  bank's admission gates are the quality filter; the archive must stay
  low-threshold because drift signals are born at the periphery (a new
  capability's first skill has minimal support). Ranking/selection belongs
  to the read side (injection, dispatch, promotion), never to the archive.
- **Observation only.** This module produces records and events; it never
  mutates skills (besides caching their behavior embedding) and never
  triggers organization edits.

The assignment rule is deliberately simple (nearest frozen leader, novelty
threshold) and swappable: the persistent archive format is the stable
interface, so a future assignment rule upgrade does not invalidate recorded
history.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sage_mas.distribution_stats import cosine_similarity
from sage_mas.schemas import Skill, SkillStatus
from sage_mas.trajectory.fragments import embed_skill

logger = logging.getLogger(__name__)

EMBEDDING_VERSION = "skill-behavior-hash-v2"
EXCLUDED_STATUSES = {SkillStatus.REJECTED, SkillStatus.RETIRED}
DEFAULT_NOVELTY_THRESHOLD = 0.30

BIRTH_CAPABILITY = "capability_birth"
BIRTH_VARIANT = "variant_birth"
EVENT_ASSIGNED = "assigned"


def novelty_distance(left: list[float], right: list[float]) -> float:
    """Behavioral novelty in [0, 1]: (1 - cosine) / 2, matching the metric
    used by ``embedding_novelty_for_skill_cluster`` in distribution_stats."""
    return (1.0 - cosine_similarity(left, right)) / 2.0


def _skill_vector(skill: Skill) -> list[float]:
    """Return the behavior embedding, computing and caching it if stale."""
    if (
        skill.embedding
        and skill.metadata.get("embedding_version") == EMBEDDING_VERSION
    ):
        return skill.embedding
    vector = embed_skill(skill)
    skill.embedding = list(vector)
    return vector


def _cluster_id(capability_key: str, leader_skill_id: str) -> str:
    digest = hashlib.sha1(f"{capability_key}|{leader_skill_id}".encode("utf-8"))
    return f"cluster-{digest.hexdigest()[:12]}"


@dataclass
class SkillCluster:
    """One persistent cluster: frozen leader + members + segment time series."""

    cluster_id: str
    capability_key: str
    leader_skill_id: str
    leader_embedding: list[float]
    protocol_template: list[str]
    born_segment: int
    birth_event: str
    member_skill_ids: list[str] = field(default_factory=list)
    segment_stats: list[dict[str, Any]] = field(default_factory=list)

    def stats_for_segment(self, segment: int) -> dict[str, Any] | None:
        for entry in self.segment_stats:
            if entry.get("segment") == segment:
                return entry
        return None


@dataclass
class ClusterEvent:
    segment: int
    event: str
    skill_id: str
    skill_name: str
    capability_key: str
    cluster_id: str
    novelty: float | None = None


@dataclass
class ClusterUpdateReport:
    segment: int
    events: list[ClusterEvent] = field(default_factory=list)

    @property
    def births(self) -> list[ClusterEvent]:
        return [e for e in self.events if e.event in {BIRTH_CAPABILITY, BIRTH_VARIANT}]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment": self.segment,
            "events": [asdict(event) for event in self.events],
        }


def _segment_stats_snapshot(
    cluster: SkillCluster,
    members: list[Skill],
    segment: int,
) -> dict[str, Any]:
    """Support-weighted observation snapshot of one cluster for one segment.

    Snapshots are idempotent per segment: re-recording the same segment
    replaces the entry rather than appending a duplicate.
    """
    support_total = 0
    evidence_total = 0
    utilities: list[float] = []
    outcome_histogram: dict[str, int] = {}
    search_prior_histogram: dict[str, int] = {}
    status_histogram: dict[str, int] = {}
    for skill in members:
        weight = max(int(skill.support_count), len(skill.evidence_ids), 1)
        support_total += int(skill.support_count)
        evidence_total += len(skill.evidence_ids)
        if skill.marginal_utility is not None:
            utilities.append(float(skill.marginal_utility))
        status = skill.status.value if isinstance(skill.status, SkillStatus) else str(skill.status)
        status_histogram[status] = status_histogram.get(status, 0) + 1
        distribution = skill.exploration_distribution
        if distribution is not None:
            for outcome, count in (distribution.outcome_histogram or {}).items():
                outcome_histogram[outcome] = outcome_histogram.get(outcome, 0) + int(count)
        prior = (skill.metadata or {}).get("search_prior") or {}
        for entry in prior.get("sources") or []:
            source = str(entry.get("source", "")).strip()
            if not source:
                continue
            search_prior_histogram[source] = (
                search_prior_histogram.get(source, 0) + int(entry.get("count", 0))
            )
    return {
        "segment": segment,
        "member_count": len(members),
        "support_total": support_total,
        "evidence_total": evidence_total,
        "utility_mean": (
            sum(utilities) / len(utilities) if utilities else None
        ),
        "outcome_histogram": outcome_histogram,
        "status_histogram": status_histogram,
        "search_prior_histogram": dict(
            sorted(
                search_prior_histogram.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }


class SkillClusterArchive:
    """Persistent archive of skill clusters across evolution segments.

    The archive is the stable interface of the clustering layer: drift
    detection and organization logic read from it, and it survives future
    changes to the assignment rule.
    """

    def __init__(self, *, novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD) -> None:
        self.novelty_threshold = float(novelty_threshold)
        self.clusters: list[SkillCluster] = []
        self.last_segment: int = -1

    # ------------------------------------------------------------------
    # lookup helpers
    # ------------------------------------------------------------------
    def cluster_of(self, skill_id: str) -> SkillCluster | None:
        for cluster in self.clusters:
            if skill_id in cluster.member_skill_ids:
                return cluster
        return None

    def capability_cluster_ids(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for cluster in self.clusters:
            mapping.setdefault(cluster.capability_key, []).append(cluster.cluster_id)
        return mapping

    # ------------------------------------------------------------------
    # core update: one call = one evolution segment
    # ------------------------------------------------------------------
    def update(
        self,
        skills: list[Skill],
        *,
        segment: int | None = None,
    ) -> ClusterUpdateReport:
        """Assign new skills to clusters and refresh per-segment snapshots.

        Assignment order is deterministic (capability, support desc, id) so
        the same bank state always reproduces the same archive. Skills keep
        their cluster identity across segments; only never-before-seen
        skills go through leader assignment.
        """
        if segment is None:
            segment = self.last_segment + 1
        eligible = [
            skill for skill in skills if skill.status not in EXCLUDED_STATUSES
        ]
        report = ClusterUpdateReport(segment=segment)

        pending = [s for s in eligible if self.cluster_of(s.skill_id) is None]
        pending.sort(
            key=lambda s: (
                s.capability_key or "",
                -max(int(s.support_count), len(s.evidence_ids), 1),
                s.skill_id,
            )
        )
        for skill in pending:
            self._assign(skill, segment=segment, report=report)

        members_by_cluster: dict[str, list[Skill]] = {
            cluster.cluster_id: [] for cluster in self.clusters
        }
        for skill in eligible:
            cluster = self.cluster_of(skill.skill_id)
            if cluster is not None:
                members_by_cluster[cluster.cluster_id].append(skill)
        for cluster in self.clusters:
            members = members_by_cluster.get(cluster.cluster_id, [])
            snapshot = _segment_stats_snapshot(cluster, members, segment)
            existing = cluster.stats_for_segment(segment)
            if existing is None:
                cluster.segment_stats.append(snapshot)
            else:
                cluster.segment_stats[cluster.segment_stats.index(existing)] = snapshot

        self.last_segment = max(self.last_segment, segment)
        return report

    def _assign(
        self,
        skill: Skill,
        *,
        segment: int,
        report: ClusterUpdateReport,
    ) -> SkillCluster:
        vector = _skill_vector(skill)
        capability = skill.capability_key or "unspecified"
        candidates = [c for c in self.clusters if c.capability_key == capability]
        best: SkillCluster | None = None
        best_novelty = float("inf")
        for cluster in candidates:
            distance = novelty_distance(vector, cluster.leader_embedding)
            if distance < best_novelty:
                best, best_novelty = cluster, distance

        if best is not None and best_novelty <= self.novelty_threshold:
            best.member_skill_ids.append(skill.skill_id)
            report.events.append(
                ClusterEvent(
                    segment=segment,
                    event=EVENT_ASSIGNED,
                    skill_id=skill.skill_id,
                    skill_name=skill.skill_name,
                    capability_key=capability,
                    cluster_id=best.cluster_id,
                    novelty=best_novelty,
                )
            )
            return best

        birth_event = BIRTH_VARIANT if candidates else BIRTH_CAPABILITY
        cluster = SkillCluster(
            cluster_id=_cluster_id(capability, skill.skill_id),
            capability_key=capability,
            leader_skill_id=skill.skill_id,
            leader_embedding=list(vector),
            protocol_template=list(skill.action_protocol or []),
            born_segment=segment,
            birth_event=birth_event,
            member_skill_ids=[skill.skill_id],
        )
        self.clusters.append(cluster)
        report.events.append(
            ClusterEvent(
                segment=segment,
                event=birth_event,
                skill_id=skill.skill_id,
                skill_name=skill.skill_name,
                capability_key=capability,
                cluster_id=cluster.cluster_id,
                novelty=None,
            )
        )
        logger.info(
            "cluster %s: %s (%s) founded by skill %s",
            birth_event,
            cluster.cluster_id,
            capability,
            skill.skill_name,
        )
        return cluster

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_version": "skill-cluster-archive-v1",
            "novelty_threshold": self.novelty_threshold,
            "last_segment": self.last_segment,
            "clusters": [asdict(cluster) for cluster in self.clusters],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillClusterArchive":
        archive = cls(
            novelty_threshold=float(
                payload.get("novelty_threshold", DEFAULT_NOVELTY_THRESHOLD)
            )
        )
        archive.last_segment = int(payload.get("last_segment", -1))
        for item in payload.get("clusters", []):
            archive.clusters.append(SkillCluster(**item))
        return archive

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SkillClusterArchive":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_or_create(
        cls,
        path: str | Path,
        *,
        novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
    ) -> "SkillClusterArchive":
        source = Path(path)
        if source.exists():
            return cls.load(source)
        return cls(novelty_threshold=novelty_threshold)


def calibrate_novelty_threshold(skills: list[Skill]) -> dict[str, Any]:
    """Calibration table for the novelty threshold τ.

    Reports pairwise novelty within each capability (variants of the same
    action grammar — the range τ must tolerate) versus across capabilities
    (different grammars — the range τ must reject). A suggested τ is the
    midpoint between the within-capability p90 and the smallest observed
    cross-capability distance, falling back to the module default when the
    two ranges overlap.
    """
    eligible = [s for s in skills if s.status not in EXCLUDED_STATUSES]
    vectors = {s.skill_id: _skill_vector(s) for s in eligible}

    within: list[float] = []
    cross: list[float] = []
    for i, left in enumerate(eligible):
        for right in eligible[i + 1 :]:
            distance = novelty_distance(
                vectors[left.skill_id], vectors[right.skill_id]
            )
            if (left.capability_key or "") == (right.capability_key or ""):
                within.append(distance)
            else:
                cross.append(distance)

    def _stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        return {
            "n": len(values),
            "min": round(ordered[0], 4),
            "median": round(ordered[len(ordered) // 2], 4),
            "max": round(ordered[-1], 4),
            "p90": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 4),
        }

    within_stats = _stats(within)
    cross_stats = _stats(cross)
    suggested = DEFAULT_NOVELTY_THRESHOLD
    if within and cross:
        within_p90 = within_stats["p90"]
        cross_min = cross_stats["min"]
        if within_p90 < cross_min:
            suggested = round((within_p90 + cross_min) / 2.0, 4)
    return {
        "within_capability": within_stats,
        "cross_capability": cross_stats,
        "suggested_novelty_threshold": suggested,
        "separable": bool(within and cross and within_stats["p90"] < cross_stats["min"]),
    }
