"""Capability-partitioned skill clustering for τ² (archive only).

Mirrors sage_mas.skill_clustering principles without ALFWorld embeddings:
capability_key is a hard pre-partition; within a capability, new skills join
the nearest protocol-leader or birth a variant cluster.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sage_tau2.schemas import SkillStatus, Tau2Skill
from sage_tau2.serialization import read_json, write_json
from sage_tau2.task_context import organizational_capability_key

BIRTH_CAPABILITY = "capability_birth"
BIRTH_VARIANT = "variant_birth"
EVENT_ASSIGNED = "assigned"
DEFAULT_NOVELTY_THRESHOLD = 0.35
EXCLUDED = {SkillStatus.REJECTED, SkillStatus.RETIRED}


def _protocol_set(protocol: list[str]) -> set[str]:
    return {step.split("(", 1)[0].strip() for step in protocol if step}


def protocol_novelty(left: list[str], right: list[str]) -> float:
    """Jaccard distance on tool names in [0, 1]."""
    a, b = _protocol_set(left), _protocol_set(right)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - (inter / float(union))


def _cluster_id(capability_key: str, leader_skill_id: str) -> str:
    digest = hashlib.sha1(f"{capability_key}|{leader_skill_id}".encode("utf-8"))
    return f"cluster-{digest.hexdigest()[:12]}"


@dataclass
class SkillCluster:
    cluster_id: str
    capability_key: str
    leader_skill_id: str
    protocol_template: list[str]
    born_segment: int
    birth_event: str
    member_skill_ids: list[str] = field(default_factory=list)
    segment_stats: list[dict[str, Any]] = field(default_factory=list)


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
        return [
            e for e in self.events if e.event in {BIRTH_CAPABILITY, BIRTH_VARIANT}
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"segment": self.segment, "events": [asdict(e) for e in self.events]}


class SkillClusterArchive:
    def __init__(self, *, novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD) -> None:
        self.novelty_threshold = float(novelty_threshold)
        self.clusters: list[SkillCluster] = []
        self.last_segment: int = -1

    def cluster_of(self, skill_id: str) -> SkillCluster | None:
        for cluster in self.clusters:
            if skill_id in cluster.member_skill_ids:
                return cluster
        return None

    def clusters_for_capability(self, capability_key: str) -> list[SkillCluster]:
        key = str(capability_key or "").strip()
        return [c for c in self.clusters if c.capability_key == key]

    def update(
        self,
        skills: list[Tau2Skill],
        *,
        segment: int,
    ) -> ClusterUpdateReport:
        report = ClusterUpdateReport(segment=segment)
        active = [s for s in skills if s.status not in EXCLUDED]
        ordered = sorted(
            active,
            key=lambda s: (
                organizational_capability_key(s) or s.capability_key or "",
                -int(s.support_count or 0),
                s.skill_id,
            ),
        )
        by_id = {s.skill_id: s for s in ordered}

        for skill in ordered:
            existing = self.cluster_of(skill.skill_id)
            if existing is not None:
                report.events.append(
                    ClusterEvent(
                        segment=segment,
                        event=EVENT_ASSIGNED,
                        skill_id=skill.skill_id,
                        skill_name=skill.skill_name,
                        capability_key=organizational_capability_key(skill)
                        or skill.capability_key,
                        cluster_id=existing.cluster_id,
                        novelty=0.0,
                    )
                )
                continue

            capability = (
                organizational_capability_key(skill)
                or skill.capability_key
                or "tau2.unknown"
            )
            peers = self.clusters_for_capability(capability)
            if not peers:
                cluster = SkillCluster(
                    cluster_id=_cluster_id(capability, skill.skill_id),
                    capability_key=capability,
                    leader_skill_id=skill.skill_id,
                    protocol_template=list(skill.action_protocol or []),
                    born_segment=segment,
                    birth_event=BIRTH_CAPABILITY,
                    member_skill_ids=[skill.skill_id],
                )
                self.clusters.append(cluster)
                report.events.append(
                    ClusterEvent(
                        segment=segment,
                        event=BIRTH_CAPABILITY,
                        skill_id=skill.skill_id,
                        skill_name=skill.skill_name,
                        capability_key=capability,
                        cluster_id=cluster.cluster_id,
                        novelty=1.0,
                    )
                )
                continue

            best: SkillCluster | None = None
            best_nov = 1.0
            for peer in peers:
                nov = protocol_novelty(skill.action_protocol, peer.protocol_template)
                if nov < best_nov:
                    best_nov = nov
                    best = peer
            if best is not None and best_nov <= self.novelty_threshold:
                if skill.skill_id not in best.member_skill_ids:
                    best.member_skill_ids.append(skill.skill_id)
                report.events.append(
                    ClusterEvent(
                        segment=segment,
                        event=EVENT_ASSIGNED,
                        skill_id=skill.skill_id,
                        skill_name=skill.skill_name,
                        capability_key=capability,
                        cluster_id=best.cluster_id,
                        novelty=best_nov,
                    )
                )
            else:
                cluster = SkillCluster(
                    cluster_id=_cluster_id(capability, skill.skill_id),
                    capability_key=capability,
                    leader_skill_id=skill.skill_id,
                    protocol_template=list(skill.action_protocol or []),
                    born_segment=segment,
                    birth_event=BIRTH_VARIANT,
                    member_skill_ids=[skill.skill_id],
                )
                self.clusters.append(cluster)
                report.events.append(
                    ClusterEvent(
                        segment=segment,
                        event=BIRTH_VARIANT,
                        skill_id=skill.skill_id,
                        skill_name=skill.skill_name,
                        capability_key=capability,
                        cluster_id=cluster.cluster_id,
                        novelty=best_nov,
                    )
                )

        # Refresh segment snapshots.
        for cluster in self.clusters:
            members = [by_id[sid] for sid in cluster.member_skill_ids if sid in by_id]
            snap = {
                "segment": segment,
                "member_count": len(members),
                "support_total": sum(int(s.support_count or 0) for s in members),
                "evidence_total": sum(len(s.evidence_ids) for s in members),
                "utility_mean": _mean_utility(members),
            }
            cluster.segment_stats = [
                e for e in cluster.segment_stats if e.get("segment") != segment
            ] + [snap]

        self.last_segment = segment
        return report

    def to_dict(self) -> dict[str, Any]:
        return {
            "novelty_threshold": self.novelty_threshold,
            "last_segment": self.last_segment,
            "clusters": [asdict(c) for c in self.clusters],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillClusterArchive":
        archive = cls(
            novelty_threshold=float(
                payload.get("novelty_threshold", DEFAULT_NOVELTY_THRESHOLD)
            )
        )
        archive.last_segment = int(payload.get("last_segment", -1))
        for item in payload.get("clusters") or []:
            archive.clusters.append(SkillCluster(**dict(item)))
        return archive

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "SkillClusterArchive":
        p = Path(path)
        if not p.exists():
            return cls()
        payload = read_json(p)
        if not isinstance(payload, dict):
            return cls()
        return cls.from_dict(payload)


def _mean_utility(skills: list[Tau2Skill]) -> float | None:
    vals: list[float] = []
    for skill in skills:
        credit = skill.metadata.get("skill_credit") or {}
        if "score" in credit:
            vals.append(float(credit["score"]))
        elif skill.metadata.get("utility") is not None:
            vals.append(float(skill.metadata["utility"]))
    if not vals:
        return None
    return sum(vals) / len(vals)
