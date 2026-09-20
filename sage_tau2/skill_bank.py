"""Simple JSON skill bank for τ² (isolated from sage_mas.SkillBank)."""

from __future__ import annotations

from pathlib import Path

from sage_tau2.credit import initialize_credit
from sage_tau2.schemas import SkillStatus, Tau2Skill
from sage_tau2.serialization import read_json, skill_from_dict, skill_to_dict, write_json


class Tau2SkillBank:
    def __init__(self, path: str | Path, *, redundancy_threshold: float = 0.92):
        self.path = Path(path)
        self.redundancy_threshold = float(redundancy_threshold)
        self.skills: list[Tau2Skill] = []
        if self.path.exists():
            payload = read_json(self.path)
            items = payload if isinstance(payload, list) else payload.get("skills") or []
            for item in items:
                if isinstance(item, dict):
                    skill = skill_from_dict(item)
                    initialize_credit(skill)
                    self.skills.append(skill)

    def save(self) -> None:
        write_json(self.path, [skill_to_dict(s) for s in self.skills])

    def add(self, skill: Tau2Skill) -> Tau2Skill:
        initialize_credit(skill)
        duplicate = self._find_duplicate(skill)
        if duplicate is not None:
            duplicate.evidence_ids = sorted(
                set(duplicate.evidence_ids + skill.evidence_ids)
            )
            duplicate.support_count = max(
                duplicate.support_count,
                skill.support_count,
                len(duplicate.evidence_ids),
            )
            if skill.description and len(skill.description) > len(duplicate.description):
                duplicate.description = skill.description
            if skill.expected_effect:
                duplicate.expected_effect = skill.expected_effect
            self._merge_gate_metadata(duplicate, skill)
            refreshed = self._maybe_refresh_protocol(duplicate, skill)
            if refreshed and skill.precondition:
                duplicate.precondition = skill.precondition
            elif not duplicate.precondition and skill.precondition:
                duplicate.precondition = skill.precondition
            return duplicate
        if skill.status == SkillStatus.CANDIDATE:
            skill.status = SkillStatus.PROVISIONAL
        self.skills.append(skill)
        return skill

    def extend(self, skills: list[Tau2Skill]) -> list[Tau2Skill]:
        return [self.add(skill) for skill in skills]

    def active(self) -> list[Tau2Skill]:
        return [
            s
            for s in self.skills
            if s.status not in {SkillStatus.REJECTED, SkillStatus.RETIRED}
        ]

    @staticmethod
    def _merge_gate_metadata(existing: Tau2Skill, incoming: Tau2Skill) -> None:
        """Recompute bug gate fields when duplicate skills merge evidence."""
        from sage_tau2.distill import _activation_gate_payload

        meta = dict(existing.metadata or {})
        incoming_meta = incoming.metadata or {}
        sigs: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for source in (meta, incoming_meta):
            for sig in source.get("evidence_bug_signatures") or source.get(
                "activation_signatures"
            ) or []:
                if not isinstance(sig, (list, tuple)):
                    continue
                key = tuple(str(t) for t in sig)
                if not key or key in seen:
                    continue
                seen.add(key)
                sigs.append(list(key))
        if not sigs:
            return
        gate = _activation_gate_payload(sigs)
        union_ordered: list[str] = []
        seen_tags: set[str] = set()
        for sig in sigs:
            for tag in sig:
                t = str(tag)
                if t and t not in seen_tags:
                    seen_tags.add(t)
                    union_ordered.append(t)
        meta.update(gate)
        meta["evidence_bug_signatures"] = sigs
        meta["activation_signatures"] = sigs
        meta["bug_tags_core"] = list(gate.get("bug_intersection") or [])
        meta["bug_tags_union"] = union_ordered
        existing.metadata = meta

    @staticmethod
    def _protocol_quality_score(protocol: list[str]) -> tuple[int, int]:
        guide_n = sum(
            1 for s in protocol if str(s).lower().startswith("guide user:")
        )
        return (len(protocol), guide_n)

    def _maybe_refresh_protocol(
        self, existing: Tau2Skill, incoming: Tau2Skill
    ) -> bool:
        """Prefer shorter / fewer-guide protocols when merging duplicates."""
        new_proto = list(incoming.action_protocol or [])
        old_proto = list(existing.action_protocol or [])
        if not new_proto:
            return False
        old_contract = (existing.metadata or {}).get("execution_contract") or {}
        new_contract = (incoming.metadata or {}).get("execution_contract") or {}
        if old_contract.get("version", 0) >= 2:
            # A shorter trace is not proof that a dependency is optional.
            return False
        if new_contract.get("version", 0) >= 2:
            existing.action_protocol = new_proto
            existing.metadata["execution_contract"] = new_contract
            existing.metadata["agent_tool_protocol"] = incoming.metadata.get("agent_tool_protocol", new_proto)
            return True
        if not old_proto or self._protocol_quality_score(new_proto) < self._protocol_quality_score(
            old_proto
        ):
            existing.action_protocol = new_proto
            meta = dict(existing.metadata or {})
            incoming_meta = incoming.metadata or {}
            for key in (
                "agent_tool_protocol",
                "guide_user_steps",
                "protocol_len",
                "bug_tags_core",
                "bug_intersection",
            ):
                if key in incoming_meta:
                    meta[key] = incoming_meta[key]
            existing.metadata = meta
            return True
        return False

    def _find_duplicate(self, skill: Tau2Skill) -> Tau2Skill | None:
        def _agent_proto(s: Tau2Skill) -> tuple[str, ...]:
            meta = s.metadata or {}
            proto = meta.get("agent_tool_protocol") or s.action_protocol or []
            # Prefer tool-shaped rows when action_protocol is mixed.
            from sage_tau2.distill import agent_tool_protocol

            tools = agent_tool_protocol(
                list(proto), domain=getattr(s, "domain", None)
            )
            return tuple(tools or proto)

        def _telecom_write_sig(s: Tau2Skill) -> tuple[str, ...] | None:
            """Coarse identity for telecom skills: success_mode + write set.

            Telecom bucket keys are coarse (capability + write set), so the
            distilled template can vary across segments in probe depth only.
            Exact-protocol dedupe would re-add the same capability every
            segment; merge by write signature instead.
            """
            if str(getattr(s, "domain", "") or "").lower() not in {
                "telecom",
                "telecom-workflow",
            }:
                return None
            from sage_tau2.distill import primary_write_names

            writes = tuple(sorted(set(primary_write_names(list(_agent_proto(s))))))
            if not writes:
                return None
            mode = str((s.metadata or {}).get("success_mode") or "").strip()
            return (mode, *writes)

        proto = _agent_proto(skill)
        sig = _telecom_write_sig(skill)
        for existing in self.skills:
            if existing.status in {SkillStatus.REJECTED, SkillStatus.RETIRED}:
                continue
            existing_proto = _agent_proto(existing)
            if existing.domain != skill.domain:
                continue
            new_v = ((skill.metadata or {}).get("execution_contract") or {}).get("version", 0)
            old_v = ((existing.metadata or {}).get("execution_contract") or {}).get("version", 0)
            if new_v >= 2 or old_v >= 2:
                # Ordered cross-role steps and parameter settings define the
                # learned contract; a shared write name alone does not.
                if existing.action_protocol == skill.action_protocol:
                    return existing
                continue
            if existing_proto == proto and proto:
                return existing
            if (
                skill.capability_key
                and existing.capability_key == skill.capability_key
                and proto
                and existing_proto == proto
            ):
                return existing
            if sig and _telecom_write_sig(existing) == sig:
                return existing
        return None
