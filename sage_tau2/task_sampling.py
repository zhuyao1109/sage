"""Train-task sampling for sage_tau2 online runs.

Default policy (``allow_task_resampling=False``): each task id from the
official split may appear at most once across an evolve run. Pools are
shuffled once per domain; segments consume the next slice without replacement.

With ``balance_difficulty=True``, pools are stratified by PERSONA
(Easy / None / Hard) and bug-count tertile, then round-robin interleaved so
each segment slice sees a similar mix of easy/hard telecom tasks.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PERSONA_RE = re.compile(r"\[PERSONA:([^\]]*)\]", re.IGNORECASE)
_ISSUE_RE = re.compile(r"^\[([^\]]+)\]")


def task_difficulty_key(task_id: str) -> tuple[str, int]:
    """Return ``(persona, bug_bucket)`` for stratified scheduling.

    ``bug_bucket``: 0 = ≤3 bugs, 1 = 4–5, 2 = ≥6 (proxy for combinatorial hard).
    """
    text = str(task_id or "")
    m = _PERSONA_RE.search(text)
    persona = (m.group(1).strip() if m else "None") or "None"
    # Normalize common labels.
    low = persona.lower()
    if low == "easy":
        persona = "Easy"
    elif low == "hard":
        persona = "Hard"
    else:
        persona = "None"
    body = _PERSONA_RE.sub("", text)
    body = _ISSUE_RE.sub("", body)
    n_bugs = len([p for p in body.split("|") if p.strip()])
    if n_bugs <= 3:
        bucket = 0
    elif n_bugs <= 5:
        bucket = 1
    else:
        bucket = 2
    return persona, bucket


def balance_task_ids_by_difficulty(task_ids: list[str], *, seed: int) -> list[str]:
    """Shuffle within difficulty strata, then interleave for even PERSONA mix.

    Contiguous slices (one segment each) therefore draw a similar Easy / None /
    Hard mix. Within each PERSONA, bug-count buckets are also interleaved.
    """
    rng = random.Random(int(seed))
    # persona -> bug_bucket -> ids
    nested: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for tid in task_ids:
        persona, bucket = task_difficulty_key(tid)
        nested[persona][bucket].append(str(tid))
    persona_order = ["Easy", "None", "Hard"]
    # Extra personas (if any) appended stably.
    for p in sorted(nested.keys()):
        if p not in persona_order:
            persona_order.append(p)
    # Flatten each persona into an interleaved bug-bucket stream, then shuffle
    # is already done inside buckets.
    persona_streams: dict[str, list[str]] = {}
    for persona in persona_order:
        buckets = nested.get(persona) or {}
        for b in buckets:
            rng.shuffle(buckets[b])
        stream: list[str] = []
        bkeys = sorted(buckets.keys())
        while True:
            progressed = False
            for b in bkeys:
                if buckets[b]:
                    stream.append(buckets[b].pop())
                    progressed = True
            if not progressed:
                break
        persona_streams[persona] = stream
    out: list[str] = []
    while True:
        progressed = False
        for persona in persona_order:
            stream = persona_streams.get(persona) or []
            if stream:
                out.append(stream.pop(0))
                progressed = True
        if not progressed:
            break
    return out


def split_task_ids(domain: str, split_name: str) -> list[str]:
    from tau2.runner.helpers import load_task_splits

    splits = load_task_splits(domain) or {}
    ids = splits.get(split_name)
    if not ids:
        path = Path("data/tau2/domains") / domain / "split_tasks.json"
        if path.exists():
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            ids = data.get(split_name) or data.get("base") or []
    if not ids:
        raise RuntimeError(f"No split '{split_name}' for domain={domain}")
    return list(ids)


def train_pool_size(split_name: str, domains: list[str]) -> dict[str, int]:
    return {domain: len(split_task_ids(domain, split_name)) for domain in domains}


def total_train_pool(split_name: str, domains: list[str]) -> int:
    return sum(train_pool_size(split_name, domains).values())


def sample_fixed_val_ids(
    *,
    domain: str,
    split_name: str,
    val_size: int,
    seed: int,
) -> list[str]:
    """Sample a fixed validation set (shuffled once; order is stable)."""
    n = max(0, int(val_size))
    if n <= 0:
        return []
    pool = list(split_task_ids(domain, split_name))
    if n > len(pool):
        raise ValueError(
            f"val_size={n} exceeds {split_name} pool for {domain} ({len(pool)})"
        )
    rng = random.Random(int(seed))
    picked = list(pool)
    rng.shuffle(picked)
    return picked[:n]


@dataclass(slots=True)
class TaskSchedule:
    """Cross-segment without-replacement schedule per domain."""

    split_name: str
    seed: int
    order: dict[str, list[str]] = field(default_factory=dict)
    cursor: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        domains: list[str],
        split_name: str,
        seed: int,
        exclude_ids: dict[str, set[str] | list[str]] | None = None,
        balance_difficulty: bool = False,
    ) -> TaskSchedule:
        rng = random.Random(seed)
        order: dict[str, list[str]] = {}
        exclude = exclude_ids or {}
        for domain in domains:
            blocked = {str(x) for x in (exclude.get(domain) or [])}
            pool = [tid for tid in split_task_ids(domain, split_name) if tid not in blocked]
            if balance_difficulty:
                pool = balance_task_ids_by_difficulty(pool, seed=seed + hash(domain) % 10_000)
            else:
                rng.shuffle(pool)
            order[domain] = pool
        return cls(
            split_name=split_name,
            seed=seed,
            order=order,
            cursor={domain: 0 for domain in domains},
        )

    @classmethod
    def from_state(cls, payload: dict[str, Any]) -> TaskSchedule:
        return cls(
            split_name=str(payload.get("split_name") or "train"),
            seed=int(payload.get("seed") or 0),
            order={
                str(domain): list(ids)
                for domain, ids in (payload.get("order") or {}).items()
            },
            cursor={
                str(domain): int(pos)
                for domain, pos in (payload.get("cursor") or {}).items()
            },
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "seed": self.seed,
            "order": self.order,
            "cursor": self.cursor,
        }

    def remaining(self, domain: str) -> int:
        ids = self.order.get(domain) or []
        pos = int(self.cursor.get(domain) or 0)
        return max(0, len(ids) - pos)

    def total_remaining(self) -> int:
        return sum(self.remaining(domain) for domain in self.order)

    def used_task_ids(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for domain, ids in self.order.items():
            pos = int(self.cursor.get(domain) or 0)
            out[domain] = list(ids[:pos])
        return out

    def sample(
        self,
        quotas: dict[str, int],
        *,
        allow_resampling: bool = False,
        segment_seed: int | None = None,
    ) -> dict[str, list[str]]:
        if allow_resampling:
            return sample_task_ids_with_replacement(
                quotas=quotas,
                split_name=self.split_name,
                seed=int(segment_seed if segment_seed is not None else self.seed),
            )
        out: dict[str, list[str]] = {}
        for domain, n in quotas.items():
            want = max(0, int(n))
            ids = self.order.get(domain) or []
            pos = int(self.cursor.get(domain) or 0)
            take = ids[pos : pos + want]
            self.cursor[domain] = pos + len(take)
            if take:
                out[domain] = take
        return out


def sample_task_ids_with_replacement(
    *,
    quotas: dict[str, int],
    split_name: str,
    seed: int,
) -> dict[str, list[str]]:
    """Legacy per-segment sampling; may repeat task ids when quota > pool."""
    rng = random.Random(seed)
    out: dict[str, list[str]] = {}
    for domain, n in quotas.items():
        pool = split_task_ids(domain, split_name)
        if not pool:
            raise RuntimeError(f"Empty pool for {domain}/{split_name}")
        if n <= len(pool):
            picked = list(pool)
            rng.shuffle(picked)
            out[domain] = picked[:n]
        else:
            picked: list[str] = []
            while len(picked) < n:
                batch = list(pool)
                rng.shuffle(batch)
                picked.extend(batch)
            out[domain] = picked[:n]
    return out


def sample_task_ids(
    *,
    quotas: dict[str, int],
    split_name: str,
    seed: int,
    schedule: TaskSchedule | None = None,
    allow_task_resampling: bool = False,
) -> tuple[dict[str, list[str]], TaskSchedule | None]:
    """Sample tasks for one segment.

    Returns ``(task_map, updated_schedule)``. When resampling is disabled,
    ``schedule`` must be provided (or will be created).
    """
    if allow_task_resampling:
        return (
            sample_task_ids_with_replacement(
                quotas=quotas,
                split_name=split_name,
                seed=seed,
            ),
            schedule,
        )
    sched = schedule or TaskSchedule.create(
        domains=list(quotas.keys()),
        split_name=split_name,
        seed=seed,
    )
    return sched.sample(quotas, allow_resampling=False), sched


def validate_evolve_budget(
    *,
    segment_size: int,
    num_segments: int,
    quotas: dict[str, int],
    split_name: str,
    allow_task_resampling: bool,
) -> None:
    if allow_task_resampling:
        return
    pools = train_pool_size(split_name, list(quotas.keys()))
    for domain, quota in quotas.items():
        pool_n = pools.get(domain, 0)
        if int(quota) > pool_n:
            raise ValueError(
                f"Per-segment quota exceeds {split_name} pool for {domain}: "
                f"quota={quota} > pool={pool_n}. Reduce segment_size."
            )
    total_requested = int(segment_size) * int(num_segments)
    total_available = sum(pools.values())
    if total_requested > total_available:
        # Legal: later segments become partial and the run stops once pools empty.
        return
