"""WebShop helpers for LLM / SAGE-MAS evaluation (fixed goal streams)."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Any, Sequence

from agent_system.environments.env_manager import WebshopEnvironmentManager

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Matches agent_system/environments/env_package/webshop/envs.py splits.
WEBSHOP_TEST_SIZE = 500

# Default per-segment quotas for category_stratified sampling (sum=50).
# fashion is plurality but not overwhelming; minority cats stay covered.
DEFAULT_CATEGORY_QUOTAS: dict[str, int] = {
    "fashion": 18,
    "garden": 20,
    "beauty": 4,
    "electronics": 4,
    "grocery": 4,
}


class _CfgNode:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def list_webshop_goal_indices(
    *,
    split: str = "test",
    num_goals: int | None = None,
    total_goals: int | None = None,
) -> list[int]:
    """Return a deterministic goal-index stream.

    - ``test``: official eval stream ``0 .. 499`` (capped at ``num_goals``).
    - ``train``: ``500 ..`` (requires ``total_goals`` when known).
    """
    normalized = str(split or "test").strip().lower()
    if normalized in {"test", "eval", "valid", "valid_unseen"}:
        indices = list(range(WEBSHOP_TEST_SIZE))
    elif normalized in {"train", "training"}:
        if total_goals is None:
            raise ValueError(
                "train split requires total_goals (len(env.server.goals))"
            )
        indices = list(range(WEBSHOP_TEST_SIZE, int(total_goals)))
    else:
        raise ValueError(f"Unknown WebShop split: {split}")
    if num_goals is not None:
        indices = indices[: max(0, int(num_goals))]
    return indices


def build_webshop_env_manager(
    *,
    env_num: int,
    seed: int,
    is_train: bool = False,
    num_cpus_per_worker: float = 0.1,
    observation_mode: str = "text",
    history_length: int = 0,
):
    """Build a WebShop env manager (call ``reset({'goal_indices': ...})``)."""
    from agent_system.environments.env_package.webshop import (
        build_webshop_envs,
        webshop_projection,
    )

    resources_per_worker = {
        "num_cpus": float(num_cpus_per_worker),
        "num_gpus": 0.0,
    }
    envs = build_webshop_envs(
        seed=seed,
        env_num=env_num,
        group_n=1,
        is_train=is_train,
        env_kwargs={"observation_mode": observation_mode},
        resources_per_worker=resources_per_worker,
    )
    cfg = _CfgNode(env=_CfgNode(history_length=int(history_length)))
    return WebshopEnvironmentManager(envs, webshop_projection, cfg)


def load_webshop_goals(
    *,
    seed: int = 1,
    num_cpus_per_worker: float = 0.1,
) -> list[dict[str, Any]]:
    """Load the full goal table once (requires Ray / WebShop data)."""
    import ray

    manager = build_webshop_env_manager(
        env_num=1,
        seed=seed,
        is_train=False,
        num_cpus_per_worker=num_cpus_per_worker,
    )
    try:
        goals = ray.get(manager.envs._workers[0].get_goals.remote())
        return [
            dict(goal) if isinstance(goal, dict) else {"raw": goal}
            for goal in goals
        ]
    finally:
        manager.close()


def goal_category(goal: dict[str, Any] | None) -> str:
    if not isinstance(goal, dict):
        return "unknown"
    value = str(goal.get("category") or "").strip().lower()
    return value or "unknown"


def indices_by_category(
    goals: Sequence[dict[str, Any]],
    *,
    split: str = "test",
) -> dict[str, list[int]]:
    """Bucket goal indices by official ``category`` for a split."""
    normalized = str(split or "test").strip().lower()
    if normalized in {"test", "eval", "valid", "valid_unseen", "valid_seen"}:
        candidates = range(min(WEBSHOP_TEST_SIZE, len(goals)))
    elif normalized in {"train", "training"}:
        candidates = range(WEBSHOP_TEST_SIZE, len(goals))
    else:
        raise ValueError(f"Unknown WebShop split: {split}")
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx in candidates:
        buckets[goal_category(goals[idx])].append(int(idx))
    return dict(buckets)


def select_category_stratified_indices(
    buckets: dict[str, list[int]],
    *,
    num_goals: int,
    segment_size: int,
    quotas: dict[str, int] | None = None,
    seed: int = 1,
    excluded: set[int] | None = None,
) -> list[int]:
    """Sample goals so each segment-sized block covers quota categories.

    Shortfalls are filled from ``fashion``, then any remaining pool. Each
    block is shuffled so category order is not fixed inside a segment.
    """
    if num_goals <= 0:
        return []
    segment_size = max(1, int(segment_size or num_goals))
    quota_map = {
        str(key).strip().lower(): max(0, int(value))
        for key, value in (quotas or DEFAULT_CATEGORY_QUOTAS).items()
    }
    if not quota_map:
        quota_map = dict(DEFAULT_CATEGORY_QUOTAS)
    excluded = {int(i) for i in (excluded or set())}
    rng = random.Random(int(seed))

    pools: dict[str, list[int]] = {}
    for category, indices in buckets.items():
        pool = [int(i) for i in indices if int(i) not in excluded]
        rng.shuffle(pool)
        pools[str(category).strip().lower()] = pool

    selected: list[int] = []
    while len(selected) < num_goals:
        block: list[int] = []
        for category, quota in quota_map.items():
            pool = pools.get(category) or []
            take = min(
                int(quota),
                len(pool),
                num_goals - len(selected) - len(block),
            )
            for _ in range(take):
                block.append(pool.pop())
        fill_order = ["fashion", *[c for c in pools if c != "fashion"]]
        while (
            len(block) < segment_size
            and len(selected) + len(block) < num_goals
        ):
            progressed = False
            for category in fill_order:
                pool = pools.get(category) or []
                if pool:
                    block.append(pool.pop())
                    progressed = True
                    break
            if not progressed:
                break
        if not block:
            break
        rng.shuffle(block)
        selected.extend(block)
        if not any(pools.values()):
            break
    return selected[:num_goals]


def task_family_from_instruction(instruction: str) -> str:
    """Coarse family tag for logging (not ALFWorld-style hard rules)."""
    text = str(instruction or "").lower()
    for keyword, family in (
        ("shirt", "apparel"),
        ("dress", "apparel"),
        ("shoe", "footwear"),
        ("boot", "footwear"),
        ("laptop", "electronics"),
        ("phone", "electronics"),
        ("headphone", "electronics"),
        ("furniture", "home"),
        ("sofa", "home"),
        ("kitchen", "home"),
        ("beauty", "beauty"),
        ("makeup", "beauty"),
    ):
        if keyword in text:
            return family
    return "webshop"


def pad_goal_indices(goal_indices: Sequence[int], batch_size: int) -> list[int]:
    """Pad the last batch by repeating the final goal (caller must mask)."""
    indices = [int(i) for i in goal_indices]
    if not indices:
        raise ValueError("goal_indices must be non-empty")
    if len(indices) >= batch_size:
        return indices[:batch_size]
    pad = [indices[-1]] * (batch_size - len(indices))
    return indices + pad
