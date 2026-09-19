"""Unit tests for WebShop SAGE online helpers."""

from __future__ import annotations

from collections import Counter

from examples.prompt_agent.gpt4o_webshop import (
    DEFAULT_CATEGORY_QUOTAS,
    select_category_stratified_indices,
)
from sage_mas.schemas import AtomicOp
from sage_mas.trajectory_adapter import (
    WebShopTrajectoryAdapter,
    build_trajectory_adapter,
)
from sage_mas.webshop_ids import goal_uri, parse_goal_idx


def test_goal_uri_roundtrip():
    assert goal_uri(42) == "webshop://goal/42"
    assert parse_goal_idx("webshop://goal/42") == 42
    assert parse_goal_idx(7) == 7


def test_build_webshop_adapter():
    adapter = build_trajectory_adapter("webshop")
    assert isinstance(adapter, WebShopTrajectoryAdapter)


def test_webshop_adapter_infers_search_click():
    adapter = WebShopTrajectoryAdapter()
    traj = {
        "gamefile": "webshop://goal/0",
        "task": "Find a red shirt under $20",
        "task_family": "apparel",
        "won": False,
        "num_steps": 2,
        "steps": [
            {
                "action": "<think>x</think><action>search[red shirt]</action>",
                "observation": "page A",
                "observation_before": "start",
                "is_action_valid": True,
            },
            {
                "action": "click[item - shirt]",
                "observation": "page A",
                "observation_before": "page A",
                "is_action_valid": False,
            },
        ],
    }
    steps = adapter.adapt(traj)
    ops = [step.atomic_op for step in steps if step.atomic_op != AtomicOp.TERMINATE]
    assert AtomicOp.ACT in ops
    assert AtomicOp.SELECT in ops
    stalled = [
        step
        for step in steps
        if step.metadata and step.metadata.get("stalled")
    ]
    assert stalled


def test_category_stratified_fashion_quota_18():
    assert DEFAULT_CATEGORY_QUOTAS["fashion"] == 18
    assert sum(DEFAULT_CATEGORY_QUOTAS.values()) == 50

    buckets = {
        "fashion": list(range(0, 200)),
        "garden": list(range(200, 280)),
        "beauty": list(range(280, 300)),
        "electronics": list(range(300, 320)),
        "grocery": list(range(320, 340)),
    }
    selected = select_category_stratified_indices(
        buckets,
        num_goals=50,
        segment_size=50,
        quotas=DEFAULT_CATEGORY_QUOTAS,
        seed=1,
    )
    assert len(selected) == 50
    assert len(set(selected)) == 50

    cat_of = {}
    for cat, indices in buckets.items():
        for idx in indices:
            cat_of[idx] = cat
    counts = Counter(cat_of[i] for i in selected)
    assert counts["fashion"] == 18
    assert counts["garden"] == 20
    assert counts["beauty"] == 4
    assert counts["electronics"] == 4
    assert counts["grocery"] == 4
