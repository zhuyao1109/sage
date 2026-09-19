"""Tests for sage_tau2 train task sampling without replacement."""

from __future__ import annotations

import pytest

from collections import Counter

from sage_tau2.task_sampling import (
    TaskSchedule,
    balance_task_ids_by_difficulty,
    sample_fixed_val_ids,
    sample_task_ids,
    sample_task_ids_with_replacement,
    task_difficulty_key,
    validate_evolve_budget,
)


def test_without_replacement_no_duplicate_across_segments(monkeypatch):
    pools = {
        "airline": ["a1", "a2", "a3"],
        "retail": ["r1", "r2", "r3", "r4"],
    }

    def fake_split(domain: str, split_name: str) -> list[str]:
        return list(pools[domain])

    monkeypatch.setattr("sage_tau2.task_sampling.split_task_ids", fake_split)

    schedule = TaskSchedule.create(domains=["airline", "retail"], split_name="train", seed=1)
    q1 = {"airline": 2, "retail": 2}
    m1, schedule = sample_task_ids(
        quotas=q1,
        split_name="train",
        seed=10,
        schedule=schedule,
        allow_task_resampling=False,
    )
    m2, schedule = sample_task_ids(
        quotas=q1,
        split_name="train",
        seed=11,
        schedule=schedule,
        allow_task_resampling=False,
    )
    m3, schedule = sample_task_ids(
        quotas={"airline": 2, "retail": 3},
        split_name="train",
        seed=12,
        schedule=schedule,
        allow_task_resampling=False,
    )

    all_ids = m1["airline"] + m2["airline"] + m3.get("airline", [])
    assert len(all_ids) == len(set(all_ids))
    all_retail = m1["retail"] + m2["retail"] + m3.get("retail", [])
    assert len(all_retail) == len(set(all_retail))
    assert schedule.total_remaining() == 0
    assert set(m1["airline"] + m2["airline"] + m3.get("airline", [])) == {"a1", "a2", "a3"}
    assert set(m1["retail"] + m2["retail"] + m3.get("retail", [])) == {"r1", "r2", "r3", "r4"}


def test_with_replacement_can_repeat(monkeypatch):
    monkeypatch.setattr(
        "sage_tau2.task_sampling.split_task_ids",
        lambda domain, split_name: ["x1", "x2"],
    )
    out = sample_task_ids_with_replacement(
        quotas={"airline": 5},
        split_name="train",
        seed=0,
    )
    assert len(out["airline"]) == 5
    assert len(set(out["airline"])) < 5


def test_validate_budget_rejects_quota_above_pool(monkeypatch):
    monkeypatch.setattr(
        "sage_tau2.task_sampling.train_pool_size",
        lambda split_name, domains: {"airline": 30, "retail": 74},
    )
    with pytest.raises(ValueError, match="Per-segment quota exceeds"):
        validate_evolve_budget(
            segment_size=100,
            num_segments=2,
            quotas={"airline": 31, "retail": 42},
            split_name="train",
            allow_task_resampling=False,
        )


def test_validate_budget_allows_partial_multi_segment(monkeypatch):
    monkeypatch.setattr(
        "sage_tau2.task_sampling.train_pool_size",
        lambda split_name, domains: {"airline": 30, "retail": 74},
    )
    validate_evolve_budget(
        segment_size=100,
        num_segments=2,
        quotas={"airline": 17, "retail": 42},
        split_name="train",
        allow_task_resampling=False,
    )


def test_sample_fixed_val_ids_stable(monkeypatch):
    monkeypatch.setattr(
        "sage_tau2.task_sampling.split_task_ids",
        lambda domain, split_name: [f"t{i}" for i in range(10)],
    )
    a = sample_fixed_val_ids(domain="airline", split_name="train", val_size=5, seed=42)
    b = sample_fixed_val_ids(domain="airline", split_name="train", val_size=5, seed=42)
    assert a == b
    assert len(a) == 5
    assert len(set(a)) == 5


def test_schedule_excludes_val_holdout(monkeypatch):
    monkeypatch.setattr(
        "sage_tau2.task_sampling.split_task_ids",
        lambda domain, split_name: ["a", "b", "c", "d", "e"],
    )
    schedule = TaskSchedule.create(
        domains=["airline"],
        split_name="train",
        seed=1,
        exclude_ids={"airline": ["b", "d"]},
    )
    assert set(schedule.order["airline"]) == {"a", "c", "e"}
    assert schedule.remaining("airline") == 3


def test_balance_difficulty_interleaves_persona_across_segments(monkeypatch):
    pool = []
    for persona in ("Easy", "None", "Hard"):
        for i in range(20):
            bugs = "|".join([f"bug{j}" for j in range((i % 3) + 2)])
            pool.append(f"[mobile_data_issue]{bugs}[PERSONA:{persona}]")

    monkeypatch.setattr(
        "sage_tau2.task_sampling.split_task_ids",
        lambda domain, split_name: list(pool),
    )
    schedule = TaskSchedule.create(
        domains=["telecom"],
        split_name="train_large",
        seed=7,
        balance_difficulty=True,
    )
    assert len(schedule.order["telecom"]) == len(pool)
    # Each 30-task segment should see all three PERSONA labels.
    for start in range(0, 60, 30):
        chunk = schedule.order["telecom"][start : start + 30]
        personas = {task_difficulty_key(t)[0] for t in chunk}
        assert personas == {"Easy", "None", "Hard"}
        counts = Counter(task_difficulty_key(t)[0] for t in chunk)
        # Round-robin → roughly even (±2).
        assert max(counts.values()) - min(counts.values()) <= 2


def test_task_difficulty_key_parses_persona_and_bugs():
    assert task_difficulty_key(
        "[service_issue]a|b[PERSONA:Easy]"
    ) == ("Easy", 0)
    assert task_difficulty_key(
        "[mms_issue]a|b|c|d|e|f[PERSONA:Hard]"
    ) == ("Hard", 2)
    assert balance_task_ids_by_difficulty(
        [
            "[x]a|b[PERSONA:Hard]",
            "[x]a[PERSONA:Easy]",
            "[x]a|b|c[PERSONA:None]",
        ],
        seed=0,
    )[0].endswith("[PERSONA:Easy]")
