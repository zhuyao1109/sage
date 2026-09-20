"""Fixed, family-balanced OOD validation stays separate from training evidence."""
from collections import Counter
from types import SimpleNamespace

import yaml

import sage_mas.online_evolution_legacy as online


FAMILIES = [
    "pick_and_place", "pick_two_obj_and_place", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep",
    "look_at_obj_in_light",
]


def runner():
    instance = online.OnlineAlfWorldEvolution.__new__(online.OnlineAlfWorldEvolution)
    instance.config = online.OnlineEvolutionConfig(
        seed=1, val_eval_enabled=True, val_pool_size=12,
        val_eval_dataset="valid_unseen", val_game_selection="family_proportional",
        val_eval_every_segment=True,
    )
    instance.llm_config = {"alfworld": {"data_path": "unused"}}
    return instance


def test_134_task_pool_yields_two_per_family_and_excludes_reserved(monkeypatch):
    # Unequal family sizes make an ordinary random sample insufficient.
    candidates = [f"/valid_unseen/{family}-{i}/trial/game.tw-pddl"
                  for family, count in zip(FAMILIES, [40, 30, 22, 18, 14, 10])
                  for i in range(count)]
    monkeypatch.setattr(online, "_list_unseen_gamefiles", lambda _: candidates)
    instance = runner()
    reserved = set(candidates[:3])
    picked = instance._build_val_pool(reserved)
    assert len(picked) == len(set(picked)) == 12
    assert not reserved.intersection(picked)
    assert Counter(online._task_family_from_gamefile(p) for p in picked) == {
        family: 2 for family in FAMILIES
    }
    assert picked == instance._build_val_pool(reserved)


def test_saved_pool_reused_in_each_of_ten_segment_evaluations(tmp_path, monkeypatch):
    instance = runner()
    instance.state_path = tmp_path / "online_state.json"
    pool = [f"fixed-{i}" for i in range(12)]
    monkeypatch.setattr(instance, "_build_val_pool", lambda excluded: pool)
    state = {}
    instance._ensure_val_pool(state)
    assert state["val_pool"] == pool
    assert state["config"]["val_game_selection"] == "family_proportional"
    monkeypatch.setattr(instance, "_build_val_pool", lambda _: (_ for _ in ()).throw(
        AssertionError("fixed pool must not be resampled")))
    calls = []
    evaluator = SimpleNamespace(evaluate=lambda **kwargs: calls.append(kwargs) or [])
    monkeypatch.setattr(instance, "_gate_evaluator", lambda: evaluator)
    for segment in range(1, 11):
        instance._maybe_run_val_eval(
            state, outcome={}, round_index=segment, agents=[], skills=[],
            train_progress=segment * 6,
        )
    assert len(calls) == 10
    assert all(call["gamefiles"] == pool for call in calls)


def test_distillation_window_does_not_read_validation_records(tmp_path):
    instance = runner()
    prior = tmp_path / "prior.jsonl"
    current = tmp_path / "current.jsonl"
    val = tmp_path / "val.jsonl"
    prior.write_text('{"task_id":"train-prior"}\n', encoding="utf-8")
    current.write_text('{"task_id":"train-current"}\n', encoding="utf-8")
    val.write_text('{"task_id":"validation-only"}\n', encoding="utf-8")
    state = {"trajectory_paths": [str(prior)],
             "val_trials": [{"trajectory_path": str(val)}], "val_pool": [str(val)]}
    merged = instance._prepare_distill_trajectory_window(
        state, trajectory_path=current, attempt_dir=tmp_path)
    evidence = merged.read_text(encoding="utf-8")
    assert "train-prior" in evidence and "train-current" in evidence
    assert "validation-only" not in evidence


def test_small_run_config_enables_balanced_ood_validation():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "examples/sage_mas/sage_config.online60_6x10.yaml")
                            .read_text(encoding="utf-8"))["sage"]
    val = config["online"]["val_eval"]
    assert val == {"enabled": True, "every_segment": True, "pool_size": 12,
                   "dataset": "valid_unseen", "game_selection": "family_proportional"}
