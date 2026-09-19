"""Plot online SAGE-MAS learning curves from online_state.json.

Produces PNGs plus an optional auto-refresh HTML live dashboard with three
curves aligned to training progress:
  - train: cumulative success on the online collection split
  - val: held-out valid_seen evaluations (periodic pool or shadow)
  - test: held-out valid_unseen evaluations (periodic pool or post-ADD_AGENT)

By default, periodic val/test runs start from the first train task; older
online1000-style runs still anchor at the first ADD_AGENT commit. Pass
``--anchor first_skill`` / ``--anchor add_agent`` to force a start point.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SPLIT_COLORS = {
    "train": "#1f4e79",
    "val": "#2a9d8f",
    "test": "#c45c26",
}
SPLIT_LABELS = {
    "train": "Train",
    "val": "Val (held-out)",
    "test": "Test (held-out)",
}


def _trial_won(trial: dict[str, Any]) -> bool:
    if "won" in trial:
        return bool(trial.get("won"))
    return bool(trial.get("success"))


def _trial_steps(trial: dict[str, Any]) -> int:
    for key in ("num_steps", "steps_taken", "env_steps"):
        if key in trial and trial[key] is not None:
            try:
                return max(0, int(trial[key]))
            except (TypeError, ValueError):
                continue
    steps = trial.get("steps")
    if isinstance(steps, list):
        return len(steps)
    return 0


def build_learning_curve_series(
    trials: list[dict[str, Any]],
    *,
    rolling_window: int = 50,
    x_key: str = "cumulative_tasks",
) -> list[dict[str, float | int]]:
    """Return per-episode cumulative success metrics."""
    rows: list[dict[str, float | int]] = []
    wins = 0
    cum_steps = 0
    recent: list[bool] = []
    for index, trial in enumerate(trials, start=1):
        won = _trial_won(trial)
        wins += int(won)
        cum_steps += _trial_steps(trial)
        recent.append(won)
        if rolling_window > 0 and len(recent) > rolling_window:
            recent.pop(0)
        rolling = sum(recent) / len(recent) if recent else 0.0
        x_value = (
            int(trial.get("train_progress_at_eval", index))
            if x_key == "train_progress"
            else index
        )
        rows.append(
            {
                "episode": index,
                "cumulative_tasks": index,
                "train_progress": x_value,
                "cumulative_env_steps": cum_steps,
                "won": int(won),
                "cumulative_wins": wins,
                "cumulative_success_rate": wins / index,
                "rolling_success_rate": rolling,
            }
        )
    return rows


def build_eval_checkpoint_series(
    trials: list[dict[str, Any]],
) -> list[dict[str, float | int]]:
    """Build val/test checkpoints keyed by train progress.

    Fixed periodic pools use per-checkpoint batch SR (snapshot quality).
    Mixed/historical shadow vals keep cumulative SR across batches.
    """
    if not trials:
        return []
    ordered = sorted(
        trials,
        key=lambda trial: (
            int(trial.get("train_progress_at_eval", 0)),
            int(trial.get("segment", 0)),
        ),
    )
    by_progress: dict[int, list[dict[str, Any]]] = {}
    for trial in ordered:
        progress = int(trial.get("train_progress_at_eval", 0))
        by_progress.setdefault(progress, []).append(trial)

    periodic = any(
        bool(trial.get("fixed_eval_pool"))
        or str(trial.get("phase", "")).startswith("periodic_")
        for trial in ordered
    )

    rows: list[dict[str, float | int]] = []
    cum_wins = 0
    cum_total = 0
    for progress in sorted(by_progress):
        batch = by_progress[progress]
        batch_wins = sum(int(_trial_won(trial)) for trial in batch)
        cum_wins += batch_wins
        cum_total += len(batch)
        batch_rate = batch_wins / len(batch)
        rows.append(
            {
                "train_progress": progress,
                "cumulative_tasks": progress,
                "batch_size": len(batch),
                "batch_success_rate": batch_rate,
                "cumulative_success_rate": (
                    batch_rate if periodic else cum_wins / cum_total
                ),
                "cumulative_wins": cum_wins,
            }
        )
    return rows


def state_uses_periodic_curve(state: dict[str, Any] | None) -> bool:
    """True when charts should start from task 0 (no ADD_AGENT anchor)."""
    if not isinstance(state, dict):
        return False
    cfg = state.get("config") or {}
    if bool(cfg.get("test_eval_every_segment")) or bool(
        cfg.get("val_eval_every_segment")
    ):
        return True
    if bool(cfg.get("freeze_organization")) or bool(cfg.get("executor_only")):
        return True
    for key in ("val_trials", "test_trials"):
        for trial in state.get(key) or []:
            if not isinstance(trial, dict):
                continue
            if trial.get("fixed_eval_pool") or str(
                trial.get("phase", "")
            ).startswith("periodic_"):
                return True
    return False


@dataclass(frozen=True)
class PlotAnchor:
    progress: int
    train_sr: float
    reason: str
    segment: int
    added_agents: tuple[str, ...] = ()
    skill_names: tuple[str, ...] = ()


def train_sr_at_progress(
    train_trials: list[dict[str, Any]],
    progress: int,
) -> float:
    if progress <= 0 or not train_trials:
        return 0.0
    subset = train_trials[:progress]
    wins = sum(int(_trial_won(trial)) for trial in subset)
    return wins / len(subset)


def _added_agents_from_record(
    state_path: Path,
    record: dict[str, Any],
) -> list[str]:
    prior = set(record.get("agent_names") or [])
    nxt = set(record.get("next_agent_names") or [])
    added = sorted(name for name in nxt if name not in prior)
    if added:
        return added
    edits_ref = record.get("organization_edits")
    if not edits_ref:
        return []
    edits_path = _resolve_artifact_path(state_path, str(edits_ref))
    if edits_path is None:
        return []
    try:
        edits = json.loads(edits_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    names: list[str] = []
    for edit in edits:
        if str(edit.get("edit_type", "")).lower() != "add_agent":
            continue
        new_agent = edit.get("new_agent")
        if isinstance(new_agent, dict) and new_agent.get("name"):
            names.append(str(new_agent["name"]))
    return names


def find_plot_anchor(
    state: dict[str, Any],
    state_path: Path | None = None,
) -> PlotAnchor | None:
    """Anchor charts only at ADD_AGENT commits."""
    train = [
        item
        for item in (state.get("all_trials") or [])
        if isinstance(item, dict)
    ]
    history = sorted(
        state.get("history") or [],
        key=lambda record: int(record.get("round", 0)),
    )
    for record in history:
        status = str(record.get("status", ""))
        if status not in {"accepted", "deferred_probation"}:
            continue
        added = _added_agents_from_record(state_path, record) if state_path else []
        if not added:
            prior = set(record.get("agent_names") or [])
            nxt = set(record.get("next_agent_names") or [])
            added = sorted(name for name in nxt if name not in prior)
        if not added:
            continue
        segment = int(record.get("round", 0))
        progress = _train_progress_after_round(state, segment)
        return PlotAnchor(
            progress=progress,
            train_sr=train_sr_at_progress(train, progress),
            reason="add_agent",
            segment=segment,
            added_agents=tuple(added),
        )

    return None


def find_first_skill_anchor(
    state: dict[str, Any],
    state_path: Path | None = None,
) -> PlotAnchor | None:
    """Anchor charts at the first history round that records distilled skills."""
    del state_path  # reserved for artifact-based skill discovery
    train = [
        item
        for item in (state.get("all_trials") or [])
        if isinstance(item, dict)
    ]
    history = sorted(
        state.get("history") or [],
        key=lambda record: int(record.get("round", 0)),
    )
    for record in history:
        verified = [
            str(name)
            for name in (record.get("verified_skill_names") or [])
            if name
        ]
        candidates = [
            str(name)
            for name in (record.get("candidate_skill_names") or [])
            if name
        ]
        bank_size = int(record.get("skill_bank_size") or 0)
        skill_names = verified or candidates
        if not skill_names and bank_size <= 0:
            continue
        segment = int(record.get("round", 0))
        progress = _train_progress_after_round(state, segment)
        if progress <= 0:
            continue
        return PlotAnchor(
            progress=progress,
            train_sr=train_sr_at_progress(train, progress),
            reason="first_skill",
            segment=segment,
            skill_names=tuple(skill_names),
        )
    return None


def resolve_plot_anchor(
    state: dict[str, Any] | None,
    state_path: Path | None = None,
    *,
    anchor_mode: str = "auto",
) -> tuple[PlotAnchor | None, bool]:
    """Return ``(anchor, require_anchor)``.

    ``require_anchor`` means: if no matching event yet, emit empty series /
    waiting placeholder instead of drawing from task 0.
    """
    mode = str(anchor_mode or "auto").strip().lower()
    if mode in {"", "auto"}:
        if state_uses_periodic_curve(state):
            return None, False
        return (
            find_plot_anchor(state or {}, state_path) if state else None,
            True,
        )
    if mode in {"none", "start", "from_start"}:
        return None, False
    if mode in {"add_agent", "agent"}:
        return (
            find_plot_anchor(state or {}, state_path) if state else None,
            True,
        )
    if mode in {"first_skill", "skill", "skills"}:
        return (
            find_first_skill_anchor(state or {}, state_path) if state else None,
            True,
        )
    raise ValueError(
        f"Unknown anchor mode {anchor_mode!r}; "
        "expected auto|none|add_agent|first_skill"
    )


def _anchor_train_series(
    rows: list[dict[str, float | int]],
    anchor: PlotAnchor,
) -> list[dict[str, float | int]]:
    filtered = [
        row
        for row in rows
        if int(row.get("train_progress", 0)) >= anchor.progress
    ]
    anchor_env_steps = 0
    anchored: list[dict[str, float | int]] = []
    for row in filtered:
        abs_progress = int(row["train_progress"])
        if abs_progress == anchor.progress:
            anchor_env_steps = int(row.get("cumulative_env_steps", 0))
        new_row = dict(row)
        new_row["train_progress"] = abs_progress - anchor.progress
        new_row["cumulative_tasks"] = abs_progress - anchor.progress
        new_row["cumulative_env_steps"] = (
            int(row.get("cumulative_env_steps", 0)) - anchor_env_steps
        )
        anchored.append(new_row)

    baseline = {
        "train_progress": 0,
        "cumulative_tasks": 0,
        "cumulative_env_steps": 0,
        "cumulative_success_rate": anchor.train_sr,
        "rolling_success_rate": anchor.train_sr,
        "is_baseline": True,
    }
    if not anchored:
        return [baseline]
    if int(anchored[0]["train_progress"]) > 0:
        anchored.insert(0, baseline)
    else:
        anchored[0] = {**anchored[0], **baseline, "is_baseline": False}
    return anchored


def _anchor_eval_series(
    rows: list[dict[str, float | int]],
    anchor: PlotAnchor,
) -> list[dict[str, float | int]]:
    """Shift eval checkpoints relative to the plot anchor.

    Do **not** inject a synthetic point at x=0 using train SR: that creates a
    fake vertical drop when the first real val/test checkpoint also maps to
    x=0 (common for ``first_skill`` / ADD_AGENT anchors).
    """
    if not rows:
        return rows
    anchored: list[dict[str, float | int]] = []
    for row in rows:
        abs_progress = int(row.get("train_progress", 0))
        if abs_progress < anchor.progress:
            continue
        anchored.append(
            {
                **row,
                "train_progress": abs_progress - anchor.progress,
                "cumulative_tasks": abs_progress - anchor.progress,
            }
        )
    return anchored


def anchor_x_label(anchor: PlotAnchor | None) -> str:
    if anchor is None:
        return "Train tasks completed"
    if anchor.reason == "first_skill":
        skills = ", ".join(anchor.skill_names) or "skill"
        return f"Tasks since first skill ({skills})"
    agents = ", ".join(anchor.added_agents) or "new agent"
    return f"Tasks since ADD_AGENT ({agents})"


def anchor_steps_x_label(anchor: PlotAnchor | None) -> str:
    if anchor is None:
        return "Train cumulative LLM steps"
    if anchor.reason == "first_skill":
        return "LLM steps since first skill (train)"
    return "LLM steps since ADD_AGENT (train)"


def anchor_tasks_title(anchor: PlotAnchor | None) -> str:
    if anchor is None:
        return "Online success rate vs train tasks"
    if anchor.reason == "first_skill":
        return "Online success rate vs tasks since first skill"
    return "Online success rate vs tasks since ADD_AGENT"


def anchor_steps_title(anchor: PlotAnchor | None) -> str:
    if anchor is None:
        return "Online success rate vs LLM steps"
    if anchor.reason == "first_skill":
        return "Online success rate vs LLM steps since first skill"
    return "Online success rate vs LLM steps since ADD_AGENT"


def _train_progress_after_round(state: dict[str, Any], round_index: int) -> int:
    total = 0
    for record in state.get("history") or []:
        if int(record.get("round", 0)) <= round_index:
            total += int(record.get("trial_count", 0))
    if total > 0:
        return total
    segment_size = int((state.get("config") or {}).get("segment_size", 0) or 0)
    if segment_size > 0:
        return round_index * segment_size
    return len(state.get("all_trials") or [])


def _resolve_artifact_path(state_path: Path, artifact_ref: str) -> Path | None:
    raw = Path(str(artifact_ref))
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(state_path.parent / raw)
        candidates.append(Path.cwd() / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def reconstruct_val_trials_from_history(
    state_path: Path,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Backfill val trials from segment shadow_result.json for older runs."""
    existing = list(state.get("val_trials") or [])
    if existing:
        return existing

    reconstructed: list[dict[str, Any]] = []
    for record in state.get("history") or []:
        shadow_ref = record.get("shadow_result")
        if not shadow_ref:
            continue
        shadow_path = _resolve_artifact_path(state_path, str(shadow_ref))
        if shadow_path is None:
            continue
        payload = json.loads(shadow_path.read_text(encoding="utf-8"))
        round_index = int(record.get("round", 0))
        train_progress = _train_progress_after_round(state, round_index)
        saved_trials = (payload.get("new") or {}).get("trials")
        if isinstance(saved_trials, list) and saved_trials:
            for trial in saved_trials:
                if isinstance(trial, dict):
                    item = dict(trial)
                    item.setdefault("dataset_split", "val")
                    item.setdefault("segment", round_index)
                    item.setdefault("phase", "shadow_new")
                    item.setdefault("train_progress_at_eval", train_progress)
                    reconstructed.append(item)
            continue

        metrics = (payload.get("new") or {}).get("metrics") or {}
        held_out = payload.get("held_out_gamefiles") or []
        success_rate = float(metrics.get("success_rate", 0.0) or 0.0)
        wins = int(round(success_rate * len(held_out))) if held_out else 0
        for index, gamefile in enumerate(held_out):
            reconstructed.append(
                {
                    "task_id": gamefile,
                    "won": index < wins,
                    "dataset_split": "val",
                    "segment": round_index,
                    "phase": "shadow_new",
                    "train_progress_at_eval": train_progress,
                    "synthetic_from_metrics": True,
                }
            )
    return reconstructed


def load_split_trials(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"train": [item for item in payload if isinstance(item, dict)], "val": [], "test": []}
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported input JSON type: {type(payload)}")

    train = [
        item
        for item in (payload.get("all_trials") or payload.get("trials") or [])
        if isinstance(item, dict)
    ]
    val = [item for item in (payload.get("val_trials") or []) if isinstance(item, dict)]
    test = [item for item in (payload.get("test_trials") or []) if isinstance(item, dict)]
    if not val:
        val = reconstruct_val_trials_from_history(path, payload)
    return {"train": train, "val": val, "test": test}


def _load_trials(path: Path) -> list[dict[str, Any]]:
    return load_split_trials(path)["train"]


def _load_state_meta(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    splits = load_split_trials(path)
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    return {
        "completed_segments": payload.get("completed_segments"),
        "num_games": cfg.get("num_games"),
        "segment_size": cfg.get("segment_size"),
        "collection_dataset": cfg.get("collection_dataset"),
        "train_trials": len(train),
        "val_trials": len(val),
        "test_trials": len(test),
        "train_success_rate": (
            sum(_trial_won(t) for t in train) / len(train) if train else None
        ),
        "val_success_rate": (
            sum(_trial_won(t) for t in val) / len(val) if val else None
        ),
        "test_success_rate": (
            sum(_trial_won(t) for t in test) / len(test) if test else None
        ),
    }


def _load_state_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_waiting_placeholder(
    output_path: Path,
    *,
    title: str,
    x_label: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.5,
        0.5,
        "Waiting for first ADD_AGENT\ncurves start at that commit",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="#666666",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_multi_curve(
    series_by_split: dict[str, list[dict[str, float | int]]],
    *,
    x_key: str,
    x_label: str,
    title: str,
    output_path: Path,
    rolling_window: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    train_rows = series_by_split.get("train") or []
    if train_rows:
        xs = [row[x_key] for row in train_rows]
        ys = [row["cumulative_success_rate"] for row in train_rows]
        ax.plot(
            xs,
            ys,
            color=SPLIT_COLORS["train"],
            linewidth=2.0,
            label=SPLIT_LABELS["train"],
        )
        if rolling_window > 1:
            rolling = [row["rolling_success_rate"] for row in train_rows]
            ax.plot(
                xs,
                rolling,
                color=SPLIT_COLORS["train"],
                linewidth=1.2,
                alpha=0.45,
                linestyle="--",
                label=f"Train rolling (w={rolling_window})",
            )

    for split in ("val", "test"):
        rows = series_by_split.get(split) or []
        if not rows:
            continue
        ax.plot(
            [row[x_key] for row in rows],
            [row["cumulative_success_rate"] for row in rows],
            color=SPLIT_COLORS[split],
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label=SPLIT_LABELS[split],
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Success rate")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _polyline(
    rows: list[dict[str, float | int]],
    *,
    x_key: str,
    y_key: str,
    width: float,
    height: float,
    pad: float = 48.0,
    global_min_x: float | None = None,
    global_max_x: float | None = None,
) -> str:
    if not rows:
        return ""
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    min_x = global_min_x if global_min_x is not None else min(xs)
    max_x = global_max_x if global_max_x is not None else max(xs)
    if max_x <= min_x:
        max_x = min_x + 1.0
    inner_w = width - 2 * pad
    inner_h = height - 2 * pad

    def sx(value: float) -> float:
        return pad + (value - min_x) / (max_x - min_x) * inner_w

    def sy(value: float) -> float:
        return pad + (1.0 - value) * inner_h

    return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))


def _svg_chart(
    series_by_split: dict[str, list[dict[str, float | int]]],
    *,
    x_key: str,
    x_label: str,
    title: str,
    rolling_window: int,
    width: int = 860,
    height: int = 360,
) -> str:
    pad = 52.0
    all_rows = [
        row
        for rows in series_by_split.values()
        for row in rows
    ]
    if not all_rows:
        return "<p>Waiting for trials…</p>"

    min_x = min(float(row[x_key]) for row in all_rows)
    max_x = max(float(row[x_key]) for row in all_rows)
    y_ticks = "".join(
        (
            f'<line x1="{pad}" y1="{pad + (1 - y) * (height - 2 * pad):.1f}" '
            f'x2="{width - pad}" y2="{pad + (1 - y) * (height - 2 * pad):.1f}" '
            f'stroke="#e6e6e6"/>'
            f'<text x="{pad - 8}" y="{pad + (1 - y) * (height - 2 * pad) + 4:.1f}" '
            f'text-anchor="end" font-size="11" fill="#666">{y:.1f}</text>'
        )
        for y in (0.0, 0.25, 0.5, 0.75, 1.0)
    )

    polylines: list[str] = []
    train_rows = series_by_split.get("train") or []
    if train_rows:
        polylines.append(
            f'<polyline fill="none" stroke="{SPLIT_COLORS["train"]}" '
            f'stroke-width="2.2" points="{_polyline(train_rows, x_key=x_key, y_key="cumulative_success_rate", width=width, height=height, pad=pad, global_min_x=min_x, global_max_x=max_x)}"/>'
        )
        if rolling_window > 1:
            polylines.append(
                f'<polyline fill="none" stroke="{SPLIT_COLORS["train"]}" '
                f'stroke-width="1.6" stroke-dasharray="5 4" opacity="0.55" '
                f'points="{_polyline(train_rows, x_key=x_key, y_key="rolling_success_rate", width=width, height=height, pad=pad, global_min_x=min_x, global_max_x=max_x)}"/>'
            )

    for split in ("val", "test"):
        rows = series_by_split.get(split) or []
        if not rows:
            continue
        points = _polyline(
            rows,
            x_key=x_key,
            y_key="cumulative_success_rate",
            width=width,
            height=height,
            pad=pad,
            global_min_x=min_x,
            global_max_x=max_x,
        )
        polylines.append(
            f'<polyline fill="none" stroke="{SPLIT_COLORS[split]}" '
            f'stroke-width="2.0" points="{points}"/>'
        )
        for row in rows:
            x = float(row[x_key])
            y = float(row["cumulative_success_rate"])
            inner_w = width - 2 * pad
            inner_h = height - 2 * pad
            cx = pad + (x - min_x) / (max_x - min_x if max_x > min_x else 1.0) * inner_w
            cy = pad + (1.0 - y) * inner_h
            polylines.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="{SPLIT_COLORS[split]}"/>'
            )

    legend_x = pad
    legend_items = []
    for split in ("train", "val", "test"):
        rows = series_by_split.get(split) or []
        if split != "train" and not rows:
            continue
        legend_items.append(
            f'<line x1="{legend_x}" y1="22" x2="{legend_x + 18}" y2="22" '
            f'stroke="{SPLIT_COLORS[split]}" stroke-width="2.2"/>'
            f'<text x="{legend_x + 24}" y="26" font-size="12" fill="{SPLIT_COLORS[split]}">'
            f"{SPLIT_LABELS[split]}</text>"
        )
        legend_x += 190

    return f"""
<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{title}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>
  <text x="{pad}" y="18" font-size="15" font-weight="600" fill="#222">{title}</text>
  {''.join(legend_items)}
  {y_ticks}
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#888"/>
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#888"/>
  {''.join(polylines)}
  <text x="{(width) / 2:.0f}" y="{height - 12}" text-anchor="middle" font-size="12" fill="#444">{x_label}</text>
  <text x="16" y="{(height) / 2:.0f}" transform="rotate(-90 16,{(height) / 2:.0f})"
        text-anchor="middle" font-size="12" fill="#444">Success rate</text>
  <text x="{width - pad}" y="{height - pad + 16}" text-anchor="end" font-size="11" fill="#666">{max_x:.0f}</text>
</svg>
""".strip()


def build_split_series(
    splits: dict[str, list[dict[str, Any]]],
    *,
    rolling_window: int,
    state: dict[str, Any] | None = None,
    state_path: Path | None = None,
    require_add_agent: bool | None = None,
    anchor_mode: str = "auto",
) -> tuple[dict[str, list[dict[str, float | int]]], PlotAnchor | None]:
    train = splits.get("train") or []
    if require_add_agent is not None:
        # Backward-compatible override used by older tests/callers.
        if require_add_agent:
            anchor = find_plot_anchor(state or {}, state_path) if state else None
            require_anchor = True
        else:
            periodic = state_uses_periodic_curve(state)
            if periodic:
                anchor, require_anchor = None, False
            else:
                anchor = find_plot_anchor(state or {}, state_path) if state else None
                require_anchor = False
    else:
        anchor, require_anchor = resolve_plot_anchor(
            state,
            state_path,
            anchor_mode=anchor_mode,
        )
    if require_anchor and anchor is None:
        # Do not draw the Executor-only warm-up curve; wait for the event.
        return {"train": [], "val": [], "test": []}, None

    series = {
        "train": build_learning_curve_series(
            train,
            rolling_window=rolling_window,
            x_key="train_progress",
        ),
        "val": build_eval_checkpoint_series(splits.get("val") or []),
        "test": build_eval_checkpoint_series(splits.get("test") or []),
    }
    if anchor is not None:
        series["train"] = _anchor_train_series(series["train"], anchor)
        series["val"] = _anchor_eval_series(series["val"], anchor)
        series["test"] = _anchor_eval_series(series["test"], anchor)
    return series, anchor


def _waiting_for_add_agent_svg(
    *,
    title: str,
    width: int = 860,
    height: int = 360,
) -> str:
    return f"""
<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{title}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>
  <text x="52" y="28" font-size="15" font-weight="600" fill="#222">{title}</text>
  <text x="{width / 2:.0f}" y="{height / 2:.0f}" text-anchor="middle"
        font-size="16" fill="#666">
    Waiting for first ADD_AGENT — curves start at that commit
  </text>
  <text x="{width / 2:.0f}" y="{height / 2 + 28:.0f}" text-anchor="middle"
        font-size="13" fill="#888">
    x=0: ADD_AGENT · initial SR: train SR at that moment
  </text>
</svg>
""".strip()


def _env_steps_at_train_progress(
    train_rows: list[dict[str, float | int]],
    progress: int,
) -> int:
    """Map a train-task index onto the corresponding cumulative env-step count."""
    if not train_rows:
        return 0
    best = train_rows[0]
    for row in train_rows:
        # Prefer anchored/relative train_progress over absolute cumulative_tasks.
        row_progress = int(
            row.get(
                "train_progress",
                row.get("cumulative_tasks", 0),
            )
        )
        if row_progress <= progress:
            best = row
        if row_progress >= progress:
            break
    return int(best.get("cumulative_env_steps", 0))


def series_vs_env_steps(
    series_by_split: dict[str, list[dict[str, float | int]]],
) -> dict[str, list[dict[str, float | int]]]:
    """Re-express all curves on train cumulative env-steps as the shared x-axis.

    Val/test checkpoints are recorded against train *task* progress; without this
    remapping they get plotted as if those task counts were env steps, which
    crushes them into the left edge of a 0–10k step axis.
    """
    train_rows = series_by_split.get("train") or []
    remapped: dict[str, list[dict[str, float | int]]] = {
        "train": [
            {**row, "x": int(row.get("cumulative_env_steps", 0))}
            for row in train_rows
        ]
    }
    for split in ("val", "test"):
        rows = series_by_split.get(split) or []
        remapped[split] = [
            {
                **row,
                "x": _env_steps_at_train_progress(
                    train_rows,
                    int(row.get("train_progress", row.get("cumulative_tasks", 0))),
                ),
            }
            for row in rows
        ]
    return remapped


def series_vs_evolution_batches(
    series_by_split: dict[str, list[dict[str, float | int]]],
    *,
    batch_size: int,
) -> dict[str, list[dict[str, float | int]]]:
    """Re-express curves on evolution steps where one train batch = one step.

    ``batch_size`` is the number of train tasks per segment/batch. Train is
    downsampled to batch boundaries so the x-axis is discrete evolution count.
    """
    batch = max(1, int(batch_size))

    def _to_step(progress: int) -> int:
        progress = max(0, int(progress))
        return (progress + batch - 1) // batch if progress > 0 else 0

    remapped: dict[str, list[dict[str, float | int]]] = {"train": [], "val": [], "test": []}
    train_rows = series_by_split.get("train") or []
    seen_steps: set[int] = set()
    for row in train_rows:
        progress = int(row.get("train_progress", row.get("cumulative_tasks", 0)))
        # Keep baseline / exact batch ends only (1 point per evolution step).
        if progress > 0 and progress % batch != 0:
            continue
        step = progress // batch
        if step in seen_steps:
            continue
        seen_steps.add(step)
        remapped["train"].append({**row, "x": step})
    if train_rows and remapped["train"]:
        last = train_rows[-1]
        last_progress = int(last.get("train_progress", last.get("cumulative_tasks", 0)))
        last_step = last_progress // batch
        if last_step not in seen_steps:
            remapped["train"].append({**last, "x": last_step})

    for split in ("val", "test"):
        rows = series_by_split.get(split) or []
        remapped[split] = [
            {
                **row,
                "x": _to_step(
                    int(row.get("train_progress", row.get("cumulative_tasks", 0)))
                ),
            }
            for row in rows
        ]
    return remapped


def _batch_size_from_meta_or_state(
    meta: dict[str, Any] | None,
    state: dict[str, Any] | None,
) -> int:
    if isinstance(meta, dict):
        for key in ("segment_size", "batch_size"):
            try:
                value = int(meta.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return value
    if isinstance(state, dict):
        cfg = state.get("config") or {}
        try:
            value = int(cfg.get("segment_size") or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 10


def write_live_html(
    series_by_split: dict[str, list[dict[str, float | int]]],
    output_path: Path,
    *,
    rolling_window: int,
    refresh_seconds: int,
    source: str,
    meta: dict[str, Any] | None = None,
) -> None:
    meta = meta or {}
    anchor = meta.get("plot_anchor")
    x_label = anchor_x_label(anchor if isinstance(anchor, PlotAnchor) else None)
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_rows = series_by_split.get("train") or []
    val_rows = series_by_split.get("val") or []
    test_rows = series_by_split.get("test") or []
    pre_anchor_train = int(meta.get("pre_anchor_train_trials") or 0)
    pre_anchor_sr = meta.get("pre_anchor_train_success_rate")
    train_sr = float(train_rows[-1]["cumulative_success_rate"]) if train_rows else float("nan")
    val_sr = float(val_rows[-1]["cumulative_success_rate"]) if val_rows else float("nan")
    test_sr = float(test_rows[-1]["cumulative_success_rate"]) if test_rows else float("nan")
    if isinstance(anchor, PlotAnchor):
        tasks_svg = _svg_chart(
            series_by_split,
            x_key="train_progress",
            x_label=x_label,
            title=anchor_tasks_title(anchor),
            rolling_window=rolling_window,
        )
        steps_series = series_vs_env_steps(series_by_split)
        steps_svg = _svg_chart(
            steps_series,
            x_key="x",
            x_label=anchor_steps_x_label(anchor),
            title=anchor_steps_title(anchor),
            rolling_window=rolling_window,
        )
    else:
        tasks_svg = _waiting_for_add_agent_svg(
            title=anchor_tasks_title(None),
        )
        steps_svg = _waiting_for_add_agent_svg(
            title=anchor_steps_title(None),
        )
    completed = meta.get("completed_segments")
    num_games = meta.get("num_games")
    val_sr_text = f"{val_sr:.1%}" if val_rows else "—"
    test_sr_text = f"{test_sr:.1%}" if test_rows else "—"
    train_sr_text = f"{train_sr:.1%}" if train_rows else (
        f"{float(pre_anchor_sr):.1%}" if pre_anchor_sr is not None else "—"
    )
    train_count_text = (
        str(len(train_rows))
        if train_rows
        else str(pre_anchor_train or 0)
    )
    anchor_note = ""
    if isinstance(anchor, PlotAnchor):
        if anchor.reason == "first_skill":
            skills = ", ".join(anchor.skill_names) or "skill"
            anchor_note = (
                f"<br/>Curves start at first skill ({skills}) · "
                f"segment {anchor.segment} · "
                f"{anchor.progress} train tasks · initial SR {anchor.train_sr:.1%}"
            )
        else:
            agents = ", ".join(anchor.added_agents) or "new agent"
            anchor_note = (
                f"<br/>Curves start at ADD_AGENT ({agents}) · "
                f"segment {anchor.segment} · "
                f"{anchor.progress} train tasks · initial SR {anchor.train_sr:.1%}"
            )
    else:
        anchor_note = (
            "<br/>No plot anchor yet — learning curves are held until the "
            "requested start event."
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta http-equiv="refresh" content="{max(1, int(refresh_seconds))}"/>
  <title>SAGE-MAS live learning curve</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; color: #222; background: #fafafa; }}
    h1 {{ font-size: 20px; margin: 0 0 8px; }}
    .meta {{ color: #555; font-size: 13px; margin-bottom: 16px; }}
    .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
    .stat {{ background: #fff; border: 1px solid #e5e5e5; padding: 12px 16px; min-width: 120px; }}
    .stat .label {{ font-size: 12px; color: #666; }}
    .stat .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
    .chart {{ background: #fff; border: 1px solid #e5e5e5; padding: 8px 8px 16px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <h1>SAGE-MAS learning curves</h1>
  <div class="meta">
    Source: {source}<br/>
    Updated: {updated} · auto-refresh every {refresh_seconds}s{anchor_note}
  </div>
  <div class="stats">
    <div class="stat"><div class="label">Train tasks (all)</div><div class="value">{train_count_text}{f' / {num_games}' if num_games else ''}</div></div>
    <div class="stat"><div class="label">Train SR (all)</div><div class="value">{train_sr_text}</div></div>
    <div class="stat"><div class="label">Val SR</div><div class="value">{val_sr_text}</div></div>
    <div class="stat"><div class="label">Test SR</div><div class="value">{test_sr_text}</div></div>
    <div class="stat"><div class="label">Val points</div><div class="value">{len(val_rows)}</div></div>
    <div class="stat"><div class="label">Test points</div><div class="value">{len(test_rows)}</div></div>
    <div class="stat"><div class="label">Segments</div><div class="value">{completed if completed is not None else '—'}</div></div>
  </div>
  <div class="chart">{tasks_svg}</div>
  <div class="chart">{steps_svg}</div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def plot_learning_curves(
    trials: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    output_dir: Path,
    *,
    rolling_window: int = 50,
    prefix: str = "learning_curve",
    live_html: bool = True,
    refresh_seconds: int = 15,
    source: str = "",
    meta: dict[str, Any] | None = None,
    anchor_mode: str = "auto",
) -> dict[str, str]:
    if isinstance(trials, dict):
        splits = trials
    else:
        splits = {"train": trials, "val": [], "test": []}
    state_path = Path(source) if source.endswith(".json") else None
    state = _load_state_payload(state_path) if state_path else None
    series_by_split, anchor = build_split_series(
        splits,
        rolling_window=rolling_window,
        state=state,
        state_path=state_path,
        anchor_mode=anchor_mode,
    )
    if meta is None:
        meta = {}
    if state_path is not None:
        meta = {**_load_state_meta(state_path), **meta}
    all_train = splits.get("train") or []
    meta["pre_anchor_train_trials"] = len(all_train)
    meta["pre_anchor_train_success_rate"] = (
        sum(_trial_won(t) for t in all_train) / len(all_train) if all_train else None
    )
    meta["plot_anchor"] = anchor
    meta["anchor_mode"] = anchor_mode
    x_label = anchor_x_label(anchor)
    train_rows = series_by_split.get("train") or []
    has_series = bool(
        train_rows
        or series_by_split.get("val")
        or series_by_split.get("test")
    )
    tasks_title = anchor_tasks_title(anchor)
    steps_title = anchor_steps_title(anchor)
    steps_xlabel = anchor_steps_x_label(anchor)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    val_csv = output_dir / f"{prefix}_val.csv"
    test_csv = output_dir / f"{prefix}_test.csv"
    tasks_png = output_dir / f"{prefix}_vs_tasks.png"
    steps_png = output_dir / f"{prefix}_vs_env_steps.png"
    llm_steps_png = output_dir / f"{prefix}_vs_llm_steps.png"
    evolution_png = output_dir / f"{prefix}_vs_evolution.png"
    live_path = output_dir / f"{prefix}_live.html"
    batch_size = _batch_size_from_meta_or_state(meta, state)
    _write_csv(train_rows, csv_path)
    _write_csv(series_by_split.get("val") or [], val_csv)
    _write_csv(series_by_split.get("test") or [], test_csv)
    if has_series:
        _plot_multi_curve(
            series_by_split,
            x_key="train_progress",
            x_label=x_label,
            title=tasks_title,
            output_path=tasks_png,
            rolling_window=rolling_window,
        )
        steps_series = series_vs_env_steps(series_by_split)
        _plot_multi_curve(
            steps_series,
            x_key="x",
            x_label=steps_xlabel,
            title=steps_title,
            output_path=steps_png,
            rolling_window=rolling_window,
        )
        _plot_multi_curve(
            steps_series,
            x_key="x",
            x_label=steps_xlabel,
            title=steps_title,
            output_path=llm_steps_png,
            rolling_window=rolling_window,
        )
        evolution_series = series_vs_evolution_batches(
            series_by_split,
            batch_size=batch_size,
        )
        evo_xlabel = f"Evolution step (1 batch = {batch_size} tasks)"
        evo_title = (
            "Learning curve"
            if anchor is None
            else (
                "Learning curve from first skill"
                if getattr(anchor, "reason", "") == "first_skill"
                else "Learning curve from ADD_AGENT"
            )
        )
        _plot_multi_curve(
            evolution_series,
            x_key="x",
            x_label=evo_xlabel,
            title=evo_title,
            output_path=evolution_png,
            rolling_window=0,
        )
    else:
        wait_tasks = (
            "Online success rate vs tasks since first skill"
            if str(anchor_mode).lower() in {"first_skill", "skill", "skills"}
            else "Online success rate vs tasks since ADD_AGENT"
        )
        wait_steps = (
            "Online success rate vs LLM steps since first skill"
            if str(anchor_mode).lower() in {"first_skill", "skill", "skills"}
            else "Online success rate vs LLM steps since ADD_AGENT"
        )
        _plot_waiting_placeholder(
            tasks_png,
            title=wait_tasks,
            x_label=x_label if anchor is None else anchor_x_label(anchor),
        )
        _plot_waiting_placeholder(
            steps_png,
            title=wait_steps,
            x_label=steps_xlabel,
        )
        _plot_waiting_placeholder(
            llm_steps_png,
            title=wait_steps,
            x_label=steps_xlabel,
        )
        _plot_waiting_placeholder(
            evolution_png,
            title="Online success rate vs evolution steps",
            x_label=f"Evolution step (1 batch = {batch_size} tasks)",
        )
    if live_html:
        write_live_html(
            series_by_split,
            live_path,
            rolling_window=rolling_window,
            refresh_seconds=refresh_seconds,
            source=source or str(output_dir),
            meta=meta,
        )
    return {
        "csv": str(csv_path),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "vs_tasks": str(tasks_png),
        "vs_env_steps": str(steps_png),
        "vs_llm_steps": str(llm_steps_png),
        "vs_evolution": str(evolution_png),
        "live_html": str(live_path) if live_html else "",
        "batch_size": str(batch_size),
        "anchor_mode": str(anchor_mode),
        "anchor_progress": str(anchor.progress if anchor else ""),
        "anchor_reason": str(anchor.reason if anchor else ""),
        "num_train_trials": str(len(train_rows)),
        "num_val_points": str(len(series_by_split.get("val") or [])),
        "num_test_points": str(len(series_by_split.get("test") or [])),
        "final_train_success_rate": (
            f"{train_rows[-1]['cumulative_success_rate']:.4f}" if train_rows else "nan"
        ),
        "final_val_success_rate": (
            f"{(series_by_split.get('val') or [])[-1]['cumulative_success_rate']:.4f}"
            if series_by_split.get("val")
            else "nan"
        ),
        "final_test_success_rate": (
            f"{(series_by_split.get('test') or [])[-1]['cumulative_success_rate']:.4f}"
            if series_by_split.get("test")
            else "nan"
        ),
    }


def watch_learning_curves(
    input_path: Path,
    output_dir: Path,
    *,
    rolling_window: int,
    prefix: str,
    interval_seconds: float,
    anchor_mode: str = "auto",
) -> None:
    print(
        json.dumps(
            {
                "watching": str(input_path),
                "output_dir": str(output_dir),
                "interval_seconds": interval_seconds,
                "anchor_mode": anchor_mode,
                "live_html": str(output_dir / f"{prefix}_live.html"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    last_signature: tuple[int, int, int, int] | None = None
    while True:
        try:
            if not input_path.exists():
                print(f"[wait] missing {input_path}", flush=True)
            else:
                splits = load_split_trials(input_path)
                signature = (
                    len(splits["train"]),
                    len(splits["val"]),
                    len(splits["test"]),
                    int(input_path.stat().st_mtime_ns),
                )
                if signature != last_signature:
                    summary = plot_learning_curves(
                        splits,
                        output_dir,
                        rolling_window=rolling_window,
                        prefix=prefix,
                        live_html=True,
                        refresh_seconds=max(1, int(interval_seconds)),
                        source=str(input_path),
                        meta=_load_state_meta(input_path),
                        anchor_mode=anchor_mode,
                    )
                    last_signature = signature
                    print(
                        json.dumps(
                            {
                                "updated": datetime.now().isoformat(timespec="seconds"),
                                **summary,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001 - keep watcher alive
            print(f"[error] {type(exc).__name__}: {exc}", flush=True)
        time.sleep(max(1.0, float(interval_seconds)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to online_state.json (or a JSON list of trials).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for PNG/CSV/HTML outputs (default: same as input).",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=0,
        help="If >1, also draw a train rolling-SR curve (default: 0 = off).",
    )
    parser.add_argument("--prefix", default="learning_curve")
    parser.add_argument(
        "--anchor",
        default="auto",
        choices=["auto", "none", "add_agent", "first_skill"],
        help=(
            "Curve start point: auto (periodic=from start, else ADD_AGENT); "
            "none=from first train task; add_agent=first accepted specialist; "
            "first_skill=first distilled skill in history."
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll input and refresh PNG/HTML whenever trials change.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Watch poll interval in seconds (default: 15).",
    )
    parser.add_argument(
        "--no-live-html",
        action="store_true",
        help="Skip writing the auto-refresh HTML dashboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent
    )
    rolling = max(0, int(args.rolling_window))
    if args.watch:
        watch_learning_curves(
            input_path,
            output_dir,
            rolling_window=rolling,
            prefix=str(args.prefix),
            interval_seconds=float(args.interval),
            anchor_mode=str(args.anchor),
        )
        return

    splits = load_split_trials(input_path)
    summary = plot_learning_curves(
        splits,
        output_dir,
        rolling_window=rolling,
        prefix=str(args.prefix),
        live_html=not bool(args.no_live_html),
        refresh_seconds=max(1, int(args.interval)),
        source=str(input_path),
        meta=_load_state_meta(input_path),
        anchor_mode=str(args.anchor),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
