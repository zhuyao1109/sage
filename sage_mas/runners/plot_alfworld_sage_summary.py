"""Plot ALFWorld SAGE-MAS summary figure (Flash evolution + OOD comparison)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs" / "sage_mas"
OUT = LOGS / "analysis" / "sage_alfworld_summary.png"

FAMILIES = [
    ("Pick", "pick_and_place"),
    ("Look", "look_at_obj_in_light"),
    ("Clean", "pick_clean_then_place_in_recep"),
    ("Heat", "pick_heat_then_place_in_recep"),
    ("Cool", "pick_cool_then_place_in_recep"),
    ("Pick2", "pick_two_obj_and_place"),
]

PALETTE = {
    "flash": "#2563eb",
    "flash_light": "#93c5fd",
    "mini": "#ea580c",
    "mini_light": "#fdba74",
    "val": "#059669",
    "test": "#7c3aed",
    "train": "#64748b",
    "grid": "#e2e8f0",
    "text": "#0f172a",
    "muted": "#64748b",
}


def load_sr(run_dir: Path) -> tuple[float, int, int]:
    summary_path = run_dir / "online_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    wins = int(summary["wins"])
    n = int(summary["num_games"])
    return wins / n, wins, n


def maybe_load_sr(run_dir: Path) -> tuple[float, int, int] | None:
    try:
        return load_sr(run_dir)
    except FileNotFoundError:
        return None


def plot_eval_point(
    ax,
    seg: int,
    path: Path,
    kind: str,
    *,
    label_prefix: str = "",
    y_offset: int = 10,
) -> None:
    loaded = maybe_load_sr(path)
    if loaded is None:
        return
    sr, wins, n = loaded
    marker = "D" if kind == "test" else "o"
    size = 150 if kind == "test" else 120
    ax.scatter(
        seg,
        100 * sr,
        s=size,
        color=PALETTE[kind],
        marker=marker,
        edgecolors="white",
        linewidths=1.5,
        zorder=6 if kind == "test" else 5,
    )
    prefix = "OOD " if kind == "test" else label_prefix
    ax.annotate(
        f"{prefix}{100 * sr:.1f}%\n({wins}/{n})",
        (seg, 100 * sr),
        textcoords="offset points",
        xytext=(0, y_offset),
        ha="center",
        fontsize=8.5,
        color=PALETTE[kind],
        fontweight="bold",
    )


def family_rates(run_dir: Path) -> list[float]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    traj = run_dir / "online_trajectories.jsonl"
    with traj.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = row.get("task_family", "?")
            counts[key][1] += 1
            counts[key][0] += int(bool(row.get("won")))
    return [
        100.0 * counts[key][0] / counts[key][1] if counts[key][1] else 0.0
        for _, key in FAMILIES
    ]


def train_curve_cumulative(run_dir: Path) -> tuple[list[int], list[float]]:
    """Cumulative train SR from learning_curve_checkpoints (historical plot)."""
    state = json.loads((run_dir / "online_state.json").read_text())
    checkpoints = state.get("learning_curve_checkpoints") or []
    xs = [int(c["segment"]) for c in checkpoints]
    ys = [100.0 * float(c["train_success_rate"]) for c in checkpoints]
    return xs, ys


def train_curve_per_segment(run_dir: Path) -> tuple[list[int], list[float], list[tuple[int, int]]]:
    """Per-segment train SR from history.segment_metrics (wins/n that segment)."""
    state = json.loads((run_dir / "online_state.json").read_text())
    xs: list[int] = []
    ys: list[float] = []
    counts: list[tuple[int, int]] = []
    for item in state.get("history") or []:
        if not isinstance(item, dict):
            continue
        seg = int(item.get("round") or 0)
        metrics = item.get("segment_metrics") or {}
        if not isinstance(metrics, dict):
            continue
        wins = metrics.get("wins")
        n = metrics.get("num_games") or metrics.get("n")
        sr = metrics.get("success_rate")
        if wins is None or n is None:
            if sr is None or not n:
                continue
            wins = int(round(float(sr) * int(n)))
        wins_i = int(wins)
        n_i = int(n)
        if n_i <= 0:
            continue
        xs.append(seg)
        ys.append(100.0 * wins_i / n_i)
        counts.append((wins_i, n_i))
    return xs, ys, counts


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "figure.facecolor": "#fafafa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": PALETTE["grid"],
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
        }
    )

    fig = plt.figure(figsize=(14, 8.5), dpi=160)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.15, 1.0], hspace=0.34, wspace=0.22)
    ax_evo = fig.add_subplot(gs[0, :])
    ax_ood = fig.add_subplot(gs[1, 0])
    ax_fam = fig.add_subplot(gs[1, 1])

    # --- Panel A: Flash 20-seg evolution ---
    # Per-segment train SR (primary). Keep a faint cumulative curve for reference.
    xs1, ys1, _ = train_curve_per_segment(LOGS / "online600_noseed_org")
    xs2, ys2, _ = train_curve_per_segment(LOGS / "continue_from_noseed95")
    xs2 = [x + 10 for x in xs2]
    xs_train = xs1 + xs2
    ys_train = ys1 + ys2

    xs1c, ys1c = train_curve_cumulative(LOGS / "online600_noseed_org")
    xs2c, ys2c = train_curve_cumulative(LOGS / "continue_from_noseed95")
    xs2c = [x + 10 for x in xs2c]
    ax_evo.plot(
        xs1c + xs2c,
        ys1c + ys2c,
        color="#cbd5e1",
        linewidth=1.4,
        linestyle="--",
        marker=None,
        label="Train cumulative (ref)",
        zorder=2,
        alpha=0.9,
    )
    ax_evo.plot(
        xs_train,
        ys_train,
        color=PALETTE["train"],
        linewidth=1.2,
        linestyle="-",
        alpha=0.35,
        zorder=3,
    )
    ax_evo.scatter(
        xs_train,
        ys_train,
        s=42,
        color=PALETTE["train"],
        marker="o",
        edgecolors="white",
        linewidths=0.8,
        label="Train per-seg",
        zorder=4,
    )

    val_points = [
        (0, LOGS / "executor_only_val140", "val", -32),
        (5, LOGS / "flash_seg05_val140", "val", 10),
        (10, LOGS / "online600_noseed_org_val140", "val", 10),
        (15, LOGS / "flash_seg15_val140", "val", 10),
        (20, LOGS / "continue_from_noseed95_val140", "val", 10),
    ]
    for seg, path, kind, y_offset in val_points:
        plot_eval_point(ax_evo, seg, path, kind, y_offset=y_offset)

    test_points = [
        (0, LOGS / "executor_only_test134", "test", 12),
        (10, LOGS / "online600_noseed_org_test134", "test", -28),
        (20, LOGS / "continue_from_noseed95_test134", "test", -28),
    ]
    for seg, path, kind, y_offset in test_points:
        plot_eval_point(ax_evo, seg, path, kind, y_offset=y_offset)

    ax_evo.axvline(10, color="#cbd5e1", linestyle=":", linewidth=1.2, zorder=1)
    ax_evo.text(10.15, 48, "continue →", color=PALETTE["muted"], fontsize=9, va="bottom")
    ax_evo.set_xlim(-0.8, 20.5)
    ax_evo.set_ylim(45, 102)
    ax_evo.set_xticks([0, 5, 10, 15, 20])
    ax_evo.set_xlabel("Training segment (60 games / seg)")
    ax_evo.set_ylabel("Success rate (%)")
    ax_evo.set_title(
        "Flash SAGE online evolution: per-seg train + val140 (every 5 seg) + OOD test134"
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=PALETTE["train"],
            markerfacecolor=PALETTE["train"],
            markersize=7,
            linewidth=0,
            label="Train per-seg (60 games)",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#cbd5e1",
            linestyle="--",
            linewidth=1.4,
            label="Train cumulative (old)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE["val"],
            markersize=8,
            label="Val140 (valid_seen)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=PALETTE["test"],
            markersize=8,
            label="Test134 (valid_unseen)",
        ),
    ]
    ax_evo.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor=PALETTE["grid"],
    )

    # --- Panel B: OOD test134 comparison ---
    ood_rows = [
        ("Flash\nsolo", LOGS / "executor_only_test134", PALETTE["flash_light"]),
        ("Flash\nSAGE@10", LOGS / "online600_noseed_org_test134", PALETTE["flash"]),
        ("Flash\nSAGE@20", LOGS / "continue_from_noseed95_test134", "#1d4ed8"),
        ("Pro skills\n+ MAS", LOGS / "pro_coldstart_agents_test134_r2", "#6366f1"),
        ("mini\nsolo", LOGS / "executor_only_test134_gpt-4o-mini", PALETTE["mini_light"]),
        ("mini\nSAGE@10", LOGS / "online600_noseed_gpt-4o-mini_test134", PALETTE["mini"]),
    ]
    labels = [r[0] for r in ood_rows]
    srs = [100 * load_sr(r[1])[0] for r in ood_rows]
    colors = [r[2] for r in ood_rows]
    x = np.arange(len(labels))
    bars = ax_ood.bar(x, srs, color=colors, width=0.68, edgecolor="white", linewidth=1.2)
    for bar, sr in zip(bars, srs):
        ax_ood.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{sr:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["text"],
        )
    ax_ood.set_xticks(x)
    ax_ood.set_xticklabels(labels)
    ax_ood.set_ylim(0, 100)
    ax_ood.set_ylabel("Success rate (%)")
    ax_ood.set_title("OOD test134: executor-only vs evolved SAGE")
    ax_ood.axhline(51.5, color=PALETTE["mini"], linestyle=":", linewidth=1, alpha=0.7)
    ax_ood.text(5.35, 52.5, "mini SAGE ceiling ≈51%", color=PALETTE["mini"], fontsize=8)

    # --- Panel C: family breakdown ---
    fam_runs = [
        ("Flash solo", LOGS / "executor_only_test134", PALETTE["flash_light"]),
        ("Flash SAGE@20", LOGS / "continue_from_noseed95_test134", PALETTE["flash"]),
        ("mini solo", LOGS / "executor_only_test134_gpt-4o-mini", PALETTE["mini_light"]),
        ("mini SAGE@10", LOGS / "online600_noseed_gpt-4o-mini_test134", PALETTE["mini"]),
    ]
    fam_labels = [f[0] for f in FAMILIES]
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(fam_runs))
    for offset, (name, path, color) in zip(offsets, fam_runs):
        rates = family_rates(path)
        xs = np.arange(len(fam_labels)) + offset
        ax_fam.bar(xs, rates, width=width, label=name, color=color, edgecolor="white", linewidth=0.8)
    ax_fam.set_xticks(np.arange(len(fam_labels)))
    ax_fam.set_xticklabels(fam_labels)
    ax_fam.set_ylim(0, 105)
    ax_fam.set_ylabel("Success rate (%)")
    ax_fam.set_title("Per-family OOD test134")
    ax_fam.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor=PALETTE["grid"])

    fig.suptitle(
        "SAGE-MAS on ALFWorld: online evolution lifts weak models, Flash continues to improve OOD",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["text"],
        y=0.98,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
