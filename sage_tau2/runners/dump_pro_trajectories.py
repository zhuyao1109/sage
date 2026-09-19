"""Export τ² ``results.json`` into PRO trajectories (v5).

Each assistant turn becomes one step with:
  observation_before / action / observation
  (+ reconstructed ``prompt`` / ``history_summary`` window by default)

System prompt policy:
  - Full ``system_prompt`` is stored **once** on the trajectory (and in
    ``runtime_context.json``), not copied into every step body.
  - Each step carries ``system_prompt_sha1`` + window-only ``prompt``
    (history + current obs). Training input = system(once) + step.prompt.

Default ``obs_mode=full`` keeps near-raw user/tool text (soft-capped).
Pass ``obs_mode=lean`` for GiGPO-style semantic compression.
runtime_inject stays compact (hashes + org/skills blocks).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sage_tau2.pro_steps import (
    compact_runtime_inject,
    find_dispatch_journal,
    format_rich_trajectory_text,
    load_runtime_inject_index,
    messages_to_rich_pro_steps,
    runtime_inject_from_sources,
    task_text_from_sources,
)

PRO_FORMAT = "sage_tau2_pro_v5"


def _stamp_system_refs(
    steps: list[dict[str, Any]],
    *,
    system_prompt_sha1: str | None,
    history_window: int,
) -> None:
    """Annotate steps with system hash; never embed full system into prompt."""
    sha = str(system_prompt_sha1 or "").strip() or None
    for step in steps:
        if sha:
            step["system_prompt_sha1"] = sha
        # Guard against accidental system leakage into the per-step window.
        step["system_in_prompt"] = False
        step["history_window"] = int(history_window)


def simulation_to_trajectory(
    sim: dict[str, Any],
    *,
    domain: str | None = None,
    source: str | None = None,
    dispatch_row: dict[str, Any] | None = None,
    results_info: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
    history_window: int = 10,
    obs_mode: str = "full",
    store_window_prompt: bool = True,
    include_system_prompt: bool = True,
) -> dict[str, Any]:
    reward_info = sim.get("reward_info") or {}
    breakdown = reward_info.get("reward_breakdown") or {}
    db_check = reward_info.get("db_check") or {}
    messages = [m for m in (sim.get("messages") or []) if isinstance(m, dict)]
    task_text = task_text_from_sources(
        sim, task=task, dispatch_row=dispatch_row
    )
    from sage_tau2.success_mode import classify_success_mode

    mode_info = classify_success_mode(sim)
    mode = str(obs_mode or "full").strip().lower() or "full"
    steps = messages_to_rich_pro_steps(
        messages,
        history_window=history_window,
        reward_info=reward_info if isinstance(reward_info, dict) else None,
        allow_synthesize_think=False,
        store_window_prompt=bool(store_window_prompt),
        obs_mode=mode,
    )
    inject_full = runtime_inject_from_sources(
        sim, dispatch_row=dispatch_row, results_info=results_info
    )
    inject = compact_runtime_inject(inject_full)
    primary = inject.get("primary_agent") or "Executor"
    system_prompt = inject_full.get("system_prompt")
    system_sha = inject.get("system_prompt_sha1")
    _stamp_system_refs(
        steps, system_prompt_sha1=system_sha, history_window=history_window
    )
    if mode == "lean":
        note = (
            "PRO v5 lean: semantic obs compression; "
            "system_prompt once on trajectory; each step has window prompt "
            f"(hist≤{history_window} + current obs) + system_prompt_sha1; "
            "no per-step system 复读; model <think> only in agent_messages reply."
        )
    else:
        note = (
            "PRO v5 full: near-raw user/tool observations (soft-capped); "
            "system_prompt once on trajectory; each step has window prompt "
            f"(hist≤{history_window} + current obs) + system_prompt_sha1; "
            "no per-step system 复读; model <think> only in agent_messages reply."
        )
    out: dict[str, Any] = {
        "framework": "sage_tau2",
        "format": PRO_FORMAT,
        "obs_mode": mode,
        "history_window": int(history_window),
        "domain": domain or inject.get("domain"),
        "source": source,
        "task_id": sim.get("task_id"),
        "task": task_text,
        "trial": sim.get("trial"),
        "won": float(reward_info.get("reward") or 0.0) >= 1.0,
        "reward": reward_info.get("reward"),
        "db_match": (reward_info.get("db_check") or {}).get("db_match"),
        "termination_reason": sim.get("termination_reason"),
        "duration": sim.get("duration"),
        "num_steps": len(steps),
        "assigned_primary_agent": primary,
        "dispatch_layer": inject.get("dispatch_layer"),
        "runtime_inject": inject,
        "system_prompt_sha1": system_sha,
        "system_prompt_chars": len(system_prompt or ""),
        "steps": steps,
        "note": note,
        "distill_meta": {
            "success_mode": mode_info.get("success_mode"),
            "action_checks": reward_info.get("action_checks"),
            "communicate_checks": reward_info.get("communicate_checks"),
            "write_tools": mode_info.get("write_tools"),
            "transfer_tools": mode_info.get("transfer_tools"),
            "gold_actions": mode_info.get("gold_actions"),
            "tools": mode_info.get("tools"),
            "db_reward": (
                None
                if breakdown.get("DB") is None and db_check.get("db_reward") is None
                else float(
                    breakdown.get("DB")
                    if breakdown.get("DB") is not None
                    else db_check.get("db_reward")
                )
            ),
            "communicate_reward": (
                None
                if breakdown.get("COMMUNICATE") is None
                else float(breakdown.get("COMMUNICATE"))
            ),
        },
        "_runtime_context": {
            "system_prompt": system_prompt,
            "domain_policy": inject_full.get("domain_policy"),
            "system_prompt_sha1": system_sha,
            "domain_policy_sha1": inject.get("domain_policy_sha1"),
        },
    }
    if include_system_prompt and isinstance(system_prompt, str) and system_prompt.strip():
        # Self-contained trajectory: one full system copy (not per-step).
        out["system_prompt"] = system_prompt
    return out


def format_trajectory_text(traj: dict[str, Any]) -> str:
    return format_rich_trajectory_text(traj)


def export_pro_trajectories(
    *,
    results_payload: dict[str, Any] | None = None,
    results_path: str | Path | None = None,
    output_dir: str | Path,
    domain: str | None = None,
    task_ids: list[str] | None = None,
    limit: int = 0,
    dispatch_log_path: str | Path | None = None,
    history_window: int = 10,
    obs_mode: str = "full",
    store_window_prompt: bool = True,
    include_system_prompt: bool = False,
) -> Path:
    """Write ``trajectories.jsonl`` + per-task ``.txt`` under ``output_dir``.

    By default each step stores a reconstructed window ``prompt`` /
    ``history_summary`` and ``system_prompt_sha1``. Full system text lives in
    ``runtime_context.json`` (and optionally on each trajectory when
    ``include_system_prompt=True`` for self-contained exports).
    """
    source: str | None = None
    if results_payload is None:
        if results_path is None:
            raise ValueError("Provide results_payload or results_path")
        path = Path(results_path).expanduser().resolve()
        results_payload = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    elif results_path is not None:
        source = str(Path(results_path).expanduser().resolve())

    sims = results_payload.get("simulations")
    if not isinstance(sims, list):
        raise ValueError("results payload has no simulations[]")

    results_info = results_payload.get("info")
    if not isinstance(results_info, dict):
        results_info = None

    tasks = results_payload.get("tasks") or []
    task_by_id = {
        str(t.get("id")): t
        for t in tasks
        if isinstance(t, dict) and t.get("id") is not None
    }

    journal = find_dispatch_journal(
        output_dir=output_dir, explicit=dispatch_log_path
    )
    inject_index = load_runtime_inject_index(journal)

    wanted = {str(t) for t in task_ids} if task_ids else None
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "trajectories.jsonl"
    mode = str(obs_mode or "full").strip().lower() or "full"

    context_by_sha: dict[str, dict[str, Any]] = {}
    pro_records: list[dict[str, Any]] = []
    n = 0
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for sim in sims:
            if not isinstance(sim, dict):
                continue
            tid = str(sim.get("task_id", ""))
            if wanted is not None and tid not in wanted:
                continue
            traj = simulation_to_trajectory(
                sim,
                domain=domain,
                source=source,
                dispatch_row=inject_index.get(tid),
                results_info=results_info,
                task=task_by_id.get(tid),
                history_window=history_window,
                obs_mode=mode,
                store_window_prompt=bool(store_window_prompt),
                include_system_prompt=bool(include_system_prompt),
            )
            ctx = traj.pop("_runtime_context", None) or {}
            sha = ctx.get("system_prompt_sha1")
            if sha and sha not in context_by_sha and ctx.get("system_prompt"):
                context_by_sha[sha] = {
                    "system_prompt": ctx.get("system_prompt"),
                    "domain_policy": ctx.get("domain_policy"),
                    "system_prompt_sha1": sha,
                    "domain_policy_sha1": ctx.get("domain_policy_sha1"),
                }
            fout.write(json.dumps(traj, ensure_ascii=False) + "\n")
            pro_records.append(traj)
            safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in tid)[:180]
            (out_dir / f"task_{safe}.txt").write_text(
                format_trajectory_text(traj), encoding="utf-8"
            )
            n += 1
            if limit and n >= limit:
                break

    if context_by_sha:
        # Prefer a single shared context file (first sha) plus map if multiple.
        contexts = list(context_by_sha.values())
        (out_dir / "runtime_context.json").write_text(
            json.dumps(
                {
                    "shared": contexts[0],
                    "by_sha1": context_by_sha,
                    "n_distinct": len(contexts),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    from sage_tau2.trajectory_adapter import adapt_pro_many, export_atomic_trajectories

    atomic = adapt_pro_many(pro_records)
    atomic_path = export_atomic_trajectories(atomic, out_dir / "atomic_trajectories.json")

    print(
        f"[sage_tau2] wrote {n} PRO v5 trajectories (obs_mode={mode}) -> {jsonl_path}"
        + f"; atomic_trajectories.json ({len(atomic)} episodes) -> {atomic_path}"
        + (f" (inject journal={journal})" if journal else " (no dispatch journal)"),
        flush=True,
    )
    return jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump τ² results as PRO v5 trajectories (obs_before/action/obs)"
    )
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--domain", default=None, help="Domain label for metadata")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Write trajectories.jsonl + per-task .txt here",
    )
    parser.add_argument(
        "--dispatch-log",
        default=None,
        help="dispatch_journal.jsonl with runtime inject snapshots",
    )
    parser.add_argument("--history-window", type=int, default=10)
    parser.add_argument(
        "--obs-mode",
        choices=("full", "lean"),
        default="full",
        help="full=near-raw observations (default); lean=semantic compression",
    )
    parser.add_argument(
        "--no-window-prompt",
        action="store_true",
        help="Omit per-step prompt/history_summary (legacy compact dump)",
    )
    parser.add_argument(
        "--include-system-prompt",
        action="store_true",
        help="Embed full system_prompt once per trajectory (else runtime_context.json only)",
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    export_pro_trajectories(
        results_path=args.results,
        output_dir=args.output_dir,
        domain=args.domain,
        task_ids=list(args.task_id) or None,
        limit=int(args.limit or 0),
        dispatch_log_path=args.dispatch_log,
        history_window=int(args.history_window or 10),
        obs_mode=str(args.obs_mode),
        store_window_prompt=not bool(args.no_window_prompt),
        include_system_prompt=bool(args.include_system_prompt),
    )


if __name__ == "__main__":
    main()
