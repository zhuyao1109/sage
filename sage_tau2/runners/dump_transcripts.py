"""Dump τ² ``results.json`` ``messages`` into readable step transcripts.

Examples:
  PYTHONPATH=.. uv run python -m sage_tau2.runners.dump_transcripts \\
    --results data/simulations/sage_tau2_executor_guard_test100_airline/results.json \\
    --task-id 19

  PYTHONPATH=.. uv run python -m sage_tau2.runners.dump_transcripts \\
    --results data/simulations/sage_tau2_executor_guard_test100_airline/results.json \\
    --output-dir /tmp/airline_transcripts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _tool_call_line(tc: dict[str, Any]) -> str:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = fn.get("name") or tc.get("name") or "?"
    raw_args = fn.get("arguments", tc.get("arguments", {}))
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            pass
    if isinstance(raw_args, dict):
        arg_s = json.dumps(raw_args, ensure_ascii=False)
    else:
        arg_s = str(raw_args)
    if len(arg_s) > 500:
        arg_s = arg_s[:500] + "..."
    return f"{name}({arg_s})"


def format_simulation(sim: dict[str, Any], *, index: int | None = None) -> str:
    task_id = sim.get("task_id", "?")
    reward_info = sim.get("reward_info") or {}
    reward = reward_info.get("reward")
    term = sim.get("termination_reason")
    duration = sim.get("duration")
    lines: list[str] = []
    header = f"Task {task_id}"
    if index is not None:
        header = f"[{index}] {header}"
    lines.append("=" * 72)
    lines.append(header)
    lines.append(
        f"reward={reward}  termination={term}  "
        f"duration={duration:.1f}s" if isinstance(duration, (int, float)) else
        f"reward={reward}  termination={term}"
    )
    db = reward_info.get("db_check") or {}
    if db:
        lines.append(f"db_match={db.get('db_match')}  db_reward={db.get('db_reward')}")
    lines.append("-" * 72)

    step = 0
    for msg in sim.get("messages") or []:
        role = str(msg.get("role") or "?")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []
        if role == "tool":
            step += 1
            name = msg.get("name") or "tool"
            body = str(content if content is not None else "")
            if len(body) > 800:
                body = body[:800] + "..."
            lines.append(f"{step:02d}. [TOOL:{name}] {body}")
            continue
        if tool_calls:
            for tc in tool_calls:
                step += 1
                lines.append(f"{step:02d}. [AGENT→TOOL] {_tool_call_line(tc)}")
        text = str(content).strip() if content else ""
        if text:
            step += 1
            label = {
                "user": "USER",
                "assistant": "AGENT",
                "system": "SYSTEM",
                "developer": "SYSTEM",
            }.get(role, role.upper())
            lines.append(f"{step:02d}. [{label}] {text}")
    lines.append("")
    return "\n".join(lines)


def load_simulations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sims = data.get("simulations")
    if not isinstance(sims, list):
        raise SystemExit(f"No simulations[] in {path}")
    return sims


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format τ² results.json messages as readable transcripts"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to a τ² results.json",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Only dump this task id (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="If set, write one .txt per task; else print to stdout",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of sims to dump (0 = all matching)",
    )
    args = parser.parse_args()

    results_path = Path(args.results).expanduser().resolve()
    sims = load_simulations(results_path)
    wanted = {str(t) for t in args.task_id} if args.task_id else None
    selected: list[tuple[int, dict[str, Any]]] = []
    for i, sim in enumerate(sims):
        tid = str(sim.get("task_id", ""))
        if wanted is not None and tid not in wanted:
            continue
        selected.append((i, sim))
        if args.limit and len(selected) >= args.limit:
            break

    if not selected:
        raise SystemExit("No matching simulations")

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, sim in selected:
        text = format_simulation(sim, index=i)
        if out_dir is None:
            print(text)
            continue
        tid = str(sim.get("task_id", i)).replace("/", "_")
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in tid)[:180]
        path = out_dir / f"task_{safe}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
