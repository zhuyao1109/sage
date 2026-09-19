"""Export / load evolve-ready trajectories for SAGE-τ².

One JSONL row per episode with **compact** dialogue (no LLM API shells).
Also writes human-readable ``evolve_readable/task_<id>.txt``.

Distill still round-trips via ``tool_protocol`` / ``tool_steps`` / ``success_mode``.

Examples:
  PYTHONPATH=. python -m sage_tau2.runners.dump_evolve_trajectories \\
    --results tau2-bench/data/simulations/.../results.json \\
    --domain airline \\
    --output-dir logs/sage_tau2/.../pro_trajectories/airline
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sage_tau2.distill import (
    protocol_has_write,
    trajectory_eligible_for_distill,
)
from sage_tau2.schemas import Tau2Trajectory, ToolCallStep
from sage_tau2.serialization import trajectory_from_dict
from sage_tau2.trajectory import simulation_to_trajectory

EVOLVE_FORMAT = "sage_tau2_evolve_v2"

_CONTENT_MAX = 600
_TOOL_RESULT_MAX = 200
_SYSTEM_IN_READABLE_MAX = 2500


def _truncate(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    s = str(text)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _slim_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    """Name + arg keys only (values are rarely needed for distill)."""
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = fn.get("name") or tc.get("name") or "?"
    args = fn.get("arguments", tc.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    # Keep concrete args short; large nested payloads become key list.
    slim_args: dict[str, Any] = {}
    for key, val in args.items():
        rendered = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
        if len(rendered) > 80:
            slim_args[str(key)] = "?"
        else:
            slim_args[str(key)] = val
    return {"name": str(name), "arguments": slim_args}


def slim_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Dialogue only: role / short content / tool names — no API shells."""
    role = str(msg.get("role") or "?")
    out: dict[str, Any] = {"role": role}
    content = msg.get("content")
    if content not in (None, ""):
        max_c = _TOOL_RESULT_MAX if role == "tool" else _CONTENT_MAX
        out["content"] = _truncate(str(content), max_c)
    if role == "tool":
        if msg.get("name"):
            out["name"] = msg.get("name")
        return out
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        out["tool_calls"] = [
            _slim_tool_call(tc) for tc in tool_calls if isinstance(tc, dict)
        ]
    return out


def slim_tool_steps(steps: list[Any]) -> list[dict[str, Any]]:
    """Keep concrete short args for grounding; only blow up huge payloads."""
    out: list[dict[str, Any]] = []
    for step in steps or []:
        if isinstance(step, ToolCallStep):
            name, args, err = step.name, dict(step.arguments or {}), step.result_error
        elif isinstance(step, dict):
            name = str(step.get("name") or "")
            args = dict(step.get("arguments") or {})
            err = bool(step.get("result_error"))
        else:
            continue
        if not name:
            continue
        slim_args: dict[str, Any] = {}
        for key, val in args.items():
            if val is None or isinstance(val, (dict, list)):
                slim_args[str(key)] = "?"
                continue
            if isinstance(val, (str, int, float, bool)):
                rendered = str(val)
            else:
                rendered = json.dumps(val, ensure_ascii=False)
            if len(rendered) > 80:
                slim_args[str(key)] = "?"
            else:
                slim_args[str(key)] = val
        out.append({"name": name, "arguments": slim_args, "result_error": err})
    return out


def slim_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Only labels distill/credit need."""
    src = dict(meta or {})
    keep = ("success_mode", "write_tools", "transfer_tools", "gold_actions")
    return {k: src[k] for k in keep if k in src and src[k] is not None}


def dialogue_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Human-skimmable turns (already slim)."""
    return [slim_message(m) for m in messages if isinstance(m, dict)]


def _tool_call_for_readable(tc: dict[str, Any]) -> str:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = fn.get("name") or tc.get("name") or "?"
    args = fn.get("arguments", tc.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    args_s = json.dumps(args, ensure_ascii=False, indent=2)
    if len(args_s) > 1200:
        args_s = args_s[:1197] + "..."
    return f"{name}(\n{args_s}\n)"


def format_interaction_transcript(
    *,
    task_id: Any,
    trial: Any,
    success_mode: Any,
    eligible: Any,
    reward: Any,
    db_match: Any,
    domain: Any,
    write_spine: Any,
    messages: list[dict[str, Any]],
    system_prompt: str = "",
    include_system: bool = False,
) -> str:
    """Human view of model ↔ τ²-bench interaction (not the compact jsonl)."""
    lines = [
        "=" * 72,
        "MODEL ↔ τ²-BENCH INTERACTION",
        f"task={task_id}  trial={trial}  mode={success_mode}  eligible={eligible}",
        f"reward={reward}  db_match={db_match}  domain={domain}",
        f"write_spine={write_spine}",
        "",
        "Legend:",
        "  [USER]     = bench simulated customer",
        "  [MODEL]    = agent reply / tool call",
        "  [BENCH]    = tool / environment result",
        "-" * 72,
    ]
    if include_system and system_prompt:
        lines.append("[SYSTEM] (agent instructions; truncated)")
        lines.append(_truncate(system_prompt, _SYSTEM_IN_READABLE_MAX) or "")
        lines.append("")

    turn = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        turn += 1
        if role == "user":
            lines.append(f"### {turn}. [USER]")
            lines.append(str(msg.get("content") or "").strip() or "(empty)")
        elif role == "assistant":
            lines.append(f"### {turn}. [MODEL]")
            content = str(msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                lines.append("(calls tool)")
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        lines.append(_tool_call_for_readable(tc))
            if content:
                if tool_calls:
                    lines.append("(also says)")
                lines.append(content)
            if not content and not tool_calls:
                lines.append("(empty)")
        elif role == "tool":
            name = msg.get("name") or "tool"
            lines.append(f"### {turn}. [BENCH] tool={name}")
            body = str(msg.get("content") if msg.get("content") is not None else "")
            # Pretty-print JSON tool payloads when possible.
            try:
                parsed = json.loads(body)
                body = json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                pass
            if len(body) > 3000:
                body = body[:2997] + "..."
            lines.append(body or "(empty)")
        else:
            lines.append(f"### {turn}. [{role.upper()}]")
            lines.append(str(msg.get("content") or ""))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_evolve_readable(
    row: dict[str, Any],
    *,
    system_prompt: str = "",
    messages: list[dict[str, Any]] | None = None,
    include_system: bool = False,
) -> str:
    """Plain-text model↔bench transcript for editors."""
    msgs = messages
    if msgs is None:
        msgs = list(row.get("dialogue") or row.get("raw_messages") or [])
    return format_interaction_transcript(
        task_id=row.get("task_id"),
        trial=row.get("trial"),
        success_mode=row.get("success_mode"),
        eligible=row.get("eligible_for_distill"),
        reward=row.get("reward"),
        db_match=row.get("db_match"),
        domain=row.get("domain"),
        write_spine=row.get("write_spine"),
        messages=msgs,
        system_prompt=system_prompt,
        include_system=include_system,
    )


def _write_spine(traj: Tau2Trajectory) -> list[str]:
    from sage_tau2.distill import canonicalize_protocol, trim_protocol_to_write_spine

    spine = trim_protocol_to_write_spine(
        canonicalize_protocol(
            list(traj.tool_protocol or []),
            domain=getattr(traj, "domain", None),
        )
    )
    return spine if protocol_has_write(spine) else []


def trajectory_to_evolve_row(
    traj: Tau2Trajectory,
    *,
    system_prompt: str = "",
    source: str | None = None,
) -> dict[str, Any]:
    """Minimal evolve JSONL row — distill fields + short dialogue only.

    Full system prompt / long tool dumps go to ``evolve_readable/``, not here.
    ``system_prompt`` is accepted for readable export but not stored in jsonl.
    """
    del system_prompt, source  # kept out of machine row on purpose
    meta = slim_metadata(traj.metadata)
    mode = str(meta.get("success_mode") or "") or None
    write_spine = _write_spine(traj)
    eligible = trajectory_eligible_for_distill(traj, require_success=True)
    slim_msgs = dialogue_from_messages(list(traj.raw_messages or []))
    return {
        "format": EVOLVE_FORMAT,
        "task_id": traj.task_id,
        "trial": traj.trial,
        "domain": traj.domain,
        "reward": traj.reward,
        "db_match": traj.db_match,
        "success_mode": mode,
        "eligible_for_distill": bool(eligible),
        "tool_protocol": list(traj.tool_protocol or []),
        "write_spine": write_spine,
        "tool_steps": slim_tool_steps(list(traj.tool_steps or [])),
        "dialogue": slim_msgs,
        "user_texts": list(traj.user_texts or [])[:6],
        "metadata": meta,
    }


def evolve_row_to_trajectory(row: dict[str, Any]) -> Tau2Trajectory:
    """Load one evolve JSONL row back into ``Tau2Trajectory``."""
    payload = dict(row)
    meta = dict(payload.get("metadata") or {})
    if payload.get("success_mode") and not meta.get("success_mode"):
        meta["success_mode"] = payload["success_mode"]
    payload["metadata"] = meta
    msgs = payload.get("raw_messages") or payload.get("dialogue") or []
    payload["raw_messages"] = msgs
    payload.setdefault("db_reward", None)
    payload.setdefault("communicate_reward", None)
    payload.setdefault("termination_reason", None)
    payload.setdefault("assistant_texts", [])
    payload.setdefault("user_texts", [])
    payload.setdefault("evidence_id", str(payload.get("task_id") or ""))
    if not payload.get("tool_steps"):
        payload["tool_steps"] = []
    return trajectory_from_dict(payload)


def load_evolve_trajectories(
    path: str | Path,
    *,
    eligible_only: bool = False,
) -> list[Tau2Trajectory]:
    """Read evolve JSONL into trajectories for distill / credit."""
    rows = [
        json.loads(line)
        for line in Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out: list[Tau2Trajectory] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if eligible_only and not row.get("eligible_for_distill"):
            continue
        out.append(evolve_row_to_trajectory(row))
    return out


def simulation_to_evolve_row(
    sim: dict[str, Any],
    *,
    domain: str,
    task: dict[str, Any] | None = None,
    results_info: dict[str, Any] | None = None,
    source: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Return ``(compact_row, system_prompt)`` for jsonl + readable export."""
    del source
    traj = simulation_to_trajectory(sim, domain=domain, task=task)
    system_prompt = ""
    try:
        from sage_tau2.runners.align_areal_trajectories import reconstruct_system_prompt

        system_prompt = reconstruct_system_prompt(sim, results_info=results_info) or ""
    except Exception:  # noqa: BLE001
        system_prompt = ""
    return trajectory_to_evolve_row(traj), system_prompt


def export_evolve_trajectories(
    *,
    results_payload: dict[str, Any] | None = None,
    results_path: str | Path | None = None,
    output_dir: str | Path,
    domain: str,
    task_ids: list[str] | None = None,
    limit: int = 0,
    filename: str = "evolve_trajectories.jsonl",
    summary_filename: str = "evolve_summary.json",
    solve_write_only_file: str = "evolve_trajectories_solve_write.jsonl",
    readable_dirname: str = "evolve_readable",
) -> Path:
    """Write compact evolve jsonl + readable txt + summary."""
    source: str | None = None
    if results_payload is None:
        if results_path is None:
            raise ValueError("Provide results_payload or results_path")
        path = Path(results_path).expanduser().resolve()
        results_payload = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    elif results_path is not None:
        source = str(Path(results_path).expanduser().resolve())

    results_info = (
        results_payload.get("info") if isinstance(results_payload, dict) else None
    )
    tasks = results_payload.get("tasks") or []
    task_by_id = {
        str(t.get("id")): t
        for t in tasks
        if isinstance(t, dict) and t.get("id") is not None
    }
    wanted = {str(t) for t in task_ids} if task_ids else None

    rows: list[dict[str, Any]] = []
    system_by_key: dict[tuple[str, int], str] = {}
    messages_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for sim in results_payload.get("simulations") or []:
        if not isinstance(sim, dict):
            continue
        task_id = str(sim.get("task_id") or "")
        if wanted is not None and task_id not in wanted:
            continue
        row, system_prompt = simulation_to_evolve_row(
            sim,
            domain=domain,
            task=task_by_id.get(task_id),
            results_info=results_info if isinstance(results_info, dict) else None,
            source=source,
        )
        key = (str(row.get("task_id")), int(row.get("trial") or 0))
        rows.append(row)
        system_by_key[key] = system_prompt
        messages_by_key[key] = [
            m for m in (sim.get("messages") or []) if isinstance(m, dict)
        ]
        if limit and len(rows) >= limit:
            break

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / filename
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    solve_rows = [r for r in rows if r.get("eligible_for_distill")]
    solve_path = out_dir / solve_write_only_file
    with solve_path.open("w", encoding="utf-8") as fout:
        for row in solve_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    readable_dir = out_dir / readable_dirname
    readable_dir.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "MODEL ↔ τ²-BENCH interaction transcripts",
        f"domain={domain}  episodes={len(rows)}  eligible={len(solve_rows)}",
        "",
    ]
    for row in rows:
        tid = str(row.get("task_id") or "unknown")
        trial = int(row.get("trial") or 0)
        key = (tid, trial)
        name = f"task_{tid}.txt" if trial == 0 else f"task_{tid}_trial{trial}.txt"
        (readable_dir / name).write_text(
            format_evolve_readable(
                row,
                system_prompt=system_by_key.get(key, ""),
                messages=messages_by_key.get(key),
                include_system=False,
            ),
            encoding="utf-8",
        )
        index_lines.append(
            f"{name}  mode={row.get('success_mode')}  "
            f"eligible={row.get('eligible_for_distill')}  reward={row.get('reward')}"
        )
    (readable_dir / "INDEX.txt").write_text(
        "\n".join(index_lines) + "\n", encoding="utf-8"
    )

    mode_counts = Counter(str(r.get("success_mode")) for r in rows)
    summary = {
        "domain": domain,
        "format": EVOLVE_FORMAT,
        "source": source,
        "num_episodes": len(rows),
        "success_mode_counts": dict(mode_counts),
        "eligible_for_distill": len(solve_rows),
        "evolve_path": str(jsonl_path),
        "solve_write_path": str(solve_path),
        "readable_dir": str(readable_dir),
        "episodes": [
            {
                "task_id": r.get("task_id"),
                "trial": r.get("trial"),
                "success_mode": r.get("success_mode"),
                "eligible_for_distill": r.get("eligible_for_distill"),
                "write_spine": r.get("write_spine"),
                "reward": r.get("reward"),
                "readable": str(
                    readable_dir
                    / (
                        f"task_{r.get('task_id')}.txt"
                        if int(r.get("trial") or 0) == 0
                        else f"task_{r.get('task_id')}_trial{r.get('trial')}.txt"
                    )
                ),
            }
            for r in rows
        ],
    }
    summary_path = out_dir / summary_filename
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[sage_tau2] wrote {len(rows)} evolve episodes "
        f"(distill-eligible={len(solve_rows)}, format={EVOLVE_FORMAT}) -> {jsonl_path}",
        flush=True,
    )
    print(
        f"[sage_tau2] readable transcripts -> {readable_dir}",
        flush=True,
    )
    print(
        f"[sage_tau2] evolve summary {dict(mode_counts)} -> {summary_path}",
        flush=True,
    )
    return jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump compact evolve trajectories + readable transcripts"
    )
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--domain", required=True, help="Domain label")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Write evolve_trajectories.jsonl + evolve_readable/ here",
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    export_evolve_trajectories(
        results_path=args.results,
        output_dir=args.output_dir,
        domain=args.domain,
        task_ids=list(args.task_id) or None,
        limit=int(args.limit or 0),
    )


__all__ = [
    "EVOLVE_FORMAT",
    "evolve_row_to_trajectory",
    "export_evolve_trajectories",
    "format_evolve_readable",
    "load_evolve_trajectories",
    "simulation_to_evolve_row",
    "slim_message",
    "trajectory_to_evolve_row",
]


if __name__ == "__main__":
    main()
