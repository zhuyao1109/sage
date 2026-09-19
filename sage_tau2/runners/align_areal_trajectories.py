"""Convert SAGE-τ² episodes to AReaL RL dump records.

Matches ``areal.infra.workflow_executor.WorkflowExecutor._dump_trajectory``:

```json
{
  "task_id": ...,
  "sample_idx": 0,
  "seqlen": null,
  "prompt_len": null,
  "head_version": null,
  "tail_version": null,
  "version_rle": null,
  "reward": 1.0,
  "original_reward": 1.0,
  "prompt": "...",
  "completion": "..."
}
```

One line per agent completion (AReaL ``export_style=individual``).
Token / version fields stay null — SAGE traces are text-only.

The agent system prompt is reconstructed and prepended to every turn's
``prompt`` (same memory role as live ``system_messages + history``).
Prefer ``dispatch_journal.jsonl`` runtime inject (system_prompt / skills /
org) when present; otherwise fall back to ``simulation.policy`` + default
Executor instructions.

Examples:
  PYTHONPATH=.. uv run python -m sage_tau2.runners.align_areal_trajectories \\
    --results data/simulations/sage_tau2_..._airline/results.json \\
    --output-dir ../logs/sage_tau2/.../pro_trajectories/airline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sage_tau2.organization import EXECUTOR_NAME
from sage_tau2.pro_steps import (
    find_dispatch_journal,
    format_msg_block as _format_msg_block,
    load_runtime_inject_index,
    runtime_inject_from_sources,
    tool_call_dict as _tool_call_dict,
)
from sage_tau2.prompts import EXECUTOR_INSTRUCTION, SPECIALIST_INSTRUCTION, SYSTEM_PROMPT
from sage_tau2.success_mode import classify_success_mode

# Keys written by AReaL _dump_trajectory (minus optional segments).
AREAL_DUMP_KEYS = (
    "task_id",
    "sample_idx",
    "seqlen",
    "prompt_len",
    "head_version",
    "tail_version",
    "version_rle",
    "reward",
    "prompt",
    "completion",
    "original_reward",
)

# AReaL dump + evolution label (written to areal_trajectories_typed.jsonl).
AREAL_TYPED_KEYS = AREAL_DUMP_KEYS + ("success_mode",)


def _to_openai_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    d = _tool_call_dict(tc)
    args = d.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    out: dict[str, Any] = {
        "type": "function",
        "function": {"name": d.get("name") or "?", "arguments": args},
    }
    if d.get("id"):
        out["id"] = d["id"]
    return out


def format_completion(content: str | None, tool_calls: list[dict[str, Any]]) -> str:
    """Approximate AReaL's decoded completion text for one agent turn."""
    parts: list[str] = []
    text = str(content or "").strip()
    if text:
        parts.append(text)
    if tool_calls:
        parts.append(
            "[TOOL_CALLS]\n" + json.dumps(tool_calls, ensure_ascii=False)
        )
    return "\n".join(parts) if parts else ""


def _episode_reward(sim: dict[str, Any]) -> float:
    reward_info = sim.get("reward_info") or {}
    raw = reward_info.get("reward")
    if raw is None:
        raw = sim.get("reward")
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _make_dump_record(
    *,
    task_id: Any,
    sample_idx: int,
    reward: float,
    prompt: str,
    completion: str,
    success_mode: str | None = None,
) -> dict[str, Any]:
    """Build one AReaL-compatible dump line (token fields null)."""
    record = {
        "task_id": task_id,
        "sample_idx": sample_idx,
        "seqlen": None,
        "prompt_len": None,
        "head_version": None,
        "tail_version": None,
        "version_rle": None,
        "reward": reward,
        "prompt": prompt,
        "completion": completion,
        "original_reward": reward,
    }
    if success_mode is not None:
        record["success_mode"] = success_mode
    return record


def _domain_policy_from_sim(
    sim: dict[str, Any],
    *,
    results_info: dict[str, Any] | None = None,
) -> str:
    """Prefer per-simulation policy; fall back to results.info.environment_info."""
    policy = sim.get("policy")
    if isinstance(policy, str) and policy.strip():
        return policy
    info = results_info or {}
    env = info.get("environment_info") if isinstance(info, dict) else None
    if isinstance(env, dict):
        env_policy = env.get("policy")
        if isinstance(env_policy, str) and env_policy.strip():
            return env_policy
    return ""


def reconstruct_system_prompt(
    sim: dict[str, Any],
    *,
    results_info: dict[str, Any] | None = None,
    agent_role_name: str = EXECUTOR_NAME,
    runtime_inject: dict[str, Any] | None = None,
) -> str:
    """Rebuild the live agent system prompt for trajectory memory.

    Prefer ``runtime_inject.system_prompt`` from dispatch_journal when present.
    Otherwise fall back to saved domain ``policy`` + default role instructions.
    """
    inject = runtime_inject or {}
    stored = inject.get("system_prompt")
    if isinstance(stored, str) and stored.strip():
        return stored
    domain_policy = _domain_policy_from_sim(sim, results_info=results_info)
    if not domain_policy and isinstance(inject.get("domain_policy"), str):
        domain_policy = inject["domain_policy"]
    if not domain_policy:
        return ""
    role_name = str(inject.get("primary_agent") or agent_role_name or EXECUTOR_NAME)
    is_specialist = role_name != EXECUTOR_NAME
    instruction = SPECIALIST_INSTRUCTION if is_specialist else EXECUTOR_INSTRUCTION
    role_line = f"Active role name: {role_name}.\n"
    org_block = str(inject.get("organization_block") or "")
    role_block = str(inject.get("role_block") or "")
    return SYSTEM_PROMPT.format(
        agent_instruction=role_line + instruction,
        domain_policy=domain_policy,
        role_block=role_block,
        organization_block="" if is_specialist else org_block,
    )


def _compose_prompt(
    *,
    system_prompt: str,
    prefix: list[dict[str, Any]],
) -> str:
    """System message first, then conversation prefix (live memory order)."""
    blocks: list[str] = []
    if system_prompt.strip():
        blocks.append(
            _format_msg_block({"role": "system", "content": system_prompt})
        )
    if prefix:
        blocks.append("\n\n".join(_format_msg_block(m) for m in prefix))
    return "\n\n".join(blocks)


def simulation_to_areal_turns(
    sim: dict[str, Any],
    *,
    domain: str | None = None,
    source: str | None = None,
    results_info: dict[str, Any] | None = None,
    with_success_mode: bool = False,
    runtime_inject: dict[str, Any] | None = None,
    history_window: int = 10,
) -> list[dict[str, Any]]:
    """Split one τ² simulation into AReaL individual-turn dump records.

    Each turn prompt is system + lean window (hist + current obs),
    reconstructed from PRO step fields (not a stored prompt 复读).
    """
    del domain, source  # not part of AReaL dump schema
    from sage_tau2.pro_steps import lean_window_from_prior_steps, messages_to_rich_pro_steps

    messages = [m for m in (sim.get("messages") or []) if isinstance(m, dict)]
    reward = _episode_reward(sim)
    task_id = sim.get("task_id")
    inject = runtime_inject or runtime_inject_from_sources(
        sim, results_info=results_info
    )
    system_prompt = reconstruct_system_prompt(
        sim, results_info=results_info, runtime_inject=inject
    )
    mode = (
        classify_success_mode(sim)["success_mode"] if with_success_mode else None
    )
    steps = messages_to_rich_pro_steps(
        messages, history_window=history_window, store_window_prompt=False
    )
    turns: list[dict[str, Any]] = []
    for sample_idx, step in enumerate(steps):
        tool_calls = [
            _to_openai_tool_call(tc) for tc in (step.get("tool_calls") or [])
        ]
        resp = step.get("response") or {}
        if not tool_calls:
            tool_calls = [
                _to_openai_tool_call(tc) for tc in (resp.get("tool_calls") or [])
            ]
        content = resp.get("content")
        if content is None and not tool_calls:
            content = step.get("action")
        elif content is None and tool_calls:
            content = None
        window = lean_window_from_prior_steps(
            steps[:sample_idx],
            observation_before=str(step.get("observation_before") or "(none)"),
            history_window=history_window,
            task_description=str(inject.get("task_text") or "") or None,
            skills_block=str(inject.get("skills_block") or "") or None,
        )
        if system_prompt.strip() and window:
            prompt = (
                _format_msg_block({"role": "system", "content": system_prompt})
                + "\n\n"
                + window
            )
        elif system_prompt.strip():
            prompt = _format_msg_block(
                {"role": "system", "content": system_prompt}
            )
        else:
            prompt = window
        turns.append(
            _make_dump_record(
                task_id=task_id,
                sample_idx=sample_idx,
                reward=reward,
                prompt=prompt,
                completion=format_completion(content, tool_calls),
                success_mode=mode,
            )
        )
    return turns


def pro_trajectory_to_areal_turns(
    traj: dict[str, Any],
    *,
    system_prompt: str = "",
    success_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Fallback: split an already-exported SAGE PRO episode (no raw messages)."""
    reward = traj.get("reward")
    try:
        reward_f = float(reward or 0.0)
    except (TypeError, ValueError):
        reward_f = 0.0
    if success_mode is None and traj.get("success_mode"):
        success_mode = str(traj["success_mode"])
    steps = traj.get("steps") or []
    system_block = (
        _format_msg_block({"role": "system", "content": system_prompt})
        if system_prompt.strip()
        else ""
    )
    turns: list[dict[str, Any]] = []
    for sample_idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        resp = step.get("response") or {}
        tool_calls = [
            _to_openai_tool_call(tc)
            for tc in (step.get("tool_calls") or resp.get("tool_calls") or [])
        ]
        content = resp.get("content")
        if content is None and not tool_calls:
            content = step.get("action")
        # Prefer bounded window prompt; legacy dumps may still have full chat.
        history = step.get("prompt") or ""
        if history == "(episode start)":
            history = ""
        if system_block and history:
            prompt = f"{system_block}\n\n{history}"
        elif system_block:
            prompt = system_block
        else:
            prompt = history
        turns.append(
            _make_dump_record(
                task_id=traj.get("task_id"),
                sample_idx=sample_idx,
                reward=reward_f,
                prompt=prompt,
                completion=format_completion(content, tool_calls),
                success_mode=success_mode,
            )
        )
    return turns


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for rec in rows:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_areal_trajectories(
    *,
    results_payload: dict[str, Any] | None = None,
    results_path: str | Path | None = None,
    pro_jsonl_path: str | Path | None = None,
    output_dir: str | Path,
    domain: str | None = None,
    task_ids: list[str] | None = None,
    limit: int = 0,
    filename: str = "areal_trajectories.jsonl",
    with_success_mode: bool = False,
    also_write_typed: bool = False,
    typed_filename: str = "areal_trajectories_typed.jsonl",
    summary_filename: str = "success_mode_summary.json",
    dispatch_log_path: str | Path | None = None,
) -> Path:
    """Write AReaL dump-schema jsonl under ``output_dir``.

    When ``also_write_typed`` is True (typical for results.json exports), also
    write ``typed_filename`` with ``success_mode`` and an episode summary JSON.
    Existing ``filename`` is overwritten only if this function writes it; callers
    that only want typed output should pass a dedicated ``filename`` or set
    ``also_write_typed=True`` after an existing plain dump.
    """
    turns: list[dict[str, Any]] = []
    typed_turns: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    wanted = {str(t) for t in task_ids} if task_ids else None

    if results_payload is None and results_path is not None:
        path = Path(results_path).expanduser().resolve()
        results_payload = json.loads(path.read_text(encoding="utf-8"))
    elif results_path is not None:
        pass

    n_episodes = 0
    results_info = (
        results_payload.get("info")
        if isinstance(results_payload, dict)
        else None
    )
    journal = find_dispatch_journal(
        output_dir=output_dir, explicit=dispatch_log_path
    )
    inject_index = load_runtime_inject_index(journal)
    if results_payload is not None:
        sims = results_payload.get("simulations")
        if not isinstance(sims, list):
            raise ValueError("results payload has no simulations[]")
        for sim in sims:
            if not isinstance(sim, dict):
                continue
            tid = str(sim.get("task_id", ""))
            if wanted is not None and tid not in wanted:
                continue
            info = results_info if isinstance(results_info, dict) else None
            label = classify_success_mode(sim)
            episode_rows.append(
                {
                    "task_id": sim.get("task_id"),
                    "trial": sim.get("trial"),
                    "domain": domain,
                    "termination_reason": sim.get("termination_reason"),
                    **label,
                }
            )
            inject = runtime_inject_from_sources(
                sim,
                dispatch_row=inject_index.get(tid),
                results_info=info,
            )
            plain = simulation_to_areal_turns(
                sim,
                domain=domain,
                source=None,
                results_info=info,
                with_success_mode=False,
                runtime_inject=inject,
            )
            typed = simulation_to_areal_turns(
                sim,
                domain=domain,
                source=None,
                results_info=info,
                with_success_mode=True,
                runtime_inject=inject,
            )
            turns.extend(plain)
            typed_turns.extend(typed)
            n_episodes += 1
            if limit and n_episodes >= limit:
                break
    elif pro_jsonl_path is not None:
        path = Path(pro_jsonl_path).expanduser().resolve()
        with path.open(encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                traj = json.loads(line)
                if not isinstance(traj, dict):
                    continue
                tid = str(traj.get("task_id", ""))
                if wanted is not None and tid not in wanted:
                    continue
                mode = traj.get("success_mode")
                if mode is None:
                    try:
                        mode = (
                            "fail"
                            if float(traj.get("reward") or 0) < 1.0
                            else "communicate_ok"
                        )
                    except (TypeError, ValueError):
                        mode = "communicate_ok"
                inject = traj.get("runtime_inject") or {}
                system_prompt = ""
                if isinstance(inject, dict):
                    system_prompt = str(inject.get("system_prompt") or "")
                plain = pro_trajectory_to_areal_turns(
                    traj, system_prompt=system_prompt, success_mode=None
                )
                typed = pro_trajectory_to_areal_turns(
                    traj, system_prompt=system_prompt, success_mode=str(mode)
                )
                turns.extend(plain)
                typed_turns.extend(typed)
                n_episodes += 1
                if limit and n_episodes >= limit:
                    break
    else:
        raise ValueError("Provide results_payload, results_path, or pro_jsonl_path")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if with_success_mode and not also_write_typed:
        # Single typed file request.
        jsonl_path = out_dir / filename
        _write_jsonl(jsonl_path, typed_turns or turns)
        print(
            f"[sage_tau2] wrote {len(typed_turns or turns)} AReaL typed turns "
            f"from {n_episodes} episodes -> {jsonl_path}",
            flush=True,
        )
        if episode_rows:
            _write_summary(out_dir / summary_filename, episode_rows, domain=domain)
        return jsonl_path

    jsonl_path = out_dir / filename
    _write_jsonl(jsonl_path, turns)
    print(
        f"[sage_tau2] wrote {len(turns)} AReaL dump turns "
        f"from {n_episodes} episodes -> {jsonl_path}",
        flush=True,
    )
    if also_write_typed:
        typed_path = out_dir / typed_filename
        _write_jsonl(typed_path, typed_turns)
        print(
            f"[sage_tau2] wrote {len(typed_turns)} AReaL typed turns "
            f"from {n_episodes} episodes -> {typed_path}",
            flush=True,
        )
        if episode_rows:
            _write_summary(out_dir / summary_filename, episode_rows, domain=domain)
    return jsonl_path


def _write_summary(
    path: Path,
    episode_rows: list[dict[str, Any]],
    *,
    domain: str | None,
) -> None:
    from collections import Counter

    counts = Counter(str(r.get("success_mode")) for r in episode_rows)
    payload = {
        "domain": domain,
        "num_episodes": len(episode_rows),
        "success_mode_counts": dict(counts),
        "episodes": episode_rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[sage_tau2] success_mode summary {dict(counts)} -> {path}",
        flush=True,
    )


def export_aligned_from_pro_export(
    *,
    results_payload: dict[str, Any] | None,
    results_path: str | Path | None,
    output_dir: str | Path,
    domain: str | None,
    task_ids: list[str] | None,
    limit: int,
    dispatch_log_path: str | Path | None = None,
) -> Path | None:
    """Called by SAGE PRO dump so both formats land in the same folder."""
    if results_payload is None and results_path is None:
        return None
    return export_areal_trajectories(
        results_payload=results_payload,
        results_path=results_path,
        output_dir=output_dir,
        domain=domain,
        task_ids=task_ids,
        limit=limit,
        also_write_typed=True,
        dispatch_log_path=dispatch_log_path,
    )


def export_typed_only(
    *,
    results_path: str | Path,
    output_dir: str | Path,
    domain: str | None = None,
    task_ids: list[str] | None = None,
    limit: int = 0,
) -> Path:
    """Write typed trajectories + summary without touching plain dump."""
    return export_areal_trajectories(
        results_path=results_path,
        output_dir=output_dir,
        domain=domain,
        task_ids=task_ids,
        limit=limit,
        filename="areal_trajectories_typed.jsonl",
        with_success_mode=True,
        also_write_typed=False,
        typed_filename="areal_trajectories_typed.jsonl",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SAGE-τ² results/PRO dumps to AReaL RL dump jsonl "
            "(prompt/completion/reward schema)"
        )
    )
    parser.add_argument("--results", default=None, help="Path to tau2 results.json")
    parser.add_argument(
        "--pro-jsonl",
        default=None,
        help="Fallback: SAGE trajectories.jsonl when results.json is unavailable",
    )
    parser.add_argument("--domain", default=None)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Write areal_trajectories.jsonl here",
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--typed-only",
        action="store_true",
        help=(
            "Only write areal_trajectories_typed.jsonl + success_mode_summary.json; "
            "do not overwrite areal_trajectories.jsonl"
        ),
    )
    parser.add_argument(
        "--also-typed",
        action="store_true",
        help="Also write typed dump + summary alongside the plain AReaL dump",
    )
    args = parser.parse_args()
    if not args.results and not args.pro_jsonl:
        raise SystemExit("Provide --results or --pro-jsonl")
    if args.typed_only:
        if not args.results:
            raise SystemExit("--typed-only requires --results")
        export_typed_only(
            results_path=args.results,
            output_dir=args.output_dir,
            domain=args.domain,
            task_ids=list(args.task_id) or None,
            limit=int(args.limit or 0),
        )
        return
    export_areal_trajectories(
        results_path=args.results,
        pro_jsonl_path=args.pro_jsonl,
        output_dir=args.output_dir,
        domain=args.domain,
        task_ids=list(args.task_id) or None,
        limit=int(args.limit or 0),
        also_write_typed=bool(args.also_typed),
    )


__all__ = [
    "AREAL_DUMP_KEYS",
    "AREAL_TYPED_KEYS",
    "export_aligned_from_pro_export",
    "export_areal_trajectories",
    "export_typed_only",
    "format_completion",
    "pro_trajectory_to_areal_turns",
    "reconstruct_system_prompt",
    "simulation_to_areal_turns",
]


if __name__ == "__main__":
    main()
