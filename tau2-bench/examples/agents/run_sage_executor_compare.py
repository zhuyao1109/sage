#!/usr/bin/env python3
"""Compare ``llm_agent`` vs ``sage_executor`` on the first N airline tasks.

Loads OpenAI-compatible relay settings from verl-agent's
``examples/prompt_agent/llm_config.yaml`` (same relay used by SAGE).

Usage:
    cd tau2-bench
    uv run python examples/agents/run_sage_executor_compare.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
VERL_ROOT = REPO.parent
LLM_CONFIG = VERL_ROOT / "examples" / "prompt_agent" / "llm_config.yaml"
OUT_DIR = VERL_ROOT / "logs" / "tau2" / "sage_executor_compare"
SIM_DIR = REPO / "data" / "simulations"
NUM_TASKS = 5
DOMAIN = "airline"
MODEL = "openai/gpt-4o-mini"
SEED = 42


def _load_relay() -> tuple[str, str | None]:
    cfg = yaml.safe_load(LLM_CONFIG.read_text())
    openai_cfg = cfg.get("openai") or {}
    api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
    base_url = openai_cfg.get("base_url") or os.environ.get("OPENAI_API_BASE")
    if not api_key:
        raise SystemExit(
            f"Missing api_key in {LLM_CONFIG} and OPENAI_API_KEY env var."
        )
    return str(api_key), (str(base_url) if base_url else None)


def _run(agent: str, save_to: str, env: dict[str, str]) -> Path:
    out_dir = SIM_DIR / save_to
    results_path = out_dir / "results.json"
    if results_path.exists():
        print(f"[skip] {results_path} already exists", flush=True)
        return results_path

    cmd = [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        DOMAIN,
        "--agent",
        agent,
        "--agent-llm",
        MODEL,
        "--user-llm",
        MODEL,
        "--num-trials",
        "1",
        "--num-tasks",
        str(NUM_TASKS),
        "--max-concurrency",
        "1",
        "--seed",
        str(SEED),
        "--save-to",
        save_to,
    ]
    print("\n=== Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO), env=env, check=True)
    if not results_path.exists():
        raise FileNotFoundError(f"Expected results at {results_path}")
    return results_path


def _summarize(path: Path) -> dict:
    data = json.loads(path.read_text())
    sims = data.get("simulations") or []
    rewards = []
    rows = []
    for sim in sims:
        task_id = sim.get("task_id") or (sim.get("task") or {}).get("id")
        reward_info = sim.get("reward_info") or {}
        reward = reward_info.get("reward")
        if reward is None and "reward" in sim:
            reward = sim["reward"]
        rewards.append(float(reward) if reward is not None else 0.0)
        rows.append({"task_id": task_id, "reward": reward})
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    return {
        "path": str(path),
        "n": len(rewards),
        "avg_reward": avg,
        "pass_at_1": avg,
        "rows": rows,
    }


def main() -> None:
    api_key, base_url = _load_relay()
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    if base_url:
        env["OPENAI_API_BASE"] = base_url
        env["OPENAI_BASE_URL"] = base_url

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_path = _run("llm_agent", "compare_llm_agent_airline5_s42", env)
    sage_path = _run("sage_executor", "compare_sage_executor_airline5_s42", env)

    baseline = _summarize(baseline_path)
    sage = _summarize(sage_path)
    report = {
        "domain": DOMAIN,
        "num_tasks": NUM_TASKS,
        "model": MODEL,
        "seed": SEED,
        "base_url": base_url,
        "llm_agent": baseline,
        "sage_executor": sage,
        "delta_avg_reward": sage["avg_reward"] - baseline["avg_reward"],
    }
    out_json = OUT_DIR / "compare_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("\n========== Compare Report ==========")
    print(f"Model: {MODEL}  Domain: {DOMAIN}  N={NUM_TASKS}  seed={SEED}")
    print(f"llm_agent     avg_reward={baseline['avg_reward']:.4f}")
    print(f"sage_executor avg_reward={sage['avg_reward']:.4f}")
    print(f"delta (sage - llm) = {report['delta_avg_reward']:+.4f}")
    print(f"Wrote {out_json}")
    for name, block in ("llm_agent", baseline), ("sage_executor", sage):
        print(f"\n{name} per-task:")
        for row in block["rows"]:
            print(f"  {row['task_id']}: {row['reward']}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
