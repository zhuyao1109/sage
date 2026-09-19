"""Frozen multi-domain eval for an evolved sage_tau2 checkpoint.

Loads skill_bank.json + organization.json from ``--checkpoint`` and runs
``sage_tau2`` with inject + dispatch, but does **not** distill / admit /
mutate the bank or org.

Default OOD suite (~278 ≈ “279”): airline base 50 + retail base 114 +
telecom base 114.

```bash
cd ~/verl-agent/tau2-bench
PYTHONPATH=.. uv run python -m sage_tau2.runners.eval_frozen \\
  --checkpoint ../logs/sage_tau2/airline_evolve_10seg \\
  --output ../logs/sage_tau2/airline_evolve_10seg_ood278 \\
  --llm-config ../examples/prompt_agent/llm_config.yaml
```
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from sage_tau2.runners.dump_pro_trajectories import export_pro_trajectories
from sage_tau2.runners.online_tau2 import (
    _apply_relay_from_llm_config,
    _register_agent,
)
from sage_tau2.serialization import read_json, write_json


DEFAULT_DOMAINS: list[tuple[str, int]] = [
    ("airline", 50),
    ("retail", 114),
    ("telecom", 114),
]


def _run_domain_eval(
    *,
    domain: str,
    num_tasks: int,
    model: str,
    user_model: str,
    seed: int,
    max_concurrency: int,
    save_to: str,
    skill_bank_path: Path,
    organization_path: Path,
    max_inject_skills: int,
    allow_provisional_inject: bool,
    agent_name: str,
    task_split_name: str,
    executor_only: bool = False,
    accepted_only: bool = False,
) -> dict[str, Any]:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import run_domain

    os.environ["SAGE_TAU2_SKILL_BANK"] = str(skill_bank_path)
    os.environ["SAGE_TAU2_ORG_PATH"] = str(organization_path)
    os.environ["SAGE_TAU2_MAX_SKILLS"] = str(max_inject_skills)
    os.environ["SAGE_TAU2_INJECT_PROVISIONAL"] = (
        "1" if allow_provisional_inject else "0"
    )
    os.environ["SAGE_TAU2_INJECT_SAME_DOMAIN_ONLY"] = "1"
    os.environ["SAGE_TAU2_ENABLE_DISPATCH"] = "0" if executor_only else "1"
    os.environ["SAGE_TAU2_DOMAIN"] = str(domain)
    dispatch_log = Path(skill_bank_path).resolve().parent / "dispatch_journal.jsonl"
    os.environ["SAGE_TAU2_DISPATCH_LOG"] = str(dispatch_log)
    if executor_only:
        os.environ["SAGE_TAU2_FORCE_PRIMARY"] = "Executor"
    else:
        os.environ.pop("SAGE_TAU2_FORCE_PRIMARY", None)

    llm_args_agent: dict[str, Any] = {
        "skill_bank_path": str(skill_bank_path),
        "organization_path": str(organization_path),
        "max_inject_skills": max_inject_skills,
        "inject_provisional": allow_provisional_inject,
        "inject_same_domain_only": True,
        "enable_executor_dispatch": not executor_only,
        "dispatch_log_path": str(dispatch_log),
    }
    if executor_only:
        llm_args_agent["force_primary"] = "Executor"
    if accepted_only:
        llm_args_agent["dispatch_config"] = {
            "require_accepted_for_primary": True,
            "probation_primary_quota": 0,
        }
    config = TextRunConfig(
        domain=domain,
        agent=agent_name,
        llm_agent=model,
        llm_user=user_model,
        num_trials=1,
        num_tasks=num_tasks,
        seed=seed,
        max_concurrency=max_concurrency,
        save_to=save_to,
        task_split_name=task_split_name,
        llm_args_agent=llm_args_agent,
    )
    run_domain(config)
    sim_path = Path("data/simulations") / save_to / "results.json"
    if not sim_path.exists():
        raise RuntimeError(f"Missing results for save_to={save_to}: {sim_path}")
    return read_json(sim_path)


def _reward_stats(results_payload: dict[str, Any]) -> dict[str, Any]:
    sims = results_payload.get("simulations") or []
    rewards: list[float] = []
    for sim in sims:
        if not isinstance(sim, dict):
            continue
        ri = sim.get("reward_info") or {}
        r = ri.get("reward")
        if isinstance(r, (int, float)):
            rewards.append(float(r))
    n = len(rewards)
    n_pass = sum(1 for r in rewards if r >= 1.0)
    return {
        "n_simulations": len(sims),
        "n_scored": n,
        "n_pass": n_pass,
        "avg_reward": (sum(rewards) / n) if n else 0.0,
        "pass_rate": (n_pass / n) if n else 0.0,
    }


def _bootstrap_executor_only_assets(output_root: Path) -> tuple[Path, Path]:
    from sage_tau2.organization import Organization

    bank_path = output_root / "skill_bank.json"
    org_path = output_root / "organization.json"
    if not bank_path.exists():
        write_json(bank_path, [])
    if not org_path.exists():
        Organization().save(org_path)
    return bank_path, org_path


def run_frozen_eval(
    *,
    checkpoint: Path | None,
    output_root: Path,
    domains: list[tuple[str, int]],
    model: str,
    user_model: str | None,
    seed: int,
    max_concurrency: int,
    max_inject_skills: int,
    allow_provisional_inject: bool,
    agent_name: str,
    task_split_name: str,
    llm_config_path: str | None,
    executor_only: bool = False,
    accepted_only: bool = False,
    dump_pro_trajectories: bool = True,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if executor_only:
        bank_path, org_path = _bootstrap_executor_only_assets(output_root)
        checkpoint_label = "executor_only"
    else:
        if checkpoint is None:
            raise SystemExit("--checkpoint is required unless --executor-only is set")
        checkpoint = checkpoint.resolve()
        src_bank = checkpoint / "skill_bank.json"
        src_org = checkpoint / "organization.json"
        if not src_bank.exists():
            raise SystemExit(f"Missing skill bank: {src_bank}")
        if not src_org.exists():
            raise SystemExit(f"Missing organization: {src_org}")

        bank_path = output_root / "skill_bank.json"
        org_path = output_root / "organization.json"
        if not bank_path.exists():
            shutil.copy2(src_bank, bank_path)
        if not org_path.exists():
            shutil.copy2(src_org, org_path)
        checkpoint_label = str(checkpoint)

    _apply_relay_from_llm_config(llm_config_path)
    _register_agent()

    user = user_model or model
    domain_rows: list[dict[str, Any]] = []
    for i, (domain, num_tasks) in enumerate(domains):
        save_to = f"sage_tau2_{output_root.name}_{domain}"
        if executor_only:
            mode = "executor_only"
        elif accepted_only:
            mode = "frozen_sage_accepted_only"
        else:
            mode = "frozen_sage"
        print(
            f"[sage_tau2:eval] domain={domain} split={task_split_name} "
            f"num_tasks={num_tasks} inject<={max_inject_skills} "
            f"mode={mode} checkpoint={Path(checkpoint_label).name}",
            flush=True,
        )
        payload = _run_domain_eval(
            domain=domain,
            num_tasks=num_tasks,
            model=model,
            user_model=user,
            seed=seed + i * 1000,
            max_concurrency=max_concurrency,
            save_to=save_to,
            skill_bank_path=bank_path,
            organization_path=org_path,
            max_inject_skills=max_inject_skills,
            allow_provisional_inject=allow_provisional_inject,
            agent_name=agent_name,
            task_split_name=task_split_name,
            executor_only=executor_only,
            accepted_only=accepted_only,
        )
        stats = _reward_stats(payload)
        row = {
            "domain": domain,
            "num_tasks": num_tasks,
            "save_to": save_to,
            **stats,
        }
        domain_rows.append(row)
        write_json(output_root / f"{domain}_summary.json", row)
        if dump_pro_trajectories:
            try:
                export_pro_trajectories(
                    results_payload=payload,
                    results_path=Path("data/simulations") / save_to / "results.json",
                    output_dir=output_root / "pro_trajectories" / domain,
                    domain=domain,
                    dispatch_log_path=output_root / "dispatch_journal.jsonl",
                )
            except Exception as exc:  # noqa: BLE001 — eval should still finish
                print(
                    f"[sage_tau2:eval] PRO trajectory dump failed for {domain}: {exc}",
                    flush=True,
                )
        print(
            f"[sage_tau2:eval] {domain} done "
            f"avg={stats['avg_reward']:.3f} "
            f"pass={stats['n_pass']}/{stats['n_scored']}",
            flush=True,
        )

    total_pass = sum(int(r["n_pass"]) for r in domain_rows)
    total_scored = sum(int(r["n_scored"]) for r in domain_rows)
    final = {
        "protocol": (
            "sage_tau2_executor_eval_v1"
            if executor_only
            else (
                "sage_tau2_frozen_eval_accepted_only_v1"
                if accepted_only
                else "sage_tau2_frozen_eval_v1"
            )
        ),
        "checkpoint": checkpoint_label,
        "executor_only": executor_only,
        "accepted_only": accepted_only,
        "model": model,
        "user_model": user,
        "task_split_name": task_split_name,
        "freeze_organization": not executor_only,
        "freeze_skill_bank": not executor_only,
        "n_domains": len(domain_rows),
        "n_tasks_requested": sum(n for _, n in domains),
        "n_scored": total_scored,
        "n_pass": total_pass,
        "micro_pass_rate": (total_pass / total_scored) if total_scored else 0.0,
        "macro_avg_reward": (
            sum(float(r["avg_reward"]) for r in domain_rows) / len(domain_rows)
            if domain_rows
            else 0.0
        ),
        "domains": domain_rows,
        "skill_bank_path": str(bank_path),
        "organization_path": str(org_path),
    }
    write_json(output_root / "eval_summary.json", final)
    print(
        f"[sage_tau2:eval] ALL done micro_pass={final['micro_pass_rate']:.3f} "
        f"({total_pass}/{total_scored}) macro_avg={final['macro_avg_reward']:.3f}",
        flush=True,
    )
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Frozen multi-domain eval of an evolved sage_tau2 checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Evolved run dir with skill_bank.json + organization.json",
    )
    parser.add_argument(
        "--executor-only",
        action="store_true",
        help="Bare Executor baseline: no dispatch, no skill inject, no checkpoint",
    )
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help=(
            "Frozen eval with probation_primary_quota=0: only accepted "
            "specialists may take primary; probation falls back to Executor"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Eval output dir under logs/sage_tau2/...",
    )
    parser.add_argument(
        "--domains",
        default="airline:50,retail:114,telecom:114",
        help="Comma list domain:num_tasks (default OOD ~278)",
    )
    parser.add_argument("--model", default="openai/gemini-2.5-flash")
    parser.add_argument("--user-model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--max-inject-skills", type=int, default=2)
    parser.add_argument(
        "--no-provisional-inject",
        action="store_true",
        help="Only inject verified/promoted skills",
    )
    parser.add_argument("--agent-name", default="sage_tau2")
    parser.add_argument(
        "--task-split-name",
        default="base",
        help="tau2 split name (base ≈ full domain pool used in Flash baseline)",
    )
    parser.add_argument("--llm-config", default=None)
    parser.add_argument(
        "--no-dump-pro-trajectories",
        action="store_true",
        help="Skip writing prompt/response/observation trajectories under output/pro_trajectories/",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).resolve()
    if "sage_mas" in output.parts and "sage_tau2" not in output.parts:
        raise SystemExit(
            f"Refusing to write under sage_mas logs: {output}. "
            "Use logs/sage_tau2/... instead."
        )

    domains: list[tuple[str, int]] = []
    for part in str(args.domains).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"Bad --domains entry (want domain:N): {part}")
        name, n_s = part.split(":", 1)
        domains.append((name.strip(), int(n_s)))
    if not domains:
        domains = list(DEFAULT_DOMAINS)

    if args.executor_only and args.accepted_only:
        raise SystemExit("--executor-only and --accepted-only are mutually exclusive")
    if not args.executor_only and not args.checkpoint:
        raise SystemExit("Provide --checkpoint or use --executor-only")

    max_inject_skills = 0 if args.executor_only else args.max_inject_skills
    allow_provisional_inject = (
        False if args.executor_only else not args.no_provisional_inject
    )

    run_frozen_eval(
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        output_root=output,
        domains=domains,
        model=args.model,
        user_model=args.user_model,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        max_inject_skills=max_inject_skills,
        allow_provisional_inject=allow_provisional_inject,
        agent_name=args.agent_name,
        task_split_name=args.task_split_name,
        llm_config_path=args.llm_config,
        executor_only=args.executor_only,
        accepted_only=args.accepted_only,
        dump_pro_trajectories=not args.no_dump_pro_trajectories,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
