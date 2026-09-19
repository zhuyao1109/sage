"""Online τ² evolution loop: collect → distill → credit → inject → optional org.

With ``freeze_organization: false``, each segment also runs capability
clustering + nominate/admit and may ADD_AGENT into ``organization.json``.

Episode start uses ``ExecutorDispatcher`` (ported from sage_mas) so admitted
specialists become sticky primary and receive only their assigned skills.

Run from the tau2-bench environment so ``tau2`` imports resolve:

```bash
cd ~/verl-agent/tau2-bench
export OPENAI_API_KEY=...
export OPENAI_API_BASE=...
PYTHONPATH=.. uv run python -m sage_tau2.runners.online_tau2 \\
  --config ../sage_tau2/configs/airline_evolve_10seg.yaml \\
  --output ../logs/sage_tau2/airline_evolve_10seg
```

This module never writes under ``logs/sage_mas`` and does not modify
``sage_mas`` / ALFWorld configs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sage_tau2.agent import create_sage_tau2_agent
from sage_tau2.credit import injectable_skills
from sage_tau2.pipeline import SegmentUpdateConfig, update_bank_from_results
from sage_tau2.runners.dump_pro_trajectories import export_pro_trajectories
from sage_tau2.success_mode import (
    merge_simulations_prefer_best,
    write_gold_fail_task_ids,
)
from sage_tau2.task_sampling import (
    TaskSchedule,
    sample_fixed_val_ids,
    sample_task_ids,
    validate_evolve_budget,
)
from sage_tau2.serialization import read_json, write_json
from sage_tau2.skill_bank import Tau2SkillBank


@dataclass(slots=True)
class OnlineConfig:
    domain: str = "airline"
    model: str = "openai/gemini-2.5-flash"
    user_model: str | None = None
    # Aligned with sage_mas online600: 60 games/segment (airline resamples ~50).
    segment_size: int = 60
    num_segments: int = 10
    seed: int = 42
    max_concurrency: int = 3
    max_inject_skills: int = 2
    allow_provisional_inject: bool = True
    freeze_organization: bool = True
    dump_pro_trajectories: bool = True
    min_support: int = 1
    require_success: bool = True
    min_protocol_len: int = 1
    max_new_skills: int = 16
    verify_score: float = 0.60
    prune_score: float = 0.20
    min_protocol_coverage: float = 0.5
    min_support_for_verified: int = 5
    # P1 credit attribution gates (mirror SegmentUpdateConfig defaults).
    credit_require_full_write_spine: bool = True
    credit_per_episode_inject: bool = True
    # Align sage_mas online600: merge last N prior segment evolve traces.
    distill_prior_segments: int = 2
    cluster_novelty_threshold: float = 0.05
    nominate_min_cluster_support: int = 1
    nominate_min_utility: float = 0.0
    max_new_agents_per_round: int = 1
    dispatch_only_new_agents: bool = True
    probation_games: int = 3
    enable_spec_vs_exec: bool = False
    admit_num_tasks: int = 8
    admit_min_advantage: float = 0.0
    admit_accept_ties: bool = False
    admit_on_fail_action: str = "remove"
    admit_min_utility: float = 0.60
    admit_min_support: int = 3
    allow_provisional_org_edits: bool = False
    no_spec_mode: str = "editor_commit"
    require_same_domain_for_nominate: bool = True
    probation_min_games: int = 3
    probation_min_wins: int = 2
    remove_after_rejected_windows: int = 2
    task_split_name: str = "train"
    allow_task_resampling: bool = False
    # Fixed val set: sample once, re-run after every train segment (no distill).
    val_size: int = 0
    val_split_name: str = "train"
    val_exclude_from_train: bool = True
    val_seed: int | None = None
    dump_val_pro_trajectories: bool = True
    # After each train collect: re-run write-gold fails up to retry_k times.
    retry_write_fails: bool = False
    retry_k: int = 2
    # On write-fail retries, force Executor primary (avoid weak specialists).
    retry_force_executor: bool = True
    # Feed environment action_check verdict back to agent on retry (verdict-
    # guided retry vs blind resampling). Pure environment signal, not expert
    # rules.
    retry_verdict_feedback: bool = False
    llm_config_path: str | None = None
    agent_name: str = "sage_tau2"
    # Opt-in seed skills: false | true/"auto" | "telecom" | "airline"
    seed_skills: bool | str | None = None
    inject_same_domain_only: bool = True
    # true = episode-scope retrieval; false = fixed top-k prefix (ablation).
    require_scope_match: bool = True
    # false = no ExecutorDispatcher / meta routing (force Executor).
    enable_executor_dispatch: bool = True
    probation_primary_quota: int = 0
    # Optional fixed task list (overrides train sampling for this run).
    task_ids: list[str] | None = None
    # Stratify train pool by PERSONA × bug-count so each segment is difficulty-balanced.
    balance_difficulty: bool = False


def load_online_config(path: str | Path) -> OnlineConfig:
    from dataclasses import fields as dc_fields

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in dc_fields(OnlineConfig)}
    payload = {k: v for k, v in raw.items() if k in known}
    return OnlineConfig(**payload)


def _apply_relay_from_llm_config(llm_config_path: str | Path | None) -> None:
    if not llm_config_path:
        return
    path = Path(llm_config_path)
    if not path.exists():
        return
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    openai_cfg = cfg.get("openai") or {}
    # yaml wins. A stale shell OPENAI_API_KEY otherwise overrides llm_config
    # and the relay returns Invalid token.
    if openai_cfg.get("api_key"):
        os.environ["OPENAI_API_KEY"] = str(openai_cfg["api_key"])
    base = openai_cfg.get("base_url")
    if base:
        os.environ["OPENAI_API_BASE"] = str(base)
        os.environ["OPENAI_BASE_URL"] = str(base)
    # Zero-cost stub so LiteLLM completion_cost does not ERROR on relay models.
    try:
        import litellm

        for name in (
            "gemini-2.5-flash",
            "openai/gemini-2.5-flash",
            "gpt-4o-mini",
            "openai/gpt-4o-mini",
        ):
            litellm.model_cost.setdefault(
                name,
                {
                    "max_tokens": 8192,
                    "max_input_tokens": 1_048_576,
                    "max_output_tokens": 8192,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                    "litellm_provider": "openai",
                    "mode": "chat",
                },
            )
    except Exception:
        pass


def _register_agent() -> None:
    from tau2.registry import registry

    if "sage_tau2" not in registry.get_agents():
        registry.register_agent_factory(create_sage_tau2_agent, "sage_tau2")


def _run_segment_collection(
    *,
    domain: str,
    agent: str,
    model: str,
    user_model: str,
    seed: int,
    max_concurrency: int,
    save_to: str,
    skill_bank_path: Path,
    organization_path: Path | None,
    max_inject_skills: int,
    allow_provisional_inject: bool,
    task_ids: list[str] | None = None,
    num_tasks: int | None = None,
    task_split_name: str = "train",
    force_primary: str | None = None,
    require_scope_match: bool = True,
    enable_executor_dispatch: bool = True,
    prior_failure_hints_path: Path | None = None,
) -> dict[str, Any]:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import run_domain

    os.environ["SAGE_TAU2_SKILL_BANK"] = str(skill_bank_path)
    os.environ["SAGE_TAU2_MAX_SKILLS"] = str(max_inject_skills)
    os.environ["SAGE_TAU2_INJECT_PROVISIONAL"] = (
        "1" if allow_provisional_inject else "0"
    )
    os.environ["SAGE_TAU2_INJECT_SAME_DOMAIN_ONLY"] = "1"
    os.environ["SAGE_TAU2_REQUIRE_SCOPE_MATCH"] = (
        "1" if require_scope_match else "0"
    )
    os.environ["SAGE_TAU2_DOMAIN"] = str(domain)
    if organization_path is not None:
        os.environ["SAGE_TAU2_ORG_PATH"] = str(organization_path)
    else:
        os.environ.pop("SAGE_TAU2_ORG_PATH", None)
    dispatch_log = Path(skill_bank_path).resolve().parent / "dispatch_journal.jsonl"
    os.environ["SAGE_TAU2_DISPATCH_LOG"] = str(dispatch_log)
    force = str(force_primary or "").strip() or None
    if not enable_executor_dispatch and not force:
        force = "Executor"
    if force:
        os.environ["SAGE_TAU2_FORCE_PRIMARY"] = force
    else:
        os.environ.pop("SAGE_TAU2_FORCE_PRIMARY", None)
    llm_args_agent: dict[str, Any] = {
        "skill_bank_path": str(skill_bank_path),
        "max_inject_skills": max_inject_skills,
        "inject_provisional": allow_provisional_inject,
        "inject_same_domain_only": True,
        "require_scope_match": require_scope_match,
        "enable_executor_dispatch": bool(enable_executor_dispatch),
        "dispatch_log_path": str(dispatch_log),
    }
    if organization_path is not None:
        llm_args_agent["organization_path"] = str(organization_path)
    if force:
        llm_args_agent["force_primary"] = force
    if prior_failure_hints_path is not None:
        llm_args_agent["prior_failure_hints_path"] = str(prior_failure_hints_path)
    config = TextRunConfig(
        domain=domain,
        agent=agent,
        llm_agent=model,
        llm_user=user_model,
        num_trials=1,
        task_ids=task_ids,
        num_tasks=num_tasks if not task_ids else None,
        task_split_name=task_split_name,
        seed=seed,
        max_concurrency=max_concurrency,
        save_to=save_to,
        # Resume silently after crashes/restarts; tau2 otherwise prompts
        # interactively when the save file exists, blocking the pipeline.
        auto_resume=True,
        llm_args_agent=llm_args_agent,
    )
    try:
        results = run_domain(config)
    finally:
        os.environ.pop("SAGE_TAU2_FORCE_PRIMARY", None)
    # Prefer the saved JSON (stable schema) when present.
    sim_path = Path("data/simulations") / save_to / "results.json"
    if sim_path.exists():
        return read_json(sim_path)
    # Fallback: serialize Results object if available.
    if hasattr(results, "model_dump"):
        return results.model_dump()
    if isinstance(results, dict):
        return results
    raise RuntimeError(f"Could not load segment results for save_to={save_to}")


def _reward_stats(results_payload: dict[str, Any]) -> dict[str, Any]:
    sims = results_payload.get("simulations") or []
    rewards: list[float] = []
    per_task: list[dict[str, Any]] = []
    for sim in sims:
        if not isinstance(sim, dict):
            continue
        ri = sim.get("reward_info") or {}
        r = ri.get("reward")
        if not isinstance(r, (int, float)):
            continue
        reward = float(r)
        rewards.append(reward)
        tid = sim.get("task_id") or (sim.get("task") or {}).get("id")
        db = (ri.get("db_check") or {}).get("db_match")
        per_task.append(
            {
                "task_id": tid,
                "reward": reward,
                "pass": reward >= 1.0,
                "db_match": db,
            }
        )
    n = len(rewards)
    n_pass = sum(1 for r in rewards if r >= 1.0)
    return {
        "n_simulations": len(sims),
        "n_scored": n,
        "n_pass": n_pass,
        "avg_reward": (sum(rewards) / n) if n else 0.0,
        "pass_rate": (n_pass / n) if n else 0.0,
        "per_task": per_task,
    }


def _extract_verdict_hints(payload: dict[str, Any]) -> dict[str, str]:
    """Build per-task verdict feedback from failed action_checks.

    For each failed write action_check, report the expected tool name and
    arguments (from the environment's own validation). This is pure
    environment signal — no expert rules or prescribed algorithms.

    Returns ``{task_id: feedback_text}``.
    """
    hints: dict[str, str] = {}
    for sim in payload.get("simulations") or []:
        if not isinstance(sim, dict):
            continue
        ri = sim.get("reward_info") or {}
        reward = float(ri.get("reward") or 0)
        if reward >= 1.0:
            continue  # passed, no feedback needed
        tid = str(sim.get("task_id") or "").strip()
        if not tid:
            continue
        checks = ri.get("action_checks") or []
        failed_writes = []
        for c in checks:
            if (c.get("action_reward") or 0) >= 1.0:
                continue
            act = c.get("action") or {}
            name = str(act.get("name") or "").strip()
            if not name:
                continue
            args = act.get("arguments") or {}
            # Only report agent-side write tools (not user-side guides).
            if str(act.get("requestor") or "assistant") != "assistant":
                continue
            args_str = ", ".join(f"{k}={v}" for k, v in sorted(args.items()))
            failed_writes.append(f"{name}({args_str})")
        if not failed_writes:
            continue
        lines = [
            "[Previous attempt feedback]",
            "The following required agent write actions were not matched in your prior attempt:",
        ]
        for i, w in enumerate(failed_writes, 1):
            lines.append(f"  {i}. {w}")
        lines.append(
            "Re-attempt the episode. Ensure your tool calls use the expected arguments. "
            "Do not guess — look up the correct entity ids before calling write tools."
        )
        hints[tid] = "\n".join(lines)
    return hints


def retry_write_gold_fails(
    *,
    results_payload: dict[str, Any],
    retry_k: int,
    domain: str,
    agent: str,
    model: str,
    user_model: str,
    seed: int,
    max_concurrency: int,
    save_to_prefix: str,
    skill_bank_path: Path,
    organization_path: Path | None,
    max_inject_skills: int,
    allow_provisional_inject: bool,
    task_split_name: str,
    seg_dir: Path,
    force_primary: str | None = "Executor",
    require_scope_match: bool = True,
    enable_executor_dispatch: bool = True,
    verdict_feedback: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-collect write-gold fails up to ``retry_k`` rounds; keep best sim per task.

    When ``verdict_feedback`` is True, each retry round feeds the agent the
    environment's own action_check verdict from the prior attempt (expected
    tool name + arguments), without prescribing the algorithm. This converts
    blind resampling into verdict-guided retry — pure environment signal, not
    expert scaffolding.
    """
    k = max(0, int(retry_k))
    payload = results_payload
    force = str(force_primary or "").strip() or None
    log: dict[str, Any] = {
        "retry_k": k,
        "force_primary": force,
        "rounds": [],
        "initial_write_gold_fails": write_gold_fail_task_ids(payload),
    }
    if k <= 0:
        return payload, log

    pending = list(log["initial_write_gold_fails"])
    for attempt in range(1, k + 1):
        if not pending:
            break
        retry_save = f"{save_to_prefix}_wretry{attempt}"
        print(
            f"[sage_tau2] write-fail retry {attempt}/{k} n={len(pending)} "
            f"force_primary={force or '-'} verdict={verdict_feedback} ids={pending}",
            flush=True,
        )
        # Build per-task verdict feedback from the current best payload.
        hints_path: Path | None = None
        if verdict_feedback:
            hints = _extract_verdict_hints(payload)
            # Only keep hints for tasks we're about to retry.
            hints = {tid: hints[tid] for tid in pending if tid in hints}
            if hints:
                retry_dir = seg_dir / f"write_retry_{attempt:02d}"
                retry_dir.mkdir(parents=True, exist_ok=True)
                hints_path = retry_dir / "prior_failure_hints.json"
                write_json(hints_path, hints)
                print(
                    f"[sage_tau2] verdict feedback: {len(hints)}/{len(pending)} "
                    f"tasks get prior-attempt hints",
                    flush=True,
                )
        retry_payload = _run_segment_collection(
            domain=domain,
            agent=agent,
            model=model,
            user_model=user_model,
            task_ids=pending,
            num_tasks=None,
            seed=int(seed) + 7000 + attempt,
            max_concurrency=max_concurrency,
            save_to=retry_save,
            skill_bank_path=skill_bank_path,
            organization_path=organization_path,
            max_inject_skills=max_inject_skills,
            allow_provisional_inject=allow_provisional_inject,
            task_split_name=task_split_name,
            force_primary=force,
            require_scope_match=require_scope_match,
            enable_executor_dispatch=enable_executor_dispatch,
            prior_failure_hints_path=hints_path,
        )
        retry_dir = seg_dir / f"write_retry_{attempt:02d}"
        retry_dir.mkdir(parents=True, exist_ok=True)
        write_json(retry_dir / "raw_results.json", retry_payload)
        write_json(
            retry_dir / "task_ids.json",
            {"attempt": attempt, "task_ids": pending, "force_primary": force},
        )
        before_fails = set(pending)
        payload = merge_simulations_prefer_best(payload, retry_payload)
        after_fails = write_gold_fail_task_ids(payload)
        rescued = sorted(before_fails - set(after_fails))
        round_stats = _reward_stats(retry_payload)
        log["rounds"].append(
            {
                "attempt": attempt,
                "retried": list(pending),
                "rescued": rescued,
                "replaced": list(payload.get("_retry_replaced_task_ids") or []),
                "retry_avg_reward": round_stats.get("avg_reward"),
                "retry_n_pass": round_stats.get("n_pass"),
                "remaining_write_gold_fails": after_fails,
                "force_primary": force,
            }
        )
        pending = [tid for tid in after_fails if tid in before_fails]

    log["final_write_gold_fails"] = write_gold_fail_task_ids(payload)
    log["n_rescued"] = len(set(log["initial_write_gold_fails"]) - set(log["final_write_gold_fails"]))
    payload = dict(payload)
    payload.pop("_retry_replaced_task_ids", None)
    return payload, log


def run_online(config: OnlineConfig, output_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bank_path = output_root / "skill_bank.json"
    state_path = output_root / "online_state.json"
    org_path = output_root / "organization.json"
    cluster_path = output_root / "skill_clusters.json"
    bank = Tau2SkillBank(bank_path)

    _apply_relay_from_llm_config(config.llm_config_path)
    _register_agent()

    from sage_tau2.seed_skills import ensure_seed_skills_in_bank, resolve_seed_skills

    seed_pack = resolve_seed_skills(config.seed_skills, domain=config.domain)
    if seed_pack:
        ensure_seed_skills_in_bank(bank_path, skills=seed_pack)
        bank = Tau2SkillBank(bank_path)
        print(
            f"[sage_tau2] seeded {len(seed_pack)} skills into {bank_path} "
            f"({', '.join(s.skill_name for s in seed_pack)})",
            flush=True,
        )

    if not config.freeze_organization:
        from sage_tau2.organization import Organization

        Organization.load(org_path)  # ensure Executor exists on disk

    user_model = config.user_model or config.model
    state: dict[str, Any] = {
        "protocol": "sage_tau2_online_v1",
        "domain": config.domain,
        "model": config.model,
        "task_split_name": config.task_split_name,
        "allow_task_resampling": config.allow_task_resampling,
        "segments": [],
        "completed_segments": 0,
        "freeze_organization": config.freeze_organization,
    }
    if state_path.exists():
        state = read_json(state_path)
        completed = int(state.get("completed_segments") or 0)
    else:
        completed = 0

    # Fixed val: sample once from train (or val_split), persist, optionally hold out.
    val_ids: list[str] = []
    if int(config.val_size) > 0:
        raw_val = state.get("val_task_ids")
        if isinstance(raw_val, list) and raw_val:
            val_ids = [str(x) for x in raw_val]
        else:
            val_seed = (
                int(config.val_seed)
                if config.val_seed is not None
                else int(config.seed) + 7919
            )
            val_ids = sample_fixed_val_ids(
                domain=config.domain,
                split_name=config.val_split_name,
                val_size=int(config.val_size),
                seed=val_seed,
            )
            state["val_task_ids"] = val_ids
            state["val_split_name"] = config.val_split_name
            state["val_exclude_from_train"] = bool(config.val_exclude_from_train)
            state["val_seed"] = val_seed
            write_json(
                output_root / "val_task_ids.json",
                {
                    "domain": config.domain,
                    "split": config.val_split_name,
                    "seed": val_seed,
                    "exclude_from_train": bool(config.val_exclude_from_train),
                    "task_ids": val_ids,
                },
            )
            write_json(state_path, state)
            print(
                f"[sage_tau2] fixed val n={len(val_ids)} from {config.val_split_name} "
                f"exclude_from_train={config.val_exclude_from_train} ids={val_ids}",
                flush=True,
            )

    quotas = {config.domain: int(config.segment_size)}
    if not config.task_ids:
        validate_evolve_budget(
            segment_size=int(config.segment_size),
            num_segments=int(config.num_segments),
            quotas=quotas,
            split_name=config.task_split_name,
            allow_task_resampling=config.allow_task_resampling,
        )

    schedule: TaskSchedule | None = None
    if config.task_ids:
        state["fixed_task_ids"] = list(config.task_ids)
        write_json(state_path, state)
        print(
            f"[sage_tau2] using fixed task_ids n={len(config.task_ids)}",
            flush=True,
        )
    elif not config.allow_task_resampling:
        raw_sched = state.get("task_schedule")
        if isinstance(raw_sched, dict) and raw_sched.get("order"):
            schedule = TaskSchedule.from_state(raw_sched)
        else:
            exclude: dict[str, list[str]] | None = None
            if val_ids and config.val_exclude_from_train:
                exclude = {config.domain: list(val_ids)}
            schedule = TaskSchedule.create(
                domains=[config.domain],
                split_name=config.task_split_name,
                seed=int(config.seed),
                exclude_ids=exclude,
                balance_difficulty=bool(config.balance_difficulty),
            )
            if config.balance_difficulty:
                print(
                    "[sage_tau2] task schedule: balance_difficulty=True "
                    "(PERSONA × bug-bucket round-robin)",
                    flush=True,
                )
            state["task_schedule"] = schedule.to_state()
            write_json(state_path, state)

    segment_cfg = SegmentUpdateConfig(
        min_support=config.min_support,
        require_success=config.require_success,
        min_protocol_len=config.min_protocol_len,
        max_new_skills=config.max_new_skills,
        max_inject_skills=config.max_inject_skills,
        allow_provisional_inject=config.allow_provisional_inject,
        freeze_organization=config.freeze_organization,
        verify_score=config.verify_score,
        prune_score=config.prune_score,
        min_protocol_coverage=config.min_protocol_coverage,
        distill_prior_segments=int(config.distill_prior_segments),
        min_support_for_verified=int(config.min_support_for_verified),
        credit_require_full_write_spine=bool(config.credit_require_full_write_spine),
        credit_per_episode_inject=bool(config.credit_per_episode_inject),
        cluster_novelty_threshold=config.cluster_novelty_threshold,
        nominate_min_cluster_support=config.nominate_min_cluster_support,
        nominate_min_utility=config.nominate_min_utility,
        max_new_agents_per_round=config.max_new_agents_per_round,
        dispatch_only_new_agents=config.dispatch_only_new_agents,
        probation_games=config.probation_games,
        enable_spec_vs_exec=config.enable_spec_vs_exec,
        admit_num_tasks=config.admit_num_tasks,
        admit_min_advantage=config.admit_min_advantage,
        admit_accept_ties=config.admit_accept_ties,
        admit_on_fail_action=config.admit_on_fail_action,
        admit_min_utility=config.admit_min_utility,
        admit_min_support=config.admit_min_support,
        allow_provisional_org_edits=config.allow_provisional_org_edits,
        no_spec_mode=config.no_spec_mode,
        require_same_domain_for_nominate=config.require_same_domain_for_nominate,
        probation_min_games=config.probation_min_games,
        probation_min_wins=config.probation_min_wins,
        remove_after_rejected_windows=config.remove_after_rejected_windows,
        dispatch_log_path=str(output_root / "dispatch_journal.jsonl"),
    )

    for seg_idx in range(completed, config.num_segments):
        seg_id = seg_idx + 1
        seg_dir = output_root / f"segment_{seg_id:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        save_to = f"sage_tau2_{output_root.name}_seg{seg_id:03d}"
        seed = int(config.seed) + seg_idx * 1000

        task_sample_path = seg_dir / "task_sample.json"
        if config.task_ids:
            # Fixed list: consume without replacement across segments.
            start = seg_idx * int(config.segment_size)
            end = start + int(config.segment_size)
            task_ids = [str(t) for t in list(config.task_ids)[start:end] if str(t).strip()]
            schedule = None
        else:
            prior_sample: dict[str, Any] | None = None
            if task_sample_path.exists():
                try:
                    prior_sample = json.loads(task_sample_path.read_text())
                except Exception:
                    prior_sample = None
            if prior_sample and prior_sample.get("task_ids"):
                # Mid-segment restart: this segment was already sampled by the
                # previous attempt and the schedule cursor advanced past its
                # slice back then. Re-sampling would consume the NEXT slice,
                # skipping tasks and desyncing from the existing simulation
                # save (try_resume then aborts with "tasks were removed").
                task_ids = [str(t) for t in prior_sample["task_ids"]]
            else:
                task_map, schedule = sample_task_ids(
                    quotas=quotas,
                    split_name=config.task_split_name,
                    seed=seed,
                    schedule=schedule,
                    allow_task_resampling=config.allow_task_resampling,
                )
                task_ids = task_map.get(config.domain) or []
        if not task_ids:
            print(
                f"[sage_tau2] train pool exhausted before segment {seg_id}; stopping.",
                flush=True,
            )
            break
        if schedule is not None:
            state["task_schedule"] = schedule.to_state()
            write_json(state_path, state)
        write_json(
            task_sample_path,
            {
                "segment": seg_id,
                "seed": seed,
                "split": config.task_split_name,
                "task_ids": task_ids,
                "fixed_task_ids": bool(config.task_ids),
            },
        )

        offered = injectable_skills(
            bank.active(),
            max_skills=config.max_inject_skills,
            allow_provisional=config.allow_provisional_inject,
        )
        injected_ids = [s.skill_id for s in offered]
        write_json(
            seg_dir / "injection.json",
            {
                "skill_ids": injected_ids,
                "skill_names": [s.skill_name for s in offered],
            },
        )

        print(
            f"[sage_tau2] segment {seg_id}/{config.num_segments} "
            f"domain={config.domain} n={len(task_ids)} inject={len(offered)} "
            f"freeze_org={config.freeze_organization} "
            f"split={config.task_split_name} resampling={config.allow_task_resampling}",
            flush=True,
        )
        results_payload = _run_segment_collection(
            domain=config.domain,
            agent=config.agent_name,
            model=config.model,
            user_model=user_model,
            task_ids=task_ids,
            num_tasks=None,
            seed=seed,
            max_concurrency=config.max_concurrency,
            save_to=save_to,
            skill_bank_path=bank_path,
            organization_path=None if config.freeze_organization else org_path,
            max_inject_skills=config.max_inject_skills,
            allow_provisional_inject=config.allow_provisional_inject,
            task_split_name=config.task_split_name,
            require_scope_match=config.require_scope_match,
            enable_executor_dispatch=config.enable_executor_dispatch,
        )
        write_json(seg_dir / "raw_results_initial.json", results_payload)

        retry_log: dict[str, Any] | None = None
        if config.retry_write_fails and int(config.retry_k) > 0:
            results_payload, retry_log = retry_write_gold_fails(
                results_payload=results_payload,
                retry_k=int(config.retry_k),
                domain=config.domain,
                agent=config.agent_name,
                model=config.model,
                user_model=user_model,
                seed=seed,
                max_concurrency=config.max_concurrency,
                save_to_prefix=save_to,
                skill_bank_path=bank_path,
                organization_path=None if config.freeze_organization else org_path,
                max_inject_skills=config.max_inject_skills,
                allow_provisional_inject=config.allow_provisional_inject,
                task_split_name=config.task_split_name,
                seg_dir=seg_dir,
                force_primary=(
                    "Executor" if config.retry_force_executor else None
                ),
                require_scope_match=config.require_scope_match,
                enable_executor_dispatch=config.enable_executor_dispatch,
                verdict_feedback=bool(config.retry_verdict_feedback),
            )
            write_json(seg_dir / "write_retry_summary.json", retry_log)
            print(
                f"[sage_tau2] segment {seg_id} write-retry "
                f"rescued={retry_log.get('n_rescued', 0)} "
                f"remain={len(retry_log.get('final_write_gold_fails') or [])}",
                flush=True,
            )

        write_json(seg_dir / "raw_results.json", results_payload)
        if config.dump_pro_trajectories:
            try:
                export_pro_trajectories(
                    results_payload=results_payload,
                    results_path=Path("data/simulations") / save_to / "results.json",
                    output_dir=seg_dir / "pro_trajectories",
                    domain=config.domain,
                    dispatch_log_path=output_root / "dispatch_journal.jsonl",
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[sage_tau2] PRO dump failed segment {seg_idx}: {exc}",
                    flush=True,
                )

        # Reload bank from disk in case of concurrent writes (single-process ok).
        bank = Tau2SkillBank(bank_path)
        probe = None
        if (not config.freeze_organization) and config.enable_spec_vs_exec:
            from sage_tau2.admission_probe import make_probe_callback

            probe = make_probe_callback(
                domain=config.domain,
                model=config.model,
                user_model=user_model,
                skill_bank_path=bank_path,
                organization_path=org_path,
                seed=seed + 17,
                max_concurrency=max(1, min(4, int(config.max_concurrency))),
                agent_name=config.agent_name,
                probe_dir=seg_dir / "admission_probes",
            )
        summary = update_bank_from_results(
            results_payload=results_payload,
            domain=config.domain,
            bank=bank,
            injected_skill_ids=injected_ids,
            config=segment_cfg,
            segment_dir=seg_dir,
            segment_index=seg_id,
            organization_path=None if config.freeze_organization else org_path,
            cluster_archive_path=None if config.freeze_organization else cluster_path,
            spec_vs_exec_probe=probe,
        )

        val_summary: dict[str, Any] | None = None
        if val_ids:
            val_dir = seg_dir / "val"
            val_dir.mkdir(parents=True, exist_ok=True)
            val_save = f"sage_tau2_{output_root.name}_seg{seg_id:03d}_val"
            val_seed = int(config.seed) + seg_idx * 1000 + 333
            print(
                f"[sage_tau2] segment {seg_id} val retest n={len(val_ids)}",
                flush=True,
            )
            # Reload bank after evolve so val sees updated skills/org.
            bank = Tau2SkillBank(bank_path)
            val_payload = _run_segment_collection(
                domain=config.domain,
                agent=config.agent_name,
                model=config.model,
                user_model=user_model,
                task_ids=val_ids,
                num_tasks=None,
                seed=val_seed,
                max_concurrency=config.max_concurrency,
                save_to=val_save,
                skill_bank_path=bank_path,
                organization_path=None if config.freeze_organization else org_path,
                max_inject_skills=config.max_inject_skills,
                allow_provisional_inject=config.allow_provisional_inject,
                task_split_name=config.val_split_name,
                require_scope_match=config.require_scope_match,
                enable_executor_dispatch=config.enable_executor_dispatch,
            )
            write_json(val_dir / "raw_results.json", val_payload)
            val_summary = _reward_stats(val_payload)
            write_json(val_dir / "val_summary.json", val_summary)
            if config.dump_val_pro_trajectories:
                try:
                    export_pro_trajectories(
                        results_payload=val_payload,
                        results_path=Path("data/simulations") / val_save / "results.json",
                        output_dir=val_dir / "pro_trajectories",
                        domain=config.domain,
                        dispatch_log_path=output_root / "dispatch_journal.jsonl",
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[sage_tau2] val PRO dump failed segment {seg_id}: {exc}",
                        flush=True,
                    )
            print(
                f"[sage_tau2] segment {seg_id} val "
                f"avg={val_summary['avg_reward']:.3f} "
                f"pass={val_summary['n_pass']}/{val_summary['n_scored']}",
                flush=True,
            )

        seg_row: dict[str, Any] = {
            "segment": seg_id,
            "save_to": save_to,
            "summary": summary,
        }
        if retry_log is not None:
            seg_row["write_retry"] = {
                "n_initial_fails": len(retry_log.get("initial_write_gold_fails") or []),
                "n_rescued": retry_log.get("n_rescued"),
                "final_write_gold_fails": retry_log.get("final_write_gold_fails"),
            }
        if val_summary is not None:
            seg_row["val"] = val_summary
        state.setdefault("segments", []).append(seg_row)
        state["completed_segments"] = seg_id
        write_json(state_path, state)
        org_note = ""
        edit = summary.get("organization_edit") or {}
        if edit.get("accepted_agents"):
            org_note = f" add_agents={edit['accepted_agents']}"
        print(
            f"[sage_tau2] segment {seg_id} done "
            f"avg_reward={summary['avg_reward']:.3f} "
            f"new_skills={summary['n_new_skills_distilled']} "
            f"bank={summary['n_active_skills']}"
            f"{org_note}",
            flush=True,
        )

    # Final summary
    rewards = [
        float(seg["summary"]["avg_reward"])
        for seg in state.get("segments") or []
        if isinstance(seg, dict) and isinstance(seg.get("summary"), dict)
    ]
    val_rewards = [
        float(seg["val"]["avg_reward"])
        for seg in state.get("segments") or []
        if isinstance(seg, dict) and isinstance(seg.get("val"), dict)
    ]
    specialists: list[str] = []
    if not config.freeze_organization and org_path.exists():
        from sage_tau2.organization import Organization

        specialists = [a.name for a in Organization.load(org_path).specialists()]
    final = {
        "protocol": "sage_tau2_online_v1",
        "domain": config.domain,
        "model": config.model,
        "num_segments": config.num_segments,
        "segment_size": config.segment_size,
        "freeze_organization": config.freeze_organization,
        "mean_segment_reward": (
            sum(rewards) / len(rewards) if rewards else 0.0
        ),
        "val_task_ids": state.get("val_task_ids") or val_ids,
        "mean_val_reward": (
            sum(val_rewards) / len(val_rewards) if val_rewards else None
        ),
        "skill_bank_path": str(bank_path),
        "n_skills": len(Tau2SkillBank(bank_path).skills),
        "organization_path": str(org_path) if not config.freeze_organization else None,
        "specialists": specialists,
        "segments": state.get("segments"),
    }
    write_json(output_root / "online_summary.json", final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAGE-τ² online evolution runner")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config under sage_tau2/configs/",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory (use logs/sage_tau2/...; never logs/sage_mas)",
    )
    parser.add_argument(
        "--llm-config",
        default=None,
        help="Optional verl-agent examples/prompt_agent/llm_config.yaml for relay",
    )
    parser.add_argument(
        "--no-dump-pro-trajectories",
        action="store_true",
        help="Skip writing PRO trajectories under each segment_*/pro_trajectories/",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).resolve()
    if "sage_mas" in output.parts and "sage_tau2" not in output.parts:
        raise SystemExit(
            f"Refusing to write under sage_mas logs: {output}. "
            "Use logs/sage_tau2/... instead."
        )

    config = load_online_config(args.config)
    if args.no_dump_pro_trajectories:
        config.dump_pro_trajectories = False
    if args.llm_config:
        config.llm_config_path = args.llm_config
    elif config.llm_config_path is None:
        # Default to sibling verl-agent llm_config if present.
        default_llm = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "prompt_agent"
            / "llm_config.yaml"
        )
        if default_llm.exists():
            config.llm_config_path = str(default_llm)

    # Ensure tau2 cwd has data/ when invoked from repo root.
    tau2_root = Path(__file__).resolve().parents[2] / "tau2-bench"
    if (tau2_root / "data").exists():
        os.chdir(tau2_root)
        src = str(tau2_root / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        parent = str(Path(__file__).resolve().parents[2])
        if parent not in sys.path:
            sys.path.insert(0, parent)

    run_online(config, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
