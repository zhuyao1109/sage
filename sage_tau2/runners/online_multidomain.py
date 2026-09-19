"""Multi-domain continue evolution: proportional train sample → distill/admit.

Default: 100 tasks with train weights airline:retail:telecom = 30:74:74
→ quotas 17 / 42 / 41 from official ``train`` splits.

```bash
cd ~/verl-agent/tau2-bench
PYTHONPATH=.. uv run python -m sage_tau2.runners.online_multidomain \\
  --config ../sage_tau2/configs/multidomain_continue_100.yaml \\
  --checkpoint ../logs/sage_tau2/airline_evolve_10seg \\
  --output ../logs/sage_tau2/multidomain_continue_100 \\
  --llm-config ../examples/prompt_agent/llm_config.yaml
```
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from sage_tau2.injection import offer_skills_for_domain
from sage_tau2.pipeline import SegmentUpdateConfig, update_bank_from_results
from sage_tau2.runners.dump_pro_trajectories import export_pro_trajectories
from sage_tau2.runners.online_tau2 import (
    _apply_relay_from_llm_config,
    _register_agent,
    _reward_stats,
)
from sage_tau2.serialization import read_json, write_json
from sage_tau2.skill_bank import Tau2SkillBank
from sage_tau2.task_sampling import (
    TaskSchedule,
    sample_fixed_val_ids,
    sample_task_ids,
    validate_evolve_budget,
)


@dataclass(slots=True)
class MultiDomainConfig:
    model: str = "openai/gemini-2.5-flash"
    user_model: str | None = None
    seed: int = 42
    max_concurrency: int = 3
    # Align with sage_mas online600 max_injected_skills: 2
    max_inject_skills: int = 2
    allow_provisional_inject: bool = True
    freeze_organization: bool = False
    # One segment size; total evolve budget ≈ segment_size * num_segments.
    segment_size: int = 100
    num_segments: int = 1
    # Backward-compatible alias used when segment_size unset in older configs.
    total_tasks: int | None = None
    task_split_name: str = "train"
    # Official train protocol: each task id at most once per evolve run.
    allow_task_resampling: bool = False
    domain_weights: dict[str, int] | None = None
    # Prefer same-domain skills for Executor inject (avoid airline→telecom pollution).
    inject_same_domain_only: bool = True
    # After each domain collect, write ALFWorld-like PRO trajectories.
    dump_pro_trajectories: bool = True
    # Fixed multi-domain val: total size allocated by domain_weights; retest after evolve.
    val_size: int = 0
    val_split_name: str = "train"
    val_exclude_from_train: bool = True
    val_seed: int | None = None
    dump_val_pro_trajectories: bool = True
    # Distill / credit gates
    min_support: int = 1
    require_success: bool = True
    min_protocol_len: int = 1
    max_new_skills: int = 16
    verify_score: float = 0.60
    prune_score: float = 0.20
    min_protocol_coverage: float = 0.5
    # Credit attribution: use requires the skill's full write spine present.
    credit_require_full_write_spine: bool = True
    # Credit attribution: per-episode injected ids from the dispatch journal
    # (fallback: segment-level offer set for legacy/offline runs).
    credit_per_episode_inject: bool = True
    min_support_for_verified: int = 5
    # Align sage_mas online600: merge last N prior segment evolve traces.
    distill_prior_segments: int = 2
    cluster_novelty_threshold: float = 0.05
    nominate_min_cluster_support: int = 1
    nominate_min_utility: float = 0.0
    max_new_agents_per_round: int = 1
    dispatch_only_new_agents: bool = True
    probation_games: int = 3
    # Match actor_promotion_probe: false
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
    # Evolve: only accepted specialists take primary (probation shadow/quota off).
    probation_primary_quota: int = 0
    llm_config_path: str | None = None
    agent_name: str = "sage_tau2"


def load_config(path: str | Path) -> MultiDomainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {f.name for f in fields(MultiDomainConfig)}
    payload = {k: v for k, v in raw.items() if k in known}
    return MultiDomainConfig(**payload)


def allocate_quotas(weights: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation."""
    names = list(weights.keys())
    w = [max(0, int(weights[n])) for n in names]
    s = sum(w) or 1
    raw = [total * x / s for x in w]
    base = [int(x) for x in raw]
    rem = total - sum(base)
    order = sorted(range(len(names)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return {names[i]: base[i] for i in range(len(names))}


def _offer_skills_for_domain(
    bank: Tau2SkillBank,
    *,
    domain: str,
    max_skills: int,
    allow_provisional: bool,
    same_domain_only: bool,
    allowed_skill_ids: set[str] | None = None,
) -> list:
    return offer_skills_for_domain(
        bank,
        domain=domain,
        max_skills=max_skills,
        allow_provisional=allow_provisional,
        same_domain_only=same_domain_only,
        allowed_skill_ids=allowed_skill_ids,
    )


def _run_domain_collection(
    *,
    domain: str,
    task_ids: list[str],
    model: str,
    user_model: str,
    seed: int,
    max_concurrency: int,
    save_to: str,
    skill_bank_path: Path,
    organization_path: Path | None,
    max_inject_skills: int,
    allow_provisional_inject: bool,
    agent_name: str,
    task_split_name: str,
    probation_primary_quota: int = 0,
    inject_same_domain_only: bool = True,
    allowed_skill_ids: set[str] | None = None,
) -> dict[str, Any]:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import run_domain

    os.environ["SAGE_TAU2_SKILL_BANK"] = str(skill_bank_path)
    os.environ["SAGE_TAU2_MAX_SKILLS"] = str(max_inject_skills)
    os.environ["SAGE_TAU2_INJECT_PROVISIONAL"] = (
        "1" if allow_provisional_inject else "0"
    )
    os.environ["SAGE_TAU2_INJECT_SAME_DOMAIN_ONLY"] = (
        "1" if inject_same_domain_only else "0"
    )
    if allowed_skill_ids is not None:
        os.environ["SAGE_TAU2_ALLOWED_SKILL_IDS"] = json.dumps(
            sorted(str(x) for x in allowed_skill_ids)
        )
    else:
        os.environ.pop("SAGE_TAU2_ALLOWED_SKILL_IDS", None)
    os.environ["SAGE_TAU2_DOMAIN"] = str(domain)
    if organization_path is not None:
        os.environ["SAGE_TAU2_ORG_PATH"] = str(organization_path)
    else:
        os.environ.pop("SAGE_TAU2_ORG_PATH", None)
    # Parent of bank is the run output root; journal lives there.
    dispatch_log = Path(skill_bank_path).resolve().parent / "dispatch_journal.jsonl"
    os.environ["SAGE_TAU2_DISPATCH_LOG"] = str(dispatch_log)

    llm_args_agent: dict[str, Any] = {
        "skill_bank_path": str(skill_bank_path),
        "max_inject_skills": max_inject_skills,
        "inject_provisional": allow_provisional_inject,
        "inject_same_domain_only": inject_same_domain_only,
        "enable_executor_dispatch": True,
        "dispatch_log_path": str(dispatch_log),
        "dispatch_config": {
            "require_accepted_for_primary": True,
            "probation_primary_quota": int(probation_primary_quota),
        },
    }
    if allowed_skill_ids is not None:
        llm_args_agent["allowed_skill_ids"] = sorted(str(x) for x in allowed_skill_ids)
    if organization_path is not None:
        llm_args_agent["organization_path"] = str(organization_path)

    # Segment save_to is unique per (run, seg, domain). Stale results from a
    # previous partial attempt cause interactive resume / task-set mismatches —
    # always start clean for this save name.
    sim_dir = Path("data/simulations") / save_to
    if sim_dir.exists():
        import shutil

        shutil.rmtree(sim_dir)

    config = TextRunConfig(
        domain=domain,
        agent=agent_name,
        llm_agent=model,
        llm_user=user_model,
        num_trials=1,
        task_ids=task_ids,
        task_split_name=task_split_name,
        seed=seed,
        max_concurrency=max_concurrency,
        save_to=save_to,
        llm_args_agent=llm_args_agent,
    )
    run_domain(config)
    sim_path = Path("data/simulations") / save_to / "results.json"
    if not sim_path.exists():
        raise RuntimeError(f"Missing results for save_to={save_to}")
    return read_json(sim_path)


def run_multidomain(
    config: MultiDomainConfig,
    *,
    checkpoint: Path | None,
    output_root: Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    bank_path = output_root / "skill_bank.json"
    org_path = output_root / "organization.json"
    cluster_path = output_root / "skill_clusters.json"

    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        src_bank = checkpoint / "skill_bank.json"
        src_org = checkpoint / "organization.json"
        if not src_bank.exists():
            raise SystemExit(f"Missing skill bank: {src_bank}")
        if not src_org.exists() and not config.freeze_organization:
            raise SystemExit(f"Missing organization: {src_org}")
        if not bank_path.exists():
            shutil.copy2(src_bank, bank_path)
        if src_org.exists() and not org_path.exists():
            shutil.copy2(src_org, org_path)
        if (checkpoint / "skill_clusters.json").exists() and not cluster_path.exists():
            shutil.copy2(checkpoint / "skill_clusters.json", cluster_path)
        ckpt_note = checkpoint.name
    else:
        # Cold start: empty bank + Executor-only org (dispatch fix applies).
        if not bank_path.exists():
            Tau2SkillBank(bank_path).save()
        if not org_path.exists() and not config.freeze_organization:
            from sage_tau2.organization import Organization

            Organization.load(org_path)  # creates Executor on disk
        ckpt_note = "cold_start"

    _apply_relay_from_llm_config(config.llm_config_path)
    _register_agent()

    weights = config.domain_weights or {
        "airline": 30,
        "retail": 74,
        "telecom": 74,
    }
    seg_size = int(config.segment_size or config.total_tasks or 100)
    num_segments = max(1, int(config.num_segments or 1))
    quotas = allocate_quotas(weights, seg_size)
    validate_evolve_budget(
        segment_size=seg_size,
        num_segments=num_segments,
        quotas=quotas,
        split_name=config.task_split_name,
        allow_task_resampling=config.allow_task_resampling,
    )

    user_model = config.user_model or config.model
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
        credit_require_full_write_spine=bool(
            config.credit_require_full_write_spine
        ),
        credit_per_episode_inject=bool(config.credit_per_episode_inject),
        distill_prior_segments=int(config.distill_prior_segments),
        min_support_for_verified=int(config.min_support_for_verified),
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

    state_path = output_root / "online_state.json"
    state: dict[str, Any] = {
        "protocol": "sage_tau2_multidomain_segments_v1",
        "checkpoint": ckpt_note,
        "segment_size": seg_size,
        "num_segments": num_segments,
        "quotas_per_segment": quotas,
        "task_split_name": config.task_split_name,
        "allow_task_resampling": config.allow_task_resampling,
        "segments": [],
        "completed_segments": 0,
    }
    if state_path.exists():
        state = read_json(state_path)
        completed = int(state.get("completed_segments") or 0)
    else:
        completed = 0

    # Fixed val per domain (proportional to weights); persist once for the run.
    val_map: dict[str, list[str]] = {}
    if int(config.val_size) > 0:
        raw_val = state.get("val_task_ids")
        if isinstance(raw_val, dict) and raw_val:
            val_map = {
                str(d): [str(x) for x in (ids or [])]
                for d, ids in raw_val.items()
            }
        else:
            val_seed = (
                int(config.val_seed)
                if config.val_seed is not None
                else int(config.seed) + 7919
            )
            val_quotas = allocate_quotas(weights, int(config.val_size))
            for domain, n_val in val_quotas.items():
                if int(n_val) <= 0:
                    continue
                val_map[domain] = sample_fixed_val_ids(
                    domain=domain,
                    split_name=config.val_split_name,
                    val_size=int(n_val),
                    seed=val_seed + abs(hash(domain)) % 10_000,
                )
            state["val_task_ids"] = val_map
            state["val_split_name"] = config.val_split_name
            state["val_exclude_from_train"] = bool(config.val_exclude_from_train)
            state["val_seed"] = val_seed
            state["val_quotas"] = val_quotas
            write_json(
                output_root / "val_task_ids.json",
                {
                    "split": config.val_split_name,
                    "seed": val_seed,
                    "exclude_from_train": bool(config.val_exclude_from_train),
                    "quotas": val_quotas,
                    "task_ids": val_map,
                },
            )
            write_json(state_path, state)
            print(
                f"[sage_tau2:multi] fixed val n={sum(len(v) for v in val_map.values())} "
                f"quotas={val_quotas} exclude_from_train={config.val_exclude_from_train}",
                flush=True,
            )

    schedule: TaskSchedule | None = None
    if not config.allow_task_resampling:
        raw_sched = state.get("task_schedule")
        if isinstance(raw_sched, dict) and raw_sched.get("order"):
            schedule = TaskSchedule.from_state(raw_sched)
        else:
            exclude: dict[str, list[str]] | None = None
            if val_map and config.val_exclude_from_train:
                exclude = {d: list(ids) for d, ids in val_map.items()}
            schedule = TaskSchedule.create(
                domains=list(quotas.keys()),
                split_name=config.task_split_name,
                seed=int(config.seed),
                exclude_ids=exclude,
            )
            state["task_schedule"] = schedule.to_state()
            write_json(state_path, state)

    print(
        f"[sage_tau2:multi] continue from={ckpt_note} "
        f"segments={completed}->{num_segments} "
        f"seg_size={seg_size} quotas/seg={quotas} split={config.task_split_name} "
        f"resampling={config.allow_task_resampling}"
        + (
            f" val={ {d: len(v) for d, v in val_map.items()} }"
            if val_map
            else ""
        ),
        flush=True,
    )
    if schedule is not None:
        pools = {d: len(schedule.order.get(d) or []) for d in quotas}
        print(
            f"[sage_tau2:multi] train pools={pools} "
            f"remaining={schedule.total_remaining()}",
            flush=True,
        )

    for seg_idx in range(completed, num_segments):
        seg_id = seg_idx + 1
        seg_seed = int(config.seed) + seg_idx * 10_000
        task_map, schedule = sample_task_ids(
            quotas=quotas,
            split_name=config.task_split_name,
            seed=seg_seed,
            schedule=schedule,
            allow_task_resampling=config.allow_task_resampling,
        )
        if not task_map:
            print(
                f"[sage_tau2:multi] train pool exhausted before segment {seg_id}; stopping.",
                flush=True,
            )
            break
        if schedule is not None:
            state["task_schedule"] = schedule.to_state()
            write_json(state_path, state)
        n_seg = sum(len(v) for v in task_map.values())
        seg_root = output_root / f"segment_{seg_id:03d}"
        seg_root.mkdir(parents=True, exist_ok=True)
        write_json(
            seg_root / "task_sample.json",
            {
                "segment": seg_id,
                "seed": seg_seed,
                "split": config.task_split_name,
                "quotas": quotas,
                "task_ids": task_map,
            },
        )
        print(
            f"[sage_tau2:multi] segment {seg_id}/{num_segments} "
            f"n={n_seg} seed={seg_seed}",
            flush=True,
        )

        bank = Tau2SkillBank(bank_path)
        # Freeze inject pool at segment start: only reuse prior-segment skills.
        inject_skill_ids = {str(s.skill_id) for s in bank.active()}
        domain_rows: list[dict[str, Any]] = []
        domain_order = [d for d in ("airline", "retail", "telecom") if d in task_map]
        for extra in task_map:
            if extra not in domain_order:
                domain_order.append(extra)

        collect_meta: list[dict[str, Any]] = []
        print(
            f"[sage_tau2:multi] seg{seg_id} phase=collect "
            f"inject_pool={len(inject_skill_ids)} (prior-segment only)",
            flush=True,
        )
        for i, domain in enumerate(domain_order):
            ids = task_map[domain]
            seg_dir = seg_root / f"domain_{domain}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            save_to = f"sage_tau2_{output_root.name}_seg{seg_id:03d}_{domain}"

            offered = _offer_skills_for_domain(
                bank,
                domain=domain,
                max_skills=config.max_inject_skills,
                allow_provisional=config.allow_provisional_inject,
                same_domain_only=config.inject_same_domain_only,
                allowed_skill_ids=inject_skill_ids,
            )
            injected_ids = [s.skill_id for s in offered]
            write_json(
                seg_dir / "injection.json",
                {
                    "skill_ids": injected_ids,
                    "skill_names": [s.skill_name for s in offered],
                    "n_tasks": len(ids),
                    "domain_filter": config.inject_same_domain_only,
                    "inject_pool_policy": "prior_segment_snapshot",
                    "inject_pool_size": len(inject_skill_ids),
                },
            )
            print(
                f"[sage_tau2:multi] seg{seg_id} collect domain={domain} "
                f"n={len(ids)} inject={len(offered)}",
                flush=True,
            )
            payload = _run_domain_collection(
                domain=domain,
                task_ids=ids,
                model=config.model,
                user_model=user_model,
                seed=seg_seed + i * 1000,
                max_concurrency=config.max_concurrency,
                save_to=save_to,
                skill_bank_path=bank_path,
                # Prior-segment specialists may still dispatch; new ADD waits
                # until the segment evolve phase below.
                organization_path=None if config.freeze_organization else org_path,
                max_inject_skills=config.max_inject_skills,
                allow_provisional_inject=config.allow_provisional_inject,
                agent_name=config.agent_name,
                task_split_name=config.task_split_name,
                probation_primary_quota=int(config.probation_primary_quota),
                inject_same_domain_only=config.inject_same_domain_only,
                allowed_skill_ids=inject_skill_ids,
            )
            write_json(seg_dir / "raw_results.json", payload)
            if config.dump_pro_trajectories:
                try:
                    export_pro_trajectories(
                        results_payload=payload,
                        results_path=Path("data/simulations") / save_to / "results.json",
                        output_dir=seg_dir / "pro_trajectories",
                        domain=domain,
                        dispatch_log_path=output_root / "dispatch_journal.jsonl",
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[sage_tau2:multi] PRO dump failed seg{seg_id} {domain}: {exc}",
                        flush=True,
                    )

            bank = Tau2SkillBank(bank_path)
            # Phase 1: distill + cluster only (no nominate / Spec mid-segment).
            summary = update_bank_from_results(
                results_payload=payload,
                domain=domain,
                bank=bank,
                injected_skill_ids=injected_ids,
                config=segment_cfg,
                segment_dir=seg_dir,
                segment_index=seg_id * 10 + i + 1,
                organization_path=None if config.freeze_organization else org_path,
                cluster_archive_path=(
                    None if config.freeze_organization else cluster_path
                ),
                distill_and_credit=True,
                update_clusters=not config.freeze_organization,
                evolve_organization=False,
            )
            row = {
                "domain": domain,
                "n_tasks": len(ids),
                "save_to": save_to,
                "avg_reward": summary.get("avg_reward"),
                "n_wins": summary.get("n_wins"),
                "n_trajectories": summary.get("n_trajectories"),
                "n_new_skills_distilled": summary.get("n_new_skills_distilled"),
                "n_active_skills": summary.get("n_active_skills"),
                "accepted_agents": [],
            }
            domain_rows.append(row)
            collect_meta.append(
                {
                    "domain": domain,
                    "index": i,
                    "seg_dir": str(seg_dir),
                    "injected_ids": injected_ids,
                }
            )
            write_json(seg_dir / "domain_summary.json", row)
            print(
                f"[sage_tau2:multi] seg{seg_id} {domain} collected "
                f"avg={summary.get('avg_reward'):.3f} "
                f"bank={summary.get('n_active_skills')} "
                f"new_skills={summary.get('n_new_skills_distilled')}",
                flush=True,
            )

        # Phase 2: after all domains distilled, nominate/admit per domain.
        if not config.freeze_organization:
            print(
                f"[sage_tau2:multi] seg{seg_id} phase=evolve_org "
                f"domains={len(collect_meta)}",
                flush=True,
            )
            for meta in collect_meta:
                domain = str(meta["domain"])
                i = int(meta["index"])
                seg_dir = Path(meta["seg_dir"])
                payload = read_json(seg_dir / "raw_results.json")
                injected_ids = list(meta["injected_ids"])
                bank = Tau2SkillBank(bank_path)
                probe = None
                if config.enable_spec_vs_exec:
                    from sage_tau2.admission_probe import make_probe_callback

                    probe = make_probe_callback(
                        domain=domain,
                        model=config.model,
                        user_model=user_model,
                        skill_bank_path=bank_path,
                        organization_path=org_path,
                        seed=seg_seed + i * 1000 + 17,
                        max_concurrency=max(1, min(2, int(config.max_concurrency))),
                        agent_name=config.agent_name,
                        probe_dir=seg_dir / "admission_probes",
                    )
                summary = update_bank_from_results(
                    results_payload=payload,
                    domain=domain,
                    bank=bank,
                    injected_skill_ids=injected_ids,
                    config=segment_cfg,
                    segment_dir=seg_dir,
                    segment_index=seg_id * 10 + i + 1,
                    organization_path=org_path,
                    cluster_archive_path=cluster_path,
                    spec_vs_exec_probe=probe,
                    distill_and_credit=False,
                    update_clusters=False,
                    evolve_organization=True,
                )
                edit = summary.get("organization_edit") or {}
                for row in domain_rows:
                    if row["domain"] == domain:
                        row["accepted_agents"] = edit.get("accepted_agents") or []
                        row["n_active_skills"] = summary.get(
                            "n_active_skills", row.get("n_active_skills")
                        )
                        write_json(seg_dir / "domain_summary.json", row)
                        break
                print(
                    f"[sage_tau2:multi] seg{seg_id} {domain} evolve "
                    f"add={edit.get('accepted_agents')}",
                    flush=True,
                )

        # Phase 3: fixed val retest after evolve (no distill).
        val_summary: dict[str, Any] | None = None
        if val_map:
            val_root = seg_root / "val"
            val_root.mkdir(parents=True, exist_ok=True)
            bank = Tau2SkillBank(bank_path)
            inject_skill_ids = {str(s.skill_id) for s in bank.active()}
            val_domain_rows: list[dict[str, Any]] = []
            print(
                f"[sage_tau2:multi] seg{seg_id} phase=val "
                f"n={sum(len(v) for v in val_map.values())}",
                flush=True,
            )
            for i, domain in enumerate(
                [d for d in ("airline", "retail", "telecom") if d in val_map]
                + [d for d in val_map if d not in ("airline", "retail", "telecom")]
            ):
                ids = val_map[domain]
                if not ids:
                    continue
                val_dir = val_root / f"domain_{domain}"
                val_dir.mkdir(parents=True, exist_ok=True)
                val_save = f"sage_tau2_{output_root.name}_seg{seg_id:03d}_val_{domain}"
                offered = _offer_skills_for_domain(
                    bank,
                    domain=domain,
                    max_skills=config.max_inject_skills,
                    allow_provisional=config.allow_provisional_inject,
                    same_domain_only=config.inject_same_domain_only,
                    allowed_skill_ids=inject_skill_ids,
                )
                payload = _run_domain_collection(
                    domain=domain,
                    task_ids=ids,
                    model=config.model,
                    user_model=user_model,
                    seed=int(config.seed) + seg_idx * 1000 + 333 + i,
                    max_concurrency=config.max_concurrency,
                    save_to=val_save,
                    skill_bank_path=bank_path,
                    organization_path=None if config.freeze_organization else org_path,
                    max_inject_skills=config.max_inject_skills,
                    allow_provisional_inject=config.allow_provisional_inject,
                    agent_name=config.agent_name,
                    task_split_name=config.val_split_name,
                    probation_primary_quota=int(config.probation_primary_quota),
                    inject_same_domain_only=config.inject_same_domain_only,
                    allowed_skill_ids=inject_skill_ids,
                )
                write_json(val_dir / "raw_results.json", payload)
                stats = _reward_stats(payload)
                row = {
                    "domain": domain,
                    "n_tasks": len(ids),
                    "injected_skills": [s.skill_name for s in offered],
                    **{k: stats[k] for k in ("n_scored", "n_pass", "avg_reward", "pass_rate")},
                }
                val_domain_rows.append(row)
                write_json(val_dir / "val_summary.json", {**row, "per_task": stats["per_task"]})
                if config.dump_val_pro_trajectories:
                    try:
                        export_pro_trajectories(
                            results_payload=payload,
                            results_path=Path("data/simulations")
                            / val_save
                            / "results.json",
                            output_dir=val_dir / "pro_trajectories",
                            domain=domain,
                            dispatch_log_path=output_root / "dispatch_journal.jsonl",
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[sage_tau2:multi] val PRO dump failed "
                            f"seg{seg_id} {domain}: {exc}",
                            flush=True,
                        )
                print(
                    f"[sage_tau2:multi] seg{seg_id} val {domain} "
                    f"avg={row['avg_reward']:.3f} "
                    f"pass={row['n_pass']}/{row['n_scored']}",
                    flush=True,
                )
            n_scored = sum(int(r["n_scored"]) for r in val_domain_rows)
            n_pass = sum(int(r["n_pass"]) for r in val_domain_rows)
            avg = (
                sum(float(r["avg_reward"]) * int(r["n_scored"]) for r in val_domain_rows)
                / n_scored
                if n_scored
                else 0.0
            )
            val_summary = {
                "n_scored": n_scored,
                "n_pass": n_pass,
                "avg_reward": avg,
                "pass_rate": (n_pass / n_scored) if n_scored else 0.0,
                "domains": val_domain_rows,
            }
            write_json(val_root / "val_summary.json", val_summary)
            print(
                f"[sage_tau2:multi] seg{seg_id} val "
                f"avg={avg:.3f} pass={n_pass}/{n_scored}",
                flush=True,
            )

        seg_mean = (
            sum(float(r["avg_reward"] or 0) for r in domain_rows) / len(domain_rows)
            if domain_rows
            else 0.0
        )
        seg_row: dict[str, Any] = {
            "segment": seg_id,
            "seed": seg_seed,
            "mean_domain_reward": seg_mean,
            "domains": domain_rows,
        }
        if val_summary is not None:
            seg_row["val"] = val_summary
        state.setdefault("segments", []).append(seg_row)
        state["completed_segments"] = seg_id
        write_json(state_path, state)
        print(
            f"[sage_tau2:multi] segment {seg_id} done mean={seg_mean:.3f}"
            + (
                f" val={val_summary['avg_reward']:.3f}"
                if val_summary is not None
                else ""
            ),
            flush=True,
        )

    specialists: list[str] = []
    if not config.freeze_organization and org_path.exists():
        from sage_tau2.organization import Organization

        specialists = [a.name for a in Organization.load(org_path).specialists()]

    seg_means = [
        float(s.get("mean_domain_reward") or 0)
        for s in state.get("segments") or []
        if isinstance(s, dict)
    ]
    used_total = 0
    if schedule is not None:
        used_total = sum(len(v) for v in schedule.used_task_ids().values())
    val_avgs = [
        float((s.get("val") or {}).get("avg_reward") or 0)
        for s in state.get("segments") or []
        if isinstance(s, dict) and isinstance(s.get("val"), dict)
    ]
    final = {
        "protocol": "sage_tau2_multidomain_segments_v1",
        "checkpoint": ckpt_note,
        "model": config.model,
        "task_split_name": config.task_split_name,
        "allow_task_resampling": config.allow_task_resampling,
        "segment_size": seg_size,
        "num_segments": num_segments,
        "total_tasks": used_total or seg_size * len(state.get("segments") or []),
        "quotas_per_segment": quotas,
        "mean_segment_reward": (
            sum(seg_means) / len(seg_means) if seg_means else 0.0
        ),
        "mean_val_reward": (
            sum(val_avgs) / len(val_avgs) if val_avgs else None
        ),
        "val_task_ids": val_map or state.get("val_task_ids"),
        "n_skills": len(Tau2SkillBank(bank_path).skills),
        "specialists": specialists,
        "segments": state.get("segments"),
        "skill_bank_path": str(bank_path),
        "organization_path": str(org_path),
    }
    write_json(output_root / "online_summary.json", final)
    print(
        f"[sage_tau2:multi] ALL done mean_segment_reward="
        f"{final['mean_segment_reward']:.3f} bank={final['n_skills']} "
        f"specialists={len(specialists)} total_tasks={final['total_tasks']}",
        flush=True,
    )
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-domain proportional continue evolution for sage_tau2"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Prior run dir with skill_bank.json + organization.json; "
        "omit for cold start (empty bank + Executor only)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm-config", default=None)
    parser.add_argument(
        "--no-dump-pro-trajectories",
        action="store_true",
        help="Skip writing PRO trajectories under each segment_*/<domain>/pro_trajectories/",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).resolve()
    if "sage_mas" in output.parts and "sage_tau2" not in output.parts:
        raise SystemExit(
            f"Refusing to write under sage_mas logs: {output}. "
            "Use logs/sage_tau2/... instead."
        )

    config = load_config(args.config)
    if args.llm_config:
        config.llm_config_path = args.llm_config
    if args.no_dump_pro_trajectories:
        config.dump_pro_trajectories = False
    run_multidomain(
        config,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        output_root=output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
