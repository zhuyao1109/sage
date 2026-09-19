"""τ² Spec-vs-Exec admission probe (paired specialist vs bare Executor).

Mirrors sage_mas ``probe_actor_promotion(..., bare_executor=True)``:
run the same capability-scoped task sample twice — once forcing the specialist
primary (with assigned skills), once forcing Executor with **no skill inject** —
and compare success rates on tasks that completed without infrastructure errors.
"""

from __future__ import annotations

import os
import random
import shutil
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any

from sage_tau2.organization import EXECUTOR_NAME, Organization
from sage_tau2.schemas import AgentSpec, Tau2Skill
from sage_tau2.serialization import read_json, write_json
from sage_tau2.skill_resolve import resolve_assigned_skills
from sage_tau2.task_context import (
    capability_scope_from_key,
    organizational_capability_key,
    probe_scopes_for_capability_keys,
    skill_matches_episode,
    task_matches_probe_scope,
)
from sage_tau2.trajectory import results_json_to_trajectories


def _clone_agent(agent: AgentSpec) -> AgentSpec:
    allowed = {f.name for f in fields(AgentSpec)}
    payload = {name: deepcopy(getattr(agent, name)) for name in allowed}
    return AgentSpec(**payload)


def _task_id_from_sim(sim: dict[str, Any]) -> str:
    return str(sim.get("task_id") or "")


def _termination_reason(sim: dict[str, Any]) -> str:
    term = sim.get("termination_reason")
    if term is None and isinstance(sim.get("info"), dict):
        term = sim["info"].get("termination_reason")
    return str(term or "").strip().lower()


def _is_infrastructure_failure(sim: dict[str, Any]) -> bool:
    term = _termination_reason(sim)
    return "infrastructure" in term


def _sim_success(sim: dict[str, Any]) -> bool:
    reward_info = sim.get("reward_info") or {}
    try:
        return float(reward_info.get("reward") or 0.0) >= 1.0
    except (TypeError, ValueError):
        return False


def paired_probe_scores(
    specialist_payload: dict[str, Any],
    executor_payload: dict[str, Any],
) -> dict[str, Any]:
    """Score Spec vs Exec on shared task ids, dropping infrastructure failures.

    A task is counted only when **both** arms finished without
    ``infrastructure_error``. This matches a fair paired contrast: API/infra
    flakes must not reject an otherwise stronger specialist.
    """
    spec_sims = {
        _task_id_from_sim(s): s
        for s in (specialist_payload.get("simulations") or [])
        if isinstance(s, dict) and _task_id_from_sim(s)
    }
    exec_sims = {
        _task_id_from_sim(s): s
        for s in (executor_payload.get("simulations") or [])
        if isinstance(s, dict) and _task_id_from_sim(s)
    }
    shared = sorted(set(spec_sims) & set(exec_sims))
    dropped_infra: list[str] = []
    counted: list[str] = []
    spec_wins = 0
    exec_wins = 0
    for tid in shared:
        s_sim = spec_sims[tid]
        e_sim = exec_sims[tid]
        if _is_infrastructure_failure(s_sim) or _is_infrastructure_failure(e_sim):
            dropped_infra.append(tid)
            continue
        counted.append(tid)
        if _sim_success(s_sim):
            spec_wins += 1
        if _sim_success(e_sim):
            exec_wins += 1
    n = len(counted)
    return {
        "specialist_wins": spec_wins,
        "executor_wins": exec_wins,
        "n_tasks": n,
        "specialist_sr": (spec_wins / float(n)) if n else 0.0,
        "executor_sr": (exec_wins / float(n)) if n else 0.0,
        "scored_task_ids": counted,
        "dropped_infra_task_ids": dropped_infra,
        "n_shared_raw": len(shared),
    }


def _avg_success(results_payload: dict[str, Any], domain: str) -> tuple[float, int, int]:
    trajs = results_json_to_trajectories(results_payload, domain=domain)
    if not trajs:
        return 0.0, 0, 0
    wins = sum(1 for t in trajs if float(t.reward or 0.0) >= 1.0)
    n = len(trajs)
    return wins / float(n), wins, n


def _task_id(task: Any) -> str:
    if isinstance(task, dict):
        return str(task.get("id") or "")
    return str(getattr(task, "id", "") or "")


def probe_scopes_for_specialist(
    specialist: AgentSpec,
    skills: list[Tau2Skill] | None = None,
) -> list[str]:
    """Collect routing scopes for capability-directed Spec probe sampling.

    Prefer **write / organizational** capability scopes from assigned skills.
    Do not widen the pool via episode ``task_families`` (that caused
    roaming specialists to be probed on unrelated mms/mobile combos).
    """
    scopes: list[str] = []
    seen: set[str] = set()
    skill_by_id = {s.skill_id: s for s in skills or []}
    for sid in specialist.assigned_skills or []:
        skill = skill_by_id.get(str(sid))
        if skill is None:
            # Names are rare; resolve_assigned_skills handles id+name.
            continue
        org = organizational_capability_key(skill)
        scope = capability_scope_from_key(org)
        if scope and scope not in seen:
            seen.add(scope)
            scopes.append(scope)
    if not scopes:
        # Fallback: resolve by name/id then org key.
        for skill in resolve_assigned_skills(
            list(specialist.assigned_skills or []), list(skills or [])
        ):
            org = organizational_capability_key(skill)
            scope = capability_scope_from_key(org)
            if scope and scope not in seen:
                seen.add(scope)
                scopes.append(scope)
    if not scopes:
        for key in specialist.capability_keys or []:
            scope = capability_scope_from_key(key)
            if scope and scope not in seen:
                seen.add(scope)
                scopes.append(scope)
    if not scopes:
        scopes = probe_scopes_for_capability_keys(
            list(specialist.capability_keys or [])
        )
    return scopes


def select_capability_probe_task_ids(
    *,
    domain: str,
    specialist: AgentSpec,
    skills: list[Tau2Skill] | None,
    num_tasks: int,
    seed: int,
    split_name: str = "train",
    task_loader: Any | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Sample probe tasks on the specialist's skill-match subset when possible."""
    if task_loader is None:
        from tau2.runner.helpers import load_tasks as task_loader

    scopes = probe_scopes_for_specialist(specialist, skills)
    assigned = resolve_assigned_skills(
        list(specialist.assigned_skills or []), list(skills or [])
    )
    meta: dict[str, Any] = {
        "capability_keys": list(specialist.capability_keys or []),
        "capability_scopes": scopes,
        "split_name": split_name,
        "requested_tasks": int(num_tasks),
        "n_assigned_skills": len(assigned),
    }
    tasks = task_loader(task_set_name=domain, task_split_name=split_name)

    pool_ids: list[str] = []
    if assigned:
        # Same gate as inject/dispatch: only episodes the skills would match.
        for t in tasks:
            tid = _task_id(t)
            if not tid:
                continue
            if any(
                skill_matches_episode(sk, domain=domain, task_id=tid)
                for sk in assigned
            ):
                pool_ids.append(tid)
        meta["pool_size"] = len(pool_ids)
        meta["reason"] = "skill_match_sample"
        if not pool_ids:
            meta["note"] = (
                "no skill-matching tasks; falling back to write-capability scopes"
            )
    if not pool_ids:
        if not scopes:
            meta["reason"] = "specialist has no capability scope for probe filtering"
            meta["pool_size"] = 0
            meta["sampled_task_ids"] = []
            return [], meta
        scope_set = set(scopes)
        pool_ids = [
            _task_id(t)
            for t in tasks
            if _task_id(t) and task_matches_probe_scope(t, scope_set)
        ]
        meta["pool_size"] = len(pool_ids)
        meta["reason"] = "capability_scoped_sample"
        if not pool_ids:
            meta["reason"] = (
                f"no {split_name} tasks in domain={domain} match scope(s) {scopes}"
            )
            meta["sampled_task_ids"] = []
            return [], meta

    want = max(1, int(num_tasks))
    rng = random.Random(int(seed))
    picked = list(pool_ids)
    rng.shuffle(picked)
    sampled = picked[: min(want, len(picked))]
    meta["sampled_task_ids"] = sampled
    if len(sampled) < want:
        meta["note"] = (
            f"capability pool smaller than requested: using {len(sampled)}/{want}"
        )
    return sampled, meta


def _empty_probe_result(
    *,
    num_tasks: int,
    selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "specialist_sr": 0.0,
        "executor_sr": 0.0,
        "specialist_wins": 0,
        "executor_wins": 0,
        "n_tasks": 0,
        "specialist_run": None,
        "executor_run": None,
        "mode": "spec_vs_exec",
        "probe_task_selection": selection,
        "requested_tasks": int(num_tasks),
        "bare_executor": True,
    }


def run_spec_vs_exec_probe(
    *,
    domain: str,
    model: str,
    user_model: str,
    specialist: AgentSpec,
    organization: Organization,
    skills: list[Tau2Skill],
    skill_bank_path: Path,
    organization_path: Path,
    num_tasks: int = 8,
    seed: int = 0,
    max_concurrency: int = 1,
    agent_name: str = "sage_tau2",
    probe_dir: Path | None = None,
    task_split_name: str = "train",
    bare_executor: bool = True,
) -> dict[str, Any]:
    """Paired forced-primary rollouts; returns SR stats for admission_would_pass."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import run_domain

    probe_dir = Path(probe_dir or (organization_path.parent / "admission_probes"))
    probe_dir.mkdir(parents=True, exist_ok=True)
    tag = specialist.name.replace(" ", "_")[:48]
    probe_org_path = probe_dir / f"{tag}_organization.json"

    task_ids, selection = select_capability_probe_task_ids(
        domain=domain,
        specialist=specialist,
        skills=skills,
        num_tasks=int(num_tasks),
        seed=int(seed),
        split_name=str(task_split_name),
    )
    write_json(probe_dir / f"{tag}_task_selection.json", selection)
    if not task_ids:
        print(
            f"[sage_tau2] Spec probe skipped specialist={specialist.name} "
            f"reason={selection.get('reason')}",
            flush=True,
        )
        return _empty_probe_result(num_tasks=num_tasks, selection=selection)

    print(
        f"[sage_tau2] Spec probe tasks specialist={specialist.name} "
        f"scopes={selection.get('capability_scopes')} "
        f"pool={selection.get('pool_size')} sample={task_ids} "
        f"bare_executor={bare_executor}",
        flush=True,
    )

    probe_agents = [_clone_agent(a) for a in organization.agents]
    if not any(a.name == specialist.name for a in probe_agents):
        probe_agents.append(_clone_agent(specialist))
    Organization(probe_agents).save(probe_org_path)

    def _one(force_primary: str, label: str, *, max_inject_skills: int) -> dict[str, Any]:
        save_to = f"sage_tau2_probe_{tag}_{label}_s{seed}"
        # Probe save names are deterministic (specialist + seed), so a stale
        # directory from an earlier run would trigger tau2's interactive
        # resume prompt and block the pipeline. Probes must reflect the
        # CURRENT bank/org — always start fresh.
        stale_dir = Path("data/simulations") / save_to
        if stale_dir.exists():
            shutil.rmtree(stale_dir, ignore_errors=True)
        os.environ["SAGE_TAU2_SKILL_BANK"] = str(skill_bank_path)
        os.environ["SAGE_TAU2_ORG_PATH"] = str(probe_org_path)
        os.environ["SAGE_TAU2_FORCE_PRIMARY"] = force_primary
        os.environ["SAGE_TAU2_ENABLE_DISPATCH"] = "1"
        os.environ["SAGE_TAU2_DOMAIN"] = str(domain)
        config = TextRunConfig(
            domain=domain,
            agent=agent_name,
            llm_agent=model,
            llm_user=user_model,
            num_trials=1,
            task_ids=list(task_ids),
            seed=int(seed),
            max_concurrency=int(max_concurrency),
            save_to=save_to,
            auto_resume=True,
            task_split_name=str(task_split_name),
            llm_args_agent={
                "skill_bank_path": str(skill_bank_path),
                "organization_path": str(probe_org_path),
                "force_primary": force_primary,
                "enable_executor_dispatch": True,
                # Specialist arm uses assigned skills via org. Executor arm uses
                # bare_executor ⇒ max_inject_skills=0 (sage_mas Spec-vs-Exec).
                "max_inject_skills": int(max_inject_skills),
                "inject_provisional": True,
                "dispatch_config": {
                    "require_accepted_for_primary": False,
                    "probation_primary_quota": 8,
                },
            },
        )
        run_domain(config)
        sim_path = Path("data/simulations") / save_to / "results.json"
        payload = read_json(sim_path) if sim_path.exists() else {}
        write_json(probe_dir / f"{tag}_{label}.json", payload if payload else {})
        raw_sr, raw_wins, raw_n = _avg_success(payload, domain=domain)
        return {
            "force_primary": force_primary,
            "save_to": save_to,
            "payload": payload if payload else {},
            "raw_sr": raw_sr,
            "raw_wins": raw_wins,
            "raw_n": raw_n,
            "max_inject_skills": int(max_inject_skills),
        }

    try:
        spec = _one(specialist.name, "spec", max_inject_skills=4)
        exec_inject = 0 if bare_executor else 4
        exec_ = _one(EXECUTOR_NAME, "exec", max_inject_skills=exec_inject)
    finally:
        os.environ.pop("SAGE_TAU2_FORCE_PRIMARY", None)

    paired = paired_probe_scores(spec["payload"], exec_["payload"])
    print(
        f"[sage_tau2] Spec probe scored specialist={specialist.name} "
        f"clean={paired['specialist_wins']}/{paired['n_tasks']} vs "
        f"Executor {paired['executor_wins']}/{paired['n_tasks']} "
        f"(dropped_infra={paired['dropped_infra_task_ids']})",
        flush=True,
    )

    return {
        "specialist_sr": float(paired["specialist_sr"]),
        "executor_sr": float(paired["executor_sr"]),
        "specialist_wins": int(paired["specialist_wins"]),
        "executor_wins": int(paired["executor_wins"]),
        "n_tasks": int(paired["n_tasks"]),
        "specialist_run": {
            "force_primary": spec["force_primary"],
            "save_to": spec["save_to"],
            "sr": spec["raw_sr"],
            "wins": spec["raw_wins"],
            "n": spec["raw_n"],
            "max_inject_skills": spec["max_inject_skills"],
        },
        "executor_run": {
            "force_primary": exec_["force_primary"],
            "save_to": exec_["save_to"],
            "sr": exec_["raw_sr"],
            "wins": exec_["raw_wins"],
            "n": exec_["raw_n"],
            "max_inject_skills": exec_["max_inject_skills"],
        },
        "mode": "spec_vs_exec",
        "probe_task_selection": selection,
        "requested_tasks": int(num_tasks),
        "bare_executor": bool(bare_executor),
        "scored_task_ids": paired["scored_task_ids"],
        "dropped_infra_task_ids": paired["dropped_infra_task_ids"],
        "n_shared_raw": paired["n_shared_raw"],
    }


def make_probe_callback(
    *,
    domain: str,
    model: str,
    user_model: str,
    skill_bank_path: Path,
    organization_path: Path,
    seed: int,
    max_concurrency: int = 1,
    agent_name: str = "sage_tau2",
    probe_dir: Path | None = None,
    bare_executor: bool = True,
):
    """Build a SpecVsExecProbe closure for ``run_nominate_admit``."""

    def _probe(
        *,
        specialist: AgentSpec,
        organization: Organization,
        skills: list[Tau2Skill],
        num_tasks: int,
    ) -> dict[str, Any]:
        scopes = probe_scopes_for_specialist(specialist, skills)
        print(
            f"[sage_tau2] Spec-vs-Exec probe specialist={specialist.name} "
            f"n={num_tasks} scopes={scopes} bare_executor={bare_executor}",
            flush=True,
        )
        return run_spec_vs_exec_probe(
            domain=domain,
            model=model,
            user_model=user_model,
            specialist=specialist,
            organization=organization,
            skills=skills,
            skill_bank_path=skill_bank_path,
            organization_path=organization_path,
            num_tasks=num_tasks,
            seed=seed,
            max_concurrency=max_concurrency,
            agent_name=agent_name,
            probe_dir=probe_dir,
            task_split_name="train",
            bare_executor=bare_executor,
        )

    return _probe
