"""Segment-level distill + credit + cluster + nominate/admit + onboarding for τ².

Defaults aligned with sage_mas online600_noseed:
  Spec-vs-Exec off, inject≤2, verified-only ADD, onboarding min_games/wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sage_tau2.credit import (
    CreditPolicy,
    apply_credit_for_episode,
    initialize_credit,
    injectable_skills,
    seed_credit_from_birth_support,
)
from sage_tau2.distill import distill_skills_from_trajectories, summarize_protocol_histogram
from sage_tau2.nominate_admit import NominateAdmitConfig, run_nominate_admit
from sage_tau2.onboarding import (
    OnboardingPolicy,
    load_dispatch_index,
    refresh_acting_statuses,
    write_onboarding_report,
)
from sage_tau2.organization import EXECUTOR_NAME, Organization
from sage_tau2.serialization import write_json
from sage_tau2.skill_bank import Tau2SkillBank
from sage_tau2.skill_clustering import SkillClusterArchive
from sage_tau2.skill_resolve import heal_assigned_skill_refs
from sage_tau2.trajectory import results_json_to_trajectories


def _pro_trajectories_dir(segment_dir: Path) -> Path:
    if segment_dir.name.startswith("domain_"):
        return segment_dir / "pro_trajectories"
    return segment_dir / "pro_trajectories"


def _load_segment_trajectories(
    *,
    results_payload: dict[str, Any] | None,
    domain: str,
    segment_dir: str | Path | None = None,
    evolve_jsonl: str | Path | None = None,
    evolve_eligible_only: bool = False,
) -> list:
    """Load distill input from PRO dump (ALFWorld-style), not raw ``results.json``.

    Priority:
      1. ``pro_trajectories/atomic_trajectories.json``
      2. ``pro_trajectories/trajectories.jsonl`` → adapt to ``Tau2Trajectory``
      3. legacy ``evolve_trajectories.jsonl`` (old runs)
      4. ``results_payload`` simulations (fallback)
    """
    from sage_tau2.trajectory_adapter import (
        adapt_pro_many,
        load_atomic_trajectories,
        load_pro_trajectories,
    )

    if segment_dir is not None:
        base = Path(segment_dir)
        atomic_path = _pro_trajectories_dir(base) / "atomic_trajectories.json"
        if atomic_path.exists():
            return load_atomic_trajectories(
                atomic_path, eligible_only=evolve_eligible_only
            )
        pro_path = _pro_trajectories_dir(base) / "trajectories.jsonl"
        if pro_path.exists():
            records = load_pro_trajectories(pro_path)
            trajectories = adapt_pro_many(records)
            if evolve_eligible_only:
                from sage_tau2.distill import trajectory_eligible_for_distill

                trajectories = [
                    t for t in trajectories if trajectory_eligible_for_distill(t)
                ]
            return trajectories

    if evolve_jsonl is not None:
        from sage_tau2.runners.dump_evolve_trajectories import load_evolve_trajectories

        return load_evolve_trajectories(
            evolve_jsonl, eligible_only=evolve_eligible_only
        )

    if segment_dir is not None:
        legacy = _pro_trajectories_dir(Path(segment_dir)) / "evolve_trajectories.jsonl"
        if legacy.exists():
            from sage_tau2.runners.dump_evolve_trajectories import load_evolve_trajectories

            return load_evolve_trajectories(
                legacy, eligible_only=evolve_eligible_only
            )

    if results_payload is None:
        raise ValueError(
            "Provide segment_dir with PRO trajectories, evolve_jsonl, or results_payload"
        )
    return results_json_to_trajectories(results_payload, domain=domain)


@dataclass(slots=True)
class SegmentUpdateConfig:
    min_support: int = 1
    require_success: bool = True
    min_protocol_len: int = 1
    max_new_skills: int = 16
    # Match online600 max_injected_skills: 2
    max_inject_skills: int = 2
    allow_provisional_inject: bool = True
    freeze_organization: bool = True
    verify_score: float = 0.60
    prune_score: float = 0.20
    min_protocol_coverage: float = 0.5
    # Credit attribution: a use requires the skill's full write spine in the
    # episode protocol (not just min_protocol_coverage tool-name overlap).
    credit_require_full_write_spine: bool = True
    # Credit attribution: resolve injected skill ids per episode from the
    # dispatch journal (fallback to the segment-level offer when unavailable).
    credit_per_episode_inject: bool = True
    # Org evolution — defaults aligned with sage_mas online600_noseed
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
    # Match allow_provisional_org_edits: false
    allow_provisional_org_edits: bool = False
    no_spec_mode: str = "editor_commit"
    require_same_domain_for_nominate: bool = True
    # Distill: default τ²-native modes (solve_write + communicate_ok).
    # Set True to restore write-only birth (legacy ALFWorld-style spine).
    distill_solve_write_only: bool = False
    # Opt-in reinjection of hand-written confirm/diagnose dialogue into protocols.
    distill_inline_dialogue: bool = False
    # Strong verify bar for birth credit (below → verified_low_support).
    min_support_for_verified: int = 5
    # Align sage_mas online600: merge last N prior segment evolve traces into distill.
    distill_prior_segments: int = 2
    # Onboarding (probation_min_games / probation_min_wins)
    probation_min_games: int = 3
    probation_min_wins: int = 2
    remove_after_rejected_windows: int = 2
    dispatch_log_path: str | None = None


def _segment_index_from_dir(segment_dir: Path) -> int | None:
    name = segment_dir.name
    if name.startswith("segment_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def load_prior_distill_trajectories(
    *,
    segment_dir: str | Path | None,
    domain: str,
    prior_segments: int,
) -> list:
    """Load eligible atomic traces from prior segments (ALFWorld-style window).

    Layouts:
      - single-domain: ``<run>/segment_XXX/pro_trajectories/atomic_trajectories.json``
      - multi-domain: ``<run>/segment_XXX/domain_<d>/pro_trajectories/atomic_trajectories.json``
    """
    n = max(0, int(prior_segments))
    if n <= 0 or segment_dir is None:
        return []
    cur = Path(segment_dir)
    domain_subdir: str | None = None
    seg_dir = cur
    if cur.name.startswith("domain_"):
        domain_subdir = cur.name
        seg_dir = cur.parent
    seg_idx = _segment_index_from_dir(seg_dir)
    if seg_idx is None or seg_idx <= 1:
        return []
    run_root = seg_dir.parent
    from sage_tau2.trajectory_adapter import load_atomic_trajectories, load_pro_trajectories
    from sage_tau2.trajectory_adapter import adapt_pro_many

    out: list = []
    start = max(1, seg_idx - n)
    for i in range(start, seg_idx):
        prior_seg = run_root / f"segment_{i:03d}"
        prior_dir = prior_seg / domain_subdir if domain_subdir else prior_seg
        pro_root = prior_dir / "pro_trajectories"
        path = pro_root / "atomic_trajectories.json"
        loaded: list = []
        if path.exists():
            try:
                loaded = load_atomic_trajectories(path, eligible_only=True)
            except Exception:
                loaded = []
        if not loaded:
            pro_path = pro_root / "trajectories.jsonl"
            if pro_path.exists():
                try:
                    from sage_tau2.distill import trajectory_eligible_for_distill

                    loaded = [
                        t
                        for t in adapt_pro_many(load_pro_trajectories(pro_path))
                        if trajectory_eligible_for_distill(t)
                    ]
                except Exception:
                    loaded = []
        if not loaded:
            legacy = pro_root / "evolve_trajectories.jsonl"
            if not legacy.exists():
                legacy = pro_root / "evolve_trajectories_solve_write.jsonl"
            if legacy.exists():
                try:
                    from sage_tau2.runners.dump_evolve_trajectories import (
                        load_evolve_trajectories,
                    )

                    loaded = load_evolve_trajectories(legacy, eligible_only=True)
                except Exception:
                    loaded = []
        for traj in loaded:
            if getattr(traj, "domain", None) and str(traj.domain) != str(domain):
                continue
            out.append(traj)
    return out


def _merge_trajectories_for_distill(
    current: list,
    prior: list,
) -> tuple[list, dict[str, int]]:
    """Append prior eligible traces; de-dupe by (domain, task_id, trial)."""
    seen: set[tuple[str, str, int]] = set()
    merged: list = []
    for traj in list(current) + list(prior):
        key = (
            str(getattr(traj, "domain", "") or ""),
            str(getattr(traj, "task_id", "") or ""),
            int(getattr(traj, "trial", 0) or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(traj)
    return merged, {
        "n_current": len(current),
        "n_prior": len(prior),
        "n_merged": len(merged),
    }


def _load_injected_ids_by_task(
    dispatch_log_path: str | Path | None,
) -> dict[str, list[str]]:
    """task_id → per-episode injected skill ids from the dispatch journal.

    The journal is append-only at run level; later rows for the same task
    (retries) overwrite earlier ones, matching ``load_dispatch_index``.
    Rows without an ``injected_skill_ids`` list are ignored (unknown → caller
    falls back to the segment-level offer set).
    """
    if not dispatch_log_path:
        return {}
    path = Path(dispatch_log_path)
    if not path.exists():
        return {}
    import json

    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(row, dict):
                continue
            task_id = str(row.get("task_id") or "").strip()
            ids = row.get("injected_skill_ids")
            if task_id and isinstance(ids, list):
                out[task_id] = [str(x) for x in ids]
    return out


def update_bank_from_results(
    *,
    results_payload: dict[str, Any] | None = None,
    domain: str,
    bank: Tau2SkillBank,
    injected_skill_ids: list[str] | None = None,
    config: SegmentUpdateConfig | None = None,
    segment_dir: str | Path | None = None,
    segment_index: int = 1,
    organization_path: str | Path | None = None,
    cluster_archive_path: str | Path | None = None,
    spec_vs_exec_probe: Callable[..., dict[str, Any]] | None = None,
    distill_and_credit: bool = True,
    update_clusters: bool | None = None,
    evolve_organization: bool | None = None,
    evolve_jsonl: str | Path | None = None,
    evolve_eligible_only: bool = False,
) -> dict[str, Any]:
    """Distill / credit / cluster / nominate for one domain batch.

    Prefer PRO ``atomic_trajectories.json`` / ``trajectories.jsonl`` under
    ``segment_dir/pro_trajectories/`` (ALFWorld-style); legacy evolve jsonl and
    raw ``results_payload`` are fallbacks only.

    Multidomain segment protocol can split phases:
      1) ``distill_and_credit=True``, ``evolve_organization=False`` → collect learning
      2) ``distill_and_credit=False``, ``evolve_organization=True`` → org admit only
    """
    cfg = config or SegmentUpdateConfig()
    do_distill = bool(distill_and_credit)
    do_org = (
        (not cfg.freeze_organization)
        if evolve_organization is None
        else bool(evolve_organization)
    )
    do_cluster = (
        (do_org or do_distill)
        if update_clusters is None
        else bool(update_clusters)
    )
    # Clustering is required before nominate; if org is frozen and caller did not
    # ask for clusters, skip the archive entirely.
    if cfg.freeze_organization and evolve_organization is None and update_clusters is None:
        do_cluster = False
        do_org = False

    current_trajectories = _load_segment_trajectories(
        results_payload=results_payload,
        domain=domain,
        segment_dir=segment_dir,
        evolve_jsonl=evolve_jsonl,
        evolve_eligible_only=evolve_eligible_only,
    )
    distill_window_meta: dict[str, int] | None = None
    distill_trajectories = current_trajectories
    if do_distill and int(cfg.distill_prior_segments) > 0 and segment_dir is not None:
        prior = load_prior_distill_trajectories(
            segment_dir=segment_dir,
            domain=domain,
            prior_segments=int(cfg.distill_prior_segments),
        )
        if prior:
            distill_trajectories, distill_window_meta = _merge_trajectories_for_distill(
                current_trajectories, prior
            )
    # Credit / onboarding use the current segment only; distill may widen the window.
    trajectories = current_trajectories
    histogram = summarize_protocol_histogram(trajectories)
    credit_policy = CreditPolicy(
        verify_score=cfg.verify_score,
        prune_score=cfg.prune_score,
        min_protocol_coverage=cfg.min_protocol_coverage,
        min_support_for_verified=int(cfg.min_support_for_verified),
        require_full_write_spine=bool(cfg.credit_require_full_write_spine),
    )
    credit_events: list[dict[str, Any]] = []
    new_skills: list = []
    accepted: list = []
    if do_distill:
        injected_by_task: dict[str, list[str]] = {}
        if bool(cfg.credit_per_episode_inject):
            dispatch_journal = cfg.dispatch_log_path
            if dispatch_journal is None and segment_dir is not None:
                dispatch_journal = str(
                    Path(segment_dir).parent / "dispatch_journal.jsonl"
                )
            injected_by_task = _load_injected_ids_by_task(dispatch_journal)
        for traj in trajectories:
            ep_injected: list[str] | None = list(injected_skill_ids or [])
            if bool(cfg.credit_per_episode_inject):
                tid = str(getattr(traj, "task_id", "") or "")
                if tid in injected_by_task:
                    ep_injected = injected_by_task[tid]
                    if not ep_injected:
                        # Journal proves nothing was injected this episode:
                        # no skill may claim a use from it.
                        continue
            credit_events.extend(
                apply_credit_for_episode(
                    bank.active(),
                    traj,
                    injected_skill_ids=ep_injected,
                    policy=credit_policy,
                )
            )

        new_skills = distill_skills_from_trajectories(
            distill_trajectories,
            min_support=cfg.min_support,
            require_success=cfg.require_success,
            require_solve_write=cfg.distill_solve_write_only,
            min_protocol_len=cfg.min_protocol_len,
            max_skills=cfg.max_new_skills,
            min_support_for_verified=int(cfg.min_support_for_verified),
            inline_dialogue_in_protocol=bool(cfg.distill_inline_dialogue),
        )
        for skill in new_skills:
            initialize_credit(skill)
            seed_credit_from_birth_support(skill, policy=credit_policy)
        accepted = bank.extend(new_skills)
        # Re-seed after merge: bank.add may bump support_count on duplicates.
        for skill in accepted:
            seed_credit_from_birth_support(skill, policy=credit_policy)
        bank.save()

    organization_edit: dict[str, Any] | None = None
    cluster_report: dict[str, Any] | None = None
    org_result: dict[str, Any] | None = None
    onboarding_report: dict[str, Any] | None = None

    if do_cluster or do_org:
        org_path = Path(organization_path) if organization_path else None
        arch_path = Path(cluster_archive_path) if cluster_archive_path else None
        if segment_dir is not None:
            seg = Path(segment_dir)
            org_path = org_path or (seg.parent / "organization.json")
            arch_path = arch_path or (seg.parent / "skill_clusters.json")
        if org_path is None or arch_path is None:
            raise ValueError(
                "organization_path and cluster_archive_path required when "
                "clustering or organization evolution is enabled"
            )

        archive = SkillClusterArchive.load(arch_path)
        archive.novelty_threshold = float(cfg.cluster_novelty_threshold)
        if do_cluster:
            report = archive.update(bank.active(), segment=int(segment_index))
            archive.save(arch_path)
            cluster_report = report.to_dict()

        if do_org:
            organization = Organization.load(org_path)

            # Heal dangling assigned_skills refs before anything else: credit
            # pruning / dedupe may retire a skill a specialist points at, which
            # would otherwise zero that specialist's dispatch eligibility
            # ("zombie specialist"). Cascade dead refs onto the best active
            # same-capability skill.
            heal_reports: dict[str, Any] = {}
            for agent in organization.agents:
                if agent.name == EXECUTOR_NAME:
                    continue
                rep = heal_assigned_skill_refs(agent, list(bank.skills))
                if rep.get("changed"):
                    heal_reports[str(agent.name)] = rep
            if heal_reports:
                organization.save(org_path)
                for name, rep in heal_reports.items():
                    moved = rep.get("reassigned") or {}
                    dropped = rep.get("dropped") or []
                    print(
                        f"[sage_tau2] healed skill refs for {name}: "
                        f"reassigned={len(moved)} dropped={len(dropped)}"
                    )

            # Onboarding first: consume this segment's primary trials.
            dispatch_path = cfg.dispatch_log_path
            if dispatch_path is None and segment_dir is not None:
                dispatch_path = str(Path(segment_dir).parent / "dispatch_journal.jsonl")
            dispatch_index = load_dispatch_index(dispatch_path)
            onboarding_report = refresh_acting_statuses(
                organization,
                trajectories,
                dispatch_by_task=dispatch_index,
                policy=OnboardingPolicy(
                    min_games=int(cfg.probation_min_games),
                    min_wins=int(cfg.probation_min_wins),
                    remove_after_rejected_windows=int(
                        cfg.remove_after_rejected_windows
                    ),
                ),
            )
            organization.save(org_path)

            na_cfg = NominateAdmitConfig()
            na_cfg.nomination.min_cluster_support = int(cfg.nominate_min_cluster_support)
            na_cfg.nomination.require_min_utility = float(cfg.nominate_min_utility)
            na_cfg.nomination.max_new_agents_per_round = int(
                cfg.max_new_agents_per_round
            )
            na_cfg.nomination.dispatch_only_new_agents = bool(
                cfg.dispatch_only_new_agents
            )
            na_cfg.nomination.probation_games = int(cfg.probation_games)
            na_cfg.nomination.allow_provisional = bool(cfg.allow_provisional_org_edits)
            if cfg.require_same_domain_for_nominate:
                na_cfg.nomination.require_same_domain = str(domain)
            na_cfg.admission.enable_spec_vs_exec = bool(cfg.enable_spec_vs_exec)
            na_cfg.admission.num_tasks = int(cfg.admit_num_tasks)
            na_cfg.admission.min_advantage = float(cfg.admit_min_advantage)
            na_cfg.admission.accept_ties = bool(cfg.admit_accept_ties)
            na_cfg.admission.on_fail_action = str(cfg.admit_on_fail_action)
            na_cfg.admission.min_utility = float(cfg.admit_min_utility)
            na_cfg.admission.min_support = int(cfg.admit_min_support)
            na_cfg.admission.no_spec_mode = str(cfg.no_spec_mode)
            org_result = run_nominate_admit(
                organization=organization,
                archive=archive,
                skills=bank.active(),
                config=na_cfg,
                spec_vs_exec_probe=spec_vs_exec_probe,
            )
            organization.save(org_path)
            bank.save()
            if org_result.get("accepted_agents"):
                organization_edit = {
                    "edit_type": "add_agent",
                    "accepted_agents": org_result["accepted_agents"],
                    "specialists": org_result.get("specialists"),
                }
            else:
                organization_edit = {
                    "edit_type": "do_nothing",
                    "accepted_agents": [],
                    "specialists": org_result.get("specialists"),
                }

    next_inject = injectable_skills(
        bank.active(),
        max_skills=cfg.max_inject_skills,
        allow_provisional=cfg.allow_provisional_inject,
    )
    summary = {
        "domain": domain,
        "n_trajectories": len(trajectories),
        "n_wins": sum(1 for t in trajectories if t.success),
        "avg_reward": (
            sum(t.reward for t in trajectories) / len(trajectories)
            if trajectories
            else 0.0
        ),
        "protocol_histogram": histogram,
        "n_new_skills_distilled": len(new_skills),
        "n_skills_in_bank": len(bank.skills),
        "n_active_skills": len(bank.active()),
        "credit_events": len(credit_events),
        "next_inject_skill_ids": [s.skill_id for s in next_inject],
        "next_inject_skill_names": [s.skill_name for s in next_inject],
        "freeze_organization": cfg.freeze_organization,
        "organization_edit": organization_edit,
        "cluster_report": cluster_report,
        "nominate_admit": org_result,
        "onboarding": onboarding_report,
        "phases": {
            "distill_and_credit": do_distill,
            "update_clusters": do_cluster,
            "evolve_organization": do_org,
        },
        "distill_prior_segments": int(cfg.distill_prior_segments),
        "distill_window": distill_window_meta,
    }
    if segment_dir is not None:
        out = Path(segment_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Merge org-phase fields into an existing collect summary when present.
        summary_path = out / "segment_summary.json"
        if summary_path.exists() and not do_distill and do_org:
            prev = {}
            try:
                from sage_tau2.serialization import read_json

                prev = read_json(summary_path)
            except Exception:
                prev = {}
            if isinstance(prev, dict):
                prev.update(
                    {
                        "organization_edit": organization_edit,
                        "nominate_admit": org_result,
                        "onboarding": onboarding_report,
                        "phases": summary["phases"],
                        "n_skills_in_bank": summary["n_skills_in_bank"],
                        "n_active_skills": summary["n_active_skills"],
                    }
                )
                summary = prev
        write_json(summary_path, summary)
        if do_distill:
            write_json(
                out / "trajectories.json",
                [
                    {
                        "task_id": t.task_id,
                        "reward": t.reward,
                        "db_match": t.db_match,
                        "tool_protocol": t.tool_protocol,
                        "evidence_id": t.evidence_id,
                        "success": t.success,
                    }
                    for t in trajectories
                ],
            )
            write_json(
                out / "new_skills.json",
                [
                    {
                        "skill_id": s.skill_id,
                        "skill_name": s.skill_name,
                        "capability_key": s.capability_key,
                        "action_protocol": s.action_protocol,
                        "status": s.status.value,
                    }
                    for s in accepted
                ],
            )
        if cluster_report is not None:
            write_json(out / "cluster_events.json", cluster_report)
        if org_result is not None:
            write_json(out / "nominate_admit.json", org_result)
        if onboarding_report is not None:
            write_onboarding_report(out / "onboarding.json", onboarding_report)
    return summary
