"""Online segmented SAGE-MAS on ALFWorld: play → distill → evolve → continue."""

from __future__ import annotations

import os
import re
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from examples.prompt_agent.gpt4o_alfworld import (
    _list_unseen_gamefiles,
    _task_family_from_gamefile,
)
from sage_mas.actor_promotion import (
    _with_status,
    apply_actor_promotion_result,
    probe_actor_promotion,
    probation_specialists,
    select_promotion_gamefiles,
)
from sage_mas.cold_start import (
    apply_admit_decisions_to_family_enable,
    dispatch_lists_from_family_enable,
    family_enable_from_dispatch_lists,
)
from sage_mas.alfworld_evaluator import (
    AlfWorldEvaluatorConfig,
    AlfWorldOrganizationEvaluator,
    EvaluationTrial,
    sample_alfworld_gamefiles,
    select_family_proportional_gamefiles,
    select_prompt_agent_gamefiles,
)
from sage_mas.distribution_stats import (
    compare_experience_distribution_snapshots,
)
from sage_mas.evaluation_services import OrganizationShadowService
from sage_mas.executor_dispatch import dispatch_config_from_mapping
from sage_mas.onboarding import (
    acting_status,
    refresh_acting_statuses,
    restore_executor_baseline,
    revive_cross_model_demotions,
)
from sage_mas.pipeline import SageEvolutionPipeline
from sage_mas.runtime import ChatBackend, OpenAIChatBackend
from sage_mas.schemas import AgentSpec, Skill, SkillStatus, ShadowDecision
from sage_mas.serialization import (
    agent_from_dict,
    load_agents,
    load_skills,
    read_json,
    write_json,
    write_jsonl,
)
from sage_mas.shadow_evaluation import ShadowEvaluator
from sage_mas.skill_bank import SkillBank
from sage_mas.skill_credit import (
    skill_credit_policy_from_mapping,
    update_skill_credits,
)
from sage_mas.skill_injection_policy import (
    DEFAULT_BLOCKED_INJECTION_PREFIXES,
    filter_injectable_skills,
    injection_policy_from_mapping,
)
from sage_mas.skill_marginal_utility import (
    apply_marginal_utility_probe,
    probe_skill_marginal_utility,
    select_probe_gamefiles,
)
from sage_mas.specialist_failure_distillation import (
    update_specialist_failure_protocols,
)


@dataclass(slots=True)
class OnlineEvolutionConfig:
    num_games: int = 134
    segment_size: int = 20
    game_selection: str = "first"
    # Online score pool: train | valid_seen | valid_unseen.
    collection_dataset: str = "valid_unseen"
    max_skills_per_round: int = 3
    shadow_pool_size: int = 0
    mechanism_dataset: str = "valid_unseen"
    seed: int = 1
    distill_after_last_segment: bool = True
    stop_when_no_candidates: bool = False
    probation_min_games: int = 8
    probation_min_wins: int = 3
    probation_epsilon: float = 0.05
    remove_after_rejected_windows: int = 2
    # Cross-segment distillation: include this many prior segment trajectories.
    distill_prior_segments: int = 2
    # Pilot: skip paired shadow gating and always commit meaningful org edits.
    skip_shadow: bool = False
    # Pilot: allow committing ADD_AGENT even on the final play segment.
    commit_org_on_last_segment: bool = False
    # Baseline: single Executor only; skip org commits after distillation.
    executor_only: bool = False
    # Skill-only ablation: keep distill/credit/injection, never commit org edits.
    freeze_organization: bool = False
    # Injection policy: block broad procedural capabilities by default.
    block_injection_capability_prefixes: tuple[str, ...] = (
        DEFAULT_BLOCKED_INJECTION_PREFIXES
    )
    inject_capability_allowlist: tuple[str, ...] | None = None
    # Only inject skills with paired MU > threshold (unprobed stay bank-only).
    require_positive_mu_for_injection: bool = False
    min_marginal_utility: float = 0.0
    # After distill, dual-eval provisional skills with vs without injection.
    paired_mu_probe: bool = False
    paired_mu_num_tasks: int = 6
    paired_mu_require_ci: bool = False
    # Skip MU as a bank gate: form_ok distilled skills become verified immediately
    # so ADD_AGENT / injection can proceed without paired ΔSR.
    auto_verify_form_ok_skills: bool = False
    # Re-run the segment without injected skills for relative credit.
    run_control_eval_for_credit: bool = False
    # Propose ADD/ASSIGN for verified bank skills not yet on any agent.
    # Needed when warm-starting C from a B_mu SkillBank (org only sees
    # this-round candidates, not a passive bank).
    propose_org_for_unassigned_skills: bool = False
    # Play like skill-only for the first N segments before committing org edits.
    org_warmup_segments: int = 0
    # Held-out probes: specialist primary vs Executor primary before acting.
    actor_promotion_probe: bool = False
    actor_promotion_num_tasks: int = 8
    # Require specialist_sr >= executor_sr + min_advantage (ties never promote).
    actor_promotion_min_advantage: float = 0.125
    # After Admit / actor-promotion probes, rewrite enabled/disabled_task_families.
    admit_updates_family_enable: bool = True
    # Fixed held-out eval (test curve). Default: only after ADD_AGENT.
    # Set test_eval_every_segment=True (or every_n_segments=1) to probe each segment.
    # every_n_segments>1 runs on multiples of N and always on the final segment.
    test_eval_enabled: bool = False
    test_pool_size: int = 20
    test_eval_dataset: str = "valid_unseen"
    test_eval_every_segment: bool = False
    test_eval_every_n_segments: int = 0
    # Optional higher concurrency for periodic test140 (mini) only.
    test_parallel_envs: int | None = None
    test_api_concurrency: int | None = None
    # Weak model for acceptance gates + held-out test/val.
    # When set, train play keeps executor_model (teacher); mini only gates.
    acceptor_model: str | None = None
    # Mid/deploy model for non-Executor agents during train play.
    # When set, Executor (and teacher collection) stays on executor_model;
    # dispatched specialists call specialist_model (e.g. gemini-2.5-flash).
    specialist_model: str | None = None
    # Probation gate baseline:
    #   same_model (default) — when teacher != gate/specialist model, replay
    #     Executor-only episodes on the gate evaluator (acceptor/flash) using
    #     the specialist's dispatched gamefiles. Never compare Flash specialist
    #     SR against Pro teacher train Executor SR.
    #   train — legacy: use recent teacher-collection Executor trials.
    onboarding_baseline: str = "same_model"
    # On resume, restore demoted specialists to probation when the prior demote
    # was likely a cross-model artifact (teacher vs acceptor/specialist).
    revive_cross_model_demotions: bool = False
    # Independent fixed val pool (valid_seen by default). When enabled,
    # shadow trials are not mixed into val_trials.
    val_eval_enabled: bool = False
    val_pool_size: int = 10
    val_eval_dataset: str = "valid_seen"
    val_game_selection: str = "shuffle"
    val_eval_every_segment: bool = False
    val_eval_every_n_segments: int = 0


def _resolve_openai_api_key(*cfgs: dict[str, Any] | None) -> str | None:
    """Prefer explicit config api_key over ambient OPENAI_API_KEY.

    Stale shell tokens otherwise silently override a working llm_config.yaml
    key and surface as OpenAI 401 Invalid token during play.
    """
    for cfg in cfgs:
        if not isinstance(cfg, dict):
            continue
        value = cfg.get("api_key")
        if value:
            return str(value)
    env = os.environ.get("OPENAI_API_KEY")
    return str(env) if env else None


def infer_dataset_split(task_id: str, explicit: str | None = None) -> str:
    """Map ALFWorld gamefile paths to train / val / test split labels."""
    if explicit:
        normalized = str(explicit).strip().lower()
        aliases = {
            "train": "train",
            "training": "train",
            "valid_seen": "val",
            "validation": "val",
            "val": "val",
            "valid_unseen": "test",
            "test": "test",
        }
        if normalized in aliases:
            return aliases[normalized]
    path = str(task_id or "").replace("\\", "/").lower()
    if "/valid_unseen/" in path:
        return "test"
    if "/valid_seen/" in path:
        return "val"
    if "/train/" in path:
        return "train"
    return "train"


def evaluation_trial_record(
    trial: EvaluationTrial,
    *,
    dataset_split: str,
    round_index: int,
    phase: str,
    train_progress: int,
) -> dict[str, Any]:
    """Serialize one evaluation episode for state + plotting."""
    return {
        "task_id": trial.task_id,
        "task": trial.task,
        "task_family": trial.task_family,
        "won": trial.won,
        "num_steps": trial.num_steps,
        "cost": trial.cost,
        "activated_skill_names": trial.activated_skill_names,
        "assigned_primary_agent": trial.assigned_primary_agent,
        "assignment_rationale": trial.assignment_rationale,
        "eligible_agents": trial.eligible_agents,
        "dispatch_layer": trial.dispatch_layer,
        "dispatch_evidence": trial.dispatch_evidence,
        "actions_by_agent": trial.actions_by_agent,
        "dataset_split": dataset_split,
        "segment": round_index,
        "phase": phase,
        "train_progress_at_eval": train_progress,
    }


def _config_int(mapping: dict[str, Any], key: str, default: int) -> int:
    """Read int configs so an explicit 0 is preserved (unlike ``or`` chains)."""
    if key in mapping and mapping[key] is not None:
        return int(mapping[key])
    return int(default)

def segment_gamefiles(
    gamefiles: list[str],
    segment_size: int,
) -> list[list[str]]:
    """Split games into contiguous, non-overlapping segments."""
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")
    return [
        gamefiles[offset : offset + segment_size]
        for offset in range(0, len(gamefiles), segment_size)
    ]


def summarize_trials(
    trials: list[EvaluationTrial],
    *,
    name: str = "online",
) -> dict[str, Any]:
    wins = sum(int(trial.won) for trial in trials)
    task_wins: dict[str, int] = defaultdict(int)
    task_total: dict[str, int] = defaultdict(int)
    for trial in trials:
        task_total[trial.task_family] += 1
        task_wins[trial.task_family] += int(trial.won)
    num_games = len(trials)
    return {
        "name": name,
        "wins": wins,
        "num_games": num_games,
        "success_rate": (wins / num_games if num_games else 0.0),
        "by_family": {
            family: {
                "wins": task_wins[family],
                "total": task_total[family],
                "success_rate": task_wins[family] / task_total[family],
            }
            for family in sorted(task_total)
        },
        "dispatch": summarize_dispatch(trials),
    }


def _trial_field(trial: Any, name: str, default: Any = None) -> Any:
    if isinstance(trial, dict):
        return trial.get(name, default)
    return getattr(trial, name, default)


def summarize_adaptation_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact segment-status / shadow diagnostics for causal ablations."""
    status_counts: Counter[str] = Counter(
        str(record.get("status") or "unknown") for record in history
    )
    shadow_compared = 0
    shadow_accepted = 0
    new_utilities: list[float] = []
    old_utilities: list[float] = []
    candidate_rounds = 0
    utility_re = re.compile(
        r"new utility ([0-9.]+).*(?:meets or exceeds|does not meet|"
        r"exceeds|does not exceed) required utility ([0-9.]+)",
        re.IGNORECASE,
    )
    for record in history:
        if record.get("candidate_skill_names"):
            candidate_rounds += 1
        status = str(record.get("status") or "")
        # History stores shadow_result as a path; use status + decision text.
        if status in {"accepted", "rolled_back"} and record.get("shadow_decision"):
            shadow_compared += 1
            if status == "accepted":
                shadow_accepted += 1
        decision_text = record.get("shadow_decision")
        if isinstance(decision_text, str):
            match = utility_re.search(decision_text)
            if match:
                new_utilities.append(float(match.group(1)))
                old_utilities.append(float(match.group(2)))
        shadow = record.get("shadow_result")
        if isinstance(shadow, dict):
            decision = shadow.get("decision") or shadow
            if isinstance(decision, dict) and (
                "accepted" in decision or "new_utility" in decision
            ):
                if status not in {"accepted", "rolled_back"}:
                    shadow_compared += 1
                    if bool(decision.get("accepted")):
                        shadow_accepted += 1
                if decision.get("new_utility") is not None:
                    new_utilities.append(float(decision["new_utility"]))
                if decision.get("old_utility") is not None:
                    old_utilities.append(float(decision["old_utility"]))
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "segments_with_candidates": candidate_rounds,
        "org_accepted": status_counts.get("accepted", 0),
        "org_rolled_back": status_counts.get("rolled_back", 0),
        "organization_frozen": status_counts.get("organization_frozen", 0),
        "shadow_compared": shadow_compared,
        "shadow_accepted": shadow_accepted,
        "shadow_accept_rate": (
            shadow_accepted / shadow_compared if shadow_compared else None
        ),
        "mean_shadow_new_utility": (
            sum(new_utilities) / len(new_utilities) if new_utilities else None
        ),
        "mean_shadow_old_utility": (
            sum(old_utilities) / len(old_utilities) if old_utilities else None
        ),
    }


def summarize_dispatch(trials: list[Any]) -> dict[str, Any]:
    """Aggregate routing utilization and conditional outcomes."""
    by_layer: Counter[str] = Counter()
    primary_count: Counter[str] = Counter()
    primary_wins: Counter[str] = Counter()
    actions_by_agent: Counter[str] = Counter()
    eligible_episode_count = 0
    executor_fallback_count = 0
    specialist_primary_count = 0

    for trial in trials:
        layer = str(_trial_field(trial, "dispatch_layer", "") or "unknown")
        by_layer[layer] += 1
        primary = str(
            _trial_field(trial, "assigned_primary_agent", "") or "unassigned"
        )
        primary_count[primary] += 1
        primary_wins[primary] += int(
            bool(_trial_field(trial, "won", False))
        )
        eligible = list(_trial_field(trial, "eligible_agents", []) or [])
        is_executor = "executor" in primary.lower()
        if eligible:
            eligible_episode_count += 1
            if is_executor:
                executor_fallback_count += 1
        if primary != "unassigned" and not is_executor:
            specialist_primary_count += 1
        for agent_name, count in dict(
            _trial_field(trial, "actions_by_agent", {}) or {}
        ).items():
            actions_by_agent[str(agent_name)] += int(count)

    total = len(trials)
    conditional_success_by_agent = {
        agent_name: {
            "wins": primary_wins[agent_name],
            "trials_as_primary": count,
            "success_rate": (
                primary_wins[agent_name] / count if count else 0.0
            ),
            "actions_executed": actions_by_agent.get(agent_name, 0),
        }
        for agent_name, count in sorted(primary_count.items())
    }
    return {
        "dispatch_count_by_agent": dict(sorted(primary_count.items())),
        "specialist_primary_count": specialist_primary_count,
        "specialist_primary_rate": (
            specialist_primary_count / total if total else 0.0
        ),
        "executor_primary_count": sum(
            count
            for agent_name, count in primary_count.items()
            if "executor" in agent_name.lower()
        ),
        "eligible_episode_count": eligible_episode_count,
        "executor_fallback_count": executor_fallback_count,
        "executor_fallback_rate": (
            executor_fallback_count / eligible_episode_count
            if eligible_episode_count
            else 0.0
        ),
        "dispatch_count_by_layer": dict(sorted(by_layer.items())),
        "conditional_success_by_agent": conditional_success_by_agent,
        "actions_executed_by_agent": dict(sorted(actions_by_agent.items())),
    }


class OnlineAlfWorldEvolution:
    """Play ALFWorld in segments; probe/verify/shadow between segments.

    Protocol per segment ``t``:
      1. Collect/evaluate with the current active organization (scores count).
      2. Distill candidates, paired probe + MU verify, update SkillBank.
      3. Shadow-evaluate candidate vs active organization.
      4. Commit accepted organization for segment ``t+1``.

    Final metric is the pooled success rate across all segment trials.
    """

    def __init__(
        self,
        llm_config: dict[str, Any],
        sage_config: dict[str, Any],
        output_root: str | Path,
        backend: ChatBackend | None = None,
        evaluator: AlfWorldOrganizationEvaluator | None = None,
    ):
        self.llm_config = llm_config
        self.sage_config = deepcopy(sage_config)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        online_cfg = self.sage_config.get("sage", {}).get("online", {})
        loop_cfg = self.sage_config.get("sage", {}).get("loop", {})
        shadow_cfg = self.sage_config.get("sage", {}).get("shadow", {})
        credit_cfg_early = self.sage_config.get("sage", {}).get(
            "skill_credit",
            {},
        )
        max_skills_per_round = int(
            online_cfg.get(
                "max_skills_per_round",
                loop_cfg.get("max_skills_per_round", 3),
            )
        )
        num_games = int(online_cfg.get("num_games", 134))
        segment_size = int(
            online_cfg.get(
                "segment_size",
                loop_cfg.get("collection_tasks_per_round", 20),
            )
        )
        num_segments = max(
            1,
            (num_games + segment_size - 1) // segment_size,
        )
        shadow_tasks_per_round = int(shadow_cfg.get("num_tasks", 2))
        shadow_enabled = bool(shadow_cfg.get("enabled", True))
        skip_shadow = bool(
            online_cfg.get("skip_shadow", not shadow_enabled)
        )
        shadow_pool_size = _config_int(online_cfg, "shadow_pool_size", -1)
        if shadow_pool_size < 0:
            shadow_pool_size = _config_int(loop_cfg, "shadow_pool_size", -1)
        if shadow_pool_size < 0:
            shadow_pool_size = _config_int(shadow_cfg, "pool_size", -1)
        if shadow_pool_size < 0:
            shadow_pool_size = (
                0
                if skip_shadow
                else shadow_tasks_per_round * num_segments
            )
        if skip_shadow:
            shadow_pool_size = 0
        distill_cfg = self.sage_config.get("sage", {}).get("distillation") or {}
        _injection_policy = injection_policy_from_mapping(
            {**distill_cfg, **online_cfg}
        )
        # Prefer sage.online.*; fall back to loop.* for convenience.
        self.config = OnlineEvolutionConfig(
            num_games=num_games,
            segment_size=segment_size,
            game_selection=str(
                online_cfg.get("game_selection", "first")
            ).lower(),
            collection_dataset=str(
                online_cfg.get("collection_dataset", "valid_unseen")
            ).lower(),
            max_skills_per_round=max_skills_per_round,
            shadow_pool_size=shadow_pool_size,
            mechanism_dataset=str(
                online_cfg.get("mechanism_dataset", "valid_unseen")
            ).lower(),
            seed=int(online_cfg.get("seed", loop_cfg.get("seed", 1))),
            distill_after_last_segment=bool(
                online_cfg.get("distill_after_last_segment", True)
            ),
            stop_when_no_candidates=bool(
                online_cfg.get(
                    "stop_when_no_candidates",
                    loop_cfg.get("stop_when_no_candidates", False),
                )
            ),
            probation_min_games=int(online_cfg.get("probation_min_games", 8)),
            probation_min_wins=int(online_cfg.get("probation_min_wins", 3)),
            probation_epsilon=float(online_cfg.get("probation_epsilon", 0.05)),
            remove_after_rejected_windows=int(
                online_cfg.get("remove_after_rejected_windows", 2)
            ),
            distill_prior_segments=int(
                online_cfg.get("distill_prior_segments", 2)
            ),
            skip_shadow=skip_shadow,
            commit_org_on_last_segment=bool(
                online_cfg.get("commit_org_on_last_segment", skip_shadow)
            ),
            executor_only=bool(online_cfg.get("executor_only", False)),
            freeze_organization=bool(
                online_cfg.get(
                    "freeze_organization",
                    online_cfg.get("executor_only", False),
                )
            ),
            block_injection_capability_prefixes=tuple(
                _injection_policy["block_prefixes"]
            ),
            inject_capability_allowlist=(
                None
                if _injection_policy["allow_prefixes"] is None
                else tuple(_injection_policy["allow_prefixes"])
            ),
            require_positive_mu_for_injection=bool(
                online_cfg.get(
                    "require_positive_mu_for_injection",
                    credit_cfg_early.get(
                        "require_positive_mu_for_verify",
                        False,
                    ),
                )
            ),
            min_marginal_utility=float(
                online_cfg.get(
                    "min_marginal_utility",
                    credit_cfg_early.get("min_marginal_utility", 0.0),
                )
            ),
            paired_mu_probe=bool(
                online_cfg.get(
                    "paired_mu_probe",
                    credit_cfg_early.get(
                        "require_positive_mu_for_verify",
                        False,
                    ),
                )
            ),
            paired_mu_num_tasks=int(
                online_cfg.get(
                    "paired_mu_num_tasks",
                    credit_cfg_early.get("paired_mu_num_tasks", 6),
                )
            ),
            paired_mu_require_ci=bool(
                online_cfg.get(
                    "paired_mu_require_ci",
                    credit_cfg_early.get("paired_mu_require_ci", False),
                )
            ),
            auto_verify_form_ok_skills=bool(
                online_cfg.get("auto_verify_form_ok_skills", False)
            ),
            run_control_eval_for_credit=bool(
                online_cfg.get(
                    "run_control_eval_for_credit",
                    credit_cfg_early.get(
                        "relative_credit_to_baseline",
                        False,
                    ),
                )
            ),
            propose_org_for_unassigned_skills=bool(
                online_cfg.get("propose_org_for_unassigned_skills", False)
            ),
            org_warmup_segments=int(
                online_cfg.get("org_warmup_segments", 0)
            ),
            actor_promotion_probe=bool(
                online_cfg.get("actor_promotion_probe", False)
            ),
            actor_promotion_num_tasks=int(
                online_cfg.get("actor_promotion_num_tasks", 8)
            ),
            actor_promotion_min_advantage=float(
                online_cfg.get("actor_promotion_min_advantage", 0.125)
            ),
            admit_updates_family_enable=bool(
                online_cfg.get("admit_updates_family_enable", True)
            ),
            test_eval_enabled=bool(
                (online_cfg.get("test_eval") or {}).get("enabled", False)
            ),
            test_pool_size=int(
                (online_cfg.get("test_eval") or {}).get(
                    "pool_size",
                    (online_cfg.get("test_eval") or {}).get("num_tasks", 20),
                )
            ),
            test_eval_dataset=str(
                (online_cfg.get("test_eval") or {}).get(
                    "dataset",
                    "valid_unseen",
                )
            ).lower(),
            test_eval_every_segment=bool(
                (online_cfg.get("test_eval") or {}).get(
                    "every_segment",
                    False,
                )
            ),
            test_eval_every_n_segments=int(
                (online_cfg.get("test_eval") or {}).get(
                    "every_n_segments",
                    0,
                )
            ),
            test_parallel_envs=(
                int((online_cfg.get("test_eval") or {})["parallel_envs"])
                if (online_cfg.get("test_eval") or {}).get("parallel_envs")
                is not None
                else None
            ),
            test_api_concurrency=(
                int((online_cfg.get("test_eval") or {})["api_concurrency"])
                if (online_cfg.get("test_eval") or {}).get("api_concurrency")
                is not None
                else None
            ),
            acceptor_model=(
                str(online_cfg["acceptor_model"]).strip()
                if online_cfg.get("acceptor_model")
                else (
                    str((online_cfg.get("test_eval") or {}).get("model")).strip()
                    if (online_cfg.get("test_eval") or {}).get("model")
                    else None
                )
            )
            or None,
            specialist_model=(
                str(online_cfg["specialist_model"]).strip()
                if online_cfg.get("specialist_model")
                else None
            )
            or None,
            onboarding_baseline=str(
                online_cfg.get("onboarding_baseline", "same_model")
            )
            .strip()
            .lower()
            or "same_model",
            revive_cross_model_demotions=bool(
                online_cfg.get("revive_cross_model_demotions", False)
            ),
            val_eval_enabled=bool(
                (online_cfg.get("val_eval") or {}).get("enabled", False)
            ),
            val_pool_size=int(
                (online_cfg.get("val_eval") or {}).get(
                    "pool_size",
                    (online_cfg.get("val_eval") or {}).get("num_tasks", 10),
                )
            ),
            val_eval_dataset=str(
                (online_cfg.get("val_eval") or {}).get(
                    "dataset",
                    "valid_seen",
                )
            ).lower(),
            val_game_selection=str(
                (online_cfg.get("val_eval") or {}).get("game_selection", "shuffle")
            ).strip().lower(),
            val_eval_every_segment=bool(
                (online_cfg.get("val_eval") or {}).get(
                    "every_segment",
                    True,
                )
            ),
            val_eval_every_n_segments=int(
                (online_cfg.get("val_eval") or {}).get(
                    "every_n_segments",
                    0,
                )
            ),
        )
        credit_cfg = credit_cfg_early
        self.skill_credit_policy = skill_credit_policy_from_mapping(
            credit_cfg
        )

        self.active_organization_path = (
            self.output_root / "active_organization.json"
        )
        self.baseline_organization_path = (
            self.output_root / "baseline_organization.json"
        )
        self.state_path = self.output_root / "online_state.json"
        self.skill_bank_path = self.output_root / "skill_bank.json"
        self.sage_config.setdefault("sage", {})["skill_bank_path"] = str(
            self.skill_bank_path
        )

        self.backend = backend or (
            getattr(evaluator, "backend", None)
            if evaluator is not None
            else self._build_backend()
        )
        self.distillation_backend = self._build_distillation_backend()
        self.specialist_backend = self._build_specialist_backend()
        if evaluator is not None:
            self.evaluator = evaluator
            if (
                self.specialist_backend is not None
                and getattr(self.evaluator, "specialist_backend", None) is None
            ):
                self.evaluator.specialist_backend = self.specialist_backend
        else:
            self.evaluator = AlfWorldOrganizationEvaluator(
                self.backend,
                self._evaluator_config(),
                specialist_backend=self.specialist_backend,
            )
        self.acceptor_backend = self._build_acceptor_backend()
        self.acceptor_evaluator = (
            AlfWorldOrganizationEvaluator(
                self.acceptor_backend,
                self._evaluator_config(),
            )
            if self.acceptor_backend is not None
            else None
        )
        self._gate_policy_meta = self._enforce_mid_model_promotion_gates()
        self._family_enable: dict[str, Any] = {}
        self._initialize_files()
        self._init_family_enable_table()

    def _dispatch_config_mapping(self) -> dict[str, Any]:
        online_cfg = self.sage_config.get("sage", {}).get("online") or {}
        return dict(
            self.sage_config.get("sage", {}).get("executor_dispatch")
            or online_cfg.get("executor_dispatch")
            or {}
        )

    def _init_family_enable_table(
        self,
        state: dict[str, Any] | None = None,
    ) -> None:
        """Load runtime enable table from state, disk, or dispatch seed lists."""
        table: dict[str, Any] | None = None
        if state and isinstance(state.get("family_enable"), dict):
            table = {
                str(key): dict(row or {})
                for key, row in state["family_enable"].items()
            }
        path = self.output_root / "family_enable.json"
        if table is None and path.is_file():
            loaded = read_json(path)
            if isinstance(loaded, dict):
                table = {
                    str(key): dict(row or {})
                    for key, row in loaded.items()
                }
        if table is None:
            raw = self._dispatch_config_mapping()
            table = family_enable_from_dispatch_lists(
                enabled=raw.get("enabled_task_families"),
                disabled=raw.get("disabled_task_families"),
            )
        self._apply_family_enable_to_runtime(table, persist=True)

    def _apply_family_enable_to_runtime(
        self,
        table: dict[str, Any],
        *,
        persist: bool = True,
    ) -> None:
        """Push enable table into sage_config + live evaluator dispatchers."""
        self._family_enable = {
            str(key): dict(row or {}) for key, row in (table or {}).items()
        }
        enabled, disabled = dispatch_lists_from_family_enable(self._family_enable)
        sage = self.sage_config.setdefault("sage", {})
        ed = sage.setdefault("executor_dispatch", {})
        ed["enabled_task_families"] = list(enabled)
        ed["disabled_task_families"] = list(disabled)
        for evaluator in (self.evaluator, self.acceptor_evaluator):
            if evaluator is None:
                continue
            dispatcher = getattr(evaluator, "dispatcher", None)
            if dispatcher is None:
                continue
            cfg = getattr(dispatcher, "config", None)
            if cfg is None:
                continue
            cfg.enabled_task_families = list(enabled)
            cfg.disabled_task_families = list(disabled)
        if persist:
            write_json(self.output_root / "family_enable.json", self._family_enable)

    def _update_family_enable_from_admit(
        self,
        *,
        agents: list[AgentSpec],
        promotion: dict[str, Any],
        state: dict[str, Any],
        attempt_dir: Path,
    ) -> dict[str, Any]:
        """Admit / actor-promotion → rewrite enabled/disabled_task_families."""
        if not self.config.admit_updates_family_enable:
            return {
                "updated": False,
                "reason": "admit_updates_family_enable=false",
                "table": dict(self._family_enable or {}),
            }
        decisions = [
            decision
            for decision in (promotion.get("decisions") or [])
            if isinstance(decision, dict)
        ]
        if not decisions:
            return {
                "updated": False,
                "reason": "no_admit_decisions",
                "table": dict(self._family_enable or {}),
            }
        table, updates = apply_admit_decisions_to_family_enable(
            self._family_enable,
            agents=agents,
            decisions=decisions,
            source="actor_promotion",
        )
        self._apply_family_enable_to_runtime(table, persist=True)
        enabled, disabled = dispatch_lists_from_family_enable(table)
        payload = {
            "updated": bool(updates),
            "updates": updates,
            "enabled_task_families": enabled,
            "disabled_task_families": disabled,
            "table": table,
        }
        state["family_enable"] = table
        write_json(attempt_dir / "family_enable_update.json", payload)
        write_json(self.state_path, state)
        return payload

    def _enforce_mid_model_promotion_gates(self) -> dict[str, Any]:
        """Teacher discovers; mid-model gates specialist acting / test.

        Exp B contract:
          - Pro (teacher): collection play + distill + LLM dispatch
          - Flash (acceptor/specialist): onboarding baseline, actor promotion,
            test140
          - Skills: form_ok auto-verify (no paired-MU bank gate by default)

        Never allow Pro train Executor SR to accept/demote Flash specialists.
        """
        meta: dict[str, Any] = {
            "teacher_model": self._teacher_model_name(),
            "gate_model": self._gate_model_name(),
            "models_differ": False,
            "forced": [],
        }
        if not self._teacher_differs_from_gate():
            return meta
        meta["models_differ"] = True
        forced: list[str] = []
        mode = str(self.config.onboarding_baseline or "").strip().lower()
        if mode in {"", "train", "teacher", "legacy"}:
            self.config = replace(self.config, onboarding_baseline="same_model")
            forced.append("onboarding_baseline=same_model")
        # Keep revive available for resumes that demoted under the old bug.
        if not self.config.revive_cross_model_demotions:
            self.config = replace(
                self.config,
                revive_cross_model_demotions=True,
            )
            forced.append("revive_cross_model_demotions=true")
        # Prefer fast form-ok bank entry unless the config explicitly wants MU gates.
        if (
            not self.config.auto_verify_form_ok_skills
            and not self.config.require_positive_mu_for_injection
            and not self.skill_credit_policy.require_positive_mu_for_verify
        ):
            self.config = replace(self.config, auto_verify_form_ok_skills=True)
            forced.append("auto_verify_form_ok_skills=true")
        meta["forced"] = forced
        meta["onboarding_baseline"] = self.config.onboarding_baseline
        meta["paired_mu_probe"] = self.config.paired_mu_probe
        meta["auto_verify_form_ok_skills"] = (
            self.config.auto_verify_form_ok_skills
        )
        meta["require_positive_mu_for_verify"] = (
            self.skill_credit_policy.require_positive_mu_for_verify
        )
        return meta

    def run(self) -> dict[str, Any]:
        state = read_json(self.state_path)
        state.setdefault("val_trials", [])
        state.setdefault("test_trials", [])
        state.setdefault("learning_curve_checkpoints", [])
        if self.config.revive_cross_model_demotions:
            agents = load_agents(self.active_organization_path)
            org_cfg = self.sage_config.get("sage", {}).get("organization") or {}
            revive = revive_cross_model_demotions(
                agents,
                probation_games=int(org_cfg.get("probation_games", 3)),
            )
            if revive.get("changed"):
                write_json(
                    self.active_organization_path,
                    {"agents": agents},
                )
                write_json(
                    self.output_root / "revive_cross_model_demotions.json",
                    revive,
                )
        if getattr(self, "_gate_policy_meta", None):
            write_json(
                self.output_root / "mid_model_gate_policy.json",
                self._gate_policy_meta,
            )
        self._ensure_test_pool(state)
        self._ensure_val_pool(state)
        self._ensure_mechanism_pools(state)
        self._init_family_enable_table(state)
        segments: list[list[str]] = state["segments"]
        start = int(state.get("completed_segments", 0)) + 1
        for round_index in range(start, len(segments) + 1):
            outcome = self.run_segment(round_index, state)
            state = self._commit_segment(state, outcome)
            if (
                outcome["status"] == "no_candidates"
                and self.config.stop_when_no_candidates
                and round_index < len(segments)
            ):
                # Still finish remaining segments with the frozen org.
                pass

        summary = self._finalize_summary(state)
        state["summary"] = summary
        write_json(self.state_path, state)
        write_json(self.output_root / "online_summary.json", summary)
        return state

    def run_segment(
        self,
        round_index: int,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = state or read_json(self.state_path)
        self._ensure_mechanism_pools(state)
        segments: list[list[str]] = state["segments"]
        is_last = round_index == len(segments)
        bank = SkillBank(self.skill_bank_path)
        gamefiles, task_prioritization = self._prioritize_probation_tasks(
            state,
            round_index,
            bank.credit_skills(),
        )

        attempt_dir = (
            self.output_root
            / f"segment_{round_index:03d}"
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        attempt_dir.mkdir(parents=True, exist_ok=False)

        active_agents = load_agents(self.active_organization_path)
        # Include provisional skills so the recall (BM25) index can surface
        # them; the evaluator's own status filters (e.g. VERIFIED-only for
        # assigned-skill mounting) still apply downstream.
        active_skills = [
            skill
            for skill in bank.skills
            if skill.status in {SkillStatus.PROVISIONAL, SkillStatus.VERIFIED}
        ]
        executor_skill_names = {
            skill_name
            for agent in active_agents
            if "executor" in f"{agent.name} {agent.role}".lower()
            for skill_name in (agent.assigned_skills or [])
        }
        specialist_only_skill_names = {
            skill_name
            for agent in active_agents
            if "executor" not in f"{agent.name} {agent.role}".lower()
            for skill_name in (agent.assigned_skills or [])
            if skill_name not in executor_skill_names
        }
        # Inject provisional/shared credit skills for Executor. Skills that exist
        # only on specialists stay specialist-owned; shared copies on Executor
        # are rendered via assigned-skill injection in the runtime prompt.
        # Broad execution.* summaries are blocked unless explicitly allowlisted.
        executor_credit_skills = filter_injectable_skills(
            [
                skill
                for skill in bank.credit_skills()
                if skill.skill_name not in specialist_only_skill_names
                and self._has_executor_skill_contract(skill)
            ],
            block_prefixes=self.config.block_injection_capability_prefixes,
            allow_prefixes=self.config.inject_capability_allowlist,
            require_positive_mu=self.config.require_positive_mu_for_injection,
            min_marginal_utility=self.config.min_marginal_utility,
        )
        online_cfg = self.sage_config.get("sage", {}).get("online", {}) or {}
        # Skill selection is prompt-based inside the evaluator; Executor skill
        # injection stays enabled so retrieved protocols can be mounted.
        if hasattr(self.evaluator, "config"):
            self.evaluator.config.allow_executor_skill_injection = True
        if self.acceptor_evaluator is not None and hasattr(
            self.acceptor_evaluator, "config"
        ):
            self.acceptor_evaluator.config.allow_executor_skill_injection = True

        train_progress_base = len(self._load_prior_trials(state))
        collection_split = infer_dataset_split(
            gamefiles[0] if gamefiles else "",
            self.config.collection_dataset,
        )

        def _record_trial(
            trial: EvaluationTrial,
            *,
            progress: int,
            phase: str = "collection",
            dataset_split: str = collection_split,
        ) -> dict[str, Any]:
            return evaluation_trial_record(
                trial,
                dataset_split=dataset_split,
                round_index=round_index,
                phase=phase,
                train_progress=progress,
            )

        trials = self.evaluator.evaluate(
            agents=active_agents,
            skills=active_skills,
            gamefiles=gamefiles,
            condition=f"online_segment_{round_index}",
            injected_skills=executor_credit_skills or None,
        )
        control_trials: list[EvaluationTrial] | None = None
        if self.config.run_control_eval_for_credit and gamefiles:
            control_trials = self.evaluator.evaluate(
                agents=active_agents,
                skills=active_skills,
                gamefiles=gamefiles,
                condition=f"online_segment_{round_index}_control",
                injected_skills=[],
            )
            write_json(
                attempt_dir / "control_segment_trials.json",
                [
                    _record_trial(
                        trial,
                        progress=train_progress_base + index + 1,
                        phase="control",
                    )
                    for index, trial in enumerate(control_trials)
                ],
            )
        credit_updates = update_skill_credits(
            bank.credit_skills(),
            trials,
            active_agents,
            self.skill_credit_policy,
            control_trials=control_trials,
        )
        controllers_enabled = bool(
            getattr(
                getattr(self.evaluator, "config", None),
                "enable_specialist_controllers",
                False,
            )
        )
        if controllers_enabled:
            failure_protocol_updates = {
                "updated": [],
                "updated_skill_count": 0,
                "skipped": {"controller_enabled": len(trials)},
            }
        else:
            failure_protocol_updates = update_specialist_failure_protocols(
                skills=bank.skills,
                agents=active_agents,
                trials=trials,
                require_no_controller=True,
            )
        bank.save()
        # Probation gate must compare specialist vs same-model Executor.
        # Teacher (Pro) train Executor SR is not a valid baseline when
        # specialists/test run on acceptor/specialist (Flash).
        prior = self._load_prior_trials(state)
        recent_window = max(30, 3 * max(1, self.config.segment_size))
        train_baseline = (prior + trials)[-recent_window:]
        baseline_trials, baseline_meta = self._resolve_onboarding_baseline(
            agents=active_agents,
            train_trials=trials,
            train_baseline=train_baseline,
            skills=bank.stable_skills(),
            attempt_dir=attempt_dir,
        )
        onboarding = refresh_acting_statuses(
            active_agents,
            trials,
            baseline_trials=baseline_trials,
            min_games=self.config.probation_min_games,
            min_wins=self.config.probation_min_wins,
            epsilon=self.config.probation_epsilon,
            remove_after_rejected_windows=(
                self.config.remove_after_rejected_windows
            ),
            require_controller=bool(
                self.sage_config.get("sage", {})
                .get("online", {})
                .get("enable_specialist_controllers", False)
            ),
            min_new_agent_call_rate=float(
                self.sage_config.get("sage", {})
                .get("organization", {})
                .get(
                    "min_new_agent_call_rate",
                    self.sage_config.get("sage", {})
                    .get("shadow", {})
                    .get("min_new_agent_call_rate", 0.5),
                )
            ),
        )
        onboarding["baseline_meta"] = baseline_meta
        for decision in onboarding.get("decisions") or []:
            if isinstance(decision, dict):
                decision.setdefault(
                    "baseline_source",
                    baseline_meta.get("source"),
                )
        if onboarding.get("changed"):
            write_json(
                self.active_organization_path,
                {"agents": active_agents},
            )
            write_json(attempt_dir / "onboarding_decisions.json", onboarding)
        elif baseline_meta.get("source"):
            write_json(attempt_dir / "onboarding_baseline_meta.json", baseline_meta)

        promotion = self._run_actor_promotion_probes(
            agents=active_agents,
            state=state,
            attempt_dir=attempt_dir,
        )
        if promotion.get("changed"):
            write_json(
                self.active_organization_path,
                {"agents": active_agents},
            )
        family_enable_update = self._update_family_enable_from_admit(
            agents=active_agents,
            promotion=promotion,
            state=state,
            attempt_dir=attempt_dir,
        )

        segment_metrics = summarize_trials(
            trials,
            name=f"segment_{round_index:03d}",
        )
        trajectory_path = attempt_dir / "trajectories.jsonl"
        write_jsonl(
            trajectory_path,
            [
                self._trial_to_trajectory(trial, round_index)
                for trial in trials
            ],
        )
        write_json(
            attempt_dir / "segment_trials.json",
            [
                _record_trial(
                    trial,
                    progress=train_progress_base + index + 1,
                )
                for index, trial in enumerate(trials)
            ],
        )
        write_jsonl(
            attempt_dir / "dispatch_log.jsonl",
            [
                {
                    "task_id": trial.task_id,
                    "task_family": trial.task_family,
                    "task": trial.task,
                    "won": trial.won,
                    "assigned_primary_agent": trial.assigned_primary_agent,
                    "eligible_agents": trial.eligible_agents,
                    "dispatch_layer": trial.dispatch_layer,
                    "dispatch_evidence": trial.dispatch_evidence,
                    "actions_by_agent": trial.actions_by_agent,
                    "assignment_rationale": trial.assignment_rationale,
                }
                for trial in trials
            ],
        )

        outcome: dict[str, Any] = {
            "round": round_index,
            "attempt_dir": str(attempt_dir),
            "gamefiles": gamefiles,
            "trajectory_path": str(trajectory_path),
            "agent_count": len(active_agents),
            "agent_names": [agent.name for agent in active_agents],
            "skill_bank_size": len(SkillBank(self.skill_bank_path).skills),
            "segment_metrics": segment_metrics,
            "task_prioritization": task_prioritization,
            "onboarding": onboarding,
            "actor_promotion": promotion,
            "family_enable_update": family_enable_update,
            "skill_credit": {
                "injected_into_executor": [
                    skill.skill_name for skill in executor_credit_skills
                ],
                "updated": [
                    {
                        "skill_name": skill.skill_name,
                        "status": skill.status.value,
                        "score": skill.metadata["skill_credit"]["score"],
                        "uses": skill.metadata["skill_credit"]["uses"],
                        "successes": skill.metadata["skill_credit"][
                            "successes"
                        ],
                    }
                    for skill in credit_updates["updated"]
                ],
                "promoted": [
                    skill.skill_name for skill in credit_updates["promoted"]
                ],
                "demoted": [
                    skill.skill_name for skill in credit_updates["demoted"]
                ],
                "retired": [
                    skill.skill_name for skill in credit_updates["retired"]
                ],
            },
            "failure_protocol_distillation": failure_protocol_updates,
            "trials": [
                _record_trial(
                    trial,
                    progress=train_progress_base + index + 1,
                )
                for index, trial in enumerate(trials)
            ],
        }

        should_distill = (not is_last) or self.config.distill_after_last_segment
        if not should_distill:
            outcome["status"] = "collect_only"
            write_json(attempt_dir / "segment_outcome.json", outcome)
            return outcome

        pipeline = SageEvolutionPipeline(
            self.sage_config,
            distillation_backend=self.distillation_backend,
        )
        distill_path = self._prepare_distill_trajectory_window(
            state,
            trajectory_path=trajectory_path,
            attempt_dir=attempt_dir,
        )
        bank_org_candidates = self._unassigned_verified_skills_for_org(
            bank,
            active_agents,
        )
        if round_index <= self.config.org_warmup_segments:
            bank_org_candidates = []
        defer_org = bool(
            self.config.paired_mu_probe
            and self.config.require_positive_mu_for_injection
        )
        distill_artifacts = pipeline.run(
            trajectory_path=distill_path,
            output_root=attempt_dir / "distillation",
            active_organization_path=self.active_organization_path,
            max_candidates=self.config.max_skills_per_round,
            additional_candidates=(
                list(credit_updates["promoted"]) + bank_org_candidates
            ),
            skip_organization=defer_org,
        )
        candidates = [
            skill
            for skill in load_skills(distill_artifacts.candidate_skills)
            if skill.status not in {SkillStatus.REJECTED, SkillStatus.RETIRED}
        ]
        write_json(attempt_dir / "selected_candidate_skills.json", candidates)
        outcome["candidate_skill_names"] = [
            skill.skill_name for skill in candidates
        ]
        outcome["verified_skill_names"] = [
            skill.skill_name
            for skill in candidates
            if skill.status == SkillStatus.VERIFIED
        ]
        outcome["distillation"] = distill_artifacts.run_dir
        outcome["skill_bank"] = distill_artifacts.skill_bank
        outcome["organization_edits"] = distill_artifacts.organization_edits
        outcome["experience_distribution_snapshot"] = (
            distill_artifacts.experience_distribution_snapshot
        )
        outcome["capability_contracts"] = (
            distill_artifacts.capability_contracts
        )
        if self.config.paired_mu_probe:
            outcome["marginal_utility_probes"] = self._run_paired_mu_probes(
                agents=active_agents,
                seed_trials=trials,
                attempt_dir=attempt_dir,
                state=state,
            )
            if defer_org:
                # ADD_AGENT only after skills prove positive paired MU.
                distill_artifacts = pipeline.repropose_organization(
                    output_root=attempt_dir / "organization_after_mu",
                    active_organization_path=self.active_organization_path,
                    skill_bank_path=self.skill_bank_path,
                )
                outcome["organization_edits"] = (
                    distill_artifacts.organization_edits
                )
                outcome["capability_contracts"] = (
                    distill_artifacts.capability_contracts
                )
                outcome["experience_distribution_snapshot"] = (
                    distill_artifacts.experience_distribution_snapshot
                )
                outcome["organization_after_mu"] = distill_artifacts.run_dir

        if not candidates:
            outcome["status"] = "no_candidates"
            write_json(attempt_dir / "segment_outcome.json", outcome)
            return outcome

        edits = read_json(distill_artifacts.organization_edits)
        meaningful_edits = [
            edit for edit in edits if edit.get("edit_type") != "do_nothing"
        ]
        allow_last_commit = self.config.commit_org_on_last_segment
        if not meaningful_edits:
            outcome["status"] = "no_edit"
            outcome["candidate_organization"] = (
                distill_artifacts.candidate_organization
            )
            write_json(attempt_dir / "segment_outcome.json", outcome)
            return outcome
        if is_last and not allow_last_commit:
            # On the last segment, bank/skills may update but there is no next
            # play window for a new organization.
            outcome["status"] = "distilled_no_org_switch"
            outcome["candidate_organization"] = (
                distill_artifacts.candidate_organization
            )
            write_json(attempt_dir / "segment_outcome.json", outcome)
            return outcome
        if self.config.executor_only:
            outcome["status"] = "executor_only_frozen_org"
            outcome["candidate_organization"] = (
                distill_artifacts.candidate_organization
            )
            write_json(attempt_dir / "segment_outcome.json", outcome)
            return outcome
        if self.config.freeze_organization:
            # Skill-only arm: SkillBank/credit may update, roster stays fixed.
            outcome["status"] = "organization_frozen"
            outcome["candidate_organization"] = (
                distill_artifacts.candidate_organization
            )
            outcome["organization_edits"] = (
                distill_artifacts.organization_edits
            )
            write_json(attempt_dir / "segment_outcome.json", outcome)
            return outcome
        if round_index <= self.config.org_warmup_segments:
            outcome["status"] = "organization_warmup"
            outcome["candidate_organization"] = (
                distill_artifacts.candidate_organization
            )
            outcome["organization_edits"] = (
                distill_artifacts.organization_edits
            )
            write_json(attempt_dir / "segment_outcome.json", outcome)
            return outcome

        candidate_agents = load_agents(distill_artifacts.candidate_organization)
        if self.config.skip_shadow:
            active_names = {agent.name for agent in active_agents}
            new_probation_agents = [
                agent.name
                for agent in candidate_agents
                if agent.name not in active_names
                and str(
                    (agent.shadow_evaluation_record or {}).get(
                        "acting_status",
                        "",
                    )
                ).lower()
                == "probation"
            ]
            selected_organization_path = attempt_dir / "selected_organization.json"
            write_json(selected_organization_path, {"agents": candidate_agents})
            outcome.update(
                {
                    "status": (
                        "deferred_probation"
                        if new_probation_agents
                        else "accepted"
                    ),
                    "selected_organization": str(selected_organization_path),
                    "candidate_organization": distill_artifacts.candidate_organization,
                    "next_agent_count": len(candidate_agents),
                    "next_agent_names": [agent.name for agent in candidate_agents],
                    "shadow_result": None,
                    "shadow_decision": (
                        "Shadow skipped; new specialists remain dispatch-only "
                        "probation until one real primary-dispatch task is won."
                        if new_probation_agents
                        else "skipped: online.skip_shadow=true"
                    ),
                    "probation_agent_names": new_probation_agents,
                }
            )
            write_json(attempt_dir / "segment_outcome.json", outcome)
            return outcome

        stable_skills = SkillBank(self.skill_bank_path).stable_skills()
        shadow_families = self._shadow_capability_families(
            active_agents,
            candidate_agents,
            stable_skills,
        )
        shadow_gamefiles = self._round_shadow_tasks(
            state.get("shadow_pool", []),
            round_index,
            required_families=shadow_families,
        )
        shadow_outcome = OrganizationShadowService(
            evaluator=self._gate_evaluator(),
            gamefiles=shadow_gamefiles,
            shadow_evaluator=self._shadow_evaluator(),
        ).run(
            old_agents=active_agents,
            new_agents=candidate_agents,
            skills=stable_skills,
        )
        old_experience = pipeline.preview_experience_distribution(
            [
                self._trial_to_trajectory(trial, round_index)
                for trial in shadow_outcome.old_trials
            ],
            stable_skills,
            round_id=f"round-{round_index}-old-organization",
        )
        new_experience = pipeline.preview_experience_distribution(
            [
                self._trial_to_trajectory(trial, round_index)
                for trial in shadow_outcome.new_trials
            ],
            stable_skills,
            round_id=f"round-{round_index}-new-organization",
        )
        experience_comparison = compare_experience_distribution_snapshots(
            old_experience,
            new_experience,
        )
        specialist_evidence = self._specialist_shadow_utilization(
            active_agents,
            candidate_agents,
            shadow_outcome.new_trials,
            required_families=shadow_families,
        )
        shadow_decision = self._add_agent_shadow_decision(
            old_trials=shadow_outcome.old_trials,
            new_trials=shadow_outcome.new_trials,
            required_families=shadow_families,
            specialist_evidence=specialist_evidence,
            fallback_decision=shadow_outcome.decision,
            skills=stable_skills,
            old_agents=active_agents,
            new_agents=candidate_agents,
        )
        new_specialist_names = set(
            specialist_evidence["new_specialist_names"]
        )
        shadow_path = attempt_dir / "shadow_result.json"
        segment_train_progress = train_progress_base + len(trials)
        shadow_new_records = [
            _record_trial(
                trial,
                progress=segment_train_progress,
                phase="shadow_new",
                dataset_split="val",
            )
            for trial in shadow_outcome.new_trials
        ]
        # Periodic val_eval owns the learning-curve val channel; do not mix
        # organization-shadow trials into state["val_trials"].
        if not self.config.val_eval_enabled:
            outcome["val_trials"] = shadow_new_records
        write_json(
            shadow_path,
            {
                "protocol": "paired_organization_fork_v4",
                "held_out_gamefiles": shadow_gamefiles,
                "old": {
                    "metrics": shadow_outcome.old_metrics,
                    "experience_distribution": old_experience,
                    "trials": [
                        _record_trial(
                            trial,
                            progress=segment_train_progress,
                            phase="shadow_old",
                            dataset_split="val",
                        )
                        for trial in shadow_outcome.old_trials
                    ],
                },
                "new": {
                    "metrics": shadow_outcome.new_metrics,
                    "experience_distribution": new_experience,
                    "trials": shadow_new_records,
                },
                "experience_distribution_comparison": experience_comparison,
                "specialist_utilization": specialist_evidence,
                "decision": shadow_decision,
                "global_decision": shadow_outcome.decision,
            },
        )
        defer_for_missing_evidence = (
            bool(new_specialist_names)
            and not specialist_evidence["evidence_sufficient"]
        )
        # Never roll back an untested ADD_AGENT: if the new specialist was not
        # primary-dispatched in shadow, defer to online probation instead of
        # treating a missing comparison as failure.
        if defer_for_missing_evidence:
            selected_organization_path = attempt_dir / "selected_organization.json"
            write_json(selected_organization_path, {"agents": candidate_agents})
            outcome.update(
                {
                    "status": "deferred_probation",
                    "selected_organization": str(selected_organization_path),
                    "candidate_organization": distill_artifacts.candidate_organization,
                    "next_agent_count": len(candidate_agents),
                    "next_agent_names": [agent.name for agent in candidate_agents],
                    "shadow_result": str(shadow_path),
                    "shadow_decision": (
                        "Decision deferred: the new specialist had no "
                        "effective shadow dispatch; retain it in dispatch-only "
                        "probation for the next online segment. "
                        f"Shadow note: {shadow_decision.reason}"
                    ),
                    "shadow_specialist_utilization": specialist_evidence,
                }
            )
        elif shadow_decision.accepted:
            selected_organization_path = attempt_dir / "selected_organization.json"
            write_json(selected_organization_path, {"agents": candidate_agents})
            outcome.update(
                {
                    "status": "accepted",
                    "selected_organization": str(selected_organization_path),
                    "candidate_organization": distill_artifacts.candidate_organization,
                    "next_agent_count": len(candidate_agents),
                    "next_agent_names": [agent.name for agent in candidate_agents],
                    "shadow_result": str(shadow_path),
                    "shadow_decision": shadow_decision.reason,
                    "shadow_specialist_utilization": specialist_evidence,
                }
            )
        else:
            outcome.update(
                {
                    "status": "rolled_back",
                    "candidate_organization": distill_artifacts.candidate_organization,
                    "shadow_result": str(shadow_path),
                    "shadow_decision": shadow_decision.reason,
                    "shadow_specialist_utilization": specialist_evidence,
                }
            )
        write_json(attempt_dir / "segment_outcome.json", outcome)
        return outcome

    def _commit_segment(
        self,
        state: dict[str, Any],
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        selected = outcome.get("selected_organization")
        if selected and outcome.get("status") in {
            "accepted",
            "deferred_probation",
        }:
            write_json(self.active_organization_path, read_json(selected))

        round_index = int(outcome["round"])
        train_trials = list(outcome.get("trials") or [])
        if not any(
            int(record.get("round", -1)) == round_index
            for record in state["history"]
        ):
            lean = {
                key: value
                for key, value in outcome.items()
                if key not in {"trials", "val_trials", "test_trials"}
            }
            lean["trial_count"] = len(train_trials)
            state["history"].append(lean)

        all_trials = state.setdefault("all_trials", [])
        all_trials.extend(train_trials)
        val_trials = list(outcome.get("val_trials") or [])
        if val_trials:
            state.setdefault("val_trials", []).extend(val_trials)

        train_progress = len(all_trials)
        attempt_dir = (
            Path(str(outcome["attempt_dir"]))
            if outcome.get("attempt_dir")
            else None
        )
        agents = self._agents_for_segment_eval(outcome)
        skills = SkillBank(self.skill_bank_path).stable_skills()
        periodic_val = self._maybe_run_val_eval(
            state,
            outcome=outcome,
            round_index=round_index,
            agents=agents,
            skills=skills,
            train_progress=train_progress,
            attempt_dir=attempt_dir,
        )
        if periodic_val:
            state.setdefault("val_trials", []).extend(periodic_val)
        test_trials = self._maybe_run_test_eval(
            state,
            outcome=outcome,
            round_index=round_index,
            agents=agents,
            skills=skills,
            train_progress=train_progress,
            attempt_dir=attempt_dir,
        )
        if test_trials:
            state.setdefault("test_trials", []).extend(test_trials)

        self._append_learning_curve_checkpoint(
            state,
            round_index=round_index,
            train_progress=train_progress,
        )

        trajectory_paths = state.setdefault("trajectory_paths", [])
        if outcome.get("trajectory_path"):
            trajectory_paths.append(outcome["trajectory_path"])

        state["completed_segments"] = max(
            int(state.get("completed_segments", 0)),
            round_index,
        )
        state["active_organization"] = str(self.active_organization_path)
        state["skill_bank"] = str(self.skill_bank_path)
        if self._family_enable:
            state["family_enable"] = dict(self._family_enable)
        write_json(self.state_path, state)
        return state

    def _finalize_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        trial_records = state.get("all_trials") or []
        wins = sum(int(trial.get("won", False)) for trial in trial_records)
        task_wins: dict[str, int] = defaultdict(int)
        task_total: dict[str, int] = defaultdict(int)
        for trial in trial_records:
            family = str(trial.get("task_family", "other"))
            task_total[family] += 1
            task_wins[family] += int(bool(trial.get("won", False)))
        num_games = len(trial_records)
        val_records = state.get("val_trials") or []
        test_records = state.get("test_trials") or []

        traj_path = self.output_root / "online_trajectories.jsonl"
        self._merge_trajectory_jsonl(
            state.get("trajectory_paths") or [],
            traj_path,
        )

        history = list(state.get("history") or [])
        segment_rows = [
            {
                "round": record.get("round"),
                "status": record.get("status"),
                "agent_count": record.get("agent_count"),
                "agent_names": record.get("agent_names"),
                "next_agent_count": record.get("next_agent_count"),
                "next_agent_names": record.get("next_agent_names"),
                "candidate_skill_names": record.get("candidate_skill_names"),
                "verified_skill_names": record.get("verified_skill_names"),
                "shadow_decision": record.get("shadow_decision"),
                "segment_metrics": record.get("segment_metrics"),
                "experience_distribution_snapshot": record.get(
                    "experience_distribution_snapshot"
                ),
                "capability_contracts": record.get("capability_contracts"),
                "paired_organization_fork": record.get("shadow_result"),
            }
            for record in history
        ]
        skill_bank = SkillBank(self.skill_bank_path)
        skill_capability_counts: Counter[str] = Counter(
            str(skill.capability_key or skill.metadata.get("capability_key") or "unknown")
            for skill in skill_bank.skills
            if skill.status
            not in {SkillStatus.REJECTED, SkillStatus.RETIRED}
        )

        return {
            "protocol": "alfworld_online_segmented_evolution_v2",
            "num_games": num_games,
            "num_segments": len(state.get("segments") or []),
            "wins": wins,
            "success_rate": (wins / num_games if num_games else 0.0),
            "ablation": {
                "executor_only": self.config.executor_only,
                "freeze_organization": self.config.freeze_organization,
                "skip_shadow": self.config.skip_shadow,
                "propose_org_for_unassigned_skills": (
                    self.config.propose_org_for_unassigned_skills
                ),
                "org_warmup_segments": self.config.org_warmup_segments,
                "actor_promotion_probe": self.config.actor_promotion_probe,
                "require_accepted_for_primary": bool(
                    (
                        self.sage_config.get("sage", {}).get(
                            "executor_dispatch"
                        )
                        or {}
                    ).get("require_accepted_for_primary", False)
                ),
                "seed_skill_bank_path": (
                    self.sage_config.get("sage", {}).get(
                        "seed_skill_bank_path"
                    )
                ),
                "collection_dataset": self.config.collection_dataset,
                "game_selection": self.config.game_selection,
                "seed": self.config.seed,
                "segment_size": self.config.segment_size,
                "block_injection_capability_prefixes": list(
                    self.config.block_injection_capability_prefixes
                ),
                "inject_capability_allowlist": (
                    None
                    if self.config.inject_capability_allowlist is None
                    else list(self.config.inject_capability_allowlist)
                ),
                "distillation_mode": str(
                    (
                        self.sage_config.get("sage", {}).get("distillation")
                        or {}
                    ).get("mode", "heuristic")
                ),
                "distillation_model": (
                    getattr(self.distillation_backend, "model", None)
                    if self.distillation_backend is not None
                    else None
                ),
                "executor_model": (
                    getattr(self.backend, "model", None)
                    or (self.llm_config.get("openai") or {}).get("model")
                ),
                "specialist_model": (
                    getattr(self.specialist_backend, "model", None)
                    if self.specialist_backend is not None
                    else self.config.specialist_model
                ),
                "acceptor_model": (
                    getattr(self.acceptor_backend, "model", None)
                    if self.acceptor_backend is not None
                    else self.config.acceptor_model
                ),
            },
            "val_trials": len(val_records),
            "val_success_rate": (
                sum(int(t.get("won", False)) for t in val_records) / len(val_records)
                if val_records
                else None
            ),
            "test_trials": len(test_records),
            "test_success_rate": (
                sum(int(t.get("won", False)) for t in test_records) / len(test_records)
                if test_records
                else None
            ),
            "learning_curve_checkpoints": state.get("learning_curve_checkpoints") or [],
            "by_family": {
                family: {
                    "wins": task_wins[family],
                    "total": task_total[family],
                    "success_rate": task_wins[family] / task_total[family],
                }
                for family in sorted(task_total)
            },
            "dispatch": summarize_dispatch(trial_records),
            "adaptation": summarize_adaptation_history(history),
            "skill_bank_capability_counts": dict(
                sorted(skill_capability_counts.items())
            ),
            "skill_bank_size": len(skill_bank.skills),
            "segments": segment_rows,
            "final_organization": str(self.active_organization_path),
            "skill_bank": str(self.skill_bank_path),
            "trajectories_jsonl": str(traj_path),
            "final_agent_count": len(
                load_agents(self.active_organization_path)
            ),
            "final_agent_names": [
                agent.name for agent in load_agents(self.active_organization_path)
            ],
        }

    @staticmethod
    def _merge_trajectory_jsonl(
        sources: list[str],
        destination: Path,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as out:
            for source in sources:
                path = Path(source)
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            out.write(line if line.endswith("\n") else line + "\n")

    def _seed_or_create_skill_bank(self, sage_cfg: dict[str, Any]) -> None:
        """Create an empty bank, or copy/filter a warm-start SkillBank."""
        seed_raw = sage_cfg.get("seed_skill_bank_path")
        if not seed_raw:
            SkillBank(self.skill_bank_path).save()
            return
        seed_path = Path(str(seed_raw)).expanduser()
        if not seed_path.is_absolute():
            seed_path = Path.cwd() / seed_path
        if not seed_path.exists():
            raise FileNotFoundError(
                f"sage.seed_skill_bank_path does not exist: {seed_path}"
            )
        statuses = {
            str(item).strip().lower()
            for item in (
                sage_cfg.get("seed_skill_statuses")
                or ["verified"]
            )
        }
        # Fast path: full bank copy when no status filter is requested.
        if not statuses or statuses == {"*"} or statuses == {"all"}:
            self.skill_bank_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seed_path, self.skill_bank_path)
            return
        source = SkillBank(seed_path)
        dest = SkillBank(self.skill_bank_path)
        allowed = {
            SkillStatus.VERIFIED
            if name == "verified"
            else SkillStatus.PROVISIONAL
            if name == "provisional"
            else SkillStatus.CANDIDATE
            if name == "candidate"
            else None
            for name in statuses
        }
        allowed.discard(None)
        for skill in source.skills:
            if skill.status in allowed:
                dest.add(deepcopy(skill))
        dest.save()

    def _unassigned_verified_skills_for_org(
        self,
        bank: SkillBank,
        agents: list[AgentSpec],
    ) -> list[Skill]:
        """Clone verified bank skills that still need an org assignment."""
        if not self.config.propose_org_for_unassigned_skills:
            return []
        assigned = {
            str(name)
            for agent in agents
            for name in (agent.assigned_skills or [])
        }
        proposals: list[Skill] = []
        for skill in bank.stable_skills():
            if skill.skill_name in assigned:
                continue
            if self.config.require_positive_mu_for_injection:
                mu = skill.marginal_utility
                if mu is None or float(mu) <= self.config.min_marginal_utility:
                    continue
            clone = deepcopy(skill)
            clone.metadata["credit_promoted_pending_org"] = True
            clone.metadata["org_proposal_source"] = "unassigned_bank_skill"
            proposals.append(clone)
        return proposals

    def _initialize_files(self) -> None:
        sage_cfg = self.sage_config.get("sage", {})
        initial_agents = [
            agent_from_dict(record)
            for record in sage_cfg.get("agents", [])
        ]
        if not initial_agents:
            seed_org_raw = sage_cfg.get("seed_organization_path")
            if seed_org_raw:
                seed_org_path = Path(str(seed_org_raw)).expanduser()
                if not seed_org_path.is_absolute():
                    seed_org_path = Path.cwd() / seed_org_path
                if not seed_org_path.exists():
                    raise FileNotFoundError(
                        f"sage.seed_organization_path does not exist: {seed_org_path}"
                    )
                initial_agents = load_agents(seed_org_path)
        if not initial_agents:
            raise ValueError(
                "sage.agents or sage.seed_organization_path must define "
                "the initial organization"
            )
        if not self.active_organization_path.exists():
            write_json(
                self.active_organization_path,
                {"agents": initial_agents},
            )
        if not self.baseline_organization_path.exists():
            write_json(
                self.baseline_organization_path,
                {"agents": initial_agents},
            )
        if not self.skill_bank_path.exists():
            self._seed_or_create_skill_bank(sage_cfg)

        if not self.state_path.exists():
            partition = self._select_and_partition_online_gamefiles()
            gamefiles = partition["collection"]
            segments = segment_gamefiles(
                gamefiles,
                self.config.segment_size,
            )
            seed_bank = sage_cfg.get("seed_skill_bank_path")
            write_json(
                self.state_path,
                {
                    "version": 2,
                    "protocol": "alfworld_online_segmented_evolution_v1",
                    "completed_segments": 0,
                    "active_organization": str(self.active_organization_path),
                    "skill_bank": str(self.skill_bank_path),
                    "config": {
                        "num_games": self.config.num_games,
                        "segment_size": self.config.segment_size,
                        "game_selection": self.config.game_selection,
                        "collection_dataset": self.config.collection_dataset,
                        "seed": self.config.seed,
                        "shadow_pool_size": self.config.shadow_pool_size,
                        "mechanism_dataset": self.config.mechanism_dataset,
                        "test_pool_size": self.config.test_pool_size,
                        "test_eval_dataset": self.config.test_eval_dataset,
                        "test_eval_every_segment": (
                            self.config.test_eval_every_segment
                        ),
                        "val_pool_size": self.config.val_pool_size,
                        "val_eval_dataset": self.config.val_eval_dataset,
                        "val_game_selection": self.config.val_game_selection,
                        "val_eval_every_segment": (
                            self.config.val_eval_every_segment
                        ),
                        "executor_only": self.config.executor_only,
                        "freeze_organization": self.config.freeze_organization,
                        "skip_shadow": self.config.skip_shadow,
                        "seed_skill_bank_path": (
                            None if not seed_bank else str(seed_bank)
                        ),
                        "propose_org_for_unassigned_skills": (
                            self.config.propose_org_for_unassigned_skills
                        ),
                        "block_injection_capability_prefixes": list(
                            self.config.block_injection_capability_prefixes
                        ),
                        "inject_capability_allowlist": (
                            None
                            if self.config.inject_capability_allowlist is None
                            else list(self.config.inject_capability_allowlist)
                        ),
                        "distillation_mode": str(
                            (
                                self.sage_config.get("sage", {}).get(
                                    "distillation"
                                )
                                or {}
                            ).get("mode", "heuristic")
                        ),
                        "distillation_model": (
                            getattr(self.distillation_backend, "model", None)
                            if self.distillation_backend is not None
                            else None
                        ),
                        "executor_model": (
                            getattr(self.backend, "model", None)
                            or (self.llm_config.get("openai") or {}).get("model")
                        ),
                        "specialist_model": (
                            getattr(self.specialist_backend, "model", None)
                            if self.specialist_backend is not None
                            else self.config.specialist_model
                        ),
                        "acceptor_model": (
                            getattr(self.acceptor_backend, "model", None)
                            if self.acceptor_backend is not None
                            else self.config.acceptor_model
                        ),
                        "mechanism_carved_from_budget": partition[
                            "mechanism_carved_from_budget"
                        ],
                        "collection_game_count": len(gamefiles),
                    },
                    "gamefiles": gamefiles,
                    "segments": segments,
                    "shadow_pool": partition["shadow_pool"],
                    "test_pool": partition.get("test_pool") or [],
                    "val_pool": partition.get("val_pool") or [],
                    "history": [],
                    "all_trials": [],
                    "val_trials": [],
                    "test_trials": [],
                    "learning_curve_checkpoints": [],
                    "trajectory_paths": [],
                },
            )

    def _select_and_partition_online_gamefiles(self) -> dict[str, Any]:
        required = self.config.shadow_pool_size
        if required <= 0:
            return self._with_test_pool(
                {
                    "shadow_pool": [],
                    "collection": self._select_gamefiles(self.config.num_games),
                    "mechanism_carved_from_budget": False,
                }
            )
        # Keep mechanism/shadow tasks on their own split whenever collection
        # is not the same valid_unseen pool (e.g. online train runs).
        if (
            self.config.mechanism_dataset != "valid_unseen"
            or self.config.collection_dataset != "valid_unseen"
        ):
            pools = self._partition_mechanism_selection(
                self._select_mechanism_gamefiles(required)
            )
            return self._with_test_pool(
                {
                    **pools,
                    "collection": self._select_gamefiles(self.config.num_games),
                    "mechanism_carved_from_budget": False,
                }
            )
        data_path = self.llm_config["alfworld"]["data_path"]
        all_unseen = _list_unseen_gamefiles(data_path)
        total = self.config.num_games + required
        if len(all_unseen) >= total:
            selected = self._select_gamefiles(total)
            return self._with_test_pool(
                {
                    "shadow_pool": selected[:required],
                    "collection": selected[required:total],
                    "mechanism_carved_from_budget": False,
                }
            )
        return self._with_test_pool(
            self._partition_selected_gamefiles(
                self._select_gamefiles(self.config.num_games),
                carve_from_budget=True,
            )
        )

    def _with_test_pool(self, partition: dict[str, Any]) -> dict[str, Any]:
        excluded = set(partition.get("collection") or [])
        excluded.update(partition.get("shadow_pool") or [])
        partition["test_pool"] = self._build_test_pool(excluded)
        excluded.update(partition.get("test_pool") or [])
        partition["val_pool"] = self._build_val_pool(excluded)
        return partition

    def _build_heldout_pool(
        self,
        *,
        dataset: str,
        pool_size: int,
        excluded: set[str] | None = None,
        seed_offset: int = 42,
        game_selection: str = "shuffle",
    ) -> list[str]:
        if pool_size <= 0:
            return []
        excluded = set(excluded or [])
        data_path = Path(self.llm_config["alfworld"]["data_path"])
        dataset = str(dataset or "").strip().lower()
        if dataset == "valid_unseen":
            pool = [
                gamefile
                for gamefile in _list_unseen_gamefiles(str(data_path))
                if gamefile not in excluded
            ]
        else:
            dataset_root = data_path / "json_2.1.1" / dataset
            pool = sorted(
                str(path) for path in dataset_root.rglob("game.tw-pddl")
            )
            pool = [gamefile for gamefile in pool if gamefile not in excluded]
        if game_selection == "family_proportional":
            return select_family_proportional_gamefiles(
                sorted(pool), pool_size, seed=self.config.seed + seed_offset
            )
        if game_selection != "shuffle":
            raise ValueError(f"Unknown held-out game_selection={game_selection!r}")
        if not pool:
            return []
        import random

        random.Random(self.config.seed + seed_offset).shuffle(pool)
        return pool[: min(pool_size, len(pool))]

    def _build_test_pool(self, excluded: set[str] | None = None) -> list[str]:
        if not self.config.test_eval_enabled or self.config.test_pool_size <= 0:
            return []
        return self._build_heldout_pool(
            dataset=self.config.test_eval_dataset,
            pool_size=self.config.test_pool_size,
            excluded=excluded,
            seed_offset=42,
        )

    def _build_val_pool(self, excluded: set[str] | None = None) -> list[str]:
        if not self.config.val_eval_enabled or self.config.val_pool_size <= 0:
            return []
        return self._build_heldout_pool(
            dataset=self.config.val_eval_dataset,
            pool_size=self.config.val_pool_size,
            excluded=excluded,
            seed_offset=91,
            game_selection=self.config.val_game_selection,
        )

    def _ensure_test_pool(self, state: dict[str, Any]) -> None:
        existing = state.get("test_pool")
        if isinstance(existing, list) and existing:
            return
        excluded = set(state.get("gamefiles") or [])
        excluded.update(state.get("shadow_pool") or [])
        excluded.update(state.get("val_pool") or [])
        pool = self._build_test_pool(excluded)
        if pool:
            state["test_pool"] = pool
            state.setdefault("config", {})["test_pool_size"] = (
                self.config.test_pool_size
            )
            state["config"]["test_eval_dataset"] = self.config.test_eval_dataset
            state["config"]["test_eval_every_segment"] = (
                self.config.test_eval_every_segment
            )
            write_json(self.state_path, state)

    def _ensure_val_pool(self, state: dict[str, Any]) -> None:
        existing = state.get("val_pool")
        if isinstance(existing, list) and existing:
            return
        excluded = set(state.get("gamefiles") or [])
        excluded.update(state.get("shadow_pool") or [])
        excluded.update(state.get("test_pool") or [])
        pool = self._build_val_pool(excluded)
        if pool:
            state["val_pool"] = pool
            state.setdefault("config", {})["val_pool_size"] = (
                self.config.val_pool_size
            )
            state["config"]["val_eval_dataset"] = self.config.val_eval_dataset
            state["config"]["val_game_selection"] = self.config.val_game_selection
            state["config"]["val_eval_every_segment"] = (
                self.config.val_eval_every_segment
            )
            write_json(self.state_path, state)

    def _partition_selected_gamefiles(
        self,
        selected: list[str],
        *,
        carve_from_budget: bool,
    ) -> dict[str, Any]:
        required = self.config.shadow_pool_size
        if required <= 0:
            return {
                "shadow_pool": [],
                "collection": list(selected),
                "mechanism_carved_from_budget": False,
            }
        if len(selected) <= required:
            raise ValueError(
                "Not enough ALFWorld tasks to reserve the shadow "
                f"pools: need at least {required + 1}, got {len(selected)}."
            )
        return {
            "shadow_pool": selected[:required],
            "collection": selected[required:],
            "mechanism_carved_from_budget": carve_from_budget,
        }

    def _build_mechanism_pools(
        self,
        collection_gamefiles: list[str],
    ) -> dict[str, list[str]]:
        required = self.config.shadow_pool_size
        if required <= 0:
            return {"shadow_pool": []}
        if self.config.mechanism_dataset != "valid_unseen":
            return self._partition_mechanism_selection(
                self._select_mechanism_gamefiles(required)
            )
        import random
        excluded = set(collection_gamefiles)
        remaining = [
            gamefile
            for gamefile in _list_unseen_gamefiles(
                self.llm_config["alfworld"]["data_path"]
            )
            if gamefile not in excluded
        ]
        random.Random(self.config.seed + 17).shuffle(remaining)
        if len(remaining) >= required:
            return {"shadow_pool": remaining[:required]}
        partition = self._partition_selected_gamefiles(
            list(collection_gamefiles),
            carve_from_budget=True,
        )
        return {"shadow_pool": partition["shadow_pool"]}

    def _partition_mechanism_selection(
        self,
        selected: list[str],
    ) -> dict[str, list[str]]:
        return {"shadow_pool": selected[: self.config.shadow_pool_size]}

    def _run_paired_mu_probes(
        self,
        *,
        agents: list[AgentSpec],
        seed_trials: list[EvaluationTrial],
        attempt_dir: Path,
        state: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Probe provisional bank skills with paired with/without injection."""
        bank = SkillBank(
            self.skill_bank_path,
            redundancy_threshold=float(
                self.sage_config.get("sage", {}).get(
                    "redundancy_threshold",
                    0.9,
                )
            ),
        )
        # Probe each credit-managed skill once until it earns a positive MU.
        pending = [
            skill
            for skill in bank.credit_skills()
            if skill.status
            in {SkillStatus.PROVISIONAL, SkillStatus.VERIFIED}
            and not bool(skill.metadata.get("mu_promoted"))
            and skill.metadata.get("marginal_utility_probe") is None
        ]
        summaries: list[dict[str, Any]] = []
        if not pending:
            write_json(attempt_dir / "marginal_utility_probes.json", summaries)
            return summaries

        fallback_pool: list[str] = []
        if state is not None:
            fallback_pool.extend(str(path) for path in (state.get("shadow_pool") or []))
            for segment in state.get("segments") or []:
                fallback_pool.extend(str(path) for path in segment)

        # Fresh bank snapshot for evaluate skill lists.
        all_skills = list(bank.skills)
        for skill in pending:
            gamefiles = select_probe_gamefiles(
                seed_trials,
                skill,
                max_tasks=self.config.paired_mu_num_tasks,
            )
            if len(gamefiles) < max(1, self.config.paired_mu_num_tasks // 2):
                # Same-family held-out tasks when this segment had no matches.
                extra = [
                    path
                    for path in fallback_pool
                    if path not in gamefiles
                    and self._path_matches_skill_families(path, skill)
                ]
                for path in extra:
                    gamefiles.append(path)
                    if len(gamefiles) >= self.config.paired_mu_num_tasks:
                        break
            if not gamefiles:
                continue
            result = probe_skill_marginal_utility(
                self._gate_evaluator(),
                agents=agents,
                skills=all_skills,
                skill=skill,
                gamefiles=gamefiles,
                min_delta=self.config.min_marginal_utility,
                require_ci_above_zero=self.config.paired_mu_require_ci,
                seed=self.config.seed + abs(hash(skill.skill_name)) % 1000,
            )
            apply_marginal_utility_probe(
                skill,
                result,
                promote_on_accept=True,
            )
            bank.save()
            summaries.append(result.as_dict())
        write_json(attempt_dir / "marginal_utility_probes.json", summaries)
        return summaries

    @staticmethod
    def _path_matches_skill_families(path: str, skill: Skill) -> bool:
        families = {
            str(family).strip()
            for family in (skill.applicable_task_families or [])
            if str(family).strip()
        }
        if not families:
            return True
        lowered = str(path or "").lower()
        return any(family.lower() in lowered for family in families)

    def _run_actor_promotion_probes(
        self,
        *,
        agents: list[AgentSpec],
        state: dict[str, Any],
        attempt_dir: Path,
    ) -> dict[str, Any]:
        """Promote probation specialists only if they beat Executor on held-out tasks."""
        if not self.config.actor_promotion_probe:
            return {"changed": False, "decisions": [], "probes": []}
        pending = probation_specialists(agents)
        if not pending:
            return {"changed": False, "decisions": [], "probes": []}

        pool = list(state.get("shadow_pool") or [])
        if not pool:
            # Fall back to upcoming collection tasks so probes still run.
            segments: list[list[str]] = state.get("segments") or []
            pool = [path for segment in segments for path in segment]
        skills = SkillBank(self.skill_bank_path).stable_skills()
        decisions: list[dict[str, Any]] = []
        probes: list[dict[str, Any]] = []
        changed = False
        for specialist in pending:
            gamefiles = select_promotion_gamefiles(
                pool,
                specialist,
                max_tasks=self.config.actor_promotion_num_tasks,
            )
            result = probe_actor_promotion(
                self._gate_evaluator(),
                agents=agents,
                skills=skills,
                specialist=specialist,
                gamefiles=gamefiles,
                min_advantage=self.config.actor_promotion_min_advantage,
                bare_executor=True,
            )
            decision = apply_actor_promotion_result(
                specialist,
                result,
                remove_after_rejected_windows=(
                    self.config.remove_after_rejected_windows
                ),
            )
            decisions.append(decision)
            probes.append(result.as_dict())
            changed = True
        payload = {
            "changed": changed,
            "decisions": decisions,
            "probes": probes,
        }
        restore = restore_executor_baseline(agents)
        payload["executor_restore"] = restore
        if restore.get("changed"):
            payload["changed"] = True
        write_json(attempt_dir / "actor_promotion_probes.json", payload)
        return payload

    @staticmethod
    def _has_executor_skill_contract(skill: Skill) -> bool:
        evidence_count = len(set(skill.evidence_ids))
        anchor_state = str(skill.metadata.get("anchor_state", "") or "").strip()
        return bool(
            evidence_count >= 1
            and skill.support_count >= 1
            and anchor_state
            and str(skill.expected_effect or "").strip()
        )

    def _prioritize_probation_tasks(
        self,
        state: dict[str, Any],
        round_index: int,
        skills: list[Skill],
    ) -> tuple[list[str], dict[str, Any]]:
        del skills
        return list(state["segments"][round_index - 1]), {
            "enabled": False,
            "reason": (
                "The scored evaluation stream remains fixed; Skill trials "
                "use the normal online segments without task reordering."
            ),
            "prioritized_task_count": 0,
        }

    def _ensure_mechanism_pools(self, state: dict[str, Any]) -> None:
        state.pop("discovery_pool", None)
        existing_pool = state.get("shadow_pool")
        need_pool = (
            not self.config.skip_shadow
            and self.config.shadow_pool_size > 0
            and (
                existing_pool is None
                or (
                    isinstance(existing_pool, list)
                    and len(existing_pool) < self.config.shadow_pool_size
                )
            )
        )
        if not need_pool:
            if self.config.skip_shadow:
                state["shadow_pool"] = list(existing_pool or [])
            return
        gamefiles = list(state.get("gamefiles") or [])
        if not gamefiles:
            raise ValueError(
                "Online state is missing collection gamefiles; "
                "use a fresh --output directory."
            )
        pools = self._build_mechanism_pools(gamefiles)
        if pools["shadow_pool"]:
            state["shadow_pool"] = pools["shadow_pool"]
            carved_from_budget = False
        else:
            partition = self._partition_selected_gamefiles(
                gamefiles,
                carve_from_budget=True,
            )
            state["shadow_pool"] = partition["shadow_pool"]
            state["gamefiles"] = partition["collection"]
            state["segments"] = segment_gamefiles(
                partition["collection"],
                self.config.segment_size,
            )
            carved_from_budget = partition["mechanism_carved_from_budget"]
        state.setdefault("config", {})["shadow_pool_size"] = (
            self.config.shadow_pool_size
        )
        state["config"]["mechanism_dataset"] = self.config.mechanism_dataset
        state["config"]["mechanism_carved_from_budget"] = carved_from_budget
        state["config"]["collection_game_count"] = len(state["gamefiles"])
        state["version"] = max(int(state.get("version", 1)), 2)
        write_json(self.state_path, state)

    def _round_shadow_tasks(
        self,
        pool: list[str],
        round_index: int,
        *,
        required_families: list[str] | None = None,
    ) -> list[str]:
        shadow_cfg = self.sage_config.get("sage", {}).get("shadow", {})
        count = int(shadow_cfg.get("num_tasks", 2))
        if not pool:
            return []
        start = ((round_index - 1) * count) % len(pool)
        rotated = [
            pool[(start + offset) % len(pool)]
            for offset in range(len(pool))
        ]
        families = list(dict.fromkeys(required_families or []))
        selected: list[str] = []
        if families:
            matching = [
                gamefile
                for gamefile in rotated
                if _task_family_from_gamefile(gamefile) in families
            ]
            for family in families:
                family_tasks = [
                    gamefile
                    for gamefile in matching
                    if _task_family_from_gamefile(gamefile) == family
                ]
                if family_tasks and len(selected) < count:
                    selected.append(family_tasks[0])
            for gamefile in matching:
                if gamefile not in selected and len(selected) < count:
                    selected.append(gamefile)
            # Keep ADD_AGENT shadow on the specialist's contract families only.
            # Padding with unrelated tasks dilutes effectiveness and confuses
            # Executor dispatch checks.
            if selected:
                return selected[:count]
        for gamefile in rotated:
            if gamefile not in selected and len(selected) < count:
                selected.append(gamefile)
        return selected

    @staticmethod
    def _shadow_capability_families(
        active_agents: list[AgentSpec],
        candidate_agents: list[AgentSpec],
        stable_skills: list[Skill],
    ) -> list[str]:
        """Families owned by newly added specialists only.

        Executor ASSIGN_SKILL copies must not pull unrelated families into the
        shadow mix; otherwise ADD_AGENT effectiveness is diluted by tasks the
        new agent is not supposed to own.
        """
        active_names = {agent.name for agent in active_agents}
        changed_skill_names: set[str] = set()
        for candidate in candidate_agents:
            if candidate.name in active_names:
                continue
            changed_skill_names.update(candidate.assigned_skills)
        families = {
            family
            for skill in stable_skills
            if skill.skill_name in changed_skill_names
            for family in skill.applicable_task_families
        }
        return sorted(families)

    @staticmethod
    def _trial_success(trial: EvaluationTrial) -> float:
        if trial.reward is not None:
            return float(trial.reward)
        if trial.won is None:
            return 0.0
        return 1.0 if trial.won else 0.0

    @classmethod
    def _specialist_shadow_utilization(
        cls,
        active_agents: list[AgentSpec],
        candidate_agents: list[AgentSpec],
        new_trials: list[EvaluationTrial],
        *,
        required_families: list[str],
    ) -> dict[str, Any]:
        new_specialist_names = {
            agent.name for agent in candidate_agents
        } - {agent.name for agent in active_agents}
        family_set = {
            str(family).strip()
            for family in required_families
            if str(family).strip()
        }
        in_contract = [
            trial
            for trial in new_trials
            if str(trial.task_family or "").strip() in family_set
        ]
        out_of_contract = [
            trial
            for trial in new_trials
            if str(trial.task_family or "").strip() not in family_set
        ]
        primary_count = sum(
            int(trial.assigned_primary_agent in new_specialist_names)
            for trial in new_trials
        )
        in_contract_primary = sum(
            int(trial.assigned_primary_agent in new_specialist_names)
            for trial in in_contract
        )
        out_of_contract_primary = sum(
            int(trial.assigned_primary_agent in new_specialist_names)
            for trial in out_of_contract
        )
        action_count = sum(
            int(trial.actions_by_agent.get(agent_name, 0))
            for trial in new_trials
            for agent_name in new_specialist_names
        )
        in_contract_n = len(in_contract)
        call_rate = (
            float(in_contract_primary) / float(in_contract_n)
            if in_contract_n > 0
            else 0.0
        )
        # Reachable dispatch: in-contract call_rate above threshold and no
        # out-of-contract primary steals. Exact 100% is not required.
        min_call_rate = 0.5
        dispatch_correct = (
            not new_specialist_names
            or (
                bool(in_contract)
                and call_rate + 1e-12 >= float(min_call_rate)
                and out_of_contract_primary == 0
            )
        )
        return {
            "new_specialist_names": sorted(new_specialist_names),
            "required_task_families": list(required_families),
            "primary_dispatch_count": primary_count,
            "in_contract_trial_count": in_contract_n,
            "in_contract_primary_count": in_contract_primary,
            "out_of_contract_primary_count": out_of_contract_primary,
            "call_rate": float(call_rate),
            "min_call_rate": float(min_call_rate),
            "dispatch_correct": dispatch_correct,
            "actions_executed": action_count,
            "evidence_sufficient": (
                not new_specialist_names
                or (primary_count > 0 and action_count > 0)
            ),
        }

    def _add_agent_shadow_decision(
        self,
        *,
        old_trials: list[EvaluationTrial],
        new_trials: list[EvaluationTrial],
        required_families: list[str],
        specialist_evidence: dict[str, Any],
        fallback_decision: ShadowDecision,
        skills: list[Skill],
        old_agents: list[AgentSpec],
        new_agents: list[AgentSpec],
    ) -> ShadowDecision:
        """Accept ADD_AGENT only when dispatch is correct and the agent helps.

        Effectiveness is measured on the new specialist's contract families,
        without an agent-count tax. Equal/worse in-contract outcomes reject.
        """
        new_specialists = list(specialist_evidence.get("new_specialist_names") or [])
        if not new_specialists:
            return fallback_decision

        if not specialist_evidence.get("dispatch_correct", False):
            return replace(
                fallback_decision,
                accepted=False,
                reason=(
                    "Rejected ADD_AGENT: Executor dispatch was incorrect on "
                    "shadow tasks "
                    f"(in-contract primary="
                    f"{specialist_evidence.get('in_contract_primary_count', 0)}/"
                    f"{specialist_evidence.get('in_contract_trial_count', 0)}, "
                    "out-of-contract primary="
                    f"{specialist_evidence.get('out_of_contract_primary_count', 0)}). "
                    "New agents must own only their contract tasks."
                ),
            )

        family_set = {
            str(family).strip()
            for family in required_families
            if str(family).strip()
        }
        old_in = [
            trial
            for trial in old_trials
            if str(trial.task_family or "").strip() in family_set
        ]
        new_in = [
            trial
            for trial in new_trials
            if str(trial.task_family or "").strip() in family_set
        ]
        if not old_in or not new_in:
            return replace(
                fallback_decision,
                accepted=False,
                reason=(
                    "Rejected ADD_AGENT: no in-contract shadow tasks were "
                    "available to prove the new agent is effective."
                ),
            )

        # Compare acting quality only on contract scope. Effectiveness is
        # success-rate first: do not let Executor's default adherence=1.0
        # (no skills) handicap a specialist whose protocol is being scored.
        # Adherence remains a skill-quality gate before ADD_AGENT.
        old_metrics = OrganizationShadowService._metrics(
            old_in,
            agent_count=1,
            skills=skills,
            agents=old_agents,
        )
        new_metrics = OrganizationShadowService._metrics(
            new_in,
            agent_count=1,
            skills=skills,
            agents=new_agents,
        )
        from dataclasses import replace as _replace_metrics

        call_rate = specialist_evidence.get("call_rate")
        if call_rate is None:
            in_n = int(specialist_evidence.get("in_contract_trial_count") or 0)
            in_primary = int(
                specialist_evidence.get("in_contract_primary_count") or 0
            )
            call_rate = (float(in_primary) / float(in_n)) if in_n else 0.0
        old_metrics = _replace_metrics(old_metrics, protocol_adherence=1.0)
        new_metrics = _replace_metrics(
            new_metrics,
            protocol_adherence=1.0,
            new_agent_call_rate=float(call_rate),
        )
        evaluator = self._shadow_evaluator()
        # Effectiveness ignores headcount; keep cost weight from config.
        evaluator.agent_count_weight = 0.0
        min_call = float(
            getattr(evaluator, "min_new_agent_call_rate", 0.0) or 0.0
        )
        if min_call <= 0.0:
            min_call = float(specialist_evidence.get("min_call_rate") or 0.5)
            evaluator.min_new_agent_call_rate = min_call
        if float(call_rate) + 1e-12 < min_call:
            return replace(
                fallback_decision,
                accepted=False,
                new_agent_call_rate=float(call_rate),
                reason=(
                    "Rejected ADD_AGENT: new agent call_rate below threshold "
                    f"({float(call_rate):.3f} < {min_call:.3f}); "
                    "dispatch must reach the specialist on in-contract tasks."
                ),
            )
        old_by_task = {trial.task_id: trial for trial in old_in}
        new_by_task = {trial.task_id: trial for trial in new_in}
        paired_deltas = [
            evaluator.trial_utility(
                self._trial_success(new_by_task[task_id]),
                new_by_task[task_id].cost,
                1,
            )
            - evaluator.trial_utility(
                self._trial_success(old_by_task[task_id]),
                old_by_task[task_id].cost,
                1,
            )
            for task_id in sorted(set(old_by_task) & set(new_by_task))
        ]
        decision = evaluator.decide(
            old_metrics,
            new_metrics,
            paired_deltas=paired_deltas,
        )
        # "Effective" means a real in-contract improvement, not a zero-zero tie
        # that the global shadow rule treats as acceptable.
        if (
            decision.accepted
            and float(new_metrics.success_rate)
            <= float(old_metrics.success_rate) + 1e-12
        ):
            return replace(
                decision,
                accepted=False,
                reason=(
                    "Rejected ADD_AGENT: new agent did not improve in-contract "
                    f"success ({new_metrics.success_rate:.3f} vs "
                    f"{old_metrics.success_rate:.3f}); dispatch_correct=true."
                ),
            )
        if (
            not decision.accepted
            and float(new_metrics.success_rate)
            > float(old_metrics.success_rate) + 1e-12
        ):
            # SR improved but utility/cost gate disagreed; prefer SR for
            # specialist effectiveness once dispatch is correct.
            return replace(
                decision,
                accepted=True,
                reason=(
                    "Accepted ADD_AGENT on in-contract success improvement "
                    f"({new_metrics.success_rate:.3f} > "
                    f"{old_metrics.success_rate:.3f}); "
                    f"utility note: {decision.reason} dispatch_correct=true"
                ),
            )
        if decision.accepted:
            return replace(
                decision,
                reason=(
                    "Accepted ADD_AGENT on in-contract effectiveness "
                    f"({decision.reason} dispatch_correct=true)"
                ),
            )
        return replace(
            decision,
            reason=(
                "Rejected ADD_AGENT: new agent was not effective on its "
                f"contract tasks ({decision.reason})"
            ),
        )

    def _shadow_evaluator(self) -> ShadowEvaluator:
        shadow_cfg = self.sage_config.get("sage", {}).get("shadow", {})
        org_cfg = self.sage_config.get("sage", {}).get("organization", {})
        return ShadowEvaluator(
            cost_weight=float(shadow_cfg.get("cost_weight", 0.1)),
            significance_threshold=float(
                shadow_cfg.get(
                    "min_relative_gain",
                    shadow_cfg.get("significance_threshold", 0.02),
                )
            ),
            agent_count_weight=float(shadow_cfg.get("agent_count_weight", 0.0)),
            bootstrap_samples=int(shadow_cfg.get("bootstrap_samples", 1000)),
            confidence_level=float(shadow_cfg.get("confidence_level", 0.95)),
            require_confident_gain=bool(
                shadow_cfg.get("require_confident_gain", False)
            ),
            min_paired_tasks=int(shadow_cfg.get("min_paired_tasks", 1)),
            seed=int(shadow_cfg.get("seed", 202)),
            min_new_agent_call_rate=float(
                shadow_cfg.get(
                    "min_new_agent_call_rate",
                    org_cfg.get("min_new_agent_call_rate", 0.5),
                )
            ),
        )

    def _excluded_collection_gamefiles(self) -> set[str]:
        """Optional exclude list for collection sampling (paths or online_state)."""
        online_cfg = self.sage_config.get("sage", {}).get("online") or {}
        raw = online_cfg.get("exclude_gamefiles_from")
        if not raw:
            return set()
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(
                f"online.exclude_gamefiles_from does not exist: {path}"
            )
        payload = read_json(path)
        if isinstance(payload, list):
            return {str(item) for item in payload}
        if isinstance(payload, dict):
            games = payload.get("gamefiles")
            if isinstance(games, list):
                return {str(item) for item in games}
        raise RuntimeError(
            "online.exclude_gamefiles_from must be a JSON list of gamefile "
            "paths or an online_state.json object with a gamefiles list: "
            f"{path}"
        )

    def _select_gamefiles(self, num_games: int | None = None) -> list[str]:
        count = int(num_games or self.config.num_games)
        data_path = self.llm_config["alfworld"]["data_path"]
        selection = self.config.game_selection
        split = self.config.collection_dataset
        excluded = self._excluded_collection_gamefiles()
        if selection in {"first", "hard", "family_proportional", "proportional"}:
            return select_prompt_agent_gamefiles(
                data_path,
                num_games=count,
                game_selection=selection,
                excluded=excluded or None,
                split=split,
                seed=self.config.seed,
            )
        if selection == "shuffle":
            return sample_alfworld_gamefiles(
                data_path,
                num_tasks=count,
                seed=self.config.seed,
                excluded=excluded or None,
                split=split,
            )
        raise RuntimeError(
            "Unknown game_selection="
            f"{selection!r}; use first|hard|shuffle|family_proportional."
        )

    def _select_mechanism_gamefiles(self, count: int) -> list[str]:
        data_path = Path(self.llm_config["alfworld"]["data_path"])
        dataset = self.config.mechanism_dataset
        if dataset not in {"valid_seen", "valid_unseen"}:
            raise ValueError(
                "online.mechanism_dataset must be valid_seen or valid_unseen, "
                f"got {dataset!r}."
            )
        if dataset == "valid_unseen":
            gamefiles = _list_unseen_gamefiles(str(data_path))
        else:
            dataset_root = data_path / "json_2.1.1" / dataset
            gamefiles = sorted(
                str(path)
                for path in dataset_root.rglob("game.tw-pddl")
            )
            if not gamefiles:
                raise RuntimeError(
                    f"No ALFWorld game files found under {dataset_root}"
                )

        import random

        random.Random(self.config.seed + 17).shuffle(gamefiles)
        if len(gamefiles) < count:
            raise RuntimeError(
                f"Requested {count} {dataset} mechanism games but only "
                f"{len(gamefiles)} are available."
            )
        return gamefiles[:count]

    def _build_backend(self, model: str | None = None) -> OpenAIChatBackend:
        openai_cfg = self.llm_config["openai"]
        online_cfg = self.sage_config.get("sage", {}).get("online", {})
        loop_cfg = self.sage_config.get("sage", {}).get("loop", {})
        # Prefer sage.online.executor_model so actor can differ from distill.
        resolved = (
            model
            or online_cfg.get("executor_model")
            or loop_cfg.get("executor_model")
            or openai_cfg.get("model")
        )
        timeout = online_cfg.get("llm_timeout", openai_cfg.get("timeout", 120.0))
        max_retries = online_cfg.get(
            "llm_max_retries",
            openai_cfg.get("max_retries", 2),
        )
        return OpenAIChatBackend(
            model=str(resolved),
            api_key=_resolve_openai_api_key(openai_cfg),
            base_url=openai_cfg.get("base_url"),
            temperature=float(
                online_cfg.get(
                    "temperature",
                    loop_cfg.get("temperature", 0.0),
                )
            ),
            timeout=float(timeout) if timeout is not None else None,
            max_retries=int(max_retries),
            group=openai_cfg.get("group"),
        )

    def _build_acceptor_backend(self) -> OpenAIChatBackend | None:
        """Optional weak model for MU / promotion / shadow / test140 gates."""
        model = self.config.acceptor_model
        if not model:
            return None
        teacher = getattr(self.backend, "model", None)
        if teacher is not None and str(teacher) == str(model):
            return None
        return self._build_backend(model=str(model))

    def _build_specialist_backend(self) -> OpenAIChatBackend | None:
        """Optional mid model for specialist acts during teacher train play."""
        model = self.config.specialist_model
        if not model:
            return None
        teacher = getattr(self.backend, "model", None)
        if teacher is not None and str(teacher) == str(model):
            return None
        return self._build_backend(model=str(model))

    def _gate_evaluator(self) -> AlfWorldOrganizationEvaluator:
        """Evaluator used for acceptance and held-out probes (mini when set)."""
        return self.acceptor_evaluator or self.evaluator

    def _teacher_model_name(self) -> str | None:
        model = getattr(self.backend, "model", None)
        return str(model).strip() if model else None

    def _gate_model_name(self) -> str | None:
        """Model used for specialist acts / test gates (Flash in Exp B)."""
        for backend in (
            self.specialist_backend,
            self.acceptor_backend,
            getattr(self._gate_evaluator(), "backend", None),
        ):
            model = getattr(backend, "model", None) if backend is not None else None
            if model:
                return str(model).strip()
        return self._teacher_model_name()

    def _teacher_differs_from_gate(self) -> bool:
        teacher = self._teacher_model_name()
        gate = self._gate_model_name()
        if not teacher or not gate:
            return False
        return teacher != gate

    def _resolve_onboarding_baseline(
        self,
        *,
        agents: list[AgentSpec],
        train_trials: list[EvaluationTrial],
        train_baseline: list[Any],
        skills: list[Skill],
        attempt_dir: Path,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Pick Executor baseline trials for probation accept/demote.

        ``same_model`` (default): when teacher != gate/specialist, replay
        Executor-only on the gate evaluator over specialist-dispatched
        gamefiles. Never judge a Flash specialist against Pro train SR.
        """
        mode = str(self.config.onboarding_baseline or "same_model").strip().lower()
        teacher = self._teacher_model_name()
        gate_model = self._gate_model_name()
        meta: dict[str, Any] = {
            "mode": mode,
            "teacher_model": teacher,
            "gate_model": gate_model,
            "models_differ": self._teacher_differs_from_gate(),
        }
        # Hard rule: never judge mid-model specialists against teacher train SR.
        if self._teacher_differs_from_gate() and mode in {
            "train",
            "teacher",
            "legacy",
        }:
            mode = "same_model"
            meta["mode"] = mode
            meta["forced_same_model"] = True
        if mode in {"train", "teacher", "legacy"}:
            meta["source"] = "train_executor"
            return list(train_baseline), meta
        if not self._teacher_differs_from_gate():
            meta["source"] = "train_executor_same_model"
            return list(train_baseline), meta

        pending = [
            agent
            for agent in agents
            if acting_status(agent) == "probation"
            and not (
                "executor" in f"{agent.name} {agent.role}".lower()
            )
        ]
        if not pending:
            meta["source"] = "same_model_no_probation"
            # No probationists to judge; empty baseline cannot demote anyone.
            return [], meta

        gamefiles: list[str] = []
        pending_names = {agent.name for agent in pending}
        for trial in train_trials:
            primary = str(getattr(trial, "assigned_primary_agent", "") or "")
            if primary not in pending_names:
                continue
            path = str(
                getattr(trial, "task_id", None)
                or getattr(trial, "gamefile", None)
                or ""
            ).strip()
            if path and path not in gamefiles:
                gamefiles.append(path)
        if not gamefiles:
            # Fall back to family-matched segment tasks (no cross-model train).
            for agent in pending:
                families = {
                    str(f).strip().lower()
                    for f in (
                        (agent.shadow_evaluation_record or {}).get(
                            "task_families"
                        )
                        or []
                    )
                    if str(f).strip()
                }
                for trial in train_trials:
                    family = str(getattr(trial, "task_family", "") or "").lower()
                    path = str(getattr(trial, "task_id", "") or "").strip()
                    if not path or path in gamefiles:
                        continue
                    if families and family not in families:
                        continue
                    gamefiles.append(path)
        meta["gamefiles"] = list(gamefiles)
        if not gamefiles:
            meta["source"] = "same_model_unavailable"
            meta["defer"] = True
            return [], meta

        gate = self._gate_evaluator()
        if gate is self.evaluator and self._teacher_differs_from_gate():
            # Should be unreachable when acceptor/specialist backends exist.
            meta["source"] = "same_model_gate_missing"
            meta["defer"] = True
            return [], meta
        executor_agents = _with_status(agents, primary_specialist=None)
        baseline_trials = gate.evaluate(
            agents=executor_agents,
            skills=skills,
            gamefiles=list(gamefiles),
            condition="onboarding_same_model_executor_baseline",
        )
        write_json(
            attempt_dir / "onboarding_same_model_baseline.json",
            {
                "teacher_model": teacher,
                "gate_model": gate_model,
                "n_gamefiles": len(gamefiles),
                "gamefiles": gamefiles,
                "executor_wins": sum(1 for t in baseline_trials if t.won),
                "executor_sr": (
                    sum(1 for t in baseline_trials if t.won)
                    / max(len(baseline_trials), 1)
                ),
            },
        )
        meta["source"] = "acceptor_executor_replay"
        meta["n_baseline"] = len(baseline_trials)
        meta["executor_sr"] = (
            sum(1 for t in baseline_trials if t.won)
            / max(len(baseline_trials), 1)
        )
        return list(baseline_trials), meta

    def _with_test_concurrency(
        self,
        evaluator: AlfWorldOrganizationEvaluator,
    ):
        """Temporarily raise concurrency for test140 only."""
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            cfg = getattr(evaluator, "config", None)
            if cfg is None:
                yield evaluator
                return
            old_parallel = cfg.parallel_envs
            old_api = cfg.api_concurrency
            try:
                if self.config.test_parallel_envs is not None:
                    cfg.parallel_envs = int(self.config.test_parallel_envs)
                if self.config.test_api_concurrency is not None:
                    cfg.api_concurrency = int(self.config.test_api_concurrency)
                yield evaluator
            finally:
                cfg.parallel_envs = old_parallel
                cfg.api_concurrency = old_api

        return _ctx()

    def _build_distillation_backend(self) -> OpenAIChatBackend | None:
        """Optional separate LLM for skill summarization (Executor stays fixed)."""
        distill_cfg = self.sage_config.get("sage", {}).get("distillation") or {}
        mode = str(distill_cfg.get("mode", "heuristic")).strip().lower()
        overlay = bool(distill_cfg.get("llm_protocol_overlay", False))
        if mode in {"heuristic", "heuristic_seed_only"} and not overlay:
            return None
        llm_distill = self.llm_config.get("distillation") or {}
        openai_cfg = self.llm_config.get("openai") or {}
        model = (
            distill_cfg.get("model")
            or llm_distill.get("model")
            or openai_cfg.get("distillation_model")
        )
        if not model:
            # trajectory_enriched / overlay without an explicit model reuses Executor LLM.
            if overlay or mode in {"trajectory_enriched", "trajectory_grounded_llm"}:
                return self.backend if isinstance(self.backend, OpenAIChatBackend) else None
            return None
        api_key = _resolve_openai_api_key(
            distill_cfg,
            llm_distill,
            openai_cfg,
        )
        base_url = (
            distill_cfg.get("base_url")
            or llm_distill.get("base_url")
            or openai_cfg.get("base_url")
        )
        temperature = float(
            distill_cfg.get(
                "temperature",
                llm_distill.get("temperature", 0.0),
            )
        )
        openai_cfg = self.llm_config.get("openai") or {}
        online_cfg = self.sage_config.get("sage", {}).get("online", {})
        timeout = distill_cfg.get(
            "timeout",
            online_cfg.get("llm_timeout", openai_cfg.get("timeout", 180.0)),
        )
        max_retries = distill_cfg.get(
            "max_retries",
            online_cfg.get("llm_max_retries", openai_cfg.get("max_retries", 2)),
        )
        return OpenAIChatBackend(
            model=str(model),
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            timeout=float(timeout) if timeout is not None else None,
            max_retries=int(max_retries),
            group=openai_cfg.get("group"),
        )

    def _evaluator_config(self) -> AlfWorldEvaluatorConfig:
        alf_cfg = self.llm_config["alfworld"]
        online_cfg = self.sage_config.get("sage", {}).get("online", {})
        loop_cfg = self.sage_config.get("sage", {}).get("loop", {})

        def _get(key: str, default: Any) -> Any:
            if key in online_cfg:
                return online_cfg[key]
            if key in loop_cfg:
                return loop_cfg[key]
            return default

        cfg = AlfWorldEvaluatorConfig(
            max_steps=int(_get("max_steps", alf_cfg.get("max_steps", 50))),
            parallel_envs=int(
                _get("parallel_envs", alf_cfg.get("parallel_envs", 10))
            ),
            api_concurrency=int(
                _get(
                    "api_concurrency",
                    alf_cfg.get("api_concurrency", 10),
                )
            ),
            num_cpus_per_worker=float(
                alf_cfg.get("num_cpus_per_worker", 0.05)
            ),
            history_length=int(
                _get("history_length", alf_cfg.get("history_length", 0))
            ),
            seed=self.config.seed,
            save_steps=bool(_get("save_steps", True)),
            max_advisors=_get("max_advisors", 1),
            max_injected_skills=int(_get("max_injected_skills", 2)),
            skill_recall_backend=str(_get("skill_recall_backend", "bm25")),
            skill_recall_top_k=int(_get("skill_recall_top_k", 8)),
            skill_recall_semantic_filter=bool(_get("skill_recall_semantic_filter", True)),
            skill_recall_query_llm=bool(_get("skill_recall_query_llm", True)),
            use_visited_location_memory=False,
            ignore_assigned_skills=bool(
                online_cfg.get(
                    "ignore_assigned_skills",
                    online_cfg.get("executor_only", False),
                )
            ),
            enable_action_guards=False,
            enable_specialist_controllers=bool(
                _get("enable_specialist_controllers", False)
            ),
            short_specialist_prompts=bool(
                _get("short_specialist_prompts", True)
            ),
            compact_alfworld_prompts=False,
            executor_dispatch=dispatch_config_from_mapping(
                self._dispatch_config_mapping()
            ),
        )
        if not cfg.enable_specialist_controllers:
            cfg.executor_dispatch.require_controller_for_eligibility = False
        if getattr(self, "_family_enable", None):
            enabled, disabled = dispatch_lists_from_family_enable(
                self._family_enable
            )
            cfg.executor_dispatch.enabled_task_families = list(enabled)
            cfg.executor_dispatch.disabled_task_families = list(disabled)
        return cfg

    @staticmethod
    def _load_prior_trials(state: dict[str, Any]) -> list[dict[str, Any]]:
        prior = state.get("all_trials") or []
        return list(prior) if isinstance(prior, list) else []

    def _prepare_distill_trajectory_window(
        self,
        state: dict[str, Any],
        *,
        trajectory_path: Path,
        attempt_dir: Path,
    ) -> Path:
        """Merge current + recent prior segment trajectories for distillation."""
        prior_n = max(0, int(self.config.distill_prior_segments))
        if prior_n <= 0:
            return trajectory_path
        prior_paths = [
            str(path)
            for path in (state.get("trajectory_paths") or [])[-prior_n:]
        ]
        if not prior_paths:
            return trajectory_path
        merged = attempt_dir / "distill_trajectories_window.jsonl"
        self._merge_trajectory_jsonl(
            prior_paths + [str(trajectory_path)],
            merged,
        )
        return merged

    def _agents_for_segment_eval(self, outcome: dict[str, Any]) -> list[AgentSpec]:
        selected = outcome.get("selected_organization")
        if selected and outcome.get("status") in {
            "accepted",
            "deferred_probation",
        }:
            return load_agents(selected)
        return load_agents(self.active_organization_path)

    @staticmethod
    def _segment_added_agents(outcome: dict[str, Any]) -> list[str]:
        prior = set(outcome.get("agent_names") or [])
        nxt = set(outcome.get("next_agent_names") or [])
        added = sorted(name for name in nxt if name not in prior)
        if added:
            return added
        edits_ref = outcome.get("organization_edits")
        if not edits_ref:
            return []
        try:
            edits = read_json(edits_ref)
        except (OSError, TypeError, ValueError):
            return []
        names: list[str] = []
        for edit in edits:
            if str(edit.get("edit_type", "")).lower() != "add_agent":
                continue
            new_agent = edit.get("new_agent")
            if isinstance(new_agent, dict) and new_agent.get("name"):
                names.append(str(new_agent["name"]))
        return names

    def _periodic_eval_cadence(self, *, every_n_segments: int, every_segment: bool) -> int:
        """Return N>0 for every-N-segments cadence, else 0 (ADD_AGENT-only)."""
        n = int(every_n_segments or 0)
        if n > 0:
            return n
        if every_segment:
            return 1
        return 0

    def _should_run_periodic_eval(
        self,
        outcome: dict[str, Any],
        *,
        enabled: bool,
        every_n_segments: int,
        every_segment: bool,
        round_index: int,
    ) -> bool:
        if not enabled:
            return False
        cadence = self._periodic_eval_cadence(
            every_n_segments=every_n_segments,
            every_segment=every_segment,
        )
        if cadence > 0:
            total_segments = max(
                1,
                (int(self.config.num_games) + int(self.config.segment_size) - 1)
                // max(1, int(self.config.segment_size)),
            )
            round_index = int(round_index)
            if round_index >= total_segments:
                return True
            if round_index == 1:
                return True
            return round_index % cadence == 0
        if outcome.get("status") not in {"accepted", "deferred_probation"}:
            return False
        return bool(self._segment_added_agents(outcome))

    def _should_run_test_eval(
        self,
        outcome: dict[str, Any],
        *,
        round_index: int = 1,
    ) -> bool:
        return self._should_run_periodic_eval(
            outcome,
            enabled=self.config.test_eval_enabled,
            every_n_segments=self.config.test_eval_every_n_segments,
            every_segment=self.config.test_eval_every_segment,
            round_index=round_index,
        )

    def _should_run_val_eval(
        self,
        outcome: dict[str, Any],
        *,
        round_index: int = 1,
    ) -> bool:
        return self._should_run_periodic_eval(
            outcome,
            enabled=self.config.val_eval_enabled,
            every_n_segments=self.config.val_eval_every_n_segments,
            every_segment=self.config.val_eval_every_segment,
            round_index=round_index,
        )

    def _fixed_test_gamefiles(self, state: dict[str, Any]) -> list[str]:
        pool = list(state.get("test_pool") or [])
        if pool:
            return pool
        excluded = set(state.get("gamefiles") or [])
        excluded.update(state.get("shadow_pool") or [])
        excluded.update(state.get("val_pool") or [])
        return self._build_test_pool(excluded)

    def _fixed_val_gamefiles(self, state: dict[str, Any]) -> list[str]:
        pool = list(state.get("val_pool") or [])
        if pool:
            return pool
        excluded = set(state.get("gamefiles") or [])
        excluded.update(state.get("shadow_pool") or [])
        excluded.update(state.get("test_pool") or [])
        return self._build_val_pool(excluded)

    def _maybe_run_heldout_eval(
        self,
        state: dict[str, Any],
        *,
        outcome: dict[str, Any],
        round_index: int,
        agents: list[AgentSpec],
        skills: list[Skill],
        train_progress: int,
        gamefiles: list[str],
        dataset: str,
        phase: str,
        condition_prefix: str,
        attempt_dir: Path | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not gamefiles:
            return []
        evaluator = self._gate_evaluator()
        use_test_conc = (
            phase in {"periodic_test", "post_add_agent_test"}
            and (
                self.config.test_parallel_envs is not None
                or self.config.test_api_concurrency is not None
            )
        )
        if use_test_conc:
            with self._with_test_concurrency(evaluator):
                trials = evaluator.evaluate(
                    agents=agents,
                    skills=skills,
                    gamefiles=gamefiles,
                    condition=f"{condition_prefix}_{round_index}",
                )
        else:
            trials = evaluator.evaluate(
                agents=agents,
                skills=skills,
                gamefiles=gamefiles,
                condition=f"{condition_prefix}_{round_index}",
            )
        records = []
        for trial in trials:
            record = evaluation_trial_record(
                trial,
                dataset_split=infer_dataset_split(
                    trial.task_id,
                    dataset,
                ),
                round_index=round_index,
                phase=phase,
                train_progress=train_progress,
            )
            record["fixed_eval_pool"] = True
            if extra_fields:
                record.update(extra_fields)
            records.append(record)
        if attempt_dir is not None:
            write_json(
                attempt_dir / f"{condition_prefix}_trials.json",
                records,
            )
            write_json(
                attempt_dir / f"{condition_prefix}_result.json",
                {
                    "protocol": f"{condition_prefix}_v1",
                    "held_out_gamefiles": gamefiles,
                    "metrics": summarize_trials(
                        trials,
                        name=f"{condition_prefix}_{round_index:03d}",
                    ),
                },
            )
        return records

    def _maybe_run_val_eval(
        self,
        state: dict[str, Any],
        *,
        outcome: dict[str, Any],
        round_index: int,
        agents: list[AgentSpec],
        skills: list[Skill],
        train_progress: int,
        attempt_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        if not self._should_run_val_eval(outcome, round_index=round_index):
            return []
        return self._maybe_run_heldout_eval(
            state,
            outcome=outcome,
            round_index=round_index,
            agents=agents,
            skills=skills,
            train_progress=train_progress,
            gamefiles=self._fixed_val_gamefiles(state),
            dataset=self.config.val_eval_dataset,
            phase="periodic_val",
            condition_prefix="online_periodic_val",
            attempt_dir=attempt_dir,
        )

    def _maybe_run_test_eval(
        self,
        state: dict[str, Any],
        *,
        outcome: dict[str, Any],
        round_index: int,
        agents: list[AgentSpec],
        skills: list[Skill],
        train_progress: int,
        attempt_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        if not self._should_run_test_eval(outcome, round_index=round_index):
            return []
        added_agents = self._segment_added_agents(outcome)
        phase = (
            "periodic_test"
            if self.config.test_eval_every_segment
            else "post_add_agent_test"
        )
        condition = (
            "online_periodic_test"
            if self.config.test_eval_every_segment
            else "online_post_add_agent_test"
        )
        return self._maybe_run_heldout_eval(
            state,
            outcome=outcome,
            round_index=round_index,
            agents=agents,
            skills=skills,
            train_progress=train_progress,
            gamefiles=self._fixed_test_gamefiles(state),
            dataset=self.config.test_eval_dataset,
            phase=phase,
            condition_prefix=condition,
            attempt_dir=attempt_dir,
            extra_fields={
                "added_agent_names": added_agents,
                "fixed_test_pool": True,
            },
        )
    @staticmethod
    def _append_learning_curve_checkpoint(
        state: dict[str, Any],
        *,
        round_index: int,
        train_progress: int,
    ) -> None:
        train_trials = [
            trial
            for trial in (state.get("all_trials") or [])
            if str(trial.get("dataset_split", "train")) == "train"
            or "dataset_split" not in trial
        ]
        val_trials = list(state.get("val_trials") or [])
        test_trials = list(state.get("test_trials") or [])
        train_wins = sum(int(trial.get("won", False)) for trial in train_trials)

        def _latest_batch_rate(trials: list[dict[str, Any]]) -> float | None:
            if not trials:
                return None
            # Prefer the batch evaluated at this train_progress (periodic pools).
            batch = [
                trial
                for trial in trials
                if int(trial.get("train_progress_at_eval", -1)) == train_progress
            ]
            if not batch:
                # Sparse cadence: no eval this segment — do not mix older pools.
                return None
            wins = sum(int(trial.get("won", False)) for trial in batch)
            return wins / len(batch)

        def _cumulative_rate(trials: list[dict[str, Any]]) -> float | None:
            if not trials:
                return None
            wins = sum(int(trial.get("won", False)) for trial in trials)
            return wins / len(trials)

        # Fixed-pool periodic evals: report the latest checkpoint batch SR.
        # Shadow / mixed historical vals keep cumulative SR.
        val_is_periodic = any(
            trial.get("fixed_eval_pool") or trial.get("phase") == "periodic_val"
            for trial in val_trials
        )
        test_is_periodic = any(
            trial.get("fixed_eval_pool")
            or trial.get("phase") in {"periodic_test", "post_add_agent_test"}
            for trial in test_trials
        )
        checkpoints = state.setdefault("learning_curve_checkpoints", [])
        previous = checkpoints[-1] if checkpoints else {}
        val_rate = (
            _latest_batch_rate(val_trials)
            if val_is_periodic
            else _cumulative_rate(val_trials)
        )
        test_rate = (
            _latest_batch_rate(test_trials)
            if test_is_periodic
            else _cumulative_rate(test_trials)
        )
        if val_rate is None and previous:
            val_rate = previous.get("val_success_rate")
        if test_rate is None and previous:
            test_rate = previous.get("test_success_rate")
        checkpoints.append(
            {
                "segment": round_index,
                "train_progress": train_progress,
                "train_success_rate": (
                    train_wins / len(train_trials) if train_trials else 0.0
                ),
                "val_success_rate": val_rate,
                "test_success_rate": test_rate,
                "train_trials": len(train_trials),
                "val_trials": len(val_trials),
                "test_trials": len(test_trials),
            }
        )

    @staticmethod
    def _trial_to_trajectory(trial: EvaluationTrial, round_index: int) -> dict[str, Any]:
        return {
            "framework": "sage_mas",
            "phase": "online_segment",
            "evolution_round": round_index,
            "gamefile": trial.task_id,
            "task_family": trial.task_family,
            "task": trial.task,
            "won": trial.won,
            "num_steps": trial.num_steps,
            "token_cost": trial.cost,
            "assigned_primary_agent": trial.assigned_primary_agent,
            "assignment_rationale": trial.assignment_rationale,
            "eligible_agents": trial.eligible_agents,
            "dispatch_layer": trial.dispatch_layer,
            "dispatch_evidence": trial.dispatch_evidence,
            "steps": trial.steps,
        }
