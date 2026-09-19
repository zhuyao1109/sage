"""Reusable ALFWorld evaluator for skill probes and organization shadows."""

from __future__ import annotations

import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from examples.prompt_agent.gpt4o_alfworld import (
    _extract_task_from_obs,
    _list_unseen_gamefiles,
    _select_hard_gamefiles,
    _task_family_from_gamefile,
    build_alfworld_env_manager,
)
from sage_mas.executor_dispatch import (
    ExecutorDispatchConfig,
    ExecutorDispatcher,
    dispatch_config_from_mapping,
)
from sage_mas.runtime import ChatBackend, MASRuntime
from sage_mas.schemas import AgentSpec, Skill, SkillStatus
from sage_mas.specialist_action_guard import SpecialistActionGuard
from sage_mas.skill_prompt_retrieval import (
    build_skill_retrieval_prompt,
    parse_skill_selection,
)
from sage_mas.skill_recall import (
    BM25Index,
    build_recall_index,
    build_recall_query_prompt,
    parse_recall_query,
    tokenize,
)

# Import enhanced components
from sage_mas.enhanced_integration import EnhancedMASRuntime
from sage_mas.enhancement_config_loader import load_enhancement_config
from pathlib import Path

# Load enhancement config once
_ENHANCEMENT_CONFIG = None
def _get_enhancement_config():
    global _ENHANCEMENT_CONFIG
    if _ENHANCEMENT_CONFIG is None:
        config_path = Path("sage_mas/enhancement_config.yaml")
        if config_path.exists():
            _ENHANCEMENT_CONFIG = load_enhancement_config(config_path)
            print("[ENHANCED] Loaded enhancement config")
        else:
            from sage_mas.enhancement_config_loader import get_default_config
            _ENHANCEMENT_CONFIG = get_default_config()
            print("[ENHANCED] Using default config (enrichment disabled)")
    return _ENHANCEMENT_CONFIG


@dataclass(slots=True)
class AlfWorldEvaluatorConfig:
    max_steps: int = 50
    parallel_envs: int = 10
    api_concurrency: int = 10
    num_cpus_per_worker: float = 0.05
    history_length: int = 0
    seed: int = 1
    save_steps: bool = True
    max_advisors: int | None = None
    # Max skills mounted from one on-demand BM25 hit list.
    max_injected_skills: int = 2
    # Probe-only: skip prompt-based retrieval and inject every candidate
    # skill directly (discovery forks / paired marginal-utility probes).
    force_inject_skills: bool = False
    # On-demand retrieval (every Executor step): the model writes a query
    # from the current observation, or "none", and BM25 searches the bank.
    # "bm25" = on; "none" = legacy catalog selection.
    skill_recall_backend: str = "bm25"
    skill_recall_top_k: int = 8
    # True: LLM writes the query (and may decline). False: task text is the query.
    skill_recall_query_llm: bool = True
    # Deprecated no-op: visited-location expert memory removed.
    use_visited_location_memory: bool = False
    # Skip org-assigned / bank skills (executor-only reporting ablation).
    ignore_assigned_skills: bool = False
    # "alfworld": call LLM on GiGPO ALFWORLD_TEMPLATE observations directly.
    # "mas": legacy custom Executor / Advisor system prompts.
    prompt_style: str = "alfworld"
    # Deprecated no-op: hand-written action rewriting removed.
    enable_action_guards: bool = False
    # Hard task-family controllers are an ablation/upper-bound only. The SAGE
    # main path keeps specialists LLM-driven with injected skill protocols.
    enable_specialist_controllers: bool = False
    short_specialist_prompts: bool = True
    # Observation-only compact path; does not inject expert rules.
    compact_alfworld_prompts: bool = False
    # Counterfactual/discovery and explicit online-credit probation may enable
    # learned Skill injection. It remains off for ordinary collection.
    allow_executor_skill_injection: bool = False
    # full | soft | sparse_soft | hybrid_soft — see skill_inject_sparse_soft
    skill_inject_mode: str = "full"
    skill_reinject_every: int = 0
    # Executor judges the task and may hand episodes to specialists.
    executor_dispatch: ExecutorDispatchConfig = field(
        default_factory=ExecutorDispatchConfig
    )

@dataclass(slots=True)
class EvaluationTrial:
    task_id: str
    task: str
    task_family: str
    condition: str
    reward: float
    cost: float
    won: bool
    num_steps: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    activated_skill_names: list[str] = field(default_factory=list)
    skill_activation_steps: dict[str, int] = field(default_factory=dict)
    runtime_error_count: int = 0
    runtime_errors: list[dict[str, Any]] = field(default_factory=list)
    assigned_primary_agent: str | None = None
    assignment_rationale: str | None = None
    eligible_agents: list[str] = field(default_factory=list)
    dispatch_layer: str | None = None
    dispatch_evidence: dict[str, Any] = field(default_factory=dict)
    actions_by_agent: dict[str, int] = field(default_factory=dict)


class AlfWorldOrganizationEvaluator:
    def __init__(
        self,
        backend: ChatBackend,
        config: AlfWorldEvaluatorConfig | None = None,
        specialist_backend: ChatBackend | None = None,
    ):
        self.backend = backend
        # When set, non-Executor agents (specialists) call this model during play.
        self.specialist_backend = specialist_backend
        self.config = config or AlfWorldEvaluatorConfig()
        controllers_on = bool(self.config.enable_specialist_controllers)
        if not controllers_on:
            self.config.executor_dispatch.require_controller_for_eligibility = (
                False
            )
        self.dispatcher = ExecutorDispatcher(
            self.config.executor_dispatch,
            backend=self.backend,
        )
        self.specialist_action_guard = SpecialistActionGuard(
            enabled=controllers_on
        )

    def evaluate(
        self,
        agents: list[AgentSpec],
        skills: list[Skill],
        gamefiles: list[str],
        condition: str,
        injected_skills: list[Skill] | None = None,
    ) -> list[EvaluationTrial]:
        trials = []
        for offset in range(0, len(gamefiles), self.config.parallel_envs):
            batch_games = gamefiles[offset : offset + self.config.parallel_envs]
            trials.extend(
                self._evaluate_batch(
                    agents=agents,
                    skills=skills,
                    gamefiles=batch_games,
                    condition=condition,
                    injected_skills=injected_skills,
                    batch_seed=self.config.seed + offset,
                )
            )
        return trials

    def _evaluate_batch(
        self,
        agents: list[AgentSpec],
        skills: list[Skill],
        gamefiles: list[str],
        condition: str,
        injected_skills: list[Skill] | None,
        batch_seed: int,
    ) -> list[EvaluationTrial]:
        env_manager = build_alfworld_env_manager(
            env_num=len(gamefiles),
            seed=batch_seed,
            eval_dataset="eval_out_of_distribution",
            game_files_list=gamefiles,
            num_cpus_per_worker=self.config.num_cpus_per_worker,
        )
        env_manager.config.env.history_length = self.config.history_length
        runtime = EnhancedMASRuntime(
            agents=agents,
            skills=skills,
            backend=self.backend,
            specialist_backend=self.specialist_backend,
            injected_skills=[],
            max_advisors=self.config.max_advisors,
            prompt_style=self.config.prompt_style,  # type: ignore[arg-type]
            enable_action_guards=self.config.enable_action_guards,
            short_specialist_prompts=self.config.short_specialist_prompts,
            compact_alfworld_prompts=self.config.compact_alfworld_prompts,
            allow_executor_skill_injection=(
                self.config.allow_executor_skill_injection
            ),
            strict_skill_selection=True,
            skill_inject_mode=str(self.config.skill_inject_mode or "full"),
            skill_reinject_every=int(self.config.skill_reinject_every or 0),
            enhancement_config=_get_enhancement_config(),
        )
        ignore_assigned = self.config.ignore_assigned_skills
        if ignore_assigned:
            assigned_skill_names: set[str] = set()
            assigned_skills: list[Skill] = []
        else:
            assigned_skill_names = {
                skill_name
                for agent in agents
                for skill_name in agent.assigned_skills
            }
            assigned_skills = [
                skill
                for skill in skills
                if skill.skill_name in assigned_skill_names
            ]

        # Episode index over the whole bank. Each Executor step decides
        # whether to query it. Probes (force_inject) and recall=none skip it.
        recall_index: BM25Index | None = None
        recall_index_skills: list[Skill] = []
        if (
            self.config.skill_recall_backend != "none"
            and not self.config.force_inject_skills
        ):
            recall_index, recall_index_skills = build_recall_index(skills)
            if not recall_index_skills:
                recall_index = None

        try:
            print(
                f"[alfworld_eval] reset begin condition={condition} "
                f"n={len(gamefiles)} cpus={self.config.num_cpus_per_worker}",
                flush=True,
            )
            obs, infos = env_manager.reset({})
            print(
                f"[alfworld_eval] reset done condition={condition} "
                f"n={len(gamefiles)}",
                flush=True,
            )
            tasks = [_extract_task_from_obs(obs["anchor"][i]) for i in range(len(gamefiles))]
            assignments = [
                self.dispatcher.assign(
                    task=tasks[index],
                    task_family=_task_family_from_gamefile(gamefiles[index]),
                    agents=agents,
                    skills=skills,
                    executor_name=runtime.executor.name,
                    gamefile=gamefiles[index],
                )
                for index in range(len(gamefiles))
            ]
            dones = [False] * len(gamefiles)
            token_costs = [0] * len(gamefiles)
            steps: list[list[dict[str, Any]]] = [[] for _ in gamefiles]
            activation_steps: list[dict[str, int]] = [
                {} for _ in gamefiles
            ]
            runtime_errors: list[list[dict[str, Any]]] = [
                [] for _ in gamefiles
            ]
            actions_by_agent: list[dict[str, int]] = [
                {} for _ in gamefiles
            ]
            final_infos: list[dict[str, Any]] = [dict(info) for info in infos]

            for step_index in range(self.config.max_steps):
                actions = ["None"] * len(gamefiles)
                action_results = {}
                prompts = []
                for index in range(len(gamefiles)):
                    if dones[index]:
                        continue
                    prompts.append(
                        (
                            index,
                            obs["text"][index],
                            obs["anchor"][index],
                            assignments[index].primary_agent,
                            tasks[index],
                            _task_family_from_gamefile(gamefiles[index]),
                        )
                    )
                with ThreadPoolExecutor(max_workers=self.config.api_concurrency) as executor:
                    prompt_by_index = {
                        index: prompt
                        for (index, prompt, *_rest) in prompts
                    }
                    action_guard_results = {}
                    retrieval_by_index: dict[int, dict[str, Any]] = {}
                    futures = {
                        executor.submit(
                            self._act_with_skill_retrieval,
                            runtime,
                            observation=prompt,
                            retrieval_observation=anchor,
                            history_steps=steps[index],
                            gamefile=gamefiles[index],
                            primary_agent=preferred_actor,
                            task=task,
                            task_family=task_family,
                            assigned_skills=assigned_skills,
                            assigned_skill_names=assigned_skill_names,
                            injected_skills=injected_skills,
                            ignore_assigned=ignore_assigned,
                            agents=agents,
                            recall_index=recall_index,
                            recall_skills=recall_index_skills,
                        ): index
                        for (
                            index,
                            prompt,
                            anchor,
                            preferred_actor,
                            task,
                            task_family,
                        ) in prompts
                    }
                    successful_calls = 0
                    if prompts:
                        print(
                            f"[alfworld_eval] llm step={step_index + 1}/"
                            f"{self.config.max_steps} condition={condition} "
                            f"calls={len(prompts)} "
                            f"alive={sum(1 for d in dones if not d)}",
                            flush=True,
                        )
                    for future in as_completed(futures):
                        index = futures[future]
                        try:
                            result, retrieval_info, shown_skills = future.result()
                            if retrieval_info is not None:
                                retrieval_by_index[index] = retrieval_info
                                token_costs[index] += int(
                                    retrieval_info.get("token_cost") or 0
                                )
                            for skill_name in shown_skills:
                                activation_steps[index].setdefault(
                                    skill_name,
                                    step_index + 1,
                                )
                            guarded = self.specialist_action_guard.guard(
                                raw_action=result.action,
                                task=tasks[index],
                                gamefile=gamefiles[index],
                                observation=obs["anchor"][index],
                                prompt=prompt_by_index.get(index, ""),
                                steps=steps[index],
                                primary_agent=assignments[index].primary_agent,
                                agents=agents,
                                skills=skills,
                            )
                            actions[index] = guarded.action
                            action_guard_results[index] = guarded
                            action_results[index] = result
                            token_costs[index] += result.token_cost
                            if result.messages:
                                actor_name = result.messages[-1].agent_name
                                actions_by_agent[index][actor_name] = (
                                    actions_by_agent[index].get(actor_name, 0)
                                    + 1
                                )
                            successful_calls += 1
                        except Exception as exc:
                            actions[index] = "<think>runtime error</think><action>look</action>"
                            action_results[index] = exc
                            error_record = {
                                "step": step_index + 1,
                                "condition": condition,
                                "task_id": gamefiles[index],
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                            runtime_errors[index].append(error_record)
                            logging.exception(
                                "SAGE-MAS LLM call failed: condition=%s "
                                "task=%s step=%s",
                                condition,
                                gamefiles[index],
                                step_index + 1,
                            )

                if prompts and successful_calls == 0:
                    latest_errors = [
                        runtime_errors[index][-1]
                        for index, *_ in prompts
                        if runtime_errors[index]
                    ]
                    raise RuntimeError(
                        "All active SAGE-MAS LLM calls failed at "
                        f"step {step_index + 1} for condition "
                        f"{condition!r}: {latest_errors}"
                    )

                previous_anchors = list(obs["anchor"])
                previous_progress = [
                    self._goal_progress(final_infos[index])
                    for index in range(len(gamefiles))
                ]
                next_obs, rewards, env_dones, infos = env_manager.step(actions)
                for index in range(len(gamefiles)):
                    if dones[index]:
                        continue
                    final_infos[index] = dict(infos[index])
                    result = action_results.get(index)
                    record = {
                        "step": step_index + 1,
                        "observation_before": previous_anchors[index],
                        "observation": next_obs["anchor"][index],
                        "action": actions[index],
                        "reward": float(rewards[index]),
                        "goal_progress_before": previous_progress[index],
                        "goal_progress_after": self._goal_progress(infos[index]),
                        "goal_progress_delta": max(
                            0.0,
                            self._goal_progress(infos[index])
                            - previous_progress[index],
                        ),
                        "is_action_valid": bool(
                            infos[index].get("is_action_valid", False)
                        ),
                    }
                    if self.config.save_steps and hasattr(result, "messages"):
                        record["agent_messages"] = [
                            {
                                "agent": message.agent_name,
                                "content": message.content,
                                "token_cost": message.token_cost,
                            }
                            for message in result.messages
                        ]
                    if index in retrieval_by_index:
                        record["skill_retrieval"] = retrieval_by_index[index]
                        if "shortlist" in retrieval_by_index[index]:
                            record["skill_recall"] = retrieval_by_index[index]
                    if index in action_guard_results:
                        record["action_guard"] = action_guard_results[index].as_dict()
                    steps[index].append(record)
                    dones[index] = bool(env_dones[index])

                obs = next_obs
                if all(dones):
                    break

            return [
                EvaluationTrial(
                    task_id=gamefile,
                    task=tasks[index],
                    task_family=_task_family_from_gamefile(gamefile),
                    condition=condition,
                    reward=float(bool(final_infos[index].get("won", False))),
                    cost=float(token_costs[index]),
                    won=bool(final_infos[index].get("won", False)),
                    num_steps=len(steps[index]),
                    steps=steps[index] if self.config.save_steps else [],
                    activated_skill_names=sorted(activation_steps[index]),
                    skill_activation_steps=activation_steps[index],
                    runtime_error_count=len(runtime_errors[index]),
                    runtime_errors=runtime_errors[index],
                    assigned_primary_agent=assignments[index].primary_agent,
                    assignment_rationale=assignments[index].rationale,
                    eligible_agents=list(assignments[index].eligible_agents),
                    dispatch_layer=assignments[index].dispatch_layer or None,
                    dispatch_evidence=dict(assignments[index].evidence),
                    actions_by_agent=dict(actions_by_agent[index]),
                )
                for index, gamefile in enumerate(gamefiles)
            ]
        finally:
            env_manager.close()

    @staticmethod
    def _executor_skill_candidates(
        assigned_skills: list[Skill],
        injected_skills: list[Skill] | None,
        *,
        ignore_assigned: bool,
    ) -> list[Skill]:
        """Catalog for prompt-based retrieval: verified assigned + injected.

        Order is stable (assigned first, then bank-injected) with no
        rule-based family re-ranking — selection is the Executor's own
        decision from the catalog.
        """
        candidates: list[Skill] = []
        seen: set[str] = set()
        if not ignore_assigned:
            for skill in assigned_skills:
                if skill.status != SkillStatus.VERIFIED:
                    continue
                if skill.skill_name in seen:
                    continue
                candidates.append(skill)
                seen.add(skill.skill_name)
        for skill in injected_skills or []:
            if skill.skill_name in seen:
                continue
            candidates.append(skill)
            seen.add(skill.skill_name)
        return candidates

    def _select_skills_with_prompt(
        self,
        observation: str,
        candidates: list[Skill],
        *,
        task: str | None = None,
    ) -> tuple[list[Skill], dict[str, Any]]:
        """The Executor model picks applicable skills from the catalog.

        Returns the selected skills plus a per-step record of the retrieval
        (candidates, selected names, raw reply) for trajectory logging.
        A failed retrieval call yields an empty selection instead of
        killing the step.
        """
        candidate_names = [skill.skill_name for skill in candidates]
        prompt = build_skill_retrieval_prompt(
            observation,
            candidates,
            self.config.max_injected_skills,
            task=task,
        )
        try:
            result = self.backend.complete("", prompt)
        except Exception as exc:
            logging.warning(
                "SAGE-MAS skill retrieval call failed: %s", exc
            )
            return [], {
                "candidates": candidate_names,
                "selected": [],
                "raw_reply": None,
                "error": f"{type(exc).__name__}: {exc}",
                "token_cost": 0,
            }
        selected = parse_skill_selection(
            result.content,
            candidates,
            self.config.max_injected_skills,
        )
        return selected, {
            "candidates": candidate_names,
            "selected": [skill.skill_name for skill in selected],
            "raw_reply": result.content,
            "error": None,
            "token_cost": result.total_tokens,
        }

    def _recall_skills_with_prompt(
        self,
        recall_index: BM25Index,
        recall_skills: list[Skill],
        *,
        task: str,
        observation: str,
    ) -> tuple[list[Skill], dict[str, Any]]:
        """One-step on-demand recall.

        The Executor writes a query for the current observation, or declines.
        A real query searches the whole bank; ``none`` / a failed call mounts
        nothing. Hits are capped by ``max_injected_skills``.
        """
        info: dict[str, Any] = {
            "backend": self.config.skill_recall_backend,
            "top_k": self.config.skill_recall_top_k,
            "query": None,
            "raw_reply": None,
            "error": None,
            "token_cost": 0,
            "skipped": False,
            "shortlist": [],
            "selected": [],
        }
        query: str | None = None
        if self.config.skill_recall_query_llm:
            prompt = build_recall_query_prompt(task, observation)
            try:
                result = self.backend.complete("", prompt)
                info["raw_reply"] = result.content
                info["token_cost"] = result.total_tokens
                query = parse_recall_query(result.content)
            except Exception as exc:
                logging.warning("SAGE-MAS skill recall query failed: %s", exc)
                info["error"] = f"{type(exc).__name__}: {exc}"
                info["skipped"] = True
                return [], info
        else:
            query = task
        if not query:
            info["skipped"] = True
            return [], info
        info["query"] = query
        mount_k = min(
            max(0, self.config.max_injected_skills),
            max(0, self.config.skill_recall_top_k),
        )
        if mount_k <= 0:
            info["skipped"] = True
            return [], info
        ranked = recall_index.search(tokenize(query), mount_k)
        chosen = [(index, score) for index, score in ranked if score > 0.0]
        shortlist = [recall_skills[index] for index, _score in chosen]
        info["shortlist"] = [
            {"name": recall_skills[index].skill_name, "score": round(score, 4)}
            for index, score in chosen
        ]
        info["selected"] = [skill.skill_name for skill in shortlist]
        return shortlist, info

    def _act_with_skill_retrieval(
        self,
        runtime: MASRuntime,
        *,
        observation: str,
        history_steps: list[dict[str, Any]],
        gamefile: str,
        primary_agent: str,
        task: str,
        task_family: str,
        assigned_skills: list[Skill],
        assigned_skill_names: set[str],
        injected_skills: list[Skill] | None,
        ignore_assigned: bool,
        agents: list[AgentSpec],
        recall_index: BM25Index | None = None,
        recall_skills: list[Skill] | None = None,
        retrieval_observation: str | None = None,
    ) -> tuple[Any, dict[str, Any] | None, list[str]]:
        """On-demand skill retrieval, then the action call.

        Returns ``(runtime_action, retrieval_info, shown_skill_names)``.
        Only an Executor primary retrieves. Each step the model may write a
        query; BM25 hits are mounted directly. Specialists keep their
        verified assigned contract for the whole episode.
        """
        active_injected: list[Skill] = []
        active_assigned_names: set[str] = set()
        shown: list[str] = []
        retrieval_info: dict[str, Any] | None = None
        if primary_agent == runtime.executor.name:
            if self.config.force_inject_skills:
                active_injected = list(injected_skills or [])
                if not ignore_assigned:
                    active_assigned_names = set(assigned_skill_names)
                shown = list(
                    dict.fromkeys(
                        [skill.skill_name for skill in active_injected]
                        + [
                            skill.skill_name
                            for skill in assigned_skills
                            if skill.skill_name in active_assigned_names
                            and skill.status == SkillStatus.VERIFIED
                        ]
                    )
                )
            elif recall_index is not None:
                shortlist, retrieval_info = self._recall_skills_with_prompt(
                    recall_index,
                    recall_skills or [],
                    task=task,
                    observation=(
                        retrieval_observation
                        if retrieval_observation is not None
                        else observation
                    ),
                )
                active_injected = list(shortlist)
                shown = [skill.skill_name for skill in shortlist]
            else:
                candidates = self._executor_skill_candidates(
                    assigned_skills,
                    injected_skills,
                    ignore_assigned=ignore_assigned,
                )
                if candidates and self.config.max_injected_skills > 0:
                    selected, retrieval_info = self._select_skills_with_prompt(
                        observation,
                        candidates,
                        task=task,
                    )
                    selected_names = {skill.skill_name for skill in selected}
                    active_injected = [
                        skill
                        for skill in (injected_skills or [])
                        if skill.skill_name in selected_names
                    ]
                    active_assigned_names = selected_names & assigned_skill_names
                    shown = [skill.skill_name for skill in selected]
        else:
            # Specialist primary: ungated assigned names keep the dispatch
            # fallback intact; the specialist prompt mounts its own verified
            # contract for the whole episode regardless.
            if not ignore_assigned:
                active_assigned_names = set(assigned_skill_names)
            primary_spec = next(
                (agent for agent in agents if agent.name == primary_agent),
                None,
            )
            if primary_spec is not None:
                primary_skill_names = set(primary_spec.assigned_skills or [])
                shown = [
                    skill.skill_name
                    for skill in assigned_skills
                    if skill.skill_name in primary_skill_names
                    and skill.status == SkillStatus.VERIFIED
                ]
        result = runtime.act(
            observation,
            active_injected,
            active_assigned_names,
            primary_agent,
            task,
            task_family,
            history_steps,
            gamefile,
        )
        return result, retrieval_info, shown

    @staticmethod
    def _goal_progress(info: dict[str, Any]) -> float:
        for key in (
            "goal_condition_success_rate",
            "goal_progress",
            "task_score",
        ):
            value = info.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0


    def replay_from_anchor(
        self,
        *,
        agents: list[AgentSpec],
        skills: list[Skill],
        gamefile: str,
        condition: str,
        replay_steps: list[dict[str, Any]],
        replay_until_step: int | None,
        injected_skills: list[Skill] | None = None,
    ) -> EvaluationTrial:
        """Replay a baseline prefix, then continue locally from an anchor.

        ALFWorld does not expose a stable state snapshot/restore API here, so
        this method reconstructs the anchor state by resetting the same game and
        replaying recorded environment actions up to the matched anchor step.
        The LLM is only used after the anchor, which makes the paired probe a
        local causal test around the AtomicOp node instead of a full free rerun.
        """
        prefix_length = max(0, int(replay_until_step or 0))
        env_manager = build_alfworld_env_manager(
            env_num=1,
            seed=self.config.seed,
            eval_dataset="eval_out_of_distribution",
            game_files_list=[gamefile],
            num_cpus_per_worker=self.config.num_cpus_per_worker,
        )
        env_manager.config.env.history_length = self.config.history_length
        runtime = EnhancedMASRuntime(
            agents=agents,
            skills=skills,
            backend=self.backend,
            specialist_backend=self.specialist_backend,
            injected_skills=[],
            max_advisors=self.config.max_advisors,
            prompt_style=self.config.prompt_style,  # type: ignore[arg-type]
            enable_action_guards=self.config.enable_action_guards,
            short_specialist_prompts=self.config.short_specialist_prompts,
            compact_alfworld_prompts=self.config.compact_alfworld_prompts,
            allow_executor_skill_injection=(
                self.config.allow_executor_skill_injection
            ),
            strict_skill_selection=True,
            skill_inject_mode=str(self.config.skill_inject_mode or "full"),
            skill_reinject_every=int(self.config.skill_reinject_every or 0),
            enhancement_config=_get_enhancement_config(),
        )
        ignore_assigned = self.config.ignore_assigned_skills
        if ignore_assigned:
            assigned_skill_names: set[str] = set()
            assigned_skills: list[Skill] = []
        else:
            assigned_skill_names = {
                skill_name
                for agent in agents
                for skill_name in agent.assigned_skills
            }
            assigned_skills = [
                skill
                for skill in skills
                if skill.skill_name in assigned_skill_names
            ]

        recall_index: BM25Index | None = None
        recall_index_skills: list[Skill] = []
        if (
            self.config.skill_recall_backend != "none"
            and not self.config.force_inject_skills
        ):
            recall_index, recall_index_skills = build_recall_index(skills)
            if not recall_index_skills:
                recall_index = None

        try:
            obs, infos = env_manager.reset({})
            task = _extract_task_from_obs(obs["anchor"][0])
            task_family = _task_family_from_gamefile(gamefile)
            assignment = self.dispatcher.assign(
                task=task,
                task_family=task_family,
                agents=agents,
                skills=skills,
                executor_name=runtime.executor.name,
                gamefile=gamefile,
            )
            done = False
            token_cost = 0
            steps: list[dict[str, Any]] = []
            activation_steps: dict[str, int] = {}
            runtime_errors: list[dict[str, Any]] = []
            final_info = dict(infos[0])

            for source_step in replay_steps[:prefix_length]:
                if done:
                    break
                action = str(source_step.get("action", ""))
                next_obs, rewards, env_dones, infos = env_manager.step([action])
                final_info = dict(infos[0])
                record = {
                    "step": len(steps) + 1,
                    "observation": next_obs["anchor"][0],
                    "action": action,
                    "reward": float(rewards[0]),
                    "is_action_valid": bool(
                        infos[0].get("is_action_valid", False)
                    ),
                    "replayed_prefix": True,
                }
                if self.config.save_steps and source_step.get("agent_messages"):
                    record["agent_messages"] = source_step["agent_messages"]
                steps.append(record)
                done = bool(env_dones[0])
                obs = next_obs

            while not done and len(steps) < self.config.max_steps:
                step_number = len(steps) + 1
                prompt = obs["text"][0]
                retrieval_info: dict[str, Any] | None = None

                try:
                    result, retrieval_info, shown_skills = (
                        self._act_with_skill_retrieval(
                            runtime,
                            observation=prompt,
                            retrieval_observation=obs["anchor"][0],
                            history_steps=steps,
                            gamefile=gamefile,
                            primary_agent=assignment.primary_agent,
                            task=task,
                            task_family=task_family,
                            assigned_skills=assigned_skills,
                            assigned_skill_names=assigned_skill_names,
                            injected_skills=injected_skills,
                            ignore_assigned=ignore_assigned,
                            agents=agents,
                            recall_index=recall_index,
                            recall_skills=recall_index_skills,
                        )
                    )
                    for skill_name in shown_skills:
                        activation_steps.setdefault(skill_name, step_number)
                    if retrieval_info is not None:
                        token_cost += int(retrieval_info.get("token_cost") or 0)
                    action = result.action
                    guarded = self.specialist_action_guard.guard(
                        raw_action=action,
                        task=task,
                        gamefile=gamefile,
                        observation=obs["anchor"][0],
                        prompt=prompt,
                        steps=steps,
                        primary_agent=assignment.primary_agent,
                        agents=agents,
                        skills=skills,
                    )
                    action = guarded.action
                    token_cost += result.token_cost
                except Exception as exc:
                    action = "<think>runtime error</think><action>look</action>"
                    result = exc
                    guarded = None
                    error_record = {
                        "step": step_number,
                        "condition": condition,
                        "task_id": gamefile,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                    runtime_errors.append(error_record)
                    logging.exception(
                        "SAGE-MAS replay LLM call failed: condition=%s "
                        "task=%s step=%s",
                        condition,
                        gamefile,
                        step_number,
                    )

                observation_before = obs["anchor"][0]
                progress_before = self._goal_progress(final_info)
                next_obs, rewards, env_dones, infos = env_manager.step([action])
                final_info = dict(infos[0])
                record = {
                    "step": step_number,
                    "observation_before": observation_before,
                    "observation": next_obs["anchor"][0],
                    "action": action,
                    "reward": float(rewards[0]),
                    "goal_progress_before": progress_before,
                    "goal_progress_after": self._goal_progress(final_info),
                    "goal_progress_delta": max(
                        0.0,
                        self._goal_progress(final_info) - progress_before,
                    ),
                    "is_action_valid": bool(
                        infos[0].get("is_action_valid", False)
                    ),
                    "replayed_prefix": False,
                }
                if self.config.save_steps and hasattr(result, "messages"):
                    record["agent_messages"] = [
                        {
                            "agent": message.agent_name,
                            "content": message.content,
                            "token_cost": message.token_cost,
                        }
                        for message in result.messages
                    ]
                if retrieval_info is not None:
                    record["skill_retrieval"] = retrieval_info
                    if "shortlist" in retrieval_info:
                        record["skill_recall"] = retrieval_info
                if guarded is not None:
                    record["action_guard"] = guarded.as_dict()
                steps.append(record)
                done = bool(env_dones[0])
                obs = next_obs

            return EvaluationTrial(
                task_id=gamefile,
                task=task,
                task_family=task_family,
                condition=condition,
                reward=float(bool(final_info.get("won", False))),
                cost=float(token_cost),
                won=bool(final_info.get("won", False)),
                num_steps=len(steps),
                steps=steps if self.config.save_steps else [],
                activated_skill_names=sorted(activation_steps),
                skill_activation_steps=activation_steps,
                runtime_error_count=len(runtime_errors),
                runtime_errors=runtime_errors,
                assigned_primary_agent=assignment.primary_agent,
                assignment_rationale=assignment.rationale,
                eligible_agents=list(assignment.eligible_agents),
                dispatch_layer=assignment.dispatch_layer or None,
                dispatch_evidence=dict(assignment.evidence),
            )
        finally:
            env_manager.close()


ALFWORLD_SPLITS = ("train", "valid_seen", "valid_unseen")


def list_alfworld_split_gamefiles(data_path: str, split: str) -> list[str]:
    """List ``game.tw-pddl`` files under ``json_2.1.1/<split>``."""
    normalized = str(split).strip().lower()
    if normalized not in ALFWORLD_SPLITS:
        raise ValueError(
            f"Unknown ALFWorld split={split!r}; use one of {ALFWORLD_SPLITS}."
        )
    if normalized == "valid_unseen":
        return _list_unseen_gamefiles(data_path)
    split_root = os.path.join(data_path, "json_2.1.1", normalized)
    gamefiles = []
    for root, _, files in os.walk(split_root):
        if "game.tw-pddl" in files:
            gamefiles.append(os.path.join(root, "game.tw-pddl"))
    gamefiles.sort()
    if not gamefiles:
        raise RuntimeError(
            f"No ALFWorld game files found under {split_root}"
        )
    return gamefiles


def select_family_proportional_gamefiles(
    candidates: list[str],
    num_games: int,
    *,
    seed: int = 1,
) -> list[str]:
    """Round-robin sample across task families so every family appears."""
    if len(candidates) < num_games:
        raise RuntimeError(
            f"Requested num_games={num_games} but only found "
            f"{len(candidates)} candidates."
        )
    by_family: dict[str, list[str]] = {}
    for gamefile in candidates:
        family = _task_family_from_gamefile(gamefile)
        by_family.setdefault(family, []).append(gamefile)
    rng = random.Random(seed)
    families = sorted(by_family.keys())
    pools = {family: list(paths) for family, paths in by_family.items()}
    for family in families:
        rng.shuffle(pools[family])
    selected: list[str] = []
    while len(selected) < num_games and any(pools[family] for family in families):
        for family in families:
            if pools[family] and len(selected) < num_games:
                selected.append(pools[family].pop())
    if len(selected) < num_games:
        raise RuntimeError(
            f"Family-proportional selection only produced {len(selected)} "
            f"of {num_games} requested games."
        )
    return selected


def select_prompt_agent_gamefiles(
    data_path: str,
    *,
    num_games: int = 134,
    game_selection: str = "first",
    excluded: set[str] | None = None,
    split: str = "valid_unseen",
    seed: int = 1,
) -> list[str]:
    """Match prompt-agent selection on a dataset split.

    Supported ``game_selection`` values: ``first``, ``hard``,
    ``family_proportional``.
    """
    excluded = excluded or set()
    all_gamefiles = list_alfworld_split_gamefiles(data_path, split)
    candidates = [gamefile for gamefile in all_gamefiles if gamefile not in excluded]
    selection = game_selection.lower()
    if selection == "first":
        if len(candidates) < num_games:
            raise RuntimeError(
                f"Requested num_games={num_games} but only found "
                f"{len(candidates)} {split} games after excluding "
                f"{len(excluded)}."
            )
        return candidates[:num_games]
    if selection == "hard":
        if len(candidates) < num_games:
            raise RuntimeError(
                f"Requested {num_games} hard games but only found "
                f"{len(candidates)} {split} games after exclusions."
            )
        return _select_hard_gamefiles(candidates, num_games)
    if selection in {"family_proportional", "proportional"}:
        return select_family_proportional_gamefiles(
            candidates,
            num_games,
            seed=seed,
        )
    raise RuntimeError(
        f"Unknown game_selection={game_selection!r}; use "
        f"'first', 'hard', or 'family_proportional'."
    )


def sample_alfworld_gamefiles(
    data_path: str,
    num_tasks: int,
    seed: int,
    excluded: set[str] | None = None,
    task_families: set[str] | None = None,
    split: str = "valid_unseen",
) -> list[str]:
    excluded = excluded or set()
    candidates = [
        gamefile
        for gamefile in list_alfworld_split_gamefiles(data_path, split)
        if gamefile not in excluded
        and (
            not task_families
            or _task_family_from_gamefile(gamefile) in task_families
        )
    ]
    if len(candidates) < num_tasks:
        raise RuntimeError(
            f"Requested {num_tasks} ALFWorld tasks but only {len(candidates)} "
            f"{split} games remain after exclusions and task-family filtering."
        )
    random.Random(seed).shuffle(candidates)
    return candidates[:num_tasks]
