"""WebShop organization evaluator for SAGE online play / shadow."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from examples.prompt_agent.gpt4o_webshop import (
    build_webshop_env_manager,
    pad_goal_indices,
    task_family_from_instruction,
)
from sage_mas.alfworld_evaluator import (
    AlfWorldEvaluatorConfig,
    EvaluationTrial,
    _get_enhancement_config,
)
from sage_mas.enhanced_integration import EnhancedMASRuntime
from sage_mas.executor_dispatch import ExecutorDispatcher
from sage_mas.runtime import ChatBackend
from sage_mas.schemas import AgentSpec, Skill
from sage_mas.skill_activation import SkillPreconditionMatcher
from sage_mas.specialist_action_guard import SpecialistActionGuard
from sage_mas.webshop_ids import goal_uri, parse_goal_idx

# Reuse the shared evaluator knobs; WebShop does not need a parallel dataclass.
WebShopEvaluatorConfig = AlfWorldEvaluatorConfig


@dataclass(slots=True)
class _WebShopBatchSlot:
    goal_idx: int
    task_id: str


class WebShopOrganizationEvaluator:
    """Play WebShop episodes; task ids are ``webshop://goal/{idx}``."""

    def __init__(
        self,
        backend: ChatBackend,
        config: WebShopEvaluatorConfig | None = None,
        specialist_backend: ChatBackend | None = None,
    ):
        self.backend = backend
        self.specialist_backend = specialist_backend
        self.config = config or WebShopEvaluatorConfig()
        self.precondition_matcher = SkillPreconditionMatcher()
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
        """``gamefiles`` accepts ``webshop://goal/{idx}`` or bare ints."""
        trials: list[EvaluationTrial] = []
        task_ids = [str(item) for item in gamefiles]
        for offset in range(0, len(task_ids), self.config.parallel_envs):
            batch = task_ids[offset : offset + self.config.parallel_envs]
            trials.extend(
                self._evaluate_batch(
                    agents=agents,
                    skills=skills,
                    task_ids=batch,
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
        task_ids: list[str],
        condition: str,
        injected_skills: list[Skill] | None,
        batch_seed: int,
    ) -> list[EvaluationTrial]:
        slots = [
            _WebShopBatchSlot(
                goal_idx=parse_goal_idx(task_id),
                task_id=(
                    task_id
                    if str(task_id).startswith("webshop://")
                    else goal_uri(parse_goal_idx(task_id))
                ),
            )
            for task_id in task_ids
        ]
        goal_indices = [slot.goal_idx for slot in slots]
        env_manager = build_webshop_env_manager(
            env_num=len(slots),
            seed=batch_seed,
            is_train=False,
            num_cpus_per_worker=self.config.num_cpus_per_worker,
            history_length=self.config.history_length,
        )
        runtime = EnhancedMASRuntime(
            agents=agents,
            skills=skills,
            backend=self.backend,
            specialist_backend=self.specialist_backend,
            injected_skills=[],
            max_advisors=self.config.max_advisors,
            prompt_style=self.config.prompt_style,  # type: ignore[arg-type]
            enable_action_guards=False,
            short_specialist_prompts=self.config.short_specialist_prompts,
            compact_alfworld_prompts=False,
            allow_executor_skill_injection=(
                self.config.allow_executor_skill_injection
            ),
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

        try:
            print(
                f"[webshop_eval] reset begin condition={condition} "
                f"n={len(slots)} goals={goal_indices}",
                flush=True,
            )
            padded = pad_goal_indices(goal_indices, len(slots))
            obs, infos = env_manager.reset({"goal_indices": padded})
            print(
                f"[webshop_eval] reset done condition={condition} "
                f"n={len(slots)}",
                flush=True,
            )
            tasks = [str(task) for task in env_manager.tasks]
            families = [
                task_family_from_instruction(tasks[index])
                for index in range(len(slots))
            ]
            assignments = [
                self.dispatcher.assign(
                    task=tasks[index],
                    task_family=families[index],
                    agents=agents,
                    skills=skills,
                    executor_name=runtime.executor.name,
                    gamefile=slots[index].task_id,
                )
                for index in range(len(slots))
            ]
            dones = [False] * len(slots)
            token_costs = [0] * len(slots)
            steps: list[list[dict[str, Any]]] = [[] for _ in slots]
            activation_steps: list[dict[str, int]] = [{} for _ in slots]
            runtime_errors: list[list[dict[str, Any]]] = [[] for _ in slots]
            actions_by_agent: list[dict[str, int]] = [{} for _ in slots]
            final_infos: list[dict[str, Any]] = [dict(info) for info in infos]

            for step_index in range(self.config.max_steps):
                actions = ["None"] * len(slots)
                action_results: dict[int, Any] = {}
                prompts = []
                for index in range(len(slots)):
                    if dones[index]:
                        continue
                    prompt = obs["text"][index]
                    if ignore_assigned:
                        active_assigned_names: set[str] = set()
                    elif assigned_skills and self.config.gate_assigned_skills:
                        active_assigned, assigned_matches = (
                            self.precondition_matcher.active_skills(
                                assigned_skills,
                                tasks[index],
                                slots[index].task_id,
                                steps[index],
                            )
                        )
                        active_assigned_names = {
                            skill.skill_name for skill in active_assigned
                        }
                        for skill_name, match in assigned_matches.items():
                            if (
                                match.applicable
                                and skill_name not in activation_steps[index]
                            ):
                                activation_steps[index][
                                    skill_name
                                ] = step_index + 1
                    else:
                        active_assigned_names = assigned_skill_names
                    if injected_skills and self.config.gate_injected_skills:
                        active_injected, _matches = (
                            self.precondition_matcher.active_skills(
                                injected_skills,
                                tasks[index],
                                slots[index].task_id,
                                steps[index],
                            )
                        )
                    else:
                        active_injected = list(injected_skills or [])
                    active_injected = self._rank_injected_skills(
                        active_injected,
                        families[index],
                    )[: self.config.max_injected_skills]
                    for skill in active_injected:
                        activation_steps[index].setdefault(
                            skill.skill_name,
                            step_index + 1,
                        )
                    prompts.append(
                        (
                            index,
                            prompt,
                            active_injected,
                            active_assigned_names,
                            assignments[index].primary_agent,
                            tasks[index],
                            families[index],
                        )
                    )

                with ThreadPoolExecutor(
                    max_workers=self.config.api_concurrency
                ) as pool:
                    prompt_by_index = {
                        index: prompt
                        for (
                            index,
                            prompt,
                            *_rest,
                        ) in prompts
                    }
                    futures = {
                        pool.submit(
                            runtime.act,
                            prompt,
                            active_injected,
                            active_assigned_names,
                            preferred_actor,
                            task,
                            task_family,
                            steps[index],
                            slots[index].task_id,
                        ): index
                        for (
                            index,
                            prompt,
                            active_injected,
                            active_assigned_names,
                            preferred_actor,
                            task,
                            task_family,
                        ) in prompts
                    }
                    successful_calls = 0
                    if prompts:
                        print(
                            f"[webshop_eval] llm step={step_index + 1}/"
                            f"{self.config.max_steps} condition={condition} "
                            f"calls={len(prompts)} "
                            f"alive={sum(1 for done in dones if not done)}",
                            flush=True,
                        )
                    for future in as_completed(futures):
                        index = futures[future]
                        try:
                            result = future.result()
                            actions[index] = result.action
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
                            actions[index] = (
                                "<think>runtime error</think>"
                                "<action>search[product]</action>"
                            )
                            action_results[index] = exc
                            error_record = {
                                "step": step_index + 1,
                                "condition": condition,
                                "task_id": slots[index].task_id,
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                            runtime_errors[index].append(error_record)
                            logging.exception(
                                "SAGE-MAS WebShop LLM call failed: "
                                "condition=%s task=%s step=%s",
                                condition,
                                slots[index].task_id,
                                step_index + 1,
                            )

                if prompts and successful_calls == 0:
                    latest_errors = [
                        runtime_errors[index][-1]
                        for index, *_ in prompts
                        if runtime_errors[index]
                    ]
                    raise RuntimeError(
                        "All active SAGE-MAS WebShop LLM calls failed at "
                        f"step {step_index + 1} for condition "
                        f"{condition!r}: {latest_errors}"
                    )

                previous_anchors = list(obs["anchor"])
                previous_progress = [
                    self._goal_progress(final_infos[index])
                    for index in range(len(slots))
                ]
                next_obs, rewards, env_dones, infos = env_manager.step(actions)
                for index in range(len(slots)):
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
                        "goal_progress_after": self._goal_progress(
                            infos[index]
                        ),
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
                    steps[index].append(record)
                    dones[index] = bool(env_dones[index])

                obs = next_obs
                if all(dones):
                    break

            return [
                EvaluationTrial(
                    task_id=slot.task_id,
                    task=tasks[index],
                    task_family=families[index],
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
                for index, slot in enumerate(slots)
            ]
        finally:
            env_manager.close()

    def _rank_injected_skills(
        self,
        skills: list[Skill],
        task_family: str,
    ) -> list[Skill]:
        prefer_proven = bool(getattr(self.config, "prefer_proven_skills", False))

        def key(skill: Skill) -> tuple[int, float, int, int, float, str]:
            credit = skill.metadata.get("skill_credit")
            credit = credit if isinstance(credit, dict) else {}
            scoped = task_family in set(skill.applicable_task_families)
            uses = int(credit.get("uses", 0))
            score = float(credit.get("score", 0.0))
            utility_raw = skill.metadata.get("utility")
            try:
                utility = (
                    float(utility_raw) if utility_raw is not None else score
                )
            except (TypeError, ValueError):
                utility = score
            mu = skill.marginal_utility
            if mu is None and skill.metadata.get("marginal_utility") is not None:
                try:
                    mu = float(skill.metadata["marginal_utility"])
                except (TypeError, ValueError):
                    mu = None
            has_positive_mu = 0 if mu is not None and float(mu) > 0.0 else 1
            mu_sort = -float(mu) if mu is not None else 0.0
            uses_sort = -uses if prefer_proven else uses
            return (
                0 if scoped else 1,
                -utility,
                uses_sort,
                has_positive_mu,
                mu_sort,
                skill.skill_name,
            )

        return sorted(skills, key=key)

    @staticmethod
    def _goal_progress(info: dict[str, Any]) -> float:
        for key in ("task_score", "goal_progress", "goal_condition_success_rate"):
            value = info.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0
