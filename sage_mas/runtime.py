"""Executable prompt-based runtime for AgentSpec organizations."""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

_ACTION_TAG_RE = re.compile(
    r"<action>\s*.+?\s*</action>",
    re.IGNORECASE | re.DOTALL,
)
_ACTION_FORMAT_RETRY_SUFFIX = (
    "\n\nFORMAT ERROR: your previous reply had no valid "
    "<action>...</action> block. Reply again with exactly:\n"
    "<think>brief reasoning</think>\n"
    "<action>one admissible action from the list</action>\n"
    "The text inside <action> must be a single admissible action only."
)
# Seeded specialists often ship with a small token_budget (e.g. 768) that
# truncates skill-patch reasoning before </think><action>. Clear the cap so
# specialists match Executor (API default completion limit).
_CLEAR_SPECIALIST_TOKEN_BUDGET = True

# GiGPO ALFWorld templates are filled into env observations (obs["text"]).
# With prompt_style="alfworld", the executor calls the LLM on that template
# directly (prompt-agent style), instead of a custom MAS system prompt.
from agent_system.environments.prompts.alfworld import (  # noqa: F401
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
)
from sage_mas.schemas import AgentSpec, Skill, SkillStatus
from sage_mas.specialist_prompt_summary import build_specialist_execution_summary

PromptStyle = Literal["alfworld", "mas"]


@dataclass(slots=True)
class LLMResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatBackend(Protocol):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int | None = None,
    ) -> LLMResult:
        ...


class OpenAIChatBackend:
    """OpenAI-compatible backend with token accounting."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        # Default OpenAI read timeout is 600s; that makes Gemini hangs look frozen.
        timeout: float | None = 120.0,
        max_retries: int = 2,
        # new-api / compatible proxies: route via request JSON "group" (e.g. default).
        group: str | None = None,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_retries = max(0, int(max_retries))
        self.group = str(group).strip() if group else None
        client_kwargs: dict = {"api_key": api_key, "base_url": base_url}
        if timeout is not None:
            client_kwargs["timeout"] = float(timeout)
        self.client = OpenAI(**client_kwargs)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int | None = None,
    ) -> LLMResult:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "n": 1,
        }
        if self.group:
            # Proxy-only field; OpenAI SDK rejects unknown kwargs unless extra_body.
            request["extra_body"] = {"group": self.group}
        # GPT-5 family endpoints commonly reject non-default temperature.
        if not self.model.lower().startswith("gpt-5"):
            request["temperature"] = self.temperature
        if max_completion_tokens is not None:
            token_parameter = (
                "max_completion_tokens"
                if self.model.lower().startswith("gpt-5")
                else "max_tokens"
            )
            request[token_parameter] = max_completion_tokens
        response = self._create_with_retries(request)
        usage = response.usage
        return LLMResult(
            content=(response.choices[0].message.content or "").strip(),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def _create_with_retries(self, request: dict):
        import logging
        import time

        from openai import APIConnectionError, APITimeoutError, RateLimitError

        attempts = self.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return self.client.chat.completions.create(**request)
            except Exception as exc:
                token_parameters = {
                    "max_completion_tokens",
                    "max_tokens",
                }
                supplied_token_parameter = next(
                    (
                        parameter
                        for parameter in token_parameters
                        if parameter in request
                    ),
                    None,
                )
                if supplied_token_parameter is not None and self._is_unsupported_parameter(
                    exc,
                    supplied_token_parameter,
                ):
                    request = dict(request)
                    request.pop(supplied_token_parameter, None)
                    return self.client.chat.completions.create(**request)

                retryable = isinstance(
                    exc,
                    (APITimeoutError, APIConnectionError, RateLimitError),
                )
                last_exc = exc
                if not retryable or attempt + 1 >= attempts:
                    raise
                sleep_s = min(8.0, 1.5 * (attempt + 1))
                logging.warning(
                    "OpenAIChatBackend retry %s/%s model=%s after %s: %s",
                    attempt + 1,
                    attempts,
                    self.model,
                    type(exc).__name__,
                    exc,
                )
                time.sleep(sleep_s)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _is_unsupported_parameter(
        exc: Exception,
        parameter: str,
    ) -> bool:
        message = str(exc).lower()
        return parameter.lower() in message and any(
            marker in message
            for marker in (
                "unsupported",
                "unknown",
                "unrecognized",
                "not permitted",
                "not allowed",
                "unexpected",
                "invalid parameter",
            )
        )


@dataclass(slots=True)
class AgentMessage:
    agent_name: str
    content: str
    token_cost: int


@dataclass(slots=True)
class RuntimeAction:
    action: str
    token_cost: int
    messages: list[AgentMessage] = field(default_factory=list)


class MASRuntime:
    """Run a skill-gated organization on ALFWorld.

    Specialists that hold ``alfworld_action`` become the environment actor when
    their assigned skills are active. The Executor remains the fallback actor
    when no specialist stage is active. Agents without ``alfworld_action`` may
    still advise the chosen actor.
    """

    def __init__(
        self,
        agents: list[AgentSpec],
        skills: list[Skill],
        backend: ChatBackend,
        injected_skills: list[Skill] | None = None,
        max_advisors: int | None = None,
        max_observation_chars: int = 16000,
        prompt_style: PromptStyle = "alfworld",
        enable_action_guards: bool = False,
        short_specialist_prompts: bool = True,
        # When True: observation (+ optional advice) only; no hand-written rules.
        compact_alfworld_prompts: bool = False,
        # Controlled counterfactual/discovery evaluations and explicit online
        # credit probation may inject a learned contract into Executor.
        allow_executor_skill_injection: bool = False,
        # full: legacy rich render + optional enriched cursor every step
        # soft: short soft SOP every step (no Step-k/N cursor)
        # sparse_soft: same soft SOP, only at start/core/stall
        # hybrid_soft: short cue every step + full soft SOP at D gates
        skill_inject_mode: str = "full",
        skill_reinject_every: int = 0,
        # Prompt-based skill retrieval (ALFWorld): the evaluator passes exactly
        # the skills the Executor selected, so an empty active-assigned set
        # means "mount nothing" instead of falling back to all assigned skills.
        strict_skill_selection: bool = False,
        # Optional mid/deploy model for non-Executor agents (e.g. flash specialists
        # while Executor/teacher play stays on ``backend``).
        specialist_backend: ChatBackend | None = None,
        # Extra LLM calls when the actor reply lacks <action>...</action>.
        # Default 1: retry once when <action> tags are missing.
        action_format_retries: int = 1,
    ):
        if not agents:
            raise ValueError("MASRuntime requires at least one agent")
        if prompt_style not in {"alfworld", "mas"}:
            raise ValueError(f"Unsupported prompt_style: {prompt_style}")
        mode = str(skill_inject_mode or "full").strip().lower()
        if mode not in {"full", "soft", "sparse_soft", "hybrid_soft"}:
            raise ValueError(f"Unsupported skill_inject_mode: {skill_inject_mode}")
        self.agents = agents
        self.backend = backend
        self.specialist_backend = specialist_backend
        self.skill_by_name = {skill.skill_name: skill for skill in skills}
        self.injected_skills = injected_skills or []
        self.max_advisors = max_advisors
        self.max_observation_chars = max_observation_chars
        self.prompt_style = prompt_style
        # Kept for API compatibility; expert action rewriting is disabled.
        self.enable_action_guards = False
        self.short_specialist_prompts = short_specialist_prompts
        self.compact_alfworld_prompts = compact_alfworld_prompts
        self.allow_executor_skill_injection = bool(
            allow_executor_skill_injection
        )
        self.skill_inject_mode = mode
        self.skill_reinject_every = max(0, int(skill_reinject_every))
        self.strict_skill_selection = bool(strict_skill_selection)
        self.action_format_retries = max(0, int(action_format_retries))
        self.executor = self._select_executor(agents)
        self.advisors = [
            agent for agent in agents if agent.agent_id != self.executor.agent_id
        ]
        self._ensure_specialist_token_budgets()

    def _ensure_specialist_token_budgets(self) -> None:
        """Drop specialist max-token caps (same as Executor: unlimited)."""
        if not _CLEAR_SPECIALIST_TOKEN_BUDGET:
            return
        for agent in self.agents:
            if agent.agent_id == self.executor.agent_id:
                continue
            agent.token_budget = None

    @staticmethod
    def _has_action_tag(content: str) -> bool:
        return bool(_ACTION_TAG_RE.search(str(content or "")))

    def _backend_for(self, agent: AgentSpec) -> ChatBackend:
        """Route specialists to ``specialist_backend`` when configured."""
        if (
            self.specialist_backend is not None
            and agent.agent_id != self.executor.agent_id
        ):
            return self.specialist_backend
        return self.backend

    def act(
        self,
        observation: str,
        injected_skills: list[Skill] | None = None,
        active_assigned_skill_names: set[str] | None = None,
        preferred_actor_name: str | None = None,
        task: str | None = None,
        task_family: str | None = None,
        history_steps: list | None = None,
        gamefile: str | None = None,
    ) -> RuntimeAction:
        observation = observation[-self.max_observation_chars :]
        actor = self._select_step_actor(
            active_assigned_skill_names,
            preferred_actor_name=preferred_actor_name,
        )
        messages = []
        # Specialists on alfworld path skip advisors to avoid noisy long prompts.
        # The GiGPO-compatible path stays a single-actor prompt. Specialists
        # differentiate through their learned Skill contract after dispatch,
        # not through an extra advisor call.
        ask_advisors = self.prompt_style != "alfworld"
        if ask_advisors:
            for advisor in self._eligible_advisors(
                active_assigned_skill_names,
                actor=actor,
            ):
                result = self._complete(
                    advisor,
                    self._advisor_system_prompt(
                        advisor,
                        active_assigned_skill_names,
                        actor_name=actor.name,
                    ),
                    observation,
                )
                messages.append(
                    AgentMessage(
                        agent_name=advisor.name,
                        content=result.content,
                        token_cost=result.total_tokens,
                    )
                )

        if self.prompt_style == "alfworld":
            # Same empty-system path for Executor and specialists.
            system_prompt = ""
            user_prompt = self._alfworld_user_prompt(
                observation,
                messages,
                injected_skills=injected_skills,
                active_assigned_skill_names=active_assigned_skill_names,
                actor=actor,
                task_family=task_family,
                task=task,
                history_steps=history_steps,
                gamefile=gamefile,
            )
        else:
            system_prompt = self._actor_system_prompt(
                actor,
                injected_skills,
                active_assigned_skill_names,
            )
            user_prompt = self._executor_user_prompt(observation, messages)

        actor_result = self._complete(actor, system_prompt, user_prompt)
        content = actor_result.content
        token_cost = actor_result.total_tokens
        retry_prompt = user_prompt
        for attempt in range(self.action_format_retries):
            if self._has_action_tag(content):
                break
            logging.warning(
                "SAGE-MAS actor reply missing <action> tag; "
                "retry %s/%s agent=%s",
                attempt + 1,
                self.action_format_retries,
                actor.name,
            )
            retry_prompt = f"{retry_prompt}{_ACTION_FORMAT_RETRY_SUFFIX}"
            actor_result = self._complete(actor, system_prompt, retry_prompt)
            content = actor_result.content
            token_cost += actor_result.total_tokens

        messages.append(
            AgentMessage(
                agent_name=actor.name,
                content=content,
                token_cost=token_cost,
            )
        )
        return RuntimeAction(
            action=content,
            token_cost=sum(message.token_cost for message in messages),
            messages=messages,
        )

    def _complete(
        self,
        agent: AgentSpec,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResult:
        backend = self._backend_for(agent)
        parameters = inspect.signature(backend.complete).parameters
        if "max_completion_tokens" in parameters:
            try:
                return backend.complete(
                    system_prompt,
                    user_prompt,
                    max_completion_tokens=agent.token_budget,
                )
            except TypeError as exc:
                if "max_completion_tokens" not in str(exc):
                    raise
                return backend.complete(
                    system_prompt,
                    user_prompt,
                )
        return backend.complete(system_prompt, user_prompt)

    def _advisor_system_prompt(
        self,
        agent: AgentSpec,
        active_assigned_skill_names: set[str] | None = None,
        *,
        actor_name: str | None = None,
    ) -> str:
        skills = self._resolve_agent_skills(
            agent,
            active_assigned_skill_names,
        )
        target = actor_name or self.executor.name
        role_spec = agent.role_specification or (
            f"You are {agent.name}, the {agent.role} in a multi-agent team."
        )
        boundary = agent.responsibility_boundary or (
            f"Advise {target} only within assigned skills; do not emit environment actions."
        )
        input_protocol = agent.input_protocol or (
            "Input: current observation and assigned skill protocols."
        )
        output_protocol = agent.output_protocol or (
            f"Output: concise advice to {target}. "
            "Do not emit <action> tags."
        )
        # Advise-only agents should never claim environment ownership.
        if self._is_stage_actor(agent):
            # Should not reach here for stage actors; defensive text.
            output_protocol = (
                f"Output: brief notes for {target}. Do not emit <action> tags."
            )
        tools = ", ".join(agent.tool_permissions) or "none listed"
        edges = ", ".join(agent.communication_edges) or target
        budgets = []
        if agent.token_budget is not None:
            budgets.append(f"token_budget={agent.token_budget}")
        if agent.turn_budget is not None:
            budgets.append(f"turn_budget={agent.turn_budget} (be selective)")
        budget_line = (
            f"Budgets: {', '.join(budgets)}\n" if budgets else ""
        )
        return (
            f"{role_spec}\n"
            f"Role: {agent.role}\n"
            f"Responsibility boundary: {boundary}\n"
            f"Responsibilities:\n{self._bullets(agent.responsibilities)}\n"
            f"Activation condition: "
            f"{agent.activation_condition or 'When your expertise is relevant.'}\n"
            f"Tool permissions (recommend only these verbs): {tools}\n"
            f"Communication edges (send advice only to): {edges}\n"
            f"{budget_line}"
            f"Input protocol: {input_protocol}\n"
            f"Output protocol: {output_protocol}\n"
            f"Assigned skill protocols:\n{self._render_skills(skills)}\n"
            "Do not claim success without observation evidence."
        )

    def _actor_system_prompt(
        self,
        actor: AgentSpec,
        injected_skills: list[Skill] | None = None,
        active_assigned_skill_names: set[str] | None = None,
    ) -> str:
        active_injected_skills = (
            self.injected_skills
            if injected_skills is None
            else injected_skills
        )
        skills = self._active_runtime_skills(
            active_assigned_skill_names,
            active_injected_skills,
            actor=actor,
        )
        role_spec = actor.role_specification or (
            f"You are {actor.name}, the {actor.role}."
        )
        if actor.agent_id == self.executor.agent_id:
            boundary = actor.responsibility_boundary or (
                "You are the only agent allowed to act as the fallback "
                "environment actor when no specialist stage is active."
            )
        else:
            boundary = actor.responsibility_boundary or (
                "You are the active stage owner for your assigned skills and "
                "must emit the environment action this turn."
            )
        input_protocol = actor.input_protocol or (
            "Input: observation, active skill protocols, and optional team advice."
        )
        output_protocol = actor.output_protocol or (
            "Return exactly one decision: "
            "<think>brief reasoning</think><action>one admissible action</action>."
        )
        tools = ", ".join(actor.tool_permissions) or "alfworld_action"
        # Specialists: short system prompt. Verbose role/contract text caused
        # weaker mini execution than Executor+skill on the same protocols.
        if actor.agent_id != self.executor.agent_id:
            return (
                f"{role_spec}\n"
                f"Active skill protocols:\n{self._render_skills(skills)}\n"
                "Bind protocol placeholders to concrete entities from the "
                "observation and admissible actions. "
                "Return exactly one decision: "
                "<think>brief reasoning</think>"
                "<action>one admissible action</action>."
            )
        return (
            f"{role_spec}\n"
            f"Responsibility boundary: {boundary}\n"
            f"Responsibilities:\n{self._bullets(actor.responsibilities)}\n"
            f"Tool permissions: {tools}\n"
            f"Input protocol: {input_protocol}\n"
            f"Output protocol: {output_protocol}\n"
            f"Active skill protocols:\n{self._render_skills(skills)}\n"
            "Follow an active skill protocol when its precondition is satisfied. "
            "Emit exactly one admissible environment action this turn."
        )

    def _executor_system_prompt(
        self,
        injected_skills: list[Skill] | None = None,
        active_assigned_skill_names: set[str] | None = None,
    ) -> str:
        return self._actor_system_prompt(
            self.executor,
            injected_skills,
            active_assigned_skill_names,
        )

    @staticmethod
    def _executor_user_prompt(observation: str, messages: list[AgentMessage]) -> str:
        if not messages:
            return observation
        advice = "\n".join(
            f"[{message.agent_name} advice]\n{message.content}" for message in messages
        )
        return f"{observation}\n\nTeam advice:\n{advice}"

    def _alfworld_user_prompt(
        self,
        observation: str,
        messages: list[AgentMessage],
        *,
        injected_skills: list[Skill] | None = None,
        active_assigned_skill_names: set[str] | None = None,
        actor: AgentSpec | None = None,
        task_family: str | None = None,
        task: str | None = None,
        history_steps: list | None = None,
        gamefile: str | None = None,
    ) -> str:
        """GiGPO observation plus learned Skill contracts when enabled."""
        active_actor = actor or self.executor
        parts = [observation]
        is_executor = active_actor.agent_id == self.executor.agent_id
        if is_executor:
            if not self.allow_executor_skill_injection:
                # Preserve the original GiGPO observation prompt. Initial
                # Executor trajectories therefore remain directly comparable.
                return observation
            assigned = [
                skill
                for skill in self._resolve_agent_skills(active_actor, None)
                if skill.status == SkillStatus.VERIFIED
            ]
            if self.strict_skill_selection:
                # Prompt-based retrieval: mount exactly the selected names.
                # An empty set means the Executor chose no skill this step.
                if active_assigned_skill_names is not None:
                    assigned = [
                        skill
                        for skill in assigned
                        if skill.skill_name in active_assigned_skill_names
                    ]
            elif active_assigned_skill_names:
                # Legacy rule-gated path: prefer step-active assigned names,
                # but keep verified assigned skills available for fallback.
                gated = [
                    skill
                    for skill in assigned
                    if skill.skill_name in active_assigned_skill_names
                ]
                if gated:
                    assigned = gated
            experimental = list(injected_skills or [])
            skills = self._unique_skills(assigned + experimental)
            if skills:
                if self.skill_inject_mode in {
                    "soft",
                    "sparse_soft",
                    "hybrid_soft",
                }:
                    from sage_mas.skill_inject_sparse_soft import (
                        render_hybrid_soft_block,
                        render_soft_skill_block,
                        should_attach_sparse_soft,
                    )

                    if self.skill_inject_mode == "hybrid_soft":
                        soft = render_hybrid_soft_block(
                            skills,
                            history_steps=history_steps,
                            reinject_every=self.skill_reinject_every,
                        )
                        if soft:
                            parts.append(soft)
                    else:
                        if self.skill_inject_mode == "soft":
                            attach = True
                        else:
                            attach, _reason = should_attach_sparse_soft(
                                history_steps=history_steps,
                                reinject_every=self.skill_reinject_every,
                            )
                        if attach:
                            soft = render_soft_skill_block(skills)
                            if soft:
                                parts.append(soft)
                else:
                    rendered = self._render_skills_contextual(
                        skills,
                        observation=observation,
                        task=task,
                        gamefile=gamefile,
                        task_family=task_family,
                        history_steps=history_steps,
                    )
                    if rendered:
                        parts.append(
                            "Active skill patch (follow concrete steps; "
                            "do not invent entities):\n"
                            f"{rendered}"
                        )
                    guidance = self._executable_step_guidance(
                        skills,
                        observation=observation,
                        history_steps=history_steps,
                        task=task,
                    )
                    if guidance:
                        parts.append(guidance)
            return "\n\n".join(parts)

        # A dispatched specialist owns the whole episode. Give it all verified
        # assigned contracts even before a step-level precondition activates.
        skills = [
            skill
            for skill in self._resolve_agent_skills(active_actor, None)
            if skill.status == SkillStatus.VERIFIED
        ]
        if skills:
            rendered = self._render_skills_contextual(
                skills,
                observation=observation,
                task=task,
                gamefile=gamefile,
                task_family=task_family,
                history_steps=history_steps,
            )
            parts.append(
                "Complete the stated environment task end-to-end with "
                "admissible actions.\n\n"
                "Active skill patch:\n"
                f"{rendered or self._render_skills(skills)}"
            )
            guidance = self._executable_step_guidance(
                skills,
                observation=observation,
                history_steps=history_steps,
                task=task,
            )
            if guidance:
                parts.append(guidance)
            parts.append(
                self._specialist_execution_discipline(
                    skills,
                    observation=observation,
                    task=task,
                    task_family=task_family,
                    history_steps=history_steps,
                )
            )
        if messages and not self.compact_alfworld_prompts:
            advice = "\n".join(
                f"[{message.agent_name} advice]\n{message.content}"
                for message in messages
            )
            parts.append(f"Team advice:\n{advice}")
        return "\n\n".join(parts)

    def _render_skills_contextual(
        self,
        skills: list[Skill],
        *,
        observation: str,
        task: str | None,
        gamefile: str | None,
        task_family: str | None,
        history_steps: list | None,
    ) -> str:
        """Prefer slot-filled student patches; fall back to static render."""
        from sage_mas.student_patch_runtime import (
            render_skills_bound,
            student_patch_of,
        )

        patched = [skill for skill in skills if student_patch_of(skill) is not None]
        plain = [skill for skill in skills if student_patch_of(skill) is None]
        blocks: list[str] = []
        if patched:
            bound = render_skills_bound(
                patched,
                observation=observation,
                task=str(task or ""),
                gamefile=str(gamefile or ""),
                task_family=task_family,
                history_steps=history_steps,
            )
            if bound:
                blocks.append(bound)
        if plain:
            blocks.append(self._render_skills(plain))
        return "\n\n".join(block for block in blocks if block.strip())

    def _executable_step_guidance(
        self,
        skills: list[Skill],
        *,
        observation: str,
        history_steps: list | None,
        task: str | None = None,
    ) -> str:
        from sage_mas.executable_protocol import render_current_step_guidance

        return render_current_step_guidance(
            skills,
            observation=observation,
            history_steps=history_steps,
            task=task,
        )

    def _specialist_execution_discipline(
        self,
        skills: list[Skill],
        *,
        observation: str,
        task: str | None,
        task_family: str | None,
        history_steps: list | None,
    ) -> str:
        """Prompt specialists as full-task actors, not skill-name personas."""
        lines = [
            "Specialist execution discipline:",
            "- Use the assigned verified skill protocol as the execution source.",
            "- Bind protocol placeholders to concrete entities from the task,",
            "  observation, and admissible actions.",
            "- Keep a mental visited list from the recent observations.",
            "- Avoid repeating recent invalid or no-progress actions.",
            "- Keep <think> short (a few sentences). Always finish with a "
            "closed <action>...</action> in the same reply—do not stop mid-thought.",
            "- Emit exactly one admissible action in <action>...</action>.",
        ]
        checklist = self._learned_protocol_checklist(skills)
        if checklist:
            lines.append("Learned protocol checklist:")
            lines.extend(checklist)
        if task:
            lines.append(f"Task text: {task}")
        if task_family:
            lines.append(f"Task family: {task_family}")
        lines.append(
            build_specialist_execution_summary(
                task=task,
                task_family=task_family,
                observation_prompt=observation,
                history_steps=history_steps,
            )
        )
        recent = self._recent_action_memory(history_steps)
        if recent:
            lines.append("Recent action memory:")
            lines.extend(recent)
        return "\n".join(lines)

    @staticmethod
    def _learned_protocol_checklist(skills: list[Skill]) -> list[str]:
        rows: list[str] = []
        for skill in skills[:2]:
            stages = skill.metadata.get("protocol_stages") or []
            if stages:
                rows.append(
                    f"- {skill.skill_name}: "
                    + " -> ".join(str(stage) for stage in stages[:8])
                )
                continue
            protocol = [str(step) for step in (skill.action_protocol or []) if step]
            if protocol:
                rows.append(
                    f"- {skill.skill_name}: "
                    + " -> ".join(protocol[:8])
                )
        return rows

    @staticmethod
    def _recent_action_memory(history_steps: list | None) -> list[str]:
        if not history_steps:
            return []
        rows: list[str] = []
        for step in list(history_steps)[-8:]:
            if isinstance(step, dict):
                action = step.get("action")
                observation = (
                    step.get("observation")
                    or step.get("observation_after")
                    or step.get("obs")
                )
            else:
                action = getattr(step, "action", None)
                observation = (
                    getattr(step, "observation", None)
                    or getattr(step, "observation_after", None)
                    or getattr(step, "obs", None)
                )
            action_text = " ".join(str(action or "").split())
            observation_text = " ".join(str(observation or "").split())
            if not action_text and not observation_text:
                continue
            if len(observation_text) > 120:
                observation_text = observation_text[:117] + "..."
            rows.append(f"- {action_text or '(no action)'} -> {observation_text}")
        return rows

    @staticmethod
    def _unique_skills(skills: list[Skill]) -> list[Skill]:
        unique: list[Skill] = []
        seen: set[str] = set()
        for skill in skills:
            if skill.skill_name in seen:
                continue
            seen.add(skill.skill_name)
            unique.append(skill)
        return unique

    def _active_runtime_skills(
        self,
        active_assigned_skill_names: set[str] | None,
        injected_skills: list[Skill],
        *,
        actor: AgentSpec | None = None,
    ) -> list[Skill]:
        active_actor = actor or self.executor
        combined = (
            self._resolve_active_assigned_skills(active_assigned_skill_names)
            + self._resolve_agent_skills(
                active_actor,
                active_assigned_skill_names,
            )
            + list(injected_skills)
        )
        unique: list[Skill] = []
        seen: set[str] = set()
        for skill in combined:
            if skill.skill_name in seen:
                continue
            seen.add(skill.skill_name)
            unique.append(skill)
        return unique

    def _resolve_agent_skills(
        self,
        agent: AgentSpec,
        active_skill_names: set[str] | None = None,
    ) -> list[Skill]:
        return [
            self.skill_by_name[name]
            for name in agent.assigned_skills
            if name in self.skill_by_name
            and (
                active_skill_names is None
                or name in active_skill_names
            )
        ]

    def _resolve_active_assigned_skills(
        self,
        active_skill_names: set[str] | None,
    ) -> list[Skill]:
        if active_skill_names is None:
            return []
        return [
            self.skill_by_name[name]
            for name in sorted(active_skill_names)
            if name in self.skill_by_name
        ]

    def _eligible_advisors(
        self,
        active_assigned_skill_names: set[str] | None,
        *,
        actor: AgentSpec | None = None,
    ) -> list[AgentSpec]:
        active_actor = actor or self.executor
        advisors = [
            advisor
            for advisor in self.advisors
            if advisor.agent_id != active_actor.agent_id
            and not self._is_stage_actor(advisor)
            and not self._is_dormant(advisor)
            and not bool(
                (advisor.shadow_evaluation_record or {}).get(
                    "dispatch_only",
                    False,
                )
            )
            and self._advisor_reports_to_executor(advisor)
        ]
        if active_assigned_skill_names is not None:
            advisors = [
                advisor
                for advisor in advisors
                if advisor.assigned_skills
                and set(advisor.assigned_skills) & active_assigned_skill_names
            ]
            advisors.sort(
                key=lambda advisor: len(
                    set(advisor.assigned_skills) & active_assigned_skill_names
                ),
                reverse=True,
            )
        if self.max_advisors is not None:
            advisors = advisors[: self.max_advisors]
        return advisors

    def _select_step_actor(
        self,
        active_assigned_skill_names: set[str] | None,
        *,
        preferred_actor_name: str | None = None,
    ) -> AgentSpec:
        """Prefer Executor-dispatched owner; else skill-gated specialist."""
        if preferred_actor_name:
            preferred = next(
                (
                    agent
                    for agent in self.agents
                    if agent.name == preferred_actor_name
                ),
                None,
            )
            if preferred is not None and self._may_take_dispatched_episode(
                preferred
            ):
                return preferred
        if active_assigned_skill_names is None:
            # Ungated mode: do not let every specialist steal every turn.
            return self.executor
        if not active_assigned_skill_names:
            return self.executor
        ranked: list[tuple[int, str, AgentSpec]] = []
        for agent in self.agents:
            if not self._is_stage_actor(agent):
                continue
            overlap = set(agent.assigned_skills) & active_assigned_skill_names
            if not overlap:
                continue
            ranked.append((len(overlap), agent.name, agent))
        if not ranked:
            return self.executor
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][2]

    @staticmethod
    def _acting_status(agent: AgentSpec) -> str:
        record = agent.shadow_evaluation_record or {}
        status = str(record.get("acting_status") or "").strip().lower()
        # Legacy agents without an onboarding gate act as accepted.
        return status or "accepted"

    @classmethod
    def _is_demoted(cls, agent: AgentSpec) -> bool:
        return cls._acting_status(agent) == "demoted"

    @classmethod
    def _is_dormant(cls, agent: AgentSpec) -> bool:
        return cls._acting_status(agent) == "dormant"

    @classmethod
    def _may_sticky_act(cls, agent: AgentSpec) -> bool:
        """Point 1: probation may only act while trial budget remains."""
        status = cls._acting_status(agent)
        if status == "accepted":
            return True
        if status in {"demoted", "dormant"}:
            return False
        if status == "probation":
            record = agent.shadow_evaluation_record or {}
            return int(record.get("trial_games_remaining") or 0) > 0
        return True

    @classmethod
    def _has_alfworld_action(cls, agent: AgentSpec) -> bool:
        """True when the agent may emit environment actions.

        Accepts legacy ``alfworld_action`` plus WebShop ``webshop_action`` /
        generic ``env_action`` so domain ports do not need fake ALFWorld tools.
        """
        permissions = {
            permission.lower() for permission in agent.tool_permissions
        }
        return bool(
            permissions
            & {"alfworld_action", "webshop_action", "env_action"}
        )

    @classmethod
    def _may_take_dispatched_episode(cls, agent: AgentSpec) -> bool:
        """Executor may hand an episode to a matching dormant specialist."""
        if cls._is_demoted(agent):
            return False
        if not cls._has_alfworld_action(agent) and not (
            "executor" in f"{agent.name} {agent.role}".lower()
        ):
            return False
        status = cls._acting_status(agent)
        if status == "dormant":
            return True
        if status == "accepted":
            return True
        if status == "probation":
            return cls._may_sticky_act(agent)
        return True

    @classmethod
    def _is_stage_actor(cls, agent: AgentSpec) -> bool:
        """Specialists that may emit environment actions when activated."""
        if cls._is_demoted(agent) or cls._is_dormant(agent):
            return False
        if bool(
            (agent.shadow_evaluation_record or {}).get(
                "dispatch_only",
                False,
            )
        ):
            return False
        if not cls._may_sticky_act(agent):
            return False
        if not cls._has_alfworld_action(agent):
            return False
        # Executor is the fallback actor, not a specialist stage owner.
        return "executor" not in f"{agent.name} {agent.role}".lower()

    def _advisor_reports_to_executor(self, advisor: AgentSpec) -> bool:
        edges = list(advisor.communication_edges or [])
        if not edges:
            # Legacy agents without explicit edges still advise the executor.
            return True
        targets = {edge.strip().lower() for edge in edges}
        executor_name = self.executor.name.strip().lower()
        return executor_name in targets or "executor" in targets

    @staticmethod
    def _select_executor(agents: list[AgentSpec]) -> AgentSpec:
        for agent in agents:
            if "executor" in f"{agent.name} {agent.role}".lower():
                return agent
        return agents[0]

    @staticmethod
    def _render_skills(skills: list[Skill]) -> str:
        if not skills:
            return "- No additional skills."
        rendered = []
        seen = set()
        for skill in skills:
            if skill.skill_name in seen:
                continue
            seen.add(skill.skill_name)
            md = skill.metadata or {}
            # Prefer short student patches compiled for mid-strong actors.
            student_patch = md.get("student_patch")
            if isinstance(student_patch, dict) and student_patch.get("schema_version") == "student_patch_v2":
                # Unbound preview must not be injected; contextual binder fills slots.
                rendered.append(
                    f"- {skill.skill_name}\n"
                    "  [student_patch_v2] waiting for runtime slot binding "
                    "(concrete actions only; no placeholders)."
                )
                continue
            patch_text = ""
            if isinstance(student_patch, dict):
                patch_text = str(student_patch.get("inject_text") or "").strip()
            if not patch_text:
                patch_text = str(md.get("inject_render_text") or "").strip()
            if patch_text:
                rendered.append(f"- {skill.skill_name}\n{patch_text}")
                continue
            protocol = "; ".join(skill.action_protocol)
            from sage_mas.rich_skill_context import render_rich_skill_block

            block = render_rich_skill_block(
                skill,
                include_header=True,
                include_contract=True,
                include_demos=True,
                include_cues=True,
                include_anti=True,
                max_demos=3,
            )
            if not block.strip():
                block = "\n".join(
                    [
                        f"- {skill.skill_name}",
                        f"  Precondition: {skill.precondition}",
                        f"  Protocol: {protocol}",
                        f"  Expected effect: "
                        f"{skill.expected_effect or 'Improve task execution.'}",
                    ]
                )
            rendered.append(block)
        return "\n".join(rendered)

    @staticmethod
    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- Follow the task goal."
