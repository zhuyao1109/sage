"""τ² HalfDuplex agent with Executor dispatch + specialist skill ownership.

Executor owns the episode and may delegate successive bounded turns.
Forced-primary mode remains available for paired admission experiments.

Registered at runtime by the sage_tau2 online runner.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Generic, List, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel, Field

from sage_tau2.executor_dispatch import (
    AgentAssignment,
    ExecutorDispatchConfig,
    ExecutorDispatcher,
    dispatch_config_from_mapping,
    record_primary_dispatch,
)
from sage_tau2.executor_tool_guard import (
    assistant_has_payload,
    ensure_assistant_payload,
    guard_assistant_message,
    with_tool_boundary_policy,
)
from sage_tau2.onboarding import append_dispatch_event
from sage_tau2.injection import (
    append_skills_to_user_prompt,
    candidate_pool_for_domain,
    format_skills_for_prompt,
    render_skill_cards,
    offer_skills_for_domain,
)
from sage_tau2.skill_retrieval import SkillIndex
from sage_tau2.organization import (
    EXECUTOR_NAME,
    Organization,
    format_organization_for_prompt,
)
from sage_tau2.pro_steps import (
    action_from_assistant_message,
    build_live_window_prompt,
    format_available_tools,
    parse_think_and_visible,
)
from sage_tau2.projection import validate_think_action
from sage_tau2.prompts import (
    SYSTEM_PROMPT,
    THINK_ACTION_RETRY_NUDGE,
    instruction_for_agent,
)
from sage_tau2.schemas import AgentSpec, SkillStatus, Tau2Skill
from sage_tau2.serialization import to_primitive
from sage_tau2.skill_bank import Tau2SkillBank

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import (
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate


class SageTau2State(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    executions: list[dict] = Field(default_factory=list)
    current_execution_id: str = ""


SageTau2StateType = TypeVar("SageTau2StateType", bound=SageTau2State)


class SageTau2Agent(
    LLMConfigMixin,
    HalfDuplexAgent[SageTau2StateType],
    Generic[SageTau2StateType],
):
    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        *,
        skills: list[Tau2Skill] | None = None,
        organization_block: str = "",
        role_block: str = "",
        agent_role_name: str = EXECUTOR_NAME,
        assignment: AgentAssignment | None = None,
        domain: str | None = None,
        task_description: str | None = None,
        # Step-level retrieval (ALFWorld-style per-step skill activation).
        candidate_pool: list[Tau2Skill] | None = None,
        task_id: str = "",
        max_active_skills: int = 2,
        allow_provisional: bool = True,
        delegates: list[AgentSpec] | None = None,
        delegate_skills: list[Tau2Skill] | None = None,
        max_delegated_turns: int = 8,
        select_skills: bool = True,
    ):
        super().__init__(
            tools=tools,
            domain_policy=with_tool_boundary_policy(
                domain_policy,
                tools=tools,
                domain=domain or os.environ.get("SAGE_TAU2_DOMAIN"),
            ),
            llm=llm,
            llm_args=llm_args,
        )
        self.delegates = list(delegates or [])
        self.delegate_skills = list(delegate_skills or [])
        self.max_delegated_turns = max(0, int(max_delegated_turns))
        self.select_skills = select_skills
        self._turn_skills: list[Tau2Skill] | None = None
        self._auxiliary_calls: list[dict] = []
        self.skills = list(skills or [])
        self.injected_skill_ids = [s.skill_id for s in self.skills]
        self.organization_block = organization_block or ""
        self.role_block = role_block or ""
        self.agent_role_name = agent_role_name
        self.assignment = assignment
        self.domain = domain
        self.task_description = (task_description or "").strip() or (
            f"Handle a {domain or 'customer-service'} episode end-to-end"
        )
        self.available_tools_text = format_available_tools(tools)
        # When True, LLM sees bounded PRO window (not full chat transcript).
        self.use_window_prompt = str(
            os.environ.get("SAGE_TAU2_USE_WINDOW_PROMPT", "1")
        ).lower() not in {"0", "false", "no"}
        self.history_window = int(os.environ.get("SAGE_TAU2_HISTORY_WINDOW", "10"))
        # Default off: forced <think> retries burned latency without improving
        # retail/airline write rates; set SAGE_TAU2_REQUIRE_THINK_ACTION=1 to re-enable.
        self.require_think_action = str(
            os.environ.get("SAGE_TAU2_REQUIRE_THINK_ACTION", "0")
        ).lower() not in {"0", "false", "no"}
        self.think_action_retries = int(
            os.environ.get("SAGE_TAU2_THINK_ACTION_RETRIES", "3")
        )
        # On-demand retrieval: when a SkillIndex is provided, each step the
        # LLM generates a natural-language query from the current dialogue
        # state, and the query is used to retrieve top-K skills from the
        # bank via BM25. Retrieved skills' full protocols are
        # injected into the prompt. This is true retrieval — no hardcoded
        # routing rules, no catalog browsing.
        self.candidate_pool = list(candidate_pool or [])
        self.task_id = task_id
        self.max_active_skills = max(0, int(max_active_skills))
        self.allow_provisional_skills = allow_provisional
        # Build the retrieval index from the candidate pool.
        self.skill_index: SkillIndex | None = (
            SkillIndex(self.candidate_pool) if self.candidate_pool else None
        )
        # Skills retrieved at the current step (full protocols in prompt).
        self.retrieved_skills: list[Tau2Skill] = []
        # Snapshot for journaling / credit attribution.
        self._last_active_skills: list[Tau2Skill] = list(self.skills)

    @property
    def skills_user_block(self) -> str:
        """Skill patch appended to the user window.

        When on-demand retrieval is enabled (``skill_index`` non-empty),
        this returns the full protocols of skills **retrieved** at the
        current step — the LLM generated a query, the index returned top-K
        matches, and their protocols are injected here.

        Specialists keep assigned-skill ownership (no retrieval).
        """
        if self._turn_skills is not None:
            return format_skills_for_prompt(self._turn_skills)
        is_specialist = self.agent_role_name != EXECUTOR_NAME
        # Specialists keep assigned-skill ownership (no retrieval).
        if is_specialist or self.skill_index is None:
            return format_skills_for_prompt(self.skills, specialist=is_specialist)
        # Executor with on-demand retrieval: inject retrieved protocols.
        return format_skills_for_prompt(self.retrieved_skills, specialist=False)

    def _generate_retrieval_query(self, state: SageTau2StateType) -> str:
        """Generate a natural-language retrieval query from the current dialogue state.

        Makes a lightweight LLM call with a short prompt (not the full window)
        asking: "Given this conversation, what skill/protocol would help?"
        The returned query is used for BM25 retrieval against the skill index.
        """
        dicts = [self._message_to_dict(m) for m in state.messages]
        # Build a compact context: last user message + recent tool results.
        recent = dicts[-8:] if len(dicts) > 8 else dicts
        context_parts: list[str] = []
        for d in recent:
            role = str(d.get("role") or "").lower()
            content = str(d.get("content") or "").strip()
            if role == "user" and content:
                context_parts.append(f"User: {content[:300]}")
            elif role == "tool":
                name = str(d.get("name") or "tool")
                context_parts.append(f"Tool[{name}]: {content[:200]}")
            elif role == "assistant":
                tool_calls = d.get("tool_calls") or []
                if tool_calls:
                    names = [str(tc.get("name") or "?") for tc in tool_calls if isinstance(tc, dict)]
                    context_parts.append(f"Agent called: {', '.join(names)}")
                elif content:
                    context_parts.append(f"Agent: {content[:200]}")
        from sage_tau2.contracts import observed_ledger
        from sage_tau2.execution import execution_prompt
        context = "\n".join(context_parts[-6:]) + "\n" + observed_ledger(dicts) + "\n" + execution_prompt(state.executions)
        if not context.strip():
            return ""
        query_prompt = (
            f"You are a skill retrieval assistant for a {self.domain or 'customer service'} agent. "
            "Given the current conversation state, output a short query (1-2 sentences) "
            "describing what kind of protocol or skill would help the agent handle this "
            "request. Focus on: the user's problem, what tools need to be called, what "
            "information needs to be looked up. Output ONLY the query, no explanation.\n\n"
            f"Conversation:\n{context}\n\nQuery:"
        )
        try:
            from tau2.utils.llm_utils import generate as _generate
            raw = _generate(
                model=self.llm,
                tools=[],
                messages=[UserMessage(role="user", content=query_prompt)],
                call_name="sage_tau2_skill_query",
                **{k: v for k, v in self.llm_args.items() if k in ("temperature", "max_tokens", "top_p")},
            )
            self._record_auxiliary_call("retrieval", raw)
            query = ""
            if hasattr(raw, "content") and raw.content:
                query = str(raw.content).strip()
            elif isinstance(raw, dict):
                query = str(raw.get("content") or "").strip()
            if len(query) > 500:
                query = query[:500]
            return query
        except Exception as exc:
            logger.warning(f"[sage_tau2] retrieval query generation failed: {exc}")
            return ""

    def _record_auxiliary_call(self, kind: str, response: Any) -> None:
        self._auxiliary_calls.append({"kind": kind, "cost": getattr(response, "cost", None),
                                      "usage": getattr(response, "usage", None)})

    @property
    def system_prompt(self) -> str:
        is_specialist = self.agent_role_name != EXECUTOR_NAME
        instruction = instruction_for_agent(
            domain=self.domain or os.environ.get("SAGE_TAU2_DOMAIN"),
            specialist=is_specialist,
        )
        role_line = f"Active role name: {self.agent_role_name}.\n"
        return SYSTEM_PROMPT.format(
            agent_instruction=role_line + instruction,
            domain_policy=self.domain_policy,
            role_block=self.role_block,
            organization_block=self.organization_block if not is_specialist else "",
        )

    def _llm_messages_full_history(
        self, state: SageTau2StateType
    ) -> tuple[list[Any], str | None, dict[str, Any]]:
        """Fallback path: system + full chat (skills stay on a trailing user note)."""
        from sage_tau2.execution import execution_prompt
        messages = list(state.system_messages) + list(state.messages)
        if state.executions:
            messages.append(UserMessage(role="user", content=execution_prompt(state.executions)))
        skills = self.skills_user_block
        if skills:
            messages = list(messages) + [
                UserMessage(role="user", content=skills)
            ]
        return messages, skills or None, {"skills_on_user": bool(skills)}

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> SageTau2StateType:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Invalid agent message history"
        )
        return SageTau2State(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )

    @staticmethod
    def _message_to_dict(msg: Any) -> dict[str, Any]:
        if isinstance(msg, dict):
            return msg
        if hasattr(msg, "model_dump"):
            return msg.model_dump()
        if hasattr(msg, "dict"):
            return msg.dict()
        return {"role": getattr(msg, "role", None), "content": getattr(msg, "content", None)}

    def _llm_messages_and_window(
        self, state: SageTau2StateType
    ) -> tuple[list[Any], str | None, dict[str, Any]]:
        """Build LLM input: system + GiGPO user window (+ skill patch)."""
        if not self.use_window_prompt:
            return self._llm_messages_full_history(state)
        dicts = [self._message_to_dict(m) for m in state.messages]
        window_prompt, meta = build_live_window_prompt(
            dicts,
            history_window=self.history_window,
            task_description=self.task_description,
        )
        from sage_tau2.execution import execution_prompt
        user_content = append_skills_to_user_prompt(
            window_prompt + "\n" + execution_prompt(state.executions), self.skills_user_block
        )
        meta = dict(meta)
        meta["skills_on_user"] = bool(self.skills_user_block)
        llm_messages: list[Any] = list(state.system_messages) + [
            UserMessage(role="user", content=user_content)
        ]
        return llm_messages, user_content, meta

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: SageTau2StateType
    ) -> tuple[AssistantMessage, SageTau2StateType]:
        self._auxiliary_calls = []
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        from sage_tau2.execution import (observe, apply_feedback, choose_execution, record_action,
                                         execution_prompt, TERMINAL)
        incoming = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        observe(state.executions, [self._message_to_dict(m) for m in incoming], f"obs-{len(state.messages)}")

        # On-demand retrieval: before building the prompt, generate a
        # natural-language query from the current dialogue state and
        # retrieve top-K skills from the index via BM25.
        # This is true retrieval — the LLM generates the query key, the
        # index does similarity search, no hardcoded routing rules.
        if self.skill_index is not None:
            query = self._generate_retrieval_query(state)
            if query:
                retrieved = self.skill_index.search(
                    query, top_k=self.max_active_skills
                )
                self.retrieved_skills = retrieved
                self._last_active_skills = list(retrieved)
                logger.info(
                    f"[sage_tau2] retrieval query={query[:120]!r} "
                    f"retrieved={[s.skill_name for s in retrieved]}"
                )
            elif not self.retrieved_skills:
                # First step with no query yet: fall back to initial skills.
                self.retrieved_skills = list(self.skills)
                self._last_active_skills = list(self.skills)

        from sage_tau2.coordination import TurnPlan, parse_plan, coordination_prompt
        from sage_tau2.contracts import skill_events_from_messages
        self._turn_skills = []
        candidates = self.retrieved_skills if self.skill_index is not None else self.skills
        pool = {s.skill_id: s for s in [*self.skills, *self.candidate_pool, *self.delegate_skills]}
        ongoing = [pool[e['skill_id']] for e in state.executions if e['status'] not in TERMINAL and e['skill_id'] in pool]
        candidates = list({s.skill_id: s for s in [*ongoing, *candidates]}.values()) if self.max_active_skills else []
        cards, offered = render_skill_cards(candidates)
        events = skill_events_from_messages([self._message_to_dict(m) for m in state.messages]) or []
        delegated_turns = sum(e.get("actor") not in (None, EXECUTOR_NAME) for e in events)
        from sage_tau2.skill_resolve import resolve_assigned_skills
        delegates = [a for a in self.delegates if resolve_assigned_skills(a.assigned_skills, offered)] if delegated_turns < self.max_delegated_turns else []
        actors = [AgentSpec(name=self.agent_role_name, role="owner", responsibilities=[]), *delegates]
        plan = TurnPlan(actor=self.agent_role_name)
        if self.select_skills and (offered or delegates):
            context_messages, context, _ = self._llm_messages_and_window(state)
            if not self.use_window_prompt:
                context = json.dumps([self._message_to_dict(m) for m in context_messages], ensure_ascii=False, default=str)
            try:
                choice = generate(model=self.llm, tools=[], messages=[UserMessage(role="user", content=coordination_prompt(
                    context=self.system_prompt + "\n" + (context or ""), cards=cards,
                    agents=actors, owner=self.agent_role_name))], call_name="sage_tau2_coordinator",
                    **{k: v for k, v in self.llm_args.items() if k in ("temperature", "max_tokens", "top_p")})
                self._record_auxiliary_call("coordination", choice)
                plan = parse_plan(choice.content or "", actors={a.name for a in actors},
                                  skill_ids={sk.skill_id for sk in offered}, fallback=self.agent_role_name)
            except Exception as exc:
                logger.warning(f"[sage_tau2] coordinator fallback: {exc}")
        elif not self.select_skills:
            plan.adopted_skill_ids = [sk.skill_id for sk in offered][:self.max_active_skills]
        chosen_delegate = next((a for a in delegates if a.name == plan.actor), None)
        if chosen_delegate is not None:
            from sage_tau2.skill_resolve import resolve_assigned_skills
            owned_ids = {s.skill_id for s in resolve_assigned_skills(chosen_delegate.assigned_skills, offered)}
            plan.adopted_skill_ids = [sid for sid in plan.adopted_skill_ids if sid in owned_ids]
            if not plan.adopted_skill_ids:
                plan = TurnPlan(actor=self.agent_role_name, reason="no_available_owned_skill")
        apply_feedback(state.executions, plan.step_update)
        previous = next((e for e in state.executions if e['execution_id'] == state.current_execution_id), None)
        if plan.disposition in {'pause', 'abandon'}:
            target = next((e for e in state.executions if e['execution_id'] == plan.execution_id), previous)
            if target and target['status'] not in TERMINAL:
                target['status'] = 'paused' if plan.disposition == 'pause' else 'abandoned'
                target['return_record'] = {'status': target['status'], 'effect_verified': False, 'reason': plan.reason}
            state.current_execution_id = ''
            plan.adopted_skill_ids = []
        elif not plan.adopted_skill_ids and previous and previous['status'] not in TERMINAL:
            if previous['skill_id'] in {s.skill_id for s in offered}:
                plan.adopted_skill_ids = [previous['skill_id']]
                plan.execution_id = previous['execution_id']
                plan.actor = previous['actor'] if previous['actor'] in {a.name for a in actors} else self.agent_role_name
        if plan.execution_id and not plan.adopted_skill_ids and plan.disposition == 'continue':
            requested = next((e for e in state.executions if e['execution_id'] == plan.execution_id and e['status'] not in TERMINAL), None)
            if requested:
                plan.adopted_skill_ids = [requested['skill_id']]
        self._turn_skills = [sk for sk in offered if sk.skill_id in plan.adopted_skill_ids][:1]
        selected_actor = next((a for a in delegates if a.name == plan.actor), None)
        if selected_actor and not resolve_assigned_skills(selected_actor.assigned_skills, self._turn_skills):
            plan.actor = self.agent_role_name
        current = None
        if self._turn_skills:
            current = choose_execution(state.executions, skill=self._turn_skills[0], actor=plan.actor,
                execution_id=plan.execution_id, objective=plan.subtask)
            if current is None:
                self._turn_skills = []
            else:
                state.current_execution_id = current['execution_id']
        else:
            state.current_execution_id = ''

        llm_messages, window_prompt, window_meta = self._llm_messages_and_window(state)
        directive = (f"Active actor for this turn: {plan.actor}. Assigned subtask: {plan.subtask}. "
                     f"Expected observable result: {plan.expected_result}. "
                     "Execute only the next turn, then return control to Executor. "
                     "Do not report an expected result as an observed fact.")
        delegate = next((a for a in delegates if a.name == plan.actor), None)
        if delegate is not None:
            delegated_system = SYSTEM_PROMPT.format(
                agent_instruction=instruction_for_agent(domain=self.domain, specialist=True),
                domain_policy=self.domain_policy, role_block=_role_block_for(delegate), organization_block="")
            llm_messages[0] = SystemMessage(role="system", content=delegated_system)
        llm_messages.append(UserMessage(role="user", content=directive))
        window_prompt = (window_prompt or "") + "\n\n" + directive

        def _produce(extra_messages: list[Any] | None = None) -> AssistantMessage:
            call_messages = list(llm_messages) + list(extra_messages or [])
            raw = generate(
                model=self.llm,
                tools=self.tools,
                messages=call_messages,
                call_name="sage_tau2_agent",
                **self.llm_args,
            )
            return guard_assistant_message(raw, tools=self.tools)

        # Non-empty payload first (existing guard).
        assistant_message = ensure_assistant_payload(
            lambda: _produce(), max_attempts=3, label="sage_tau2"
        )

        # GiGPO-style: require <think> + action; regenerate with nudge on failure.
        if self.require_think_action:
            ok, reason = validate_think_action(assistant_message)
            attempts = 1
            while (
                (not ok or not assistant_has_payload(assistant_message))
                and attempts < max(1, self.think_action_retries)
            ):
                attempts += 1
                logger.warning(
                    f"[sage_tau2] think/action invalid ({reason or 'empty'}); "
                    f"retry {attempts}/{self.think_action_retries}"
                )
                assistant_message = _produce(
                    [UserMessage(role="user", content=THINK_ACTION_RETRY_NUDGE)]
                )
                if not assistant_has_payload(assistant_message):
                    reason = "empty_payload"
                    ok = False
                    continue
                ok, reason = validate_think_action(assistant_message)
            if not ok:
                logger.warning(
                    f"[sage_tau2] think/action still invalid after retries "
                    f"({reason}); accepting best-effort without synthetic think"
                )

        # Keep model <think> for trajectory dumps; never synthesize into formal
        # fields. Hide think tags from the customer-visible content — but never
        # emit an empty AssistantMessage (τ² raises infrastructure_error).
        think, visible = parse_think_and_visible(assistant_message.content)

        raw = assistant_message.raw_data
        if not isinstance(raw, dict):
            raw = {}
        else:
            raw = dict(raw)
        raw["sage_auxiliary_calls"] = list(self._auxiliary_calls)
        costs = [c["cost"] for c in self._auxiliary_calls if isinstance(c.get("cost"), (int, float))]
        if costs:
            assistant_message.cost = float(assistant_message.cost or 0) + sum(costs)
        usage = dict(assistant_message.usage or {})
        for call in self._auxiliary_calls:
            for key, value in (call.get("usage") or {}).items():
                if isinstance(value, (int, float)) and isinstance(usage.get(key, 0), (int, float)):
                    usage[key] = usage.get(key, 0) + value
        if usage:
            assistant_message.usage = usage
        from copy import deepcopy
        call_ownership = record_action(current, self._message_to_dict(assistant_message))
        raw["sage_skill_event"] = {
            "version": 2, "execution_id": current['execution_id'] if current else None,
            "call_ownership": call_ownership, "executions": deepcopy(state.executions),
            "actor": plan.actor, "owner": self.agent_role_name,
            "offered_skill_ids": [sk.skill_id for sk in offered],
            "adopted_skill_ids": [sk.skill_id for sk in self._turn_skills],
            "subtask": plan.subtask, "expected_result": plan.expected_result,
            "reason": plan.reason, "tool_calls": self._message_to_dict(assistant_message).get("tool_calls") or [],
        }
        if self.retrieved_skills:
            raw["sage_retrieved_skill_ids"] = [s.skill_id for s in self.retrieved_skills]
            raw["sage_retrieved_skill_names"] = [s.skill_name for s in self.retrieved_skills]
        if think:
            raw["sage_think"] = think
            raw["sage_think_source"] = "model"
        else:
            raw.pop("sage_think", None)
            raw["sage_think_source"] = "missing"
        if window_prompt:
            raw["sage_window_prompt"] = window_prompt
            raw["sage_prompt_source"] = "live"
        if window_meta:
            raw["sage_window_meta"] = {
                k: window_meta.get(k)
                for k in (
                    "step_count",
                    "current_step",
                    "history_length",
                )
            }
        msg_dict = self._message_to_dict(assistant_message)
        raw["sage_action"] = action_from_assistant_message(msg_dict)
        assistant_message.raw_data = raw

        has_tools = bool(getattr(assistant_message, "tool_calls", None))
        if visible and str(visible).strip():
            assistant_message.content = visible
        elif has_tools:
            # Tool-only turn: content may be empty after stripping <think>.
            assistant_message.content = None
        else:
            # Text-only turn that was entirely <think>…</think> — keep a
            # non-empty customer-visible line so the episode continues.
            assistant_message.content = (
                "Thanks for the details — let me continue helping with your request."
            )
            raw["sage_content_recovered"] = True
            assistant_message.raw_data = raw

        state.messages.append(assistant_message)
        return assistant_message, state


from sage_tau2.task_context import (
    agent_facing_task_text,
    clean_task_phrase,
    episode_routing_scope,
    resolve_episode_domain,
    task_text_and_domain as _task_text,
)

def _pop_llm_arg(kwargs: dict, key: str, default: Any = None) -> Any:
    llm_args = dict(kwargs.get("llm_args") or {})
    value = llm_args.pop(key, default)
    kwargs["llm_args"] = llm_args
    return value


def _load_bank(kwargs: dict) -> tuple[Tau2SkillBank | None, Path | None]:
    bank_path = _pop_llm_arg(kwargs, "skill_bank_path", None) or os.environ.get(
        "SAGE_TAU2_SKILL_BANK"
    )
    if not bank_path:
        return None, None
    path = Path(bank_path)
    if not path.exists():
        logger.warning(f"[sage_tau2] skill bank missing: {path}")
        return None, path
    return Tau2SkillBank(path), path


def _load_org(kwargs: dict) -> tuple[Organization | None, Path | None]:
    org_path = _pop_llm_arg(kwargs, "organization_path", None) or os.environ.get(
        "SAGE_TAU2_ORG_PATH"
    )
    if not org_path:
        return None, None
    path = Path(org_path)
    if not path.exists():
        return None, path
    return Organization.load(path), path


def _dispatch_config_from_kwargs(kwargs: dict) -> ExecutorDispatchConfig:
    raw = _pop_llm_arg(kwargs, "dispatch_config", None)
    if isinstance(raw, dict):
        return dispatch_config_from_mapping(raw)
    enabled = str(
        _pop_llm_arg(
            kwargs,
            "enable_executor_dispatch",
            os.environ.get("SAGE_TAU2_ENABLE_DISPATCH", "1"),
        )
    ).lower() not in {"0", "false", "no"}
    cfg = ExecutorDispatchConfig(enabled=enabled)
    return cfg


def _skills_for_primary(
    *,
    primary: AgentSpec,
    bank: Tau2SkillBank | None,
    max_inject: int,
    allow_provisional: bool,
    domain: str,
    same_domain_only: bool = True,
    allowed_skill_ids: set[str] | None = None,
    task_id: str = "",
    episode_scope: str = "",
    task_text: str = "",
    require_scope_match: bool = True,
) -> list[Tau2Skill]:
    """Pick skills for the sticky primary.

    Executor / generalist uses ``offer_skills_for_domain`` so multidomain banks
    do not leak retail/telecom recipes into airline (and vice versa).
    Specialists keep assigned-skill ownership only.
    """
    if bank is None:
        return []
    active = bank.active()
    if primary.name == EXECUTOR_NAME or primary.role == "generalist":
        return offer_skills_for_domain(
            bank,
            domain=domain,
            max_skills=max_inject,
            allow_provisional=allow_provisional,
            same_domain_only=same_domain_only,
            allowed_skill_ids=allowed_skill_ids,
            task_id=task_id,
            episode_scope=episode_scope,
            task_text=task_text,
            require_scope_match=require_scope_match,
        )
    # Specialist owns all assigned active skills (verified + optional provisional).
    allowed = {SkillStatus.VERIFIED}
    if allow_provisional:
        allowed.add(SkillStatus.PROVISIONAL)
    from sage_tau2.skill_resolve import resolve_assigned_skills

    out: list[Tau2Skill] = []
    for skill in resolve_assigned_skills(
        primary.assigned_skills,
        active,
        allowed_status=allowed,
    ):
        if not skill.action_protocol:
            continue
        if allowed_skill_ids is not None and skill.skill_id not in allowed_skill_ids:
            continue
        from sage_tau2.credit import skill_allowed_for_inject
        if not skill_allowed_for_inject(skill, allow_provisional=allow_provisional):
            continue
        # Extra safety: never hand a specialist a cross-domain protocol.
        if same_domain_only:
            skill_domain = str(getattr(skill, "domain", "") or "").lower()
            if skill_domain and skill_domain != str(domain or "").lower():
                continue
        out.append(skill)
    return out[:max(0, max_inject)]


def _parse_allowed_skill_ids(raw: Any) -> set[str] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (set, list, tuple)):
        out = {str(x) for x in raw if str(x).strip()}
        return out
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("["):
        import json

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            out = {str(x) for x in parsed if str(x).strip()}
            return out
    out = {part.strip() for part in text.split(",") if part.strip()}
    return out or None


def _role_block_for(agent: AgentSpec) -> str:
    if agent.name == EXECUTOR_NAME:
        return ""
    lines = ["<active_specialist>"]
    lines.append(f"name: {agent.name}")
    if agent.capability_keys:
        lines.append(f"capabilities: {', '.join(agent.capability_keys)}")
    if agent.role_specification:
        lines.append(f"role: {agent.role_specification}")
    if agent.activation_condition:
        lines.append(f"activate_when: {agent.activation_condition}")
    if agent.assigned_skills:
        lines.append(
            "assigned_skill_names: " + ", ".join(agent.assigned_skills[:12])
        )
    lines.append(
        "Execute the assigned objective using observed evidence. Return actual results and unresolved work to Executor after this turn."
    )
    lines.append("</active_specialist>")
    return "\n".join(lines)


def create_sage_tau2_agent(tools, domain_policy, **kwargs):
    """Factory used by τ² registry; receives ``task=`` from build_agent."""
    task = kwargs.pop("task", None)
    kwargs.pop("audio_native_config", None)
    kwargs.pop("audio_taps_dir", None)

    max_skills = int(
        _pop_llm_arg(
            kwargs,
            "max_inject_skills",
            os.environ.get("SAGE_TAU2_MAX_SKILLS", 2),
        )
    )
    allow_provisional = str(
        _pop_llm_arg(
            kwargs,
            "inject_provisional",
            os.environ.get("SAGE_TAU2_INJECT_PROVISIONAL", "1"),
        )
    ).lower() not in {"0", "false", "no"}
    same_domain_only = str(
        _pop_llm_arg(
            kwargs,
            "inject_same_domain_only",
            os.environ.get("SAGE_TAU2_INJECT_SAME_DOMAIN_ONLY", "1"),
        )
    ).lower() not in {"0", "false", "no"}
    require_scope_match = str(
        _pop_llm_arg(
            kwargs,
            "require_scope_match",
            os.environ.get("SAGE_TAU2_REQUIRE_SCOPE_MATCH", "1"),
        )
    ).lower() not in {"0", "false", "no"}
    allowed_skill_ids = _parse_allowed_skill_ids(
        _pop_llm_arg(
            kwargs,
            "allowed_skill_ids",
            os.environ.get("SAGE_TAU2_ALLOWED_SKILL_IDS"),
        )
    )

    bank, bank_path = _load_bank(kwargs)
    org, org_path = _load_org(kwargs)
    dispatch_cfg = _dispatch_config_from_kwargs(kwargs)
    force_primary = (
        _pop_llm_arg(kwargs, "force_primary", None)
        or os.environ.get("SAGE_TAU2_FORCE_PRIMARY")
        or ""
    )
    force_primary = str(force_primary).strip()

    # Prior-attempt verdict feedback (retry with environment signal, not
    # expert rules). Sidecar JSON keyed by task_id → feedback text.
    prior_hints_path = _pop_llm_arg(
        kwargs,
        "prior_failure_hints_path",
        os.environ.get("SAGE_TAU2_PRIOR_HINTS_PATH"),
    )
    prior_hints_path = str(prior_hints_path or "").strip()

    enable_delegation = str(_pop_llm_arg(kwargs, "enable_delegation", True)).lower() not in {"0", "false", "no"}
    max_delegated_turns = int(_pop_llm_arg(kwargs, "max_delegated_turns", 8))
    select_skills = str(_pop_llm_arg(kwargs, "select_skills", True)).lower() not in {"0", "false", "no"}
    fixed_skill_ids = _pop_llm_arg(kwargs, "fixed_skill_ids", None)
    # Remaining llm_args are for the LLM call itself.
    llm = kwargs.get("llm")
    llm_args = dict(kwargs.get("llm_args") or {})

    task_text, task_domain, task_id = _task_text(task)
    # Window/PROMPT task line: customer scenario only (never "Purpose: ...").
    prompt_task = agent_facing_task_text(task) or clean_task_phrase(task_text)
    domain = resolve_episode_domain(task_domain)
    if not domain:
        logger.warning(
            "[sage_tau2] episode domain unknown (task + SAGE_TAU2_DOMAIN empty); "
            "dispatch may mis-route"
        )
        domain = "airline"

    agents = list(org.agents) if org is not None else []
    if not agents:
        agents = [
            AgentSpec(
                name=EXECUTOR_NAME,
                role="generalist",
                responsibilities=["Handle episode end-to-end"],
                acting_status="accepted",
                tool_permissions=["env_action"],
            )
        ]
        org = Organization(agents)

    skills = list(bank.active()) if bank is not None else []
    routing_scope = episode_routing_scope(task)
    if force_primary:
        primary = next((a for a in agents if a.name == force_primary), None)
        if primary is None:
            primary = org.executor() if org is not None else agents[0]
        assignment = AgentAssignment(
            primary_agent=primary.name,
            rationale=f"force_primary={force_primary}",
            dispatch_layer="force_primary",
            evidence={"force_primary": force_primary, "task_id": task_id},
        )
    elif enable_delegation and dispatch_cfg.enabled:
        primary = org.executor()
        assignment = AgentAssignment(primary_agent=primary.name, dispatch_layer="executor_coordinator")
    else:
        dispatcher = ExecutorDispatcher(config=dispatch_cfg, backend=None)
        assignment = dispatcher.assign(
            task=task_text or domain,
            domain=domain,
            agents=agents,
            skills=skills,
            executor_name=EXECUTOR_NAME,
            task_id=task_id,
            episode_scope=routing_scope,
        )
        primary = next(
            (a for a in agents if a.name == assignment.primary_agent),
            org.executor() if org is not None else agents[0],
        )

    if primary.name != EXECUTOR_NAME:
        record_primary_dispatch(primary)
        # Do not persist org here: τ² runs episodes concurrently and would race.

    journal = (
        _pop_llm_arg(kwargs, "dispatch_log_path", None)
        or os.environ.get("SAGE_TAU2_DISPATCH_LOG")
        or ""
    )

    injected = _skills_for_primary(
        primary=primary,
        bank=bank,
        max_inject=max_skills,
        allow_provisional=allow_provisional,
        domain=domain,
        same_domain_only=same_domain_only,
        allowed_skill_ids=allowed_skill_ids,
        task_id=str(task_id or ""),
        episode_scope=str(routing_scope or ""),
        task_text=str(task_text or ""),
        require_scope_match=require_scope_match,
    )
    if fixed_skill_ids is not None:
        injected = [sk for sk in skills if sk.skill_id in set(fixed_skill_ids)][:max_skills]
    is_specialist = primary.name != EXECUTOR_NAME

    # Build the full candidate pool for step-level retrieval (Executor only).
    # Specialists keep assigned-skill ownership; no per-step gate needed.
    executor_candidate_pool: list[Tau2Skill] = []
    if max_skills > 0 and not is_specialist and bank is not None and fixed_skill_ids is None:
        # Step-level retrieval: search the full injectable domain pool.
        # Pre-filtering with skill_matches_episode here would bypass BM25
        # retrieval and often leave pool=0 when scope gates are strict.
        executor_candidate_pool = candidate_pool_for_domain(
            bank,
            domain=domain,
            allow_provisional=allow_provisional,
            same_domain_only=same_domain_only,
            allowed_skill_ids=allowed_skill_ids,
            task_id=str(task_id or ""),
            episode_scope=str(routing_scope or ""),
            task_text=str(task_text or ""),
            require_scope_match=False,
        )
    org_block = format_organization_for_prompt(org) if org is not None else (
        "<organization>\nExecutor: Executor (default generalist)\n"
        "Specialists: (none yet)\n</organization>"
    )
    if is_specialist:
        # Specialists still see org roster for context in journal; prompt omits it.
        pass
    role_block = _role_block_for(primary)

    # Load prior-attempt verdict feedback for this specific task (retry only).
    prior_hint = ""
    if prior_hints_path:
        try:
            hints_doc = json.loads(Path(prior_hints_path).read_text())
            prior_hint = str(hints_doc.get(str(task_id or "")) or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[sage_tau2] prior hints load failed: {exc}")
    if prior_hint:
        # Prepend to task_description so it enters the system prompt's task line.
        prompt_task = f"{prior_hint}\n\n{prompt_task}"

    delegates = []
    delegate_skills = []
    if max_skills > 0 and enable_delegation and dispatch_cfg.enabled and not force_primary and primary.name == EXECUTOR_NAME:
        gate = ExecutorDispatcher(config=dispatch_cfg)
        for candidate in agents:
            if candidate.name == EXECUTOR_NAME or not gate._is_dispatchable(candidate, primary):
                continue
            # Unproven legacy specialists cannot bypass the paired admission gate.
            if not (candidate.shadow_evaluation_record or {}).get("same_skill_probe_passed"):
                continue
            owned = _skills_for_primary(primary=candidate, bank=bank, max_inject=max_skills,
                allow_provisional=allow_provisional, domain=domain, same_domain_only=True,
                allowed_skill_ids=allowed_skill_ids, require_scope_match=False)
            if owned:
                delegates.append(candidate)
                delegate_skills.extend(owned)
    agent = SageTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        llm_args=llm_args,
        skills=injected,
        organization_block=org_block if not is_specialist else "",
        role_block=role_block,
        agent_role_name=primary.name,
        assignment=assignment,
        domain=domain,
        task_description=prompt_task or None,
        candidate_pool=executor_candidate_pool or None,
        task_id=str(task_id or ""),
        max_active_skills=max_skills,
        allow_provisional=allow_provisional,
        delegates=delegates, delegate_skills=delegate_skills,
        max_delegated_turns=max_delegated_turns, select_skills=select_skills,
    )
    skills_block = format_skills_for_prompt(injected, specialist=is_specialist)

    active_skills = list(bank.active()) if bank is not None else []
    domain_skills = [
        s
        for s in active_skills
        if str(getattr(s, "domain", "") or "").lower() == str(domain or "").lower()
    ]

    append_dispatch_event(
        journal or None,
        task_id=str(task_id or ""),
        primary=assignment.primary_agent,
        domain=domain,
        layer=assignment.dispatch_layer,
        system_prompt=agent.system_prompt,
        role_block=role_block,
        organization_block=org_block,
        skills_block=skills_block,
        injected_skill_ids=[s.skill_id for s in injected],
        injected_skill_names=[s.skill_name for s in injected],
        bank_active_count=len(active_skills),
        bank_domain_active_count=len(domain_skills),
        bank_domain_skill_names=[s.skill_name for s in domain_skills[:20]],
        task_text=(prompt_task or task_text or "")[:500],
        assignment=assignment_to_dict(assignment),
        eligible_agents=list(assignment.eligible_agents or []),
        candidate_pool_size=len(executor_candidate_pool),
        step_level_retrieval=bool(executor_candidate_pool),
    )

    logger.info(
        f"[sage_tau2] dispatch primary={assignment.primary_agent} "
        f"layer={assignment.dispatch_layer} "
        f"eligible={assignment.eligible_agents} "
        f"inject={len(injected)} same_domain_only={same_domain_only} "
        f"domain={domain} bank={bank_path} org={org_path}"
    )
    # Also print: tau2 loguru often hides INFO in the runner tee stream.
    print(
        f"[sage_tau2] dispatch primary={assignment.primary_agent} "
        f"layer={assignment.dispatch_layer} "
        f"eligible={assignment.eligible_agents} inject={len(injected)} "
        f"same_domain_only={int(same_domain_only)} domain={domain}",
        flush=True,
    )

    return agent


def assignment_to_dict(assignment: AgentAssignment | None) -> dict[str, Any] | None:
    if assignment is None:
        return None
    return to_primitive(asdict(assignment))
