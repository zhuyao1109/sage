"""SAGE Executor agent for τ²-bench.

Mirrors SAGE-MAS Executor behavior on a tool-calling customer-service domain:

- Single primary actor owns the full episode (no specialist handoff yet).
- Policy is binding; tools are the only way to change world state.
- Brief internal planning (think) before each tool/user act, analogous to
  SAGE's ``<think>...</think><action>...</action>`` turn discipline.

This is an evaluation adapter, not the full SAGE online skill/org pipeline.
"""

from __future__ import annotations

from typing import Generic, List, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel

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

try:
    from sage_tau2.executor_tool_guard import (
        ensure_assistant_payload,
        guard_assistant_message,
        with_tool_boundary_policy,
    )
except ImportError:  # pragma: no cover - tau2-only checkout without sage_tau2
    ensure_assistant_payload = None  # type: ignore
    guard_assistant_message = None  # type: ignore
    with_tool_boundary_policy = None  # type: ignore

# Executor role — adapted from sage_mas Runtime actor prompts for tool domains.
EXECUTOR_INSTRUCTION = """
You are Executor, the primary customer-service actor for this episode.
You own the full conversation until the task is resolved or transferred.

Operating rules (SAGE Executor discipline):
1. Policy is binding. Never invent fees, waivers, eligibility, or database facts.
2. World state only changes through tools. Read before you write.
3. Prefer the smallest correct tool sequence; do not call tools you do not need.
4. Do not claim an action succeeded unless a tool result confirms it.
5. Ask the user only for information you cannot obtain from tools/policy.
6. Each turn: either talk to the user OR call tool(s) — never both.
7. Stay concise, professional, and grounded in tool observations.
8. Only call tools from your agent toolkit. Phone/device actions named in the
   policy (status bar, airplane mode, SIM reseat, speed test, APN, permissions,
   roaming toggle, etc.) are USER actions — instruct the user to perform them;
   never invoke those names as your own tool calls.
9. Do not transfer until you have made a genuine resolve attempt: relevant
   agent-side tools and/or guided user device steps with results. Looking up
   the customer and immediately transferring is not enough.
10. Escalate only when policy says out of scope, or after those attempts fail.

Always produce valid structured tool calls when using tools.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

THINK_PROMPT = """
Before acting, write a short internal plan (SAGE-style think):
- Goal of this turn
- Facts already confirmed by tools/user
- Missing facts (tool vs ask user)
- Policy constraints that apply
- Next action: which tool(s), or what to say

Be brief (under 120 words). Do not call tools in this step.
""".strip()

ACT_PROMPT = """
Your internal plan:
{reasoning}

Now execute as Executor:
- If you need environment state or a mutation, call the appropriate tool(s).
- If you can reply to the user, send a concise message that follows policy.
- Do not contradict tool results. Do not both talk and call tools.
""".strip()


class SageExecutorState(BaseModel):
    """Conversation state for the SAGE Executor agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


SageExecutorStateType = TypeVar("SageExecutorStateType", bound=SageExecutorState)


class SageExecutorAgent(
    LLMConfigMixin,
    HalfDuplexAgent[SageExecutorStateType],
    Generic[SageExecutorStateType],
):
    """SAGE Executor half-duplex agent with think-then-act turns."""

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        *,
        enable_think: bool = True,
    ):
        policy = domain_policy
        if with_tool_boundary_policy is not None:
            policy = with_tool_boundary_policy(domain_policy, tools=tools)
        super().__init__(
            tools=tools,
            domain_policy=policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.enable_think = bool(enable_think)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            agent_instruction=EXECUTOR_INSTRUCTION,
            domain_policy=self.domain_policy,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> SageExecutorStateType:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, "
            "or ToolMessage to Agent."
        )
        return SageExecutorState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: SageExecutorStateType
    ) -> tuple[AssistantMessage, SageExecutorStateType]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        if self.enable_think:
            reasoning = self._think(state)
            logger.debug(f"[sage_executor] think: {reasoning[:200]}")

            def _produce() -> AssistantMessage:
                return self._act(state, reasoning)

        else:

            def _produce() -> AssistantMessage:
                return self._act_direct(state)

        def _produce_guarded() -> AssistantMessage:
            msg = _produce()
            if guard_assistant_message is not None:
                msg = guard_assistant_message(msg, tools=self.tools)
            return msg

        if ensure_assistant_payload is not None:
            assistant_message = ensure_assistant_payload(
                _produce_guarded, max_attempts=3, label="sage_executor"
            )
        else:
            assistant_message = _produce_guarded()
        state.messages.append(assistant_message)
        return assistant_message, state

    def _think(self, state: SageExecutorStateType) -> str:
        think_messages = (
            state.system_messages
            + state.messages
            + [UserMessage(role="user", content=THINK_PROMPT)]
        )
        response = generate(
            model=self.llm,
            tools=[],
            messages=think_messages,
            call_name="sage_executor_think",
            **self.llm_args,
        )
        return str(response.content) if response.content else ""

    def _act(self, state: SageExecutorStateType, reasoning: str) -> AssistantMessage:
        act_messages = (
            state.system_messages
            + state.messages
            + [
                UserMessage(
                    role="user",
                    content=ACT_PROMPT.format(reasoning=reasoning or "(none)"),
                )
            ]
        )
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=act_messages,
            call_name="sage_executor_act",
            **self.llm_args,
        )

    def _act_direct(self, state: SageExecutorStateType) -> AssistantMessage:
        messages = state.system_messages + state.messages
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="sage_executor_act",
            **self.llm_args,
        )


def create_sage_executor_agent(tools, domain_policy, **kwargs):
    """Factory for registry / ``tau2 run --agent sage_executor``."""
    llm_args = dict(kwargs.get("llm_args") or {})
    enable_think = llm_args.pop("enable_think", True)
    if isinstance(enable_think, str):
        enable_think = enable_think.strip().lower() not in {"0", "false", "no"}
    return SageExecutorAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=llm_args,
        enable_think=bool(enable_think),
    )
