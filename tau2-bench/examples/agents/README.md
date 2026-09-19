# Agent Examples

Runnable examples showing how to create and evaluate custom tau2 agents.

## Examples

### `minimal_text_agent.py` -- Start here

A single-file example that creates a minimal agent, registers it, and runs it against the mock domain. Shows:

- Implementing `HalfDuplexAgent` (the two required methods)
- Writing a factory function
- Registering with the registry
- Running via `run_single_task` or `run_domain`

```bash
python examples/agents/minimal_text_agent.py
```

### `sage_executor_agent.py` -- SAGE Executor adapter

SAGE-MAS Executor style for τ²-bench: policy-bound single actor with
think-then-act turns. Registered as ``sage_executor`` in the core registry.

```bash
# single mock task
uv run python examples/agents/sage_executor_agent.py

# compare vs llm_agent on first 5 airline tasks (uses verl-agent llm_config.yaml relay)
uv run python examples/agents/run_sage_executor_compare.py

# or via CLI
uv run tau2 run --domain airline --agent sage_executor \
  --agent-llm openai/gpt-4o-mini --user-llm openai/gpt-4o-mini \
  --num-tasks 5 --num-trials 1 --seed 42
```

### `react_agent.py` -- ReAct pattern

A ReAct (Reasoning + Acting) agent that explicitly thinks before acting. Each turn follows:

1. **THINK** -- reason about the situation (LLM call without tools)
2. **ACT** -- choose a tool call or text response based on the reasoning (LLM call with tools)

Shows how to customize the agent's decision-making process to improve tool-use accuracy.

```bash
python examples/agents/react_agent.py
```

### `custom_agent_eval.py` -- Power user path

Builds all components manually without the registry. Shows:

- Building environment, agent, user, and orchestrator by hand
- Running `run_simulation()` directly
- Inspecting results (messages, rewards, evaluation details)
- Adding custom behavior (logging, call counting)

```bash
python examples/agents/custom_agent_eval.py
```

## The Agent Interface

Every text agent must subclass `HalfDuplexAgent` and implement two methods:

```python
class MyAgent(HalfDuplexAgent[MyState]):

    def get_init_state(self, message_history=None) -> MyState:
        """Return the initial state (e.g., system prompt + history)."""
        ...

    def generate_next_message(self, message, state) -> tuple[AssistantMessage, MyState]:
        """Given a user/tool message and current state, return (response, new_state)."""
        ...
```

The agent receives `tools: list[Tool]` and `domain_policy: str` in `__init__`.

See `src/tau2/agent/README.md` for the full developer guide.
