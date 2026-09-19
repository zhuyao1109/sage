#!/usr/bin/env python3
"""Runnable example: SAGE Executor agent on a single mock task.

Usage:
    cd tau2-bench
    export OPENAI_API_KEY=...
    export OPENAI_API_BASE=http://.../v1   # optional OpenAI-compatible relay
    uv run python examples/agents/sage_executor_agent.py
"""

from __future__ import annotations

import os

from tau2.agent.sage_executor_agent import create_sage_executor_agent
from tau2.data_model.simulation import TextRunConfig
from tau2.registry import registry
from tau2.runner import get_tasks, run_single_task


def main() -> None:
    # Already registered in tau2.registry; re-register is harmless if names collide
    # only when running this file in isolation before importing registry defaults.
    if "sage_executor" not in registry.get_agents():
        registry.register_agent_factory(create_sage_executor_agent, "sage_executor")

    llm = os.environ.get("TAU2_AGENT_LLM", "openai/gpt-4o-mini")
    tasks = get_tasks("mock", task_ids=["create_task_1"])
    result = run_single_task(
        TextRunConfig(
            domain="mock",
            agent="sage_executor",
            llm_agent=llm,
            llm_user=llm,
        ),
        tasks[0],
        seed=42,
    )
    print(f"Task: {result.task_id}")
    print(f"Reward: {result.reward_info.reward}")
    print(f"Messages: {len(result.messages)}")


if __name__ == "__main__":
    main()
