"""Retry when actor replies lack <action>...</action>."""

from __future__ import annotations

import unittest

from sage_mas.runtime import LLMResult, MASRuntime
from sage_mas.schemas import AgentSpec


class SequencedBackend:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[str] = []

    def complete(self, system_prompt, user_prompt, max_completion_tokens=None):
        self.calls.append(user_prompt)
        if not self.replies:
            raise AssertionError("unexpected extra complete() call")
        return LLMResult(self.replies.pop(0), prompt_tokens=1, completion_tokens=1)


class ActionFormatRetryTests(unittest.TestCase):
    def test_clears_specialist_token_budget(self):
        backend = SequencedBackend(
            ["<think>ok</think><action>look</action>"]
        )
        specialist = AgentSpec(
            name="InspectWithLightSpecialist",
            role="Capability specialist",
            responsibilities=["Inspect"],
            assigned_skills=["inspect skill"],
            token_budget=768,
            shadow_evaluation_record={"acting_status": "accepted"},
        )
        runtime = MASRuntime(
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Issue actions"],
                ),
                specialist,
            ],
            skills=[],
            backend=backend,
            action_format_retries=0,
        )
        self.assertIsNone(specialist.token_budget)
        self.assertIsNone(runtime.executor.token_budget)

    def test_retries_once_when_action_tag_missing(self):
        backend = SequencedBackend(
            [
                "<think>I should go</think>",
                "<think>go north</think><action>go north</action>",
            ]
        )
        runtime = MASRuntime(
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Issue actions"],
                )
            ],
            skills=[],
            backend=backend,
            action_format_retries=1,
        )
        result = runtime.act("obs text")
        self.assertEqual(len(backend.calls), 2)
        self.assertIn("FORMAT ERROR", backend.calls[1])
        self.assertIn("<action>go north</action>", result.action)
        self.assertEqual(result.token_cost, 4)

    def test_no_retry_when_action_present(self):
        backend = SequencedBackend(
            ["<think>ok</think><action>look</action>"]
        )
        runtime = MASRuntime(
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Issue actions"],
                )
            ],
            skills=[],
            backend=backend,
            action_format_retries=1,
        )
        result = runtime.act("obs")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result.action, "<think>ok</think><action>look</action>")

    def test_retries_disabled(self):
        backend = SequencedBackend(["<think>no tag</think>"])
        runtime = MASRuntime(
            agents=[
                AgentSpec(
                    name="Executor",
                    role="Environment executor",
                    responsibilities=["Issue actions"],
                )
            ],
            skills=[],
            backend=backend,
            action_format_retries=0,
        )
        result = runtime.act("obs")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result.action, "<think>no tag</think>")


if __name__ == "__main__":
    unittest.main()
