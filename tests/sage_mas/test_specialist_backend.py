"""Specialist backend routing: Executor vs specialist use different LLMs."""

from __future__ import annotations

import unittest

from sage_mas.runtime import LLMResult, MASRuntime
from sage_mas.schemas import AgentSpec


class RecordingBackend:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[str] = []

    def complete(self, system_prompt, user_prompt, max_completion_tokens=None):
        self.calls.append(user_prompt)
        return LLMResult(f"<action>{self.name}</action>")


class SpecialistBackendTests(unittest.TestCase):
    def _runtime(self, *, with_specialist_backend: bool):
        teacher = RecordingBackend("teacher")
        flash = RecordingBackend("flash")
        executor = AgentSpec(
            name="Executor",
            role="Environment executor",
            responsibilities=["Issue actions"],
        )
        specialist = AgentSpec(
            name="HeatSpecialist",
            role="Capability specialist",
            responsibilities=["Heat tasks"],
            assigned_skills=["heat skill"],
            shadow_evaluation_record={"acting_status": "accepted"},
        )
        runtime = MASRuntime(
            agents=[executor, specialist],
            skills=[],
            backend=teacher,
            specialist_backend=flash if with_specialist_backend else None,
        )
        return runtime, teacher, flash, executor, specialist

    def test_backend_for_routes_specialist(self):
        runtime, teacher, flash, executor, specialist = self._runtime(
            with_specialist_backend=True
        )
        self.assertIs(runtime._backend_for(executor), teacher)
        self.assertIs(runtime._backend_for(specialist), flash)

    def test_complete_uses_specialist_backend(self):
        runtime, teacher, flash, executor, specialist = self._runtime(
            with_specialist_backend=True
        )
        result = runtime._complete(specialist, "", "obs")
        self.assertIn("flash", result.content)
        self.assertEqual(len(flash.calls), 1)
        self.assertEqual(len(teacher.calls), 0)

        result2 = runtime._complete(executor, "", "obs2")
        self.assertIn("teacher", result2.content)
        self.assertEqual(len(teacher.calls), 1)

    def test_without_specialist_backend_all_use_teacher(self):
        runtime, teacher, flash, executor, specialist = self._runtime(
            with_specialist_backend=False
        )
        self.assertIs(runtime._backend_for(specialist), teacher)
        runtime._complete(specialist, "", "obs")
        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(len(flash.calls), 0)


if __name__ == "__main__":
    unittest.main()
