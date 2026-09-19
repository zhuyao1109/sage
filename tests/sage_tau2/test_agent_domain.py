"""Episode domain resolution for sage_tau2 dispatch (no τ² runtime)."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from sage_tau2.task_context import domain_from_task, resolve_episode_domain, task_text_and_domain
from sage_tau2.credit import initialize_credit
from sage_tau2.executor_dispatch import ExecutorDispatchConfig, ExecutorDispatcher
from sage_tau2.organization import EXECUTOR_NAME, Organization, default_executor
from sage_tau2.schemas import AgentSpec, SkillStatus, Tau2Skill


def _load_retail_task() -> dict:
    path = Path(__file__).resolve().parents[2] / "tau2-bench/data/tau2/domains/retail/tasks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data["73"]
    return next(item for item in data if str(item.get("id")) == "73")


class AgentDomainTests(unittest.TestCase):
    def test_domain_from_task_instructions(self) -> None:
        task = _load_retail_task()
        self.assertEqual(domain_from_task(task), "retail")
        text, domain, task_id = task_text_and_domain(task)
        self.assertEqual(domain, "retail")
        self.assertEqual(task_id, "73")
        self.assertTrue(text)

    def test_domain_from_mock_task_object(self) -> None:
        raw = _load_retail_task()

        class MockTask:
            id = raw["id"]
            description = raw.get("description")
            user_scenario = raw.get("user_scenario")

        self.assertEqual(domain_from_task(MockTask()), "retail")

    def test_resolve_prefers_task_over_env(self) -> None:
        os.environ["SAGE_TAU2_DOMAIN"] = "telecom"
        try:
            self.assertEqual(resolve_episode_domain("retail"), "retail")
        finally:
            os.environ.pop("SAGE_TAU2_DOMAIN", None)

    def test_resolve_uses_env_when_task_empty(self) -> None:
        os.environ["SAGE_TAU2_DOMAIN"] = "telecom"
        try:
            self.assertEqual(resolve_episode_domain(""), "telecom")
        finally:
            os.environ.pop("SAGE_TAU2_DOMAIN", None)

    def test_retail_specialist_eligible_with_correct_domain(self) -> None:
        raw = _load_retail_task()
        text, domain, _ = task_text_and_domain(raw)
        skill = Tau2Skill(
            skill_name="find user id skill",
            description="find user",
            precondition="return items",
            action_protocol=["find_user_id_by_email(email=?)"],
            expected_effect="ok",
            capability_key="tau2.find_user_id_by_email",
            status=SkillStatus.VERIFIED,
            domain="retail",
        )
        initialize_credit(skill)
        specialist = AgentSpec(
            name="FindUserIdSpecialist",
            role="specialist",
            responsibilities=["find user"],
            assigned_skills=[skill.skill_name],
            capability_keys=[skill.capability_key],
            tool_permissions=["env_action"],
            acting_status="probation",
            metadata={"domains": ["retail"]},
            shadow_evaluation_record={
                "promotion_probe_passed": False,
                "dispatch_only": True,
                "applicable_dispatched_games": 0,
                "trial_games_remaining": 3,
            },
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(prefer_matching_specialist=True)
        )
        assignment = dispatcher.assign(
            task=text,
            domain=domain,
            agents=org.agents,
            skills=[skill],
            executor_name=EXECUTOR_NAME,
            task_id="73",
        )
        self.assertEqual(assignment.primary_agent, "FindUserIdSpecialist")

    def test_agent_facing_skips_purpose_label(self) -> None:
        from sage_tau2.task_context import agent_facing_task_text, clean_task_phrase

        raw = {
            "id": "15",
            "description": {
                "purpose": "Test finding cheapest Economy cabin class.",
            },
            "user_scenario": {
                "instructions": {
                    "reason_for_call": "For your upcoming trip from ATL to PHL, change to cheapest Economy.",
                    "task_instructions": "Consider EWR and PHL.",
                    "known_info": "Your user id is aarav_garcia_1177.",
                }
            },
        }
        facing = agent_facing_task_text(raw)
        self.assertNotIn("Purpose:", facing)
        self.assertTrue(facing.startswith("For your upcoming trip"))
        self.assertNotIn("Consider EWR and PHL", facing)
        self.assertIn("aarav_garcia_1177", facing)
        self.assertEqual(
            clean_task_phrase("Purpose: Test finding cheapest."),
            "Test finding cheapest.",
        )

    def test_object_description_not_str_labeled(self) -> None:
        from sage_tau2.task_context import flatten_task_description, task_text_and_domain

        class _Desc:
            purpose = "Test finding cheapest."
            notes = None
            summary = None

            def __str__(self) -> str:
                return f"Purpose: {self.purpose}"

        class _Task:
            id = "15"
            description = _Desc()
            user_scenario = {
                "instructions": {
                    "reason_for_call": "Change my ATL to PHL flight.",
                    "task_instructions": None,
                    "known_info": None,
                }
            }

        text, _, tid = task_text_and_domain(_Task())
        self.assertEqual(tid, "15")
        self.assertNotIn("Purpose:", text)
        self.assertEqual(flatten_task_description(_Desc()), "Test finding cheapest.")

    def test_retail_specialist_blocked_when_domain_wrong(self) -> None:
        raw = _load_retail_task()
        text, _, _ = task_text_and_domain(raw)
        skill = Tau2Skill(
            skill_name="find user id skill",
            description="find user",
            precondition="return items",
            action_protocol=["find_user_id_by_email(email=?)"],
            expected_effect="ok",
            capability_key="tau2.find_user_id_by_email",
            status=SkillStatus.VERIFIED,
            domain="retail",
        )
        initialize_credit(skill)
        specialist = AgentSpec(
            name="FindUserIdSpecialist",
            role="specialist",
            responsibilities=["find user"],
            assigned_skills=[skill.skill_name],
            capability_keys=[skill.capability_key],
            tool_permissions=["env_action"],
            acting_status="probation",
            metadata={"domains": ["retail"]},
            shadow_evaluation_record={
                "promotion_probe_passed": False,
                "dispatch_only": True,
                "applicable_dispatched_games": 0,
                "trial_games_remaining": 3,
            },
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(ExecutorDispatchConfig())
        assignment = dispatcher.assign(
            task=text,
            domain="airline",
            agents=org.agents,
            skills=[skill],
            executor_name=EXECUTOR_NAME,
            task_id="73",
        )
        self.assertEqual(assignment.primary_agent, EXECUTOR_NAME)
        self.assertEqual(assignment.dispatch_layer, "eligibility_empty")


if __name__ == "__main__":
    unittest.main()
