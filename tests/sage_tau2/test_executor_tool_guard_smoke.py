"""Smoke tests: Executor-source tool guard (no LLM / no tau2 runtime)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from sage_tau2.executor_tool_guard import (
    build_user_guidance,
    canonicalize_passenger_dob_args,
    classify_tool_names,
    ensure_assistant_payload,
    guard_assistant_message,
    telecom_tool_boundary_addendum,
    with_tool_boundary_policy,
)
from sage_tau2.tool_sides import TELECOM_AGENT_TOOLS


class _FakeTool:
    def __init__(self, name: str):
        self.name = name


class ExecutorToolGuardSmokeTests(unittest.TestCase):
    def test_classify_blocks_user_device_calls(self) -> None:
        allowed = set(TELECOM_AGENT_TOOLS)
        decision = classify_tool_names(
            [
                "get_customer_by_phone",
                "toggle_airplane_mode",
                "enable_roaming",
                "run_speed_test",
            ],
            allowed=allowed,
        )
        self.assertEqual(
            decision.kept_names,
            ("get_customer_by_phone", "enable_roaming"),
        )
        self.assertEqual(
            decision.blocked_names,
            ("toggle_airplane_mode", "run_speed_test"),
        )
        self.assertIsNone(decision.guidance_content)

    def test_all_illegal_becomes_user_guidance(self) -> None:
        decision = classify_tool_names(
            ["toggle_airplane_mode", "check_network_status"],
            allowed=set(TELECOM_AGENT_TOOLS),
        )
        self.assertEqual(decision.kept_names, ())
        self.assertIsNotNone(decision.guidance_content)
        self.assertIn("Airplane Mode", decision.guidance_content)
        self.assertIn("Network Status", decision.guidance_content)

    def test_build_user_guidance_dedupes(self) -> None:
        text = build_user_guidance(
            ["toggle_airplane_mode", "toggle_airplane_mode", "reseat_sim_card"]
        )
        self.assertEqual(text.count("Airplane Mode"), 1)
        self.assertIn("SIM", text)

    def test_policy_addendum_lists_agent_tools_only(self) -> None:
        addendum = telecom_tool_boundary_addendum(
            agent_tool_names=["get_customer_by_phone", "refuel_data"]
        )
        self.assertIn("`get_customer_by_phone`", addendum)
        self.assertIn("`refuel_data`", addendum)
        self.assertIn("NOT callable by you", addendum)
        self.assertIn("toggle_airplane_mode", addendum)
        self.assertIn("Do not transfer to a human after only looking up", addendum)
        # Principle-level escalate guidance — no write-tool checklist.
        self.assertNotIn("must call refuel_data", addendum.lower())

    def test_with_tool_boundary_appends_once_for_telecom_tools(self) -> None:
        tools = [_FakeTool("get_customer_by_phone"), _FakeTool("enable_roaming")]
        policy = with_tool_boundary_policy("BASE POLICY", tools=tools)
        self.assertIn("BASE POLICY", policy)
        self.assertIn("<agent_user_tool_boundary>", policy)
        again = with_tool_boundary_policy(policy, tools=tools)
        self.assertEqual(policy.count("<agent_user_tool_boundary>"), 1)
        self.assertEqual(again.count("<agent_user_tool_boundary>"), 1)

    def test_guard_rewrites_message_without_tau2(self) -> None:
        tools = [_FakeTool(n) for n in sorted(TELECOM_AGENT_TOOLS)]
        msg = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(name="toggle_airplane_mode"),
                SimpleNamespace(name="check_sim_status"),
            ],
            content=None,
            cost=None,
            usage=None,
        )
        out = guard_assistant_message(msg, tools=tools)
        self.assertIsNone(out.tool_calls)
        self.assertIn("can't change phone settings", out.content.lower())

    def test_guard_keeps_legal_subset(self) -> None:
        tools = [_FakeTool(n) for n in sorted(TELECOM_AGENT_TOOLS)]
        msg = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(name="get_customer_by_phone"),
                SimpleNamespace(name="toggle_airplane_mode"),
                SimpleNamespace(name="refuel_data"),
            ],
            content=None,
            cost=None,
            usage=None,
        )
        # model_copy missing → duck-patch path
        out = guard_assistant_message(msg, tools=tools)
        names = [tc.name for tc in (out.tool_calls or [])]
        self.assertEqual(names, ["get_customer_by_phone", "refuel_data"])

    def test_canonicalize_passenger_dob_alias(self) -> None:
        fixed = canonicalize_passenger_dob_args(
            "book_reservation",
            {
                "user_id": "u",
                "passengers": [
                    {
                        "first_name": "Ada",
                        "last_name": "Lovelace",
                        "date_of_birth": "1815-12-10",
                    }
                ],
            },
        )
        self.assertEqual(fixed["passengers"][0]["dob"], "1815-12-10")
        self.assertNotIn("date_of_birth", fixed["passengers"][0])

    def test_guard_rewrites_date_of_birth_to_dob(self) -> None:
        tools = [_FakeTool("book_reservation"), _FakeTool("get_user_details")]
        msg = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    name="book_reservation",
                    arguments={
                        "passengers": [
                            {
                                "first_name": "A",
                                "last_name": "B",
                                "date_of_birth": "1990-01-02",
                            }
                        ]
                    },
                )
            ],
            content=None,
            cost=None,
            usage=None,
        )
        out = guard_assistant_message(msg, tools=tools)
        args = out.tool_calls[0].arguments
        self.assertEqual(args["passengers"][0]["dob"], "1990-01-02")
        self.assertNotIn("date_of_birth", args["passengers"][0])

    def test_ensure_assistant_retries_then_returns(self) -> None:
        calls = {"n": 0}

        def produce():
            calls["n"] += 1
            if calls["n"] < 3:
                return SimpleNamespace(content="", tool_calls=None, cost=None, usage=None)
            return SimpleNamespace(
                content="hello", tool_calls=None, cost=None, usage=None
            )

        out = ensure_assistant_payload(produce, max_attempts=3, label="test")
        self.assertEqual(out.content, "hello")
        self.assertEqual(calls["n"], 3)

    def test_ensure_assistant_fallback_after_all_empty(self) -> None:
        out = ensure_assistant_payload(
            lambda: SimpleNamespace(content="  ", tool_calls=None, cost=1, usage=None),
            max_attempts=2,
            label="test",
        )
        self.assertTrue(str(out.content).strip())
        self.assertIn("restate", out.content.lower())


if __name__ == "__main__":
    unittest.main()
