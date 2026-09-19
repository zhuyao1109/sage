"""Smoke tests: telecom agent/user tool-side separation (no LLM)."""

from __future__ import annotations

import unittest
from pathlib import Path

from sage_tau2.distill import (
    canonicalize_protocol,
    distill_skills_from_trajectories,
    normalize_protocol,
)
from sage_tau2.injection import format_skills_for_prompt
from sage_tau2.schemas import SkillStatus, Tau2Skill, Tau2Trajectory
from sage_tau2.tool_sides import (
    TELECOM_USER_TOOLS,
    is_tool_not_found_result,
    strip_user_side_protocol_steps,
)
from sage_tau2.trajectory import (
    extract_tool_steps,
    protocol_from_tool_steps,
    simulation_to_trajectory,
)


def _traj(**kwargs) -> Tau2Trajectory:
    base = dict(
        task_id="[mobile_data_issue]airplane_mode_on|data_usage_exceeded",
        trial=0,
        domain="telecom",
        reward=1.0,
        db_reward=1.0,
        communicate_reward=None,
        db_match=True,
        termination_reason="user_stop",
        tool_protocol=[],
        tool_steps=[],
        assistant_texts=[],
        user_texts=["My data is not working abroad"],
    )
    base.update(kwargs)
    return Tau2Trajectory(**base)


class TelecomToolSideSmokeTests(unittest.TestCase):
    def test_strip_drops_user_device_steps(self) -> None:
        proto = [
            "get_customer_by_phone(phone_number=?)",
            "toggle_airplane_mode()",
            "check_network_status()",
            "enable_roaming(customer_id=?, line_id=?)",
            "run_speed_test()",
            "refuel_data(customer_id=?, gb_amount=?, line_id=?)",
        ]
        cleaned = strip_user_side_protocol_steps(proto, domain="telecom")
        names = [s.split("(", 1)[0] for s in cleaned]
        self.assertEqual(
            names,
            ["get_customer_by_phone", "enable_roaming", "refuel_data"],
        )
        for name in TELECOM_USER_TOOLS:
            self.assertNotIn(name, names)

    def test_canonicalize_and_normalize_drop_user_tools(self) -> None:
        messy = [
            "get_customer_by_phone(phone_number=?)",
            "toggle_airplane_mode()",
            "toggle_airplane_mode()",
            "enable_roaming(line_id=?)",
            "run_speed_test()",
            "refuel_data(gb_amount=?)",
        ]
        canon = canonicalize_protocol(messy, domain="telecom")
        self.assertEqual(
            [s.split("(", 1)[0] for s in canon],
            ["get_customer_by_phone", "enable_roaming", "refuel_data"],
        )
        # Write-spine trim is opt-in; default normalize keeps full agent tools.
        self.assertEqual(
            [s.split("(", 1)[0] for s in normalize_protocol(messy, domain="telecom")],
            ["get_customer_by_phone", "enable_roaming", "refuel_data"],
        )
        self.assertEqual(
            [
                s.split("(", 1)[0]
                for s in normalize_protocol(
                    messy, domain="telecom", trim_to_write_spine=True
                )
            ],
            ["get_customer_by_phone", "enable_roaming"],
        )

    def test_distill_does_not_birth_user_tool_spine(self) -> None:
        # Legacy bug: toggle_airplane_mode was treated as a write spine.
        traj = _traj(
            tool_protocol=[
                "get_customer_by_phone(phone_number=?)",
                "toggle_airplane_mode()",
                "check_sim_status()",
            ],
        )
        skills = distill_skills_from_trajectories([traj], min_support=1)
        self.assertEqual(skills, [])

        traj_ok = _traj(
            tool_protocol=[
                "get_customer_by_phone(phone_number=?)",
                "toggle_airplane_mode()",
                "enable_roaming(line_id=?)",
                "run_speed_test()",
            ],
        )
        skills_ok = distill_skills_from_trajectories([traj_ok], min_support=1)
        self.assertEqual(len(skills_ok), 1)
        self.assertEqual(skills_ok[0].capability_key, "tau2.enable_roaming")
        agent_proto = skills_ok[0].metadata.get("agent_tool_protocol") or []
        proto_names = [s.split("(", 1)[0] for s in agent_proto]
        self.assertNotIn("toggle_airplane_mode", proto_names)
        self.assertNotIn("run_speed_test", proto_names)
        # Default normalize keeps the full agent write sequence (no spine trim).
        self.assertEqual(proto_names, ["get_customer_by_phone", "enable_roaming"])
        # Full action_protocol may inline dialogue/branches; still no user-side tools.
        full_names = [s.split("(", 1)[0] for s in skills_ok[0].action_protocol]
        self.assertNotIn("toggle_airplane_mode", full_names)
        self.assertNotIn("run_speed_test", full_names)
        self.assertIn("enable_roaming", " ".join(skills_ok[0].action_protocol))

    def test_trajectory_filters_not_found_and_openai_style_calls(self) -> None:
        sim = {
            "id": "sim-1",
            "task_id": "[mms_issue]x",
            "trial": 0,
            "termination_reason": "user_stop",
            "reward_info": {
                "reward": 1.0,
                "reward_breakdown": {"ENV_ASSERTION": 1.0},
                "db_check": {"db_match": True, "db_reward": 1.0},
            },
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "get_customer_by_phone",
                                "arguments": '{"phone_number": "555-123-2002"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": '{"customer_id": "C1001"}',
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c2",
                            "function": {
                                "name": "toggle_airplane_mode",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c2",
                    "content": "Error: Tool 'toggle_airplane_mode' not found.",
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c3",
                            "function": {
                                "name": "enable_roaming",
                                "arguments": '{"customer_id": "C1001", "line_id": "L1002"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c3",
                    "content": "Roaming enabled successfully",
                },
            ],
        }
        steps = extract_tool_steps(sim["messages"])
        self.assertEqual(len(steps), 3)
        self.assertTrue(steps[1].result_error)
        self.assertTrue(is_tool_not_found_result(steps[1].result_content))

        protocol = protocol_from_tool_steps(steps, domain="telecom")
        self.assertEqual(
            [s.split("(", 1)[0] for s in protocol],
            ["get_customer_by_phone", "enable_roaming"],
        )

        traj = simulation_to_trajectory(sim, domain="telecom")
        self.assertEqual(
            [s.split("(", 1)[0] for s in traj.tool_protocol],
            ["get_customer_by_phone", "enable_roaming"],
        )

    def test_injection_strips_legacy_user_tools_and_prompt_warns(self) -> None:
        skill = Tau2Skill(
            skill_name="legacy roaming skill",
            description="legacy",
            precondition="roaming",
            action_protocol=[
                "get_customer_by_phone(phone_number=?)",
                "toggle_airplane_mode()",
                "enable_roaming(line_id=?)",
                "run_speed_test()",
            ],
            expected_effect="ok",
            capability_key="tau2.enable_roaming",
            status=SkillStatus.VERIFIED,
            support_count=2,
            domain="telecom",
            metadata={"utility": 0.7},
        )
        text = format_skills_for_prompt([skill])
        self.assertIn("get_customer_by_phone", text)
        self.assertIn("enable_roaming", text)
        self.assertNotIn("toggle_airplane_mode", text)
        self.assertNotIn("run_speed_test", text)
        self.assertIn("Device/phone steps are never agent tools", text)
        # Agent prompt lives in prompts.py (avoid importing tau2 via agent).
        prompt_src = (
            Path(__file__).resolve().parents[2] / "sage_tau2" / "prompts.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Only call tools from your agent toolkit", prompt_src)
        self.assertIn("instruct the user", prompt_src)


if __name__ == "__main__":
    unittest.main()
