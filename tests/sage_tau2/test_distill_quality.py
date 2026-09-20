"""Tests for τ² distill quality gates (no runtime)."""

from __future__ import annotations

import unittest

from sage_tau2.distill import (
    distill_skills_from_trajectories,
    is_early_escalate_protocol,
    is_escalate_only_capability,
    is_read_only_protocol,
    trajectory_eligible_for_distill,
)
from sage_tau2.schemas import Tau2Trajectory
from sage_tau2.task_context import episode_routing_scope, episode_scope_from_task_id


def _traj(**kwargs) -> Tau2Trajectory:
    base = dict(
        task_id="0",
        trial=0,
        domain="airline",
        reward=1.0,
        db_reward=1.0,
        communicate_reward=1.0,
        db_match=True,
        termination_reason="user_stop",
        tool_protocol=[],
        tool_steps=[],
        assistant_texts=[],
        user_texts=["Please cancel my reservation"],
    )
    base.update(kwargs)
    return Tau2Trajectory(**base)


class DistillQualityTests(unittest.TestCase):
    def test_episode_scope_from_telecom_task_id(self) -> None:
        self.assertEqual(
            episode_scope_from_task_id(
                "[mms_issue]airplane_mode_on|data_usage_exceeded"
            ),
            "mms_issue",
        )
        self.assertEqual(episode_scope_from_task_id("73"), "")

    def test_bug_tags_and_clean_precondition(self) -> None:
        from sage_tau2.distill import (
            _clean_user_intent_snippet,
            _expected_effect,
            bind_protocol_slots,
            user_side_hints_from_trajs,
        )
        from sage_tau2.task_context import bug_tags_from_task_id

        tid = (
            "[mobile_data_issue]airplane_mode_on|bad_vpn|"
            "user_abroad_roaming_enabled_off[PERSONA:Easy]"
        )
        self.assertEqual(
            bug_tags_from_task_id(tid),
            [
                "airplane_mode_on",
                "bad_vpn",
                "user_abroad_roaming_enabled_off",
            ],
        )
        dirty = (
            "I'm in France [tool_call] toggle_roaming({}) "
            "and check_network_status({}) please"
        )
        clean = _clean_user_intent_snippet(dirty)
        self.assertNotIn("toggle_roaming", clean)
        self.assertNotIn("[tool_call]", clean)
        self.assertIn("France", clean)

        traj = _traj(
            domain="telecom",
            task_id=tid,
            reward=1.0,
            db_match=False,
            db_reward=None,
            tool_protocol=[
                "get_customer_by_phone(phone_number=?)",
                "get_details_by_id(id=?)",
                "enable_roaming(customer_id=?, line_id=?)",
            ],
            user_texts=[dirty],
            metadata={
                "action_checks": [
                    {
                        "action": {"requestor": "user", "name": "disconnect_vpn"},
                        "action_match": True,
                    }
                ]
            },
        )
        skills = distill_skills_from_trajectories(
            [traj], min_support=1, inline_dialogue_in_protocol=True
        )
        self.assertEqual(len(skills), 1)
        skill = skills[0]
        self.assertEqual(skill.status.value, "candidate")
        self.assertNotIn("[tool_call]", skill.precondition)
        self.assertNotIn("toggle_roaming({})", skill.precondition)
        self.assertIn("env_reward_ok=", skill.expected_effect)
        self.assertIn("db_match_ok=", skill.expected_effect)
        proto = " | ".join(skill.action_protocol)
        self.assertIn("diagnose boolean flags", proto)
        self.assertIn("resolve target line_id", proto)
        self.assertIn("user says yes", proto)
        self.assertIn("user says no", proto)
        self.assertIn("skip write", proto)
        # line_id resolve before diagnostic if-branches and before write
        resolve_i = next(
            i for i, s in enumerate(skill.action_protocol) if s.startswith("resolve ")
        )
        if_i = next(
            i for i, s in enumerate(skill.action_protocol) if s.startswith("if airplane")
        )
        yes_i = next(
            i
            for i, s in enumerate(skill.action_protocol)
            if "user says yes" in s and "enable_roaming(" in s
        )
        self.assertLess(resolve_i, if_i)
        self.assertLess(if_i, yes_i)
        self.assertIn("never_default_first_listed_line", proto)
        self.assertNotIn("line_id=<line_id from details; ask if multiple>", proto)
        # Branches must not instruct an unconditional agent enable_roaming call
        branch_blob = " | ".join(
            s for s in skill.action_protocol if s.startswith("if ")
        )
        self.assertIn("mark need_enable_roaming", branch_blob)
        self.assertIn("do not call toggle_roaming or enable_roaming in this step", branch_blob)
        agent_tools = skill.metadata.get("agent_tool_protocol") or []
        self.assertEqual(
            sum(1 for s in agent_tools if s.startswith("enable_roaming(")),
            1,
        )
        self.assertEqual(
            skill.capability_key, "tau2.enable_roaming"
        )
        self.assertEqual(
            skill.metadata.get("protocol_bucket")[0],
            "solve_write",
        )
        self.assertEqual(
            skill.metadata.get("protocol_bucket")[1],
            "tau2.enable_roaming",
        )
        self.assertEqual(
            skill.metadata.get("write_capability_key"), "tau2.enable_roaming"
        )
        self.assertEqual(
            skill.metadata.get("parent_capability_key"),
            "tau2.mobile_data_issue_abroad",
        )
        self.assertEqual(
            skill.metadata.get("protocol_bucket_write_spine")[0],
            "solve_write",
        )
        self.assertEqual(skill.metadata.get("success_mode"), "solve_write")
        self.assertEqual(skill.metadata.get("primary_task_family"), "mobile_data_issue")
        self.assertEqual(
            skill.metadata.get("bug_bucket_mode"), "union_with_signature_gate"
        )
        attr = skill.metadata.get("success_attribution") or {}
        self.assertTrue(attr.get("env_user_side_primary"))
        cues = skill.metadata.get("intent_cues") or []
        for noisy in ("bad", "data", "mode", "off", "user", "enabled"):
            self.assertNotIn(noisy, cues)
        self.assertTrue(
            any(
                c in cues
                for c in (
                    "airplane_mode_on",
                    "bad_vpn",
                    "france",
                    "roaming",
                    "mobile_data_slow",
                )
            )
        )
        # Legacy builders still available / stored.
        self.assertIn("precondition_legacy", skill.metadata)
        self.assertIn("db_ok=", skill.metadata.get("expected_effect_legacy") or "")
        self.assertIn("env_reward_ok=", _expected_effect([traj]))
        bound = bind_protocol_slots(
            [
                "get_customer_by_phone(phone_number=?)",
                "get_details_by_id(id=?)",
                "enable_roaming(customer_id=?, line_id=?)",
            ]
        )
        self.assertIn("<entity_id_from_prior_tool_result", bound[1])
        self.assertTrue(
            any("disconnect_vpn" in h for h in user_side_hints_from_trajs([traj]))
        )

        # signature gate: shared core required
        from sage_tau2.task_context import skill_matches_episode

        self.assertTrue(
            skill_matches_episode(
                skill,
                domain="telecom",
                task_id=tid,
            )
        )
        self.assertFalse(
            skill_matches_episode(
                skill,
                domain="telecom",
                task_id="[mobile_data_issue]airplane_mode_on[PERSONA:None]",
            )
        )

        # exact bug bucketing keeps mixed root causes apart
        traj_b = _traj(
            domain="telecom",
            task_id="[mobile_data_issue]bad_network_preference|user_abroad_roaming_enabled_off",
            reward=1.0,
            db_match=False,
            db_reward=None,
            tool_protocol=list(traj.tool_protocol),
            user_texts=["France roaming slow"],
        )
        split = distill_skills_from_trajectories(
            [traj, traj_b],
            min_support=1,
            bug_bucket_mode="exact",
            inline_dialogue_in_protocol=True,
        )
        self.assertEqual(len(split), 2)
        merged = distill_skills_from_trajectories(
            [traj, traj_b],
            min_support=2,
            bug_bucket_mode="union_with_signature_gate",
            inline_dialogue_in_protocol=True,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            merged[0].metadata.get("bug_intersection"),
            ["user_abroad_roaming_enabled_off"],
        )
        self.assertGreaterEqual(len(merged[0].metadata.get("bug_tags_union") or []), 3)
        # airplane-only still rejected by shared-core gate on the merged skill
        self.assertFalse(
            skill_matches_episode(
                merged[0],
                domain="telecom",
                task_id="[mobile_data_issue]airplane_mode_on[PERSONA:None]",
            )
        )
        self.assertTrue(
            skill_matches_episode(
                merged[0],
                domain="telecom",
                task_id=(
                    "[mobile_data_issue]airplane_mode_on|"
                    "user_abroad_roaming_enabled_off[PERSONA:None]"
                ),
            )
        )

    def test_skips_transfer_only_protocol(self) -> None:
        traj = _traj(
            domain="telecom",
            task_id="[mms_issue]x",
            tool_protocol=[
                "get_customer_by_phone(phone_number=?)",
                "transfer_to_human_agents(summary=?)",
            ],
            user_texts=["No service on my phone"],
        )
        skills = distill_skills_from_trajectories([traj], min_support=1)
        self.assertEqual(skills, [])
        self.assertTrue(is_escalate_only_capability("tau2.transfer_to_human_agents"))
        self.assertTrue(
            is_read_only_protocol(
                [
                    "get_customer_by_phone(phone_number=?)",
                    "transfer_to_human_agents(summary=?)",
                ]
            )
        )
        self.assertTrue(
            is_early_escalate_protocol(
                [
                    "get_customer_by_phone(phone_number=?)",
                    "transfer_to_human_agents(summary=?)",
                ]
            )
        )
        self.assertFalse(
            trajectory_eligible_for_distill(traj, require_success=True)
        )

    def test_communicate_ok_read_protocol_can_distill(self) -> None:
        traj = _traj(
            domain="airline",
            task_id="14",
            reward=1.0,
            db_match=False,
            db_reward=None,
            tool_protocol=[
                "get_user_details(user_id=?)",
                "calculate(expression=?)",
            ],
            user_texts=["How much on gift cards and certificates?"],
            metadata={"success_mode": "communicate_ok"},
        )
        self.assertTrue(trajectory_eligible_for_distill(traj))
        skills = distill_skills_from_trajectories([traj], min_support=1)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].metadata.get("success_mode"), "communicate_ok")
        self.assertTrue(all("confirm" not in s.lower() for s in skills[0].action_protocol))

    def test_default_distill_keeps_trajectory_tool_rows(self) -> None:
        traj = _traj(
            domain="telecom",
            task_id="[mms_issue]airplane_mode_on",
            tool_protocol=[
                "get_customer_by_phone(phone_number=?)",
                "enable_roaming(line_id=?)",
            ],
            user_texts=["I need roaming"],
            metadata={"success_mode": "solve_write"},
        )
        skills = distill_skills_from_trajectories([traj], min_support=1)
        self.assertEqual(len(skills), 1)
        proto = " | ".join(skills[0].action_protocol)
        self.assertNotIn("user says yes", proto)
        self.assertNotIn("diagnose boolean flags", proto)
        self.assertIn("enable_roaming(", proto)

    def test_telecom_requires_full_success_not_db_proxy(self) -> None:
        # Failed ENV (reward 0) must not teach, even if a write appears.
        bad = _traj(
            domain="telecom",
            task_id="[mobile_data_issue]x",
            reward=0.0,
            db_reward=None,
            db_match=None,
            tool_protocol=[
                "get_customer_by_phone(phone_number=?)",
                "refuel_data(line_id=?)",
            ],
        )
        self.assertFalse(trajectory_eligible_for_distill(bad))
        self.assertEqual(
            distill_skills_from_trajectories([bad], min_support=1),
            [],
        )
        good = _traj(
            domain="telecom",
            task_id="[mobile_data_issue]x",
            reward=1.0,
            db_reward=None,
            db_match=None,
            tool_protocol=[
                "get_customer_by_phone(phone_number=?)",
                "refuel_data(line_id=?)",
            ],
        )
        self.assertTrue(trajectory_eligible_for_distill(good))
        skills = distill_skills_from_trajectories([good], min_support=1)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].capability_key, "tau2.refuel_data")

    def test_write_spine_and_task_families(self) -> None:
        traj = _traj(
            domain="telecom",
            task_id="[mms_issue]airplane_mode_on",
            tool_protocol=[
                "get_customer_by_phone(phone_number=?)",
                "enable_roaming(line_id=?)",
            ],
            user_texts=["I need roaming enabled for MMS"],
        )
        skills = distill_skills_from_trajectories([traj], min_support=1)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].capability_key, "tau2.enable_roaming")
        self.assertEqual(
            skills[0].metadata.get("task_families"),
            ["enable_roaming", "mms_issue"],
        )
        self.assertEqual(
            skills[0].metadata.get("primary_task_family"), "enable_roaming"
        )
        self.assertIn("mms", skills[0].metadata.get("intent_cues") or [])

    def test_episode_routing_scope_from_eval_criteria(self) -> None:
        task = {
            "id": "73",
            "domain": "retail",
            "evaluation_criteria": {
                "actions": [
                    {"requestor": "assistant", "name": "return_delivered_order_items"},
                ]
            },
        }
        self.assertEqual(
            episode_routing_scope(task), "return_delivered_order_items"
        )
        self.assertEqual(
            episode_routing_scope(
                {"id": "[mms_issue]x", "evaluation_criteria": {"actions": []}}
            ),
            "mms_issue",
        )


class TelecomBugBranchCoverageTests(unittest.TestCase):
    def test_train_large_tags_have_protocol_branches(self) -> None:
        from pathlib import Path

        from sage_tau2.distill import (
            _BUG_TAG_PROTOCOL_BRANCHES,
            compose_executable_protocol,
            diagnostic_branch_steps,
        )
        from sage_tau2.task_context import bug_tags_from_task_id

        root = Path(__file__).resolve().parents[2]
        split_path = root / "tau2-bench/data/tau2/domains/telecom/split_tasks.json"
        tasks_path = root / "tau2-bench/data/tau2/domains/telecom/tasks.json"
        if not split_path.exists() or not tasks_path.exists():
            self.skipTest("tau2-bench telecom data not present")
        import json

        split = json.loads(split_path.read_text())
        tasks = json.loads(tasks_path.read_text())
        train_ids = set(split.get("train") or [])
        tags: set[str] = set()
        for task in tasks:
            if task.get("id") not in train_ids:
                continue
            tags.update(bug_tags_from_task_id(str(task.get("id") or "")))
        missing = sorted(t for t in tags if t not in _BUG_TAG_PROTOCOL_BRANCHES)
        self.assertEqual(missing, [], f"missing branches for {missing}")
        branches = diagnostic_branch_steps(sorted(tags))
        self.assertGreaterEqual(len(branches), 10)
        proto = compose_executable_protocol(
            [
                "get_customer_by_phone(phone_number=?)",
                "get_details_by_id(id=?)",
                "enable_roaming(customer_id=?, line_id=?)",
            ],
            domain="telecom",
            bug_tags=[
                "data_mode_off",
                "unseat_sim_card",
                "break_apn_mms_setting",
                "user_abroad_roaming_disabled_on",
            ],
        )
        blob = " | ".join(proto)
        self.assertIn("data_mode_off", blob)
        self.assertIn("unseat_sim_card", blob)
        self.assertIn("break_apn_mms_setting", blob)
        self.assertIn("user_abroad_roaming_disabled_on", blob)
        self.assertIn("do not call toggle_data", blob)
        self.assertIn("do not call reseat_sim_card", blob)


if __name__ == "__main__":
    unittest.main()
