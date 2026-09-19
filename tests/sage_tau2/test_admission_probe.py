"""Tests for capability-scoped Spec-vs-Exec admission probe sampling."""

from __future__ import annotations

import unittest

from sage_tau2.admission_probe import (
    paired_probe_scores,
    probe_scopes_for_specialist,
    select_capability_probe_task_ids,
)
from sage_tau2.schemas import AgentSpec, Tau2Skill
from sage_tau2.task_context import task_matches_probe_scope


def _sim(task_id: str, reward: float, termination: str = "user_stop") -> dict:
    return {
        "task_id": task_id,
        "reward_info": {"reward": reward},
        "termination_reason": termination,
    }


class PairedProbeScoreTests(unittest.TestCase):
    def test_drops_infra_and_scores_clean_pair(self) -> None:
        spec = {
            "simulations": [
                _sim("a", 1.0),
                _sim("b", 0.0, "infrastructure_error"),
                _sim("c", 1.0),
            ]
        }
        exe = {
            "simulations": [
                _sim("a", 0.0),
                _sim("b", 1.0),
                _sim("c", 1.0),
            ]
        }
        out = paired_probe_scores(spec, exe)
        self.assertEqual(out["n_tasks"], 2)
        self.assertEqual(out["scored_task_ids"], ["a", "c"])
        self.assertEqual(out["dropped_infra_task_ids"], ["b"])
        self.assertEqual(out["specialist_wins"], 2)
        self.assertEqual(out["executor_wins"], 1)

    def test_either_side_infra_drops_task(self) -> None:
        spec = {"simulations": [_sim("x", 1.0), _sim("y", 1.0)]}
        exe = {
            "simulations": [
                _sim("x", 0.0, "infrastructure_error"),
                _sim("y", 0.0),
            ]
        }
        out = paired_probe_scores(spec, exe)
        self.assertEqual(out["n_tasks"], 1)
        self.assertEqual(out["scored_task_ids"], ["y"])
        self.assertEqual(out["specialist_wins"], 1)
        self.assertEqual(out["executor_wins"], 0)


def _retail_return_task(task_id: str) -> dict:
    return {
        "id": task_id,
        "domain": "retail",
        "evaluation_criteria": {
            "actions": [
                {"requestor": "assistant", "name": "return_delivered_order_items"},
            ]
        },
    }


def _retail_cancel_task(task_id: str) -> dict:
    return {
        "id": task_id,
        "domain": "retail",
        "evaluation_criteria": {
            "actions": [
                {"requestor": "assistant", "name": "cancel_pending_order"},
            ]
        },
    }


class AdmissionProbeScopeTests(unittest.TestCase):
    def test_task_matches_probe_scope(self) -> None:
        task = _retail_return_task("73")
        self.assertTrue(
            task_matches_probe_scope(task, {"return_delivered_order_items"})
        )
        self.assertFalse(task_matches_probe_scope(task, {"cancel_pending_order"}))

    def test_probe_scopes_from_specialist_capability_keys(self) -> None:
        agent = AgentSpec(
            name="ReturnSpecialist",
            role="specialist",
            responsibilities=["return"],
            capability_keys=["tau2.return_delivered_order_items"],
        )
        self.assertEqual(
            probe_scopes_for_specialist(agent, []),
            ["return_delivered_order_items"],
        )

    def test_select_capability_probe_task_ids_filters_pool(self) -> None:
        agent = AgentSpec(
            name="ReturnSpecialist",
            role="specialist",
            responsibilities=["return"],
            capability_keys=["tau2.return_delivered_order_items"],
        )
        fake_tasks = [
            _retail_return_task("r1"),
            _retail_return_task("r2"),
            _retail_return_task("r3"),
            _retail_cancel_task("c1"),
            _retail_cancel_task("c2"),
        ]

        def fake_load_tasks(*, task_set_name: str, task_split_name: str | None = None):
            self.assertEqual(task_set_name, "retail")
            self.assertEqual(task_split_name, "train")
            return fake_tasks

        ids, meta = select_capability_probe_task_ids(
            domain="retail",
            specialist=agent,
            skills=[],
            num_tasks=2,
            seed=7,
            split_name="train",
            task_loader=fake_load_tasks,
        )

        self.assertEqual(meta["pool_size"], 3)
        self.assertEqual(len(ids), 2)
        self.assertTrue(set(ids).issubset({"r1", "r2", "r3"}))
        self.assertEqual(meta["capability_scopes"], ["return_delivered_order_items"])

    def test_select_probe_empty_when_no_matching_pool(self) -> None:
        agent = AgentSpec(
            name="ReturnSpecialist",
            role="specialist",
            responsibilities=["return"],
            capability_keys=["tau2.return_delivered_order_items"],
        )

        def fake_load_tasks(*, task_set_name: str, task_split_name: str | None = None):
            return [_retail_cancel_task("c1")]

        ids, meta = select_capability_probe_task_ids(
            domain="retail",
            specialist=agent,
            skills=[],
            num_tasks=6,
            seed=0,
            task_loader=fake_load_tasks,
        )

        self.assertEqual(ids, [])
        self.assertEqual(meta["pool_size"], 0)
        self.assertIn("no train tasks", str(meta.get("reason")))

    def test_probe_scopes_prefer_write_not_task_families(self) -> None:
        skill = Tau2Skill(
            skill_name="roaming skill",
            description="d",
            precondition="p",
            action_protocol=["enable_roaming(customer_id=?, line_id=?)"],
            expected_effect="e",
            capability_key="tau2.enable_roaming",
            skill_id="skill-1",
            metadata={
                "write_capability_key": "tau2.enable_roaming",
                "primary_task_family": "mms_issue",
                "task_families": ["mms_issue", "mobile_data_issue", "enable_roaming"],
            },
        )
        agent = AgentSpec(
            name="EnableRoamingSpecialist",
            role="specialist",
            responsibilities=["roaming"],
            capability_keys=["tau2.enable_roaming"],
            assigned_skills=["skill-1"],
        )
        scopes = probe_scopes_for_specialist(agent, [skill])
        self.assertEqual(scopes, ["enable_roaming"])
        self.assertNotIn("mms_issue", scopes)
        self.assertNotIn("mobile_data_issue", scopes)

    def test_select_probe_uses_skill_match_subset(self) -> None:
        skill = Tau2Skill(
            skill_name="roaming skill",
            description="d",
            precondition="p",
            action_protocol=["enable_roaming(customer_id=?, line_id=?)"],
            expected_effect="e",
            capability_key="tau2.enable_roaming",
            skill_id="skill-roam",
            domain="telecom",
            metadata={
                "write_capability_key": "tau2.enable_roaming",
                "bug_bucket_mode": "union_with_signature_gate",
                "activation_signatures": [
                    ["user_abroad_roaming_enabled_off", "airplane_mode_on"]
                ],
                "bug_intersection": ["user_abroad_roaming_enabled_off"],
                "bug_union": [
                    "user_abroad_roaming_enabled_off",
                    "airplane_mode_on",
                    "data_mode_off",
                ],
            },
        )
        agent = AgentSpec(
            name="EnableRoamingSpecialist",
            role="specialist",
            responsibilities=["roaming"],
            capability_keys=["tau2.enable_roaming"],
            assigned_skills=["skill-roam"],
        )
        fake_tasks = [
            {
                "id": (
                    "[mobile_data_issue]airplane_mode_on|"
                    "user_abroad_roaming_enabled_off[PERSONA:Easy]"
                )
            },
            {
                "id": (
                    "[mms_issue]airplane_mode_on|data_mode_off|"
                    "user_abroad_roaming_disabled_on[PERSONA:None]"
                )
            },
            {"id": "[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]"},
        ]

        def fake_load_tasks(*, task_set_name: str, task_split_name: str | None = None):
            return fake_tasks

        ids, meta = select_capability_probe_task_ids(
            domain="telecom",
            specialist=agent,
            skills=[skill],
            num_tasks=6,
            seed=0,
            split_name="train",
            task_loader=fake_load_tasks,
        )
        self.assertEqual(meta["reason"], "skill_match_sample")
        # Roaming tags stay exact: only the enabled_off intersection hit is kept.
        self.assertEqual(meta["pool_size"], 1)
        self.assertIn(
            "[mobile_data_issue]airplane_mode_on|"
            "user_abroad_roaming_enabled_off[PERSONA:Easy]",
            ids,
        )
        self.assertNotIn(
            "[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]",
            ids,
        )
        self.assertNotIn(
            "[mms_issue]airplane_mode_on|data_mode_off|"
            "user_abroad_roaming_disabled_on[PERSONA:None]",
            ids,
        )


if __name__ == "__main__":
    unittest.main()
