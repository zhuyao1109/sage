"""Smoke tests for τ² distill / admit quality patches (no LLM)."""

from __future__ import annotations

import unittest

from sage_tau2.credit import (
    initialize_credit,
    protocol_coverage,
    seed_credit_from_birth_support,
)
from sage_tau2.distill import (
    canonicalize_protocol,
    distill_skills_from_trajectories,
    normalize_protocol,
    trim_protocol_to_write_spine,
)
from sage_tau2.executor_dispatch import ExecutorDispatchConfig, ExecutorDispatcher
from sage_tau2.nominate_admit import NominateAdmitConfig, run_nominate_admit
from sage_tau2.organization import EXECUTOR_NAME, Organization, default_executor
from sage_tau2.schemas import AgentSpec, SkillStatus, Tau2Skill, Tau2Trajectory
from sage_tau2.skill_clustering import SkillClusterArchive
from sage_tau2.skill_resolve import resolve_assigned_skills


def _traj(**kwargs) -> Tau2Trajectory:
    base = dict(
        task_id="0",
        trial=0,
        domain="retail",
        reward=1.0,
        db_reward=1.0,
        communicate_reward=1.0,
        db_match=True,
        termination_reason="user_stop",
        tool_protocol=[],
        tool_steps=[],
        assistant_texts=[],
        user_texts=["I need to return items"],
    )
    base.update(kwargs)
    return Tau2Trajectory(**base)


class DistillNormalizeSmokeTests(unittest.TestCase):
    """Reason: exact protocol tuples + post-write noise exploded the skill bank."""

    def test_canonicalize_collapses_consecutive_duplicates(self) -> None:
        proto = canonicalize_protocol(
            [
                "find_user_id_by_email(email=?)",
                "get_order_details(order_id=?)",
                "get_order_details(order_id=?)",
                "get_order_details(order_id=?)",
                "return_delivered_order_items(item_ids=?, order_id=?)",
                "return_delivered_order_items(item_ids=?, order_id=?)",
                "transfer_to_human_agents(summary=?)",
            ]
        )
        names = [s.split("(", 1)[0] for s in proto]
        self.assertEqual(
            names,
            [
                "find_user_id_by_email",
                "get_order_details",
                "return_delivered_order_items",
            ],
        )

    def test_trim_drops_post_write_pollution(self) -> None:
        trimmed = trim_protocol_to_write_spine(
            [
                "get_customer_by_phone(phone_number=?)",
                "enable_roaming(line_id=?)",
                "refuel_data(gb_amount=?)",
                "run_speed_test()",
            ]
        )
        self.assertEqual(
            [s.split("(", 1)[0] for s in trimmed],
            ["get_customer_by_phone", "enable_roaming"],
        )

    def test_near_duplicate_traces_merge_into_one_skill(self) -> None:
        t1 = _traj(
            task_id="1",
            tool_protocol=[
                "find_user_id_by_email(email=?)",
                "get_order_details(order_id=?)",
                "get_order_details(order_id=?)",
                "return_delivered_order_items(item_ids=?, order_id=?)",
            ],
        )
        t2 = _traj(
            task_id="2",
            tool_protocol=[
                "find_user_id_by_email(email=?)",
                "get_order_details(order_id=?)",
                "return_delivered_order_items(item_ids=?, order_id=?)",
                "return_delivered_order_items(item_ids=?, order_id=?)",
                "transfer_to_human_agents(summary=?)",
            ],
        )
        skills = distill_skills_from_trajectories([t1, t2], min_support=2)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].support_count, 2)
        self.assertEqual(skills[0].capability_key, "tau2.return_delivered_order_items")
        agent_proto = skills[0].metadata.get("agent_tool_protocol") or []
        self.assertEqual(normalize_protocol(agent_proto), agent_proto)
        # Default distill keeps trajectory tool rows only (no expert confirm inject).
        self.assertFalse(
            any("user says yes" in s for s in skills[0].action_protocol)
        )
        # Opt-in legacy inline dialogue still available for ablations.
        with_inline = distill_skills_from_trajectories(
            [t1, t2], min_support=2, inline_dialogue_in_protocol=True
        )
        self.assertTrue(
            any("user says yes" in s for s in with_inline[0].action_protocol)
            or any("confirm with user" in s for s in with_inline[0].action_protocol)
        )


class BirthCreditSmokeTests(unittest.TestCase):
    """Reason: without birth seeding, verified-only nominate deadlocks."""

    def test_birth_seed_verifies_support_ge_2(self) -> None:
        skill = Tau2Skill(
            skill_name="return skill",
            description="r",
            precondition="r",
            action_protocol=[
                "find_user_id_by_email(email=?)",
                "return_delivered_order_items(item_ids=?)",
            ],
            expected_effect="ok",
            capability_key="tau2.return_delivered_order_items",
            status=SkillStatus.PROVISIONAL,
            support_count=2,
            evidence_ids=["e1", "e2"],
            domain="retail",
        )
        seed_credit_from_birth_support(skill)
        # n=2 is below default min_support_for_verified=5 → stay provisional
        # (birth alone no longer creates verified_low_support).
        self.assertEqual(skill.status, SkillStatus.PROVISIONAL)
        credit = skill.metadata["skill_credit"]
        self.assertEqual(credit["uses"], 2)
        self.assertGreaterEqual(float(credit["score"]), 0.60)
        self.assertTrue(skill.metadata.get("credit_seeded_from_birth"))

    def test_coverage_matches_raw_vs_normalized(self) -> None:
        skill_proto = [
            "find_user_id_by_email(email=?)",
            "get_order_details(order_id=?)",
            "return_delivered_order_items(item_ids=?)",
        ]
        raw_traj = [
            "find_user_id_by_email(email=?)",
            "get_order_details(order_id=?)",
            "get_order_details(order_id=?)",
            "return_delivered_order_items(item_ids=?)",
            "return_delivered_order_items(item_ids=?)",
        ]
        self.assertGreaterEqual(protocol_coverage(skill_proto, raw_traj), 0.99)

    def test_birth_verify_then_nominate_without_provisional_org(self) -> None:
        trajs = [
            _traj(
                task_id="1",
                tool_protocol=[
                    "find_user_id_by_email(email=?)",
                    "get_order_details(order_id=?)",
                    "return_delivered_order_items(item_ids=?)",
                ],
            ),
            _traj(
                task_id="2",
                tool_protocol=[
                    "find_user_id_by_email(email=?)",
                    "get_order_details(order_id=?)",
                    "get_order_details(order_id=?)",
                    "return_delivered_order_items(item_ids=?)",
                ],
            ),
        ]
        skills = distill_skills_from_trajectories(trajs, min_support=2)
        self.assertEqual(len(skills), 1)
        # Explicitly lower the verify bar so this smoke still exercises nominate.
        from sage_tau2.credit import CreditPolicy

        low_bar = CreditPolicy(min_support_for_verified=2)
        for skill in skills:
            initialize_credit(skill)
            seed_credit_from_birth_support(skill, policy=low_bar)
        self.assertEqual(skills[0].status, SkillStatus.VERIFIED)

        archive = SkillClusterArchive(novelty_threshold=0.35)
        archive.update(skills, segment=1)
        org = Organization()
        cfg = NominateAdmitConfig()
        cfg.nomination.allow_provisional = False
        cfg.nomination.min_cluster_support = 2
        cfg.nomination.require_min_utility = 0.50
        cfg.nomination.require_same_domain = "retail"
        result = run_nominate_admit(
            organization=org,
            archive=archive,
            skills=skills,
            config=cfg,
            spec_vs_exec_probe=None,
        )
        self.assertEqual(
            result["accepted_agents"], ["ReturnDeliveredOrderItemsSpecialist"]
        )
        self.assertEqual(org.specialists()[0].assigned_skills, [skills[0].skill_id])


class AdmitVerifySmokeTests(unittest.TestCase):
    """Reason: editor_commit was stamping VERIFIED with zero credit uses."""

    def test_editor_commit_does_not_stamp_verified(self) -> None:
        skill = Tau2Skill(
            skill_name="return delivered order items skill (n=1)",
            description="return",
            precondition="return",
            action_protocol=[
                "find_user_id_by_email(email=?)",
                "return_delivered_order_items(item_ids=?, order_id=?)",
            ],
            expected_effect="db_ok",
            capability_key="tau2.return_delivered_order_items",
            status=SkillStatus.PROVISIONAL,
            support_count=2,
            domain="retail",
        )
        initialize_credit(skill)
        archive = SkillClusterArchive()
        archive.update([skill], segment=1)
        org = Organization()
        cfg = NominateAdmitConfig()
        cfg.nomination.allow_provisional = True
        result = run_nominate_admit(
            organization=org,
            archive=archive,
            skills=[skill],
            config=cfg,
            spec_vs_exec_probe=None,
        )
        self.assertEqual(
            result["accepted_agents"], ["ReturnDeliveredOrderItemsSpecialist"]
        )
        self.assertEqual(skill.status, SkillStatus.PROVISIONAL)
        self.assertTrue(skill.metadata.get("nominate_admit_pending_credit_verify"))
        specialist = org.specialists()[0]
        self.assertEqual(specialist.assigned_skills, [skill.skill_id])

    def test_spec_pass_still_verifies(self) -> None:
        skill = Tau2Skill(
            skill_name="cancel reservation skill",
            description="cancel",
            precondition="cancel",
            action_protocol=["cancel_reservation(reservation_id=?)"],
            expected_effect="db_ok",
            capability_key="tau2.cancel_reservation",
            status=SkillStatus.PROVISIONAL,
            support_count=3,
            domain="airline",
        )
        initialize_credit(skill)
        archive = SkillClusterArchive()
        archive.update([skill], segment=1)
        org = Organization()
        cfg = NominateAdmitConfig()
        cfg.nomination.allow_provisional = True
        cfg.admission.enable_spec_vs_exec = True

        def _probe(**kwargs):
            return {
                "specialist_sr": 0.8,
                "executor_sr": 0.4,
                "specialist_wins": 4,
                "executor_wins": 2,
                "n_tasks": 5,
            }

        run_nominate_admit(
            organization=org,
            archive=archive,
            skills=[skill],
            config=cfg,
            spec_vs_exec_probe=_probe,
        )
        self.assertEqual(skill.status, SkillStatus.VERIFIED)

    def test_spec_zero_tasks_refuses(self) -> None:
        skill = Tau2Skill(
            skill_name="cancel reservation skill",
            description="cancel",
            precondition="cancel",
            action_protocol=["cancel_reservation(reservation_id=?)"],
            expected_effect="db_ok",
            capability_key="tau2.cancel_reservation",
            status=SkillStatus.VERIFIED,
            support_count=3,
            domain="airline",
        )
        initialize_credit(skill)
        archive = SkillClusterArchive()
        archive.update([skill], segment=1)
        org = Organization()
        cfg = NominateAdmitConfig()
        cfg.admission.enable_spec_vs_exec = True

        def _probe(**kwargs):
            return {
                "specialist_sr": 1.0,
                "executor_sr": 0.0,
                "specialist_wins": 0,
                "executor_wins": 0,
                "n_tasks": 0,
            }

        result = run_nominate_admit(
            organization=org,
            archive=archive,
            skills=[skill],
            config=cfg,
            spec_vs_exec_probe=_probe,
        )
        self.assertEqual(result["accepted_agents"], [])
        self.assertEqual(len(org.specialists()), 0)


class SkillIdResolveSmokeTests(unittest.TestCase):
    """Reason: colliding skill_name inflated assigned_skills / inject."""

    def test_resolve_prefers_skill_id_and_dedupes_name(self) -> None:
        a = Tau2Skill(
            skill_name="return delivered order items skill (n=1)",
            description="a",
            precondition="a",
            action_protocol=["return_delivered_order_items(item_ids=?)"],
            expected_effect="ok",
            capability_key="tau2.return_delivered_order_items",
            status=SkillStatus.VERIFIED,
            support_count=1,
        )
        b = Tau2Skill(
            skill_name="return delivered order items skill (n=1)",
            description="b",
            precondition="b",
            action_protocol=["return_delivered_order_items(item_ids=?)"],
            expected_effect="ok",
            capability_key="tau2.return_delivered_order_items",
            status=SkillStatus.PROVISIONAL,
            support_count=3,
        )
        by_id = resolve_assigned_skills([a.skill_id, a.skill_id], [a, b])
        self.assertEqual([s.skill_id for s in by_id], [a.skill_id])
        by_name = resolve_assigned_skills(
            [a.skill_name, a.skill_name, a.skill_name], [a, b]
        )
        self.assertEqual(len(by_name), 1)
        self.assertEqual(by_name[0].skill_id, a.skill_id)


class ProbationQuotaSmokeTests(unittest.TestCase):
    """Reason: evolve/eval should not let weak probation steal primary."""

    def test_quota_zero_blocks_probation_primary(self) -> None:
        skill = Tau2Skill(
            skill_name="refuel data skill (n=1)",
            description="refuel",
            precondition="data low",
            action_protocol=[
                "get_customer_by_phone(phone_number=?)",
                "refuel_data(gb_amount=?)",
            ],
            expected_effect="ok",
            capability_key="tau2.refuel_data",
            status=SkillStatus.VERIFIED,
            support_count=1,
            domain="telecom",
            metadata={
                "task_families": ["refuel_data", "mobile_data_issue"],
                "primary_task_family": "refuel_data",
            },
        )
        initialize_credit(skill)
        specialist = AgentSpec(
            name="RefuelDataSpecialist",
            role="specialist",
            responsibilities=["own"],
            assigned_skills=[skill.skill_id],
            capability_keys=["tau2.refuel_data"],
            tool_permissions=["env_action"],
            acting_status="probation",
            metadata={"dispatch_only": True},
            shadow_evaluation_record={
                "dispatch_only": True,
                "applicable_dispatched_games": 0,
                "trial_games_remaining": 3,
            },
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(
                prefer_matching_specialist=True,
                require_accepted_for_primary=True,
                probation_primary_quota=0,
            )
        )
        assignment = dispatcher.assign(
            task="Please add more mobile data",
            domain="telecom",
            agents=org.agents,
            skills=[skill],
            executor_name=EXECUTOR_NAME,
            task_id="[mobile_data_issue]x",
            episode_scope="refuel_data",
        )
        self.assertEqual(assignment.primary_agent, EXECUTOR_NAME)
        self.assertEqual(assignment.dispatch_layer, "eligibility_empty")

    def test_spec_probe_passed_still_dispatchable_with_quota_zero(self) -> None:
        skill = Tau2Skill(
            skill_name="return skill",
            description="r",
            precondition="r",
            action_protocol=[
                "find_user_id_by_email(email=?)",
                "return_delivered_order_items(item_ids=?)",
            ],
            expected_effect="ok",
            capability_key="tau2.return_delivered_order_items",
            status=SkillStatus.VERIFIED,
            support_count=2,
            domain="retail",
            metadata={
                "task_families": ["return_delivered_order_items"],
                "primary_task_family": "return_delivered_order_items",
            },
        )
        initialize_credit(skill)
        specialist = AgentSpec(
            name="ReturnDeliveredOrderItemsSpecialist",
            role="specialist",
            responsibilities=["own"],
            assigned_skills=[skill.skill_id],
            capability_keys=["tau2.return_delivered_order_items"],
            tool_permissions=["env_action"],
            acting_status="probation",
            metadata={"dispatch_only": False},
            shadow_evaluation_record={
                "dispatch_only": False,
                "promotion_probe_passed": True,
                "applicable_dispatched_games": 0,
                "trial_games_remaining": 3,
            },
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(
                prefer_matching_specialist=True,
                require_accepted_for_primary=True,
                probation_primary_quota=0,
            )
        )
        assignment = dispatcher.assign(
            task="I need to return delivered items",
            domain="retail",
            agents=org.agents,
            skills=[skill],
            executor_name=EXECUTOR_NAME,
            task_id="11",
            episode_scope="return_delivered_order_items",
        )
        self.assertEqual(
            assignment.primary_agent, "ReturnDeliveredOrderItemsSpecialist"
        )


if __name__ == "__main__":
    unittest.main()
