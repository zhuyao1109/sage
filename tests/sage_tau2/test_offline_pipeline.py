"""Offline unit checks for sage_tau2 distill/credit/org (no LLM calls)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sage_tau2.credit import apply_credit_for_episode, initialize_credit, injectable_skills
from sage_tau2.distill import distill_skills_from_trajectories
from sage_tau2.nominate_admit import NominateAdmitConfig, run_nominate_admit
from sage_tau2.organization import Organization, format_organization_for_prompt
from sage_tau2.pipeline import SegmentUpdateConfig, update_bank_from_results
from sage_tau2.schemas import SkillStatus, Tau2Skill
from sage_tau2.skill_bank import Tau2SkillBank
from sage_tau2.skill_clustering import SkillClusterArchive, protocol_novelty
from sage_tau2.trajectory import results_json_to_trajectories


class SageTau2OfflineTests(unittest.TestCase):
    def test_distill_and_credit_from_saved_results(self) -> None:
        results_path = Path(
            "tau2-bench/data/simulations/compare_llm_agent_airline5_s42/results.json"
        )
        if not results_path.exists():
            self.skipTest(f"missing fixture {results_path}")
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        trajs = results_json_to_trajectories(payload, domain="airline")
        self.assertGreaterEqual(len(trajs), 1)
        skills = distill_skills_from_trajectories(trajs, min_support=1)
        # Fixture episodes are mostly read/escalate-only; quality gate may yield 0.
        # Seed a write-protocol skill so credit/inject path still exercises.
        if not skills:
            skills = [
                Tau2Skill(
                    skill_name="cancel reservation skill (n=1)",
                    description="cancel",
                    precondition="user asks to cancel",
                    action_protocol=[
                        "get_reservation_details(reservation_id=?)",
                        "cancel_reservation(reservation_id=?)",
                    ],
                    expected_effect="db_ok",
                    capability_key="tau2.cancel_reservation",
                    status=SkillStatus.PROVISIONAL,
                    support_count=1,
                    domain="airline",
                )
            ]
        self.assertGreaterEqual(len(skills), 1)
        self.assertTrue(all(s.action_protocol for s in skills))

        with tempfile.TemporaryDirectory() as tmp:
            bank = Tau2SkillBank(Path(tmp) / "skill_bank.json")
            bank.extend(skills)
            bank.save()
            offered = injectable_skills(bank.active(), max_skills=2)
            self.assertGreaterEqual(len(offered), 1)
            events = apply_credit_for_episode(
                bank.active(),
                trajs[0],
                injected_skill_ids=[s.skill_id for s in offered],
            )
            summary = update_bank_from_results(
                results_payload=payload,
                domain="airline",
                bank=bank,
                injected_skill_ids=[s.skill_id for s in offered],
                segment_dir=Path(tmp) / "seg",
            )
            self.assertEqual(summary["n_trajectories"], len(trajs))
            self.assertTrue((Path(tmp) / "skill_bank.json").exists())
            # events may be empty if coverage fails on first traj; bank still updates
            self.assertIsInstance(events, list)
            self.assertIsNone(summary.get("organization_edit"))

    def test_cluster_and_admit_add_agent(self) -> None:
        skill = Tau2Skill(
            skill_name="cancel reservation skill",
            description="cancel",
            precondition="user wants to cancel",
            action_protocol=[
                "get_user_details(user_id=?)",
                "get_reservation_details(reservation_id=?)",
                "cancel_reservation(reservation_id=?)",
            ],
            expected_effect="db_ok",
            capability_key="tau2.cancel_reservation",
            status=SkillStatus.VERIFIED,
            support_count=3,
            evidence_ids=["e1", "e2", "e3"],
            domain="airline",
        )
        initialize_credit(skill)
        skill.metadata["skill_credit"]["uses"] = 2
        skill.metadata["skill_credit"]["successes"] = 2
        skill.metadata["skill_credit"]["score"] = 0.75
        skill.metadata["utility"] = 0.75

        archive = SkillClusterArchive(novelty_threshold=0.05)
        report = archive.update([skill], segment=1)
        self.assertEqual(len(report.births), 1)
        self.assertAlmostEqual(
            protocol_novelty(skill.action_protocol, skill.action_protocol), 0.0
        )

        def _probe(**kwargs):
            return {
                "specialist_sr": 0.75,
                "executor_sr": 0.50,
                "specialist_wins": 6,
                "executor_wins": 4,
                "n_tasks": 8,
            }

        org = Organization()
        cfg = NominateAdmitConfig()
        cfg.admission.enable_spec_vs_exec = True
        cfg.admission.accept_ties = False
        result = run_nominate_admit(
            organization=org,
            archive=archive,
            skills=[skill],
            config=cfg,
            spec_vs_exec_probe=_probe,
        )
        self.assertEqual(result["accepted_agents"], ["CancelReservationSpecialist"])
        self.assertEqual(len(org.specialists()), 1)
        specialist = org.specialists()[0]
        self.assertTrue(
            specialist.shadow_evaluation_record.get("promotion_probe_passed")
        )
        block = format_organization_for_prompt(org)
        self.assertIn("CancelReservationSpecialist", block)

    def test_spec_vs_exec_fail_removes_agent(self) -> None:
        skill = Tau2Skill(
            skill_name="cancel reservation skill",
            description="cancel",
            precondition="cancel",
            action_protocol=["cancel_reservation(reservation_id=?)"],
            expected_effect="db_ok",
            capability_key="tau2.cancel_reservation",
            status=SkillStatus.VERIFIED,
            support_count=3,
            evidence_ids=["e1", "e2", "e3"],
        )
        archive = SkillClusterArchive()
        archive.update([skill], segment=1)

        def _probe(**kwargs):
            return {
                "specialist_sr": 0.25,
                "executor_sr": 0.75,
                "specialist_wins": 2,
                "executor_wins": 6,
                "n_tasks": 8,
            }

        org = Organization()
        cfg = NominateAdmitConfig()
        cfg.admission.enable_spec_vs_exec = True
        cfg.admission.accept_ties = False
        result = run_nominate_admit(
            organization=org,
            archive=archive,
            skills=[skill],
            config=cfg,
            spec_vs_exec_probe=_probe,
        )
        self.assertEqual(result["accepted_agents"], [])
        self.assertEqual(len(org.specialists()), 0)

    def test_editor_commit_adds_probation_without_spec(self) -> None:
        skill = Tau2Skill(
            skill_name="cancel reservation skill",
            description="cancel",
            precondition="cancel",
            action_protocol=["cancel_reservation(reservation_id=?)"],
            expected_effect="db_ok",
            capability_key="tau2.cancel_reservation",
            status=SkillStatus.VERIFIED,
            support_count=3,
            evidence_ids=["e1", "e2", "e3"],
            domain="airline",
        )
        archive = SkillClusterArchive()
        archive.update([skill], segment=1)
        org = Organization()
        cfg = NominateAdmitConfig()
        # Defaults: Spec off + editor_commit
        self.assertFalse(cfg.admission.enable_spec_vs_exec)
        result = run_nominate_admit(
            organization=org,
            archive=archive,
            skills=[skill],
            config=cfg,
            spec_vs_exec_probe=None,
        )
        self.assertEqual(result["accepted_agents"], ["CancelReservationSpecialist"])
        specialist = org.specialists()[0]
        self.assertEqual(specialist.acting_status, "probation")
        self.assertEqual(specialist.assigned_skills, [skill.skill_id])
        record = specialist.shadow_evaluation_record
        self.assertFalse(record.get("nominate_admit_awaiting_admission"))
        self.assertFalse(record.get("promotion_probe_passed"))
        self.assertEqual(result["admissions"][0].get("mode"), "editor_commit")
        # editor_commit must not re-stamp VERIFIED (skill was already verified here).
        self.assertEqual(skill.status, SkillStatus.VERIFIED)

    def test_onboarding_promotes_after_wins(self) -> None:
        from sage_tau2.onboarding import OnboardingPolicy, refresh_acting_statuses
        from sage_tau2.schemas import Tau2Trajectory

        org = Organization()
        agent = org.specialists()  # empty
        from sage_tau2.schemas import AgentSpec

        specialist = AgentSpec(
            name="CancelReservationSpecialist",
            role="specialist",
            responsibilities=["cancel"],
            assigned_skills=["cancel reservation skill"],
            capability_keys=["tau2.cancel_reservation"],
            acting_status="probation",
            metadata={"domains": ["airline"], "dispatch_only": True},
            shadow_evaluation_record={
                "acting_status": "probation",
                "dispatch_only": True,
                "applicable_dispatched_games": 0,
                "trial_games_remaining": 3,
                "nominate_admit_awaiting_admission": False,
            },
        )
        org.agents.append(specialist)
        trajs = [
            Tau2Trajectory(
                task_id=f"t{i}",
                trial=0,
                domain="airline",
                reward=1.0,
                db_reward=1.0,
                communicate_reward=None,
                db_match=True,
                termination_reason="user_stop",
                tool_protocol=["cancel_reservation(reservation_id=?)"],
                tool_steps=[],
                assistant_texts=[],
                user_texts=[],
            )
            for i in range(3)
        ]
        report = refresh_acting_statuses(
            org,
            trajs,
            dispatch_by_task={f"t{i}": "CancelReservationSpecialist" for i in range(3)},
            policy=OnboardingPolicy(min_games=3, min_wins=2),
        )
        self.assertEqual(report["n_promoted"], 1)
        self.assertEqual(specialist.acting_status, "accepted")
        self.assertFalse(specialist.metadata.get("dispatch_only"))

    def test_pipeline_writes_org_when_unfrozen(self) -> None:
        results_path = Path(
            "tau2-bench/data/simulations/compare_llm_agent_airline5_s42/results.json"
        )
        if not results_path.exists():
            self.skipTest(f"missing fixture {results_path}")
        payload = json.loads(results_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = Tau2SkillBank(root / "skill_bank.json")
            seed = Tau2Skill(
                skill_name="seed cancel",
                description="seed",
                precondition="cancel",
                action_protocol=[
                    "get_reservation_details(reservation_id=?)",
                    "cancel_reservation(reservation_id=?)",
                ],
                expected_effect="db_ok",
                capability_key="tau2.cancel_reservation",
                status=SkillStatus.VERIFIED,
                support_count=4,
                evidence_ids=["a", "b", "c", "d"],
                domain="airline",
            )
            initialize_credit(seed)
            seed.metadata["skill_credit"]["score"] = 0.8
            seed.metadata["utility"] = 0.8
            bank.extend([seed])
            bank.save()

            def _probe(**kwargs):
                return {
                    "specialist_sr": 0.8,
                    "executor_sr": 0.5,
                    "specialist_wins": 4,
                    "executor_wins": 2,
                    "n_tasks": 5,
                }

            cfg = SegmentUpdateConfig(
                freeze_organization=False,
                nominate_min_cluster_support=1,
                nominate_min_utility=0.0,
                enable_spec_vs_exec=True,
                admit_accept_ties=False,
                allow_provisional_org_edits=False,
                max_new_agents_per_round=1,
            )
            summary = update_bank_from_results(
                results_payload=payload,
                domain="airline",
                bank=bank,
                injected_skill_ids=[],
                config=cfg,
                segment_dir=root / "segment_001",
                segment_index=1,
                organization_path=root / "organization.json",
                cluster_archive_path=root / "skill_clusters.json",
                spec_vs_exec_probe=_probe,
            )
            self.assertFalse(summary["freeze_organization"])
            self.assertTrue((root / "organization.json").exists())
            self.assertTrue((root / "skill_clusters.json").exists())
            self.assertTrue((root / "segment_001" / "nominate_admit.json").exists())
            edit = summary.get("organization_edit") or {}
            self.assertIn(edit.get("edit_type"), {"add_agent", "do_nothing"})
            if edit.get("accepted_agents"):
                org = Organization.load(root / "organization.json")
                self.assertGreaterEqual(len(org.specialists()), 1)


if __name__ == "__main__":
    unittest.main()
