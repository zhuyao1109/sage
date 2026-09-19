"""Unit tests for nominate-then-admit (no environment / no LLM).

Locks the gate boundary:
  - injection MU must NOT nominate ADD by itself
  - uncovered / shift MAY nominate ADD from provisional form_ok skills
  - Spec vs Exec PASS verifies skills; FAIL leaves them provisional
"""

from __future__ import annotations

import unittest

from sage_mas.actor_promotion import ActorPromotionResult
from sage_mas.nominate_admit import (
    AdmissionPolicy,
    NominateAdmitConfig,
    NominateAdmitController,
    NominationPolicy,
    admission_would_pass,
    skill_is_nomination_eligible,
)
from sage_mas.onboarding import acting_status
from sage_mas.schemas import (
    AgentSpec,
    AtomicOp,
    DistributionShift,
    OrganizationEditType,
    Skill,
    SkillStatus,
)


def _executor() -> AgentSpec:
    return AgentSpec(
        name="Executor",
        role="Environment executor",
        responsibilities=["Execute actions"],
        tool_permissions=["alfworld_action"],
        shadow_evaluation_record={"acting_status": "accepted"},
    )


def _clean_skill(
    *,
    status: SkillStatus = SkillStatus.PROVISIONAL,
    mu: float | None = 0.9,
    evidence_ids: list[str] | None = None,
) -> Skill:
    """Provisional form_ok clean skill with optional high injection MU.

    High MU must never be sufficient evidence for ADD on this path.
    """
    return Skill(
        skill_name="Clean Object and Place in Receptacle",
        description="clean then place",
        precondition="need clean object",
        action_protocol=[
            "go to <location>",
            "take <object> from <receptacle>",
            "clean <object> with <tool>",
            "put <object> in/on <receptacle>",
        ],
        applicable_atomic_ops=[AtomicOp.ACT],
        suggested_role="TransformCleanSpecialist",
        status=status,
        capability_key="transform.clean",
        applicable_task_families=["pick_clean_then_place_in_recep"],
        expected_effect="Object is cleaned and placed.",
        evidence_ids=evidence_ids or ["t1", "t2"],
        support_count=2,
        marginal_utility=mu,
        distribution_shift=DistributionShift(
            kl_divergence=0.5,
            shift_score=0.5,
            baseline_source="skill_bank",
        ),
        metadata={
            "primary_task_family": "pick_clean_then_place_in_recep",
            "protocol_form_ok": True,
            "required_tools": ["alfworld_action"],
            # Injection MU probe residue — must be ignored for nomination.
            "inject_ready": True if mu is not None and mu > 0 else False,
            "paired_mu": mu,
            "executable_protocol_source": "trajectory",
            "skill_credit": {
                "uses": 0,
                "successes": 0,
                "score": 0.5,
                "mean_protocol_adherence": 0.0,
                "events": [],
            },
        },
    )


class NominateAdmitTests(unittest.TestCase):
    def test_eligibility_ignores_injection_mu(self):
        high_mu = _clean_skill(mu=0.99)
        no_mu = _clean_skill(mu=None)
        no_mu.metadata["inject_ready"] = False
        rejected = _clean_skill()
        rejected.status = SkillStatus.REJECTED

        self.assertTrue(skill_is_nomination_eligible(high_mu))
        self.assertTrue(skill_is_nomination_eligible(no_mu))
        self.assertFalse(skill_is_nomination_eligible(rejected))

    def test_uncovered_provisional_nominates_add_without_mu(self):
        """Uncovered capability nominates ADD even with MU=None / flat."""
        skill = _clean_skill(mu=None)
        skill.metadata["inject_ready"] = False
        skill.metadata.pop("paired_mu", None)

        controller = NominateAdmitController(
            NominateAdmitConfig(
                nomination=NominationPolicy(
                    min_cluster_support=1,
                    min_assignment_gap=0.0,
                    add_when_uncovered_capability=True,
                    require_executable_protocol=True,
                    require_positive_mu_for_add_agent=False,
                )
            )
        )
        agents = [_executor()]
        edits = controller.nominate(agents=agents, skills=[skill], bank_before=[])
        self.assertTrue(edits)
        add_edits = [
            edit
            for edit in edits
            if edit.edit_type == OrganizationEditType.ADD_AGENT
        ]
        self.assertEqual(len(add_edits), 1, msg=[e.rationale for e in edits])
        self.assertIn("nominate_admit:nomination", add_edits[0].rationale)
        # Real skill must still be provisional after nominate().
        self.assertEqual(skill.status, SkillStatus.PROVISIONAL)

    def test_positive_injection_mu_alone_does_not_force_add(self):
        """Covered cluster + no shift → MU cannot force ADD_AGENT."""
        # Historical bank already has the same capability with a specialist
        # carrier, so uncovered=False and shift vs self should be low.
        historical = _clean_skill(status=SkillStatus.VERIFIED, mu=0.1)
        historical.skill_name = "Historical clean"
        historical.skill_id = "hist-clean"
        historical.evidence_ids = ["h1", "h2"]

        specialist = AgentSpec(
            name="TransformCleanSpecialist",
            role="TransformCleanSpecialist",
            responsibilities=["Handle clean tasks"],
            assigned_skills=["Historical clean"],
            tool_permissions=["alfworld_action"],
            activation_condition="Capability transform.clean",
            shadow_evaluation_record={
                "acting_status": "accepted",
                "capability_contract": {
                    "name": "transform.clean",
                    "task_families": ["pick_clean_then_place_in_recep"],
                },
            },
        )
        # New skill: huge injection MU, but same cluster already has a carrier.
        # With prefer_assign_before_add and no novelty, expect ASSIGN or NOTHING,
        # never ADD driven by MU.
        skill = _clean_skill(mu=0.95)
        skill.distribution_shift = DistributionShift(
            kl_divergence=0.0,
            shift_score=0.0,
            novelty_score=0.0,
            baseline_source="skill_bank",
        )

        controller = NominateAdmitController(
            NominateAdmitConfig(
                nomination=NominationPolicy(
                    min_cluster_support=1,
                    distribution_shift_threshold=0.15,
                    add_when_uncovered_capability=True,
                    prefer_assign_before_add=True,
                    require_executable_protocol=True,
                )
            )
        )
        edits = controller.nominate(
            agents=[_executor(), specialist],
            skills=[skill],
            bank_before=[historical],
        )
        for edit in edits:
            self.assertNotEqual(
                edit.edit_type,
                OrganizationEditType.ADD_AGENT,
                msg=(
                    "Positive injection MU must not nominate ADD when the "
                    f"cluster is already covered: {edit.rationale}"
                ),
            )

    def test_commit_keeps_skills_provisional(self):
        skill = _clean_skill(mu=None)
        controller = NominateAdmitController(
            NominateAdmitConfig(
                nomination=NominationPolicy(
                    min_cluster_support=1,
                    min_assignment_gap=0.0,
                    add_when_uncovered_capability=True,
                )
            )
        )
        agents = [_executor()]
        edits = controller.nominate(agents=agents, skills=[skill], bank_before=[])
        agents2, committed = controller.commit_probation_add(agents, edits, [skill])
        self.assertEqual(len(committed), 1)
        self.assertEqual(skill.status, SkillStatus.PROVISIONAL)
        self.assertTrue(skill.metadata.get("nominate_admit_awaiting_admission"))
        new_agent = next(a for a in agents2 if a.name == committed[0])
        self.assertEqual(acting_status(new_agent), "probation")
        self.assertTrue(new_agent.shadow_evaluation_record.get("dispatch_only"))

    def test_admission_pass_verifies_skills(self):
        skill = _clean_skill(mu=None)
        controller = NominateAdmitController(
            NominateAdmitConfig(
                nomination=NominationPolicy(
                    min_cluster_support=1,
                    min_assignment_gap=0.0,
                ),
                admission=AdmissionPolicy(
                    min_advantage=0.0,
                    on_pass_acting_status="probation",
                    verify_skills_on_pass=True,
                ),
            )
        )
        agents = [_executor()]
        edits = controller.nominate(agents=agents, skills=[skill], bank_before=[])
        agents, committed = controller.commit_probation_add(agents, edits, [skill])
        name = committed[0]

        result = ActorPromotionResult(
            agent_name=name,
            n_tasks=8,
            specialist_wins=4,
            executor_wins=2,
            specialist_sr=0.5,
            executor_sr=0.25,
            accepted=True,
            reason="promote: synthetic",
            gamefiles=[],
        )
        self.assertTrue(
            admission_would_pass(0.5, 0.25, min_advantage=0.0, accept_ties=True)
        )
        # Cold-start: ties pass (ΔSR>=0).
        self.assertTrue(
            admission_would_pass(0.5, 0.5, min_advantage=0.0, accept_ties=True)
        )
        # Strict online-style rule still rejects ties.
        self.assertFalse(
            admission_would_pass(0.5, 0.5, min_advantage=0.0, accept_ties=False)
        )
        decision = controller.apply_admission(
            agents=agents,
            skills=[skill],
            specialist_name=name,
            result=result,
            related_skill_names=[skill.skill_name],
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(skill.status, SkillStatus.VERIFIED)
        self.assertEqual(
            skill.metadata.get("nominate_admit_verified_by"),
            "spec_vs_exec",
        )
        self.assertIn(skill.skill_name, decision.verified_skill_names)
        specialist = next(a for a in agents if a.name == name)
        self.assertEqual(acting_status(specialist), "probation")

    def test_admission_fail_keeps_provisional_and_removes(self):
        skill = _clean_skill(mu=0.8)  # MU still irrelevant on fail path
        controller = NominateAdmitController(
            NominateAdmitConfig(
                nomination=NominationPolicy(
                    min_cluster_support=1,
                    min_assignment_gap=0.0,
                ),
                admission=AdmissionPolicy(on_fail_action="remove"),
            )
        )
        agents = [_executor()]
        edits = controller.nominate(agents=agents, skills=[skill], bank_before=[])
        agents, committed = controller.commit_probation_add(agents, edits, [skill])
        name = committed[0]

        result = ActorPromotionResult(
            agent_name=name,
            n_tasks=8,
            specialist_wins=1,
            executor_wins=3,
            specialist_sr=0.125,
            executor_sr=0.375,
            accepted=False,
            reason="keep Executor primary: synthetic",
            gamefiles=[],
        )
        self.assertFalse(
            admission_would_pass(0.125, 0.375, min_advantage=0.0, accept_ties=True)
        )
        decision = controller.apply_admission(
            agents=agents,
            skills=[skill],
            specialist_name=name,
            result=result,
            related_skill_names=[skill.skill_name],
        )
        self.assertFalse(decision.accepted)
        self.assertTrue(decision.removed)
        self.assertEqual(skill.status, SkillStatus.PROVISIONAL)
        self.assertEqual(decision.verified_skill_names, [])
        self.assertFalse(any(a.name == name for a in agents))
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0].name, "Executor")


if __name__ == "__main__":
    unittest.main()
