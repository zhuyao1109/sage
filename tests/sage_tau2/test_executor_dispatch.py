"""Unit tests for τ² ExecutorDispatcher (no LLM / no tau2 runtime)."""

from __future__ import annotations

import unittest

from sage_tau2.credit import initialize_credit
from sage_tau2.executor_dispatch import (
    ExecutorDispatchConfig,
    ExecutorDispatcher,
)
from sage_tau2.organization import EXECUTOR_NAME, Organization, default_executor
from sage_tau2.schemas import AgentSpec, SkillStatus, Tau2Skill


def _skill(name: str, cap: str, protocol: list[str], *, status=SkillStatus.VERIFIED) -> Tau2Skill:
    skill = Tau2Skill(
        skill_name=name,
        description=name,
        precondition=f"user wants {cap}",
        action_protocol=protocol,
        expected_effect="db_ok",
        capability_key=cap,
        status=status,
        support_count=3,
    )
    initialize_credit(skill)
    return skill


def _admitted_specialist(name: str, skill_name: str, cap: str) -> AgentSpec:
    return AgentSpec(
        name=name,
        role="specialist",
        responsibilities=["own cap"],
        assigned_skills=[skill_name],
        capability_keys=[cap],
        tool_permissions=["env_action"],
        acting_status="probation",
        metadata={"dispatch_only": False},
        shadow_evaluation_record={
            "promotion_probe_passed": True,
            "dispatch_only": False,
            "applicable_dispatched_games": 0,
        },
    )


class ExecutorDispatchTests(unittest.TestCase):
    def test_matching_specialist_takes_primary(self) -> None:
        cancel = _skill(
            "cancel skill",
            "tau2.cancel_reservation",
            ["get_reservation_details(reservation_id=?)", "cancel_reservation(reservation_id=?)"],
        )
        specialist = _admitted_specialist(
            "CancelReservationSpecialist", cancel.skill_name, "tau2.cancel_reservation"
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(prefer_matching_specialist=True)
        )
        assignment = dispatcher.assign(
            task="Please cancel my flight reservation XEHM4B",
            domain="airline",
            agents=org.agents,
            skills=[cancel],
            executor_name=EXECUTOR_NAME,
            task_id="t1",
        )
        self.assertEqual(assignment.primary_agent, "CancelReservationSpecialist")
        self.assertIn(
            assignment.dispatch_layer,
            {
                "eligibility_single",
                "prefer_matching_specialist",
                "prefer_matching_no_backend",
            },
        )

    def test_awaiting_admission_not_dispatchable(self) -> None:
        cancel = _skill(
            "cancel skill",
            "tau2.cancel_reservation",
            ["cancel_reservation(reservation_id=?)"],
        )
        specialist = AgentSpec(
            name="CancelReservationSpecialist",
            role="specialist",
            responsibilities=["cancel"],
            assigned_skills=[cancel.skill_name],
            capability_keys=["tau2.cancel_reservation"],
            tool_permissions=["env_action"],
            acting_status="probation",
            metadata={"dispatch_only": True},
            shadow_evaluation_record={
                "nominate_admit_awaiting_admission": True,
                "dispatch_only": True,
            },
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(ExecutorDispatchConfig())
        assignment = dispatcher.assign(
            task="I want to cancel my flight reservation ABC123",
            domain="airline",
            agents=org.agents,
            skills=[cancel],
            executor_name=EXECUTOR_NAME,
        )
        self.assertEqual(assignment.primary_agent, EXECUTOR_NAME)

    def test_unrelated_task_keeps_executor(self) -> None:
        cancel = _skill(
            "cancel skill",
            "tau2.cancel_reservation",
            ["cancel_reservation(reservation_id=?)"],
        )
        specialist = _admitted_specialist(
            "CancelReservationSpecialist", cancel.skill_name, "tau2.cancel_reservation"
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(ExecutorDispatchConfig())
        assignment = dispatcher.assign(
            task="What is the weather in Paris today?",
            domain="airline",
            agents=org.agents,
            skills=[cancel],
            executor_name=EXECUTOR_NAME,
        )
        self.assertEqual(assignment.primary_agent, EXECUTOR_NAME)
        self.assertEqual(assignment.dispatch_layer, "eligibility_empty")

    def test_cross_domain_skill_keeps_executor(self) -> None:
        """Airline cancel specialist must not own telecom episodes."""
        cancel = _skill(
            "cancel skill",
            "tau2.cancel_reservation",
            [
                # Polluted protocol (real distill artifact): status token used to
                # falsely match telecom assert_*_status text.
                "get_flight_status(date=?, flight_number=?)",
                "cancel_reservation(reservation_id=?)",
                "calculate(expression=?)",
            ],
        )
        cancel.domain = "airline"
        specialist = _admitted_specialist(
            "CancelReservationSpecialist", cancel.skill_name, "tau2.cancel_reservation"
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(ExecutorDispatchConfig())
        assignment = dispatcher.assign(
            task=(
                "[mobile_data_issue]airplane_mode_on|data_usage_exceeded "
                "assert_mobile_data_status expected_status=True"
            ),
            domain="telecom",
            agents=org.agents,
            skills=[cancel],
            executor_name=EXECUTOR_NAME,
            task_id="telecom_1",
        )
        self.assertEqual(assignment.primary_agent, EXECUTOR_NAME)
        self.assertEqual(assignment.dispatch_layer, "eligibility_empty")
        self.assertEqual(assignment.eligible_agents, [])

    def test_same_domain_cancel_still_matches(self) -> None:
        cancel = _skill(
            "cancel skill",
            "tau2.cancel_reservation",
            [
                "get_flight_status(date=?, flight_number=?)",
                "cancel_reservation(reservation_id=?)",
            ],
        )
        cancel.domain = "airline"
        specialist = _admitted_specialist(
            "CancelReservationSpecialist", cancel.skill_name, "tau2.cancel_reservation"
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(prefer_matching_specialist=True)
        )
        assignment = dispatcher.assign(
            task="Please cancel my flight reservation XEHM4B",
            domain="airline",
            agents=org.agents,
            skills=[cancel],
            executor_name=EXECUTOR_NAME,
        )
        self.assertEqual(assignment.primary_agent, "CancelReservationSpecialist")

    def test_telecom_scope_match_like_alfworld_family(self) -> None:
        skill = _skill(
            "roaming skill",
            "tau2.enable_roaming",
            ["get_customer_by_phone(phone_number=?)", "enable_roaming(line_id=?)"],
        )
        skill.domain = "telecom"
        skill.metadata["task_families"] = ["mms_issue", "enable_roaming"]
        skill.metadata["primary_task_family"] = "enable_roaming"
        specialist = _admitted_specialist(
            "EnableRoamingSpecialist", skill.skill_name, "tau2.enable_roaming"
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(ExecutorDispatchConfig())
        hit = dispatcher.assign(
            task="I cannot send MMS abroad; please enable roaming on my line",
            domain="telecom",
            agents=org.agents,
            skills=[skill],
            executor_name=EXECUTOR_NAME,
            task_id="[mms_issue]airplane_mode_on|data_usage_exceeded",
            episode_scope="enable_roaming",
        )
        miss_scope = dispatcher.assign(
            task="I cannot send MMS messages",
            domain="telecom",
            agents=org.agents,
            skills=[skill],
            executor_name=EXECUTOR_NAME,
            task_id="[service_issue]overdue_bill_suspension",
        )
        miss_capability = dispatcher.assign(
            task="I cannot send MMS messages",
            domain="telecom",
            agents=org.agents,
            skills=[skill],
            executor_name=EXECUTOR_NAME,
            task_id="[mms_issue]airplane_mode_on|data_usage_exceeded",
        )
        self.assertEqual(hit.primary_agent, "EnableRoamingSpecialist")
        self.assertEqual(miss_scope.primary_agent, EXECUTOR_NAME)
        self.assertEqual(miss_capability.primary_agent, EXECUTOR_NAME)

    def test_return_specialist_not_on_cancel_retail(self) -> None:
        ret = _skill(
            "return skill",
            "tau2.return_delivered_order_items",
            [
                "find_user_id_by_email(email=?)",
                "return_delivered_order_items(item_ids=?, order_id=?, payment_method_id=?)",
            ],
        )
        ret.domain = "retail"
        ret.metadata["task_families"] = ["return_delivered_order_items"]
        ret.metadata["primary_task_family"] = "return_delivered_order_items"
        specialist = _admitted_specialist(
            "ReturnDeliveredOrderItemsSpecialist",
            ret.skill_name,
            "tau2.return_delivered_order_items",
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(prefer_matching_specialist=True)
        )
        assignment = dispatcher.assign(
            task="Please cancel my pending order #12345",
            domain="retail",
            agents=org.agents,
            skills=[ret],
            executor_name=EXECUTOR_NAME,
            task_id="99",
            episode_scope="cancel_pending_order",
        )
        self.assertEqual(assignment.primary_agent, EXECUTOR_NAME)

    def test_return_specialist_on_return_retail(self) -> None:
        ret = _skill(
            "return skill",
            "tau2.return_delivered_order_items",
            [
                "find_user_id_by_email(email=?)",
                "return_delivered_order_items(item_ids=?, order_id=?, payment_method_id=?)",
            ],
        )
        ret.domain = "retail"
        ret.metadata["task_families"] = ["return_delivered_order_items"]
        ret.metadata["primary_task_family"] = "return_delivered_order_items"
        specialist = _admitted_specialist(
            "ReturnDeliveredOrderItemsSpecialist",
            ret.skill_name,
            "tau2.return_delivered_order_items",
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(prefer_matching_specialist=True)
        )
        assignment = dispatcher.assign(
            task="I need to return delivered items from my order",
            domain="retail",
            agents=org.agents,
            skills=[ret],
            executor_name=EXECUTOR_NAME,
            task_id="11",
            episode_scope="return_delivered_order_items",
        )
        self.assertEqual(assignment.primary_agent, "ReturnDeliveredOrderItemsSpecialist")

    def test_return_specialist_not_on_exchange_retail(self) -> None:
        ret = _skill(
            "return skill",
            "tau2.return_delivered_order_items",
            [
                "find_user_id_by_email(email=?)",
                "return_delivered_order_items(item_ids=?, order_id=?, payment_method_id=?)",
            ],
        )
        ret.domain = "retail"
        ret.metadata["task_families"] = ["return_delivered_order_items"]
        ret.metadata["primary_task_family"] = "return_delivered_order_items"
        specialist = _admitted_specialist(
            "ReturnDeliveredOrderItemsSpecialist",
            ret.skill_name,
            "tau2.return_delivered_order_items",
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(prefer_matching_specialist=True)
        )
        assignment = dispatcher.assign(
            task="I want to exchange delivered items for new sizes",
            domain="retail",
            agents=org.agents,
            skills=[ret],
            executor_name=EXECUTOR_NAME,
            task_id="88",
            episode_scope="exchange_delivered_order_items",
        )
        self.assertEqual(assignment.primary_agent, EXECUTOR_NAME)

    def test_phone_token_alone_does_not_own_telecom(self) -> None:
        skill = _skill(
            "phone skill",
            "tau2.get_customer_by_phone",
            ["get_customer_by_phone(phone_number=?)"],
        )
        skill.domain = "telecom"
        skill.metadata["primary_task_family"] = "get_customer_by_phone"
        specialist = _admitted_specialist(
            "GetCustomerByPhoneSpecialist", skill.skill_name, "tau2.get_customer_by_phone"
        )
        org = Organization([default_executor(), specialist])
        dispatcher = ExecutorDispatcher(ExecutorDispatchConfig())
        assignment = dispatcher.assign(
            task="My mobile data is slow. My phone number is 555-123-2002.",
            domain="telecom",
            agents=org.agents,
            skills=[skill],
            executor_name=EXECUTOR_NAME,
            task_id="[mobile_data_issue]data_saver_mode_on",
            episode_scope="mobile_data_issue",
        )
        self.assertEqual(assignment.primary_agent, EXECUTOR_NAME)

    def test_accepted_preferred_over_probation_when_both_match(self) -> None:
        skill_a = _skill(
            "email a",
            "tau2.find_user_id_by_email",
            ["find_user_id_by_email(email=?)"],
        )
        skill_a.domain = "retail"
        skill_b = _skill(
            "email b",
            "tau2.find_user_id_by_email_get_order_details",
            ["find_user_id_by_email(email=?)", "get_order_details(order_id=?)"],
        )
        skill_b.domain = "retail"
        probation = _admitted_specialist(
            "FindUserIdProbation", skill_a.skill_name, "tau2.find_user_id_by_email"
        )
        accepted = AgentSpec(
            name="FindUserIdAccepted",
            role="specialist",
            responsibilities=["own"],
            assigned_skills=[skill_b.skill_name],
            capability_keys=["tau2.find_user_id_by_email_get_order_details"],
            tool_permissions=["env_action"],
            acting_status="accepted",
            metadata={"nominated_utility": 0.9, "domains": ["retail"]},
            shadow_evaluation_record={"applicable_dispatched_games": 0},
        )
        # Put probation first in org order; ranking should still prefer accepted.
        org = Organization([default_executor(), probation, accepted])
        dispatcher = ExecutorDispatcher(
            ExecutorDispatchConfig(prefer_matching_specialist=True)
        )
        assignment = dispatcher.assign(
            task="I want to return an item. My email is a@b.com",
            domain="retail",
            agents=org.agents,
            skills=[skill_a, skill_b],
            executor_name=EXECUTOR_NAME,
            task_id="73",
        )
        self.assertEqual(assignment.primary_agent, "FindUserIdAccepted")


if __name__ == "__main__":
    unittest.main()
