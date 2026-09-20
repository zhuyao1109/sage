"""Record what the current tau2 prompts and learning loop actually do.

A failure means this behavior changed. Passing does not mean SAGE compliance.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from sage_tau2.admission_probe import run_spec_vs_exec_probe
from sage_tau2.contracts import local_skill_outcome
from sage_tau2.coordination import coordination_prompt
from sage_tau2.credit import CreditPolicy, seed_credit_from_birth_support
from sage_tau2.injection import render_skill_cards
from sage_tau2.onboarding import OnboardingPolicy, refresh_delegated_statuses
from sage_tau2.organization import EXECUTOR_NAME, Organization
from sage_tau2.pipeline import _load_injected_ids_by_task
from sage_tau2.prompts import EXECUTOR_INSTRUCTION, SPECIALIST_INSTRUCTION
from sage_tau2.schemas import AgentSpec, SkillStatus, Tau2Skill, Tau2Trajectory, ToolCallStep


def _skill(sid: str = "sk-roam", protocol: list[str] | None = None) -> Tau2Skill:
    return Tau2Skill(
        skill_id=sid,
        skill_name="enable roaming",
        description="learned protocol",
        precondition="line is abroad and roaming is off",
        action_protocol=protocol or ["enable_roaming(line_id=?)"],
        expected_effect="roaming enabled on the bound line",
        capability_key="telecom.roaming",
        domain="telecom",
        status=SkillStatus.VERIFIED,
        support_count=3,
    )


def _traj(**kwargs) -> Tau2Trajectory:
    base = dict(
        task_id="0",
        trial=0,
        domain="telecom",
        reward=0.0,
        db_reward=None,
        communicate_reward=None,
        db_match=None,
        termination_reason="user_stop",
        tool_protocol=[],
        tool_steps=[],
        assistant_texts=[],
        user_texts=[],
    )
    base.update(kwargs)
    return Tau2Trajectory(**base)


def test_skill_prompt_treats_protocol_as_optional_candidate():
    text, offered = render_skill_cards([_skill()])
    assert offered and "Skill sk-roam:" in text
    assert "candidates, not mandatory actions" in text
    assert "Skip inapplicable steps" not in text
    assert "Skip inapplicable steps" in EXECUTOR_INSTRUCTION
    assert "follow its protocol as written" not in EXECUTOR_INSTRUCTION
    assert "follow its protocol as written" not in SPECIALIST_INSTRUCTION


def test_coordinator_plans_one_turn_instead_of_a_full_episode_owner():
    prompt = coordination_prompt(
        context="user cannot use data abroad",
        cards="Skill sk-roam",
        agents=[
            AgentSpec(name="Executor", role="owner", responsibilities=[]),
            AgentSpec(name="RoamingSpec", role="specialist", responsibilities=["roaming"]),
        ],
        owner="Executor",
    )
    assert "Choose the next ONE turn" in prompt
    assert "returns control after its turn" in prompt
    assert "owns the full episode" not in prompt
    assert "primary Agent owns the full episode" not in prompt


def test_admission_default_compares_same_skills_not_bare_executor():
    default = inspect.signature(run_spec_vs_exec_probe).parameters["bare_executor"].default
    assert default is False


def test_birth_seed_does_not_erase_online_failures():
    skill = _skill()
    skill.support_count = 7
    skill.evidence_ids = [f"e{i}" for i in range(7)]
    skill.status = SkillStatus.PROVISIONAL
    skill.metadata["skill_credit"] = {
        "uses": 5,
        "successes": 3,
        "online_uses": 2,
        "online_successes": 0,
        "birth_uses": 3,
        "birth_successes": 3,
        "score": 4 / 7,
    }
    seed_credit_from_birth_support(skill, policy=CreditPolicy(min_support_for_verified=5))
    credit = skill.metadata["skill_credit"]
    assert credit["online_uses"] == 2
    assert credit["online_successes"] == 0
    assert credit["uses"] == 9
    assert credit["successes"] == 7
    assert credit["successes"] < credit["uses"]


def test_whole_task_win_does_not_verify_an_unexecuted_skill():
    skill = _skill()
    traj = _traj(
        reward=1.0,
        metadata={"skill_events": [{"actor": "RoamingSpec", "adopted_skill_ids": ["sk-roam"], "tool_calls": []}]},
    )
    outcome = local_skill_outcome(skill, traj, traj.metadata["skill_events"])
    assert traj.success is True
    assert outcome["success"] is None
    assert outcome["outcome"] == "local_effect_unverified"


def test_mixed_local_outcomes_are_dropped_as_unverified():
    done = _skill("sk-done", ["enable_roaming(line_id=?)"])
    pending = _skill("sk-pending", ["suspend_line(line_id=?)"])
    done.metadata["execution_contract"] = {"verification_tools": ["enable_roaming"]}
    traj = _traj(
        reward=1.0,
        tool_steps=[
            ToolCallStep(name="enable_roaming", arguments={"line_id": "L1"}, tool_call_id="c1", result_content="{}"),
        ],
        metadata={
            "skill_events": [
                {
                    "actor": "RoamingSpec",
                    "adopted_skill_ids": ["sk-done", "sk-pending"],
                    "tool_calls": [{"id": "c1"}],
                }
            ],
            "action_checks": [
                {
                    "action_match": True,
                    "action": {
                        "requestor": "assistant",
                        "name": "enable_roaming",
                        "arguments": {"line_id": "L1"},
                    },
                }
            ],
        },
    )
    org = Organization(
        [
            AgentSpec(
                name="RoamingSpec",
                role="specialist",
                responsibilities=["roaming"],
                acting_status="probation",
                shadow_evaluation_record={"same_skill_probe_passed": True, "acting_status": "probation"},
            )
        ]
    )
    decisions = refresh_delegated_statuses(
        org, [traj], [done, pending], policy=OnboardingPolicy(min_games=1, min_wins=1)
    )
    assert decisions[0]["played"] == 0
    assert decisions[0]["local_wins"] == 0
    assert decisions[0]["unverified"] == 1
    assert org.agents[1].acting_status == "probation"


def test_dispatch_journal_keys_task_id_without_domain(tmp_path: Path):
    journal = tmp_path / "dispatch_journal.jsonl"
    journal.write_text(
        "\n".join(
            [
                '{"task_id":"0","domain":"airline","injected_skill_ids":["air-1"]}',
                '{"task_id":"0","domain":"retail","injected_skill_ids":["retail-9"]}',
            ]
        ),
        encoding="utf-8",
    )
    loaded = _load_injected_ids_by_task(journal)
    assert loaded == {"0": ["retail-9"]}
    assert EXECUTOR_NAME == "Executor"
