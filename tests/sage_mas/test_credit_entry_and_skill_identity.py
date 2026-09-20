"""Regression coverage for on-demand credit entry and persisted skill identity."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from sage_mas.alfworld_evaluator import EvaluationTrial
from sage_mas import executable_protocol
from sage_mas.executable_protocol import protocol_adherence_score
from sage_mas.pipeline import SageEvolutionPipeline
from sage_mas.schemas import AtomicOp, Skill, SkillStatus
from sage_mas.skill_bank import SkillBank
from sage_mas.skill_credit import SkillCreditPolicy, initialize_skill_credit, update_skill_credits
from sage_mas.skill_recall import dedup_recallable_skills


def heat_skill():
    return Skill(skill_name="learned heat", description="observed procedure", precondition="observed need",
        action_protocol=["go to <source>", "take <target> from <source>", "go to <tool>",
                         "open <tool>", "heat <target> with <tool>", "go to <destination>",
                         "move <target> to <destination>"],
        applicable_atomic_ops=[AtomicOp.ACT], capability_key="transform.heat",
        status=SkillStatus.PROVISIONAL)


def successful_steps():
    actions = ["go to countertop 1", "take egg 1 from countertop 1", "go to microwave 1",
               "open microwave 1", "heat egg 1 with microwave 1", "go to countertop 2",
               "move egg 1 to countertop 2"]
    observations = ["You arrive at countertop 1.", "You pick up the egg 1 from countertop 1.",
                    "You arrive at microwave 1.", "You open the microwave 1.",
                    "You heat the egg 1 using microwave 1.", "You arrive at countertop 2.",
                    "You move the egg 1 to countertop 2."]
    return [dict(action=a, observation=o, is_action_valid=True) for a, o in zip(actions, observations)]


def trial(skill, steps, activation):
    return EvaluationTrial(task_id="entry-test", task="put a hot egg on countertop",
        task_family="pick_heat_then_place_in_recep", condition="offline", reward=1.0,
        cost=0, won=True, num_steps=len(steps), steps=steps,
        activated_skill_names=[skill.skill_name], skill_activation_steps={skill.skill_name: activation},
        assigned_primary_agent="Executor", actions_by_agent={"Executor":len(steps)})


def apply(skill, steps, activation, **overrides):
    policy = SkillCreditPolicy(**overrides)
    initialize_skill_credit(skill, policy)
    return update_skill_credits([skill], [trial(skill, steps, activation)], [], policy)


@pytest.mark.parametrize("activation", [1, 4, 5])
def test_mid_task_retrieval_scores_only_remaining_work_and_promotes(activation):
    skill = heat_skill()
    steps = successful_steps()
    assert protocol_adherence_score(skill, steps, activation_step=activation) == 1
    result = apply(skill, steps, activation)
    assert result["promoted"] == [skill]
    credit = skill.metadata["skill_credit"]
    assert credit["uses"] == 1 and credit["successes"] == 1
    assert credit["events"][0]["entry_step_index"] == activation - 1


@pytest.mark.parametrize("activation", [6, 8, 20])
def test_prior_transform_or_completed_protocol_cannot_earn_new_credit(activation):
    skill = heat_skill()
    result = apply(skill, successful_steps(), activation)
    assert result["promoted"] == []
    assert skill.metadata["skill_credit"]["uses"] == 0
    if activation >= 8:
        assert protocol_adherence_score(skill, successful_steps(), activation_step=activation) == 0


def test_failed_prefix_is_not_assumed_complete_even_when_episode_wins():
    skill = heat_skill()
    steps = successful_steps()
    steps[1]["observation"] = "Nothing happens."
    # The wrapper's optimistic flag must not override a failed result.
    assert len(executable_protocol.protocol_after_activation(skill, steps, activation_step=4)) == 6
    apply(skill, steps, 4)
    assert skill.metadata["skill_credit"]["uses"] == 0


def test_unobserved_prefix_does_not_skip_prerequisites():
    skill = heat_skill()
    steps = successful_steps()
    for row in steps[:3]:
        row.pop("observation")
        row.pop("is_action_valid")
    assert len(executable_protocol.protocol_after_activation(skill, steps, activation_step=4)) == 7
    assert protocol_adherence_score(skill, steps, activation_step=4) < .34


def test_confirmed_observations_override_noisy_invalid_flags():
    skill = heat_skill()
    steps = successful_steps()
    for row in steps[:3]:
        row["is_action_valid"] = False
    assert protocol_adherence_score(skill, steps, activation_step=4) == 1


def test_partial_suffix_does_not_get_full_adherence():
    skill = heat_skill()
    # Entry consumes three steps; only open+heat of four remaining steps occurs.
    assert protocol_adherence_score(skill, successful_steps()[:5], activation_step=4) == .5


def test_productive_coverage_also_uses_remaining_protocol():
    skill = heat_skill()
    skill.capability_key = "track.place"
    skill.action_protocol = ["take <target> from <source>", "move <target> to <destination>"]
    steps = [successful_steps()[1], successful_steps()[-1]]
    result = apply(skill, steps, 2, min_protocol_coverage=1.0)
    assert result["promoted"] == [skill]


def org_eligible(candidates):
    # Use the actual pipeline gate, without constructing model/environment backends.
    config = SimpleNamespace(require_inject_ready_for_org=False,
        injection_block_prefixes=[], injection_allow_prefixes=None,
        require_positive_mu_for_org=False, min_marginal_utility=0)
    return SageEvolutionPipeline._org_eligible_skills(config, candidates)


@pytest.mark.parametrize("activation", [1, 4])
def test_promoted_variant_keeps_identity_through_bank_reload_and_org_gate(tmp_path, activation):
    bank = SkillBank(tmp_path / "bank.json", disable_merge=True)
    old = heat_skill()
    initialize_skill_credit(old, SkillCreditPolicy())
    bank.add(old)
    new = heat_skill()
    new.support_count = 4
    apply(new, successful_steps(), activation)
    assert new.status == SkillStatus.VERIFIED
    bank.add(new)
    assert bank.canonical_skill(new) is new
    assert dedup_recallable_skills(bank.skills) == [new]
    assert org_eligible([bank.canonical_skill(new)]) == [new]
    bank.save()
    restored = SkillBank(bank.path, disable_merge=True)
    resolved = restored.canonical_skill(deepcopy(new))
    assert resolved.skill_id == new.skill_id and resolved.status == SkillStatus.VERIFIED
    assert org_eligible([resolved]) == [resolved]
    assert dedup_recallable_skills(restored.skills)[0].skill_id == resolved.skill_id
    assert restored.canonical_skill(old).status == SkillStatus.PROVISIONAL


def test_no_merge_does_not_alias_unknown_id_to_existing_capability(tmp_path):
    bank = SkillBank(tmp_path / "bank.json", disable_merge=True)
    bank.add(heat_skill())
    assert bank.canonical_skill(heat_skill()) is None


def test_same_id_updates_without_duplicate_rows_when_merge_disabled(tmp_path):
    bank = SkillBank(tmp_path / "bank.json", disable_merge=True)
    old = heat_skill()
    bank.add(old)
    assert bank.add(old) is False
    updated = deepcopy(old)
    updated.status = SkillStatus.VERIFIED
    assert bank.add(updated) is False
    assert len(bank.skills) == 1
    assert bank.canonical_skill(updated).status == SkillStatus.VERIFIED


def test_merge_enabled_still_resolves_merged_alias(tmp_path):
    bank = SkillBank(tmp_path / "bank.json")
    old, new = heat_skill(), heat_skill()
    bank.add(old)
    assert bank.add(new) is False
    assert bank.canonical_skill(new) is old
