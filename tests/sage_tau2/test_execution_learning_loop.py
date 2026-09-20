"""Offline regression cases for the execution -> learning -> delegation loop."""
import json
from copy import deepcopy

from sage_tau2.contracts import (interleave_evidence_guides, learn_contract,
    local_skill_outcome, observed_ledger, interaction_sequence)
from sage_tau2.credit import apply_credit_for_episode, seed_credit_from_birth_support
from sage_tau2.injection import render_skill_cards
from sage_tau2.pro_steps import build_live_window_prompt, messages_to_rich_pro_steps, semantic_summarize_json
from sage_tau2.schemas import Tau2Skill, Tau2Trajectory, ToolCallStep, SkillStatus, AgentSpec
from sage_tau2.skill_bank import Tau2SkillBank
from sage_tau2.trajectory_adapter import pro_record_to_trajectory
from sage_tau2.organization import Organization
from sage_tau2.onboarding import refresh_delegated_statuses, OnboardingPolicy


def skill(sid='refuel', protocol=None):
    return Tau2Skill(skill_id=sid, skill_name=sid, description='learned',
        precondition='observed need', expected_effect='verified effect', domain='telecom',
        capability_key='tau2.refuel_data', status=SkillStatus.PROVISIONAL,
        action_protocol=protocol or ['refuel_data(customer_id=?, line_id=?, gb_amount=2)'])


def trajectory(reward=0, steps=None, events=None, checks=None):
    return Tau2Trajectory(task_id='opaque-test-task', trial=0, domain='telecom', reward=reward,
        db_reward=None, communicate_reward=None, db_match=None, termination_reason='user_stop',
        tool_protocol=[s.name + '()' for s in steps or []], tool_steps=steps or [],
        assistant_texts=[], user_texts=[], metadata={'skill_events': events or [], 'action_checks': checks or []})


def event(sid='refuel', cid='c1', name='refuel_data', actor='Executor'):
    return {'actor': actor, 'offered_skill_ids': [sid], 'adopted_skill_ids': [sid],
            'tool_calls': [{'id': cid, 'name': name}]}


def check(name='refuel_data', requestor='assistant', match=True, **args):
    return {'action': {'name': name, 'requestor': requestor, 'arguments': args}, 'action_match': match}


def test_telecom_observations_and_past_evidence_survive_window():
    payload = {'line_id': 'L1002', 'phone_number': '5550002', 'data_used_gb': 10,
               'data_limit_gb': 10, 'roaming_enabled': False}
    summary = semantic_summarize_json(json.dumps(payload))
    for key in payload:
        assert key in summary
    messages = [{'role': 'assistant', 'tool_calls': [{'id': 'usage', 'name': 'get_data_usage', 'arguments': {'line_id': 'L1002'}}]},
                {'role': 'tool', 'id': 'usage', 'content': json.dumps(payload)}]
    for _ in range(15):
        messages.extend([{'role': 'assistant', 'content': 'checking'}, {'role': 'user', 'content': 'continue'}])
    prompt, _ = build_live_window_prompt(messages, history_window=2, task_description='help')
    assert 'Observed tool evidence' in prompt
    assert '5550002' in prompt and '"roaming_enabled": false' in prompt
    assert 'data_used_gb' in prompt and 'L1002' in prompt


def test_ledger_updates_exact_call_and_never_invents_state():
    messages = []
    for i, value in enumerate([False, True]):
        messages.extend([{'role': 'assistant', 'tool_calls': [{'id': str(i), 'function': {'name': 'details', 'arguments': '{"id":"L2"}'}}]},
                         {'role': 'tool', 'tool_call_id': str(i), 'content': json.dumps({'enabled': value})}])
    ledger = observed_ledger(messages)
    assert ledger.count('"tool": "details"') == 1
    assert 'true' in ledger and 'false' not in ledger


def test_complete_cards_skip_oversize_without_truncating_next_card():
    huge = skill('huge', ['x' * 9000])
    small = skill()
    text, offered = render_skill_cards([huge, small], max_chars=1800)
    assert [s.skill_id for s in offered] == ['refuel']
    assert len(text) <= 1800 and 'huge' not in text and 'truncated' not in text
    assert 'Stop/return:' in text and small.action_protocol[0] in text
    assert render_skill_cards([small], max_chars=10) == ('', [])


def test_cross_role_payment_dependency_and_repeated_lookup_are_preserved():
    t = trajectory(reward=1)
    t.metadata['interaction_sequence'] = [
        {'requestor': 'assistant', 'name': 'get_details_by_id', 'arguments': {'id': 'C1'}},
        {'requestor': 'assistant', 'name': 'get_details_by_id', 'arguments': {'id': 'L2'}},
        {'requestor': 'assistant', 'name': 'send_payment_request', 'arguments': {'bill_id': 'B1'}},
        {'requestor': 'user', 'name': 'make_payment', 'arguments': {}},
        {'requestor': 'assistant', 'name': 'resume_line', 'arguments': {'line_id': 'L2'}},
    ]
    proto = ['get_details_by_id(id=?)', 'send_payment_request(bill_id=?)', 'resume_line(line_id=?)', 'Guide user: make_payment']
    ordered = interleave_evidence_guides(proto, [t])
    assert len(ordered) == 5
    assert 'make_payment' in ordered[3] and 'resume_line' in ordered[4]
    assert sum('get_details_by_id' in p for p in ordered) == 2
    assert 'C1' not in str(ordered) and 'L2' not in str(ordered)


def test_contract_sources_learn_line_binding_without_leaking_ids():
    t = trajectory(reward=1, steps=[
        ToolCallStep('get_customer_by_phone', {'phone_number': '5550000'}, result_content='{"customer_id":"C123", "lines":["L456"]}'),
        ToolCallStep('get_details_by_id', {'id': 'L456'}, result_content='{"line_id":"L456"}'),
        ToolCallStep('refuel_data', {'customer_id': 'C123', 'line_id': 'L456', 'gb_amount': 2}, result_content='ok'),
    ])
    contract = learn_contract(skill(), [t])
    assert 'get_customer_by_phone.result.lines[]' in contract['bindings']['get_details_by_id.id']
    assert 'C123' not in str(contract) and 'L456' not in str(contract)
    assert contract['version'] == 2 and 'current' in contract['condition']
    assert 'refuel_data' in contract['verification']


def test_birth_reseeding_does_not_erase_online_failure():
    s = skill(); s.support_count = 3
    credit = seed_credit_from_birth_support(s)
    t = trajectory(events=[event()], checks=[check(line_id='L2')])
    apply_credit_for_episode([s], t)
    assert credit['uses'] == 4 and credit['successes'] == 3
    seed_credit_from_birth_support(s)
    assert credit['uses'] == 4 and credit['successes'] == 3
    s.support_count = 4
    seed_credit_from_birth_support(s)
    assert credit['uses'] == 5 and credit['successes'] == 4
    assert credit['online_uses'] == 1 and credit['online_successes'] == 0


def test_offered_not_adopted_does_not_get_credit():
    s = skill(); e = event(); e['adopted_skill_ids'] = []
    t = trajectory(reward=1, events=[e])
    assert apply_credit_for_episode([s], t) == []
    credit = s.metadata['skill_credit']
    assert credit['offered_episodes'] == 1 and credit['uses'] == 0


def test_correct_local_effect_credited_despite_whole_episode_failure():
    s = skill(); args = {'customer_id': 'C1', 'line_id': 'L2', 'gb_amount': 2}
    t = trajectory(events=[event()], steps=[ToolCallStep('refuel_data', args, 'c1', 'ok')], checks=[check(**args)])
    result = apply_credit_for_episode([s], t)[0]
    assert result['success'] is True and result['task_success'] is False
    assert result['outcome'] == 'verified_local_effect'


def test_wrong_entity_and_other_turn_cannot_claim_success():
    s = skill()
    t = trajectory(reward=1, events=[event()],
        steps=[ToolCallStep('refuel_data', {'line_id': 'L1'}, 'c1', 'ok'),
               ToolCallStep('refuel_data', {'line_id': 'L2'}, 'other', 'ok')],
        checks=[check(line_id='L2')])
    assert local_skill_outcome(s, t, t.metadata['skill_events'])['success'] is False


def test_unverified_write_is_neutral_but_execution_is_counted():
    s = skill()
    t = trajectory(reward=1, events=[event()], steps=[ToolCallStep('refuel_data', {}, 'c1', 'ok')])
    assert apply_credit_for_episode([s], t)[0]['success'] is None
    assert s.metadata['skill_credit']['executed_episodes'] == 1
    assert s.metadata['skill_credit']['uses'] == 0


def test_db_match_alone_is_not_success_and_infrastructure_is_not_credit():
    t = trajectory(); t.db_match = True
    assert not t.success and t.has_positive_db_effect
    t.metadata['skill_events'] = [event()]; t.termination_reason = 'infrastructure_error'
    s = skill()
    assert apply_credit_for_episode([s], t) == []


def test_pro_roundtrip_retains_adoption_and_results_by_call_id():
    messages = [{'role': 'assistant', 'tool_calls': [
        {'id': 'a', 'name': 'get_details_by_id', 'arguments': {'id': 'C1'}},
        {'id': 'b', 'name': 'get_details_by_id', 'arguments': {'id': 'L2'}}],
        'raw_data': {'sage_skill_event': event()}},
        {'role': 'tool', 'id': 'b', 'content': '{"line_id":"L2"}'},
        {'role': 'tool', 'id': 'a', 'content': '{"customer_id":"C1"}'},
        {'role': 'user', 'content': 'done'}]
    steps = messages_to_rich_pro_steps(messages)
    t = pro_record_to_trajectory({'task_id': 'roundtrip', 'domain': 'telecom', 'reward': 1, 'steps': steps})
    assert t.metadata['skill_events'][0]['adopted_skill_ids'] == ['refuel']
    assert json.loads(t.tool_steps[0].result_content) == {'customer_id': 'C1'}
    assert json.loads(t.tool_steps[1].result_content) == {'line_id': 'L2'}


def test_user_side_local_verification_requires_adopted_guidance():
    s = skill('pay', ['Guide user: make_payment()'])
    e = event('pay'); e['tool_calls'] = []
    messages = [{'role': 'assistant', 'content': 'Please pay', 'raw_data': {'sage_skill_event': e}},
        {'role': 'user', 'tool_calls': [{'id': 'u', 'name': 'make_payment', 'arguments': {'bill_id': 'B2'}}]},
        {'role': 'tool', 'id': 'u', 'content': 'paid'}]
    t = trajectory(events=[e], checks=[check('make_payment', 'user', bill_id='B2')])
    t.metadata['interaction_sequence'] = interaction_sequence(messages)
    assert local_skill_outcome(s, t, [e])['success'] is True
    t.metadata['interaction_sequence'][0]['adopted_skill_ids'] = []
    assert local_skill_outcome(s, t, [e])['success'] is False


def test_bank_cannot_merge_distinct_ordered_contracts(tmp_path):
    bank = Tau2SkillBank(tmp_path / 'bank.json')
    first = skill('a', ['send_payment_request()', 'Guide user: make_payment()', 'resume_line()'])
    other = skill('b', ['send_payment_request()', 'resume_line()'])
    for s in [first, other]: s.metadata['execution_contract'] = {'version': 2}
    bank.extend([first, other])
    assert len(bank.skills) == 2
    assert bank.skills[0].action_protocol == first.action_protocol
    same = deepcopy(first); same.skill_id = 'a-copy'; same.evidence_ids = ['new']
    assert bank.add(same) is first and len(bank.skills) == 2


def test_delegated_onboarding_uses_local_outcomes_not_primary_wins():
    specialist = AgentSpec(name='Data', role='specialist', responsibilities=[], assigned_skills=['refuel'],
        shadow_evaluation_record={'same_skill_probe_passed': True})
    org = Organization([specialist])
    s = skill()
    t = trajectory(events=[event(actor='Data')], steps=[ToolCallStep('refuel_data', {'line_id':'L2'}, 'c1', 'ok')], checks=[check(line_id='L2')])
    decisions = refresh_delegated_statuses(org, [t], [s], policy=OnboardingPolicy(min_games=1, min_wins=1))
    assert decisions[0]['decision'] == 'accepted'
    assert specialist.shadow_evaluation_record['delegated_local_wins'] == 1
    assert 'wins_as_primary' not in specialist.shadow_evaluation_record

def test_conditions_are_scoped_to_the_target_object():
    s = skill()
    t = trajectory(reward=1, steps=[
        ToolCallStep('get_details_by_id', {'id':'L1'}, result_content='{"roaming_enabled":true}'),
        ToolCallStep('get_details_by_id', {'id':'L2'}, result_content='{"roaming_enabled":false}'),
        ToolCallStep('refuel_data', {'line_id':'L2'}, result_content='ok'),
    ])
    conditions = learn_contract(s, [t])['observed_conditions']
    assert len(conditions) == 1 and '<line_id>' in conditions[0] and 'false' in conditions[0]
    t.tool_steps.insert(2, ToolCallStep('get_details_by_id', {'id':'L2'}, result_content='{"roaming_enabled":true}'))
    assert learn_contract(s,[t])['observed_conditions'] == []


def test_recovered_tool_error_does_not_override_verified_local_effect():
    s = skill(); e = event()
    e['tool_calls'].append({'id':'retry','name':'refuel_data'})
    t = trajectory(events=[e], steps=[ToolCallStep('refuel_data', {'line_id':'L2'}, 'c1', 'retry later', True),
        ToolCallStep('refuel_data', {'line_id':'L2'}, 'retry', 'ok')], checks=[check(line_id='L2')])
    assert local_skill_outcome(s,t,[e])['success'] is True


def test_admission_excludes_training_evidence_and_fixed_validation():
    from sage_tau2.admission_probe import select_capability_probe_task_ids
    from types import SimpleNamespace
    s=skill();s.metadata['task_ids']=['seen']
    spec=AgentSpec(name='Data',role='specialist',responsibilities=[],assigned_skills=[s.skill_id])
    # Unstructured task identifiers all match via explicit episode scope text.
    def tasks(**kwargs):
        return [{'id':tid, 'description':'refuel_data'} for tid in ['seen','val','fresh']]
    ids, meta=select_capability_probe_task_ids(domain='telecom', specialist=spec,skills=[s],num_tasks=3,
        seed=1, task_loader=tasks, excluded_task_ids=['val'])
    assert 'seen' not in ids and 'val' not in ids
    assert set(meta['excluded_task_ids']) == {'seen','val'}
