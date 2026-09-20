"""Regression checks for ALFWorld selection, bounded dispatch and canonical SOPs."""
import json
from types import SimpleNamespace

import pytest

from sage_mas.alfworld_evaluator import AlfWorldEvaluatorConfig, AlfWorldOrganizationEvaluator
from sage_mas.alfworld_coordination import parse_turn
from sage_mas.executor_dispatch import ExecutorDispatchConfig, dispatch_config_from_mapping
from sage_mas.executable_protocol import ensure_executable_protocol, protocol_adherence_score
from sage_mas.runtime import MASRuntime, LLMResult
from sage_mas.schemas import AgentSpec, AtomicOp, Skill, SkillStatus
from sage_mas.skill_recall import build_recall_index


def skill(name, operation='heat'):
    return Skill(skill_name=name, skill_id=name, description=operation,
        precondition=f'held cup needs {operation}', expected_effect=f'cup is {operation}ed',
        action_protocol=[f'{operation} <object> with microwave'], applicable_atomic_ops=[AtomicOp.ACT],
        capability_key='transform.' + operation, status=SkillStatus.VERIFIED,
        applicable_task_families=['household'])


def agents(*skills):
    owner = AgentSpec(name='Executor', role='Executor', responsibilities=['own task'])
    return [owner] + [AgentSpec(name='Expert'+s.skill_id, role='specialist', responsibilities=['bounded step'],
        assigned_skills=[s.skill_name], tool_permissions=['alfworld_action']) for s in skills]


class Backend:
    def __init__(self, selection=None, actor='Executor'):
        self.selection, self.actor, self.calls = selection, actor, []

    def complete(self, system_prompt, user_prompt):
        self.calls.append(user_prompt)
        if 'Decide whether to look up' in user_prompt:
            return LLMResult('<query>heat cool cup microwave</query>', 1, 1)
        if 'BM25 similarity is not applicability' in user_prompt:
            return LLMResult(json.dumps({'selected': [
                {'skill_id': sid, 'reason': 'Observed unmet goal', 'evidence_quote': 'holding cup'}
                for sid in (self.selection or [])]}), 2, 3)
        if 'Choose the actor for the next ONE' in user_prompt:
            return LLMResult(json.dumps({'actor': self.actor,
                'skill_id': self.selection[0] if self.actor != 'Executor' else '',
                'subtask': 'Resolve the next local effect', 'reason': 'Current evidence matches',
                'expected_result': 'Observe the local effect'}), 3, 4)
        return LLMResult('<action>look</action>', 1, 1)


def act(evaluator, backend, pool, roster, history=None):
    runtime = MASRuntime(agents=roster, skills=pool, backend=backend,
                         allow_executor_skill_injection=True, strict_skill_selection=True,
                         turn_delegation=True)
    index, indexed = build_recall_index(pool)
    return evaluator._act_with_skill_retrieval(runtime, observation='holding cup',
        retrieval_observation='holding cup', task='prepare cup', task_family='household',
        history_steps=history or [], gamefile='g', primary_agent='Executor',
        assigned_skills=pool, assigned_skill_names={s.skill_name for s in pool},
        injected_skills=None, ignore_assigned=False, agents=roster,
        recall_index=index, recall_skills=indexed)


def test_bm25_shortlist_is_wider_than_mount_limit_and_semantically_rejected():
    pool = [skill('heat'), skill('cool', 'cool')]
    backend = Backend(selection=['cool'])
    evaluator = AlfWorldOrganizationEvaluator(backend, AlfWorldEvaluatorConfig(
        max_injected_skills=1, skill_recall_top_k=8,
        executor_dispatch=ExecutorDispatchConfig(enabled=False)))
    _, info, shown = act(evaluator, backend, pool, agents())
    assert len(info['shortlist']) == 2
    assert info['selected'] == shown == ['cool']
    assert info['token_cost'] == 7  # query + semantic filter
    backend.selection = []
    _, info, shown = act(evaluator, backend, pool, agents())
    assert not shown and not info['selected']


def test_bad_semantic_response_mounts_nothing():
    backend = Backend(selection=['nonexistent'])
    evaluator = AlfWorldOrganizationEvaluator(backend)
    _, info, shown = act(evaluator, backend, [skill('heat')], agents())
    assert shown == [] and info['selection_error']


def test_single_expert_does_not_force_dispatch():
    s = skill('heat'); roster = agents(s)
    backend = Backend(selection=['heat'], actor='Executor')
    evaluator = AlfWorldOrganizationEvaluator(backend)
    result, info, shown = act(evaluator, backend, [s], roster)
    assert result.messages[-1].agent_name == 'Executor'
    assert info['turn_dispatch']['eligible_agents'] == ['Expertheat']
    assert info['turn_dispatch']['plan']['actor'] == 'Executor'
    assert shown == []
    assert any('Choose the actor for the next ONE' in p for p in backend.calls)


def test_experts_switch_after_actual_feedback_and_receive_only_selected_skill():
    heat, cool = skill('heat'), skill('cool', 'cool'); roster = agents(heat, cool)
    roster[1].assigned_skills.append(cool.skill_name)  # A multi-skill expert must not mount all skills.
    backend = Backend(selection=['heat'], actor='Expertheat')
    evaluator = AlfWorldOrganizationEvaluator(backend)
    first, info, shown = act(evaluator, backend, [heat, cool], roster)
    assert first.messages[-1].agent_name == 'Expertheat' and shown == ['heat']
    assert 'cool <object>' not in backend.calls[-1]
    assert 'bounded next action' in backend.calls[-1]
    history = [{'step': 1, 'action': 'heat cup 1 with microwave 1',
        'observation': 'You heat the cup.', 'is_action_valid': True,
        'agent_messages': [{'agent': 'Expertheat'}], 'skill_retrieval': info}]
    backend.selection, backend.actor = ['cool'], 'Expertcool'
    second, info, shown = act(evaluator, backend, [heat, cool], roster, history)
    assert second.messages[-1].agent_name == 'Expertcool' and shown == ['cool']
    assert info['turn_dispatch']['delegated_turns_used'] == 1
    assert any('You heat the cup.' in p for p in backend.calls if 'Choose the actor' in p)


def test_budget_and_unavailable_actor_return_control_to_executor():
    s = skill('heat'); roster = agents(s)
    backend = Backend(selection=['heat'], actor='ghost')
    evaluator = AlfWorldOrganizationEvaluator(backend)
    result, info, shown = act(evaluator, backend, [s], roster)
    assert result.messages[-1].agent_name == 'Executor' and shown == []
    assert info['turn_dispatch']['error']
    evaluator.config.executor_dispatch.max_delegated_turns = 0
    backend.calls.clear()
    result, info, _ = act(evaluator, backend, [s], roster)
    assert result.messages[-1].agent_name == 'Executor'
    assert info['turn_dispatch']['budget_exhausted']
    assert not any('Choose the actor' in p for p in backend.calls)


def test_unowned_skill_is_rejected_by_turn_contract():
    s = skill('heat')
    roster = [{'name': 'Executor', 'owned_skill_ids': ['heat']}, {'name': 'Expert', 'owned_skill_ids': []}]
    with pytest.raises(ValueError, match='own available'):
        parse_turn(json.dumps({'actor': 'Expert', 'skill_id': 'heat'}), 'Executor', [s], roster)


def test_delegate_uses_selected_revision_not_same_name_cache():
    good, stale = skill('same'), skill('same', 'cool')
    stale.skill_id = 'old-version'
    backend = Backend(); roster = agents(good)
    runtime = MASRuntime(agents=roster, skills=[good, stale], backend=backend, turn_delegation=True)
    runtime.act('holding cup', [good], {'same'}, roster[1].name)
    assert 'heat <object> with microwave' in backend.calls[-1]
    assert 'cool <object> with microwave' not in backend.calls[-1]


def test_default_owner_and_explicit_episode_control():
    s = skill('heat'); roster = agents(s)
    evaluator = AlfWorldOrganizationEvaluator(Backend())
    args = dict(task='heat cup', task_family='household', agents=roster, skills=[s], executor_name='Executor')
    assert evaluator._initial_assignment(**args).primary_agent == 'Executor'
    evaluator.config.executor_dispatch = dispatch_config_from_mapping({'mode': 'episode'})
    assert evaluator._initial_assignment(**args).primary_agent == 'Expertheat'
    evaluator.config.executor_dispatch.mode = 'turn'
    evaluator.config.force_inject_skills = True
    assert evaluator._initial_assignment(**args).primary_agent == 'Expertheat'


def test_cleaned_protocol_is_canonical_despite_long_search_prefix_and_old_cache():
    s = skill('heat')
    s.action_protocol = ['go to fridge', 'open fridge', 'take <object> from fridge',
        'go to microwave', 'heat <object> with microwave', 'go to countertop', 'move <object> to countertop']
    records = [{'action': action, 'observation': 'You open cabinet.'} for _ in range(9)
               for action in ['go to cabinet 1', 'open cabinet 1']]
    records += [{'action': action.replace('<object>', 'cup 1'), 'observation': 'confirmed '+str(i)}
                for i, action in enumerate(s.action_protocol)]
    s.metadata['executable_protocol'] = [{'index': 0, 'action_template': 'go to <location>'}]
    s.metadata['executable_protocol_source'] = 'trajectory'
    executable = ensure_executable_protocol(s, trajectory_steps=records, max_steps=2)
    assert len(executable) == 7
    assert executable[3].action_template == 'go to microwave'
    assert executable[4].action_template == 'heat <target> with microwave'
    assert executable[-1].expected_obs_hint == 'confirmed 6'
    assert protocol_adherence_score(s, records[18:]) == 1.0
    s.action_protocol = ['cool <object> with fridge']
    assert ensure_executable_protocol(s)[0].action_template == 'cool <target> with fridge'
    assert not ensure_executable_protocol(s)[0].expected_obs_hint


def test_transitions_cannot_override_protocol_or_remove_repeated_steps():
    s = skill('heat')
    s.action_protocol = ['take <object> from <source>'] * 2 + ['heat <object> with microwave']
    s.metadata['confirmed_transitions'] = [{'action': 'cool cup 1 with fridge 1', 'observation': 'You cool cup.'}]
    assert len(ensure_executable_protocol(s)) == 3
    assert ensure_executable_protocol(s)[-1].action_template == 'heat <target> with microwave'
    assert all(not step.expected_obs_hint for step in ensure_executable_protocol(s))


def test_full_soft_card_keeps_terminal_steps():
    from sage_mas.skill_inject_sparse_soft import render_soft_skill_block
    s = skill('heat'); s.action_protocol = [f'go to cabinet {i}' for i in range(18)] + ['heat <object> with microwave']
    assert '19. heat <target> with microwave' in render_soft_skill_block([s])


def test_evaluator_records_owner_turn_handoffs_and_costs(monkeypatch):
    import sage_mas.alfworld_evaluator as mod
    class Env:
        config = SimpleNamespace(env=SimpleNamespace(history_length=0))
        index = 0
        closed = False
        def obs(self):
            text = f'Your task is to: prepare cup. holding cup phase{self.index + 1}'
            return {'anchor': [text], 'text': [text]}
        def reset(self, _):
            return self.obs(), [{}]
        def step(self, actions):
            assert ('heat' if self.index == 0 else 'cool') in actions[0]
            self.index += 1
            return self.obs(), [float(self.index == 2)], [self.index == 2], [
                {'won': self.index == 2, 'is_action_valid': True}]
        def close(self):
            self.closed = True
    class RoutingBackend(Backend):
        def complete(self, system_prompt, user_prompt):
            if 'BM25 similarity is not applicability' in user_prompt:
                self.selection = ['cool' if 'phase2' in user_prompt else 'heat']
                self.actor = 'Expert' + self.selection[0]
            if 'Executor turn assignment:' in user_prompt:
                return LLMResult(f'<action>{self.selection[0]} cup 1 with microwave 1</action>', 1, 1)
            return super().complete(system_prompt, user_prompt)
    env = Env(); backend = RoutingBackend()
    pool = [skill('heat'), skill('cool', 'cool')]
    monkeypatch.setattr(mod, 'build_alfworld_env_manager', lambda **kwargs: env)
    monkeypatch.setattr(mod, '_task_family_from_gamefile', lambda _: 'household')
    evaluator = AlfWorldOrganizationEvaluator(backend, AlfWorldEvaluatorConfig(
        max_steps=2, parallel_envs=1, api_concurrency=1, allow_executor_skill_injection=True))
    trial = evaluator.evaluate(agents(*pool), pool, ['fake'], 'delegation')[0]
    assert env.closed and trial.won
    assert trial.assigned_primary_agent == 'Executor'
    assert trial.dispatch_layer == 'executor_turn_coordinator'
    assert trial.actions_by_agent == {'Expertheat': 1, 'Expertcool': 1}
    assert [t['plan']['actor'] for t in trial.dispatch_evidence['turns']] == ['Expertheat', 'Expertcool']
    assert trial.cost == 32  # Two query(2) + semantic(5) + coordinator(7) + actor(2) calls.
    assert trial.skill_activation_steps == {'heat': 1, 'cool': 2}
