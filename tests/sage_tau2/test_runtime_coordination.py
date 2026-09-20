"""Exercise actual tau2 agent message/state integration with offline model stubs."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
pytest.importorskip('tau2')
from tau2.data_model.message import AssistantMessage, UserMessage
from sage_tau2.agent import SageTau2Agent, create_sage_tau2_agent, _parse_allowed_skill_ids
from sage_tau2.schemas import AgentSpec, Tau2Skill, SkillStatus
from sage_tau2.organization import Organization
from sage_tau2.skill_bank import Tau2SkillBank
from sage_tau2.coordination import parse_plan
from sage_tau2.admission_probe import run_spec_vs_exec_probe


def make_skill(sid):
    return Tau2Skill(skill_id=sid, skill_name=sid, description='learned', precondition='current evidence',
                    action_protocol=['get_details_by_id(id=?)'], expected_effect='observed details', domain='telecom',
                    status=SkillStatus.VERIFIED)


def specialist(name, sid):
    return AgentSpec(name=name, role='specialist', responsibilities=['bounded subtask'], assigned_skills=[sid],
        acting_status='accepted', shadow_evaluation_record={'same_skill_probe_passed':True, 'promotion_probe_passed': True})


def test_two_specialists_then_budget_returns_to_executor(monkeypatch):
    import sage_tau2.agent as mod
    one, two = make_skill('s1'), make_skill('s2')
    a, b = specialist('A', 's1'), specialist('B', 's2')
    plans = iter([('A','s1'), ('B','s2'), ('A','s1')])
    seen = []
    def fake_generate(**kwargs):
        text = '\n'.join(m.content or '' for m in kwargs['messages'])
        seen.append((kwargs['call_name'], text))
        if kwargs['call_name'] == 'sage_tau2_coordinator':
            actor, sid = next(plans)
            return AssistantMessage(role='assistant', content=json.dumps({'actor': actor, 'adopted_skill_ids':[sid], 'subtask':'next check'}))
        return AssistantMessage(role='assistant', content='Please report the observed result.')
    monkeypatch.setattr(mod, 'generate', fake_generate)
    agent = SageTau2Agent(tools=[], domain_policy='Follow policy', llm='offline', domain='telecom',
        skills=[one,two], delegates=[a,b], delegate_skills=[one,two], max_delegated_turns=2)
    state = agent.get_init_state()
    actors = []
    for feedback in ['help', 'A result: first check complete', 'B result: second check complete']:
        reply, state = agent.generate_next_message(UserMessage(role='user', content=feedback), state)
        actors.append(reply.raw_data['sage_skill_event']['actor'])
    assert actors == ['A', 'B', 'Executor']
    actor_prompts = [text for name,text in seen if name == 'sage_tau2_agent']
    assert 'A result: first check complete' in actor_prompts[1]
    assert 'B result: second check complete' in actor_prompts[2]
    assert 'Do not hand off mid-episode' not in '\n'.join(actor_prompts)
    assert 'Skill s1:' in actor_prompts[0] and 'Skill s2:' not in actor_prompts[0]
    assert 'Skill s2:' in actor_prompts[1] and 'Skill s1:' not in actor_prompts[1]


def test_invalid_plan_and_unowned_skill_fall_back(monkeypatch):
    assert parse_plan('not json', actors={'Executor'}, skill_ids={'s'}).actor == 'Executor'
    assert parse_plan('{"actor":"unknown"}', actors={'Executor'}, skill_ids=set()).adopted_skill_ids == []
    import sage_tau2.agent as mod
    def fake_generate(**kwargs):
        if kwargs['call_name'] == 'sage_tau2_coordinator':
            return AssistantMessage(role='assistant', content='{"actor":"A","adopted_skill_ids":["s2"]}')
        return AssistantMessage(role='assistant', content='Continue gathering facts.')
    monkeypatch.setattr(mod, 'generate', fake_generate)
    agent = SageTau2Agent(tools=[], domain_policy='policy', llm='offline', skills=[make_skill('s1'), make_skill('s2')],
        delegates=[specialist('A', 's1')])
    reply, _ = agent.generate_next_message(UserMessage(role='user', content='help'), agent.get_init_state())
    event = reply.raw_data['sage_skill_event']
    assert event['actor'] == 'Executor' and event['adopted_skill_ids'] == []


def test_full_history_coordinator_sees_user_evidence(monkeypatch):
    import sage_tau2.agent as mod
    monkeypatch.setenv('SAGE_TAU2_USE_WINDOW_PROMPT', '0')
    observed = []
    def fake_generate(**kwargs):
        if kwargs['call_name'] == 'sage_tau2_coordinator':
            observed.append(kwargs['messages'][0].content)
            return AssistantMessage(role='assistant', content='{"actor":"Executor","adopted_skill_ids":[]}')
        return AssistantMessage(role='assistant', content='Checking.')
    monkeypatch.setattr(mod, 'generate', fake_generate)
    agent = SageTau2Agent(tools=[], domain_policy='policy', llm='offline', skills=[make_skill('s1')])
    agent.generate_next_message(UserMessage(role='user', content='Target is line ending 5678'), agent.get_init_state())
    assert 'Target is line ending 5678' in observed[0]


def test_factory_obeys_empty_freeze_and_zero_skill_budget(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith('SAGE_TAU2_'): monkeypatch.delenv(key)
    bank = Tau2SkillBank(tmp_path/'bank.json'); bank.add(make_skill('s1')); bank.save()
    org = Organization([specialist('A','s1')]); org.save(tmp_path/'org.json')
    task = {'id':'opaque', 'domain':'telecom'}
    for extra in [{'allowed_skill_ids':[]}, {'max_inject_skills':0}]:
        agent = create_sage_tau2_agent([], 'policy', llm='offline', task=task, llm_args={
            'skill_bank_path':str(bank.path), 'organization_path':str(tmp_path/'org.json'), **extra})
        assert agent.skills == [] and agent.candidate_pool == [] and agent.delegates == []
    assert _parse_allowed_skill_ids([]) == set()


def test_factory_requires_same_skill_probe_before_delegation(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith('SAGE_TAU2_'): monkeypatch.delenv(key)
    bank=Tau2SkillBank(tmp_path/'bank.json');bank.add(make_skill('s1'));bank.save()
    old=specialist('Old','s1');old.shadow_evaluation_record.pop('same_skill_probe_passed')
    valid=specialist('New','s1')
    org=Organization([old,valid]);org.save(tmp_path/'org.json')
    agent=create_sage_tau2_agent([], 'policy', llm='offline', task={'id':'opaque','domain':'telecom'}, llm_args={
        'skill_bank_path':str(bank.path),'organization_path':str(tmp_path/'org.json')})
    assert [a.name for a in agent.delegates] == ['New']


def test_probe_uses_identical_skills_and_restores_environment(tmp_path, monkeypatch):
    import sage_tau2.admission_probe as mod
    import tau2.runner
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('SAGE_TAU2_FORCE_PRIMARY','sentinel')
    monkeypatch.setattr(mod, 'select_capability_probe_task_ids', lambda **kwargs: (['t1'], {'pool_size':1}))
    configs=[]
    def fake_run(config):
        configs.append(config)
        output=Path('data/simulations')/config.save_to
        output.mkdir(parents=True)
        (output/'results.json').write_text(json.dumps({'simulations':[{'task_id':'t1','termination_reason':'user_stop','reward_info':{'reward':1}}]}),encoding='utf-8')
    monkeypatch.setattr(tau2.runner,'run_domain',fake_run)
    s=make_skill('s1'); spec=specialist('A','s1')
    result=run_spec_vs_exec_probe(domain='telecom',model='offline',user_model='offline', specialist=spec,
        organization=Organization([spec]),skills=[s],skill_bank_path=tmp_path/'bank.json',organization_path=tmp_path/'org.json',
        task_split_name='train_large',probe_dir=tmp_path/'probe')
    assert len(configs)==2
    for c in configs:
        assert c.llm_args_agent['fixed_skill_ids']==['s1']
        assert c.llm_args_agent['enable_delegation'] is False
        assert c.llm_args_agent['select_skills'] is True
        assert c.task_split_name=='train_large'
    assert configs[0].seed==configs[1].seed
    assert configs[0].max_steps==configs[1].max_steps
    assert result['comparison']=='same_skills_same_model_same_turn_budget'
    assert result['n_tasks']==1 and os.environ['SAGE_TAU2_FORCE_PRIMARY']=='sentinel'

def test_frozen_executor_with_skills_keeps_bank_and_disables_dispatch(tmp_path, monkeypatch):
    import sage_tau2.runners.eval_frozen as mod
    import tau2.runner
    monkeypatch.chdir(tmp_path)
    configs=[]
    def fake_run(config):
        configs.append(config)
        out=Path('data/simulations')/config.save_to;out.mkdir(parents=True)
        (out/'results.json').write_text('{"simulations":[]}',encoding='utf-8')
    monkeypatch.setattr(tau2.runner,'run_domain',fake_run)
    mod._run_domain_eval(domain='telecom',num_tasks=2,model='offline',user_model='offline',seed=1,
        max_concurrency=1,save_to='control',skill_bank_path=tmp_path/'same_bank.json',
        organization_path=tmp_path/'org.json',max_inject_skills=2,allow_provisional_inject=True,
        agent_name='sage_tau2',task_split_name='val_large',executor_with_skills=True)
    args=configs[0].llm_args_agent
    assert args['skill_bank_path'].endswith('same_bank.json')
    assert args['max_inject_skills']==2 and args['force_primary']=='Executor'
    assert args['enable_executor_dispatch'] is False and args['enable_delegation'] is False


def test_stage_configs_keep_same_sampling_and_separate_interventions():
    from sage_tau2.runners.online_tau2 import load_online_config
    root=Path(__file__).resolve().parents[2]/'sage_tau2/configs'
    stages=[load_online_config(root/name) for name in ['telecom_stage1_execution.yaml','telecom_stage2_skills.yaml','telecom_stage3_organization.yaml']]
    for field in ['model','user_model','seed','segment_size','num_segments','val_size','task_split_name','val_split_name']:
        assert len({getattr(s,field) for s in stages})==1
    assert [s.max_inject_skills for s in stages]==[0,2,2]
    assert [s.enable_delegation for s in stages]==[False,False,True]
    assert stages[2].enable_spec_vs_exec and not stages[2].admit_accept_ties
