"""Persistent instances, exclusive attribution, and solution-driven organization."""
from copy import deepcopy
import json

from sage_tau2.schemas import Tau2Skill, SkillStatus, Tau2Trajectory, ToolCallStep, AgentSpec
from sage_tau2.execution import (start_execution, next_step, observe, record_action, apply_feedback,
                                 choose_execution, action_is_ready, execution_prompt)
from sage_tau2.execution_credit import execution_outcomes
from sage_tau2.credit import apply_credit_for_episode
from sage_tau2.skill_clustering import SkillClusterArchive, protocol_novelty
from sage_tau2.solutions import solution_fingerprint, solution_distance, solution_contract
from sage_tau2.organization import Organization
from sage_tau2.nominate_admit import propose_add_agents, run_nominate_admit, NominateAdmitConfig
from sage_tau2.lifecycle import reconcile_specialists, specialist_revision, refresh_instance_statuses
from sage_tau2.onboarding import OnboardingPolicy


def skill(sid='s', protocol=None):
    return Tau2Skill(skill_id=sid, skill_name=sid, description='demo', precondition='observed need', expected_effect='checked',
        domain='telecom', capability_key='tau2.refuel_data', status=SkillStatus.VERIFIED, support_count=5,
        action_protocol=protocol or ['refuel_data(line_id=?, gb_amount=2)'])


def call(name, cid, **args):
    return {'role':'assistant', 'tool_calls':[{'id':cid,'name':name,'arguments':args}]}


def trace(instances, events, steps, checks=None, reward=0):
    events=deepcopy(events)
    events[-1]['executions']=deepcopy(instances)
    return Tau2Trajectory(task_id='case', trial=0, domain='telecom', reward=reward, db_match=None,
        db_reward=None, communicate_reward=None, termination_reason='user_stop',tool_protocol=[],tool_steps=steps,
        assistant_texts=[],user_texts=[],metadata={'skill_events':events,'action_checks':checks or []})


def event(inst, message):
    claims=record_action(inst,message)
    return {'version':2,'execution_id':inst['execution_id'],'actor':inst['actor'],
        'offered_skill_ids':[inst['skill_id']], 'adopted_skill_ids':[inst['skill_id']],
        'tool_calls':message.get('tool_calls',[]), 'call_ownership':claims, 'executions':[deepcopy(inst)]}


def check(name='refuel_data', line='L2', requestor='assistant'):
    return {'action_match':True,'action':{'requestor':requestor,'name':name,'arguments':{'line_id':line}}}


def test_dependency_waits_for_user_and_valid_feedback_reference():
    s=skill(protocol=['send_payment_request(bill_id=?)','Guide user: make_payment()','resume_line(line_id=?)'])
    inst=start_execution(s,'Payments')
    assert not action_is_ready(inst,call('resume_line','bad',line_id='L2'))
    assert action_is_ready(inst,call('send_payment_request','pay',bill_id='B2'))
    record_action(inst,call('send_payment_request','pay',bill_id='B2'))
    assert next_step(inst)['status']=='inflight'
    assert not action_is_ready(inst,call('resume_line','bad',line_id='L2'))
    observe([inst],[{'role':'tool','id':'other','content':'ok'}],'obs-1')
    assert next_step(inst)['status']=='inflight'
    observe([inst],[{'role':'tool','id':'pay','content':'sent'}],'obs-2')
    assert next_step(inst)['index']==1
    record_action(inst,{'role':'assistant','content':'Please complete payment.'})
    observe([inst],[{'role':'user','content':'Payment completed.'}],'obs-3')
    update={'execution_id':inst['execution_id'],'step_index':1,'verdict':'observed','evidence_ref':'invented'}
    assert not apply_feedback([inst],update)
    assert not action_is_ready(inst,call('resume_line','resume',line_id='L2'))
    update['evidence_ref']='obs-3'
    assert apply_feedback([inst],update)
    assert action_is_ready(inst,call('resume_line','resume',line_id='L2'))
    record_action(inst,call('resume_line','resume',line_id='L2'))
    observe([inst],[{'role':'tool','id':'resume','content':'active'}],'obs-4')
    assert inst['status']=='protocol_complete'
    assert inst['return_record']['effect_verified'] is False


def test_pause_resume_and_serialization_preserve_instance_identity():
    a,b=skill('a'),skill('b')
    instances=[]
    first=choose_execution(instances,skill=a,actor='A')
    record_action(first,call('refuel_data','a1',line_id='L2'))
    second=choose_execution(instances,skill=b,actor='B')
    assert first['status']=='paused'
    restored=json.loads(json.dumps(instances))
    observe(restored,[{'role':'tool','id':'a1','content':'ok'}],'obs-x')
    assert restored[0]['steps'][0]['status']=='observed'
    assert first['execution_id'] in execution_prompt(restored)
    assert choose_execution(restored,skill=a,actor='Executor',execution_id=first['execution_id'])['execution_id']==first['execution_id']


def test_dependent_tools_cannot_run_in_same_batch():
    s=skill(protocol=['send_payment_request()','resume_line()'])
    inst=start_execution(s,'A')
    batch=call('send_payment_request','a');batch['tool_calls']+=call('resume_line','b')['tool_calls']
    assert not action_is_ready(inst,batch)
    assert action_is_ready(inst,call('get_customer_by_phone','lookup',phone_number='555'))


def test_one_call_cannot_credit_two_instances_and_replay_is_idempotent():
    a,b=skill('a'),skill('b')
    ia,ib=start_execution(a,'A'),start_execution(b,'B')
    ev=event(ia,call('refuel_data','c1',line_id='L2'))
    t=trace([ia,ib],[ev],[ToolCallStep('refuel_data',{'line_id':'L2'},'c1','ok')],[check()])
    outcomes=execution_outcomes(t)
    assert [o['success'] for o in outcomes]==[True,False]
    first=apply_credit_for_episode([a,b],t)
    assert a.metadata['skill_credit']['online_successes']==1
    assert b.metadata['skill_credit']['online_successes']==0
    assert apply_credit_for_episode([a,b],t)==[]
    assert a.metadata['skill_credit']['online_successes']==1


def test_conflicting_call_owners_are_neutral():
    a,b=skill('a'),skill('b');ia,ib=start_execution(a,'A'),start_execution(b,'B')
    ea=event(ia,call('refuel_data','same',line_id='L2'))
    eb=event(ib,call('refuel_data','same',line_id='L2'))
    t=trace([ia,ib],[ea,eb],[ToolCallStep('refuel_data',{'line_id':'L2'},'same','ok')],[check()])
    assert all(o['success'] is None and o['outcome']=='ambiguous_call_owner' for o in execution_outcomes(t))


def test_instances_on_different_objects_are_scored_separately():
    s=skill();ia,ib=start_execution(s,'A'),start_execution(s,'A')
    ea=event(ia,call('refuel_data','a',line_id='L1'));eb=event(ib,call('refuel_data','b',line_id='L2'))
    t=trace([ia,ib],[ea,eb],[ToolCallStep('refuel_data',{'line_id':'L1'},'a','ok'),ToolCallStep('refuel_data',{'line_id':'L2'},'b','ok')],
            [check(line='L1'),check(line='L2')])
    assert [o['success'] for o in execution_outcomes(t)]==[True,True]


def test_repeated_operation_does_not_reuse_same_verification():
    s=skill();ia,ib=start_execution(s,'A'),start_execution(s,'A')
    ea=event(ia,call('refuel_data','a',line_id='L2'));eb=event(ib,call('refuel_data','b',line_id='L2'))
    t=trace([ia,ib],[ea,eb],[ToolCallStep('refuel_data',{'line_id':'L2'},'a','ok'),ToolCallStep('refuel_data',{'line_id':'L2'},'b','ok')],[check()])
    assert [o['success'] for o in execution_outcomes(t)]==[True,None]


def test_unfinished_post_write_check_prevents_premature_credit():
    s=skill(protocol=['refuel_data(line_id=?)','get_data_usage(line_id=?)']);inst=start_execution(s,'A')
    ev=event(inst,call('refuel_data','a',line_id='L2'))
    t=trace([inst],[ev],[ToolCallStep('refuel_data',{'line_id':'L2'},'a','ok')],[check()])
    assert execution_outcomes(t)[0]['outcome']=='execution_incomplete'


def test_protocol_order_parameters_conditions_and_multiplicity_change_identity():
    proto=['send_payment_request()','Guide user: make_payment()','resume_line()']
    assert protocol_novelty(proto,list(reversed(proto)))>.35
    assert protocol_novelty(['get_details_by_id()'],['get_details_by_id()','get_details_by_id()'])>0
    a=skill('a');b=skill('b',['refuel_data(line_id=?, gb_amount=5)'])
    assert solution_fingerprint(a)!=solution_fingerprint(b)
    b=deepcopy(a);b.metadata['execution_contract']={'observed_conditions':['roaming_enabled == false']}
    assert solution_distance(solution_contract(a),solution_contract(b))==1


def test_changed_skill_is_reclustered_and_same_capability_can_spawn_two_experts():
    a=skill('a',['send_payment_request()','Guide user: make_payment()','resume_line()'])
    b=skill('b',list(reversed(a.action_protocol)))
    archive=SkillClusterArchive(novelty_threshold=.35);archive.update([a,b],segment=1)
    assert len(archive.clusters)==2
    cfg=NominateAdmitConfig();cfg.nomination.max_new_agents_per_round=3
    proposals=propose_add_agents(organization=Organization(),archive=archive,skills=[a,b],config=cfg)
    assert len([p for p in proposals if p.new_agent])==2
    assert len({p.new_agent.name for p in proposals})==2
    old=archive.cluster_of(a.skill_id).cluster_id
    a.action_protocol=['refuel_data(line_id=?)']
    archive.update([a,b],segment=2)
    assert archive.cluster_of(a.skill_id).cluster_id!=old
    restored=SkillClusterArchive.from_dict(archive.to_dict())
    assert restored.cluster_of(a.skill_id).cluster_id==archive.cluster_of(a.skill_id).cluster_id


def test_existing_variant_does_not_block_new_variant():
    a=skill('a',['refuel_data()']);b=skill('b',['get_data_usage()','Guide user: toggle_data()','refuel_data()'])
    archive=SkillClusterArchive(novelty_threshold=.2);archive.update([a,b],segment=1)
    agent=AgentSpec(name='Data',role='specialist',responsibilities=[],assigned_skills=['a'],capability_keys=['tau2.refuel_data'],
                   metadata={'cluster_id':archive.cluster_of('a').cluster_id},acting_status='accepted')
    proposals=propose_add_agents(organization=Organization([agent]),archive=archive,skills=[a,b])
    assert proposals[0].new_agent and proposals[0].new_agent.assigned_skills==['b']


def test_legacy_expert_revalidation_and_dead_skill_retirement():
    s=skill();archive=SkillClusterArchive();archive.update([s],segment=1)
    agent=AgentSpec(name='Old',role='specialist',responsibilities=[],assigned_skills=['s'],acting_status='accepted',
                   capability_keys=['tau2.refuel_data'])
    org=Organization([agent]);cfg=NominateAdmitConfig();cfg.admission.enable_spec_vs_exec=True
    calls=[]
    def probe(**kwargs):
        calls.append(kwargs['specialist'].name)
        return {'specialist_sr':1,'executor_sr':0,'n_tasks':8,'comparison':'same_skills_same_model_same_turn_budget'}
    run_nominate_admit(organization=org,archive=archive,skills=[s],config=cfg,spec_vs_exec_probe=probe)
    assert calls==['Old']
    agent=next(a for a in org.agents if a.name=='Old')
    assert agent.shadow_evaluation_record['same_skill_probe_passed']
    assert reconcile_specialists(org,archive,[s])==[]
    s.action_protocol.append('get_data_usage()');archive.update([s],segment=2)
    assert reconcile_specialists(org,archive,[s])
    assert not agent.shadow_evaluation_record['same_skill_probe_passed']
    s.status=SkillStatus.RETIRED
    reconcile_specialists(org,archive,[s])
    assert agent.acting_status=='dormant'


def test_accepted_expert_retires_after_verified_instance_failures():
    s=skill();agent=AgentSpec(name='A',role='specialist',responsibilities=[],assigned_skills=['s'],acting_status='accepted',
        shadow_evaluation_record={'same_skill_probe_passed':True,'admitted_solution_revision':specialist_revision([s])})
    org=Organization([agent]);pol=OnboardingPolicy(min_games=1,min_wins=1,remove_after_rejected_windows=2)
    for i in range(2):
        inst=start_execution(s,'A');ev=event(inst,call('refuel_data',str(i),line_id='wrong'))
        t=trace([inst],[ev],[ToolCallStep('refuel_data',{'line_id':'wrong'},str(i),'ok')],[check()])
        refresh_instance_statuses(org,[t],pol)
        assert refresh_instance_statuses(org,[t],pol)==[]
    assert agent.acting_status=='dormant'
