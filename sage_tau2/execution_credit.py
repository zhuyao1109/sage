"""Offline instance attribution. Verification evidence is never injected into actors."""
from copy import deepcopy
import json

from sage_tau2.execution import execution_snapshots


def execution_outcomes(trajectory):
    from sage_tau2.contracts import local_skill_outcome, interaction_sequence
    from sage_tau2.schemas import Tau2Skill
    events = trajectory.metadata.get('skill_events') or []
    instances = execution_snapshots(events)
    owners = {}
    claims = {}
    for event in events:
        for claim in event.get('call_ownership', []):
            cid = claim.get('call_id')
            owners.setdefault(cid, set()).add(claim.get('execution_id'))
            claims.setdefault(claim.get('execution_id'), set()).add(cid)
    sequence = trajectory.metadata.get('interaction_sequence') or interaction_sequence(trajectory.raw_messages)
    used_checks = set()
    outcomes = []
    for instance in instances:
        iid = instance['execution_id']
        ids = claims.get(iid, set())
        ambiguous = any(len(owners[cid]) != 1 for cid in ids)
        actual = [s for s in trajectory.tool_steps if s.tool_call_id in ids]
        checks = trajectory.metadata.get('action_checks') or []
        bindings = instance.get('bindings') or {}
        scoped = [c for c in checks if all((c.get('action') or {}).get('arguments', {}).get(k, v) == v for k,v in bindings.items())]
        # Do not turn a wrong target into neutral simply by excluding every check.
        if scoped:
            checks = scoped
        clone = deepcopy(trajectory)
        clone.metadata['action_checks'] = checks
        clone.metadata['interaction_sequence'] = [c for c in sequence if c.get('execution_id') == iid]
        for c in clone.metadata['interaction_sequence']:
            c['adopted_skill_ids'] = [instance['skill_id']]
        sk = Tau2Skill(skill_id=instance['skill_id'], skill_name=instance['skill_id'], description='', precondition='',
            expected_effect='', domain=trajectory.domain, action_protocol=[s['instruction'] for s in instance['steps']])
        own_events = [{'adopted_skill_ids': [sk.skill_id], 'tool_calls': [{'id':cid} for cid in ids]}]
        verdict = local_skill_outcome(sk, clone, own_events)
        step_complete = []
        for step in instance['steps']:
            if step['requestor'] == 'user' or step['tool'] is None:
                user_observed = any(c.get('name') == step['tool'] and c.get('result') is not None and not c.get('result_error')
                                    for c in clone.metadata['interaction_sequence'])
                step_complete.append(step['status'] == 'reported' or user_observed)
            else:
                cids = {a['call_id'] for a in step.get('attempts', [])}
                step_complete.append(any(t.tool_call_id in cids and t.result_content is not None and not t.result_error for t in actual))
        if verdict['success'] is True and not all(step_complete):
            verdict = {'success': False if not trajectory.success else None, 'executed': bool(actual),
                       'outcome': 'execution_incomplete'}
        check_keys = {json.dumps(c['action'], sort_keys=True) for c in checks
                      if c.get('action_match') and any(s.name == c['action'].get('name') and all(
                          s.arguments.get(k) == v for k,v in c['action'].get('arguments', {}).items()) for s in actual)}
        if ambiguous:
            verdict = {'success': None, 'executed': bool(actual), 'outcome': 'ambiguous_call_owner'}
        elif instance.get('violations'):
            verdict = {'success': False, 'executed': bool(actual), 'outcome': 'execution_contract_violation'}
        elif verdict['success'] is True and check_keys & used_checks:
            verdict = {'success': None, 'executed': bool(actual), 'outcome': 'verification_already_attributed'}
        elif verdict['success'] is True:
            used_checks.update(check_keys)
            verdict['outcome'] = 'verified_local_operation'
        for user_call in clone.metadata['interaction_sequence']:
            if user_call.get('requestor') == 'user' and verdict['success'] is True:
                key = json.dumps({'requestor':'user', 'name':user_call.get('name'), 'arguments':user_call.get('arguments')}, sort_keys=True)
                if key in used_checks:
                    verdict = {'success': None, 'executed': True, 'outcome': 'verification_already_attributed'}
                else:
                    used_checks.add(key)
        actors = {a['actor'] for step in instance['steps'] for a in step.get('attempts', [])}
        actors.update(s['guidance_actor'] for s in instance['steps'] if s.get('guidance_actor'))
        outcomes.append({**verdict, 'execution_id': iid, 'skill_id': sk.skill_id,
                         'skill_version': instance.get('skill_version'), 'actors': sorted(actors),
                         'status': instance['status'], 'task_id': trajectory.task_id, 'task_success': trajectory.success})
    return outcomes


def apply_execution_credit(skills, trajectory, policy):
    from sage_tau2.credit import initialize_credit, _smoothed, _maybe_promote_or_prune
    events = trajectory.metadata.get('skill_events') or []
    offered = {sid for e in events for sid in e.get('offered_skill_ids', [])}
    outcome_list = execution_outcomes(trajectory)
    result = []
    for skill in skills:
        credit = initialize_credit(skill)
        seen = set(credit.get('evaluated_execution_ids') or [])
        episode_key = str(trajectory.evidence_id)
        episode_seen = set(credit.get('offered_evidence_ids') or [])
        if skill.skill_id in offered and episode_key not in episode_seen:
            credit['offered_episodes'] = int(credit.get('offered_episodes') or 0) + 1
            credit['offered_evidence_ids'] = sorted(episode_seen | {episode_key})
        for outcome in outcome_list:
            if outcome['skill_id'] != skill.skill_id or outcome['execution_id'] in seen:
                continue
            seen.add(outcome['execution_id'])
            credit['adopted_instances'] = int(credit.get('adopted_instances') or 0) + 1
            credit['executed_instances'] = int(credit.get('executed_instances') or 0) + int(outcome['executed'])
            if outcome['success'] is not None:
                credit['uses'] += 1
                credit['online_uses'] += 1
                if outcome['success']:
                    credit['successes'] += 1
                    credit['online_successes'] += 1
                    credit['online_fail_streak'] = 0
                else:
                    credit['online_fail_streak'] = int(credit.get('online_fail_streak') or 0) + 1
                credit['score'] = _smoothed(credit['successes'], credit['uses'])
                credit['online_score'] = _smoothed(credit['online_successes'], credit['online_uses'])
                skill.metadata['utility'] = credit['score']
                _maybe_promote_or_prune(skill, policy)
            record = {**outcome, 'online': True}
            credit.setdefault('events', []).append(record)
            credit['events'] = credit['events'][-100:]
            result.append(record)
        credit['evaluated_execution_ids'] = sorted(seen)
    return result
