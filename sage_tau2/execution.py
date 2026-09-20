"""Persistent skill invocations. No benchmark labels enter runtime state."""
from __future__ import annotations

import copy
import json
import re
from uuid import uuid4

TERMINAL = {'protocol_complete', 'abandoned'}


def operation(row):
    text = str(row).strip()
    user = text.lower().startswith('guide user:')
    if user:
        text = text.split(':', 1)[1].strip()
    match = re.match(r'([\w]+)\s*(?:\(|$)', text)
    return ('user' if user else 'assistant', match.group(1) if match else None)


def start_execution(skill, actor, objective=''):
    from sage_tau2.solutions import solution_fingerprint
    return {'execution_id': str(uuid4()), 'skill_id': skill.skill_id,
            'skill_version': solution_fingerprint(skill), 'actor': actor,
            'objective': objective, 'status': 'running', 'bindings': {}, 'violations': [],
            'steps': [{'index': i, 'instruction': row, 'requestor': operation(row)[0],
                       'tool': operation(row)[1], 'status': 'pending', 'attempts': [], 'evidence': []}
                      for i, row in enumerate(skill.action_protocol)],
            'return_record': None}


def next_step(instance):
    return next((s for s in instance['steps'] if s['status'] not in {'observed', 'reported'}), None)


def refresh_status(instance):
    step = next_step(instance)
    if step is None:
        instance['status'] = 'protocol_complete'
        instance['return_record'] = {'status': 'protocol_complete', 'effect_verified': False,
                                     'remaining_steps': [], 'reason': 'All steps have observations; effect requires verification.'}
    elif step['status'] == 'failed':
        instance['status'] = 'blocked'
    elif step['status'] in {'waiting_user', 'user_feedback'}:
        instance['status'] = 'waiting_user'
    else:
        instance['status'] = 'running'


def observe(instances, messages, observation_ref):
    """Advance tool steps only on matching returned call IDs; user text awaits review."""
    for inst in instances:
        if inst['status'] in TERMINAL:
            continue
        for message in messages:
            if message.get('role') == 'tool':
                cid = str(message.get('tool_call_id') or message.get('id') or '')
                for step in inst['steps']:
                    attempt = next((a for a in step['attempts'] if a['call_id'] == cid), None)
                    if attempt is None or 'result' in attempt:
                        continue
                    attempt['result'] = message.get('content')
                    attempt['error'] = bool(message.get('error'))
                    try:
                        payload = json.loads(message.get('content') or '')
                        attempt['error'] |= isinstance(payload, dict) and bool(payload.get('error'))
                    except (ValueError, TypeError):
                        pass
                    step['evidence'].append({'ref': cid, 'source': 'tool', 'error': attempt['error']})
                    step['status'] = 'failed' if attempt['error'] else 'observed'
            elif message.get('role') == 'user' and message.get('content') and inst['status'] == 'waiting_user':
                step = next_step(inst)
                if step and step['status'] in {'waiting_user', 'user_feedback'}:
                    step['status'] = 'user_feedback'
                    step['feedback_ref'] = observation_ref
                    step['feedback'] = str(message['content'])
        if inst['status'] != 'paused':
            refresh_status(inst)


def apply_feedback(instances, update):
    """A model may acknowledge a real user report, never invent tool completion."""
    if not isinstance(update, dict):
        return False
    inst = next((i for i in instances if i['execution_id'] == update.get('execution_id')), None)
    if not inst or inst['status'] in TERMINAL:
        return False
    step = next_step(inst)
    if not step or step['status'] != 'user_feedback' or step['index'] != update.get('step_index'):
        return False
    if not update.get('evidence_ref') or update['evidence_ref'] != step.get('feedback_ref'):
        return False
    if update.get('verdict') not in {'observed', 'failed'}:
        return False
    step['evidence'].append({'ref': update['evidence_ref'], 'source': 'user_report', 'verdict': update['verdict']})
    step['status'] = 'reported' if update['verdict'] == 'observed' else 'failed'
    refresh_status(inst)
    return True


def choose_execution(instances, *, skill, actor, execution_id='', objective=''):
    if execution_id:
        inst = next((i for i in instances if i['execution_id'] == execution_id
                     and i['skill_id'] == skill.skill_id and i['status'] not in TERMINAL), None)
        if inst is None:
            return None
    else:
        inst = next((i for i in reversed(instances) if i['skill_id'] == skill.skill_id and i['status'] not in TERMINAL), None)
        if inst is None:
            inst = start_execution(skill, actor, objective)
            instances.append(inst)
    for other in instances:
        if other is not inst and other['status'] not in TERMINAL:
            other['status'] = 'paused'
    inst['actor'] = actor
    refresh_status(inst)
    return inst


def record_action(instance, message):
    """Own only calls to the ready step; parallel dependent operations are deviations."""
    claims = []
    if instance is None:
        return claims
    from sage_tau2.distill import primary_write_names
    writes = set(primary_write_names([s['instruction'] for s in instance['steps']]))
    step = next_step(instance)
    for call in message.get('tool_calls') or []:
        fn = call.get('function') or call
        cid = call.get('id')
        args = fn.get('arguments') or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if not (step and step['requestor'] == 'assistant' and step['tool'] == fn.get('name')
                and step['status'] in {'pending', 'failed'} and cid):
            instance['violations'].append({'call_id': cid, 'reason': 'call_outside_ready_step', 'tool': fn.get('name')})
            continue
        if fn.get('name') in writes:
            for key, value in args.items():
                if key.endswith('_id'):
                    old = instance['bindings'].get(key)
                    if old is not None and old != value:
                        instance['violations'].append({'call_id': cid, 'reason': 'object_changed', 'slot': key})
                    instance['bindings'].setdefault(key, value)
        step['attempts'].append({'call_id': cid, 'arguments': args, 'actor': instance['actor']})
        step['status'] = 'inflight'
        claims.append({'call_id': cid, 'execution_id': instance['execution_id'],
                       'skill_id': instance['skill_id'], 'step_index': step['index']})
    if step and not message.get('tool_calls') and message.get('content') and (step['requestor'] == 'user' or step['tool'] is None):
        step['status'] = 'waiting_user'
        step['guidance'] = message['content']
        step['guidance_actor'] = instance['actor']
    refresh_status(instance)
    return claims


def execution_prompt(instances):
    if not instances:
        return ''
    # Preserve every unfinished step, not a truncated protocol prefix.
    rows = []
    for inst in instances:
        if inst['status'] in TERMINAL:
            rows.append({k: inst[k] for k in ('execution_id','skill_id','status','return_record')})
            continue
        row = {k: inst[k] for k in ('execution_id','skill_id','actor','objective','status','bindings')}
        row['steps'] = [{k:v for k,v in s.items() if k != 'attempts'} for s in inst['steps']]
        rows.append(row)
    return ('Persistent execution state. Resume by execution_id. Observed/reported steps must not be repeated. '
            'Only the first unresolved step is ready. Do not skip dependencies. A user report is not a verified environment effect.\n'
            + json.dumps(rows, ensure_ascii=False))


def execution_snapshots(events):
    result = {}
    for event in events:
        for instance in event.get('executions', []):
            result[instance['execution_id']] = copy.deepcopy(instance)
    return list(result.values())
