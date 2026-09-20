"""Evidence-based execution contracts and per-turn learning records.

No task-family policy or benchmark answer is exposed to the actor.
"""
from __future__ import annotations

import json
import re
from typing import Any

from sage_tau2.schemas import Tau2Skill


def contract_for_skill(skill: Tau2Skill) -> dict[str, Any]:
    saved = (skill.metadata or {}).get('execution_contract') or {}
    return {
        'condition': saved.get('condition') or skill.precondition or 'Only when current evidence establishes a need for this capability.',
        'verification': saved.get('verification') or (
            'Check actual tool results for errors and the intended state change on the bound object. '
            'For user actions, obtain their result before dependent steps. A successful call alone does not prove task completion.'
        ),
        'stop_condition': saved.get('stop_condition') or (
            'If prerequisites are unknown, request evidence; if inapplicable, skip. '
            'On error or unresolved symptoms, return the result and remaining problem to Executor.'
        ),
        'bindings': saved.get('bindings') or {},
        'observed_conditions': saved.get('observed_conditions') or [],
    }


def _leaf_values(value, path="result"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaf_values(item, f"{path}.{key}")
    elif isinstance(value, list):
        for item in value:
            yield from _leaf_values(item, f"{path}[]")
    else:
        yield path, value


def learn_contract(skill: Tau2Skill, trajectories: list) -> dict[str, Any]:
    """Learn input sources and observable checks; never store concrete identities.

    These are observational execution contracts, not causal preconditions inferred
    from a single win. Missing sources explicitly require fresh evidence.
    """
    from sage_tau2.distill import primary_write_names
    bindings: dict[str, set[str]] = {}
    reads = set()
    condition_sets = []
    writes = set(primary_write_names(skill.action_protocol))
    for traj in trajectories:
        observations = []
        conditions = {}
        target_slots = {}
        for operation in traj.tool_steps:
            if operation.name in writes:
                for key, value in operation.arguments.items():
                    if key == 'id' or key.endswith('_id'):
                        target_slots.setdefault(str(value), set()).add(key)
        seen_write = False
        for call in traj.tool_steps:
            seen_write = seen_write or call.name in writes
            for key, value in call.arguments.items():
                if key == 'id' or key.endswith('_id'):
                    sources = {f"{name}.{path}" for name, path, old in observations
                               if value is not None and old == value}
                    if not sources:
                        sources = {'current user input or fresh tool evidence; verify object identity'}
                    bindings.setdefault(f'{call.name}.{key}', set()).update(sources)
            if call.name not in writes and not call.result_error:
                reads.add(call.name)
            if not call.result_error and call.result_content:
                try:
                    result = json.loads(call.result_content)
                except (ValueError, TypeError):
                    continue
                leaves = list(_leaf_values(result))
                observations.extend((call.name, path, value) for path, value in leaves)
                if not seen_write:
                    slots = sorted({slot for key, value in call.arguments.items()
                                    if key == 'id' or key.endswith('_id')
                                    for slot in target_slots.get(str(value), [])})
                    # Do not combine observations from different objects into
                    # contradictory unbound conditions.
                    if slots:
                        scope = f"{call.name}(target=<{','.join(slots)}>)"
                        for path, value in leaves:
                            if isinstance(value, bool) or (path.endswith('.status') and isinstance(value, str)):
                                conditions.setdefault(f"{scope}.{path}", set()).add(json.dumps(value))
        condition_sets.append({f"{key} == {next(iter(values))}" for key, values in conditions.items() if len(values) == 1})
    base = contract_for_skill(skill)
    base['observed_conditions'] = sorted(set.intersection(*condition_sets)) if condition_sets else []
    base['bindings'] = {k: sorted(v) for k, v in bindings.items()}
    base['condition'] = (
        f"The current request requires {skill.capability_key or skill.skill_name}; "
        "establish the target object and applicable state from current user/tool evidence. "
        + (f"Supporting traces obtained evidence with {', '.join(sorted(reads))}. " if reads else '')
        + "Do not infer applicability from a task label or from this card's presence."
    )
    base['verification'] = (
        (f"Check {', '.join(sorted(writes))} results on the same bound object; " if writes else '')
        + "observe the intended state change or a subsequent user/tool check. "
        "User actions must return feedback before dependent operations. "
        "If the local effect cannot be checked, report unverified rather than success."
    )
    base['evidence_ids'] = sorted({t.evidence_id for t in trajectories})
    base['verification_tools'] = sorted(writes)
    base['version'] = 2
    return base


def skill_events_from_messages(messages: list[dict]) -> list[dict] | None:
    events = []
    known = False
    for message in messages:
        raw = message.get('raw_data') or {}
        if isinstance(raw, dict) and isinstance(raw.get('sage_skill_event'), dict):
            known = True
            events.append(raw['sage_skill_event'])
    return events if known else None


def skill_events_from_steps(steps: list[dict]) -> list[dict] | None:
    events = [s['skill_event'] for s in steps if isinstance(s.get('skill_event'), dict)]
    return events if events else None


def observed_ledger(messages: list[dict], *, max_chars: int = 10000) -> str:
    """Persist tool-backed facts beyond the sliding dialogue window.

    Retain the most recent result for each tool+argument set, with exact values.
    Never infer task state from benchmark metadata or a hidden task ID.
    """
    calls: dict[str, dict] = {}
    results: dict[str, str] = {}
    for m in messages:
        for call in m.get('tool_calls') or []:
            fn = call.get('function') or call
            args = fn.get('arguments', {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            calls[str(call.get('id') or '')] = {'tool': fn.get('name'), 'arguments': args}
        if m.get('role') == 'tool':
            call = calls.get(str(m.get('tool_call_id') or m.get('id') or ''))
            if not call:
                continue
            key = json.dumps(call, ensure_ascii=False, sort_keys=True)
            result = m.get('content')
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except ValueError:
                    pass
            row = json.dumps({'call': call, 'result': result, 'error': m.get('error')}, ensure_ascii=False)
            results.pop(key, None)
            results[key] = row
    chosen = []
    used = 0
    for row in reversed(list(results.values())):
        if used + len(row) + 1 > max_chars:
            continue
        chosen.append(row)
        used += len(row) + 1
    if not chosen:
        return ''
    return 'Observed tool evidence (latest per call; older facts may be stale after mutations):\n' + '\n'.join(reversed(chosen))


def local_skill_outcome(skill, trajectory, events: list[dict]) -> dict:
    """Attribute local writes to actually adopted turns; gold is evaluator-only.

    Unknown local effects are neutral, never silently promoted using whole-task reward.
    """
    if any(skill.skill_id in e.get('adopted_skill_ids', []) and len(e.get('adopted_skill_ids', [])) > 1 for e in events):
        return {'success': None, 'executed': False, 'outcome': 'ambiguous_legacy_adoption'}
    from sage_tau2.distill import primary_write_names
    required = set(primary_write_names(skill.action_protocol))
    user_required = {row.split(':', 1)[1].strip().split('(', 1)[0]
                     for row in skill.action_protocol if row.lower().startswith('guide user:')}
    ids = {c.get('id') for e in events if skill.skill_id in e.get('adopted_skill_ids', [])
           for c in e.get('tool_calls', []) if c.get('id')}
    actual = [s for s in trajectory.tool_steps if s.tool_call_id in ids and s.name in required]
    sequence = trajectory.metadata.get('interaction_sequence') or interaction_sequence(trajectory.raw_messages)
    user_calls = [c for c in sequence if c.get('requestor') == 'user'
                  and c.get('name') in user_required
                  and skill.skill_id in c.get('adopted_skill_ids', [])]
    executed = bool(actual or user_calls)
    had_error = any(s.result_error for s in actual) or any(c.get('result_error') for c in user_calls)
    checks = [c for c in (trajectory.metadata.get('action_checks') or [])
              if ((c.get('action') or {}).get('requestor') == 'assistant'
                  and (c.get('action') or {}).get('name') in required)
              or ((c.get('action') or {}).get('requestor') == 'user'
                  and (c.get('action') or {}).get('name') in user_required)]
    matched = []
    for check in checks:
        action = check['action']
        args = action.get('arguments') or {}
        compare = action.get('compare_args')
        keys = compare if isinstance(compare, list) else list(args)
        if action.get('requestor') == 'assistant':
            own_match = any(s.name == action.get('name') and not s.result_error and s.result_content is not None
                            and all(s.arguments.get(k) == args.get(k) for k in keys) for s in actual)
        else:
            own_match = any(c.get('name') == action.get('name') and not c.get('result_error') and c.get('result') is not None
                            and all(c.get('arguments', {}).get(k) == args.get(k) for k in keys) for c in user_calls)
        matched.append(bool(check.get('action_match')) and own_match)
    if matched and not all(matched):
        return {'success': False, 'executed': executed, 'outcome': 'missing_or_wrong_operation'}
    expected = {('assistant', n) for n in required} | {('user', n) for n in user_required}
    checked = {(c['action'].get('requestor'), c['action'].get('name')) for c in checks}
    if matched and expected <= checked:
        return {'success': True, 'executed': executed, 'outcome': 'verified_local_effect'}
    if had_error:
        return {'success': False, 'executed': executed, 'outcome': 'tool_error'}
    performed = {('assistant', s.name) for s in actual} | {('user', c['name']) for c in user_calls}
    if expected and not expected <= performed and not trajectory.success:
        return {'success': False, 'executed': executed, 'outcome': 'adopted_but_incomplete'}
    return {'success': None, 'executed': executed, 'outcome': 'local_effect_unverified'}


def interaction_sequence(messages: list[dict]) -> list[dict]:
    """Tool call order with requestor, for distillation; no synthetic branches."""
    result = []
    adopted = []
    execution_id = None
    results = {str(m.get('tool_call_id') or m.get('id') or ''): m for m in messages if m.get('role') == 'tool'}
    for m in messages:
        if m.get('role') == 'assistant':
            raw = m.get('raw_data') or {}
            event = raw.get('sage_skill_event', {}) if isinstance(raw, dict) else {}
            adopted = event.get('adopted_skill_ids', [])
            execution_id = event.get('execution_id')
        for call in m.get('tool_calls') or []:
            fn = call.get('function') or call
            args = fn.get('arguments') or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            tool_result = results.get(str(call.get('id') or ''), {})
            result.append({'name': fn.get('name'), 'requestor': call.get('requestor') or m.get('role'),
                           'arguments': args, 'id': call.get('id'), 'adopted_skill_ids': list(adopted), 'execution_id': execution_id,
                           'result': tool_result.get('content'), 'result_error': bool(tool_result.get('error'))})
    return result


def interleave_evidence_guides(protocol: list[str], trajectories: list) -> list[str]:
    """Restore cross-role ordering using the first supporting observed sequence.

    Each source instruction is retained once; no reordering is guessed for legacy data.
    """
    def signature(text):
        if text.lower().startswith('guide user:'):
            return ('user', text.split(':', 1)[1].strip().split('(', 1)[0].strip())
        return ('assistant', text.split('(', 1)[0].strip())
    from sage_tau2.trajectory import canonicalize_tool_step
    from sage_tau2.distill import bind_protocol_slots
    # Keep all occurrences (e.g. customer lookup then line lookup), including
    # user actions interleaved between writes. Pick one observed complete path.
    ordered_trajs = sorted(trajectories, key=lambda t: len(t.tool_steps))
    for traj in ordered_trajs:
        seq = traj.metadata.get('interaction_sequence') or interaction_sequence(traj.raw_messages)
        seq = [e for e in seq if not e.get('result_error')]
        wanted = {signature(p) for p in protocol}
        available = {(e.get('requestor'), e.get('name')) for e in seq}
        if not seq or not wanted <= available:
            continue
        ordered = []
        observations = []
        for event in seq:
            match = (event.get('requestor'), event.get('name'))
            prior = list(observations)
            payload = event.get('result')
            if payload and not event.get('result_error'):
                try:
                    parsed = json.loads(payload) if isinstance(payload, str) else payload
                    observations.extend((event['name'], path, value) for path, value in _leaf_values(parsed))
                except ValueError:
                    pass
            if match not in wanted:
                continue
            row = canonicalize_tool_step(event['name'], event.get('arguments') or {})
            bare_id = (event.get('arguments') or {}).get('id')
            if bare_id is not None:
                sources = sorted({f"{name}.{path}" for name, path, value in prior if bare_id == value})
                if sources:
                    row = re.sub(r'(?<![\w])id=\?', 'id=<from ' + ' or '.join(sources) + '; match requested object>', row)
            if event.get('requestor') == 'user':
                row = 'Guide user: ' + row
            ordered.append(row)
        return bind_protocol_slots(ordered)
    return protocol
