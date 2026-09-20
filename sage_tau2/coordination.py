"""Executor-controlled, sequential specialist turns on one shared environment."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from sage_tau2.schemas import AgentSpec, Tau2Skill


@dataclass
class TurnPlan:
    actor: str = 'Executor'
    subtask: str = ''
    adopted_skill_ids: list[str] = field(default_factory=list)
    reason: str = ''
    expected_result: str = ''
    execution_id: str = ''
    disposition: str = 'continue'
    step_update: dict = field(default_factory=dict)


def parse_plan(text: str, *, actors: set[str], skill_ids: set[str], fallback: str = 'Executor') -> TurnPlan:
    """Reject unknown actors/skills and malformed plans; never execute model code."""
    try:
        text = text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0]
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('plan must be an object')
        actor = str(data.get('actor') or fallback)
        if actor not in actors:
            raise ValueError('unavailable actor')
        ids = data.get('adopted_skill_ids', [])
        if not isinstance(ids, list):
            raise ValueError('skills must be a list')
        return TurnPlan(actor=actor, subtask=str(data.get('subtask') or '')[:1000],
                        adopted_skill_ids=list(dict.fromkeys(str(x) for x in ids if str(x) in skill_ids))[:1],
                        reason=str(data.get('reason') or '')[:800],
                        expected_result=str(data.get('expected_result') or '')[:800],
                        execution_id=str(data.get('execution_id') or ''),
                        disposition=data.get('disposition') if data.get('disposition') in {'continue','pause','abandon'} else 'continue',
                        step_update=data.get('step_update') if isinstance(data.get('step_update'), dict) else {})
    except (ValueError, TypeError):
        return TurnPlan(actor=fallback, reason='invalid_plan_fallback')


def coordination_prompt(*, context: str, cards: str, agents: list[AgentSpec], owner: str) -> str:
    roster = [{'name': a.name, 'role': a.role_specification, 'skills': a.assigned_skills,
               'capabilities': a.capability_keys} for a in agents]
    return (
        'You are Executor coordinating a customer-service task. Choose the next ONE turn. '
        'You remain responsible for the whole task. A specialist performs a bounded subtask '
        'and returns control after its turn; you will see actual user/tool feedback before routing again. '
        'You may select different specialists on later turns, or keep control. Do not repeat completed '
        'operations. Use observed evidence, not assumptions, to bind objects and assess prerequisites. '
        'Keep an unfinished execution across turns by execution_id. Only one skill execution owns a turn. '
        'Adopt at most one skill; pause other executions explicitly when switching. An empty selection continues '
        'the current execution unless disposition is pause or abandon. Do not redo observed steps or skip dependencies. '
        'For a waiting user step, review the actual feedback and set step_update with execution_id, step_index, '
        'verdict (observed or failed), and the exact feedback_ref as evidence_ref. Tool steps advance only from tool results. '
        'A proposed expected result is not a verified result. Domain policy remains binding. '
        'Return ONLY JSON: {"actor":"name", "subtask":"next bounded objective", '
        '"adopted_skill_ids":[], "execution_id":"", "disposition":"continue", "step_update":{}, '
        '"reason":"evidence for choice", "expected_result":"observable check"}.\n'
        f'Owner/fallback: {owner}\nAvailable actors: {json.dumps(roster, ensure_ascii=False)}\n'
        f'{cards}\nCurrent task, history, policy and tool evidence:\n{context}'
    )
