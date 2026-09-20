"""Reconcile specialists with solution revisions and instance-level performance."""
from copy import deepcopy
import hashlib
import json

from sage_tau2.organization import EXECUTOR_NAME
from sage_tau2.schemas import OrganizationEdit, OrganizationEditType, SkillStatus
from sage_tau2.skill_resolve import resolve_assigned_skills
from sage_tau2.solutions import solution_fingerprint


def specialist_revision(skills):
    payload = sorted((s.skill_id, solution_fingerprint(s)) for s in skills)
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:20]


def reconcile_specialists(organization, archive, skills, *, domain=None):
    """Refresh owned solutions; changed or legacy experts must be admitted again."""
    active = [s for s in skills if s.status not in {SkillStatus.RETIRED, SkillStatus.REJECTED}]
    by_id = {s.skill_id: s for s in active}
    proposals = []
    for agent in organization.agents:
        if agent.name == EXECUTOR_NAME:
            continue
        owned = resolve_assigned_skills(agent.assigned_skills, active)
        domains = set(agent.metadata.get('domains') or [])
        if domain and domains and domain not in domains:
            continue
        cluster = next((c for c in archive.clusters if c.cluster_id == agent.metadata.get('cluster_id')
                        and any(sid in by_id for sid in c.member_skill_ids)), None)
        if cluster is None and owned:
            cluster = archive.cluster_of(owned[0].skill_id)
        if cluster is not None:
            owned = [by_id[sid] for sid in cluster.member_skill_ids if sid in by_id]
        if not owned:
            agent.acting_status = 'dormant'
            agent.shadow_evaluation_record.update(acting_status='dormant', same_skill_probe_passed=False)
            agent.metadata['lifecycle_reason'] = 'no_supported_solution'
            continue
        revision = specialist_revision(owned)
        record = agent.shadow_evaluation_record
        changed = record.get('admitted_solution_revision') != revision
        if agent.acting_status == 'dormant' and not changed:
            continue
        agent.assigned_skills = [s.skill_id for s in owned]
        agent.metadata['solution_revision'] = revision
        agent.metadata['domains'] = sorted({s.domain for s in owned})
        if cluster:
            agent.metadata['cluster_id'] = cluster.cluster_id
            agent.capability_keys = [cluster.capability_key]
        leader = owned[0]
        agent.activation_condition = (leader.metadata.get('execution_contract') or {}).get('condition') or leader.precondition
        agent.role_specification = (
            f"Specialist for solution cluster {agent.metadata.get('cluster_id', '')}. "
            "Follow complete owned contracts and persistent execution steps. "
            "Return observed results, blocked dependencies and unresolved work to Executor.")
        if changed or not record.get('same_skill_probe_passed'):
            record['same_skill_probe_passed'] = False
            record['promotion_probe_passed'] = False
            record['nominate_admit_awaiting_admission'] = True
            agent.acting_status = 'probation'
            record['acting_status'] = 'probation'
            agent.metadata['lifecycle_reason'] = 'solution_changed_or_unvalidated'
            proposals.append(OrganizationEdit(edit_type=OrganizationEditType.ADD_AGENT,
                rationale='Revalidate updated or legacy specialist against same-skill Executor.',
                new_agent=deepcopy(agent), assigned_skill_names=list(agent.assigned_skills),
                cluster_id=agent.metadata.get('cluster_id'), capability_key=leader.capability_key,
                status='revalidation'))
    return proposals


def refresh_instance_statuses(organization, trajectories, policy):
    from sage_tau2.execution_credit import execution_outcomes
    decisions = []
    outcomes = [o for t in trajectories if 'infrastructure' not in str(t.termination_reason or '').lower()
                for o in execution_outcomes(t)]
    for agent in organization.agents:
        if agent.name == EXECUTOR_NAME:
            continue
        record = agent.shadow_evaluation_record
        seen = set(record.get('evaluated_execution_ids') or [])
        played = wins = unknown = 0
        recent = list(record.get('recent_instance_outcomes') or [])
        for outcome in outcomes:
            if agent.name not in outcome['actors'] or outcome['execution_id'] in seen:
                continue
            seen.add(outcome['execution_id'])
            # Handoffs are recorded, but cannot be wholly credited to either actor.
            if outcome['success'] is None or len(outcome['actors']) != 1:
                unknown += 1
                continue
            played += 1
            wins += int(outcome['success'])
            recent.append(bool(outcome['success']))
        if not (played or unknown):
            continue
        record['evaluated_execution_ids'] = sorted(seen)
        record['delegated_verified_games'] = int(record.get('delegated_verified_games') or 0) + played
        record['delegated_local_wins'] = int(record.get('delegated_local_wins') or 0) + wins
        record['delegated_unverified_games'] = int(record.get('delegated_unverified_games') or 0) + unknown
        window = max(1, policy.min_games)
        record['recent_instance_outcomes'] = recent[-window:]
        decision = 'continue_probation'
        if played and len(recent) >= window and sum(recent[-window:]) < policy.min_wins:
            rejected = int(record.get('instance_rejected_windows') or 0) + 1
            record['instance_rejected_windows'] = rejected
            if rejected >= policy.remove_after_rejected_windows:
                agent.acting_status = 'dormant'
                record['acting_status'] = 'dormant'
                record['same_skill_probe_passed'] = False
                agent.metadata['lifecycle_reason'] = 'verified_instance_failures'
                decision = 'dormant'
        elif record.get('same_skill_probe_passed') and len(recent) >= window and sum(recent[-window:]) >= policy.min_wins:
            agent.acting_status = 'accepted'
            record['acting_status'] = 'accepted'
            record['instance_rejected_windows'] = 0
            decision = 'accepted'
        decisions.append({'agent': agent.name, 'decision': decision, 'played': played,
                          'local_wins': wins, 'unverified': unknown, 'mode': 'execution_instances'})
    return decisions
