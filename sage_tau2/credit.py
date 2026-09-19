"""Laplace-smoothed credit for τ² tool-protocol skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sage_tau2.distill import DEFAULT_MIN_SUPPORT_FOR_VERIFIED, normalize_protocol
from sage_tau2.schemas import SkillStatus, Tau2Skill, Tau2Trajectory
from sage_tau2.task_context import episode_scope_from_task_id, skill_credit_should_apply

# Agent writes whose gold action_checks encode arg correctness (e.g. line_id).
_HARD_WRITE_ARG_TOOLS = frozenset(
    {
        "refuel_data",
        "enable_roaming",
        "disable_roaming",
        "send_payment_request",
        "resume_line",
        "suspend_line",
    }
)


@dataclass(slots=True)
class CreditPolicy:
    verify_score: float = 0.60
    prune_score: float = 0.20
    min_uses_for_promotion: int = 1
    min_uses_for_pruning: int = 3
    min_protocol_coverage: float = 0.5
    # Skip credit on irrelevant / partial multi-bug failures.
    gate_irrelevant: bool = True
    # A use also requires every write in the skill's spine to appear in the
    # episode protocol (write-level attribution, not just tool-name coverage).
    require_full_write_spine: bool = True
    # Strong verification bar (support_count). Below this → verified_low_support.
    min_support_for_verified: int = DEFAULT_MIN_SUPPORT_FOR_VERIFIED
    # When action_checks exist, require skill hard writes to arg-match gold.
    require_hard_write_arg_match: bool = True
    # Online-only prune (ignores birth-inflated totals once online uses exist).
    min_online_uses_for_pruning: int = 2
    # Block further inject when online Laplace score falls below this.
    min_online_score_for_inject: float = 0.35
    # Birth seed alone never creates verified_low_support (stay provisional).
    birth_seed_promotes_low_support: bool = False


def _smoothed(successes: int, uses: int) -> float:
    return (successes + 1.0) / (uses + 2.0)


def initialize_credit(skill: Tau2Skill) -> dict[str, Any]:
    existing = skill.metadata.get("skill_credit")
    if isinstance(existing, dict):
        existing.setdefault("online_uses", 0)
        existing.setdefault("online_successes", 0)
        existing.setdefault("online_fail_streak", 0)
        return existing
    credit = {
        "uses": 0,
        "successes": 0,
        "score": _smoothed(0, 0),
        "online_uses": 0,
        "online_successes": 0,
        "online_fail_streak": 0,
        "events": [],
    }
    skill.metadata["skill_credit"] = credit
    skill.metadata["utility"] = credit["score"]
    if skill.status == SkillStatus.CANDIDATE:
        skill.status = SkillStatus.PROVISIONAL
    return credit


def protocol_coverage(skill_protocol: list[str], traj_protocol: list[str]) -> float:
    """Ordered coverage of skill tool-names inside the episode protocol.

    Both sides are canonicalized (collapse consecutive duplicates; no write
    trim). Matching is by tool **name** (args ignored) so distill templates
    still cover raw success traces. Dialogue / branch rows in a mixed
    ``action_protocol`` are ignored.
    """
    from sage_tau2.distill import _tool_name, agent_tool_protocol

    skill_n = normalize_protocol(
        agent_tool_protocol(list(skill_protocol or []))
    )
    traj_n = normalize_protocol(list(traj_protocol or []))
    if not skill_n:
        return 0.0
    if not traj_n:
        return 0.0
    skill_names = [_tool_name(step) for step in skill_n]
    traj_names = [_tool_name(step) for step in traj_n]
    i = 0
    for name in traj_names:
        if i < len(skill_names) and name == skill_names[i]:
            i += 1
    return i / float(len(skill_names))


def skill_hard_write_names(skill: Tau2Skill) -> list[str]:
    """Hard write tool names owned by a skill (primary first when known)."""
    from sage_tau2.distill import _tool_name, agent_tool_protocol

    meta = skill.metadata or {}
    primary = str(meta.get("primary_write") or "").strip()
    out: list[str] = []
    seen: set[str] = set()
    if primary and primary in _HARD_WRITE_ARG_TOOLS:
        out.append(primary)
        seen.add(primary)
    proto = agent_tool_protocol(
        list(meta.get("agent_tool_protocol") or skill.action_protocol or []),
        domain=getattr(skill, "domain", None),
    )
    for step in proto:
        name = _tool_name(step)
        if name in _HARD_WRITE_ARG_TOOLS and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def hard_write_arg_match_score(
    skill: Tau2Skill,
    trajectory: Tau2Trajectory,
) -> float | None:
    """Fraction of skill hard writes with gold ``action_match`` (args correct).

    Returns ``None`` when the episode has no action_checks or the skill has no
    hard writes — callers should leave legacy success logic unchanged.
    ``action_match=True`` already means name+args matched gold (e.g. line_id).
    """
    checks = (trajectory.metadata or {}).get("action_checks")
    if not isinstance(checks, list) or not checks:
        return None
    hard = skill_hard_write_names(skill)
    if not hard:
        return None
    matched: dict[str, bool] = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        action = check.get("action") or {}
        if not isinstance(action, dict):
            continue
        requestor = str(action.get("requestor") or "").lower()
        if requestor == "user":
            continue
        name = str(action.get("name") or "").strip()
        if name not in _HARD_WRITE_ARG_TOOLS:
            continue
        ok = bool(check.get("action_match"))
        matched[name] = bool(matched.get(name)) or ok
    hits = sum(1 for name in hard if matched.get(name) is True)
    return hits / float(len(hard))


def skill_success_modes(skill: Tau2Skill) -> frozenset[str] | None:
    """Return the skill's success_mode allowlist, or None if unset (legacy)."""
    meta = skill.metadata or {}
    raw = meta.get("success_modes")
    if raw is None:
        raw = meta.get("success_mode")
    if raw is None:
        return None
    if isinstance(raw, str):
        modes = {raw.strip()} if raw.strip() else set()
    elif isinstance(raw, (list, tuple, set, frozenset)):
        modes = {str(m).strip() for m in raw if str(m).strip()}
    else:
        return None
    return frozenset(modes) if modes else None


def episode_counts_as_credit_success(
    skill: Tau2Skill,
    trajectory: Tau2Trajectory,
    *,
    require_hard_write_arg_match: bool = True,
) -> bool:
    """Layer credit success by skill success_mode when present.

    When gold ``action_checks`` exist and the skill owns hard writes, also
    require those writes to arg-match (blocks credit success for wrong line_id).
    """
    from sage_tau2.distill import infer_success_mode

    modes = skill_success_modes(skill)
    ep_mode = infer_success_mode(trajectory)
    if modes is not None and ep_mode not in modes:
        return False
    if ep_mode == "solve_write" or (modes is None):
        base = bool(trajectory.success or trajectory.has_positive_db_effect)
    else:
        base = bool(trajectory.success)
    if not base:
        return False
    if not require_hard_write_arg_match:
        return True
    arg_score = hard_write_arg_match_score(skill, trajectory)
    if arg_score is None:
        return True
    return arg_score >= 1.0


def seed_credit_from_birth_support(
    skill: Tau2Skill,
    *,
    policy: CreditPolicy | None = None,
) -> dict[str, Any]:
    """Seed credit from distill support so skills can reach VERIFIED without inject.

    Distill only retains successful episodes. Without birth seeding, a run with
    ``allow_provisional_inject=false`` and ``allow_provisional_org_edits=false``
    deadlocks: provisional skills never get credit uses → never VERIFIED → never
    nominated.

    Birth evidence alone only promotes to full ``verified`` when
    ``support_count >= min_support_for_verified``. Low-n birth stays
    ``provisional`` (unless ``birth_seed_promotes_low_support``), so thin
    distill cards must prove themselves online before looking "verified".
    """
    policy = policy or CreditPolicy()
    credit = initialize_credit(skill)
    support = max(int(skill.support_count or 0), len(skill.evidence_ids or []), 0)
    if support <= 0:
        return credit

    uses = int(credit.get("uses") or 0)
    successes = int(credit.get("successes") or 0)
    if uses < support:
        credit["uses"] = support
        # Birth trajectories were success-gated by distill.
        credit["successes"] = max(successes, support)
        credit["score"] = _smoothed(
            int(credit["successes"]),
            int(credit["uses"]),
        )
        credit.setdefault("online_uses", 0)
        credit.setdefault("online_successes", 0)
        credit.setdefault("online_fail_streak", 0)
        credit["birth_uses"] = support
        credit["birth_successes"] = support
        skill.metadata["utility"] = float(credit["score"])
        meta = dict(skill.metadata or {})
        meta["credit_seeded_from_birth"] = True
        skill.metadata = meta
        min_verified = int(policy.min_support_for_verified)
        score = float(credit["score"])
        if (
            support >= min_verified
            and score >= policy.verify_score
            and skill.status
            in {
                SkillStatus.PROVISIONAL,
                SkillStatus.VERIFIED_LOW_SUPPORT,
                SkillStatus.CANDIDATE,
            }
        ):
            skill.status = SkillStatus.VERIFIED
        elif (
            bool(policy.birth_seed_promotes_low_support)
            and score >= policy.verify_score
            and skill.status
            in {
                SkillStatus.PROVISIONAL,
                SkillStatus.CANDIDATE,
            }
        ):
            skill.status = SkillStatus.VERIFIED_LOW_SUPPORT
            meta = dict(skill.metadata or {})
            meta["verified_low_support"] = True
            meta["support_tier"] = "low"
            skill.metadata = meta
        elif skill.status == SkillStatus.CANDIDATE:
            skill.status = SkillStatus.PROVISIONAL
    return credit


def apply_credit_for_episode(
    skills: list[Tau2Skill],
    trajectory: Tau2Trajectory,
    *,
    injected_skill_ids: list[str] | None = None,
    policy: CreditPolicy | None = None,
) -> list[dict[str, Any]]:
    """Count uses for skills that matched coverage (optionally inject-gated)."""
    policy = policy or CreditPolicy()
    injected = set(injected_skill_ids or [])
    events: list[dict[str, Any]] = []
    episode_scope = episode_scope_from_task_id(trajectory.task_id)
    for skill in skills:
        if skill.status in {SkillStatus.REJECTED, SkillStatus.RETIRED}:
            continue
        credit = initialize_credit(skill)
        # If injection list is provided and non-empty, only credit offered skills.
        if injected and skill.skill_id not in injected:
            continue
        if policy.gate_irrelevant and not skill_credit_should_apply(
            skill,
            domain=trajectory.domain,
            task_id=trajectory.task_id,
            success=bool(trajectory.success or trajectory.has_positive_db_effect),
            episode_scope=episode_scope,
        ):
            continue
        skill_proto = list(
            (skill.metadata or {}).get("agent_tool_protocol")
            or skill.action_protocol
            or []
        )
        coverage = protocol_coverage(skill_proto, trajectory.tool_protocol)
        if coverage < policy.min_protocol_coverage:
            continue
        if policy.require_full_write_spine:
            from sage_tau2.distill import _tool_name, primary_write_names

            skill_writes = set(primary_write_names(skill_proto))
            if skill_writes:
                traj_tool_names = {
                    _tool_name(step)
                    for step in normalize_protocol(
                        list(trajectory.tool_protocol or []),
                        domain=getattr(trajectory, "domain", None),
                    )
                }
                if not skill_writes.issubset(traj_tool_names):
                    continue
        arg_score = hard_write_arg_match_score(skill, trajectory)
        success = episode_counts_as_credit_success(
            skill,
            trajectory,
            require_hard_write_arg_match=bool(policy.require_hard_write_arg_match),
        )
        credit["uses"] = int(credit.get("uses") or 0) + 1
        credit["online_uses"] = int(credit.get("online_uses") or 0) + 1
        if success:
            credit["successes"] = int(credit.get("successes") or 0) + 1
            credit["online_successes"] = int(credit.get("online_successes") or 0) + 1
            credit["online_fail_streak"] = 0
        else:
            credit["online_fail_streak"] = int(credit.get("online_fail_streak") or 0) + 1
        credit["score"] = _smoothed(
            int(credit["successes"]),
            int(credit["uses"]),
        )
        credit["online_score"] = _smoothed(
            int(credit.get("online_successes") or 0),
            int(credit.get("online_uses") or 0),
        )
        skill.metadata["utility"] = float(credit["score"])
        event = {
            "skill_id": skill.skill_id,
            "task_id": trajectory.task_id,
            "coverage": coverage,
            "arg_match": arg_score,
            "success": success,
            "reward": trajectory.reward,
            "success_mode": (trajectory.metadata or {}).get("success_mode"),
            "online": True,
        }
        events_list = credit.setdefault("events", [])
        if isinstance(events_list, list):
            events_list.append(event)
            credit["events"] = events_list[-50:]
        events.append(event)
        _maybe_promote_or_prune(skill, policy)
    return events


def _support_count(skill: Tau2Skill) -> int:
    return max(int(skill.support_count or 0), len(skill.evidence_ids or []), 0)


def _online_score(credit: dict[str, Any]) -> float | None:
    online_uses = int(credit.get("online_uses") or 0)
    if online_uses <= 0:
        return None
    if "online_score" in credit:
        try:
            return float(credit["online_score"])
        except (TypeError, ValueError):
            pass
    return _smoothed(
        int(credit.get("online_successes") or 0),
        online_uses,
    )


def _maybe_promote_or_prune(skill: Tau2Skill, policy: CreditPolicy) -> None:
    credit = skill.metadata.get("skill_credit") or {}
    uses = int(credit.get("uses") or 0)
    score = float(credit.get("score") or 0.0)
    support = _support_count(skill)
    min_verified = int(policy.min_support_for_verified)
    promotable = {
        SkillStatus.PROVISIONAL,
        SkillStatus.VERIFIED_LOW_SUPPORT,
        SkillStatus.CANDIDATE,
    }
    if uses >= policy.min_uses_for_promotion and score >= policy.verify_score:
        if skill.status in promotable:
            online_uses = int(credit.get("online_uses") or 0)
            # Full verified needs support bar. Low-support verified requires
            # at least one online trial (birth alone is not enough).
            if support >= min_verified:
                skill.status = SkillStatus.VERIFIED
            elif online_uses >= 1:
                skill.status = SkillStatus.VERIFIED_LOW_SUPPORT
                meta = dict(skill.metadata or {})
                meta["verified_low_support"] = True
                meta["support_tier"] = "low"
                skill.metadata = meta
    # Prefer online-only evidence for prune when birth inflated totals.
    online_uses = int(credit.get("online_uses") or 0)
    online_score = _online_score(credit)
    prune_by_online = (
        online_uses >= int(policy.min_online_uses_for_pruning)
        and online_score is not None
        and online_score < policy.prune_score
    )
    prune_by_total = (
        uses >= policy.min_uses_for_pruning and score < policy.prune_score
    )
    if (prune_by_online or prune_by_total) and skill.status in {
        SkillStatus.PROVISIONAL,
        SkillStatus.VERIFIED,
        SkillStatus.VERIFIED_LOW_SUPPORT,
    }:
        skill.status = SkillStatus.RETIRED
        meta = dict(skill.metadata or {})
        meta["retired_reason"] = (
            "online_score" if prune_by_online and not prune_by_total else "credit_score"
        )
        skill.metadata = meta


def skill_allowed_for_inject(
    skill: Tau2Skill,
    *,
    allow_provisional: bool = True,
    policy: CreditPolicy | None = None,
) -> bool:
    """Status + online-performance gate before a skill may be injected."""
    policy = policy or CreditPolicy()
    if skill.status in {SkillStatus.REJECTED, SkillStatus.RETIRED}:
        return False
    if skill.status == SkillStatus.VERIFIED:
        allowed = True
    elif allow_provisional and skill.status in {
        SkillStatus.PROVISIONAL,
        SkillStatus.VERIFIED_LOW_SUPPORT,
        SkillStatus.CANDIDATE,
    }:
        allowed = True
    else:
        return False
    if not allowed or not skill.action_protocol:
        return False
    credit = skill.metadata.get("skill_credit") or {}
    online_uses = int(credit.get("online_uses") or 0)
    online_score = _online_score(credit)
    # After enough online fails, stop injecting even if birth score looks fine.
    if (
        online_uses >= int(policy.min_online_uses_for_pruning)
        and online_score is not None
        and online_score < float(policy.min_online_score_for_inject)
    ):
        return False
    return True


def injectable_skills(
    skills: list[Tau2Skill],
    *,
    max_skills: int = 2,
    allow_provisional: bool = True,
    task_id: str = "",
    dedupe_writes: bool = True,
    policy: CreditPolicy | None = None,
) -> list[Tau2Skill]:
    """Rank injectable skills; optionally by episode protocol-coverage.

    When ``task_id`` is set, primary sort key is
    :func:`sage_tau2.task_context.episode_skill_coverage_score` (how the skill's
    own protocol covers hinted writes), then **online** credit when present,
    else birth/total score. With ``dedupe_writes``, at most one skill per
    primary write name is kept so two near-duplicate roaming cards cannot
    crowd out refuel/payment.
    """
    from sage_tau2.task_context import (
        episode_skill_coverage_score,
        skill_write_name,
    )

    policy = policy or CreditPolicy()
    ranked = []
    tid = str(task_id or "").strip()
    for skill in skills:
        if not skill_allowed_for_inject(
            skill, allow_provisional=allow_provisional, policy=policy
        ):
            continue
        credit = skill.metadata.get("skill_credit") or {}
        online_score = _online_score(credit)
        credit_score = (
            float(online_score)
            if online_score is not None
            else float(credit.get("score") or skill.metadata.get("utility") or 0.5)
        )
        # Birth-only provisional / low-support: slight demotion vs online-proven.
        if (
            online_score is None
            and bool((skill.metadata or {}).get("credit_seeded_from_birth"))
            and skill.status != SkillStatus.VERIFIED
        ):
            credit_score *= 0.85
        cover = episode_skill_coverage_score(skill, task_id=tid) if tid else 0.0
        ranked.append((cover, credit_score, skill.support_count, skill))
    ranked.sort(
        key=lambda item: (-item[0], -item[1], -item[2], item[3].skill_name)
    )
    if not dedupe_writes or max_skills <= 0:
        return [skill for _, _, _, skill in ranked[: max(0, max_skills)]]

    chosen: list[Tau2Skill] = []
    seen_writes: set[str] = set()
    for _cover, _credit, _sup, skill in ranked:
        write = skill_write_name(skill) or str(skill.skill_name or skill.skill_id)
        if write in seen_writes:
            continue
        seen_writes.add(write)
        chosen.append(skill)
        if len(chosen) >= max(0, max_skills):
            break
    return chosen
