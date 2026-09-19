"""Episode task/domain extraction for sage_tau2 (no τ² runtime imports in helpers)."""

from __future__ import annotations

import os
import re
from typing import Any

# Primary write tools used as routing scopes (mirrors distill._WRITE_OPS).
WRITE_TOOL_NAMES = frozenset(
    {
        "cancel_reservation",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
        "book_reservation",
        "send_certificate",
        "cancel_pending_order",
        "modify_pending_order_items",
        "modify_pending_order_address",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
        "enable_roaming",
        "disable_roaming",
        "refuel_data",
        "resume_line",
        "suspend_line",
        "send_payment_request",
        "get_bills_for_customer",
        "toggle_airplane_mode",
        "toggle_wifi_calling",
        "toggle_data",
        "toggle_roaming",
        "reset_apn_settings",
        "reseat_sim_card",
        "reboot_device",
        "grant_app_permission",
        "set_network_mode_preference",
    }
)

# Telecom (and similar) task ids look like: "[mms_issue]airplane_mode_on|..."
# These brackets are benchmark-native scopes — analogous to ALFWorld task_family
# from gamefile — not hand-written expert policies.
_BRACKET_SCOPE_RE = re.compile(r"^\[([^\]]+)\]")
# Strip trailing persona / suffix tags from telecom task ids.
_TASK_ID_SUFFIX_RE = re.compile(r"\[(?:PERSONA|persona):[^\]]*\]")


def _normalized_scope(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def episode_scope_from_task_id(task_id: str) -> str:
    """Extract a stable episode scope label from a τ² task id when present."""
    tid = str(task_id or "").strip()
    if not tid:
        return ""
    match = _BRACKET_SCOPE_RE.match(tid)
    if match:
        return _normalized_scope(match.group(1))
    return ""


def bug_tags_from_task_id(task_id: str) -> list[str]:
    """Parse benchmark bug tags after the ``[scope]`` prefix (telecom-style).

    Example::
        [mobile_data_issue]airplane_mode_on|bad_vpn[PERSONA:Easy]
        -> ["airplane_mode_on", "bad_vpn"]
    """
    tid = str(task_id or "").strip()
    if not tid:
        return []
    match = _BRACKET_SCOPE_RE.match(tid)
    if not match:
        return []
    rest = tid[match.end() :]
    rest = _TASK_ID_SUFFIX_RE.sub("", rest).strip()
    if not rest:
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for part in rest.split("|"):
        tag = str(part or "").strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def normalize_bug_tag(tag: str) -> str:
    """Collapse near-equivalent telecom bug tags for inject gating.

    Distill evidence often sees ``user_abroad_roaming_enabled_off`` while a
    later episode uses ``user_abroad_roaming_disabled_on`` — both are abroad
    roaming root-causes and should share the same skill activation family.
    """
    t = str(tag or "").strip().lower()
    if not t:
        return ""
    if t.startswith("user_abroad_roaming_"):
        return "user_abroad_roaming_*"
    if t in {
        "break_app_sms_permission",
        "break_app_storage_permission",
        "break_app_both_permissions",
    }:
        return "break_app_permission_*"
    if t in {"break_apn_mms_setting", "break_apn_settings", "bad_apn"}:
        return "break_apn_*"
    if t in {"unseat_sim_card", "lock_sim_card_pin", "sim_locked"}:
        return "sim_seating_*"
    return t


def normalize_bug_tags(tags: list[str] | set[str] | tuple[str, ...] | None) -> set[str]:
    """Set of :func:`normalize_bug_tag` over ``tags`` (drops empties)."""
    out: set[str] = set()
    for tag in tags or []:
        norm = normalize_bug_tag(str(tag))
        if norm:
            out.add(norm)
    return out


def capability_scope_from_key(capability_key: str) -> str:
    """Normalize a distilled capability key to a routing scope label."""
    key = str(capability_key or "").strip()
    if key.startswith("tau2."):
        key = key.split(".", 1)[1]
    return _normalized_scope(key)


def probe_scopes_for_capability_keys(capability_keys: list[str]) -> list[str]:
    """Routing scopes used to filter Spec-vs-Exec probe tasks."""
    scopes: list[str] = []
    seen: set[str] = set()
    for key in capability_keys or []:
        scope = capability_scope_from_key(key)
        if scope and scope not in seen:
            seen.add(scope)
            scopes.append(scope)
    return scopes


def probe_scopes_from_skill_metadata(skill: Any) -> list[str]:
    """Extra probe scopes from distill metadata (``primary_task_family``)."""
    if skill is None:
        return []
    meta = skill if isinstance(skill, dict) else getattr(skill, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}
    scopes: list[str] = []
    seen: set[str] = set()
    primary = meta.get("primary_task_family")
    if primary:
        scope = _normalized_scope(primary)
        if scope and scope not in seen:
            seen.add(scope)
            scopes.append(scope)
    for fam in meta.get("task_families") or []:
        scope = _normalized_scope(fam)
        if scope and scope not in seen:
            seen.add(scope)
            scopes.append(scope)
    return scopes


def task_matches_probe_scope(task: Any, scopes: set[str]) -> bool:
    """True when the episode routing scope is in ``scopes``."""
    if not scopes:
        return False
    return episode_routing_scope(task) in scopes


def _task_evaluation_actions(task: Any) -> list[dict[str, Any]]:
    if task is None:
        return []
    if isinstance(task, dict):
        criteria = task.get("evaluation_criteria") or {}
    else:
        criteria = getattr(task, "evaluation_criteria", None)
    if criteria is None:
        return []
    if isinstance(criteria, dict):
        actions = criteria.get("actions") or []
    else:
        actions = getattr(criteria, "actions", None) or []
    out: list[dict[str, Any]] = []
    for action in actions:
        if isinstance(action, dict):
            out.append(action)
        else:
            out.append(
                {
                    "requestor": getattr(action, "requestor", None),
                    "name": getattr(action, "name", None),
                }
            )
    return out


def episode_routing_scope(task: Any) -> str:
    """Single routing scope for an episode (ALFWorld ``task_family`` analogue).

    Priority:
    1. Primary assistant write tool from task ``evaluation_criteria`` (retail/airline).
    2. Bracket scope from telecom-style task ids.
    """
    task_id = ""
    if isinstance(task, dict):
        task_id = str(task.get("id") or "")
    elif task is not None:
        task_id = str(getattr(task, "id", "") or "")

    for action in _task_evaluation_actions(task):
        requestor = str(action.get("requestor") or "").strip().lower()
        if requestor not in {"assistant", "agent"}:
            continue
        name = _normalized_scope(action.get("name"))
        if name in {_normalized_scope(item) for item in WRITE_TOOL_NAMES}:
            return name

    return episode_scope_from_task_id(task_id)


def domain_from_task(task: Any) -> str:
    """Resolve episode domain from a τ² Task dict/object."""
    if task is None:
        return ""
    if isinstance(task, dict):
        top = str(task.get("domain") or "").strip()
        if top:
            return top
        scenario = task.get("user_scenario")
    else:
        top = str(getattr(task, "domain", None) or "").strip()
        if top:
            return top
        scenario = getattr(task, "user_scenario", None)

    if isinstance(scenario, dict):
        instr = scenario.get("instructions")
        if isinstance(instr, dict):
            dom = str(instr.get("domain") or "").strip()
            if dom:
                return dom
    elif scenario is not None:
        instr = getattr(scenario, "instructions", None)
        if instr is not None:
            dom = str(getattr(instr, "domain", None) or "").strip()
            if dom:
                return dom
            if isinstance(instr, dict):
                dom = str(instr.get("domain") or "").strip()
                if dom:
                    return dom
    return ""


def resolve_episode_domain(task_domain: str) -> str:
    """Task metadata first, then runner-provided ``SAGE_TAU2_DOMAIN``."""
    task_domain = str(task_domain or "").strip()
    if task_domain:
        return task_domain
    env_domain = str(os.environ.get("SAGE_TAU2_DOMAIN") or "").strip()
    if env_domain:
        return env_domain
    return ""


def canonical_domain(domain: str | None) -> str:
    """Normalize domain aliases (``telecom-workflow`` → ``telecom``)."""
    d = str(domain or "").strip().lower()
    if d in {"telecom-workflow", "telecom_workflow"}:
        return "telecom"
    return d


def domains_match(a: str | None, b: str | None) -> bool:
    ca = canonical_domain(a)
    cb = canonical_domain(b)
    return bool(ca) and ca == cb


_DESC_LABEL_RE = re.compile(
    r"^(?:Purpose|Notes|Summary|Relevant policies)\s*:\s*",
    re.IGNORECASE,
)


def clean_task_phrase(text: str) -> str:
    """Normalize whitespace and strip τ² Description ``__str__`` labels."""
    cleaned = " ".join(str(text or "").split()).strip()
    # Description.__str__ emits "Purpose: ..."; never feed that into templates.
    while True:
        nxt = _DESC_LABEL_RE.sub("", cleaned).strip()
        if nxt == cleaned:
            break
        cleaned = nxt
    return cleaned


def _field(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def flatten_task_description(desc: Any) -> str:
    """Extract purpose/notes/summary without using Description.__str__."""
    if desc is None:
        return ""
    if isinstance(desc, str):
        return clean_task_phrase(desc)
    purpose = _field(desc, "purpose")
    notes = _field(desc, "notes")
    summary = _field(desc, "summary")
    if purpose or notes or summary:
        return clean_task_phrase(
            " ".join(str(p) for p in (purpose, notes, summary) if p)
        )
    # Last resort: string may still be labeled; clean it.
    return clean_task_phrase(str(desc))


def scenario_text_from_task(task: Any) -> str:
    """Customer-facing scenario text (reason / instructions / known_info)."""
    if task is None:
        return ""
    scenario = (
        task.get("user_scenario")
        if isinstance(task, dict)
        else getattr(task, "user_scenario", None)
    )
    if scenario is None:
        return ""
    if isinstance(scenario, dict):
        instr = scenario.get("instructions")
        if isinstance(instr, dict):
            parts = [
                instr.get("reason_for_call"),
                instr.get("task_instructions"),
                instr.get("known_info"),
            ]
            return clean_task_phrase(" ".join(str(p) for p in parts if p))
        return clean_task_phrase(
            str(
                scenario.get("instructions")
                or scenario.get("task_instructions")
                or ""
            )
        )
    instr = getattr(scenario, "instructions", None)
    if instr is None:
        return clean_task_phrase(str(scenario))
    reason = getattr(instr, "reason_for_call", None)
    task_i = getattr(instr, "task_instructions", None)
    known = getattr(instr, "known_info", None)
    if reason or task_i or known:
        return clean_task_phrase(
            " ".join(str(p) for p in (reason, task_i, known) if p)
        )
    return clean_task_phrase(str(instr))


def _instructions_obj(task: Any) -> Any:
    scenario = (
        task.get("user_scenario")
        if isinstance(task, dict)
        else getattr(task, "user_scenario", None)
    )
    if scenario is None:
        return None
    if isinstance(scenario, dict):
        return scenario.get("instructions")
    return getattr(scenario, "instructions", None)


def agent_facing_task_text(task: Any) -> str:
    """Text for ``Your task is to:`` — customer goal, not user-sim script.

    Prefer ``reason_for_call`` + ``known_info`` only. Do **not** inject
    ``task_instructions`` (those are secret user-simulator directives and
    leak mid-conversation intents / payment hints into the agent prompt).
    """
    if task is None:
        return ""
    instr = _instructions_obj(task)
    reason = _field(instr, "reason_for_call") if instr is not None else None
    known = _field(instr, "known_info") if instr is not None else None
    facing = clean_task_phrase(
        " ".join(str(p) for p in (reason, known) if p)
    )
    if facing:
        return facing[:500]
    # Fallback: cleaned purpose only (never full task_instructions script).
    desc = (
        task.get("description")
        if isinstance(task, dict)
        else getattr(task, "description", None)
    )
    return flatten_task_description(desc)[:500]


def task_text_and_domain(task: Any) -> tuple[str, str, str]:
    """Return (task_text, domain, task_id) from a τ² Task or dict.

    ``task_text`` is for routing/journal (scenario + cleaned purpose). Agent
    window prompts should use :func:`agent_facing_task_text` instead.
    """
    if task is None:
        return "", "", ""

    if isinstance(task, dict):
        desc = flatten_task_description(task.get("description"))
        scenario = scenario_text_from_task(task)
        text = clean_task_phrase(" ".join(x for x in (scenario, desc) if x))
        return text, domain_from_task(task), str(task.get("id") or "")

    desc = flatten_task_description(getattr(task, "description", None))
    scenario = scenario_text_from_task(task)
    text = clean_task_phrase(" ".join(x for x in (scenario, desc) if x))
    task_id = str(getattr(task, "id", "") or "")
    return text, domain_from_task(task), task_id


def _tool_name_from_protocol_step(step: str) -> str:
    return str(step or "").split("(", 1)[0].strip()


def skill_write_name(skill: Any) -> str:
    """First write-tool name owned by a skill protocol / capability key."""
    meta = getattr(skill, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    write_key = str(meta.get("write_capability_key") or "").strip()
    if write_key:
        write_scope = capability_scope_from_key(write_key)
        if write_scope and write_scope in {
            _normalized_scope(n) for n in WRITE_TOOL_NAMES
        }:
            return write_scope
    key = capability_scope_from_key(getattr(skill, "capability_key", "") or "")
    if key and key in {_normalized_scope(n) for n in WRITE_TOOL_NAMES}:
        return key
    for step in getattr(skill, "action_protocol", None) or []:
        name = _tool_name_from_protocol_step(step)
        if _normalized_scope(name) in {_normalized_scope(n) for n in WRITE_TOOL_NAMES}:
            return _normalized_scope(name)
    return key


def organizational_capability_key(skill: Any) -> str:
    """Capability used for cluster / nominate / specialist ownership.

    Prefers the agent **write** spine (``write_capability_key``) when present so
    specialists align with executable tools (e.g. ``tau2.enable_roaming``), not
    wide episode labels (e.g. ``tau2.mms_issue_abroad``).
    """
    meta = getattr(skill, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    write = str(meta.get("write_capability_key") or "").strip()
    if write:
        return write
    write_scope = skill_write_name(skill)
    if write_scope and write_scope in {
        _normalized_scope(n) for n in WRITE_TOOL_NAMES
    }:
        return f"tau2.{write_scope}" if not write_scope.startswith("tau2.") else write_scope
    return str(getattr(skill, "capability_key", "") or "").strip()


def skill_routing_scopes(skill: Any) -> set[str]:
    """Normalized scopes a skill may own (families + capability write)."""
    scopes: set[str] = set()
    for scope in probe_scopes_from_skill_metadata(skill):
        if scope and scope != "other":
            scopes.add(scope)
    write = skill_write_name(skill)
    if write:
        scopes.add(write)
    cap = capability_scope_from_key(getattr(skill, "capability_key", "") or "")
    if cap and cap != "other":
        scopes.add(cap)
    org = capability_scope_from_key(organizational_capability_key(skill))
    if org and org != "other":
        scopes.add(org)
    return scopes


def _semantic_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(tok) >= 4
    }


# Benchmark task-id bug tokens that imply an agent write beyond a single spine.
# Used only to avoid blaming a narrow skill for multi-bug episode failures.
_TASK_ID_WRITE_HINTS: tuple[tuple[str, str], ...] = (
    ("data_usage_exceeded", "refuel_data"),
    ("roaming_disabled", "enable_roaming"),
    ("abroad_roaming", "enable_roaming"),
    ("roaming_enabled_off", "enable_roaming"),
    ("roaming_disabled_off", "enable_roaming"),
    ("roaming_disabled_on", "enable_roaming"),
    # Routing only: overdue suspension implies payment / line resume writes.
    ("overdue_bill", "send_payment_request"),
    ("overdue_bill", "resume_line"),
)


def hinted_agent_writes_from_task_id(task_id: str) -> set[str]:
    """Agent write tools hinted by telecom-style bug flags in ``task_id``."""
    tid = str(task_id or "").lower()
    out: set[str] = set()
    for needle, write in _TASK_ID_WRITE_HINTS:
        if needle in tid:
            out.add(_normalized_scope(write))
    return out


def skill_protocol_tool_names(skill: Any) -> list[str]:
    """Ordered agent/guide step names from a skill ``action_protocol``."""
    out: list[str] = []
    for step in getattr(skill, "action_protocol", None) or []:
        text = str(step or "").strip()
        if not text:
            continue
        low = text.lower()
        if low.startswith("guide user:"):
            name = text.split(":", 1)[1].strip().split("(", 1)[0].strip()
            if name:
                out.append(_normalized_scope(name))
            continue
        if low.startswith(
            (
                "if ",
                "confirm ",
                "resolve ",
                "extract ",
                "communicate ",
                "instruct ",
                "ask ",
                "diagnose ",
                "mark ",
            )
        ):
            continue
        name = _tool_name_from_protocol_step(text)
        if name:
            out.append(_normalized_scope(name))
    return out


def skill_agent_write_names(skill: Any) -> list[str]:
    """Write-class tool names appearing in the skill protocol (ordered, unique)."""
    write_norm = {_normalized_scope(n) for n in WRITE_TOOL_NAMES}
    seen: set[str] = set()
    out: list[str] = []
    for name in skill_protocol_tool_names(skill):
        if name in write_norm and name not in seen:
            seen.add(name)
            out.append(name)
    primary = skill_write_name(skill)
    if primary and primary not in seen and primary in write_norm:
        out.insert(0, primary)
    return out


def episode_skill_coverage_score(
    skill: Any,
    *,
    task_id: str = "",
) -> float:
    """Rank skills by how their *own protocol* covers episode write hints.

    Uses only task_id bug→write routing cues and the skill protocol — not
    hand-written repair recipes. Lookup-only cards score low when the episode
    hints hard writes (refuel / roaming / payment).

    When several skills cover the same hints, prefer the card whose **primary
    write** is hinted and whose write set is sharper (fewer extra writes).
    Prefer distilled ``bug_tags_union`` / ``bug_signature`` that is a subset of
    the episode's bug tags (complete causal card for that superset).
    """
    hinted = hinted_agent_writes_from_task_id(task_id)
    skill_writes = set(skill_agent_write_names(skill))
    skill_tools = set(skill_protocol_tool_names(skill))
    meta = getattr(skill, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    episode_bugs = set(bug_tags_from_task_id(task_id))
    skill_bugs = {
        str(t).strip()
        for t in (meta.get("bug_signature") or meta.get("bug_tags_union") or [])
        if str(t).strip()
    }
    bug_bonus = 0.0
    if skill_bugs and episode_bugs:
        if skill_bugs <= episode_bugs:
            # Applicable exact/subset card: prefer richer signatures.
            bug_bonus = 0.2 + 0.12 * min(5, len(skill_bugs))
        else:
            # Partial overlap only.
            bug_bonus = 0.04 * len(skill_bugs & episode_bugs)
    if not hinted:
        # No hard cue: prefer cards that at least expose a write, else neutral.
        if skill_writes:
            return 0.35 + 0.05 * min(3, len(skill_writes)) + bug_bonus
        return (0.2 if skill_tools else 0.0) + bug_bonus

    hit = hinted & skill_writes
    # Also count protocol tool names that equal a hinted write.
    hit |= hinted & skill_tools
    cover = len(hit) / float(len(hinted))
    primary = skill_write_name(skill)
    primary_bonus = 0.25 if primary and primary in hinted else 0.0
    # Prefer focused cards (fewer extra writes) when covering the same hints.
    sharpness = (len(hit) / float(len(skill_writes))) if skill_writes else 0.0
    if not hit:
        return max(0.0, 0.05 * min(2, len(skill_tools))) + bug_bonus
    return cover + primary_bonus + 0.25 * sharpness + bug_bonus


def skill_matches_episode(
    skill: Any,
    *,
    domain: str,
    task_id: str = "",
    episode_scope: str = "",
    task_text: str = "",
) -> bool:
    """True when a skill is in-domain and relevant to this episode.

    Matching order:
    1. Domain must match when both sides set.
    2. Optional bug-signature gate (``union_with_signature_gate`` / exact).
    3. Episode scope equals skill primary / owned write.
    4. Skill write hinted by task_id bug flags (or write stem ≥5 chars in id).
    5. Soft family labels (e.g. ``mms_issue`` in metadata) only count when (4) holds.
    6. Unstructured ids may use capability↔task token overlap; structured ids do not.
    """
    skill_domain = str(getattr(skill, "domain", "") or "").strip()
    episode_domain = str(domain or "").strip()
    if skill_domain and episode_domain and not domains_match(
        skill_domain, episode_domain
    ):
        return False

    meta = getattr(skill, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    if not _bug_signature_gate_allows(skill, meta, task_id):
        return False

    scopes = skill_routing_scopes(skill)
    current = _normalized_scope(episode_scope) or episode_scope_from_task_id(task_id)
    primary = _normalized_scope(meta.get("primary_task_family"))
    write = skill_write_name(skill)
    hinted = hinted_agent_writes_from_task_id(task_id)
    tid = str(task_id or "").lower()

    if current and (current == primary or current == write):
        return True

    capability_cue = False
    if write and write in hinted:
        capability_cue = True
    elif write:
        stem = write.split("_")[-1] if "_" in write else write
        if len(stem) >= 5 and stem in tid:
            capability_cue = True
    # Parent capability e.g. mobile_data_issue_abroad ↔ mobile_data_issue scope.
    parent = _normalized_scope(
        meta.get("parent_capability_key") or getattr(skill, "capability_key", "")
    )
    if current and parent and (
        parent == current or parent.startswith(current + "_") or current in parent
    ):
        capability_cue = True

    if capability_cue:
        return True

    # Soft family (bracket label in metadata) requires a capability cue.
    if current and current in scopes and current not in {primary, write, ""}:
        return False

    if current or hinted or _BRACKET_SCOPE_RE.match(str(task_id or "").strip()):
        return False

    contract_tokens = _semantic_tokens(
        " ".join(
            [
                str(getattr(skill, "capability_key", "") or ""),
                str(getattr(skill, "skill_name", "") or ""),
                " ".join(scopes),
                write,
            ]
        )
    )
    task_tokens = _semantic_tokens(f"{current} {task_id} {task_text}")
    return bool(contract_tokens & task_tokens)


def _bug_signature_gate_allows(skill: Any, meta: dict[str, Any], task_id: str) -> bool:
    """Enforce activation signatures when distill recorded a gated bucket mode.

    For ``union_with_signature_gate``, compare **normalized** bug families so
    roaming / APN / permission / SIM variants can still activate a skill whose
    evidence used a sibling tag. Strict raw equality remains as a fast path.
    When evidence is too diverse to share a core (or too thin to cover the
    episode's combo), the fallback is the episode's write cue for the skill's
    own write tool.
    """
    mode = str(meta.get("bug_bucket_mode") or "").lower().strip()
    if mode not in {"union_with_signature_gate", "signature_match", "exact"}:
        return True
    sigs = meta.get("activation_signatures") or meta.get("evidence_bug_signatures") or []
    if not isinstance(sigs, list) or not sigs:
        return True
    episode = set(bug_tags_from_task_id(task_id))
    # Unstructured / no-bug episodes: do not hard-fail (fall through to soft cues).
    if not episode:
        return True
    sig_sets = [set(sig) for sig in sigs if isinstance(sig, (list, tuple, set))]
    if not sig_sets:
        return True
    if mode in {"signature_match", "exact"}:
        ep_n = normalize_bug_tags(episode)
        return any(ep_n == normalize_bug_tags(sig) for sig in sig_sets)
    # union_with_signature_gate: shared core required, no unknown tags outside union.
    inter = set(meta.get("bug_intersection") or [])
    if not inter:
        inter = set.intersection(*sig_sets)
    union = set(meta.get("bug_union") or [])
    if not union:
        for sig in sig_sets:
            union |= sig
    if inter and inter <= episode and episode <= union:
        return True
    if any(episode == sig for sig in sig_sets):
        return True
    # Soft path: normalized families (roaming_*/apn_*/permission_*/sim_*).
    ep_n = normalize_bug_tags(episode)
    inter_n = (
        normalize_bug_tags(inter)
        if inter
        else set.intersection(*(normalize_bug_tags(sig) for sig in sig_sets))
    )
    union_n = normalize_bug_tags(union)
    if not union_n:
        for sig in sig_sets:
            union_n |= normalize_bug_tags(sig)
    if inter_n and inter_n <= ep_n and ep_n <= union_n:
        return True
    # Compositional bug tasks: diverse evidence yields an empty shared core,
    # and a single-evidence seed has a narrow union — both would hard-block
    # every activation above. Fall back to the write cue: the episode's bug
    # flags must hint the skill's own write tool (e.g. a roaming flag for
    # enable_roaming, data_usage_exceeded for refuel_data). Sharing an
    # unrelated bug family (airplane_mode_on) is not enough.
    write = skill_write_name(skill)
    if write and write in hinted_agent_writes_from_task_id(task_id):
        return True
    return False


def skill_credit_should_apply(
    skill: Any,
    *,
    domain: str,
    task_id: str,
    success: bool,
    episode_scope: str = "",
) -> bool:
    """Gate credit so irrelevant / partial multi-bug fails do not prune skills.

    - Always require :func:`skill_matches_episode`.
    - On failure, skip when task_id hints additional agent writes the skill
      does not own (e.g. roaming skill on a ``data_usage_exceeded`` episode).
    """
    if not skill_matches_episode(
        skill,
        domain=domain,
        task_id=task_id,
        episode_scope=episode_scope,
    ):
        return False
    if success:
        return True
    write = skill_write_name(skill)
    hinted = hinted_agent_writes_from_task_id(task_id)
    if not hinted:
        return True
    # Failure: only blame the skill when it solely owns every hinted write.
    return bool(write) and hinted <= {write}
