"""Heuristic distillation of τ² tool-protocol skills."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from sage_tau2.schemas import SkillStatus, Tau2Skill, Tau2Trajectory
from sage_tau2.task_context import (
    bug_tags_from_task_id,
    capability_scope_from_key,
    episode_scope_from_task_id,
)
from sage_tau2.tool_sides import (
    TELECOM_USER_TOOLS,
    is_telecom_user_side_tool,
    strip_user_side_protocol_steps,
)

# Birth support below this stays candidate/provisional or verified_low_support.
DEFAULT_MIN_SUPPORT_FOR_VERIFIED = 5

# Domain-mutating tools used as capability spines (ALFWorld-style: own a real write).
# Telecom user-device actions (toggle_*, check_*, reseat_*, …) are NOT writes here —
# they are user-simulator tools; treating them as spines amplified Tool-not-found misuse.
_WRITE_OPS = {
    # airline
    "cancel_reservation",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
    "book_reservation",
    "send_certificate",
    # retail
    "cancel_pending_order",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    # telecom (agent backend only)
    "enable_roaming",
    "disable_roaming",
    "refuel_data",
    "resume_line",
    "suspend_line",
    "send_payment_request",
}

# Escalation is not a specialty to own (Executor can escalate).
_ESCALATE_OPS = {"transfer_to_human_agents"}

_READISH_PREFIXES = (
    "get_",
    "find_",
    "search_",
    "check_",
    "calculate",
    "list_",
    "think",
)


def _tool_name(step: str) -> str:
    return step.split("(", 1)[0].strip()


def canonicalize_protocol(
    protocol: list[str],
    *,
    domain: str | None = None,
) -> list[str]:
    """Drop think/escalate/user-device noise and collapse consecutive duplicates.

    Exact arg strings still differ across episodes; we keep the first occurrence
    of each consecutive tool name so near-duplicate traces merge.
    """
    cleaned = strip_user_side_protocol_steps(protocol, domain=domain)
    out: list[str] = []
    prev_name = ""
    for step in cleaned:
        name = _tool_name(step)
        if not name or name == "think" or name.startswith("think"):
            continue
        if name in _ESCALATE_OPS:
            continue
        if name == prev_name:
            continue
        out.append(step)
        prev_name = name
    return out


def trim_protocol_to_write_spine(protocol: list[str]) -> list[str]:
    """Keep reads up to the first write; drop post-write long-script noise.

    Capability ownership is the first write tool. Extra writes after that
    (e.g. enable_roaming skill that also refuels) pollute the spine and create
    overlapping specialists.
    """
    if not protocol:
        return []
    write_idx = None
    for i, step in enumerate(protocol):
        if _tool_name(step) in _WRITE_OPS:
            write_idx = i
            break
    if write_idx is None:
        return list(protocol)
    return list(protocol[: write_idx + 1])


def normalize_protocol(
    protocol: list[str],
    *,
    domain: str | None = None,
    trim_to_write_spine: bool = False,
) -> list[str]:
    """Canonicalize protocols for distill buckets / credit coverage.

    By default keeps the full agent tool trace (not ALFWorld-style write-spine
    trim). Pass ``trim_to_write_spine=True`` only for diagnostics / ablations.
    """
    cleaned = canonicalize_protocol(protocol, domain=domain)
    if trim_to_write_spine:
        return trim_protocol_to_write_spine(cleaned)
    return cleaned


def _capability_key(protocol: list[str]) -> str:
    names = [_tool_name(step) for step in protocol if _tool_name(step)]
    if not names:
        return "tau2.empty"
    # Prefer a real write spine; never prefer escalation over a write.
    for name in names:
        if name in _WRITE_OPS:
            return f"tau2.{name}"
    non_escalate = [n for n in names if n not in _ESCALATE_OPS]
    if non_escalate:
        return "tau2." + "_".join(non_escalate[:3])
    # Pure escalate protocols are filtered later; keep a stable key for diagnostics.
    return f"tau2.{names[0]}"


def _protocol_bucket_key(protocol: list[str]) -> tuple[str, ...]:
    """Bucket by capability + ordered tool-name spine (ignore arg literals)."""
    capability = _capability_key(protocol)
    names = tuple(_tool_name(step) for step in protocol if _tool_name(step))
    return (capability, *names)


def _bug_signature_tuple(traj: Tau2Trajectory) -> tuple[str, ...]:
    return tuple(sorted(bug_tags_from_task_id(traj.task_id)))


def _coarse_telecom_bucket_base(protocol: list[str]) -> tuple[str, ...]:
    """Capability-level bucket for telecom: capability + sorted write-name set.

    Long dialogue traces with variable line-probe counts almost never repeat a
    full tool spine, so spine-keyed buckets all carry support 1 and can never
    reach the verified support bar. Bucket by ``(capability, write set)``
    instead; read/probe steps only decide which member becomes the template
    inside a bucket (see ``pick_richest_line_enum_member``).
    """
    capability = _capability_key(protocol)
    writes = sorted(set(primary_write_names(protocol)))
    if writes:
        return (capability, *writes)
    return (capability,)


def _protocol_bucket_key_with_bugs(
    protocol: list[str],
    traj: Tau2Trajectory,
    *,
    bug_bucket_mode: str = "off",
    domain: str | None = None,
) -> tuple[str, ...]:
    """Extend the tool-spine bucket with a bug signature when requested.

    - ``off``: tool spine only (legacy merge).
    - ``exact`` / ``signature_match``: spine + full bug-tag set.
    - ``union`` / ``union_with_signature_gate``: spine only at bucket time;
      diagnostics are unioned later (gate applied at match time).
    - ``primary_write``: removed; treated like ``union_with_signature_gate``.

    Telecom/telecom-workflow domains bucket by the coarse capability+write-set
    base instead of the full tool spine (support fragmentation fix).
    """
    domain_l = str(domain or "").strip().lower()
    if domain_l in {"telecom", "telecom-workflow"}:
        base = _coarse_telecom_bucket_base(protocol)
    else:
        base = _protocol_bucket_key(protocol)
    mode = str(bug_bucket_mode or "off").lower().strip()
    if mode not in {"exact", "signature_match"}:
        return base
    bugs = _bug_signature_tuple(traj)
    if not bugs:
        return base
    return (*base, "bugs", *bugs)


def primary_write_names(protocol: list[str]) -> list[str]:
    """Ordered unique agent write names in a protocol."""
    out: list[str] = []
    seen: set[str] = set()
    for step in protocol or []:
        name = _tool_name(str(step))
        if name in _WRITE_OPS and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def bucket_writes_from_key(key: tuple[str, ...]) -> frozenset[str]:
    """Extract sorted write-tool names encoded in a distill bucket key."""
    if len(key) < 2:
        return frozenset()
    parts = list(key[1:])
    for marker in ("bugs", "guides"):
        if marker in parts:
            parts = parts[: parts.index(marker)]
    if not parts:
        return frozenset()
    writes = [p for p in parts[1:] if p in _WRITE_OPS]
    if writes:
        return frozenset(writes)
    cap_name = str(parts[0] or "").replace("tau2.", "")
    if cap_name in _WRITE_OPS:
        return frozenset({cap_name})
    return frozenset()


def filter_protocol_to_bucket_writes(
    protocol: list[str],
    bucket_writes: frozenset[str] | set[str],
) -> list[str]:
    """Drop agent writes outside the bucket's intended write set."""
    if not bucket_writes:
        return list(protocol or [])
    allowed = set(bucket_writes)
    out: list[str] = []
    seen_writes: set[str] = set()
    for step in protocol or []:
        text = str(step or "").strip()
        if not text:
            continue
        name = _tool_name(text)
        if name in _WRITE_OPS:
            if name not in allowed or name in seen_writes:
                continue
            seen_writes.add(name)
        out.append(text)
    return out


def collapse_nonconsecutive_duplicate_tools(protocol: list[str]) -> list[str]:
    """Keep first occurrence of each agent tool name; keep all guide/enum rows."""
    out: list[str] = []
    seen_tools: set[str] = set()
    for step in protocol or []:
        text = str(step or "").strip()
        if not text:
            continue
        low = text.lower()
        if low.startswith("guide user:") or low.startswith("guide "):
            out.append(text)
            continue
        if low.startswith(("enumerate:", "probe:", "select:")):
            # Preserve evidence-backed line-enumeration procedure rows.
            if text not in out:
                out.append(text)
            continue
        name = _tool_name(text)
        if not name:
            continue
        if name in seen_tools:
            continue
        seen_tools.add(name)
        out.append(text)
    return out


def stable_guide_names_from_trajs(
    trajs: list[Tau2Trajectory],
    *,
    min_fraction: float = 0.67,
) -> list[str]:
    """User-side guides that co-occur in enough successful demos."""
    if not trajs:
        return []
    counts: Counter[str] = Counter()
    for traj in trajs:
        for name in matched_user_side_tool_names(traj):
            counts[name] += 1
    need = max(1, int(len(trajs) * float(min_fraction) + 0.999))
    if len(trajs) == 1:
        need = 1
    ordered = matched_user_side_tool_names_from_trajs(trajs, require_all=False)
    return [n for n in ordered if counts[n] >= need]


def bind_write_args_from_evidence(
    protocol: list[str],
    trajs: list[Tau2Trajectory],
) -> list[str]:
    """Bind reusable write args from successful tool_steps when stable.

    Opaque identifiers (``bill_id``, ``line_id``, …) stay semantic templates —
    binding a single demo's literal ID breaks OOD. Numeric/plan args like
    ``gb_amount`` may be bound when stable across demos. ``line_id`` is always
    rebound to the affected-line slot (never a literal like L1001).
    """
    hist: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for traj in trajs:
        for step in getattr(traj, "tool_steps", None) or []:
            name = str(getattr(step, "name", "") or "").strip()
            if name not in _WRITE_OPS:
                continue
            args = getattr(step, "arguments", None) or {}
            if not isinstance(args, dict):
                continue
            for key, val in args.items():
                if key in {
                    "customer_id",
                    "line_id",
                    "id",
                    "phone_number",
                    "bill_id",
                    "user_id",
                    "reservation_id",
                    "order_id",
                }:
                    continue
                if val is None or val == "" or val == "?":
                    continue
                hist[(name, str(key))][str(val)] += 1

    def _stable(name: str, key: str) -> str | None:
        counter = hist.get((name, key))
        if not counter:
            return None
        val, n = counter.most_common(1)[0]
        if n < max(1, (len(trajs) + 1) // 2):
            return None
        return val

    has_usage_lookup = any(
        _tool_name(s) == "get_data_usage" for s in (protocol or [])
    )
    line_slot = _LINE_ID_SLOT
    if has_usage_lookup:
        line_slot = _LINE_ID_SLOT

    out: list[str] = []
    for step in protocol or []:
        text = str(step or "").strip()
        name = _tool_name(text)
        if name == "refuel_data" and "gb_amount=?" in text:
            gb = _stable("refuel_data", "gb_amount")
            if gb is not None:
                text = text.replace("gb_amount=?", f"gb_amount={gb}")
            else:
                text = text.replace(
                    "gb_amount=?",
                    "gb_amount=<from get_data_usage remaining/plan>",
                )
        if name == "send_payment_request" and "bill_id=?" in text:
            text = text.replace(
                "bill_id=?",
                "bill_id=<from get_bills_for_customer>",
            )
        if name in _WRITE_OPS or name == "get_data_usage":
            text = _rewrite_line_id_slot(text, line_slot)
        out.append(text)
    return out


def _rewrite_line_id_slot(step: str, slot: str) -> str:
    """Replace any line_id=… binding with the semantic affected-line slot."""
    text = str(step or "")
    if "line_id=" not in text:
        return text
    # Match either a <template> or a bare token up to comma / close-paren.
    return re.sub(
        r"line_id=(?:<[^>]*>|[^\s,)\]]+)",
        slot,
        text,
        count=1,
    )


def is_escalate_only_capability(capability_key: str) -> bool:
    key = str(capability_key or "").replace("tau2.", "").strip()
    return key in _ESCALATE_OPS or key == "transfer_to_human_agents"


def protocol_has_write(protocol: list[str]) -> bool:
    return any(_tool_name(step) in _WRITE_OPS for step in protocol)


def protocol_has_escalate(protocol: list[str]) -> bool:
    return any(_tool_name(step) in _ESCALATE_OPS for step in protocol)


def is_read_only_protocol(protocol: list[str]) -> bool:
    names = [_tool_name(step) for step in protocol if _tool_name(step)]
    if not names:
        return True
    for name in names:
        if name in _WRITE_OPS:
            return False
        if name in _ESCALATE_OPS:
            continue
        if not any(name.startswith(p) for p in _READISH_PREFIXES):
            return False
    # Only reads and/or escalate.
    return True


def is_trivial_lookup_protocol(protocol: list[str]) -> bool:
    """True for identity-only get/find traces (not enough to teach communicate)."""
    names = [_tool_name(step) for step in protocol if _tool_name(step)]
    if not names:
        return True
    if any(name in _WRITE_OPS or name in _ESCALATE_OPS for name in names):
        return False
    if any(name.startswith(("calculate", "list_", "search_")) for name in names):
        return False
    return all(name.startswith(("get_", "find_")) for name in names)


def is_early_escalate_protocol(protocol: list[str]) -> bool:
    """True when the trace is escalate without any domain write.

    Kept for diagnostics / ablations. Default distill gating uses
    ``infer_success_mode`` + ``allowed_success_modes`` instead.
    """
    if not protocol_has_escalate(protocol):
        return False
    return not protocol_has_write(protocol)


# Default distill modes: DB writes and communicate/read resolutions.
# transfer_ok is excluded — escalation stays Executor fallback, not a specialty.
DEFAULT_DISTILL_SUCCESS_MODES: frozenset[str] = frozenset(
    {"solve_write", "communicate_ok"}
)


def infer_success_mode(traj: Tau2Trajectory) -> str:
    """Prefer labeled ``metadata.success_mode``; else infer from the tool trace."""
    labeled = (traj.metadata or {}).get("success_mode")
    if labeled is not None and str(labeled).strip():
        return str(labeled).strip()
    raw = list(traj.tool_protocol or [])
    if protocol_has_write(raw):
        return "solve_write"
    if protocol_has_escalate(raw) and is_early_escalate_protocol(raw):
        return "transfer_ok"
    if protocol_has_escalate(raw) and not protocol_has_write(raw):
        return "transfer_ok"
    return "communicate_ok"


def resolve_allowed_success_modes(
    *,
    require_solve_write: bool | None = None,
    allowed_success_modes: frozenset[str] | set[str] | list[str] | None = None,
) -> frozenset[str]:
    """Map legacy ``require_solve_write`` onto an allowlist of success modes."""
    if allowed_success_modes is not None:
        return frozenset(str(m).strip() for m in allowed_success_modes if str(m).strip())
    if require_solve_write is True:
        return frozenset({"solve_write"})
    # False or None → τ²-native layered modes.
    return frozenset(DEFAULT_DISTILL_SUCCESS_MODES)


def trajectory_eligible_for_distill(
    traj: Tau2Trajectory,
    *,
    require_success: bool = True,
    require_solve_write: bool | None = None,
    allowed_success_modes: frozenset[str] | set[str] | list[str] | None = None,
    require_action_checks_ok: bool | None = None,
) -> bool:
    """Gate which episodes may birth skills.

    - Airline/retail: success (reward/DB).
    - Telecom: full episode success (ENV/reward), never DB-only proxies.
    - Default modes: ``solve_write`` + ``communicate_ok`` (not transfer-only).
    - ``require_solve_write=True`` restores the old write-only gate.
    - Inferred (unlabeled) communicate_ok skips trivial get/find-only lookups.
    - Telecom default also requires complete gold ``action_checks`` (all match,
      with at least one non-transfer action) so half-solved / transfer-only
      demos do not birth incomplete payment/refuel cards.
    """
    domain = str(getattr(traj, "domain", "") or "").lower()
    modes = resolve_allowed_success_modes(
        require_solve_write=require_solve_write,
        allowed_success_modes=allowed_success_modes,
    )
    mode = infer_success_mode(traj)
    if mode not in modes:
        return False
    labeled = (traj.metadata or {}).get("success_mode")
    if mode == "communicate_ok" and (
        labeled is None or str(labeled).strip() == ""
    ):
        # After stripping user-device noise, leftover identity lookups must not
        # birth communicate skills unless the episode was labeled as such.
        proto = normalize_protocol(
            list(traj.tool_protocol or []),
            domain=getattr(traj, "domain", None),
        )
        if is_trivial_lookup_protocol(proto):
            return False
    if not require_success:
        return True
    if domain in {"telecom", "telecom-workflow"}:
        # ENV_ASSERTION domains: only fully solved episodes teach.
        if not bool(traj.success):
            return False
        checks_gate = require_action_checks_ok
        checks = (traj.metadata or {}).get("action_checks")
        has_checks = isinstance(checks, list) and bool(checks)
        if checks_gate is True:
            return trajectory_action_checks_complete(traj)
        if checks_gate is False:
            return True
        # None (default): if gold checks exist, require them complete; else allow.
        if has_checks:
            return trajectory_action_checks_complete(traj)
        return True
    if mode == "solve_write":
        return bool(traj.success or traj.has_positive_db_effect)
    return bool(traj.success)


def trajectory_action_checks_complete(traj: Tau2Trajectory) -> bool:
    """True when gold action_checks exist, all match, and are non-trivial.

    Rejects transfer-only \"success\" rows that would distill empty guide cards.
    Episodes without action_checks metadata fail closed (cannot prove causality).
    """
    checks = (traj.metadata or {}).get("action_checks") or []
    if not isinstance(checks, list) or not checks:
        return False
    names: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            return False
        if check.get("action_match") is False:
            return False
        action = check.get("action") or {}
        if not isinstance(action, dict):
            continue
        name = str(action.get("name") or "").strip()
        if name:
            names.append(name)
    meaningful = [n for n in names if n not in _ESCALATE_OPS]
    return bool(meaningful)


_HARD_AGENT_WRITES = frozenset(
    {
        "enable_roaming",
        "disable_roaming",
        "refuel_data",
        "send_payment_request",
        "resume_line",
        "suspend_line",
    }
)


def reorder_protocol_hard_writes_first(protocol: list[str]) -> list[str]:
    """Put lookup → hard agent writes → other agent steps → guide rows.

    Keeps compliance easier: refuel/payment are not buried after long roam/usage
    loops. Payment spine keeps ``guide user: make_payment`` before ``resume_line``
    (bill must clear before line resume). Other guide rows stay last.
    """
    lookups: list[str] = []
    hards: list[str] = []
    others: list[str] = []
    guides: list[str] = []
    enums: list[str] = []
    for step in protocol or []:
        text = str(step or "").strip()
        if not text:
            continue
        low = text.lower()
        if low.startswith("guide user:") or low.startswith("guide "):
            guides.append(step)
            continue
        if low.startswith(("enumerate:", "probe:", "select:")):
            enums.append(step)
            continue
        name = _tool_name(text)
        if name in _HARD_AGENT_WRITES:
            hards.append(step)
        elif name.startswith(("get_", "find_", "search_", "list_", "check_", "calculate")):
            lookups.append(step)
        else:
            others.append(step)
    payment_guides = [
        g for g in guides if "make_payment" in str(g).lower()
    ]
    other_guides = [g for g in guides if g not in payment_guides]
    hards_non_resume = [h for h in hards if _tool_name(h) != "resume_line"]
    resumes = [h for h in hards if _tool_name(h) == "resume_line"]
    # Keep evidence line-enumeration after customer identity lookups and before
    # per-line get_data_usage / writes (successful demos probe all lines first).
    identity_lookups: list[str] = []
    rest_lookups: list[str] = []
    for step in lookups:
        name = _tool_name(step)
        if name in {
            "get_customer_by_phone",
            "get_customer_by_id",
            "get_customer_by_name",
        }:
            identity_lookups.append(step)
        elif name == "get_details_by_id" and (
            "customer_id_from_prior" in step
            or "do_not_pass_guessed_line_id" in step
            or "customer_id from prior" in step
        ):
            identity_lookups.append(step)
        else:
            rest_lookups.append(step)
    return (
        identity_lookups
        + enums
        + rest_lookups
        + hards_non_resume
        + others
        + payment_guides
        + resumes
        + other_guides
    )


# Semantic line_id slot after evidence-backed line enumeration (not a literal id).
# Keep free of commas / ')' so it stays one protocol-arg token.
_LINE_ID_SLOT = (
    "line_id=<target_line_id_after_enumerating_all_lines_via_get_data_usage;"
    "never_default_first_listed_line>"
)


def _looks_like_line_id(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or text in {"?", ""}:
        return False
    # Telecom line ids are typically L####; also accept explicit line_id args.
    return text[:1].upper() == "L" and any(ch.isdigit() for ch in text)


def traj_line_probe_count(
    traj: Tau2Trajectory,
    primary_write: str,
) -> int:
    """Distinct line ids probed via get_details / get_data_usage before the write."""
    primary = str(primary_write or "").strip()
    steps = list(getattr(traj, "tool_steps", None) or [])
    write_idx = len(steps)
    for i, step in enumerate(steps):
        if str(getattr(step, "name", "") or "").strip() == primary:
            write_idx = i
            break
    probed: set[str] = set()
    for step in steps[:write_idx]:
        name = str(getattr(step, "name", "") or "").strip()
        args = getattr(step, "arguments", None) or {}
        if not isinstance(args, dict):
            continue
        if name == "get_data_usage":
            lid = args.get("line_id")
            if _looks_like_line_id(lid):
                probed.add(str(lid))
        elif name == "get_details_by_id":
            oid = args.get("id")
            if _looks_like_line_id(oid):
                probed.add(str(oid))
    return len(probed)


def pick_richest_line_enum_member(
    members: list[tuple[list[str], Tau2Trajectory, str]],
    primary_write: str,
) -> tuple[list[str], Tau2Trajectory, str]:
    """Prefer the demo that probed the most distinct lines before the write."""
    if not members:
        raise ValueError("members must be non-empty")
    scored = sorted(
        members,
        key=lambda m: (
            -traj_line_probe_count(m[1], primary_write),
            len(m[0]),
            m[0],
        ),
    )
    return scored[0]


def pick_cleanest_protocol_member(
    members: list[tuple[list[str], Tau2Trajectory, str]],
    primary_write: str,
    bucket_writes: frozenset[str] | set[str] | None = None,
) -> tuple[list[str], Tau2Trajectory, str]:
    """Prefer the shortest reusable template after write-filter and dedupe."""
    if not members:
        raise ValueError("members must be non-empty")

    def _prepare(m: tuple[list[str], Tau2Trajectory, str]) -> list[str]:
        proto = list(m[0])
        if bucket_writes:
            proto = filter_protocol_to_bucket_writes(proto, bucket_writes)
        return collapse_nonconsecutive_duplicate_tools(proto)

    def _score(m: tuple[list[str], Tau2Trajectory, str]) -> tuple[int, int, int, tuple]:
        proto = _prepare(m)
        guide_n = sum(
            1 for s in proto if str(s).lower().startswith("guide user:")
        )
        return (len(proto), guide_n, -traj_line_probe_count(m[1], primary_write), tuple(proto))

    best = min(members, key=_score)
    return _prepare(best), best[1], best[2]


def _skill_name(
    capability_key: str,
    support: int,
    *,
    bug_tags: list[str] | tuple[str, ...] | None = None,
) -> str:
    del bug_tags  # routing uses metadata gates, not name pollution
    short = capability_key.replace("tau2.", "").replace("_", " ")
    return f"{short} skill (n={support})"


_TOOL_CALL_FRAGMENT_RE = re.compile(
    r"\[tool_call\]\s*[a-zA-Z0-9_]+\s*\([^)]*\)",
    re.IGNORECASE,
)
_BARE_USER_TOOL_RE = re.compile(
    r"\b("
    + "|".join(sorted(TELECOM_USER_TOOLS, key=len, reverse=True))
    + r")\s*\([^)]*\)",
    re.IGNORECASE,
)


def _clean_user_intent_snippet(text: str, *, max_len: int = 120) -> str:
    """Drop tool-call / user-device fragments from user utterance snippets."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = _TOOL_CALL_FRAGMENT_RE.sub(" ", cleaned)
    cleaned = _BARE_USER_TOOL_RE.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


def _precondition_from_trajs_legacy(trajs: list[Tau2Trajectory]) -> str:
    """Original precondition builder (kept for compatibility / offline replay)."""
    hints = []
    for traj in trajs[:3]:
        for text in traj.user_texts[:2]:
            if text:
                hints.append(text[:160])
    if not hints:
        return "User intent matches this capability; follow domain policy."
    joined = " | ".join(hints)
    return (
        "Triggered when user requests similar goals (examples: "
        f"{joined}). Follow domain policy and grounded tools."
    )


def _precondition_from_trajs(
    trajs: list[Tau2Trajectory],
    *,
    primary_write: str = "",
    bug_tags_core: list[str] | None = None,
) -> str:
    """Activation summary from shared bug core + primary write (not raw user text)."""
    tags = list(bug_tags_core or [])
    if not tags and trajs:
        per = [set(bug_tags_from_task_id(t.task_id)) for t in trajs]
        if len(per) == 1:
            tags = sorted(per[0])
        elif per:
            tags = sorted(set.intersection(*per))
    write = str(primary_write or "").replace("tau2.", "").strip()
    if write and tags:
        tag_str = ", ".join(tags[:5])
        if len(tags) > 5:
            tag_str += f" (+{len(tags) - 5} more)"
        return (
            f"Apply when troubleshooting involves: {tag_str}. "
            f"Primary agent write: {write}. "
            "Follow the learned protocol; guide the user only for device-side steps."
        )
    if write:
        return (
            f"Apply when {write} is needed after lookup. "
            "Follow the learned protocol and domain policy."
        )
    hints: list[str] = []
    for traj in trajs[:3]:
        for text in traj.user_texts[:2]:
            snippet = _clean_user_intent_snippet(text)
            if snippet:
                hints.append(snippet)
    if not hints:
        return (
            "User intent matches this capability. Use agent toolkit tools from "
            "the protocol; guide the user for device-side steps; obey domain policy."
        )
    seen: set[str] = set()
    uniq: list[str] = []
    for h in hints:
        key = h.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    joined = " | ".join(uniq[:4])
    return (
        "Triggered when user requests similar goals (examples: "
        f"{joined}). Follow the learned protocol and domain policy. "
        "Do not call user-device APIs as agent tools."
    )


def _expected_effect_legacy(trajs: list[Tau2Trajectory]) -> str:
    """Original expected_effect wording (kept for compatibility)."""
    db_ok = sum(1 for t in trajs if t.has_positive_db_effect)
    rew_ok = sum(1 for t in trajs if t.success)
    return (
        f"Local DB / communicate outcome consistent with successful episodes "
        f"(db_ok={db_ok}/{len(trajs)}, reward_ok={rew_ok}/{len(trajs)}). "
        "Prefer matching the grounded tool protocol; do not invent facts."
    )


def _expected_effect(trajs: list[Tau2Trajectory]) -> str:
    """Clarify ENV reward vs DB match (telecom often succeeds without db_match)."""
    n = len(trajs)
    db_match_ok = sum(1 for t in trajs if t.has_positive_db_effect)
    env_reward_ok = sum(1 for t in trajs if t.success)
    domain = str(getattr(trajs[0], "domain", "") or "").lower() if trajs else ""
    note = ""
    if domain in {"telecom", "telecom-workflow"}:
        note = (
            " For telecom, env_reward_ok is the teaching signal; "
            "db_match_ok may be 0 even on solved episodes (ENV/user-side fixes)."
        )
        if db_match_ok == 0 and env_reward_ok > 0:
            note += (
                " Agent write may be partial; follow inlined user-side branches "
                "as well as the write tool."
            )
    return (
        f"Outcome consistent with successful episodes "
        f"(env_reward_ok={env_reward_ok}/{n}, db_match_ok={db_match_ok}/{n}; "
        f"legacy db_ok={db_match_ok}/{n}, reward_ok={env_reward_ok}/{n})."
        f"{note} Prefer the grounded agent-tool protocol; do not invent facts."
    )


def bind_protocol_slots(
    protocol: list[str],
    *,
    resolved_line_id: bool = False,
) -> list[str]:
    """Rewrite opaque ``?`` args into semantic slot names where the spine is known.

    Does not invent runtime values — only documents binding sources for inject.
    When ``resolved_line_id`` is True (resolve step already in the full protocol),
    write tools bind the affected-line slot instead of re-asking.
    """
    out: list[str] = []
    saw_customer_lookup = False
    saw_details = False
    for step in protocol or []:
        name = _tool_name(step)
        if name in {"get_customer_by_phone", "get_customer_by_id", "get_customer_by_name"}:
            saw_customer_lookup = True
            if name == "get_customer_by_phone" and "phone_number=?" in step:
                out.append(
                    "get_customer_by_phone(phone_number=<from_user_or_ask>)"
                )
            else:
                out.append(step)
            continue
        if name == "get_details_by_id":
            saw_details = True
            if saw_customer_lookup and "id=?" in step:
                out.append(
                    "get_details_by_id(id=<entity_id_from_prior_tool_result;"
                    "preserve_entity_type_and_match_requested_entity>)"
                )
            else:
                out.append(step)
            continue
        if name == "get_data_usage":
            rewritten = step
            if "customer_id=?" in rewritten and saw_customer_lookup:
                rewritten = rewritten.replace(
                    "customer_id=?",
                    "customer_id=<customer_id from lookup>",
                )
            if "line_id=" in rewritten and (
                resolved_line_id or saw_details or saw_customer_lookup
            ):
                rewritten = _rewrite_line_id_slot(rewritten, _LINE_ID_SLOT)
            elif "line_id=?" in rewritten:
                rewritten = rewritten.replace("line_id=?", _LINE_ID_SLOT)
            out.append(rewritten)
            continue
        if name in _WRITE_OPS and (
            "customer_id=?" in step
            or "line_id=" in step
            or "id=?" in step
        ):
            rewritten = step
            if "customer_id=?" in rewritten:
                rewritten = rewritten.replace(
                    "customer_id=?",
                    "customer_id=<customer_id from lookup>",
                )
            if "line_id=" in rewritten:
                if resolved_line_id or saw_details or saw_customer_lookup:
                    rewritten = _rewrite_line_id_slot(rewritten, _LINE_ID_SLOT)
                elif "line_id=?" in rewritten:
                    rewritten = rewritten.replace(
                        "line_id=?",
                        "line_id=<target line_id; ask if multiple>",
                    )
            out.append(rewritten)
            continue
        out.append(step)
    return out


def dialogue_gates_for_protocol(
    protocol: list[str],
    *,
    domain: str | None = None,
) -> list[str]:
    """Deprecated expert dialogue gates — always empty.

    Confirm / disambiguate recipes belong in domain ``policy.md`` or in
    trajectory-grounded protocol rows, not hand-written distill injectors.
    """
    del protocol, domain
    return []


# Map telecom bug tags → user-side companion hints (not agent tools).
_BUG_TAG_USER_HINTS: dict[str, str] = {
    "airplane_mode_on": "user: turn airplane mode OFF (toggle_airplane_mode)",
    "bad_vpn": "user: disconnect VPN (disconnect_vpn)",
    "bad_network_preference": "user: fix network mode preference (set_network_mode_preference)",
    "user_abroad_roaming_enabled_off": (
        "user: turn Data Roaming ON (toggle_roaming); agent may also enable_roaming"
    ),
    "user_abroad_roaming_disabled_on": (
        "agent may enable_roaming after confirming line; user may need Data Roaming ON"
    ),
    "user_abroad_roaming_disabled_off": (
        "agent may enable_roaming; user may also need Data Roaming ON (toggle_roaming)"
    ),
    "data_usage_exceeded": "agent may refuel_data after confirming usage",
    "data_mode_off": "user: turn Mobile Data ON (toggle_data)",
    "data_saver_mode_on": "user: turn Data Saver OFF (toggle_data_saver_mode)",
    "bad_wifi_calling": "user: toggle Wi-Fi Calling as needed (toggle_wifi_calling)",
    "break_apn_mms_setting": "user: reset APN / MMS APN (reset_apn_settings) then reboot",
    "break_apn_settings": "user: reset APN settings (reset_apn_settings)",
    "bad_apn": "user: reset/fix APN settings",
    "break_app_sms_permission": "user: grant SMS permission to messaging app",
    "break_app_storage_permission": "user: grant storage permission to messaging app",
    "break_app_both_permissions": "user: grant SMS + storage permissions to messaging app",
    "unseat_sim_card": "user: reseat SIM card (reseat_sim_card)",
    "lock_sim_card_pin": "user: unlock SIM PIN / reseat SIM",
    "sim_locked": "user: reseat SIM / check SIM status",
    "overdue_bill_suspension": (
        "agent: send_payment_request / resume_line after payment; else transfer"
    ),
    "contract_end_suspension": "agent: check line status; may need transfer_to_human_agents",
}

# Bug tag → user-simulator tool for conditional ``guide user:`` rows.
_BUG_TAG_TO_USER_TOOL: dict[str, str] = {
    "airplane_mode_on": "toggle_airplane_mode",
    "bad_vpn": "disconnect_vpn",
    "bad_network_preference": "set_network_mode_preference",
    "user_abroad_roaming_enabled_off": "toggle_roaming",
    "user_abroad_roaming_disabled_on": "toggle_roaming",
    "user_abroad_roaming_disabled_off": "toggle_roaming",
    "data_mode_off": "toggle_data",
    "data_saver_mode_on": "toggle_data_saver_mode",
    "bad_wifi_calling": "toggle_wifi_calling",
    "break_apn_mms_setting": "reset_apn_settings",
    "break_apn_settings": "reset_apn_settings",
    "bad_apn": "reset_apn_settings",
    "break_app_sms_permission": "grant_app_permission",
    "break_app_storage_permission": "grant_app_permission",
    "break_app_both_permissions": "grant_app_permission",
    "unseat_sim_card": "reseat_sim_card",
    "lock_sim_card_pin": "reseat_sim_card",
    "sim_locked": "reseat_sim_card",
    "overdue_bill_suspension": "make_payment",
}


def user_tool_for_bug_tag(tag: str) -> str | None:
    """Map a telecom bug tag to an evidence-backed user-side tool name."""
    tool = _BUG_TAG_TO_USER_TOOL.get(str(tag or "").strip())
    if tool:
        return tool
    hint = _BUG_TAG_USER_HINTS.get(str(tag or "").strip(), "")
    for candidate in sorted(TELECOM_USER_TOOLS, key=len, reverse=True):
        if candidate in hint:
            return candidate
    return None


def bug_tags_for_conditional_guides(
    trajs: list[Tau2Trajectory],
    *,
    require_shared: bool,
) -> list[str]:
    """Bug tags that drive conditional user guides (intersection for multi-traj)."""
    if not trajs:
        return []
    per = [bug_tags_from_task_id(t.task_id) for t in trajs]
    if len(trajs) == 1:
        return list(per[0])
    sets = [set(p) for p in per if p]
    if not sets:
        return []
    if require_shared:
        return sorted(set.intersection(*sets))
    union: set[str] = set()
    for s in sets:
        union |= s
    return sorted(union)


def conditional_guide_steps_from_bug_tags(bug_tags: list[str]) -> list[str]:
    """Format bug-conditioned user guides instead of flat union pepper."""
    steps: list[str] = []
    seen_steps: set[str] = set()
    for tag in bug_tags:
        tool = user_tool_for_bug_tag(tag)
        if not tool:
            continue
        step = f"guide user: {tool} (when {tag})"
        key = step.lower()
        if key in seen_steps:
            continue
        seen_steps.add(key)
        steps.append(step)
    return steps


# Conditional diagnostic steps: guide user / mark flags only — never agent writes.
_BUG_TAG_PROTOCOL_BRANCHES: dict[str, str] = {
    "airplane_mode_on": (
        "if airplane_mode_on (source: user report OR ask user to read status bar/"
        "check_network_status — never call it yourself): instruct user to turn "
        "airplane mode OFF; do not call toggle_airplane_mode"
    ),
    "bad_vpn": (
        "if bad_vpn (source: user report OR ask user to check VPN status — never "
        "call check_vpn_status yourself): instruct user to disconnect VPN; do not "
        "call disconnect_vpn"
    ),
    "bad_network_preference": (
        "if bad_network_preference (source: user report OR ask user to check "
        "network mode preference — never call it yourself): instruct user to fix "
        "network mode preference; do not call set_network_mode_preference"
    ),
    "user_abroad_roaming_enabled_off": (
        "if user_abroad_roaming_enabled_off (source: user report OR line roaming "
        "field from get_details_by_id OR ask user to read Data Roaming status): "
        "mark need_enable_roaming=true; instruct user to turn Data Roaming ON; "
        "do not call toggle_roaming or enable_roaming in this step"
    ),
    "user_abroad_roaming_disabled_on": (
        "if user_abroad_roaming_disabled_on (source: user report OR line roaming "
        "disabled from get_details_by_id): mark need_enable_roaming=true; confirm "
        "before enable_roaming; do not call toggle_roaming in this step"
    ),
    "user_abroad_roaming_disabled_off": (
        "if user_abroad_roaming_disabled_off (source: user report OR line details): "
        "mark need_enable_roaming=true; instruct user to turn Data Roaming ON; "
        "do not call toggle_roaming or enable_roaming in this step"
    ),
    "data_usage_exceeded": (
        "if data_usage_exceeded (source: get_data_usage / user report): "
        "mark need_refuel_data=true; do not call refuel_data in this step"
    ),
    "data_mode_off": (
        "if data_mode_off (source: user report OR ask user to read Mobile Data "
        "status — never call toggle_data yourself): instruct user to turn Mobile "
        "Data ON; do not call toggle_data"
    ),
    "data_saver_mode_on": (
        "if data_saver_mode_on (source: user report OR ask user to check Data "
        "Saver — never call it yourself): instruct user to turn Data Saver OFF; "
        "do not call toggle_data_saver_mode"
    ),
    "bad_wifi_calling": (
        "if bad_wifi_calling (source: user report OR ask user to check Wi-Fi "
        "Calling — never call it yourself): instruct user to toggle Wi-Fi Calling; "
        "do not call toggle_wifi_calling"
    ),
    "break_apn_mms_setting": (
        "if break_apn_mms_setting (source: user report OR ask user to check APN/"
        "MMS APN — never call it yourself): instruct user to reset APN settings "
        "then reboot; do not call reset_apn_settings or reboot_device"
    ),
    "break_apn_settings": (
        "if break_apn_settings (source: user report OR ask user to check APN — "
        "never call it yourself): instruct user to reset APN; do not call "
        "reset_apn_settings"
    ),
    "bad_apn": (
        "if bad_apn (source: user report OR ask user to check APN — never call "
        "it yourself): instruct user to reset/fix APN; do not call reset_apn_settings"
    ),
    "break_app_sms_permission": (
        "if break_app_sms_permission (source: user report OR ask user to check "
        "messaging app permissions — never call it yourself): instruct user to "
        "grant SMS permission; do not call grant_app_permission"
    ),
    "break_app_storage_permission": (
        "if break_app_storage_permission (source: user report OR ask user to check "
        "messaging app permissions — never call it yourself): instruct user to "
        "grant storage permission; do not call grant_app_permission"
    ),
    "break_app_both_permissions": (
        "if break_app_both_permissions (source: user report OR ask user to check "
        "messaging app permissions — never call it yourself): instruct user to "
        "grant SMS and storage permissions; do not call grant_app_permission"
    ),
    "unseat_sim_card": (
        "if unseat_sim_card (source: user report OR ask user to check SIM status — "
        "never call it yourself): instruct user to reseat the SIM; do not call "
        "reseat_sim_card"
    ),
    "lock_sim_card_pin": (
        "if lock_sim_card_pin (source: user report OR ask user to check SIM PIN): "
        "instruct user to unlock SIM PIN / reseat SIM; do not call reseat_sim_card"
    ),
    "sim_locked": (
        "if sim_locked (source: user report OR ask user to check SIM status): "
        "instruct user to reseat SIM; do not call reseat_sim_card"
    ),
    "overdue_bill_suspension": (
        "if overdue_bill_suspension (source: get_details_by_id / bill tools): "
        "mark need_payment_or_resume=true; use send_payment_request / resume_line "
        "only after confirming; else transfer_to_human_agents"
    ),
    "contract_end_suspension": (
        "if contract_end_suspension (source: line status from get_details_by_id): "
        "explain out-of-scope if policy requires; prefer transfer_to_human_agents "
        "after a genuine lookup attempt"
    ),
}

_DIALOGUE_STEP_PREFIXES = (
    "if ",
    "confirm ",
    "resolve ",
    "extract ",
    "communicate ",
    "guide ",
    "instruct ",
    "ask ",
    "diagnose ",
    "mark ",
)


def is_dialogue_or_branch_step(step: str) -> bool:
    """True for non-tool protocol rows (confirm / disambiguate / if-branches)."""
    text = str(step or "").strip()
    if not text:
        return True
    low = text.lower()
    if low.startswith(_DIALOGUE_STEP_PREFIXES):
        return True
    # Tool-shaped steps contain name(...); bare instructions do not.
    if "(" not in text:
        return True
    return False


def is_agent_tool_step(step: str, *, domain: str | None = None) -> bool:
    """True when the step is an agent-callable tool invocation."""
    if is_dialogue_or_branch_step(step):
        return False
    name = _tool_name(step)
    if not name:
        return False
    if domain is None or str(domain).lower() in {"", "telecom", "telecom-workflow"}:
        if is_telecom_user_side_tool(name):
            return False
    return True


def agent_tool_protocol(
    protocol: list[str],
    *,
    domain: str | None = None,
) -> list[str]:
    """Extract agent-callable tool rows from a (possibly mixed) action_protocol.

    Also recovers tool calls embedded in ``if … and user says yes: tool(...)``
    rows so credit/coverage still see the write spine. Dialogue-only rows are
    skipped. **Executors must follow ``action_protocol``**, not this subset.
    """
    out: list[str] = []
    for step in protocol or []:
        text = str(step or "").strip()
        if not text:
            continue
        low = text.lower()
        # Recover: "if need_x and user says yes: enable_roaming(...)"
        if low.startswith("if ") and " user says yes:" in low:
            _, _, tail = text.partition(":")
            embedded = tail.strip()
            if is_agent_tool_step(embedded, domain=domain):
                out.append(embedded)
            continue
        if is_agent_tool_step(step, domain=domain):
            out.append(step)
    return out


def diagnostic_branch_steps(bug_tags: list[str] | None) -> list[str]:
    """Ordered conditional instruct-user branches from evidence bug tags."""
    out: list[str] = []
    seen: set[str] = set()
    for tag in bug_tags or []:
        branch = _BUG_TAG_PROTOCOL_BRANCHES.get(str(tag).strip())
        if not branch or branch in seen:
            continue
        seen.add(branch)
        out.append(branch)
    return out


def compose_executable_protocol(
    agent_protocol: list[str],
    *,
    domain: str | None = None,
    bug_tags: list[str] | None = None,
) -> list[str]:
    """Inline confirm / disambiguate / diagnostic branches into action_protocol.

    Order: reads → resolve line_id → diagnose (guide/mark only) → confirm with
    yes/no exits → single write → communicate. Conditional branches never call
    write tools.
    """
    agent = [
        step
        for step in (agent_protocol or [])
        if is_agent_tool_step(step, domain=domain)
    ]
    if not agent:
        return list(agent_protocol or [])

    domain_l = str(domain or "").lower()
    out: list[str] = []
    if any(_tool_name(s) == "get_customer_by_phone" for s in agent):
        out.append("extract phone_number from user; if missing, ask")

    reads = [s for s in agent if _tool_name(s) not in _WRITE_OPS]
    writes = [s for s in agent if _tool_name(s) in _WRITE_OPS]
    out.extend(reads)

    needs_line = domain_l in {"telecom", "telecom-workflow"} or any(
        _tool_name(s)
        in {
            "get_details_by_id",
            "enable_roaming",
            "refuel_data",
            "resume_line",
            "suspend_line",
        }
        for s in agent
    )
    if needs_line and writes:
        out.append(
            "resolve target line_id from get_details_by_id; if multiple lines, "
            "ask user to disambiguate before any write"
        )

    tags = [str(t).strip() for t in (bug_tags or []) if str(t).strip()]
    if tags:
        out.append(
            "diagnose boolean flags from (in order): user description; "
            "agent get_details_by_id / get_data_usage fields when present; "
            "ask user to read device status (check_network_status / status bar / "
            "VPN / network preference) and report back — never call user-side APIs. "
            f"Relevant flags: {', '.join(tags)}"
        )
        out.extend(diagnostic_branch_steps(tags))

    # Re-bind write args so line_id points at the resolve step, not a re-ask.
    writes = bind_protocol_slots(writes, resolved_line_id=bool(needs_line and writes))

    for step in writes:
        wname = _tool_name(step)
        if wname == "enable_roaming":
            out.append(
                "if need_enable_roaming and user says yes: "
                f"{step}"
            )
            out.append(
                "if need_enable_roaming and user says no: skip write; communicate "
                "that user-side fixes may still resolve the issue; do not call "
                "enable_roaming"
            )
        elif wname == "refuel_data":
            out.append(f"if need_refuel_data and user says yes: {step}")
            out.append(
                "if need_refuel_data and user says no: skip write; communicate "
                "outcome; do not call refuel_data"
            )
        else:
            out.append(
                f"if user says yes: {step}"
            )
            out.append(
                f"if user says no: skip {wname}; communicate without mutating"
            )

    if writes:
        out.append(
            "communicate outcome; if symptoms remain, guide remaining user-side "
            "checks (never call user-side APIs)"
        )
    return out


def user_side_hints_from_trajs(trajs: list[Tau2Trajectory]) -> list[str]:
    """Aggregate companion user-device hints from evidence action_checks only."""
    hints: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        key = text.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        hints.append(text.strip())

    for name in matched_user_side_tool_names_from_trajs(trajs, require_all=False):
        _add(f"user-side fix seen in evidence: {name}")
    return hints


def matched_user_side_tool_names(traj: Tau2Trajectory) -> list[str]:
    """User-device tool names with ``action_match`` in episode evidence."""
    out: list[str] = []
    seen: set[str] = set()
    checks = (traj.metadata or {}).get("action_checks") or []
    if not isinstance(checks, list):
        return out
    for check in checks:
        if not isinstance(check, dict):
            continue
        # Prefer matched gold; if match flag absent, keep requestor=user rows.
        matched = check.get("action_match")
        if matched is False:
            continue
        action = check.get("action") or {}
        if not isinstance(action, dict):
            continue
        if str(action.get("requestor") or "").lower() != "user":
            continue
        uname = str(action.get("name") or "").strip()
        if not uname or not is_telecom_user_side_tool(uname):
            continue
        if uname in seen:
            continue
        seen.add(uname)
        out.append(uname)
    return out


def matched_user_side_tool_names_from_trajs(
    trajs: list[Tau2Trajectory],
    *,
    require_all: bool = True,
) -> list[str]:
    """Merge matched user-side tools across trajs (intersection or ordered union)."""
    if not trajs:
        return []
    per = [matched_user_side_tool_names(t) for t in trajs]
    if require_all and len(per) > 1:
        common = set(per[0])
        for names in per[1:]:
            common &= set(names)
        return [n for n in per[0] if n in common]
    # Union preserving first-seen order.
    out: list[str] = []
    seen: set[str] = set()
    for names in per:
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
    return out


def guide_user_steps_from_names(names: list[str]) -> list[str]:
    """Format evidence user-device fixes as non-tool protocol rows."""
    return [f"guide user: {name}" for name in names if str(name).strip()]


def compose_protocol_with_user_guides(
    agent_protocol: list[str],
    trajs: list[Tau2Trajectory],
    *,
    require_shared_guides: bool = True,
    bug_tags_core: list[str] | None = None,
) -> list[str]:
    """Append conditional ``guide user: tool (when tag)`` rows after agent steps."""
    base = list(agent_protocol or [])
    if bug_tags_core is not None:
        tags = list(bug_tags_core)
        if not tags and len(trajs) == 1:
            tags = bug_tags_for_conditional_guides(
                trajs, require_shared=require_shared_guides
            )
    else:
        tags = bug_tags_for_conditional_guides(
            trajs, require_shared=require_shared_guides
        )
    guides = conditional_guide_steps_from_bug_tags(tags)
    covered_tools: set[str] = set()
    for g in guides:
        if "(when " not in g:
            continue
        tool_part = g.split("guide user:", 1)[1].split("(when", 1)[0].strip()
        if tool_part:
            covered_tools.add(tool_part)
    if len(trajs) == 1 and trajs:
        stable = matched_user_side_tool_names(trajs[0])
    else:
        stable = stable_guide_names_from_trajs(trajs)
    for name in stable:
        if name in covered_tools:
            continue
        guides.append(f"guide user: {name}")
    if not guides:
        return base
    existing = {str(s).strip().lower() for s in base}
    extra = [g for g in guides if g.lower() not in existing]
    return base + extra


def _success_attribution(trajs: list[Tau2Trajectory]) -> dict[str, Any]:
    """Describe whether wins look DB-backed vs ENV/user-side."""
    n = len(trajs)
    db_match_ok = sum(1 for t in trajs if t.has_positive_db_effect)
    env_reward_ok = sum(1 for t in trajs if t.success)
    user_fix_tags: set[str] = set()
    user_fix_tools: set[str] = set()
    for traj in trajs:
        user_fix_tags.update(bug_tags_from_task_id(traj.task_id))
        for check in (traj.metadata or {}).get("action_checks") or []:
            if not isinstance(check, dict):
                continue
            action = check.get("action") or {}
            if str(action.get("requestor") or "").lower() != "user":
                continue
            uname = str(action.get("name") or "").strip()
            if uname:
                user_fix_tools.add(uname)
    domain = str(getattr(trajs[0], "domain", "") or "").lower() if trajs else ""
    env_primary = domain in {"telecom", "telecom-workflow"} and env_reward_ok > 0 and (
        db_match_ok == 0 or bool(user_fix_tools)
    )
    return {
        "env_reward_ok": env_reward_ok,
        "db_match_ok": db_match_ok,
        "n": n,
        "env_user_side_primary": bool(env_primary),
        "user_fix_tags": sorted(user_fix_tags),
        "user_fix_tools": sorted(user_fix_tools),
        "agent_write_role": (
            "secondary_or_partial"
            if env_primary
            else ("primary" if db_match_ok == n and n else "mixed")
        ),
    }


def _primary_task_family_for_skill(
    trajs: list[Tau2Trajectory],
    capability_key: str,
    task_families: list[str],
    attribution: dict[str, Any],
    *,
    success_mode: str | None = None,
) -> str | None:
    """Prefer write scope for solve_write; episode scope for ENV-user wins."""
    cap = capability_scope_from_key(capability_key)
    if str(success_mode or "") == "solve_write" and not bool(
        attribution.get("env_user_side_primary")
    ):
        if cap:
            return cap
    prefer_episode = bool(attribution.get("env_user_side_primary")) or str(
        success_mode or ""
    ) == "communicate_ok"
    if prefer_episode:
        scopes = [
            episode_scope_from_task_id(t.task_id)
            for t in trajs
            if episode_scope_from_task_id(t.task_id)
        ]
        if scopes:
            # Most common bracket scope (e.g. mobile_data_issue).
            return Counter(scopes).most_common(1)[0][0]
    if task_families:
        if str(success_mode or "") == "communicate_ok":
            # Prefer a real episode bracket over capability lookup stems.
            for fam in task_families:
                if fam and fam != cap and not str(fam).startswith("get_"):
                    return fam
        return cap or task_families[0]
    return cap


def _routing_capability_key(
    write_capability: str,
    trajs: list[Tau2Trajectory],
    attribution: dict[str, Any],
) -> str:
    """Parent capability when ENV/user-side fixes dominate the win.

    Keep the write spine key when there is no known multi-bug / user-fix
    evidence (e.g. a clean ``refuel_data`` solve).
    """
    if not attribution.get("env_user_side_primary"):
        return write_capability
    tags = {
        str(t).strip()
        for t in (attribution.get("user_fix_tags") or [])
        if str(t).strip()
    }
    known = tags & set(_BUG_TAG_PROTOCOL_BRANCHES)
    tools = attribution.get("user_fix_tools") or []
    if not known and not tools:
        return write_capability
    scopes = [
        episode_scope_from_task_id(t.task_id)
        for t in trajs
        if episode_scope_from_task_id(t.task_id)
    ]
    scope = Counter(scopes).most_common(1)[0][0] if scopes else "mobile_data_issue"
    abroadish = any(("abroad" in t) or ("roaming" in t) for t in known)
    if abroadish:
        return f"tau2.{scope}_abroad"
    return f"tau2.{scope}"


def _activation_gate_payload(
    evidence_bug_signatures: list[list[str]],
) -> dict[str, Any]:
    """Shared-core / union used by ``union_with_signature_gate`` matching."""
    sets = [set(sig) for sig in evidence_bug_signatures if sig]
    if not sets:
        return {
            "activation_signatures": evidence_bug_signatures,
            "bug_intersection": [],
            "bug_union": [],
        }
    inter = set.intersection(*sets) if sets else set()
    union: set[str] = set()
    for s in sets:
        union |= s
    return {
        "activation_signatures": evidence_bug_signatures,
        "bug_intersection": sorted(inter),
        "bug_union": sorted(union),
    }


def status_for_birth_support(
    support: int,
    *,
    min_support_for_verified: int = DEFAULT_MIN_SUPPORT_FOR_VERIFIED,
) -> SkillStatus:
    """Initial distill status before credit seeding.

    Low support stays ``candidate`` (honest); credit may later raise to
    ``verified_low_support`` or ``verified``.
    """
    n = int(support or 0)
    if n <= 0:
        return SkillStatus.CANDIDATE
    if n < int(min_support_for_verified):
        return SkillStatus.CANDIDATE
    return SkillStatus.PROVISIONAL


def _scopes_from_trajs(trajs: list[Tau2Trajectory], capability_key: str) -> list[str]:
    """Learned routing scopes: benchmark bracket + capability write stem."""
    scopes: set[str] = set()
    for traj in trajs:
        scope = episode_scope_from_task_id(traj.task_id)
        if scope:
            scopes.add(scope)
    cap_scope = capability_scope_from_key(capability_key)
    if cap_scope:
        scopes.add(cap_scope)
    return sorted(scopes)


# High-signal lexical cues only (avoid generic tokens like data/mode/off/user).
_INTENT_CUE_VOCAB = {
    "cancel",
    "return",
    "exchange",
    "refund",
    "book",
    "baggage",
    "passenger",
    "address",
    "roaming",
    "airplane",
    "mms",
    "sms",
    "wifi",
    "refuel",
    "sim",
    "vpn",
    "france",
    "abroad",
    "apn",
}

# Generic fragments never kept as standalone intent cues.
_INTENT_CUE_BLOCKLIST = {
    "bad",
    "data",
    "enabled",
    "mode",
    "network",
    "off",
    "on",
    "user",
    "slow",
    "mobile",
    "preference",
    "issue",
    "the",
    "and",
}


def _intent_cues_from_trajs_legacy(trajs: list[Tau2Trajectory]) -> list[str]:
    """Original lexical cue extraction (small vocab only)."""
    cue_vocab = {
        "cancel",
        "return",
        "exchange",
        "refund",
        "book",
        "baggage",
        "passenger",
        "address",
        "roaming",
        "airplane",
        "mms",
        "sms",
        "wifi",
        "refuel",
        "sim",
    }
    found: set[str] = set()
    for traj in trajs:
        blob = " ".join(traj.user_texts[:4]).lower()
        for cue in cue_vocab:
            if cue in blob:
                found.add(cue)
    return sorted(found)


def _intent_cues_from_trajs(trajs: list[Tau2Trajectory]) -> list[str]:
    """High-signal cues: curated vocab + full bug tags + a few phrases."""
    found: set[str] = set()
    for traj in trajs:
        blob = " ".join(traj.user_texts[:4]).lower()
        for cue in _INTENT_CUE_VOCAB:
            if cue in blob:
                found.add(cue)
        if ("mobile data" in blob or "mobile_data" in blob) and (
            "slow" in blob or "not working" in blob or "isn't working" in blob
        ):
            found.add("mobile_data_slow")
        for tag in bug_tags_from_task_id(traj.task_id):
            tag_l = tag.lower().strip()
            if tag_l and tag_l not in _INTENT_CUE_BLOCKLIST:
                found.add(tag_l)
    return sorted(c for c in found if c not in _INTENT_CUE_BLOCKLIST)


def distill_skills_from_trajectories(
    trajectories: list[Tau2Trajectory],
    *,
    min_support: int = 1,
    require_success: bool = True,
    require_solve_write: bool | None = None,
    allowed_success_modes: frozenset[str] | set[str] | list[str] | None = None,
    min_protocol_len: int = 1,
    max_skills: int = 32,
    min_support_for_verified: int = DEFAULT_MIN_SUPPORT_FOR_VERIFIED,
    bind_slots: bool = True,
    bug_bucket_mode: str | None = None,
    inline_dialogue_in_protocol: bool = False,
    require_action_checks_ok: bool | None = None,
    require_shared_guides: bool | None = None,
) -> list[Tau2Skill]:
    """Build provisional skills from successful tool / communicate protocols.

    Only trajectories with a grounded tool protocol and (by default) a
    successful reward/DB signal can create skills — mirrors SAGE's rule that
    failed no-progress traces cannot birth skills.

    Protocols are canonicalized (collapse consecutive duplicates; write-spine
    trim is off by default) and bucketed by
    ``(success_mode, capability, tool_name_spine)``.

    Default modes: ``solve_write`` + ``communicate_ok``.
    ``require_solve_write=True`` restores write-only distill.
    ``inline_dialogue_in_protocol`` (default False) optionally reinjects
    hand-written confirm/diagnose branches.

    ``min_support_for_verified`` gates initial status honesty: low-n skills are
    born as ``candidate`` (credit may later mark ``verified_low_support``).

    ``bug_bucket_mode`` (telecom default ``union_with_signature_gate``):
      - ``off``: ignore bug tags when merging
      - ``exact`` / ``signature_match``: only merge identical bug signatures
      - ``union``: merge by tool spine; inline union of diagnostic branches
      - ``union_with_signature_gate``: like union, but inject matching requires
        shared bug core ⊆ episode ⊆ union (see skill_matches_episode)
      - ``primary_write``: deprecated alias of ``union_with_signature_gate``
        (no write-whitelist sharpening)

    ``require_action_checks_ok`` (telecom default True): only distill from
    episodes whose gold action_checks all match and are non-transfer-only.
    ``require_shared_guides`` defaults to True (intersection / conditional guides).
    Evidence-only user tools (e.g. ``make_payment``) still attach via stable co-occurrence.
    """
    modes = resolve_allowed_success_modes(
        require_solve_write=require_solve_write,
        allowed_success_modes=allowed_success_modes,
    )
    shared_guides = True if require_shared_guides is None else bool(require_shared_guides)
    buckets: dict[
        tuple[str, ...], list[tuple[list[str], Tau2Trajectory, str]]
    ] = defaultdict(list)
    for traj in trajectories:
        if not trajectory_eligible_for_distill(
            traj,
            require_success=require_success,
            require_solve_write=require_solve_write,
            allowed_success_modes=modes,
            require_action_checks_ok=require_action_checks_ok,
        ):
            continue
        success_mode = infer_success_mode(traj)
        protocol = normalize_protocol(
            list(traj.tool_protocol or []),
            domain=getattr(traj, "domain", None),
        )
        if len(protocol) < min_protocol_len:
            continue
        # Drop pure-think / empty.
        if all(step.startswith("think(") for step in protocol):
            continue
        # Escalation-only stays Executor fallback, never a specialty teacher.
        if is_escalate_only_capability(_capability_key(protocol)):
            continue
        if success_mode == "transfer_ok":
            continue
        # Read-only protocols are allowed only for communicate_ok.
        if is_read_only_protocol(protocol) and success_mode != "communicate_ok":
            continue
        # solve_write still needs a real write in the normalized protocol.
        if success_mode == "solve_write" and not protocol_has_write(protocol):
            continue
        domain = str(getattr(traj, "domain", "") or "").lower()
        mode = bug_bucket_mode
        if mode is None:
            mode = (
                "union_with_signature_gate"
                if domain in {"telecom", "telecom-workflow"}
                else "off"
            )
        mode_s = str(mode).lower().strip()
        # Legacy alias: sharp write-whitelist bucketing removed.
        if mode_s == "primary_write":
            mode_s = "union_with_signature_gate"
        # exact / signature_match both split buckets by full bug signature.
        bucket_mode = (
            "exact"
            if mode_s in {"exact", "signature_match"}
            else mode_s
        )
        guide_names = matched_user_side_tool_names(traj)
        # Communicate repairs differ mainly by user-side fixes — keep them apart.
        # Write spines still merge on tools/bugs only.
        if success_mode == "communicate_ok" and guide_names:
            key = (
                success_mode,
                *_protocol_bucket_key_with_bugs(
                    protocol, traj, bug_bucket_mode=bucket_mode, domain=domain
                ),
                "guides",
                *guide_names,
            )
        else:
            key = (
                success_mode,
                *_protocol_bucket_key_with_bugs(
                    protocol, traj, bug_bucket_mode=bucket_mode, domain=domain
                ),
            )
        buckets[key].append((protocol, traj, ""))

    ranked = sorted(
        buckets.items(),
        key=lambda item: (
            -len(item[1]),
            -len(item[0]),
            item[0],
        ),
    )
    skills: list[Tau2Skill] = []
    for _key, members in ranked:
        if len(members) < min_support:
            continue
        if len(skills) >= max_skills:
            break
        bug_sig: tuple[str, ...] = ()
        if "bugs" in _key:
            bi = _key.index("bugs")
            bug_sig = tuple(str(b) for b in _key[bi + 1 :])
        domain0 = str(getattr(members[0][1], "domain", "") or "").lower()
        bucket_writes = bucket_writes_from_key(_key)
        ordered_writes = sorted(bucket_writes) or primary_write_names(members[0][0])
        if domain0 in {"telecom", "telecom-workflow"} and ordered_writes:
            protocol_raw = pick_cleanest_protocol_member(
                members, ordered_writes[0], bucket_writes
            )[0]
        else:
            members_sorted = sorted(members, key=lambda m: (len(m[0]), m[0]))
            protocol_raw = members_sorted[0][0]
            if bucket_writes:
                protocol_raw = filter_protocol_to_bucket_writes(
                    protocol_raw, bucket_writes
                )
            protocol_raw = collapse_nonconsecutive_duplicate_tools(protocol_raw)
        trajs = [traj for _, traj, _ in members]
        domain = trajs[0].domain
        agent_bound = (
            bind_protocol_slots(protocol_raw) if bind_slots else list(protocol_raw)
        )
        if bind_slots:
            agent_bound = bind_write_args_from_evidence(agent_bound, trajs)
        success_mode = str(_key[0])
        write_capability = _capability_key(protocol_raw)
        evidence_ids = [t.evidence_id for t in trajs]
        attribution = _success_attribution(trajs)
        # Organizational identity = capability key from tools (write or read).
        # Episode / ENV-primary labels stay in parent_capability_key for soft cues.
        parent_capability = _routing_capability_key(
            write_capability, trajs, attribution
        )
        capability = write_capability
        task_families = _scopes_from_trajs(trajs, write_capability)
        # Ensure parent routing scope is listed.
        parent_scope = capability_scope_from_key(parent_capability)
        if parent_scope and parent_scope not in task_families:
            task_families = sorted(set(task_families) | {parent_scope})
        primary_family = _primary_task_family_for_skill(
            trajs,
            parent_capability,
            task_families,
            attribution,
            success_mode=success_mode,
        )
        intent_cues = _intent_cues_from_trajs(trajs)
        dialogue_gates = dialogue_gates_for_protocol(agent_bound, domain=domain)
        user_hints = user_side_hints_from_trajs(trajs)
        bug_union: list[str] = []
        seen_bugs: set[str] = set()
        evidence_bug_signatures: list[list[str]] = []
        for traj in trajs:
            tags = list(bug_tags_from_task_id(traj.task_id))
            evidence_bug_signatures.append(tags)
            for tag in tags:
                if tag not in seen_bugs:
                    seen_bugs.add(tag)
                    bug_union.append(tag)
        if inline_dialogue_in_protocol:
            protocol = compose_executable_protocol(
                agent_bound,
                domain=domain,
                bug_tags=bug_union,
            )
        else:
            protocol = list(
                bind_protocol_slots(agent_bound, resolved_line_id=False)
            )
        meta_flags: dict[str, Any] = {}
        gate_payload = _activation_gate_payload(evidence_bug_signatures)
        bug_core = list(gate_payload.get("bug_intersection") or [])
        protocol = compose_protocol_with_user_guides(
            protocol,
            trajs,
            require_shared_guides=bool(shared_guides),
        )
        from sage_tau2.contracts import interleave_evidence_guides
        protocol = interleave_evidence_guides(protocol, trajs)
        # Preserve causal dependencies; do not move writes ahead of user actions.
        # Agent-tool subset is derived from the final action_protocol (incl. yes-branch embeds).
        agent_tools = agent_tool_protocol(protocol, domain=domain)
        if not agent_tools:
            agent_tools = list(
                bind_protocol_slots(agent_bound, resolved_line_id=True)
            )
        support = len(trajs)
        birth_status = status_for_birth_support(
            support,
            min_support_for_verified=min_support_for_verified,
        )
        mode_used = bug_bucket_mode
        if mode_used is None:
            mode_used = (
                "union_with_signature_gate"
                if str(domain).lower() in {"telecom", "telecom-workflow"}
                else "off"
            )
        elif str(mode_used).lower().strip() == "primary_write":
            mode_used = "union_with_signature_gate"
        tool_names = [
            _tool_name(step) for step in protocol_raw if _tool_name(step)
        ]
        protocol_bucket = [success_mode, capability, *tool_names]
        primary_write_name = (primary_write_names(protocol_raw)[:1] or [None])[0]
        skill = Tau2Skill(
            skill_name=_skill_name(capability, support),
            description=(
                f"Reusable τ² {success_mode} protocol distilled from successful "
                "episodes: " + " -> ".join(agent_tools or agent_bound)
            ),
            precondition=_precondition_from_trajs(
                trajs,
                primary_write=str(primary_write_name or ""),
                bug_tags_core=bug_core,
            ),
            action_protocol=list(protocol),
            expected_effect=_expected_effect(trajs),
            capability_key=capability,
            status=birth_status,
            support_count=support,
            evidence_ids=evidence_ids,
            domain=domain,
            metadata={
                "source": "tau2_heuristic_distill",
                "task_ids": sorted({t.task_id for t in trajs}),
                "protocol_len": len(protocol),
                "task_families": task_families,
                "primary_task_family": primary_family,
                "intent_cues": intent_cues,
                "protocol_bucket": protocol_bucket,
                "protocol_bucket_key": list(_key),
                "protocol_bucket_write_spine": list(_key),  # legacy alias
                "protocol_unbound": list(protocol_raw),
                "agent_tool_protocol": list(agent_tools),
                "agent_tool_protocol_purpose": (
                    "credit/coverage subset derived from action_protocol"
                ),
                "write_capability_key": write_capability,
                "parent_capability_key": parent_capability,
                "primary_write": primary_write_name,
                "bug_tags_core": bug_core,
                "sharp_card": False,
                "bug_signature": list(bug_sig),
                **meta_flags,
                "dialogue_gates": dialogue_gates,
                "user_side_hints": user_hints,
                "guide_user_steps": [
                    s for s in protocol if str(s).lower().startswith("guide user:")
                ],
                "bug_tags_union": bug_union,
                "evidence_bug_signatures": evidence_bug_signatures,
                "bug_bucket_mode": str(mode_used),
                **gate_payload,
                "success_attribution": attribution,
                "success_mode": success_mode,
                "success_modes": [success_mode],
                "inline_dialogue_in_protocol": bool(inline_dialogue_in_protocol),
                "support_tier": (
                    "strong" if support >= min_support_for_verified else "low"
                ),
                "min_support_for_verified": int(min_support_for_verified),
                # Preserve legacy text builders for debugging / A-B.
                "precondition_legacy": _precondition_from_trajs_legacy(trajs),
                "expected_effect_legacy": _expected_effect_legacy(trajs),
                "intent_cues_legacy": _intent_cues_from_trajs_legacy(trajs),
            },
        )
        from sage_tau2.contracts import learn_contract
        skill.metadata["execution_contract"] = learn_contract(skill, trajs)
        skills.append(skill)
    return skills


def summarize_protocol_histogram(
    trajectories: list[Tau2Trajectory],
) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    for traj in trajectories:
        key = " -> ".join(traj.tool_protocol) if traj.tool_protocol else "<empty>"
        counter[key] += 1
    return {
        "n_trajectories": len(trajectories),
        "n_unique_protocols": len(counter),
        "top": counter.most_common(20),
    }
