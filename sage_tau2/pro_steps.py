"""ALF-like PRO step shaping for τ² trajectories (v4).

Builds per-assistant-turn records with:
  observation_before / action / observation (semantic)
  agent_messages (full reply; ``<think>`` stays here like sage_mas ALF dumps)
  history_summary as causal (observation_before, action) pairs
  prompt = live window actually usable for cloning
  runtime_inject (skills/org/dispatch; bulky prompts stored once)

Does **not** copy the ALFRED prompt template; it only mirrors the structured
logging that GiGPO uses for distillation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from sage_tau2.prompts import TAU2_TEMPLATE, TAU2_TEMPLATE_NO_HIS

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_DEFAULT_TASK = "(resolve the customer's request per domain policy)"
_DEFAULT_TOOLS_PLACEHOLDER = "'(tools provided via function calling)'"
_DEFAULT_HISTORY_WINDOW = 10
_DEFAULT_OBS_CHARS = 520
_DEFAULT_ACT_CHARS = 160
_SEMANTIC_OBS_CHARS = 480
# Full dump: keep near-raw user/tool text (still soft-capped for huge payloads).
_FULL_OBS_CHARS = 8000
_FULL_TOOL_CHARS = 8000

# ---------------------------------------------------------------------------
# Semantic compression design (GiGPO-style: dense, not complete)
#
# Goal: rich enough for the *next* tool args / decisions, short enough for a
# history window of ~10 steps.
#
# Keep (actionable):
#   ids, last_four, cabin/status, flight dates+od+$price, search eco/biz,
#   bag counts, dob/name when present
# Drop / defer (decorative):
#   address, email, scheduled clock times (except status tools), nested blobs
# Encoding:
#   short tokens (05-24, ATL>PHL, eco=), atomic units packed into a char budget
# Never:
#   mid-token / mid-option clips that invent broken ids or prices
# ---------------------------------------------------------------------------

# Prefer these JSON keys when compressing tool payloads.
# Keep actionable ids (reservations / payment_methods) early so short history
# clips do not truncate mid-id (e.g. gift_card_8190333 → gift_card_819).
_PRIORITY_KEYS = (
    "phone_number",
    "line_id",
    "customer_id",
    "data_used_gb",
    "data_limit_gb",
    "data_refueling_gb",
    "roaming_enabled",
    "contract_end_date",
    "suspension_start_date",
    "reservation_id",
    "user_id",
    "order_id",
    "item_id",
    "name",
    "dob",
    "membership",
    "reservations",
    "payment_methods",
    "payment_history",
    "payment_method_id",
    "payment_id",
    "certificate_id",
    "flight_number",
    "origin",
    "destination",
    "cabin",
    "flight_type",
    "status",
    "flights",
    "passengers",
    "created_at",
    "total_baggages",
    "nonfree_baggages",
    "insurance",
    "error",
    "message",
)

# Low-value blobs: only include if there is still room after priority fields.
_DEFER_KEYS = frozenset(
    {
        "saved_passengers",
    }
)

# Always omit from window obs (never actionable for τ² writes).
_OMIT_KEYS = frozenset(
    {
        "email",
        "address",
    }
)

# Dict maps whose *keys* are actionable ids (keep key list + last4 when present).
_ID_MAP_KEYS = frozenset(
    {
        "payment_methods",
        "payments",
        "payment_method",
    }
)

_LIST_ID_KEYS = (
    "reservation_id",
    "order_id",
    "flight_number",
    "user_id",
    "payment_id",
    "item_id",
    "id",
)

# Scalar profile / booking fields that must survive compression.
_KEEP_SCALARS = frozenset(
    {
        "date",
        "price",
        "cabin",
        "dob",
        "created_at",
        "total_baggages",
        "nonfree_baggages",
        "insurance",
        "last_four",
    }
)


def _fmt_price_val(val: Any) -> str:
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def _price_bits(prices: Any, *, cabins: Sequence[str] | None = None) -> str:
    """Compact cabin prices: ``be=68/eco=110/biz=400``."""
    if not isinstance(prices, dict) or not prices:
        return ""
    wanted = cabins or ("basic_economy", "economy", "business")
    short_map = {
        "basic_economy": "be",
        "economy": "eco",
        "business": "biz",
    }
    parts: list[str] = []
    for key in wanted:
        if key in prices and prices[key] is not None:
            short = short_map.get(key, key)
            parts.append(f"{short}={_fmt_price_val(prices[key])}")
    return "/".join(parts)


def _short_date(date: Any) -> str:
    """``2024-05-24`` → ``05-24`` to save search-list budget."""
    text = str(date or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[5:10]
    return text


def _economy_sort_key(item: Any) -> float:
    if isinstance(item, dict):
        prices = item.get("prices")
        if isinstance(prices, dict) and prices.get("economy") is not None:
            return float(prices["economy"])
        return 1e18
    if isinstance(item, list):
        total = 0.0
        any_price = False
        for seg in item:
            if not isinstance(seg, dict):
                continue
            prices = seg.get("prices")
            if isinstance(prices, dict) and prices.get("economy") is not None:
                total += float(prices["economy"])
                any_price = True
        return total if any_price else 1e18
    return 1e18


def _format_direct_flight_option(
    item: dict[str, Any],
    *,
    compact: bool = False,
) -> str:
    fn = item.get("flight_number") or "?"
    date = item.get("date")
    origin = item.get("origin")
    dest = item.get("destination")
    bit = str(fn)
    if date:
        bit += f"@{_short_date(date) if compact else date}"
    if origin or dest:
        sep = ">" if compact else "->"
        bit += f"({origin or '?'}{sep}{dest or '?'})"
    # Search lists: keep eco (+biz) only — be rarely decides "cheapest economy".
    cabins = ("economy", "business") if compact else None
    pb = _price_bits(item.get("prices"), cabins=cabins)
    if pb:
        bit += f" {pb}"
    return bit


def _format_onestop_option(
    segs: list[Any],
    *,
    compact: bool = False,
) -> str | None:
    flights = [s for s in segs if isinstance(s, dict) and s.get("flight_number")]
    if len(flights) < 2:
        if len(flights) == 1:
            return _format_direct_flight_option(flights[0], compact=compact)
        return None
    nums = "+".join(str(f.get("flight_number") or "?") for f in flights)
    dates = [str(f.get("date")) for f in flights if f.get("date")]
    date_s = f"@{_short_date(dates[0]) if compact else dates[0]}" if dates else ""
    o0 = flights[0].get("origin") or "?"
    d_last = flights[-1].get("destination") or "?"
    via = flights[0].get("destination") or "?"
    sep = ">" if compact else "->"
    route = (
        f"({o0}{sep}{via}{sep}{d_last})"
        if len(flights) == 2
        else f"({o0}{sep}{d_last})"
    )
    totals: dict[str, float] = {}
    for f in flights:
        prices = f.get("prices")
        if not isinstance(prices, dict):
            continue
        for k, v in prices.items():
            if isinstance(v, (int, float)):
                totals[k] = totals.get(k, 0.0) + float(v)
    cabins = ("economy", "business") if compact else None
    pb = _price_bits(totals, cabins=cabins)
    bit = f"{nums}{date_s}{route}"
    if pb:
        bit += f" {pb}"
    return bit


def _looks_like_flight_option(item: Any) -> bool:
    if isinstance(item, dict) and item.get("flight_number"):
        return True
    if isinstance(item, list) and item:
        return all(isinstance(x, dict) and x.get("flight_number") for x in item)
    return False


def _format_flight_search_list(
    val: list[Any],
    *,
    max_items: int = 8,
    max_chars: int | None = None,
) -> str:
    """Keep searchable options with prices; pack whole options into budget.

    Never emit a half-option (avoids ``eco=2`` mid-price clips that cause
    misquotes). Prefer shorter per-option format so ~8 fits in the window.
    """
    if not val:
        return "list=[]"
    budget = max_chars if max_chars and max_chars > 0 else _SEMANTIC_OBS_CHARS
    # Reserve room for tool-name prefix in semantic_summarize_json.
    budget = max(120, budget - 28)
    ranked = sorted(val, key=_economy_sort_key)
    parts: list[str] = []
    for item in ranked[:max_items]:
        if isinstance(item, dict):
            bit = _format_direct_flight_option(item, compact=True)
        elif isinstance(item, list):
            bit = _format_onestop_option(item, compact=True)
        else:
            continue
        if not bit:
            continue
        trial = parts + [bit]
        leftover = len(ranked) - len(trial)
        body = f"list=[{'; '.join(trial)}]"
        suffix = f" (+{leftover})" if leftover > 0 else ""
        if len(body + suffix) <= budget:
            parts.append(bit)
            continue
        if parts:
            break
        # First option alone exceeds budget: still keep one complete line.
        parts.append(bit)
        break
    if not parts:
        return f"list_n={len(val)}"
    leftover = len(ranked) - len(parts)
    body = f"list=[{'; '.join(parts)}]"
    if leftover > 0:
        body += f" (+{leftover})"
    return body


def _format_payment_ref(item: dict[str, Any] | str, *, amount: Any = None) -> str:
    """Dense payment token: ``credit_card_2408938#2135`` or ``...:$296``."""
    if isinstance(item, str):
        bit = item
    else:
        pid = str(
            item.get("payment_id")
            or item.get("id")
            or item.get("payment_method_id")
            or ""
        ).strip()
        last4 = item.get("last_four")
        bit = pid or "?"
        if last4 is not None and str(last4).strip():
            bit += f"#{last4}"
        if amount is None:
            amount = item.get("amount")
    if amount is not None and str(amount).strip() != "":
        bit += f":${_fmt_price_val(amount)}"
    return bit


def _payment_history_paid(history: Any) -> float | None:
    """Sum of payment_history amounts (credits and debits) as currently listed."""
    if not isinstance(history, list) or not history:
        return None
    total = 0.0
    found = False
    for item in history:
        if not isinstance(item, dict) or item.get("amount") is None:
            continue
        try:
            total += float(item["amount"])
            found = True
        except (TypeError, ValueError):
            continue
    return total if found else None


def _flights_look_upcoming(flights: Any, *, ref_date: str) -> bool:
    """True if any segment date is on/after ``ref_date`` (YYYY-MM-DD)."""
    if not isinstance(flights, list):
        return False
    ref = str(ref_date or "").strip()[:10]
    if len(ref) < 10:
        return False
    for item in flights:
        if not isinstance(item, dict):
            continue
        date = str(item.get("date") or "").strip()[:10]
        if len(date) >= 10 and date >= ref:
            return True
    return False


def _format_payment_methods_map(val: dict[str, Any], *, max_items: int = 12) -> str:
    """Keep full payment ids + last_four so 'card ending in 2135' is matchable."""
    parts: list[str] = []
    for key in list(val.keys())[:max_items]:
        entry = val[key]
        if isinstance(entry, dict):
            payload = dict(entry)
            payload.setdefault("id", key)
            parts.append(_format_payment_ref(payload))
        else:
            parts.append(str(key))
    body = f"payment_methods=[{', '.join(parts)}]"
    if len(val) > max_items:
        body += f" (+{len(val) - max_items})"
    return body


def _format_payment_history(val: list[Any], *, max_items: int = 8) -> str:
    parts: list[str] = []
    for item in val[:max_items]:
        if isinstance(item, dict):
            parts.append(_format_payment_ref(item))
        elif item is not None:
            parts.append(str(item))
    if not parts:
        return "payment_history=[]"
    paid = _payment_history_paid(val)
    body = f"payment_history=[{'; '.join(parts)}]"
    if paid is not None:
        # Explicit total so models do not sum segment flight prices instead.
        body = f"paid=${_fmt_price_val(paid)}, {body}"
    if len(val) > max_items:
        body += f" (+{len(val) - max_items})"
    return body

def _format_passengers(val: list[Any], *, max_items: int = 4) -> str:
    """``passengers=[Daiki Muller/1954-07-04, ...]`` — enough for passenger writes."""
    parts: list[str] = []
    for item in val[:max_items]:
        if not isinstance(item, dict):
            continue
        first = str(item.get("first_name") or "").strip()
        last = str(item.get("last_name") or "").strip()
        dob = item.get("dob") or item.get("date_of_birth")
        name = f"{first} {last}".strip() or "?"
        bit = f"{name}/{dob}" if dob else name
        parts.append(bit)
    if not parts:
        return f"passengers_n={len(val)}"
    body = f"passengers=[{'; '.join(parts)}]"
    if len(val) > max_items:
        body += f" (+{len(val) - max_items})"
    return body


def _format_flight_segments(val: list[Any], *, max_items: int = 6) -> str:
    """Keep flight_number+date(+od)+$price so cost questions stay grounded."""
    parts: list[str] = []
    for item in val[:max_items]:
        if not isinstance(item, dict):
            continue
        fn = item.get("flight_number")
        date = item.get("date")
        origin = item.get("origin")
        dest = item.get("destination")
        bit = str(fn or "?")
        if date:
            bit += f"@{_short_date(date)}"
        if origin or dest:
            bit += f"({origin or '?'}>{dest or '?'})"
        price = item.get("price")
        if price is not None:
            bit += f"${_fmt_price_val(price)}"
        parts.append(bit)
    if not parts:
        return f"flights_n={len(val)}"
    body = f"flights=[{'; '.join(parts)}]"
    if len(val) > max_items:
        body += f" (+{len(val) - max_items})"
    return body


def _clip_at_boundary(text: str, max_chars: int) -> str:
    """Truncate without splitting mid-token (avoids mutilating payment ids / $amounts)."""
    body = str(text or "")
    if max_chars <= 0 or len(body) <= max_chars:
        return body
    if max_chars <= 3:
        return body[:max_chars]
    limit = max_chars - 3
    cut = body[:limit]
    for sep in (", ", "; ", " ", ",", ";"):
        idx = cut.rfind(sep)
        if idx >= max(24, limit // 2):
            cut = cut[:idx]
            break
    # Drop a trailing partial token so ``$1560`` never becomes ``$156``.
    cut = re.sub(r"[$€#]?[A-Za-z0-9_./@>\-:%]*$", "", cut).rstrip(",; ")
    if not cut:
        cut = body[:limit].rstrip(",; ")
    return cut + "..."

def _format_compact_list(
    key: str,
    val: list[Any],
    *,
    max_items: int = 12,
    max_chars: int | None = None,
) -> str:
    """Keep short id/scalar lists; avoid collapsing to ``key_n`` only."""
    if key == "flights":
        return _format_flight_segments(val, max_items=min(max_items, 6))
    if key == "payment_history":
        return _format_payment_history(val, max_items=min(max_items, 8))
    if key == "passengers":
        return _format_passengers(val, max_items=min(max_items, 4))
    if not val:
        return f"{key}=[]"
    if key == "list" and all(_looks_like_flight_option(x) for x in val):
        return _format_flight_search_list(
            val, max_items=min(max_items, 8), max_chars=max_chars
        )
    if all(isinstance(x, (str, int, float, bool)) or x is None for x in val):
        items = [str(x) for x in val if x is not None]
        shown = items[:max_items]
        body = f"{key}=[{', '.join(shown)}]"
        if len(items) > max_items:
            body += f" (+{len(items) - max_items})"
        return body
    if all(_looks_like_flight_option(x) for x in val):
        # Nested under a named key (rare); still keep prices.
        body = _format_flight_search_list(
            val, max_items=min(max_items, 8), max_chars=max_chars
        )
        return body.replace("list=", f"{key}=", 1)
    extracted: list[str] = []
    for item in val[:max_items]:
        if not isinstance(item, dict):
            continue
        for ik in _LIST_ID_KEYS:
            if item.get(ik) is not None:
                extracted.append(str(item[ik]))
                break
    if extracted:
        body = f"{key}=[{', '.join(extracted)}]"
        if len(val) > max_items:
            body += f" (+{len(val) - max_items})"
        return body
    return f"{key}_n={len(val)}"


def _format_compact_dict(key: str, val: dict[str, Any], *, max_items: int = 12) -> str:
    if key == "name" and ("first_name" in val or "last_name" in val):
        return (
            f"name={str(val.get('first_name') or '').strip()} "
            f"{str(val.get('last_name') or '').strip()}"
        ).strip()
    if key == "payment_methods" or key in _ID_MAP_KEYS or key.endswith("_methods"):
        if key == "payment_methods" or (
            val
            and any(
                isinstance(v, dict) and ("last_four" in v or "source" in v)
                for v in val.values()
            )
        ):
            return _format_payment_methods_map(val, max_items=max_items).replace(
                "payment_methods=", f"{key}=", 1
            )
        keys = [str(k) for k in list(val.keys())[:max_items]]
        body = f"{key}=[{', '.join(keys)}]"
        if len(val) > max_items:
            body += f" (+{len(val) - max_items})"
        return body
    return f"{key}={{…{len(val)}}}"


def semantic_summarize_json(
    content: str | None,
    *,
    tool_name: str | None = None,
    max_chars: int = _SEMANTIC_OBS_CHARS,
) -> str:
    """Compress tool/user payloads into short ALF-like observations.

    Dense and actionable (ids, last4, dates, prices), not lossless. Uses a
    fixed char budget and packs whole atomic fields — never mid-id clips.
    """
    raw = str(content if content is not None else "").strip()
    prefix = f"{tool_name}: " if tool_name else ""
    if not raw:
        return (prefix + "(empty)").strip()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        text = " ".join(raw.split())
        body = text if len(text) <= max_chars else text[: max_chars - 3] + "..."
        return (prefix + body).strip()

    if isinstance(data, dict):
        if data.get("error"):
            err = " ".join(str(data.get("error")).split())
            body = f"error={err}"
            if len(body) > max_chars:
                body = _clip_at_boundary(body, max_chars)
            return (prefix + body).strip()
        bits: list[str] = []
        seen: set[str] = set()

        def _append_bit(key: str, val: Any) -> None:
            if isinstance(val, list):
                bits.append(_format_compact_list(key, val))
            elif isinstance(val, dict):
                bits.append(_format_compact_dict(key, val))
            else:
                bits.append(f"{key}={val}")

        for key in _PRIORITY_KEYS:
            if key in _OMIT_KEYS:
                continue
            if key not in data or data[key] is None or key in seen:
                continue
            seen.add(key)
            _append_bit(key, data[key])
        for key, val in data.items():
            if key in seen or key in _DEFER_KEYS or key in _OMIT_KEYS:
                continue
            if isinstance(val, list):
                bits.append(_format_compact_list(key, val))
                seen.add(key)
            elif isinstance(val, dict):
                bits.append(_format_compact_dict(key, val))
                seen.add(key)
            elif isinstance(val, (str, int, float, bool)):
                # Unknown scalar fields can be decision-critical in new domains.
                bits.append(f"{key}={val}")
                seen.add(key)
            if len(bits) >= 10:
                break
        # Deferred blobs only if still under budget.
        body = ", ".join(bits) if bits else ""
        if len(body) < max_chars - 24:
            for key in _DEFER_KEYS:
                if key not in data or data[key] is None or key in seen:
                    continue
                if isinstance(data[key], dict):
                    trial = _format_compact_dict(key, data[key])
                elif isinstance(data[key], list):
                    trial = _format_compact_list(key, data[key])
                else:
                    trial = f"{key}={data[key]}"
                candidate = f"{body}, {trial}" if body else trial
                if len(candidate) <= max_chars:
                    bits.append(trial)
                    body = candidate
                    seen.add(key)
        body = ", ".join(bits) if bits else f"keys={list(data.keys())[:6]}"
        if len(body) > max_chars:
            body = _clip_at_boundary(body, max_chars)
        return (prefix + body).strip()

    if isinstance(data, list):
        # Flight search lists pack whole options into budget — do not re-clip
        # mid-option (that produced broken prices like ``eco=2``).
        body_budget = max(80, max_chars - len(prefix))
        body = _format_compact_list("list", data, max_chars=body_budget)
        if (
            not all(_looks_like_flight_option(x) for x in data)
            and len(body) > body_budget
        ):
            body = _clip_at_boundary(body, body_budget)
        return (prefix + body).strip()

    text = " ".join(str(data).split())
    body = text if len(text) <= max_chars else _clip_at_boundary(text, max_chars)
    return (prefix + body).strip()


def parse_think_and_visible(content: str | None) -> tuple[str | None, str]:
    """Split optional ``<think>`` block from user-visible text."""
    text = str(content or "")
    if not text.strip():
        return None, ""
    thinks = [m.group(1).strip() for m in _THINK_RE.finditer(text) if m.group(1).strip()]
    visible = _THINK_RE.sub("", text).strip()
    think = "\n\n".join(thinks) if thinks else None
    return think, visible


def think_from_message(msg: dict[str, Any]) -> tuple[str | None, str]:
    """Recover think from ``raw_data.sage_think`` or content tags.

    Returns ``(think, source)`` where source is ``model`` / ``missing``.
    Synthesized (``fallback``) thinks are ignored so they never enter formal
    trajectory dumps.
    """
    raw = msg.get("raw_data")
    if isinstance(raw, dict):
        stored = raw.get("sage_think")
        source = str(raw.get("sage_think_source") or "").strip()
        if source == "fallback":
            return None, "missing"
        if isinstance(stored, str) and stored.strip() and source != "fallback":
            return stored.strip(), (source or "model")
    think, _ = parse_think_and_visible(msg.get("content"))
    if think:
        return think, "model"
    return None, "missing"


def synthesize_think(
    *,
    action: str | None,
    observation_before: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> str:
    """Deterministic think fallback so distillation never sees null think."""
    obs = " ".join(str(observation_before or "").split())
    if len(obs) > 160:
        obs = obs[:157] + "..."
    if tool_calls:
        names = [str(t.get("name") or "?") for t in tool_calls]
        return (
            f"Given observation [{obs or 'none'}], call tool(s) "
            f"{', '.join(names)} to progress the request."
        )
    act = " ".join(str(action or "").split())
    if len(act) > 120:
        act = act[:117] + "..."
    return (
        f"Given observation [{obs or 'none'}], reply to the user: "
        f"{act or '(empty)'}."
    )


def tool_call_dict(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = fn.get("name") or tc.get("name") or "?"
    raw_args = fn.get("arguments", tc.get("arguments", {}))
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            pass
    if not isinstance(raw_args, dict):
        raw_args = {"_raw": raw_args}
    return {"name": name, "arguments": raw_args, "id": tc.get("id")}


def _tool_call_dict(tc: dict[str, Any]) -> dict[str, Any]:
    return tool_call_dict(tc)


def build_tool_name_index(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map tool_call id → function name from assistant tool_calls."""
    out: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            d = _tool_call_dict(tc)
            cid = str(d.get("id") or "").strip()
            name = str(d.get("name") or "").strip()
            if cid and name:
                out[cid] = name
    return out


def resolve_tool_name(
    msg: dict[str, Any],
    *,
    tool_names: dict[str, str] | None = None,
) -> str:
    name = msg.get("name")
    if isinstance(name, str) and name.strip() and name.strip().lower() != "tool":
        return name.strip()
    cid = str(msg.get("id") or msg.get("tool_call_id") or "").strip()
    if cid and tool_names and cid in tool_names:
        return tool_names[cid]
    return "tool"


def format_msg_block(
    msg: dict[str, Any],
    *,
    tool_names: dict[str, str] | None = None,
    semantic: bool = True,
    max_tool_chars: int | None = None,
) -> str:
    role = str(msg.get("role") or "?")
    content = msg.get("content")
    tool_calls = msg.get("tool_calls") or []
    parts: list[str] = []
    if tool_calls:
        for tc in tool_calls:
            d = _tool_call_dict(tc)
            parts.append(
                f"[tool_call] {d['name']}({json.dumps(d['arguments'], ensure_ascii=False)})"
            )
    text = str(content).strip() if content else ""
    if role == "tool":
        name = resolve_tool_name(msg, tool_names=tool_names)
        body = (
            semantic_summarize_json(content, tool_name=None)
            if semantic
            else str(content or "")
        )
        limit = (
            _SEMANTIC_OBS_CHARS
            if semantic
            else (max_tool_chars if max_tool_chars is not None else _FULL_TOOL_CHARS)
        )
        if limit > 0 and len(body) > limit:
            body = _clip_at_boundary(body, limit)
        return f"[tool_result:{name}] {body}"
    if role == "system":
        # Never semantically crush system prompts (AReaL / inject dumps need them).
        return f"[SYSTEM] {text}" if text else "[SYSTEM]"
    if text:
        if semantic and role == "user":
            compact = " ".join(text.split())
            if len(compact) > _SEMANTIC_OBS_CHARS:
                compact = _clip_at_boundary(compact, _SEMANTIC_OBS_CHARS)
            parts.append(compact)
        elif not semantic and role == "user":
            # Keep line breaks for readable dumps; only soft-cap extreme length.
            body = text
            limit = max_tool_chars if max_tool_chars is not None else _FULL_OBS_CHARS
            if limit > 0 and len(body) > limit:
                body = _clip_at_boundary(body, limit)
            parts.append(body)
        else:
            parts.append(text)
    label = {"user": "USER", "assistant": "ASSISTANT"}.get(role, role.upper())
    if not parts:
        return f"[{label}]"
    return f"[{label}] " + "\n".join(parts)


def action_from_assistant_message(msg: dict[str, Any]) -> str | None:
    """Single action string: tool call(s) or visible text reply."""
    tool_calls = [_tool_call_dict(tc) for tc in (msg.get("tool_calls") or [])]
    if tool_calls:
        return "; ".join(
            f"{t['name']}({json.dumps(t['arguments'], ensure_ascii=False)})"
            for t in tool_calls
        )
    _, visible = parse_think_and_visible(msg.get("content"))
    return visible or None


def agent_reply_content(
    msg: dict[str, Any],
    *,
    action: str | None,
    observation_before: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    allow_synthesize_think: bool = False,
) -> str:
    """ALF-style full reply string (``<think>`` + action body) for dump only.

    History windows keep using the clean ``action`` field; think is not inlined
    into ``action`` so GiGPO history stays short.
    """
    think, _source = think_from_message(msg)
    if not think and allow_synthesize_think:
        think = synthesize_think(
            action=action,
            observation_before=observation_before,
            tool_calls=tool_calls,
        )
    raw_content = str(msg.get("content") or "").strip()
    think_in_content, visible = parse_think_and_visible(raw_content)
    if think_in_content and raw_content:
        if visible or not (msg.get("tool_calls")):
            return raw_content
        body = (action or "").strip()
        if body:
            return f"<think>{think_in_content}</think>\n{body}"
        return raw_content
    body = (action or visible or "").strip()
    if think:
        return f"<think>{think}</think>\n{body}".strip() if body else f"<think>{think}</think>"
    return body


def observation_from_messages(
    obs_msgs: list[dict[str, Any]],
    *,
    tool_names: dict[str, str] | None = None,
    semantic: bool = True,
    max_tool_chars: int | None = None,
) -> str:
    if not obs_msgs:
        return ""
    return "\n\n".join(
        format_msg_block(
            m,
            tool_names=tool_names,
            semantic=semantic,
            max_tool_chars=max_tool_chars,
        )
        for m in obs_msgs
    )


def _clip_line(text: str | None, max_chars: int) -> str:
    body = " ".join(str(text or "").split()).strip()
    if not body:
        return "(empty)"
    if max_chars > 0 and len(body) > max_chars:
        return _clip_at_boundary(body, max_chars)
    return body


def build_booking_scratchpad(
    messages: list[dict[str, Any]],
    *,
    max_items: int = 8,
    max_chars: int = 720,
    env_date: str | None = None,
) -> str:
    """Sticky booking facts across the *full* episode (survives history window).

    Keeps first-seen cabin / paid separately from latest cabin/status so upgrades
    and cancels do not erase facts needed for later cabin/cost questions.
    ``upcoming_paid_sum`` sums first-seen ``paid`` for bookings that had a future
    flight relative to ``env_date`` (default τ² airline wall clock).
    """
    ref = (env_date or os.environ.get("SAGE_TAU2_ENV_DATE") or "2024-05-15").strip()[
        :10
    ]
    bookings: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        raw = msg.get("content")
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        rid = str(data.get("reservation_id") or "").strip()
        if not rid:
            continue
        if rid not in bookings:
            bookings[rid] = {}
            order.append(rid)
        entry = bookings[rid]
        cabin = data.get("cabin")
        if cabin is not None and str(cabin).strip():
            cabin_s = str(cabin).strip()
            if "first_cabin" not in entry:
                entry["first_cabin"] = cabin_s
            entry["cabin"] = cabin_s
        status = data.get("status")
        if status is not None and str(status).strip():
            entry["status"] = str(status).strip()
        paid = _payment_history_paid(data.get("payment_history"))
        if paid is not None and "paid" not in entry:
            # Sticky first-seen payment total (before upgrade/refund noise).
            entry["paid"] = _fmt_price_val(paid)
        if data.get("flights") is not None and "upcoming" not in entry:
            if _flights_look_upcoming(data.get("flights"), ref_date=ref):
                entry["upcoming"] = "1"
            else:
                entry["upcoming"] = "0"
    if not order:
        return "(none)"

    upcoming_paid = 0.0
    upcoming_paid_any = False
    for rid in order:
        entry = bookings[rid]
        if entry.get("upcoming") == "1" and entry.get("paid") is not None:
            try:
                upcoming_paid += float(entry["paid"])
                upcoming_paid_any = True
            except (TypeError, ValueError):
                pass
    sum_suffix = (
        f"; upcoming_paid_sum=${_fmt_price_val(upcoming_paid)}"
        if upcoming_paid_any
        else ""
    )
    # Reserve room for the sum suffix; pack whole booking units only.
    budget = max(80, max_chars - len(sum_suffix))
    parts: list[str] = []
    for rid in order[:max_items]:
        entry = bookings[rid]
        bits = [rid]
        first = entry.get("first_cabin")
        cabin = entry.get("cabin")
        if first:
            bits.append(f"first_cabin={first}")
        if cabin and cabin != first:
            bits.append(f"cabin={cabin}")
        elif cabin and not first:
            bits.append(f"cabin={cabin}")
        if entry.get("paid") is not None:
            bits.append(f"paid=${entry['paid']}")
        if entry.get("upcoming") == "1":
            bits.append("upcoming")
        if entry.get("status"):
            bits.append(f"status={entry['status']}")
        bit = " ".join(bits)
        trial = parts + [bit]
        body = "; ".join(trial)
        leftover = len(order) - len(trial)
        extra = f" (+{leftover})" if leftover > 0 else ""
        if len(body + extra) <= budget:
            parts.append(bit)
            continue
        if parts:
            break
        # Single booking alone exceeds budget: keep one complete unit.
        parts.append(bit)
        break
    if not parts:
        return sum_suffix.lstrip("; ") or "(none)"
    body = "; ".join(parts)
    leftover = len(order) - len(parts)
    if leftover > 0:
        body += f" (+{leftover})"
    body += sum_suffix
    return body

def format_available_tools(tools: Sequence[Any] | None) -> str:
    """Render tool list for TAU2_TEMPLATE (GiGPO admissible-actions analogue)."""
    if not tools:
        return _DEFAULT_TOOLS_PLACEHOLDER
    lines: list[str] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if not name and isinstance(tool, dict):
            name = tool.get("name")
        name = str(name or "?").strip() or "?"
        desc = (
            getattr(tool, "short_desc", None)
            or getattr(tool, "long_desc", None)
            or getattr(tool, "description", None)
        )
        if not desc and isinstance(tool, dict):
            desc = tool.get("short_desc") or tool.get("description") or ""
        desc = " ".join(str(desc or "").split())
        if len(desc) > 160:
            desc = desc[:157] + "..."
        lines.append(f"'{name}': {desc}" if desc else f"'{name}'")
    return "\n".join(lines) if lines else _DEFAULT_TOOLS_PLACEHOLDER


def build_history_summary(
    prior_steps: list[dict[str, Any]],
    *,
    max_steps: int = _DEFAULT_HISTORY_WINDOW,
) -> str:
    """Compact prior PRO window using causal (observation_before, action) pairs.

    Format mirrors GiGPO ``SimpleMemory.fetch``:
    ``[Observation N: '...', Action N: '...']``.
    """
    if not prior_steps:
        return "(none)"
    window = prior_steps[-max_steps:] if max_steps > 0 else prior_steps
    lines: list[str] = []
    for step in window:
        idx = step.get("step")
        obs = step.get("observation_before")
        if obs in (None, "", "(none)"):
            obs = "(none)"
        else:
            # Keep ids intact: payment_methods / reservations need ~300 chars.
            obs = _clip_line(str(obs), max(_SEMANTIC_OBS_CHARS, 300))

        act = _clip_line(str(step.get("action") or ""), _DEFAULT_ACT_CHARS)
        lines.append(f"[Observation {idx}: '{obs}', Action {idx}: '{act}']")
    omitted = len(prior_steps) - len(window)
    if omitted > 0:
        lines.insert(0, f"(omitted {omitted} earlier step(s))")
    return "\n".join(lines)


def clip_observation(text: str | None, *, max_chars: int = _DEFAULT_OBS_CHARS) -> str:
    body = str(text or "").strip()
    if not body:
        return "(none)"
    if max_chars > 0 and len(body) > max_chars:
        return _clip_at_boundary(body, max_chars)
    return body


def _booking_scratchpad_block(booking_scratchpad: str | None) -> str:
    scratch = (booking_scratchpad or "").strip()
    if not scratch or scratch == "(none)":
        return ""
    return f"\nKnown booking facts: {scratch}\n"


def compose_window_prompt(
    *,
    history_summary: str,
    observation_before: str,
    step_count: int,
    current_step: int,
    history_length: int,
    task_description: str | None = None,
    available_tools: str | None = None,
    booking_scratchpad: str | None = None,
) -> str:
    """GiGPO / ALFWorld-shell step window (+ optional sticky booking facts).

    ``available_tools`` ignored (tools via function calling). Task line uses
    ``agent_facing`` text when provided by the caller.
    """
    del available_tools
    from sage_tau2.task_context import clean_task_phrase

    task = clean_task_phrase(task_description or "") or (
        "Handle this customer-service episode end-to-end"
    )
    current_obs = clip_observation(observation_before)
    scratch_block = _booking_scratchpad_block(booking_scratchpad)
    if step_count <= 0:
        return TAU2_TEMPLATE_NO_HIS.format(
            task_description=task,
            current_observation=current_obs,
            booking_scratchpad_block=scratch_block,
        )
    return TAU2_TEMPLATE.format(
        task_description=task,
        step_count=step_count,
        history_length=history_length,
        action_history=history_summary or "(none)",
        current_step=current_step,
        current_observation=current_obs,
        booking_scratchpad_block=scratch_block,
    )


def build_live_window_prompt(
    messages: list[dict[str, Any]],
    *,
    history_window: int = _DEFAULT_HISTORY_WINDOW,
    task_description: str | None = None,
    available_tools: str | None = None,
    tools: Sequence[Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the window prompt for the *next* assistant turn (online inference).

    ``messages`` must already include the latest user/tool observation(s) and
    must not include the assistant turn about to be generated.
    """
    del available_tools, tools  # tools bound via function calling
    tool_names = build_tool_name_index(messages)
    completed = messages_to_rich_pro_steps(
        messages,
        history_window=history_window,
        annotate_signals=False,
        allow_synthesize_think=False,
        store_window_prompt=False,
    )
    # Observation conditioning the next action = msgs after last assistant.
    leading: list[dict[str, Any]] = []
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            break
        leading.append(m)
    leading.reverse()
    observation_before = observation_from_messages(leading, tool_names=tool_names)
    if not observation_before and completed:
        # Fallback: last step's after-observation (should match leading).
        observation_before = str(completed[-1].get("observation") or "")
    history_summary = build_history_summary(completed, max_steps=history_window)
    step_count = len(completed)
    history_length = min(step_count, history_window)
    current_step = step_count + 1
    booking_scratchpad = build_booking_scratchpad(messages)
    prompt = compose_window_prompt(
        history_summary=history_summary,
        observation_before=observation_before or "(none)",
        step_count=step_count,
        current_step=current_step,
        history_length=history_length,
        task_description=task_description,
        booking_scratchpad=booking_scratchpad,
    )
    meta = {
        "step_count": step_count,
        "current_step": current_step,
        "history_length": history_length,
        "history_summary": history_summary,
        "observation_before": clip_observation(observation_before),
        "booking_scratchpad": booking_scratchpad,
        "task_description": task_description or "",
    }
    from sage_tau2.contracts import observed_ledger
    ledger = observed_ledger(messages)
    if ledger:
        prompt += "\n\n" + ledger
    return prompt, meta


def _match_action_check(
    action: str | None,
    checks: list[dict[str, Any]],
    cursor: int,
) -> tuple[int | None, int]:
    """Return (check_index, next_cursor) when action matches a remaining check."""
    act = str(action or "")
    for i in range(cursor, len(checks)):
        item = checks[i] if isinstance(checks[i], dict) else {}
        expected = item.get("action") if isinstance(item.get("action"), dict) else {}
        name = str(expected.get("name") or "")
        if name and name in act:
            return i, i + 1
    return None, cursor


def annotate_step_signals(
    steps: list[dict[str, Any]],
    *,
    reward_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Attach ALF-like per-step reward / validity / goal_progress fields."""
    info = reward_info if isinstance(reward_info, dict) else {}
    episode_reward = float(info.get("reward") or 0.0)
    checks = [c for c in (info.get("action_checks") or []) if isinstance(c, dict)]
    n_checks = len(checks)
    matched = 0
    cursor = 0
    for i, step in enumerate(steps):
        progress_before = (matched / n_checks) if n_checks else (0.0 if i == 0 else float(i) / max(len(steps), 1))
        obs = str(step.get("observation") or "").lower()
        errored = "error=" in obs or "not found" in obs or "error=true" in obs
        check_idx, cursor = _match_action_check(step.get("action"), checks, cursor)
        check_match = None
        if check_idx is not None:
            check_match = bool(checks[check_idx].get("action_match"))
            if check_match:
                matched += 1
        if n_checks:
            progress_after = matched / n_checks
        else:
            progress_after = float(i + 1) / max(len(steps), 1)
            if i == len(steps) - 1 and episode_reward >= 1.0:
                progress_after = 1.0
        step["is_action_valid"] = (not errored) if check_match is None else bool(check_match) and not errored
        step["reward"] = 0.0
        step["goal_progress_before"] = round(progress_before, 4)
        step["goal_progress_after"] = round(progress_after, 4)
        step["goal_progress_delta"] = round(progress_after - progress_before, 4)
        if check_idx is not None:
            step["action_check_match"] = check_match
    if steps:
        steps[-1]["reward"] = episode_reward
        if episode_reward >= 1.0:
            steps[-1]["goal_progress_after"] = 1.0
            steps[-1]["goal_progress_delta"] = round(
                1.0 - float(steps[-1].get("goal_progress_before") or 0.0), 4
            )
    return steps


def messages_to_rich_pro_steps(
    messages: list[dict[str, Any]],
    *,
    history_window: int = _DEFAULT_HISTORY_WINDOW,
    max_obs_chars: int | None = None,
    annotate_signals: bool = True,
    reward_info: dict[str, Any] | None = None,
    allow_synthesize_think: bool = False,
    task_description: str | None = None,
    available_tools: str | None = None,
    store_window_prompt: bool = False,
    obs_mode: str = "lean",
) -> list[dict[str, Any]]:
    """One PRO step per assistant turn.

    Default dump fields: ``observation_before`` / ``action`` / ``observation``
    (+ ``agent_messages`` reply, tool_calls, signals). Full window prompt is
    **not** stored unless ``store_window_prompt=True``.

    ``obs_mode``:
      - ``lean``: GiGPO-style semantic compression (~520 chars) for training windows
      - ``full``: near-raw user/tool text (soft-capped) for inspection / richer dumps

    Model ``<think>`` is kept only inside ``agent_messages[].content`` (ALF-style),
    not as separate ``think`` / ``think_source`` fields. ``allow_synthesize_think``
    may embed a fallback think into that reply for debug dumps only.
    """
    del task_description, available_tools
    mode = str(obs_mode or "lean").strip().lower()
    semantic = mode != "full"
    if max_obs_chars is None:
        max_obs_chars = _DEFAULT_OBS_CHARS if semantic else _FULL_OBS_CHARS
    tool_chars = _SEMANTIC_OBS_CHARS if semantic else _FULL_TOOL_CHARS
    tool_names = build_tool_name_index(messages)
    steps: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        prefix = [m for m in messages[:i] if isinstance(m, dict)]
        tool_calls = [_tool_call_dict(tc) for tc in (msg.get("tool_calls") or [])]
        action = action_from_assistant_message(msg)

        j = i + 1
        obs_msgs: list[dict[str, Any]] = []
        while j < n and isinstance(messages[j], dict) and messages[j].get("role") != "assistant":
            obs_msgs.append(messages[j])
            j += 1
        observation = observation_from_messages(
            obs_msgs,
            tool_names=tool_names,
            semantic=semantic,
            max_tool_chars=tool_chars,
        )

        if steps:
            observation_before = str(steps[-1].get("observation") or "")
        else:
            leading: list[dict[str, Any]] = []
            for m in reversed(prefix):
                if m.get("role") == "assistant":
                    break
                leading.append(m)
            leading.reverse()
            observation_before = observation_from_messages(
                leading,
                tool_names=tool_names,
                semantic=semantic,
                max_tool_chars=tool_chars,
            )

        reply = agent_reply_content(
            msg,
            action=action,
            observation_before=observation_before,
            tool_calls=tool_calls or None,
            allow_synthesize_think=allow_synthesize_think,
        )

        history_summary = build_history_summary(steps, max_steps=history_window)
        step_count = len(steps)
        history_length = min(step_count, history_window)
        current_step = step_count + 1

        step: dict[str, Any] = {
            "step": current_step,
            "action": action,
            "tool_calls": tool_calls,
            "observation": clip_observation(observation, max_chars=max_obs_chars * 2),
            "observation_before": clip_observation(
                observation_before, max_chars=max_obs_chars
            ),
            "history_length": history_length,
            "step_count": step_count,
            "obs_mode": mode,
        }
        from sage_tau2.contracts import interaction_sequence
        step["interaction_sequence"] = interaction_sequence([msg, *obs_msgs])
        step["tool_results"] = [dict(m) for m in obs_msgs if m.get("role") == "tool"]
        raw_data = msg.get("raw_data")
        raw_event = raw_data.get("sage_skill_event") if isinstance(raw_data, dict) else None
        if isinstance(raw_event, dict):
            step["skill_event"] = raw_event
        if reply:
            step["agent_messages"] = [{"content": reply}]
        if store_window_prompt:
            raw = msg.get("raw_data") if isinstance(msg.get("raw_data"), dict) else {}
            live_prompt = raw.get("sage_window_prompt")
            if isinstance(live_prompt, str) and live_prompt.strip():
                prompt = live_prompt.strip()
                prompt_source = "live"
            else:
                prompt = compose_window_prompt(
                    history_summary=history_summary,
                    observation_before=observation_before or "(none)",
                    step_count=step_count,
                    current_step=current_step,
                    history_length=history_length,
                    booking_scratchpad=build_booking_scratchpad(messages[:j]),
                )
                prompt_source = "reconstructed"
            step["prompt"] = prompt
            step["prompt_source"] = prompt_source
            step["history_summary"] = history_summary
        steps.append(step)
        i = j if j > i else i + 1

    if annotate_signals:
        annotate_step_signals(steps, reward_info=reward_info)
    return steps


def lean_window_from_prior_steps(
    prior_steps: list[dict[str, Any]],
    *,
    observation_before: str,
    history_window: int = _DEFAULT_HISTORY_WINDOW,
    booking_scratchpad: str | None = None,
    task_description: str | None = None,
    skills_block: str | None = None,
) -> str:
    """Rebuild GiGPO-style online window from dumped prior steps + current obs."""
    from sage_tau2.injection import append_skills_to_user_prompt

    history_summary = build_history_summary(prior_steps, max_steps=history_window)
    step_count = len(prior_steps)
    history_length = min(step_count, history_window)
    prompt = compose_window_prompt(
        history_summary=history_summary,
        observation_before=observation_before or "(none)",
        step_count=step_count,
        current_step=step_count + 1,
        history_length=history_length,
        task_description=task_description,
        booking_scratchpad=booking_scratchpad,
    )
    return append_skills_to_user_prompt(prompt, skills_block or "")


def format_rich_trajectory_text(traj: dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        f"Task {traj.get('task_id')}  won={traj.get('won')}  "
        f"reward={traj.get('reward')}  steps={traj.get('num_steps')}",
        f"task={traj.get('task')}",
        f"db_match={traj.get('db_match')}  termination={traj.get('termination_reason')}",
        f"format={traj.get('format')}  primary={traj.get('assigned_primary_agent')}",
        "-" * 72,
    ]
    inject = traj.get("runtime_inject") or {}
    if inject:
        lines.append("[RUNTIME_INJECT]")
        lines.append(
            f"primary={inject.get('primary_agent')} layer={inject.get('dispatch_layer')} "
            f"skills={inject.get('injected_skill_names') or inject.get('injected_skill_ids')} "
            f"bank_domain={inject.get('bank_domain_active_count')}"
        )
        org = inject.get("organization_block") or ""
        if org:
            preview = org if len(org) <= 800 else org[:800] + "\n...[truncated]"
            lines.append(preview)
        skills = inject.get("skills_block") or ""
        if skills:
            preview = skills if len(skills) <= 800 else skills[:800] + "\n...[truncated]"
            lines.append(preview)
        lines.append(
            f"system_prompt_chars={inject.get('system_prompt_chars')} "
            f"sha1={inject.get('system_prompt_sha1')}"
        )
        lines.append("-" * 72)
    sys_body = traj.get("system_prompt")
    if isinstance(sys_body, str) and sys_body.strip():
        lines.append("[SYSTEM_PROMPT once-per-trajectory]")
        preview = sys_body if len(sys_body) <= 1200 else sys_body[:1200] + "\n...[truncated]"
        lines.append(preview)
        lines.append(f"system_prompt_sha1={traj.get('system_prompt_sha1')}")
        lines.append("-" * 72)
    for step in traj.get("steps") or []:
        lines.append(f"### step {step.get('step')}")
        lines.append(
            f"step_count_before={step.get('step_count')} "
            f"history_length={step.get('history_length')} "
            f"reward={step.get('reward')} valid={step.get('is_action_valid')} "
            f"progress={step.get('goal_progress_before')}→{step.get('goal_progress_after')} "
            f"system_sha1={step.get('system_prompt_sha1')}"
        )
        if step.get("prompt"):
            lines.append("[WINDOW_PROMPT hist+current_obs; system not inlined]")
            wp = str(step.get("prompt"))
            lines.append(wp if len(wp) <= 2000 else wp[:2000] + "\n...[truncated]")
        lines.append("[OBSERVATION_BEFORE]")
        lines.append(str(step.get("observation_before") or "(none)"))
        msgs = step.get("agent_messages") or []
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("content"):
            lines.append("[REPLY]")
            reply = str(msgs[0].get("content"))
            lines.append(reply if len(reply) <= 2000 else reply[:2000] + "\n...[truncated]")
        lines.append("[ACTION]")
        lines.append(str(step.get("action") or "(empty)"))
        lines.append("[OBSERVATION]")
        lines.append(str(step.get("observation") or "(none)"))
        lines.append("")
    return "\n".join(lines)


def load_runtime_inject_index(
    path: str | Path | None,
) -> dict[str, dict[str, Any]]:
    """task_id → last dispatch/runtime-inject row from journal."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        tid = str(row.get("task_id") or "").strip()
        if tid:
            out[tid] = row
    return out


def find_dispatch_journal(
    *,
    output_dir: str | Path | None = None,
    explicit: str | Path | None = None,
) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    if not output_dir:
        return None
    cur = Path(output_dir).expanduser().resolve()
    for _ in range(6):
        candidate = cur / "dispatch_journal.jsonl"
        if candidate.exists():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _sha1_text(text: str | None) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def compact_runtime_inject(inject: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky duplicated prompts; keep hashes + inject blocks."""
    out = dict(inject)
    system_prompt = out.pop("system_prompt", None)
    domain_policy = out.pop("domain_policy", None)
    out["system_prompt_chars"] = len(system_prompt or "")
    out["domain_policy_chars"] = len(domain_policy or "")
    out["system_prompt_sha1"] = _sha1_text(system_prompt)
    out["domain_policy_sha1"] = _sha1_text(domain_policy)
    # Keep blocks but clip extreme size.
    for key in ("skills_block", "organization_block", "role_block"):
        val = out.get(key)
        if isinstance(val, str) and len(val) > 4000:
            out[key] = val[:3970] + "\n...[truncated]"
    return out


def runtime_inject_from_sources(
    sim: dict[str, Any],
    *,
    dispatch_row: dict[str, Any] | None = None,
    results_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble runtime inject snapshot for one episode."""
    row = dict(dispatch_row or {})
    policy = sim.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        env = (results_info or {}).get("environment_info") if results_info else None
        if isinstance(env, dict) and isinstance(env.get("policy"), str):
            policy = env["policy"]
    system_prompt = row.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        system_prompt = None
    skills_block = row.get("skills_block")
    if not skills_block:
        skills_block = ""
    org_block = row.get("organization_block")
    if not org_block:
        org_block = (
            "<organization>\n"
            "Executor: Executor (default generalist)\n"
            "Specialists: (none recorded)\n"
            "</organization>"
        )
    inject = {
        "primary_agent": row.get("primary") or row.get("primary_agent"),
        "dispatch_layer": row.get("layer") or row.get("dispatch_layer"),
        "domain": row.get("domain") or sim.get("domain"),
        "system_prompt": system_prompt,
        "domain_policy": policy if isinstance(policy, str) else None,
        "role_block": row.get("role_block") or "",
        "organization_block": org_block,
        "skills_block": skills_block,
        "injected_skill_ids": list(row.get("injected_skill_ids") or []),
        "injected_skill_names": list(row.get("injected_skill_names") or []),
        "bank_active_count": row.get("bank_active_count"),
        "bank_domain_active_count": row.get("bank_domain_active_count"),
        "bank_domain_skill_names": list(row.get("bank_domain_skill_names") or []),
        "task_text": row.get("task_text"),
        "assignment": row.get("assignment"),
        "source": "dispatch_journal" if row else "results_policy_only",
    }
    return inject


def task_text_from_sources(
    sim: dict[str, Any],
    *,
    task: dict[str, Any] | None = None,
    dispatch_row: dict[str, Any] | None = None,
) -> str:
    from sage_tau2.task_context import (
        agent_facing_task_text,
        clean_task_phrase,
    )

    # Prefer structured task → customer scenario (never "Purpose: ..." labels).
    if task:
        facing = agent_facing_task_text(task)
        if facing:
            return facing[:500]

    row = dispatch_row or {}
    text = row.get("task_text")
    if isinstance(text, str) and text.strip():
        return clean_task_phrase(text)[:500]

    for m in sim.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            return clean_task_phrase(str(m.get("content")))[:500]
    return ""
