"""LLM-proposed skill protocols with programmatic grounding verification.

The heuristic distiller (:mod:`sage_mas.skill_protocol_distill`) merges
trajectories through a hand-shaped verb grammar and stage ontology. This
module offers an alternative *proposer*: an LLM reads the raw winning
trajectories and directly writes the minimal causal protocol, a natural
-language search hint, anti-patterns, and an object-disambiguation note —
no verb tables or stage rules are given to it, so it copes with whatever
actions the environment actually produced.

Division of labor:

- **LLM (comprehension).** Turns noisy trajectories into a clean recipe.
  The prompt demands that every protocol step be grounded in the evidence.
- **Code (verification).** Every proposed step must match an action that
  literally appears in at least one winning trajectory (instance ids
  stripped, ``<slot>`` placeholders absorbing entity names). Proposals
  with unverifiable steps are rejected and the caller falls back to the
  heuristic merge.
- **Evolution (arbitration).** Accepted protocols enter the bank as
  provisional skills through the unchanged credit path — utility is still
  earned by use, never assumed at write time.

The proposer is optional and side-effect free: on any backend error,
parse failure, or grounding rejection it returns ``None``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .trajectory.abstraction import trajectory_action_steps

logger = logging.getLogger(__name__)

_SLOT_PATTERN = re.compile(r"<[a-z_]+>")
_INSTANCE_ID_SUFFIX = re.compile(r"\s+\d+$")
_WHITESPACE = re.compile(r"\s+")
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

MAX_PROTOCOL_STEPS = 16


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a skill distiller for an embodied household agent. You receive
several WINNING trajectories for the same task family (action logs of an
agent that succeeded), and sometimes short digests of FAILED attempts.

Your job: extract the minimal causal recipe that explains why the wins
succeeded, as a short ordered list of environment actions.

Rules:
1. Output ONLY a JSON object. No prose, no markdown fences.
2. protocol: the minimal causal recipe — only navigation, access,
   acquisition, transform, and placement steps that are strictly
   necessary. Drop search wandering, wrong-location visits, redundant
   open/close cycles, and failed attempts.
3. GROUNDING: every protocol step MUST correspond to an action that
   literally appears in at least one winning trajectory. Object instance
   numbers may differ (they are scene-specific), but verbs and entity
   types must come from the evidence. Never invent actions. A program
   will verify each step against the trajectories and discard
   ungrounded output.
4. Use slots where the evidence generalizes: <object> for the target
   object, <source> for the receptacle it is taken from, <destination>
   for the receptacle it is placed on/in. Keep appliance/tool names
   concrete exactly as observed (e.g. "microwave", "sinkbasin") — a
   transform depends on the specific tool.
5. Keep each step's surface form identical to the environment actions
   (same verb phrases), replacing only entity names with slots.
6. If the wins disagree about where the object is found, do NOT hard-code
   one guess into the protocol. Leave "<source>" in the take step and put
   the observed search order into search_hint.
7. search_hint: one short sentence stating where the target was usually
   found and in what order to search, or "" if the evidence is unanimous
   or uninformative.
8. anti_patterns: 0-3 short imperative sentences describing mistakes that
   are visible in the detours or failure digests and that the protocol
   avoids (e.g. putting the object into the appliance instead of
   transforming it while holding it).
9. object_note: one short sentence clarifying which object counts as the
   target when similar-looking objects exist, or "" if unambiguous.
10. rationale: 1-2 sentences on what makes this recipe causal and minimal.

Output schema (all keys required):
{
  "protocol": ["go to <source>", "take <object> from <source>", "..."],
  "search_hint": "",
  "anti_patterns": [],
  "object_note": "",
  "rationale": ""
}

Example (for a heat task):

Input gist: 3 wins, each finds a mug or plate, heats it in a microwave,
then places it on a countertop or shelf. Wins include long opening
searches; one failure digest shows the agent putting a potato into the
microwave instead of holding it.

Correct output:
{
  "protocol": [
    "go to <source>",
    "take <object> from <source>",
    "go to microwave",
    "heat <object> with microwave",
    "go to <destination>",
    "move <object> to <destination>"
  ],
  "search_hint": "The target was usually found on countertops or shelves; check flat surfaces first, then appliances.",
  "anti_patterns": [
    "Do not put the object into the microwave; hold it and use the heat action."
  ],
  "object_note": "",
  "rationale": "Acquisition precedes the transform, the transform uses the concrete appliance, and placement ends the task; all search detours are omitted."
}\
"""

_USER_TEMPLATE = """\
Task family: {family}
Skill capability: {capability}

Below are {n_wins} winning trajectories for this task family. Each line is
one environment action the agent issued, in order.

{wins_block}
{failures_block}
Distill the skill now. Output ONLY the JSON object described in the
instructions.\
"""


def _format_trajectory(steps: list[Any], *, max_steps: int) -> str:
    actions = [str(getattr(step, "action", "") or "") for step in trajectory_action_steps(steps)]
    if len(actions) > max_steps:
        head = actions[:8]
        tail = actions[-(max_steps - 8) :]
        omitted = len(actions) - len(head) - len(tail)
        lines = [f" {i}. {a}" for i, a in enumerate(head, 1)]
        lines.append(f" ... ({omitted} middle steps omitted) ...")
        offset = len(head) + omitted
        lines.extend(f" {offset + i}. {a}" for i, a in enumerate(tail, 1))
        return "\n".join(lines)
    return "\n".join(f" {i}. {a}" for i, a in enumerate(actions, 1))


def _format_wins(wins: list[list[Any]], *, max_evidence: int, max_steps: int) -> str:
    blocks = []
    for idx, steps in enumerate(wins[:max_evidence], 1):
        body = _format_trajectory(steps, max_steps=max_steps)
        blocks.append(f"[WIN {idx}]\n{body}")
    return "\n\n".join(blocks)


def _format_failures(failures: list[list[Any]] | None, *, max_failures: int, tail: int = 12) -> str:
    if not failures:
        return ""
    blocks = []
    for idx, steps in enumerate(failures[:max_failures], 1):
        actions = [str(getattr(step, "action", "") or "") for step in trajectory_action_steps(steps)]
        shown = actions[-tail:]
        body = "\n".join(f" - {a}" for a in shown)
        blocks.append(f"[FAILURE DIGEST {idx}] (did not succeed; last {len(shown)} actions)\n{body}")
    return "\n\nFailed attempts for contrast (learn what to avoid, not what to copy):\n" + "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Proposal parsing
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LLMProtocolProposal:
    """Validated LLM distillation output."""

    protocol: list[str]
    search_hint: str = ""
    anti_patterns: list[str] = field(default_factory=list)
    object_note: str = ""
    rationale: str = ""
    grounding: dict[str, Any] = field(default_factory=dict)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    fenced = _FENCED_JSON.search(text)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_step(text: Any) -> str:
    normalized = _WHITESPACE.sub(" ", str(text or "")).strip()
    # Drop leading enumeration the model may add ("1. go to ...").
    normalized = re.sub(r"^\d+\s*[.)、]\s*", "", normalized)
    return normalized


def parse_proposal(text: str) -> LLMProtocolProposal | None:
    """Parse the LLM response into a proposal; ``None`` on contract violation."""
    payload = _extract_json_object(text)
    if payload is None:
        return None
    raw_protocol = payload.get("protocol")
    if not isinstance(raw_protocol, list):
        return None
    protocol = []
    for item in raw_protocol:
        step = _clean_step(item)
        if not step or len(step) > 80:
            return None
        # Slots must be well-formed; no other angle-bracket content allowed.
        stripped = _SLOT_PATTERN.sub("", step)
        if "<" in stripped or ">" in stripped:
            return None
        protocol.append(step)
    if not 1 <= len(protocol) <= MAX_PROTOCOL_STEPS:
        return None
    anti_patterns = payload.get("anti_patterns") or []
    if not isinstance(anti_patterns, list):
        return None
    anti_patterns = [
        _WHITESPACE.sub(" ", str(item)).strip() for item in anti_patterns if str(item or "").strip()
    ][:3]
    search_hint = _WHITESPACE.sub(" ", str(payload.get("search_hint") or "")).strip()
    object_note = _WHITESPACE.sub(" ", str(payload.get("object_note") or "")).strip()
    rationale = _WHITESPACE.sub(" ", str(payload.get("rationale") or "")).strip()
    return LLMProtocolProposal(
        protocol=protocol,
        search_hint=search_hint,
        anti_patterns=anti_patterns,
        object_note=object_note,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Grounding verification (grammar-free: pure evidence matching)
# ---------------------------------------------------------------------------


def _normalize_evidence(text: str) -> str:
    normalized = _WHITESPACE.sub(" ", str(text or "").strip().lower())
    # Strip trailing instance ids segment by segment: "take mug 1 from shelf 2".
    normalized = re.sub(r"\s+\d+(?=\s|$)", "", normalized)
    return normalized


def _normalize_proposed(step: str) -> str:
    return _WHITESPACE.sub(" ", str(step or "").strip().lower())


def _compiled_step_pattern(step: str) -> re.Pattern[str]:
    """Slot-aware regex for one proposed step against normalized evidence."""
    normalized = _normalize_proposed(step)
    literal_parts = [re.escape(part) for part in _SLOT_PATTERN.split(normalized)]
    body = ".+?".join(literal_parts)
    return re.compile("^" + body + "$")


def _evidence_action_lists(wins: list[list[Any]]) -> list[list[str]]:
    lists: list[list[str]] = []
    for steps in wins:
        actions = [
            _normalize_evidence(getattr(step, "action", "") or "")
            for step in trajectory_action_steps(steps)
        ]
        actions = [a for a in actions if a]
        if actions:
            lists.append(actions)
    return lists


def verify_proposal_grounding(
    proposal: LLMProtocolProposal,
    wins: list[list[Any]],
) -> dict[str, Any]:
    """Check every proposed step against the evidence trajectories.

    Returns a grounding report. ``ok`` is True only when *every* step matches
    at least one observed action (instance ids stripped, slots absorbing
    entity names). ``subsequence_support`` counts trajectories containing the
    whole protocol as an ordered subsequence — a stronger, advisory signal
    that is recorded but not required.
    """
    evidence_lists = _evidence_action_lists(wins)
    per_step: list[dict[str, Any]] = []
    ungrounded: list[str] = []
    for step in proposal.protocol:
        pattern = _compiled_step_pattern(step)
        support = sum(1 for actions in evidence_lists if any(pattern.match(a) for a in actions))
        per_step.append({"step": step, "support": support})
        if support == 0:
            ungrounded.append(step)
    subsequence_support = 0
    for actions in evidence_lists:
        pos = 0
        for step in proposal.protocol:
            pattern = _compiled_step_pattern(step)
            found = False
            while pos < len(actions):
                if pattern.match(actions[pos]):
                    pos += 1
                    found = True
                    break
                pos += 1
            if not found:
                break
        else:
            subsequence_support += 1
    return {
        "ok": not ungrounded and bool(evidence_lists),
        "ungrounded_steps": ungrounded,
        "per_step_support": per_step,
        "subsequence_support": subsequence_support,
        "evidence_trajectories": len(evidence_lists),
    }


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------


class _ChatBackend(Protocol):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int | None = None,
    ) -> Any:
        ...


class LLMProtocolProposer:
    """Ask an LLM to distill one skill protocol from winning trajectories.

    Never raises: backend errors, contract violations, and grounding
    rejections all yield ``None`` so the caller can fall back to the
    heuristic merge.
    """

    def __init__(
        self,
        backend: _ChatBackend,
        *,
        max_evidence: int = 6,
        max_steps_per_trajectory: int = 80,
        max_failures: int = 2,
        require_grounding: bool = True,
        # Thinking models (Gemini 2.5 Pro) burn most of the completion budget
        # on reasoning tokens; 1200 truncates the JSON mid-stream.
        max_completion_tokens: int = 8000,
    ) -> None:
        self.backend = backend
        self.max_evidence = max(1, int(max_evidence))
        self.max_steps_per_trajectory = max(16, int(max_steps_per_trajectory))
        self.max_failures = max(0, int(max_failures))
        self.require_grounding = bool(require_grounding)
        self.max_completion_tokens = int(max_completion_tokens)
        self.last_call_stats: dict[str, Any] = {}

    def propose(
        self,
        *,
        capability: str,
        family: str,
        wins: list[list[Any]],
        failures: list[list[Any]] | None = None,
    ) -> LLMProtocolProposal | None:
        if not wins:
            return None
        user_prompt = _USER_TEMPLATE.format(
            family=family or capability,
            capability=capability,
            n_wins=min(len(wins), self.max_evidence),
            wins_block=_format_wins(
                wins,
                max_evidence=self.max_evidence,
                max_steps=self.max_steps_per_trajectory,
            ),
            failures_block=_format_failures(failures, max_failures=self.max_failures),
        )
        try:
            result = self.backend.complete(
                SYSTEM_PROMPT,
                user_prompt,
                max_completion_tokens=self.max_completion_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - proposer must never raise
            logger.warning("LLM protocol proposal failed for %s: %s", capability, exc)
            self.last_call_stats = {"error": str(exc)}
            return None
        self.last_call_stats = {
            "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(result, "completion_tokens", 0) or 0),
        }
        raw_content = getattr(result, "content", "") or ""
        proposal = parse_proposal(raw_content)
        if proposal is None:
            logger.warning("LLM protocol proposal for %s violated the JSON contract", capability)
            self.last_call_stats["rejected"] = "parse"
            # Keep the raw response for post-mortem diagnosis (truncated).
            self.last_call_stats["raw_response"] = raw_content[:4000]
            return None
        grounding = verify_proposal_grounding(proposal, wins)
        proposal.grounding = grounding
        if self.require_grounding and not grounding["ok"]:
            logger.warning(
                "LLM protocol proposal for %s has ungrounded steps: %s",
                capability,
                "; ".join(grounding["ungrounded_steps"]),
            )
            self.last_call_stats["rejected"] = "grounding"
            return None
        return proposal
