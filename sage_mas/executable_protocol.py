"""Trajectory-derived executable skill protocols with soft step alignment.

Content comes only from skill.action_protocol / successful trajectory actions.
No ALFWorld family hard rules or action rewriting.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from sage_mas.schemas import Skill

_ADMISSIBLE_RE = re.compile(
    r"admissible actions[^:]*:\s*\[(.*?)\]",
    re.IGNORECASE | re.DOTALL,
)
_WEAK_VERBS = frozenset({"go", "look", "examine", "inventory", "open", "close"})
_METADATA_KEY = "executable_protocol"
_INSTANCE_RE = re.compile(r"\b[a-z][a-z0-9]*\s+\d+\b", re.IGNORECASE)
_PLACEHOLDER_TOKENS = frozenset(
    {
        "object",
        "tool",
        "receptacle",
        "location",
        "place",
        "target",
        "source",
        "destination",
        "entity",
    }
)


@dataclass(slots=True)
class ExecutableStep:
    index: int
    action_template: str
    expected_obs_hint: str = ""
    verb: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExecutableStep":
        return cls(
            index=int(raw.get("index", 0)),
            action_template=str(raw.get("action_template", "") or ""),
            expected_obs_hint=str(raw.get("expected_obs_hint", "") or ""),
            verb=str(raw.get("verb", "") or ""),
        )


def ensure_executable_protocol(
    skill: Skill,
    *,
    trajectory_steps: list[dict[str, Any]] | None = None,
    force: bool = False,
    max_steps: int = 16,
) -> list[ExecutableStep]:
    """Compile the distilled protocol; trajectory records only annotate evidence.

    The fingerprint invalidates legacy detour caches and edited protocols.
    Never truncate the canonical action sequence to a trajectory-prefix budget.
    """
    import hashlib
    import json

    existing = get_executable_steps(skill)
    protocol = list(skill.action_protocol or [])
    signature = hashlib.sha256(json.dumps(protocol, ensure_ascii=False).encode()).hexdigest()
    canonical = steps_from_action_protocol(protocol)
    if canonical:
        if (not force and not trajectory_steps
                and skill.metadata.get("executable_protocol_version") == 2
                and skill.metadata.get("executable_protocol_fingerprint") == signature
                and [s.action_template for s in existing] == [s.action_template for s in canonical]):
            return existing
        records = trajectory_steps or skill.metadata.get("confirmed_transitions") or []
        # Reuse hints only from a previously compiled, identical canonical list.
        if (not records and skill.metadata.get("executable_protocol_version") == 2
                and skill.metadata.get("executable_protocol_fingerprint") == signature
                and [s.action_template for s in existing] == [s.action_template for s in canonical]):
            records = [{"action": s.action_template, "observation": s.expected_obs_hint} for s in existing]
        cursor = 0
        matched = 0
        for step in canonical:
            for index in range(cursor, len(records)):
                record = records[index]
                if not isinstance(record, dict):
                    continue
                observation = str(record.get("observation") or "")
                if "nothing happens" in observation.lower() or record.get("is_action_valid") is False:
                    continue
                if action_matches_template(str(record.get("action") or ""), step.action_template):
                    step.expected_obs_hint = " ".join(observation.split())[:160]
                    matched += bool(step.expected_obs_hint)
                    cursor = index + 1
                    break
        steps = canonical
        source = "trajectory" if trajectory_steps and matched else (
            "confirmed_transitions" if matched else "action_protocol")
        skill.metadata["executable_evidence_matched_steps"] = matched
    else:
        # Compatibility for old evidence-only skills. No second protocol exists
        # to disagree with here; once action_protocol is populated it wins.
        if existing and not force and not trajectory_steps and not protocol_has_instance_literals(existing):
            return existing
        records = trajectory_steps or skill.metadata.get("confirmed_transitions") or []
        steps = steps_from_trajectory(records, max_steps=max(1, int(max_steps)))
        source = "trajectory" if trajectory_steps else "confirmed_transitions"
    skill.metadata[_METADATA_KEY] = [step.to_dict() for step in steps]
    skill.metadata["executable_protocol_source"] = source
    skill.metadata["executable_protocol_role_slotted"] = True
    skill.metadata["executable_protocol_version"] = 2
    skill.metadata["executable_protocol_fingerprint"] = signature
    return steps


def get_executable_steps(skill: Skill) -> list[ExecutableStep]:
    raw = skill.metadata.get(_METADATA_KEY) or []
    if not isinstance(raw, list):
        return []
    steps: list[ExecutableStep] = []
    for item in raw:
        if isinstance(item, dict) and str(item.get("action_template", "")).strip():
            steps.append(ExecutableStep.from_dict(item))
    return steps


def protocol_has_instance_literals(steps: list[ExecutableStep]) -> bool:
    return any(_INSTANCE_RE.search(step.action_template or "") for step in steps)


def steps_from_action_protocol(protocol: list[str]) -> list[ExecutableStep]:
    steps: list[ExecutableStep] = []
    for instruction in protocol:
        text = _environment_action(" ".join(str(instruction or "").split()))
        if not text:
            continue
        text = re.sub(r"\s+\d+(?=\s|$)", "", text)
        if "<" in text:
            text = role_slot_template(text)
        else:
            text = re.sub(r"^(take) .+?( from .+)$", r"\1 <target>\2", text)
            text = re.sub(r"^(clean|heat|cool) .+?( with .+)$", r"\1 <target>\2", text)
            text = re.sub(r"^(?:put|move) .+? (?:in|on|to) (.+)$", r"move <target> to \1", text)
        # Repeated steps can encode distinct-object or repeated-operation tasks.
        steps.append(ExecutableStep(index=len(steps), action_template=text, verb=_first_verb(text)))
    return steps


def steps_from_trajectory(
    trajectory_steps: list[dict[str, Any]],
    *,
    max_steps: int = 16,
) -> list[ExecutableStep]:
    """Compress productive env actions into role-slotted protocol steps."""
    steps: list[ExecutableStep] = []
    last_template = ""
    for record in trajectory_steps:
        action = _environment_action(record.get("action", ""))
        if not action:
            continue
        template = role_slot_template(action)
        if not template or template == last_template:
            # Dedup identical consecutive templates (kills action/obs double-count).
            continue
        verb = _first_verb(template) or _first_verb(action)
        obs = " ".join(str(record.get("observation", "") or "").split())
        steps.append(
            ExecutableStep(
                index=len(steps),
                action_template=template,
                expected_obs_hint=obs[:160],
                verb=verb,
            )
        )
        last_template = template
        if len(steps) >= max_steps:
            break
    return steps


def role_slot_template(action: str) -> str:
    """Map a concrete env action to a role-slotted transferable template."""
    text = _environment_action(action)
    if not text:
        return ""
    lower = text.lower()
    # Already slotted.
    if "<" in text and ">" in text:
        return re.sub(
            r"<(object|entity|location|receptacle|place)>",
            _legacy_slot_to_role,
            text,
            flags=re.IGNORECASE,
        )

    if re.match(r"^take .+ from .+$", lower):
        return "take <target> from <source>"
    if re.match(r"^(?:put|move) .+ (?:in|on|to) .+$", lower):
        return "move <target> to <destination>"
    if re.match(r"^(clean|heat|cool) .+ with .+$", lower):
        verb = _first_verb(lower)
        return f"{verb} <target> with <tool>"
    if re.match(r"^go to .+$", lower):
        return "go to <location>"
    if re.match(r"^(open|close) .+$", lower):
        return f"{_first_verb(lower)} <source>"
    if re.match(r"^use .+$", lower):
        return "use <tool>"
    if re.match(r"^examine .+$", lower):
        return "examine <source>"
    # Drop non-structural noise verbs from protocols.
    if _first_verb(lower) in {"look", "inventory"}:
        return ""
    return text


def _legacy_slot_to_role(match: re.Match[str]) -> str:
    token = match.group(1).lower()
    mapping = {
        "object": "target",
        "entity": "source",
        "location": "location",
        "receptacle": "source",
        "place": "destination",
    }
    return f"<{mapping.get(token, token)}>"


def bind_action_template(
    template: str,
    *,
    target: str | None = None,
    destination: str | None = None,
    source: str | None = None,
    tool: str | None = None,
) -> str:
    """Fill role slots for prompt display only (does not choose an action)."""
    text = str(template or "")
    replacements = {
        "<target>": target,
        "<destination>": destination,
        "<source>": source or destination,
        "<tool>": tool,
        "<location>": source or destination or tool,
    }
    for slot, value in replacements.items():
        if value:
            text = text.replace(slot, str(value))
    return text


def infer_current_step_index(
    skill: Skill,
    *,
    history_steps: list[Any] | None,
    observation: str,
) -> int:
    """Return the next unmatched executable step index (0-based)."""
    steps = ensure_executable_protocol(skill)
    if not steps:
        return 0
    actions = _history_actions(history_steps)
    cursor = 0
    for action in actions:
        if cursor >= len(steps):
            break
        if action_matches_template(action, steps[cursor].action_template):
            cursor += 1
            continue
        if _first_verb(action) in _WEAK_VERBS and steps[cursor].verb not in _WEAK_VERBS:
            continue
    if cursor < len(steps) and observation:
        while cursor < len(steps):
            hint = steps[cursor].expected_obs_hint.lower().strip()
            if hint and hint[:40] in observation.lower():
                cursor += 1
                continue
            break
    return min(cursor, max(len(steps) - 1, 0))


def parse_admissible_actions(observation: str) -> list[str]:
    match = _ADMISSIBLE_RE.search(observation or "")
    if not match:
        return []
    body = match.group(1)
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", body)
    if quoted:
        return [item.strip() for item in quoted if item.strip()]
    return [part.strip(" '\"") for part in body.split(",") if part.strip(" '\"")]


def soft_rank_admissible(
    action_template: str,
    admissible: list[str],
    *,
    top_k: int = 5,
) -> list[str]:
    if not admissible:
        return []
    scored: list[tuple[float, str]] = []
    template_tokens = _tokens(action_template) - _PLACEHOLDER_TOKENS
    template_verb = _first_verb(action_template)
    for action in admissible:
        score = 0.0
        action_tokens = _tokens(action)
        if template_verb and _first_verb(action) == template_verb:
            score += 3.0
        score += float(len(template_tokens & action_tokens))
        if action_matches_template(action, action_template):
            score += 5.0
        scored.append((score, action))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [action for score, action in scored if score > 0]
    if not ranked:
        return list(admissible[:top_k])
    return ranked[:top_k]


def protocol_after_activation(
    skill: Skill,
    trial_steps: list[dict[str, Any]],
    *,
    activation_step: int = 1,
) -> list[ExecutableStep]:
    """Use observed prior progress as entry state, never as credited execution.

    Only an in-order prefix with successful environment feedback is consumed.
    Missing/failed prerequisites remain in the suffix; a completed protocol has
    no remaining work to credit. This uses existing observation semantics and
    does not add task-family execution policies.
    """
    from sage_mas.trajectory.abstraction import (
        _observation_confirms_environment_effect,
        _observation_is_noop,
    )

    protocol = ensure_executable_protocol(skill)
    cursor = 0
    start = max(0, int(activation_step) - 1)
    for record in trial_steps[:start]:
        if cursor >= len(protocol):
            break
        observation = record.get("observation")
        if record.get("error") or record.get("result_error") or _observation_is_noop(observation):
            continue
        confirmed = _observation_confirms_environment_effect(observation)
        # Historical wrappers sometimes flag successful ALFWorld actions false;
        # explicit success observations take precedence over that noisy flag.
        if not confirmed and record.get("is_action_valid") is not True:
            continue
        if action_matches_template(record.get("action", ""), protocol[cursor].action_template):
            cursor += 1
    return protocol[cursor:]


def protocol_adherence_score(
    skill: Skill,
    trial_steps: list[dict[str, Any]],
    *,
    activation_step: int = 1,
) -> float:
    """In-order coverage of the remaining protocol after observed entry progress."""
    protocol = protocol_after_activation(skill, trial_steps, activation_step=activation_step)
    if not protocol:
        return 0.0
    start = max(0, int(activation_step) - 1)
    actions = [
        _environment_action(step.get("action", ""))
        for step in trial_steps[start:]
        if _environment_action(step.get("action", ""))
    ]
    matched = 0
    cursor = 0
    for step in protocol:
        found = False
        while cursor < len(actions):
            if action_matches_template(actions[cursor], step.action_template):
                matched += 1
                cursor += 1
                found = True
                break
            cursor += 1
        if not found:
            break
    return matched / max(len(protocol), 1)


def mean_protocol_adherence(
    skill: Skill,
    trials: list[Any],
) -> float:
    scores: list[float] = []
    for trial in trials:
        steps = getattr(trial, "steps", None) or []
        if not steps:
            continue
        activation_map = getattr(trial, "skill_activation_steps", None) or {}
        activation = int(activation_map.get(skill.skill_name, 1) or 1)
        scores.append(
            protocol_adherence_score(
                skill,
                steps,
                activation_step=activation,
            )
        )
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def recent_take_object(history_steps: list[Any] | None) -> str | None:
    for action in reversed(_history_actions(history_steps)):
        if not action.lower().startswith("take "):
            continue
        match = re.match(r"take\s+(.+?)\s+from\s+", action, flags=re.IGNORECASE)
        if match:
            return " ".join(match.group(1).split())
        parts = action.split()
        if len(parts) >= 2:
            return parts[1]
    return None


def protocol_mismatch_lines(
    *,
    target: str | None,
    history_steps: list[Any] | None,
) -> list[str]:
    """Prompt-only mismatch cues (never rewrites actions)."""
    lines: list[str] = []
    taken = recent_take_object(history_steps)
    if target:
        lines.append(f"Task target: {target}")
    if taken:
        lines.append(f"Recent take: {taken}")
    if target and taken:
        t_norm = re.sub(r"[^a-z0-9]", "", target.lower())
        taken_norm = re.sub(r"[^a-z0-9]", "", taken.lower())
        if t_norm and taken_norm and t_norm not in taken_norm and taken_norm not in t_norm:
            lines.append(
                "Mismatch: recent taken object does not match task target."
            )
    # Repeated identical actions.
    actions = _history_actions(history_steps)
    if len(actions) >= 2 and actions[-1] == actions[-2]:
        lines.append(f"Avoid repeating: `{actions[-1]}`")
    return lines


def render_current_step_guidance(
    skills: list[Skill],
    *,
    observation: str,
    history_steps: list[Any] | None,
    top_k: int = 5,
    task: str | None = None,
    gamefile: str | None = None,
) -> str:
    """Prompt block: slotted protocol progress + soft admissible prefs + mismatch."""
    if not skills:
        return ""
    from sage_mas.task_parser import parse_alfworld_task

    parsed = parse_alfworld_task(str(task or ""), str(gamefile or ""))
    admissible = parse_admissible_actions(observation)
    blocks: list[str] = [
        "Executable protocol guidance (trajectory-derived; follow current step):"
    ]
    mismatch = protocol_mismatch_lines(
        target=parsed.target,
        history_steps=history_steps,
    )
    if mismatch:
        blocks.append("Protocol binding / mismatch:")
        blocks.extend(f"- {line}" for line in mismatch)

    for skill in skills[:2]:
        steps = ensure_executable_protocol(skill)
        if not steps:
            continue
        index = infer_current_step_index(
            skill,
            history_steps=history_steps,
            observation=observation,
        )
        current = steps[index]
        bound = bind_action_template(
            current.action_template,
            target=parsed.target,
            destination=parsed.destination,
        )
        ranked = soft_rank_admissible(bound, admissible, top_k=top_k)
        if not ranked:
            ranked = soft_rank_admissible(
                current.action_template,
                admissible,
                top_k=top_k,
            )
        progress = [
            f"{i + 1}. {step.action_template}"
            + (" [done]" if i < index else " [current]" if i == index else "")
            for i, step in enumerate(steps)
        ]
        lines = [
            f"- Skill `{skill.skill_name}` step {index + 1}/{len(steps)}:",
            f"  Learned template: {current.action_template}",
            f"  Bound for this task: {bound}",
            "  Protocol progress:",
            *[f"    {row}" for row in progress],
        ]
        if ranked:
            lines.append(
                "  Closest admissible actions (soft preference, not forced): "
                + "; ".join(ranked)
            )
        else:
            lines.append(
                "  Bind the template to concrete names from the observation "
                "and admissible actions."
            )
        blocks.append("\n".join(lines))
    if len(blocks) == 1:
        return ""
    return "\n".join(blocks)


def action_matches_template(action: str, template: str) -> bool:
    env_action = _environment_action(action)
    if not env_action or not template:
        return False
    if env_action == " ".join(template.split()):
        return True
    action_verb = _first_verb(env_action)
    template_verb = _first_verb(template)
    if action_verb and template_verb and action_verb != template_verb:
        if not ({action_verb, template_verb} <= {"put", "move"}):
            return False
    action_tokens = _tokens(env_action) - {"a", "an", "the", "with", "from", "in", "on", "to"}
    template_tokens = _tokens(template) - {"a", "an", "the", "with", "from", "in", "on", "to"}
    template_tokens -= _PLACEHOLDER_TOKENS
    if template_verb:
        template_tokens.discard(template_verb)
    if action_verb:
        action_tokens.discard(action_verb)
    if not template_tokens:
        if action_verb and template_verb and (
            action_verb == template_verb
            or {action_verb, template_verb} <= {"put", "move"}
        ):
            return True
        return False
    overlap = action_tokens & template_tokens
    return len(overlap) >= max(1, min(2, len(template_tokens)))


def _history_actions(history_steps: list[Any] | None) -> list[str]:
    actions: list[str] = []
    for step in history_steps or []:
        if isinstance(step, dict):
            action = step.get("action", "")
        else:
            action = getattr(step, "action", "")
        env_action = _environment_action(action)
        if env_action:
            actions.append(env_action)
    return actions


def _environment_action(action: Any) -> str:
    text = str(action or "").strip()
    if not text:
        return ""
    match = re.search(r"<action>\s*(.*?)\s*</action>", text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    return " ".join(text.split())


def _first_verb(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", str(text or "").lower())
    return tokens[0] if tokens else ""


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) >= 2 and token not in {"lt", "gt"}
    }
