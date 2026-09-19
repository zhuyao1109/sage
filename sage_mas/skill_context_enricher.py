"""Enhanced skill context for weak models: goal-aware, step-by-step guidance.

This module enriches skill presentation in the prompt to help weak models execute
abstract protocols. It adds:
1. Task goal reminder (why this skill)
2. Multi-step lookahead (what comes next)
3. State-aware hints (what to check in observation)
4. Common failure patterns to avoid
"""

from __future__ import annotations

from typing import Any

from sage_mas.executable_protocol import (
    bind_action_template,
    ensure_executable_protocol,
    infer_current_step_index,
    parse_admissible_actions,
    soft_rank_admissible,
)
from sage_mas.schemas import Skill
from sage_mas.task_parser import parse_alfworld_task


def render_enriched_skill_guidance(
    skills: list[Skill],
    *,
    observation: str,
    task: str | None = None,
    gamefile: str | None = None,
    history_steps: list[Any] | None = None,
    max_lookahead: int = 3,
    include_rationale: bool = True,
) -> str:
    """Render skill guidance with goal context, lookahead, and state hints."""
    if not skills:
        return ""

    parsed = parse_alfworld_task(str(task or ""), str(gamefile or ""))
    admissible = parse_admissible_actions(observation)

    blocks: list[str] = []

    for skill in skills[:2]:  # Focus on top 2 active skills
        steps = ensure_executable_protocol(skill)
        if not steps:
            continue

        current_index = infer_current_step_index(
            skill,
            history_steps=history_steps,
            observation=observation,
        )

        # Build skill context block
        skill_block = _build_skill_context_block(
            skill=skill,
            steps=steps,
            current_index=current_index,
            parsed_task=parsed,
            observation=observation,
            admissible=admissible,
            max_lookahead=max_lookahead,
            include_rationale=include_rationale,
        )

        if skill_block:
            blocks.append(skill_block)

    if not blocks:
        return ""

    header = (
        "Active Skill Guidance (trajectory-learned protocols with execution context):\n"
        "Follow the current step; check state hints before acting."
    )
    return f"{header}\n\n" + "\n\n".join(blocks)


def _build_skill_context_block(
    *,
    skill: Skill,
    steps: list,
    current_index: int,
    parsed_task: Any,
    observation: str,
    admissible: list[str],
    max_lookahead: int,
    include_rationale: bool,
) -> str:
    """Build a single skill's enriched guidance block."""
    lines: list[str] = []

    # Header: skill name and progress
    lines.append(
        f"[{skill.skill_name}] Step {current_index + 1}/{len(steps)}"
    )

    # Few-shot examples from successful trajectories (at skill level)
    skill_examples = _extract_skill_level_examples(skill, max_examples=2)
    if skill_examples:
        lines.append(f"Examples: {skill_examples}")

    # Goal reminder (why this skill matters)
    if include_rationale and skill.expected_effect:
        lines.append(f"Goal: {skill.expected_effect}")
    elif include_rationale and parsed_task.operation:
        operation_goals = {
            "cool": "Cool the target object in fridge, then place it at destination",
            "heat": "Heat the target object in microwave, then place it at destination",
            "clean": "Clean the target object with sinkbasin, then place it at destination",
        }
        if parsed_task.operation in operation_goals:
            lines.append(f"Goal: {operation_goals[parsed_task.operation]}")

    # Current step with concrete binding
    current_step = steps[current_index]
    bound_current = bind_action_template(
        current_step.action_template,
        target=parsed_task.target,
        destination=parsed_task.destination,
    )
    lines.append(f"Current: {bound_current}")

    # State check hint (what to verify in observation)
    state_hint = _generate_state_hint(
        step_template=current_step.action_template,
        parsed_task=parsed_task,
        observation=observation,
    )
    if state_hint:
        lines.append(f"Check: {state_hint}")

    # Concrete action suggestions from admissible
    ranked = soft_rank_admissible(bound_current, admissible, top_k=3)
    if ranked:
        lines.append(f"Suggested: {' OR '.join(ranked)}")
    else:
        # Fallback: try template matching
        ranked = soft_rank_admissible(
            current_step.action_template,
            admissible,
            top_k=3,
        )
        if ranked:
            lines.append(f"Suggested: {' OR '.join(ranked)}")

    # Lookahead (next few steps for context)
    if current_index + 1 < len(steps):
        lookahead_end = min(current_index + 1 + max_lookahead, len(steps))
        next_steps = []
        for i in range(current_index + 1, lookahead_end):
            bound_next = bind_action_template(
                steps[i].action_template,
                target=parsed_task.target,
                destination=parsed_task.destination,
            )
            next_steps.append(bound_next)
        if next_steps:
            lines.append(f"Next: {' → '.join(next_steps)}")

    # Common failure patterns for this step type
    # Note: history_steps not available in this context, skip warning for now
    # TODO: pass history_steps through the call chain if needed
    warning = None
    # warning = _generate_step_warning(
    #     step_template=current_step.action_template,
    #     history_steps=history_steps,
    # )
    if warning:
        lines.append(f"⚠ {warning}")

    return "\n  ".join(lines)


def _generate_state_hint(
    *,
    step_template: str,
    parsed_task: Any,
    observation: str,
    state_hint_patterns: dict[str, dict[str, str]] | None = None,
) -> str:
    """Generate a hint about what to check in the observation before acting."""
    if state_hint_patterns is None:
        state_hint_patterns = _get_default_state_hint_patterns()

    template_lower = step_template.lower()
    obs_lower = observation.lower()

    # Check each pattern category
    for pattern_key, pattern_config in state_hint_patterns.items():
        keywords = pattern_config.get("keywords", [])
        if not keywords:
            continue

        # Check if template matches this pattern
        if all(kw in template_lower for kw in keywords):
            # Check conditions
            conditions = pattern_config.get("conditions", {})
            holding_check = conditions.get("holding")
            visible_check = conditions.get("visible")

            if holding_check is not None:
                is_holding = "you are carrying" in obs_lower and "nothing" not in obs_lower
                if is_holding == holding_check:
                    return pattern_config.get("hint_true", "")
                else:
                    return pattern_config.get("hint_false", "")

            if visible_check and parsed_task.target:
                target_lower = parsed_task.target.lower()
                if target_lower in obs_lower:
                    return pattern_config.get("hint_true", "").format(target=parsed_task.target)
                else:
                    return pattern_config.get("hint_false", "").format(target=parsed_task.target)

            # Default hint for this pattern
            return pattern_config.get("hint_default", "")

    return ""


def _get_default_state_hint_patterns() -> dict[str, dict[str, str]]:
    """Get default state hint patterns (can be overridden via config)."""
    return {
        "navigation": {
            "keywords": ["go to"],
            "hint_default": "Verify the location exists and is reachable",
        },
        "take_object": {
            "keywords": ["take", "from"],
            "conditions": {"visible": True},
            "hint_true": "Confirmed: {target} is visible",
            "hint_false": "Search for {target} in this location",
        },
        "cool_operation": {
            "keywords": ["cool", "with"],
            "conditions": {"holding": True},
            "hint_true": "You're holding the object; ready to cool",
            "hint_false": "Must be holding object before cooling",
        },
        "heat_operation": {
            "keywords": ["heat", "with"],
            "conditions": {"holding": True},
            "hint_true": "You're holding the object; ready to heat",
            "hint_false": "Must be holding object before heating",
        },
        "clean_operation": {
            "keywords": ["clean", "with"],
            "conditions": {"holding": True},
            "hint_true": "You're holding the object; ready to clean",
            "hint_false": "Must be holding object before cleaning",
        },
        "place_object": {
            "keywords": ["move", "to"],
            "conditions": {"holding": True},
            "hint_true": "You're holding the object; ready to place at destination",
            "hint_false": "Must be holding object to place it",
        },
        "open_container": {
            "keywords": ["open"],
            "hint_default": "Check if container is currently closed",
        },
        "close_container": {
            "keywords": ["close"],
            "hint_default": "Check if container is currently open",
        },
    }


def _generate_step_warning(
    *,
    step_template: str,
    history_steps: list[Any] | None,
    warning_patterns: dict[str, Any] | None = None,
    loop_threshold: int = 2,
) -> str:
    """Generate warnings about common failure patterns for this step type."""
    if not history_steps:
        return ""

    if warning_patterns is None:
        warning_patterns = _get_default_warning_patterns()

    template_lower = step_template.lower()
    recent_actions = [
        str(step.get("action", "") if isinstance(step, dict) else getattr(step, "action", ""))
        for step in (history_steps[-5:] if len(history_steps) > 5 else history_steps)
    ]

    # Detect loops
    if recent_actions:
        last_action = recent_actions[-1]
        action_count = recent_actions.count(last_action)

        if action_count >= loop_threshold:
            loop_config = warning_patterns.get("loop_detection", {})
            for pattern_key, pattern_data in loop_config.items():
                keywords = pattern_data.get("keywords", [])
                if all(kw in template_lower for kw in keywords):
                    return pattern_data.get("warning", "")

    # Pattern-based warnings
    for pattern_key, pattern_data in warning_patterns.get("operation_warnings", {}).items():
        keywords = pattern_data.get("keywords", [])
        if all(kw in template_lower for kw in keywords):
            return pattern_data.get("warning", "")

    return ""


def _get_default_warning_patterns() -> dict[str, Any]:
    """Get default warning patterns (can be overridden via config)."""
    return {
        "loop_detection": {
            "navigation_loop": {
                "keywords": ["go to"],
                "warning": "Avoid revisiting same location repeatedly; explore new areas",
            },
            "container_loop": {
                "keywords": ["open", "close"],
                "warning": "Avoid open/close loops; take or place object before closing",
            },
        },
        "operation_warnings": {
            "cool_sequence": {
                "keywords": ["cool"],
                "warning": "Ensure container is open BEFORE operation; close AFTER",
            },
            "heat_sequence": {
                "keywords": ["heat"],
                "warning": "Ensure container is open BEFORE operation; close AFTER",
            },
        },
    }


def render_task_decomposition(
    *,
    task: str,
    task_family: str | None = None,
    skills: list[Skill],
    decomposition_templates: dict[str, list[str]] | None = None,
) -> str:
    """Render high-level task decomposition to help weak models understand the plan."""
    if not task or not skills:
        return ""

    # Use provided templates or extract from skills
    if decomposition_templates is None:
        decomposition_templates = _infer_decomposition_from_skills(skills, task_family)

    if task_family and task_family in decomposition_templates:
        steps = decomposition_templates[task_family]
        lines = ["Task Breakdown:"] + [f"{i}. {step}" for i, step in enumerate(steps, 1)]
        lines.append(f"\nCurrent task: {task}")
        return "\n".join(lines)

    return ""


def _infer_decomposition_from_skills(
    skills: list[Skill],
    task_family: str | None,
) -> dict[str, list[str]]:
    """
    Infer task decomposition from skill protocols instead of hardcoding.

    Groups protocol steps into semantic phases based on verb patterns.
    """
    decompositions = {}

    for skill in skills:
        families = skill.applicable_task_families or []
        if task_family and task_family not in families:
            continue

        protocol = skill.action_protocol or []
        if not protocol:
            continue

        # Group protocol steps into phases
        phases = _group_protocol_into_phases(protocol)

        for family in families:
            if family not in decompositions and phases:
                decompositions[family] = phases

    return decompositions


def _group_protocol_into_phases(protocol: list[str]) -> list[str]:
    """
    Group protocol steps into high-level phases based on verb patterns.

    Example:
        ["go to X", "go to Y", "take A", "go to Z", "cool A", "move A"]
        → ["Navigate to find target", "Pick up object", "Navigate to tool",
           "Apply operation", "Place at destination"]
    """
    phases = []
    current_phase = []
    last_verb_category = None

    verb_categories = {
        "navigate": {"go", "look", "examine"},
        "acquire": {"take", "pick", "grab", "get"},
        "transform": {"cool", "heat", "clean", "wash"},
        "container": {"open", "close"},
        "place": {"move", "put", "place"},
        "use": {"use", "toggle", "turn"},
    }

    for step in protocol:
        step_lower = step.lower()
        first_word = step_lower.split()[0] if step_lower.split() else ""

        # Determine verb category
        verb_category = None
        for category, verbs in verb_categories.items():
            if first_word in verbs:
                verb_category = category
                break

        # Group consecutive steps of same category
        if verb_category != last_verb_category and last_verb_category is not None:
            phase_desc = _describe_phase(last_verb_category, current_phase)
            if phase_desc:
                phases.append(phase_desc)
            current_phase = []

        current_phase.append(step)
        last_verb_category = verb_category

    # Add final phase
    if current_phase and last_verb_category:
        phase_desc = _describe_phase(last_verb_category, current_phase)
        if phase_desc:
            phases.append(phase_desc)

    return phases


def _describe_phase(category: str, steps: list[str]) -> str:
    """Describe a phase in natural language."""
    phase_descriptions = {
        "navigate": "Navigate to required location",
        "acquire": "Pick up the target object",
        "transform": "Apply required operation to object",
        "container": "Manage container state",
        "place": "Place object at destination",
        "use": "Use the required tool",
    }
    return phase_descriptions.get(category, f"Perform {category} actions")


def _extract_skill_level_examples(
    skill: Skill,
    max_examples: int = 2,
) -> str:
    """Extract representative examples from skill's key_fragments.

    Key fragments contain the most important actions from successful trajectories
    (e.g., 'heat apple 1 with microwave 1', 'use desklamp 1').

    Args:
        skill: The skill containing key_fragments
        max_examples: Maximum number of examples to include

    Returns:
        Formatted string with examples, or empty string if none found
    """
    key_fragments = skill.key_fragments if hasattr(skill, 'key_fragments') else None
    if not key_fragments:
        return ""

    # Select diverse examples (avoid duplicates)
    seen_actions = set()
    selected_fragments = []

    for fragment in key_fragments:
        if not isinstance(fragment, dict):
            continue

        action = fragment.get('action', '').strip()
        observation = fragment.get('observation', '').strip()

        if not action or not observation:
            continue

        # Normalize action to detect duplicates (e.g., "use desklamp 1" -> "use desklamp")
        action_normalized = ' '.join(action.lower().split()[:2])

        if action_normalized not in seen_actions:
            seen_actions.add(action_normalized)
            selected_fragments.append({
                'action': action,
                'observation': observation,
            })

        if len(selected_fragments) >= max_examples:
            break

    if not selected_fragments:
        return ""

    # Format examples compactly
    example_lines = []
    for frag in selected_fragments:
        # Truncate long observations
        obs = frag['observation']
        if len(obs) > 60:
            obs = obs[:57] + "..."
        example_lines.append(f"'{frag['action']}' → {obs}")

    return " | ".join(example_lines)


def _extract_few_shot_examples(
    skill: Skill,
    current_step_template: str,
    max_examples: int = 2,
) -> str:
    """Extract few-shot examples from skill's key_fragments.

    Selects fragments that match the current step template to provide
    concrete examples of successful execution.

    Args:
        skill: The skill containing key_fragments
        current_step_template: The current step template (e.g., "use <entity>")
        max_examples: Maximum number of examples to include

    Returns:
        Formatted string with examples, or empty string if none found
    """
    key_fragments = skill.key_fragments if hasattr(skill, 'key_fragments') else None
    if not key_fragments:
        return ""

    # Extract the verb and pattern from current step template
    template_lower = current_step_template.lower().strip()
    template_parts = template_lower.split()
    if not template_parts:
        return ""

    template_verb = template_parts[0]

    # For templates with specific entities (e.g., "use desklamp"), match exactly
    # For generic templates (e.g., "go to <location>"), match by verb only
    has_placeholder = '<' in template_lower and '>' in template_lower

    # Find matching fragments
    matching_fragments = []
    for fragment in key_fragments:
        if not isinstance(fragment, dict):
            continue

        action = fragment.get('action', '').lower().strip()
        observation = fragment.get('observation', '').strip()

        if not action or not observation:
            continue

        action_parts = action.split()
        if not action_parts:
            continue

        action_verb = action_parts[0]

        # Match logic
        is_match = False
        if has_placeholder:
            # Generic template: match by verb only
            is_match = (action_verb == template_verb)
        else:
            # Specific template: match verb and check if action starts with template pattern
            # e.g., template "use desklamp" matches action "use desklamp 1"
            template_prefix = ' '.join(template_parts)
            is_match = action.startswith(template_prefix)

        if is_match:
            matching_fragments.append({
                'action': fragment['action'],
                'observation': observation,
            })

        if len(matching_fragments) >= max_examples:
            break

    if not matching_fragments:
        return ""

    # Format examples compactly
    example_lines = []
    for i, frag in enumerate(matching_fragments, 1):
        # Truncate long observations
        obs = frag['observation']
        if len(obs) > 80:
            obs = obs[:77] + "..."
        example_lines.append(f"'{frag['action']}' → {obs}")

    return " | ".join(example_lines)
