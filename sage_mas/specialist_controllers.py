"""Controller registry for scoped ALFWorld specialist execution packages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from sage_mas.schemas import AgentSpec, Skill
from sage_mas.specialist_state import (
    SpecialistState,
    contains_entity,
    extract_action_text,
    infer_specialist_state,
)
from sage_mas.task_parser import ParsedTask


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    selected_action_text: str | None
    reason: str
    phase: str
    state: dict[str, Any]
    controller: str


class SpecialistController(Protocol):
    name: str
    supported_task_families: tuple[str, ...]

    def decide(
        self,
        *,
        parsed: ParsedTask,
        raw_action_text: str,
        observation: str,
        admissible_actions: list[str],
        steps: list[dict[str, Any]],
    ) -> ControllerDecision:
        ...


class TransformController:
    """Generic pick-transform-place controller for clean/heat/cool tasks."""

    def __init__(
        self,
        *,
        name: str,
        task_family: str,
        operation: str,
        station_entities: tuple[str, ...],
    ):
        self.name = name
        self.supported_task_families = (task_family,)
        self.operation = operation
        self.station_entities = station_entities

    def decide(
        self,
        *,
        parsed: ParsedTask,
        raw_action_text: str,
        observation: str,
        admissible_actions: list[str],
        steps: list[dict[str, Any]],
    ) -> ControllerDecision:
        state = infer_specialist_state(
            parsed,
            steps=steps,
            observation=observation,
            admissible_actions=admissible_actions,
        )
        selected, reason = self._select_action(
            parsed,
            state,
            admissible_actions,
            raw_action_text,
        )
        return ControllerDecision(
            selected_action_text=selected,
            reason=reason,
            phase=state.phase,
            state=state.as_dict(),
            controller=self.name,
        )

    def _select_action(
        self,
        parsed: ParsedTask,
        state: SpecialistState,
        admissible: list[str],
        raw_text: str,
    ) -> tuple[str | None, str]:
        target = parsed.target
        destination = parsed.destination
        if not admissible:
            return None, "missing admissible action list"

        if target and not state.held_target:
            take = _first_matching(
                admissible,
                lambda action: action.startswith("take ")
                and contains_entity(action, target),
            )
            if take:
                return take, "target visible; take before search"

        if target and state.held_target and not state.operation_done:
            transform = _first_matching(
                admissible,
                lambda action: action.startswith(f"{self.operation} ")
                and contains_entity(action, target)
                and any(entity in action for entity in self.station_entities),
            )
            if transform:
                return transform, (
                    f"holding target at {self.station_entities[0]}; "
                    f"{self.operation} now"
                )
            open_station = _first_matching(
                admissible,
                lambda action: action.startswith("open ")
                and any(contains_entity(action, entity) for entity in self.station_entities),
            )
            if open_station:
                return open_station, f"open {self.station_entities[0]} before {self.operation}"
            goto_station = _best_go_to(
                admissible,
                preferred_entities=self.station_entities,
                visited=(),
            )
            if goto_station:
                return goto_station, (
                    f"holding un{self.operation}ed target; "
                    f"go to {self.station_entities[0]}"
                )
            if _raw_is_aligned(raw_text, state, parsed, admissible, self.operation):
                return raw_text, f"raw action is aligned with {self.operation} phase"
            return _first_matching(admissible, lambda action: action == "look"), (
                f"need {self.station_entities[0]} but no station action available"
            )

        if target and state.held_target and state.operation_done:
            put = _first_matching(
                admissible,
                lambda action: action.startswith(("put ", "move "))
                and contains_entity(action, target)
                and (destination is None or contains_entity(action, destination)),
            )
            if put:
                return put, "target transformed; place in destination"
            if destination:
                open_destination = _first_matching(
                    admissible,
                    lambda action: action.startswith("open ")
                    and contains_entity(action, destination)
                    and action not in state.opened_receptacles,
                )
                if open_destination:
                    return open_destination, "open destination before placement"
                goto_destination = _best_go_to(
                    admissible,
                    preferred_entities=(destination,),
                    visited=(),
                )
                if goto_destination:
                    return goto_destination, "target transformed; go to destination"
            if _raw_is_aligned(raw_text, state, parsed, admissible, self.operation):
                return raw_text, "raw action is aligned with placement phase"
            return _first_matching(admissible, lambda action: action == "look"), (
                "need destination but no destination action available"
            )

        search = _target_search_action(parsed, state, admissible, raw_text, self.operation)
        if search:
            return search
        if _raw_is_aligned(raw_text, state, parsed, admissible, self.operation):
            return raw_text, "raw search action is admissible and non-repeated"
        return _first_matching(admissible, lambda action: action == "look"), (
            "fallback look during target search"
        )


class LightInspectController:
    supported_task_families = ("look_at_obj_in_light",)

    def __init__(self, name: str = "GeneratedLightInspectController"):
        self.name = name

    def decide(
        self,
        *,
        parsed: ParsedTask,
        raw_action_text: str,
        observation: str,
        admissible_actions: list[str],
        steps: list[dict[str, Any]],
    ) -> ControllerDecision:
        held_target, lamp_used = _held_and_used(
            steps,
            target=parsed.target,
            use_entities=("desklamp", "floorlamp", "lamp"),
        )
        state = infer_specialist_state(
            parsed,
            steps=steps,
            observation=observation,
            admissible_actions=admissible_actions,
        )
        phase = "USE_LIGHT" if held_target and not lamp_used else state.phase
        selected, reason = self._select_action(
            parsed,
            state,
            admissible_actions,
            raw_action_text,
            held_target=held_target,
            lamp_used=lamp_used,
        )
        payload = state.as_dict()
        payload["held_target"] = held_target
        payload["lamp_used"] = lamp_used
        return ControllerDecision(
            selected_action_text=selected,
            reason=reason,
            phase=phase,
            state=payload,
            controller=self.name,
        )

    def _select_action(
        self,
        parsed: ParsedTask,
        state: SpecialistState,
        admissible: list[str],
        raw_text: str,
        *,
        held_target: bool,
        lamp_used: bool,
    ) -> tuple[str | None, str]:
        target = parsed.target
        if target and not held_target:
            take = _first_matching(
                admissible,
                lambda action: action.startswith("take ")
                and contains_entity(action, target),
            )
            if take:
                return take, "target visible; take before light use"
        if held_target and not lamp_used:
            use_lamp = _first_matching(
                admissible,
                lambda action: action.startswith("use ")
                and any(entity in action for entity in ("desklamp", "floorlamp", "lamp")),
            )
            if use_lamp:
                return use_lamp, "holding target; use lamp"
            goto_lamp = _best_go_to(
                admissible,
                preferred_entities=("desklamp", "floorlamp", "lamp"),
                visited=(),
            )
            if goto_lamp:
                return goto_lamp, "holding target; go to lamp"
        search = _target_search_action(parsed, state, admissible, raw_text, "inspect_light")
        if search:
            return search
        if raw_text in admissible and not _is_recent_repeat(raw_text, state.recent_actions):
            return raw_text, "raw action admissible for light inspection"
        return _first_matching(admissible, lambda action: action == "look"), (
            "fallback look during light inspection"
        )


class PickTwoController:
    supported_task_families = ("pick_two_obj_and_place",)

    def __init__(self, name: str = "GeneratedPickTwoController"):
        self.name = name

    def decide(
        self,
        *,
        parsed: ParsedTask,
        raw_action_text: str,
        observation: str,
        admissible_actions: list[str],
        steps: list[dict[str, Any]],
    ) -> ControllerDecision:
        state = infer_specialist_state(
            parsed,
            steps=steps,
            observation=observation,
            admissible_actions=admissible_actions,
        )
        placed = _placement_count(steps, parsed.target, parsed.destination)
        selected, reason = self._select_action(
            parsed,
            state,
            admissible_actions,
            raw_action_text,
            placed=placed,
        )
        payload = state.as_dict()
        payload["placed_count"] = placed
        return ControllerDecision(
            selected_action_text=selected,
            reason=reason,
            phase="VERIFY" if placed >= 2 else state.phase,
            state=payload,
            controller=self.name,
        )

    def _select_action(
        self,
        parsed: ParsedTask,
        state: SpecialistState,
        admissible: list[str],
        raw_text: str,
        *,
        placed: int,
    ) -> tuple[str | None, str]:
        if placed >= 2:
            if raw_text in admissible:
                return raw_text, "two targets appear placed; keep admissible raw action"
            return _first_matching(admissible, lambda action: action == "look"), (
                "verify after placing two targets"
            )
        target = parsed.target
        destination = parsed.destination
        if target and not state.held_target:
            take = _first_matching(
                admissible,
                lambda action: action.startswith("take ")
                and contains_entity(action, target),
            )
            if take:
                return take, "target visible; take next instance"
        if target and state.held_target:
            put = _first_matching(
                admissible,
                lambda action: action.startswith(("put ", "move "))
                and contains_entity(action, target)
                and (destination is None or contains_entity(action, destination)),
            )
            if put:
                return put, "holding target instance; place in destination"
            if destination:
                open_destination = _first_matching(
                    admissible,
                    lambda action: action.startswith("open ")
                    and contains_entity(action, destination),
                )
                if open_destination:
                    return open_destination, "open destination for target instance"
                goto_destination = _best_go_to(
                    admissible,
                    preferred_entities=(destination,),
                    visited=(),
                )
                if goto_destination:
                    return goto_destination, "holding target instance; go to destination"
        search = _target_search_action(parsed, state, admissible, raw_text, "pick_two")
        if search:
            return search
        if raw_text in admissible and not _is_recent_repeat(raw_text, state.recent_actions):
            return raw_text, "raw action admissible for pick-two"
        return _first_matching(admissible, lambda action: action == "look"), (
            "fallback look during pick-two"
        )


class SpecialistControllerRegistry:
    """Synthesize scoped controllers from generated Skill contracts.

    The registry stores generic ALFWorld execution templates, not a fixed roster
    of specialists. Each controller instance is generated from the active
    agent's verified Skill / CapabilityContract scope when the agent is
    actually dispatched.
    """

    TEMPLATE_FAMILIES = {
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "look_at_obj_in_light",
        "pick_two_obj_and_place",
    }

    def controller_for(
        self,
        *,
        parsed: ParsedTask,
        primary_agent: str | None,
        agents: list[AgentSpec],
        skills: list[Skill],
    ) -> SpecialistController | None:
        if not primary_agent or primary_agent == "Executor":
            return None
        agent = next((item for item in agents if item.name == primary_agent), None)
        if agent is None:
            return None
        family = parsed.task_family
        if not family:
            return None
        return self.synthesize_controller(agent, skills, parsed)

    def synthesize_controller(
        self,
        agent: AgentSpec,
        skills: list[Skill],
        parsed: ParsedTask,
    ) -> SpecialistController | None:
        family = str(parsed.task_family or "").strip().lower()
        if family not in self.TEMPLATE_FAMILIES:
            return None
        contract = _controller_contract(agent, skills, family)
        if contract is None:
            return None
        skill, contract_text = contract
        controller_name = _generated_controller_name(agent, skill, family)
        if family in {
            "pick_clean_then_place_in_recep",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
        }:
            operation = _operation_for_family_or_contract(family, contract_text)
            if operation is None:
                return None
            station_entities = _station_entities_for_operation(
                operation,
                contract_text,
            )
            if not station_entities:
                return None
            return TransformController(
                name=controller_name,
                task_family=family,
                operation=operation,
                station_entities=station_entities,
            )
        if family == "look_at_obj_in_light":
            return LightInspectController(name=controller_name)
        if family == "pick_two_obj_and_place":
            return PickTwoController(name=controller_name)
        return None

    def supported_task_families(self) -> set[str]:
        return set(self.TEMPLATE_FAMILIES)


DEFAULT_CONTROLLER_REGISTRY = SpecialistControllerRegistry()


def parse_admissible_actions(prompt: str) -> list[str]:
    match = re.search(
        r"admissible actions of the current situation are:\s*\[(.*?)\]\s*\.",
        str(prompt or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    source = match.group(1) if match else str(prompt or "")
    actions = []
    for quoted in re.finditer(r"'([^']+)'|\"([^\"]+)\"", source):
        action = extract_action_text(quoted.group(1) or quoted.group(2))
        if action and action not in actions:
            actions.append(action)
    return actions


def specialist_has_controller_for_family(
    agent: AgentSpec,
    skills: list[Skill],
    task_family: str,
) -> bool:
    family = str(task_family or "").strip().lower()
    if not family:
        return False
    parsed = ParsedTask(
        task="",
        gamefile="",
        task_family=family,
        operation=None,
        target=None,
        destination=None,
    )
    return DEFAULT_CONTROLLER_REGISTRY.synthesize_controller(
        agent,
        skills,
        parsed,
    ) is not None


def specialist_has_controller_for_scopes(
    agent: AgentSpec,
    skills: list[Skill],
) -> bool:
    scopes = _agent_scopes(agent)
    if not scopes:
        assigned = set(agent.assigned_skills)
        for skill in skills:
            if skill.skill_name in assigned:
                scopes.update(_skill_scopes(skill))
    return bool(scopes) and all(
        specialist_has_controller_for_family(agent, skills, scope)
        for scope in scopes
    )


def _agent_scopes(agent: AgentSpec) -> set[str]:
    record = agent.shadow_evaluation_record or {}
    values = list(record.get("task_families") or [])
    contract = record.get("capability_contract") or {}
    if isinstance(contract, dict):
        values.extend(contract.get("task_families") or [])
    return {
        str(value).strip().lower()
        for value in values
        if str(value).strip() and str(value).strip().lower() != "other"
    }


def _skill_scopes(skill: Skill) -> set[str]:
    values = list(skill.applicable_task_families)
    values.extend(skill.metadata.get("task_families", []) or [])
    values.append(skill.metadata.get("primary_task_family", ""))
    return {
        str(value).strip().lower()
        for value in values
        if str(value).strip() and str(value).strip().lower() != "other"
    }


def _capability_implies_family(capability: str, family: str) -> bool:
    markers = {
        "pick_clean_then_place_in_recep": ("transform.clean", " clean", "clean_operation"),
        "pick_heat_then_place_in_recep": ("transform.heat", " heat", "heat_operation"),
        "pick_cool_then_place_in_recep": ("transform.cool", " cool", "cool_operation"),
        "look_at_obj_in_light": ("inspect.with_light", "inspect", "desklamp", "light"),
        "pick_two_obj_and_place": ("pick_two", "two object", "multi_object"),
    }
    return any(marker in capability for marker in markers.get(family, ()))


def _controller_contract(
    agent: AgentSpec,
    skills: list[Skill],
    family: str,
) -> tuple[Skill | None, str] | None:
    assigned = set(agent.assigned_skills)
    for skill in skills:
        if assigned and skill.skill_name not in assigned:
            continue
        text = _skill_contract_text(skill)
        if family in _skill_scopes(skill) or _capability_implies_family(text, family):
            return skill, text

    record = agent.shadow_evaluation_record or {}
    contract = record.get("capability_contract") or {}
    values = list(record.get("task_families") or [])
    text_parts = [
        str(agent.name),
        str(agent.role),
        " ".join(agent.responsibilities or []),
        str(agent.activation_condition or ""),
        str(agent.role_specification or ""),
    ]
    if isinstance(contract, dict):
        values.extend(contract.get("task_families") or [])
        text_parts.extend(
            [
                str(contract.get("name", "")),
                str(contract.get("description", "")),
                " ".join(str(item) for item in contract.get("preconditions", []) or []),
                " ".join(
                    " ".join(str(action) for action in protocol)
                    for protocol in contract.get("action_protocols", []) or []
                    if isinstance(protocol, list)
                ),
                " ".join(str(item) for item in contract.get("expected_effects", []) or []),
            ]
        )
    record_scopes = {
        str(value).strip().lower()
        for value in values
        if str(value).strip() and str(value).strip().lower() != "other"
    }
    text = " ".join(text_parts).lower()
    if family in record_scopes or _capability_implies_family(text, family):
        return None, text
    return None


def _skill_contract_text(skill: Skill) -> str:
    return " ".join(
        [
            skill.capability_key or "",
            skill.skill_name,
            skill.description,
            skill.precondition,
            " ".join(skill.action_protocol or []),
            str(skill.expected_effect or ""),
            str(skill.metadata.get("source_signal", "")),
            str(skill.metadata.get("capability_operation", "")),
            str(skill.metadata.get("primary_task_family", "")),
            " ".join(str(item) for item in skill.metadata.get("task_families", []) or []),
        ]
    ).lower()


def _generated_controller_name(
    agent: AgentSpec,
    skill: Skill | None,
    family: str,
) -> str:
    source = skill.skill_name if skill is not None else agent.name
    source = re.sub(r"[^A-Za-z0-9]+", "", source.title()) or "Specialist"
    family_suffix = {
        "pick_clean_then_place_in_recep": "Clean",
        "pick_heat_then_place_in_recep": "Heat",
        "pick_cool_then_place_in_recep": "Cool",
        "look_at_obj_in_light": "LightInspect",
        "pick_two_obj_and_place": "PickTwo",
    }.get(family, "Scoped")
    return f"Generated{family_suffix}Controller:{source}"


def _operation_for_family_or_contract(
    family: str,
    contract_text: str,
) -> str | None:
    if family == "pick_clean_then_place_in_recep":
        return "clean"
    if family == "pick_heat_then_place_in_recep":
        return "heat"
    if family == "pick_cool_then_place_in_recep":
        return "cool"
    for operation in ("clean", "heat", "cool"):
        if f"transform.{operation}" in contract_text or f"{operation} " in contract_text:
            return operation
    return None


def _station_entities_for_operation(
    operation: str,
    contract_text: str,
) -> tuple[str, ...]:
    if "sinkbasin" in contract_text:
        return ("sinkbasin", "sink")
    if re.search(r"\bsink\b", contract_text):
        return ("sinkbasin", "sink")
    if "microwave" in contract_text:
        return ("microwave",)
    if "fridge" in contract_text or "refrigerator" in contract_text:
        return ("fridge",)
    defaults = {
        "clean": ("sinkbasin", "sink"),
        "heat": ("microwave",),
        "cool": ("fridge",),
    }
    return defaults.get(operation, ())


def _target_search_action(
    parsed: ParsedTask,
    state: SpecialistState,
    admissible: list[str],
    raw_text: str,
    operation: str,
) -> tuple[str, str] | None:
    open_action = _best_open_action(parsed, state, admissible)
    if open_action:
        return open_action, "open searchable receptacle before moving on"
    if _raw_is_aligned(raw_text, state, parsed, admissible, operation):
        if raw_text.startswith("go to ") and _strip_prefix(raw_text, "go to ") not in state.visited_locations:
            return raw_text, "raw go-to explores a new location"
        if raw_text.startswith("open "):
            return raw_text, "raw open explores current receptacle"
    preferred = _search_priority(parsed.target)
    go_action = _best_go_to(
        admissible,
        preferred_entities=preferred,
        visited=state.visited_locations,
    )
    if go_action:
        return go_action, "search target using object-type priority"
    repeated_go = _best_go_to(
        admissible,
        preferred_entities=preferred,
        visited=(),
    )
    if repeated_go and not _is_recent_repeat(repeated_go, state.recent_actions):
        return repeated_go, "search target by revisiting best location"
    return None


def _best_open_action(
    parsed: ParsedTask,
    state: SpecialistState,
    admissible: list[str],
) -> str | None:
    candidates = [
        action
        for action in admissible
        if action.startswith("open ") and action not in state.opened_receptacles
    ]
    if not candidates:
        return None
    priority = _search_priority(parsed.target)
    return sorted(
        candidates,
        key=lambda action: (
            _entity_rank(action, priority),
            _action_number(action),
            action,
        ),
    )[0]


def _best_go_to(
    admissible: list[str],
    *,
    preferred_entities: tuple[str, ...],
    visited: tuple[str, ...] | set[str],
) -> str | None:
    candidates = [action for action in admissible if action.startswith("go to ")]
    if not candidates:
        return None
    unvisited = [
        action
        for action in candidates
        if _strip_prefix(action, "go to ") not in set(visited)
    ]
    pool = unvisited or candidates
    ranked = sorted(
        pool,
        key=lambda action: (
            _entity_rank(action, preferred_entities),
            _action_number(action),
            action,
        ),
    )
    return ranked[0] if ranked else None


def _raw_is_aligned(
    raw_text: str,
    state: SpecialistState,
    parsed: ParsedTask,
    admissible: list[str],
    operation: str,
) -> bool:
    if not raw_text or raw_text not in admissible:
        return False
    if raw_text in state.invalid_actions:
        return False
    if _is_recent_repeat(raw_text, state.recent_actions):
        return False
    target = parsed.target
    destination = parsed.destination
    if target and raw_text.startswith((f"{operation} ", "put ", "move ")) and not contains_entity(raw_text, target):
        return False
    if raw_text.startswith(f"{operation} "):
        return bool(state.held_target and not state.operation_done)
    if raw_text.startswith(("put ", "move ")):
        return bool(
            state.held_target
            and state.operation_done
            and (destination is None or contains_entity(raw_text, destination))
        )
    if raw_text.startswith("take "):
        return bool(target and not state.held_target and contains_entity(raw_text, target))
    return raw_text.startswith(("go to ", "open ", "look", "inventory"))


def _held_and_used(
    steps: list[dict[str, Any]],
    *,
    target: str | None,
    use_entities: tuple[str, ...],
) -> tuple[bool, bool]:
    held = False
    used = False
    for step in steps:
        action = extract_action_text(step.get("action", ""))
        if not bool(step.get("is_action_valid", True)):
            continue
        if target and action.startswith("take ") and contains_entity(action, target):
            held = True
        if target and action.startswith(("put ", "move ")) and contains_entity(action, target):
            held = False
        if action.startswith("use ") and any(entity in action for entity in use_entities):
            used = True
    return held, used


def _placement_count(
    steps: list[dict[str, Any]],
    target: str | None,
    destination: str | None,
) -> int:
    count = 0
    for step in steps:
        if not bool(step.get("is_action_valid", True)):
            continue
        action = extract_action_text(step.get("action", ""))
        if not action.startswith(("put ", "move ")):
            continue
        if target and not contains_entity(action, target):
            continue
        if destination and not contains_entity(action, destination):
            continue
        count += 1
    return count


def _first_matching(actions: list[str], predicate) -> str | None:
    return next((action for action in actions if predicate(action)), None)


def _search_priority(target: str | None) -> tuple[str, ...]:
    target = target or ""
    if target in {"butterknife", "fork", "knife", "spoon", "ladle", "spatula"}:
        return ("countertop", "drawer", "cabinet", "diningtable", "sinkbasin", "sink")
    if target in {"dishsponge", "cloth", "soapbottle", "spraybottle"}:
        return ("sinkbasin", "sink", "countertop", "cabinet", "drawer", "diningtable")
    if target in {"mug", "cup", "bowl", "plate"}:
        return ("countertop", "cabinet", "diningtable", "coffeemachine", "microwave", "sinkbasin")
    return ("countertop", "drawer", "cabinet", "diningtable", "sinkbasin", "coffeemachine")


def _entity_rank(action: str, preferred_entities: tuple[str, ...]) -> int:
    for index, entity in enumerate(preferred_entities):
        if contains_entity(action, entity):
            return index
    return len(preferred_entities)


def _action_number(action: str) -> int:
    match = re.search(r"\b(\d+)\b", action)
    return int(match.group(1)) if match else 999


def _strip_prefix(action: str, prefix: str) -> str:
    return action[len(prefix) :].strip()


def _is_recent_repeat(action: str, recent_actions: list[str]) -> bool:
    if not recent_actions:
        return False
    if recent_actions[-1:] == [action]:
        return True
    return recent_actions[-3:].count(action) >= 2
