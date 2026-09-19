"""Environment-grounded discovery stage for online skill credit."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.prompt_agent.gpt4o_alfworld import _task_family_from_gamefile
from sage_mas.alfworld_evaluator import (
    AlfWorldOrganizationEvaluator,
    EvaluationTrial,
)
from sage_mas.discovery_fork import (
    DiscoveryForkOutcome,
    DiscoveryForkService,
    ExperimentalProtocolGenerator,
    merge_discovery_outcomes,
)
from sage_mas.runtime import ChatBackend
from sage_mas.schemas import (
    AgentSpec,
    ExplorationDistribution,
    Skill,
    SkillStatus,
)
from sage_mas.serialization import read_json, skill_from_dict
from sage_mas.serialization import to_primitive, write_json


@dataclass(slots=True)
class OnlineDiscoveryConfig:
    enabled: bool = False
    auto_capability_queue: bool = True
    operation: str | None = None
    task_family: str | None = None
    min_failures: int = 3
    discovery_tasks: int = 8
    attempts_per_task: int = 2
    min_confirmed_tasks: int = 3
    max_hypotheses: int = 2
    grounded_seed_path: str | None = None


@dataclass(slots=True)
class OnlineDiscoveryOutcome:
    status: str
    skill: Skill | None
    discovery: DiscoveryForkOutcome | None
    summary: dict[str, Any]
    evidence_trajectories: list[dict[str, Any]]


class OnlineDiscoveryCoordinator:
    def __init__(
        self,
        *,
        backend: ChatBackend,
        evaluator: AlfWorldOrganizationEvaluator,
        config: OnlineDiscoveryConfig,
        ledger_path: str | Path | None = None,
    ):
        self.backend = backend
        self.evaluator = evaluator
        self.config = config
        self.ledger_path = Path(ledger_path) if ledger_path else None

    def run(
        self,
        *,
        collection_trials: list[EvaluationTrial | dict[str, Any]],
        agents: list[AgentSpec],
        known_skills: list[Skill],
        discovery_pool: list[str],
    ) -> OnlineDiscoveryOutcome:
        cfg = self.config
        if not cfg.enabled:
            return self._empty("disabled")
        seeded = self._grounded_seed()
        if (
            seeded is not None
            and seeded.skill is not None
            and bool(seeded.skill.applicable_task_families)
            and not self._family_covered(
                seeded.skill.applicable_task_families[0],
                known_skills,
            )
        ):
            return seeded
        all_failures = [
            trial
            for trial in collection_trials
            if not bool(self._value(trial, "won"))
        ]
        if cfg.auto_capability_queue:
            clusters: dict[str, list[EvaluationTrial | dict[str, Any]]] = {}
            for trial in all_failures:
                family = str(self._value(trial, "task_family", "") or "")
                if family:
                    clusters.setdefault(family, []).append(trial)
            queue = [
                (family, members)
                for family, members in clusters.items()
                if len(members) >= cfg.min_failures
                and not self._family_covered(family, known_skills)
            ]
            if not queue:
                return self._empty(
                    "no_uncovered_capability",
                    {
                        "failure_clusters": {
                            family: len(members)
                            for family, members in sorted(clusters.items())
                        }
                    },
                )
            task_family, failures = min(
                queue,
                key=lambda item: (-len(item[1]), item[0]),
            )
            capability_key = (
                "experience."
                + "".join(
                    character
                    if character.isalnum() or character in "._-"
                    else "_"
                    for character in task_family.lower()
                )
            )
            operation = None
        else:
            task_family = str(cfg.task_family or "").strip()
            operation = str(cfg.operation or "").strip().lower() or None
            if not task_family:
                return self._empty("missing_task_family")
            failures = [
                trial
                for trial in all_failures
                if self._value(trial, "task_family") == task_family
            ]
            capability_key = (
                f"transform.{operation}"
                if operation
                else f"experience.{task_family}"
            )
            if any(
                skill.capability_key == capability_key
                for skill in known_skills
                if skill.status not in {
                    SkillStatus.REJECTED,
                    SkillStatus.RETIRED,
                }
            ):
                return self._empty("capability_exists")
        if len(failures) < cfg.min_failures:
            return self._empty(
                "insufficient_failures",
                {"failure_count": len(failures)},
            )
        family_games = [
            gamefile
            for gamefile in discovery_pool
            if _task_family_from_gamefile(gamefile) == task_family
        ]
        if len(family_games) < cfg.min_confirmed_tasks:
            return self._empty(
                "insufficient_discovery_pool",
                {
                    "available": len(family_games),
                    "required": cfg.min_confirmed_tasks,
                    "task_family": task_family,
                },
            )
        ledger = self._load_ledger()
        entry = ledger["capabilities"].setdefault(
            capability_key,
            {
                "task_family": task_family,
                "task_attempt_counts": {},
                "confirmed_transitions": [],
                "evidence_trajectories": [],
                "grounded_skill": None,
            },
        )
        attempt_counts = dict(entry.get("task_attempt_counts") or {})
        games = self._select_rotated_games(
            family_games,
            attempt_counts,
            cfg.discovery_tasks,
        )
        failed_raw = [self._trial_trajectory(trial) for trial in failures]
        evaluator = AlfWorldOrganizationEvaluator(
            self.backend,
            deepcopy(self.evaluator.config),
        )
        outcomes: list[DiscoveryForkOutcome] = []
        merged = None
        for _ in range(cfg.max_hypotheses):
            hypothesis = ExperimentalProtocolGenerator(self.backend).generate(
                operation=operation,
                capability_key=capability_key,
                task_family=task_family,
                failed_trajectories=failed_raw,
            )
            outcomes.append(
                DiscoveryForkService(
                    evaluator,
                    operation=operation,
                    capability_key=capability_key,
                    task_family=task_family,
                    attempts_per_task=cfg.attempts_per_task,
                ).run(
                    experimental_skill=hypothesis,
                    agents=agents,
                    baseline_skills=[
                        skill
                        for skill in known_skills
                        if skill.status == SkillStatus.VERIFIED
                    ],
                    gamefiles=games,
                )
            )
            merged = merge_discovery_outcomes(outcomes)
            if merged.confirmed_task_count >= cfg.min_confirmed_tasks:
                break
        assert merged is not None
        for gamefile in games:
            attempt_counts[gamefile] = int(
                attempt_counts.get(gamefile, 0)
            ) + cfg.attempts_per_task * len(outcomes)
        entry["task_attempt_counts"] = attempt_counts
        skill, cumulative_count, cumulative_evidence = (
            self._accumulate_discovery(entry, merged)
        )
        entry["last_round_confirmed_task_count"] = (
            merged.confirmed_task_count
        )
        entry["cumulative_confirmed_task_count"] = cumulative_count
        self._save_ledger(ledger)
        if (
            skill is None
            or cumulative_count < cfg.min_confirmed_tasks
        ):
            return OnlineDiscoveryOutcome(
                status="insufficient_confirmed_transitions",
                skill=None,
                discovery=merged,
                summary={
                    "confirmed_task_count": cumulative_count,
                    "round_confirmed_task_count": merged.confirmed_task_count,
                    "required": cfg.min_confirmed_tasks,
                    "rotated_discovery_tasks": games,
                },
                evidence_trajectories=cumulative_evidence,
            )
        total = len(failures) + cumulative_count
        skill.exploration_distribution = ExplorationDistribution(
            task_family_histogram={task_family: total},
            outcome_histogram={
                "failure": len(failures),
                "success": cumulative_count,
            },
            signal_histogram={"environment_progress": 1},
            total_trajectories=total,
            success_rate=cumulative_count / max(total, 1),
        )
        skill.metadata.update(
            {
                "grounding_protocol": "environment_confirmed_v1",
                "confirmed_task_count": cumulative_count,
                "discovery_cluster_support": len(failures),
            }
        )
        entry["grounded_skill"] = to_primitive(skill)
        self._save_ledger(ledger)
        return OnlineDiscoveryOutcome(
            status="grounded",
            skill=skill,
            discovery=merged,
            summary={
                "confirmed_task_count": cumulative_count,
                "round_confirmed_task_count": merged.confirmed_task_count,
                "capability_key": capability_key,
                "task_family": task_family,
                "failure_support": len(failures),
            },
            evidence_trajectories=cumulative_evidence,
        )

    def _load_ledger(self) -> dict[str, Any]:
        if self.ledger_path is None or not self.ledger_path.exists():
            return {"version": 1, "capabilities": {}}
        payload = read_json(self.ledger_path)
        if not isinstance(payload, dict):
            return {"version": 1, "capabilities": {}}
        payload.setdefault("version", 1)
        payload.setdefault("capabilities", {})
        return payload

    @staticmethod
    def _select_rotated_games(
        gamefiles: list[str],
        attempt_counts: dict[str, Any],
        count: int,
    ) -> list[str]:
        return sorted(
            gamefiles,
            key=lambda gamefile: (
                int(attempt_counts.get(gamefile, 0)),
                gamefile,
            ),
        )[: max(0, count)]

    def _save_ledger(self, ledger: dict[str, Any]) -> None:
        if self.ledger_path is not None:
            write_json(self.ledger_path, ledger)

    def _accumulate_discovery(
        self,
        entry: dict[str, Any],
        discovery: DiscoveryForkOutcome,
    ) -> tuple[Skill | None, int, list[dict[str, Any]]]:
        transitions = list(entry.get("confirmed_transitions") or [])
        transitions.extend(to_primitive(discovery.confirmed_transitions))
        unique_transitions: dict[tuple[str, str, str], dict[str, Any]] = {}
        for record in transitions:
            key = (
                str(record.get("task_id", "") or ""),
                str(record.get("action", "") or ""),
                str(record.get("observation", "") or ""),
            )
            if key[0]:
                unique_transitions[key] = dict(record)
        accumulated_transitions = list(unique_transitions.values())

        evidence_by_id = {
            str(item.get("trajectory_id", "") or ""): dict(item)
            for item in (entry.get("evidence_trajectories") or [])
            if str(item.get("trajectory_id", "") or "")
        }
        for item in self._grounding_evidence(discovery):
            evidence_by_id[str(item["trajectory_id"])] = item
        evidence = list(evidence_by_id.values())
        task_ids = sorted(
            {
                str(record["task_id"])
                for record in accumulated_transitions
            }
        )

        source = discovery.grounded_skill
        if source is None and isinstance(entry.get("grounded_skill"), dict):
            source = skill_from_dict(entry["grounded_skill"])
        skill = deepcopy(source) if source is not None else None
        if skill is not None:
            skill.evidence_ids = task_ids
            skill.support_count = len(task_ids)
            skill.metadata["confirmed_transitions"] = accumulated_transitions
            skill.metadata["confirmed_task_count"] = len(task_ids)
            operations = [
                str(record.get("operation", "") or "").strip().lower()
                for record in accumulated_transitions
                if str(record.get("operation", "") or "").strip()
            ]
            if operations:
                skill.metadata["capability_operation"] = max(
                    set(operations),
                    key=lambda item: (operations.count(item), item),
                )

        entry["confirmed_transitions"] = accumulated_transitions
        entry["evidence_trajectories"] = evidence
        entry["grounded_skill"] = to_primitive(skill) if skill else None
        return skill, len(task_ids), evidence

    def _family_covered(
        self,
        task_family: str,
        known_skills: list[Skill],
    ) -> bool:
        for skill in known_skills:
            if skill.status in {SkillStatus.REJECTED, SkillStatus.RETIRED}:
                continue
            if task_family not in set(skill.applicable_task_families):
                continue
            credit = skill.metadata.get("skill_credit")
            has_grounded_contract = bool(
                len(set(skill.evidence_ids)) >= 3
                and skill.support_count >= 3
                and str(skill.expected_effect or "").strip()
                and str(skill.metadata.get("anchor_state", "") or "").strip()
            )
            if not has_grounded_contract:
                continue
            if skill.status == SkillStatus.VERIFIED:
                return True
            if not isinstance(credit, dict):
                continue
            # A grounded provisional Skill remains the current experiment for
            # this scope until online credit explicitly retires it.
            return True
        return False

    def _grounded_seed(self) -> OnlineDiscoveryOutcome | None:
        path = self.config.grounded_seed_path
        if not path:
            return None
        payload = read_json(path)
        skill = skill_from_dict(payload.get("skill", payload))
        evidence = list(payload.get("evidence_trajectories") or [])
        if len(set(skill.evidence_ids)) < self.config.min_confirmed_tasks:
            raise ValueError(
                "grounded_seed_path has insufficient independent evidence"
            )
        existing_anchor = str(
            skill.metadata.get("anchor_state", "") or ""
        ).strip()
        task_texts = {
            str(item.get("task", "") or "").strip()
            for item in evidence
            if str(item.get("task", "") or "").strip()
        }
        if not existing_anchor or existing_anchor in task_texts:
            skill.metadata.pop("anchor_state", None)
            transitions_by_task: dict[str, list[dict[str, Any]]] = {}
            for record in skill.metadata.get("confirmed_transitions", []) or []:
                transitions_by_task.setdefault(
                    str(record.get("task_id", "") or ""),
                    [],
                ).append(record)
            for trajectory in evidence:
                steps = list(trajectory.get("steps") or [])
                task_id = str(
                    trajectory.get(
                        "trajectory_id",
                        trajectory.get("gamefile", ""),
                    )
                    or ""
                )
                for index, step in enumerate(steps):
                    if not bool(step.get("is_action_valid", False)):
                        continue
                    transitions = transitions_by_task.get(task_id, [])
                    if transitions and not any(
                        str(record.get("action", "") or "")
                        in str(step.get("action", "") or "")
                        and str(record.get("observation", "") or "")
                        in str(step.get("observation", "") or "")
                        for record in transitions
                    ):
                        continue
                    anchor = (
                        step.get("observation_before")
                        or (
                            steps[index - 1].get("observation")
                            if index > 0
                            else None
                        )
                    )
                    if anchor:
                        skill.metadata["anchor_state"] = str(anchor)[:500]
                        break
                if skill.metadata.get("anchor_state"):
                    break
        return OnlineDiscoveryOutcome(
            status="grounded_seed",
            skill=skill,
            discovery=None,
            summary=dict(payload.get("summary") or {}),
            evidence_trajectories=evidence,
        )

    @classmethod
    def _grounding_evidence(
        cls,
        discovery: DiscoveryForkOutcome,
    ) -> list[dict[str, Any]]:
        selected = []
        for transition in discovery.confirmed_transitions:
            if any(item["trajectory_id"] == transition.task_id for item in selected):
                continue
            for trial in discovery.trials:
                if trial.task_id != transition.task_id:
                    continue
                if any(
                    transition.action in str(step.get("action", ""))
                    and transition.observation
                    in str(step.get("observation", ""))
                    for step in trial.steps
                ):
                    selected.append(cls._trial_trajectory(trial))
                    break
        return selected

    @staticmethod
    def _trial_trajectory(
        trial: EvaluationTrial | dict[str, Any],
    ) -> dict[str, Any]:
        value = OnlineDiscoveryCoordinator._value
        task_id = value(trial, "task_id")
        return {
            "trajectory_id": task_id,
            "gamefile": task_id,
            "task": value(trial, "task"),
            "task_family": value(trial, "task_family"),
            "won": value(trial, "won"),
            "num_steps": value(trial, "num_steps"),
            "token_cost": value(trial, "cost", 0.0),
            "steps": value(trial, "steps", []),
        }

    @staticmethod
    def _value(
        trial: EvaluationTrial | dict[str, Any],
        key: str,
        default: Any = None,
    ) -> Any:
        if isinstance(trial, dict):
            if key == "task_id":
                return trial.get("task_id", trial.get("gamefile", default))
            return trial.get(key, default)
        return getattr(trial, key, default)

    @staticmethod
    def _empty(
        status: str,
        summary: dict[str, Any] | None = None,
    ) -> OnlineDiscoveryOutcome:
        return OnlineDiscoveryOutcome(
            status=status,
            skill=None,
            discovery=None,
            summary=summary or {},
            evidence_trajectories=[],
        )
