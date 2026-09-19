"""Organization shadow and reporting services."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from sage_mas.alfworld_evaluator import (
    AlfWorldOrganizationEvaluator,
    EvaluationTrial,
)
from sage_mas.schemas import (
    AgentSpec,
    ShadowDecision,
    ShadowMetrics,
    Skill,
)
from sage_mas.shadow_evaluation import ShadowEvaluator


@dataclass(slots=True)
class OrganizationShadowOutcome:
    old_trials: list[EvaluationTrial]
    new_trials: list[EvaluationTrial]
    old_metrics: ShadowMetrics
    new_metrics: ShadowMetrics
    decision: ShadowDecision


class OrganizationShadowService:
    def __init__(
        self,
        evaluator: AlfWorldOrganizationEvaluator,
        gamefiles: list[str],
        shadow_evaluator: ShadowEvaluator,
    ):
        self.evaluator = evaluator
        self.gamefiles = list(gamefiles)
        self.shadow_evaluator = shadow_evaluator

    def run(
        self,
        old_agents: list[AgentSpec],
        new_agents: list[AgentSpec],
        skills: list[Skill],
    ) -> OrganizationShadowOutcome:
        old_trials = self.evaluator.evaluate(
            agents=old_agents,
            skills=skills,
            gamefiles=self.gamefiles,
            condition="old_organization",
        )
        new_trials = self.evaluator.evaluate(
            agents=new_agents,
            skills=skills,
            gamefiles=self.gamefiles,
            condition="new_organization",
        )
        old_metrics = self._metrics(old_trials, len(old_agents), skills, old_agents)
        new_metrics = self._metrics(new_trials, len(new_agents), skills, new_agents)
        old_by_task = {trial.task_id: trial for trial in old_trials}
        new_by_task = {trial.task_id: trial for trial in new_trials}
        paired_deltas = [
            self.shadow_evaluator.trial_utility(
                new_by_task[task_id].reward,
                new_by_task[task_id].cost,
                len(new_agents),
            )
            - self.shadow_evaluator.trial_utility(
                old_by_task[task_id].reward,
                old_by_task[task_id].cost,
                len(old_agents),
            )
            for task_id in sorted(set(old_by_task) & set(new_by_task))
        ]
        return OrganizationShadowOutcome(
            old_trials=old_trials,
            new_trials=new_trials,
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            decision=self.shadow_evaluator.decide(
                old_metrics,
                new_metrics,
                paired_deltas=paired_deltas,
            ),
        )

    @staticmethod
    def _metrics(
        trials: list[EvaluationTrial],
        agent_count: int,
        skills: list[Skill] | None = None,
        agents: list[AgentSpec] | None = None,
    ) -> ShadowMetrics:
        from sage_mas.executable_protocol import protocol_adherence_score

        success_rate = mean(trial.reward for trial in trials) if trials else 0.0
        token_cost = mean(trial.cost for trial in trials) if trials else 0.0
        adherence_scores: list[float] = []
        skill_by_name = {
            skill.skill_name: skill for skill in (skills or [])
        }
        owners = {
            agent.name: set(agent.assigned_skills)
            for agent in (agents or [])
        }
        for trial in trials:
            primary = str(trial.assigned_primary_agent or "")
            owned = owners.get(primary) or set(trial.activated_skill_names or [])
            names = [
                name
                for name in (trial.activated_skill_names or [])
                if name in owned or not owned
            ]
            if not names and owned:
                names = sorted(owned)
            trial_scores = []
            for name in names:
                skill = skill_by_name.get(name)
                if skill is None:
                    continue
                activation = int(
                    (trial.skill_activation_steps or {}).get(name, 1)
                )
                trial_scores.append(
                    protocol_adherence_score(
                        skill,
                        trial.steps,
                        activation_step=activation,
                    )
                )
            if trial_scores:
                adherence_scores.append(sum(trial_scores) / len(trial_scores))
            else:
                # No skill in play: full adherence so Executor baseline is unchanged.
                adherence_scores.append(1.0)
        protocol_adherence = (
            mean(adherence_scores) if adherence_scores else 1.0
        )
        return ShadowMetrics(
            success_rate=success_rate,
            token_cost=token_cost,
            active_agent_count=float(agent_count),
            protocol_adherence=float(protocol_adherence),
        )


@dataclass(slots=True)
class ReportingConditionResult:
    name: str
    trials: list[EvaluationTrial]
    wins: int
    num_games: int
    success_rate: float
    mean_token_cost: float
    by_family: dict[str, dict[str, float | int]]


@dataclass(slots=True)
class OrganizationReportingOutcome:
    protocol: str
    gamefiles: list[str]
    excluded_leakage: list[str]
    conditions: dict[str, ReportingConditionResult]


class OrganizationReportingService:
    """Run prompt-agent-compatible full-split reporting for one or more orgs."""

    def __init__(
        self,
        evaluator: AlfWorldOrganizationEvaluator,
        gamefiles: list[str],
    ):
        self.evaluator = evaluator
        self.gamefiles = list(gamefiles)

    def run(
        self,
        conditions: dict[str, list[AgentSpec]],
        skills: list[Skill],
    ) -> OrganizationReportingOutcome:
        results: dict[str, ReportingConditionResult] = {}
        for name, agents in conditions.items():
            trials = self.evaluator.evaluate(
                agents=agents,
                skills=skills,
                gamefiles=self.gamefiles,
                condition=name,
            )
            results[name] = self._summarize(name, trials)
        return OrganizationReportingOutcome(
            protocol="alfworld_valid_unseen_reporting_v1",
            gamefiles=self.gamefiles,
            excluded_leakage=[],
            conditions=results,
        )

    @staticmethod
    def _summarize(
        name: str,
        trials: list[EvaluationTrial],
    ) -> ReportingConditionResult:
        wins = sum(int(trial.won) for trial in trials)
        task_wins: dict[str, int] = defaultdict(int)
        task_total: dict[str, int] = defaultdict(int)
        for trial in trials:
            family = trial.task_family
            task_total[family] += 1
            task_wins[family] += int(trial.won)
        by_family = {
            family: {
                "wins": task_wins[family],
                "total": task_total[family],
                "success_rate": task_wins[family] / task_total[family],
            }
            for family in sorted(task_total)
        }
        num_games = len(trials)
        return ReportingConditionResult(
            name=name,
            trials=trials,
            wins=wins,
            num_games=num_games,
            success_rate=(wins / num_games if num_games else 0.0),
            mean_token_cost=(
                mean(trial.cost for trial in trials) if trials else 0.0
            ),
            by_family=by_family,
        )
