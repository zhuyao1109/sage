"""Accept/rollback decisions for candidate organization edits."""

from __future__ import annotations

import math
import random
from statistics import mean

from sage_mas.schemas import ShadowDecision, ShadowMetrics


class ShadowEvaluator:
    def __init__(
        self,
        cost_weight: float = 0.1,
        significance_threshold: float = 0.02,
        agent_count_weight: float = 0.0,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        require_confident_gain: bool = False,
        min_paired_tasks: int = 1,
        seed: int = 0,
        min_new_agent_call_rate: float = 0.0,
    ):
        self.cost_weight = cost_weight
        self.min_relative_gain = significance_threshold
        self.agent_count_weight = agent_count_weight
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.require_confident_gain = require_confident_gain
        self.min_paired_tasks = min_paired_tasks
        self.seed = seed
        # When >0 and new metrics carry call_rate, reject unreachable agents.
        self.min_new_agent_call_rate = float(min_new_agent_call_rate)

    def utility(self, metrics: ShadowMetrics) -> float:
        # Blend task success with protocol adherence so soft specialists that
        # win without following learned steps are not over-credited, and
        # partial adherence still contributes when wins are scarce.
        adherence = max(0.0, min(1.0, float(metrics.protocol_adherence)))
        effective_success = float(metrics.success_rate) * (0.5 + 0.5 * adherence)
        denominator = (
            1.0
            + self.cost_weight * math.log1p(max(0.0, metrics.token_cost))
            + self.agent_count_weight
            * max(0.0, metrics.active_agent_count - 1.0)
        )
        return effective_success / denominator

    def trial_utility(
        self,
        reward: float,
        token_cost: float,
        agent_count: int,
    ) -> float:
        return self.utility(
            ShadowMetrics(
                success_rate=reward,
                token_cost=token_cost,
                active_agent_count=float(agent_count),
            )
        )

    def decide(
        self,
        old: ShadowMetrics,
        new: ShadowMetrics,
        paired_deltas: list[float] | None = None,
    ) -> ShadowDecision:
        old_utility = self.utility(old)
        new_utility = self.utility(new)
        # Allow ties: candidate organizations that are not worse may commit.
        # Strict ">" previously froze evolution when both sides scored 0.
        required = old_utility * (1.0 + self.min_relative_gain)
        paired_supplied = paired_deltas is not None
        paired_deltas = paired_deltas or []
        ci_low, ci_high = self._bootstrap_ci(paired_deltas)
        call_rate = new.new_agent_call_rate
        call_rate_ok = True
        if (
            self.min_new_agent_call_rate > 0.0
            and call_rate is not None
        ):
            call_rate_ok = float(call_rate) + 1e-12 >= float(
                self.min_new_agent_call_rate
            )
        accepted = (
            new_utility + 1e-12 >= required
            and (
                not paired_supplied
                or len(paired_deltas) >= self.min_paired_tasks
            )
            and call_rate_ok
        )
        if self.require_confident_gain:
            accepted = (
                accepted
                and ci_low is not None
                and ci_low > 0.0
            )
        relative_gain = (
            (new_utility - old_utility) / old_utility
            if old_utility > 0
            else (float("inf") if new_utility > 0 else 0.0)
        )
        relation = (
            "meets or exceeds"
            if new_utility + 1e-12 >= required
            else "does not meet"
        )
        reason = (
            f"new utility {new_utility:.6f} {relation} "
            f"required utility {required:.6f}; "
            f"paired tasks={len(paired_deltas)}, "
            f"CI={self._format_ci(ci_low, ci_high)}"
        )
        if call_rate is not None:
            reason += (
                f"; new_agent_call_rate={float(call_rate):.3f} "
                f"(min={float(self.min_new_agent_call_rate):.3f})"
            )
        if not call_rate_ok:
            reason = (
                "Rejected: new agent call_rate below threshold "
                f"({float(call_rate or 0.0):.3f} < "
                f"{float(self.min_new_agent_call_rate):.3f}); {reason}."
            )
        else:
            reason += "."
        return ShadowDecision(
            accepted=accepted,
            old_utility=old_utility,
            new_utility=new_utility,
            relative_gain=relative_gain,
            paired_delta_ci_low=ci_low,
            paired_delta_ci_high=ci_high,
            paired_task_count=len(paired_deltas),
            new_agent_call_rate=(
                float(call_rate) if call_rate is not None else None
            ),
            reason=reason,
        )

    def _bootstrap_ci(
        self,
        deltas: list[float],
    ) -> tuple[float | None, float | None]:
        if not deltas:
            return None, None
        if len(deltas) == 1 or self.bootstrap_samples <= 0:
            value = deltas[0]
            return value, value
        rng = random.Random(self.seed)
        samples: list[float] = []
        n = len(deltas)
        for _ in range(self.bootstrap_samples):
            draw = [deltas[rng.randrange(n)] for _ in range(n)]
            samples.append(mean(draw))
        samples.sort()
        alpha = 1.0 - self.confidence_level
        low_index = int(alpha / 2.0 * (len(samples) - 1))
        high_index = int((1.0 - alpha / 2.0) * (len(samples) - 1))
        return samples[low_index], samples[high_index]

    @staticmethod
    def _format_ci(low: float | None, high: float | None) -> str:
        if low is None or high is None:
            return "n/a"
        return f"[{low:.4f}, {high:.4f}]"
