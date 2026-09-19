"""Online SAGE-MAS evolution on WebShop (play → distill → org)."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from examples.prompt_agent.gpt4o_webshop import (
    DEFAULT_CATEGORY_QUOTAS,
    WEBSHOP_TEST_SIZE,
    indices_by_category,
    list_webshop_goal_indices,
    load_webshop_goals,
    select_category_stratified_indices,
)
from sage_mas.alfworld_evaluator import EvaluationTrial
from sage_mas.online_evolution_legacy import OnlineAlfWorldEvolution
from sage_mas.runtime import ChatBackend
from sage_mas.serialization import read_json, write_json
from sage_mas.webshop_evaluator import (
    WebShopEvaluatorConfig,
    WebShopOrganizationEvaluator,
)
from sage_mas.webshop_ids import goal_uri, parse_goal_idx


class OnlineWebShopEvolution(OnlineAlfWorldEvolution):
    """ALFWorld online loop with WebShop task ids and evaluator."""

    PROTOCOL_V1 = "webshop_online_segmented_evolution_v1"
    PROTOCOL_V2 = "webshop_online_segmented_evolution_v2"

    def __init__(
        self,
        llm_config: dict[str, Any],
        sage_config: dict[str, Any],
        output_root: str | Path,
        backend: ChatBackend | None = None,
        evaluator: WebShopOrganizationEvaluator | None = None,
    ):
        sage_config = deepcopy(sage_config)
        sage = sage_config.setdefault("sage", {})
        sage.setdefault("trajectory_adapter", "webshop")
        # WebShop smoke defaults: no ALFWorld path keys required.
        online = sage.setdefault("online", {})
        online.setdefault("collection_dataset", "test")
        online.setdefault("mechanism_dataset", "test")
        online.setdefault("game_selection", "category_stratified")

        self._webshop_total_goals: int | None = None
        self._webshop_goals: list[dict[str, Any]] | None = None
        self._category_quotas = self._parse_category_quotas(online)
        super().__init__(
            llm_config=llm_config,
            sage_config=sage_config,
            output_root=output_root,
            backend=backend,
            evaluator=None,
        )
        # Replace ALFWorld evaluators created by the parent constructor.
        self.evaluator = evaluator or WebShopOrganizationEvaluator(
            self.backend,
            self._evaluator_config(),
            specialist_backend=self.specialist_backend,
        )
        if self.acceptor_backend is not None:
            self.acceptor_evaluator = WebShopOrganizationEvaluator(
                self.acceptor_backend,
                self._evaluator_config(),
            )
        else:
            self.acceptor_evaluator = None
        self._retarget_state_protocol()

    def _retarget_state_protocol(self) -> None:
        if not self.state_path.exists():
            return
        state = read_json(self.state_path)
        if not isinstance(state, dict):
            return
        protocol = str(state.get("protocol") or "")
        if protocol.startswith("alfworld_"):
            state["protocol"] = self.PROTOCOL_V1
            write_json(self.state_path, state)

    def _webshop_cfg(self) -> dict[str, Any]:
        cfg = self.llm_config.get("webshop")
        if isinstance(cfg, dict) and cfg:
            return cfg
        # Allow shared knobs under openai-adjacent defaults.
        return {}

    def _parse_category_quotas(self, online_cfg: dict[str, Any]) -> dict[str, int]:
        raw = online_cfg.get("category_quotas") or DEFAULT_CATEGORY_QUOTAS
        if not isinstance(raw, dict) or not raw:
            raw = DEFAULT_CATEGORY_QUOTAS
        quotas = {
            str(key).strip().lower(): max(0, int(value))
            for key, value in raw.items()
        }
        # Keep fashion at the configured value (default 18) even if callers
        # only override a subset of keys.
        for key, value in DEFAULT_CATEGORY_QUOTAS.items():
            quotas.setdefault(key, value)
        return quotas

    def _ensure_webshop_goals(self) -> list[dict[str, Any]]:
        if self._webshop_goals is not None:
            return self._webshop_goals
        ws_cfg = self._webshop_cfg()
        goals = load_webshop_goals(
            seed=self.config.seed,
            num_cpus_per_worker=float(ws_cfg.get("num_cpus_per_worker", 0.1)),
        )
        self._webshop_goals = goals
        self._webshop_total_goals = len(goals)
        return goals

    def _total_goals(self) -> int:
        if self._webshop_total_goals is not None:
            return self._webshop_total_goals
        cfg = self._webshop_cfg()
        if cfg.get("total_goals") is not None:
            self._webshop_total_goals = int(cfg["total_goals"])
            return self._webshop_total_goals
        # Probe once when train / stratified selection needs a real count.
        return len(self._ensure_webshop_goals())

    def _normalize_split(self, split: str) -> str:
        value = str(split or "test").strip().lower()
        if value in {"test", "eval", "valid", "valid_unseen", "valid_seen"}:
            return "test"
        if value in {"train", "training"}:
            return "train"
        raise ValueError(
            "WebShop split must be test|train "
            f"(aliases: valid_unseen/valid_seen→test); got {split!r}"
        )

    def _list_goal_uris(
        self,
        *,
        split: str,
        num_goals: int | None = None,
        excluded: set[str] | None = None,
    ) -> list[str]:
        split_name = self._normalize_split(split)
        total = self._total_goals() if split_name == "train" else None
        indices = list_webshop_goal_indices(
            split=split_name,
            num_goals=None,
            total_goals=total,
        )
        uris = [goal_uri(idx) for idx in indices]
        excluded = set(excluded or [])
        if excluded:
            uris = [uri for uri in uris if uri not in excluded]
        if num_goals is not None:
            uris = uris[: max(0, int(num_goals))]
        return uris

    def _excluded_goal_indices(self) -> set[int]:
        excluded: set[int] = set()
        for uri in self._excluded_collection_gamefiles() or set():
            try:
                excluded.add(parse_goal_idx(uri))
            except ValueError:
                continue
        return excluded

    def _select_gamefiles(self, num_games: int | None = None) -> list[str]:
        count = int(num_games or self.config.num_games)
        selection = str(self.config.game_selection or "first").lower()
        split = self.config.collection_dataset
        excluded = self._excluded_collection_gamefiles()
        if selection in {
            "category_stratified",
            "category_proportional",
            "stratified",
        }:
            goals = self._ensure_webshop_goals()
            buckets = indices_by_category(goals, split=split)
            indices = select_category_stratified_indices(
                buckets,
                num_goals=count,
                segment_size=int(self.config.segment_size),
                quotas=self._category_quotas,
                seed=self.config.seed,
                excluded=self._excluded_goal_indices(),
            )
            if not indices:
                raise RuntimeError(
                    f"category_stratified sampling produced no goals for "
                    f"split={split!r}"
                )
            return [goal_uri(idx) for idx in indices]

        pool = self._list_goal_uris(
            split=split,
            excluded=excluded or None,
        )
        if not pool:
            raise RuntimeError(
                f"No WebShop goals available for split={split!r}"
            )
        if selection in {"first", "hard", "family_proportional", "proportional"}:
            return pool[:count]
        if selection == "shuffle":
            shuffled = list(pool)
            random.Random(self.config.seed).shuffle(shuffled)
            return shuffled[:count]
        raise RuntimeError(
            "Unknown game_selection="
            f"{selection!r}; use first|shuffle|category_stratified for WebShop."
        )

    def _select_mechanism_gamefiles(self, count: int) -> list[str]:
        return self._list_goal_uris(
            split=self.config.mechanism_dataset,
            num_goals=int(count),
        )

    def _select_and_partition_online_gamefiles(self) -> dict[str, Any]:
        required = self.config.shadow_pool_size
        collection = self._select_gamefiles(self.config.num_games)
        if required <= 0:
            return self._with_test_pool(
                {
                    "shadow_pool": [],
                    "collection": collection,
                    "mechanism_carved_from_budget": False,
                }
            )
        excluded = set(collection)
        shadow = self._list_goal_uris(
            split=self.config.mechanism_dataset,
            num_goals=required,
            excluded=excluded,
        )
        if len(shadow) < required:
            # Fall back to carving from a larger collection stream.
            expanded = self._select_gamefiles(self.config.num_games + required)
            shadow = expanded[:required]
            collection = expanded[required : required + self.config.num_games]
            return self._with_test_pool(
                {
                    "shadow_pool": shadow,
                    "collection": collection,
                    "mechanism_carved_from_budget": True,
                }
            )
        return self._with_test_pool(
            {
                "shadow_pool": shadow,
                "collection": collection,
                "mechanism_carved_from_budget": False,
            }
        )

    def _build_heldout_pool(
        self,
        *,
        dataset: str,
        pool_size: int,
        excluded: set[str] | None = None,
        seed_offset: int = 42,
    ) -> list[str]:
        if pool_size <= 0:
            return []
        pool = self._list_goal_uris(
            split=dataset,
            excluded=excluded,
        )
        if not pool:
            return []
        shuffled = list(pool)
        random.Random(self.config.seed + seed_offset).shuffle(shuffled)
        return shuffled[: min(pool_size, len(shuffled))]

    def _evaluator_config(self) -> WebShopEvaluatorConfig:
        from sage_mas.executor_dispatch import dispatch_config_from_mapping

        ws_cfg = self._webshop_cfg()
        online_cfg = self.sage_config.get("sage", {}).get("online", {})
        loop_cfg = self.sage_config.get("sage", {}).get("loop", {})

        def _get(key: str, default: Any) -> Any:
            if key in online_cfg:
                return online_cfg[key]
            if key in loop_cfg:
                return loop_cfg[key]
            return default

        cfg = WebShopEvaluatorConfig(
            max_steps=int(_get("max_steps", ws_cfg.get("max_steps", 50))),
            parallel_envs=int(
                _get("parallel_envs", ws_cfg.get("parallel_envs", 4))
            ),
            api_concurrency=int(
                _get(
                    "api_concurrency",
                    ws_cfg.get("api_concurrency", 4),
                )
            ),
            num_cpus_per_worker=float(
                ws_cfg.get("num_cpus_per_worker", 0.1)
            ),
            history_length=int(
                _get("history_length", ws_cfg.get("history_length", 0))
            ),
            seed=self.config.seed,
            save_steps=bool(_get("save_steps", True)),
            max_advisors=_get("max_advisors", 0),
            max_injected_skills=int(_get("max_injected_skills", 2)),
            gate_injected_skills=bool(_get("gate_injected_skills", True)),
            gate_assigned_skills=bool(_get("gate_assigned_skills", True)),
            use_visited_location_memory=False,
            ignore_assigned_skills=bool(
                online_cfg.get(
                    "ignore_assigned_skills",
                    online_cfg.get("executor_only", False),
                )
            ),
            enable_action_guards=False,
            enable_specialist_controllers=False,
            short_specialist_prompts=bool(
                _get("short_specialist_prompts", True)
            ),
            compact_alfworld_prompts=False,
            prompt_style="alfworld",
            executor_dispatch=dispatch_config_from_mapping(
                self._dispatch_config_mapping()
            ),
        )
        if not cfg.enable_specialist_controllers:
            cfg.executor_dispatch.require_controller_for_eligibility = False
        return cfg

    def _gate_evaluator(self) -> WebShopOrganizationEvaluator:
        return self.acceptor_evaluator or self.evaluator

    def _finalize_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        summary = super()._finalize_summary(state)
        summary["protocol"] = self.PROTOCOL_V2
        summary["domain"] = "webshop"
        summary["webshop_test_size"] = WEBSHOP_TEST_SIZE
        return summary

    @staticmethod
    def _trial_to_trajectory(
        trial: EvaluationTrial, round_index: int
    ) -> dict[str, Any]:
        goal_idx = None
        try:
            goal_idx = parse_goal_idx(trial.task_id)
        except ValueError:
            goal_idx = None
        return {
            "framework": "sage_mas",
            "phase": "online_segment",
            "evolution_round": round_index,
            "gamefile": trial.task_id,
            "task_id": trial.task_id,
            "goal_idx": goal_idx,
            "task_family": trial.task_family,
            "task": trial.task,
            "won": trial.won,
            "num_steps": trial.num_steps,
            "token_cost": trial.cost,
            "assigned_primary_agent": trial.assigned_primary_agent,
            "assignment_rationale": trial.assignment_rationale,
            "eligible_agents": trial.eligible_agents,
            "dispatch_layer": trial.dispatch_layer,
            "dispatch_evidence": trial.dispatch_evidence,
            "steps": trial.steps,
        }
