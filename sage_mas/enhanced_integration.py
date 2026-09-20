"""Integration module: connect enhanced components to existing SAGE-MAS pipeline.

This module provides drop-in replacements and wrappers for existing pipeline
components, making it easy to enable the enhancements.

All behavior is configurable via enhancement_config.yaml - no hardcoded values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sage_mas.enhancement_config_loader import (
    EnhancementConfig,
    extract_dispatcher_params,
    extract_phase_descriptions,
    extract_state_hint_patterns,
    extract_verb_categories,
    extract_verifier_params,
    extract_warning_patterns,
    load_enhancement_config,
)
from sage_mas.meta_agent_dispatcher import MetaAgentDispatcher
from sage_mas.runtime import MASRuntime
from sage_mas.schemas import AgentSpec, Skill
from sage_mas.skill_context_enricher import (
    render_enriched_skill_guidance,
    render_task_decomposition,
)
from sage_mas.skill_verification_enhanced import (
    EnhancedSkillVerifier,
    batch_verify_skills,
)


def _enhanced_debug_enabled() -> bool:
    return os.environ.get("SAGE_ENHANCED_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug(msg: str) -> None:
    if _enhanced_debug_enabled():
        print(msg, flush=True)

class EnhancedMASRuntime(MASRuntime):
    """
    Drop-in replacement for MASRuntime with enhanced skill context rendering.

    All behavior is controlled via EnhancementConfig - no hardcoded values.

    Usage:
        config = load_enhancement_config("enhancement_config.yaml")
        runtime = EnhancedMASRuntime(
            agents, skills, backend,
            enhancement_config=config,
        )
    """

    def __init__(
        self,
        agents: list[AgentSpec],
        skills: list[Skill],
        backend: Any,
        *,
        enhancement_config: EnhancementConfig | None = None,
        enhancement_config_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(agents, skills, backend, **kwargs)

        # Load enhancement config
        if enhancement_config is None:
            if enhancement_config_path:
                self.enhancement_config = load_enhancement_config(enhancement_config_path)
            else:
                # Look for default config file
                default_path = Path("sage_mas/enhancement_config.yaml")
                if default_path.exists():
                    self.enhancement_config = load_enhancement_config(default_path)
                else:
                    from sage_mas.enhancement_config_loader import get_default_config
                    self.enhancement_config = get_default_config()
        else:
            self.enhancement_config = enhancement_config

        # Extract runtime config
        runtime_config = self.enhancement_config.get_runtime_config()
        # Soft / sparse / hybrid soft SOP must not layer enriched Step-k/N.
        if getattr(self, "skill_inject_mode", "full") in {
            "soft",
            "sparse_soft",
            "hybrid_soft",
        }:
            self.enable_enriched_context = False
            self.enable_task_decomposition = False
        else:
            self.enable_enriched_context = runtime_config.get(
                "enable_enriched_context", False
            )
            self.enable_task_decomposition = runtime_config.get(
                "enable_task_decomposition", False
            )
        self.enable_meta_dispatcher = runtime_config.get("enable_meta_dispatcher", False)

        # Extract component-specific configs
        skill_context_config = self.enhancement_config.get_skill_context_config()
        self.max_lookahead = skill_context_config.get("max_lookahead", 3)
        self.include_rationale = skill_context_config.get("include_rationale", True)
        self.loop_threshold = skill_context_config.get("loop_threshold", 2)

        # Extract patterns
        self.state_hint_patterns = extract_state_hint_patterns(self.enhancement_config)
        self.warning_patterns = extract_warning_patterns(self.enhancement_config)
        self.verb_categories = extract_verb_categories(self.enhancement_config)
        self.phase_descriptions = extract_phase_descriptions(self.enhancement_config)

        # Initialize meta dispatcher if enabled
        self.meta_dispatcher = None
        if self.enable_meta_dispatcher:
            dispatcher_params = extract_dispatcher_params(self.enhancement_config)
            dispatcher_config = self.enhancement_config.get_meta_dispatcher_config()
            llm_config = dispatcher_config.get("llm_dispatch", {})

            self.meta_dispatcher = MetaAgentDispatcher(
                agents=agents,
                skills=skills,
                enable_llm_dispatch=llm_config.get("enabled", False),
                **dispatcher_params,
            )

    def _alfworld_user_prompt(
        self,
        observation: str,
        messages: list,
        *,
        injected_skills: list[Skill] | None = None,
        active_assigned_skill_names: set[str] | None = None,
        actor: AgentSpec | None = None,
        task_family: str | None = None,
        task: str | None = None,
        history_steps: list | None = None,
        gamefile: str | None = None,
    ) -> str:
        """Override to inject enhanced skill context."""
        _debug(
            f"[ENHANCED_DEBUG] _alfworld_user_prompt called, "
            f"enable_enriched_context={self.enable_enriched_context}"
        )

        # Get base prompt from parent
        base_prompt = super()._alfworld_user_prompt(
            observation,
            messages,
            injected_skills=injected_skills,
            active_assigned_skill_names=active_assigned_skill_names,
            actor=actor,
            task_family=task_family,
            task=task,
            history_steps=history_steps,
            gamefile=gamefile,
        )

        # Add enriched skill context if enabled
        if not self.enable_enriched_context:
            return base_prompt

        active_skills = self._get_active_skills(
            actor or self.executor,
            active_assigned_skill_names,
            injected_skills,
        )

        _debug(
            f"[ENHANCED_DEBUG] Actor: "
            f"{(actor or self.executor).name if actor or self.executor else 'None'}"
        )
        _debug(
            f"[ENHANCED_DEBUG] Actor assigned_skills: "
            f"{(actor or self.executor).assigned_skills if actor or self.executor else []}"
        )
        _debug(
            f"[ENHANCED_DEBUG] active_assigned_skill_names: {active_assigned_skill_names}"
        )
        _debug(
            f"[ENHANCED_DEBUG] injected_skills: "
            f"{[s.skill_name for s in (injected_skills or [])]}"
        )
        _debug(f"[ENHANCED_DEBUG] Total skills in runtime: {len(self.skill_by_name)}")
        _debug(f"[ENHANCED_DEBUG] Active skills count: {len(active_skills)}")
        if active_skills:
            _debug(
                f"[ENHANCED_DEBUG] Active skill names: "
                f"{[s.skill_name for s in active_skills]}"
            )

        if not active_skills:
            _debug("[ENHANCED_DEBUG] No active skills, returning base prompt")
            return base_prompt

        enriched_parts = [base_prompt]

        # Add task decomposition (high-level plan)
        if self.enable_task_decomposition and task_family:
            decomp_config = self.enhancement_config.get_task_decomposition_config()
            decomp_mode = decomp_config.get("mode", "infer_from_skills")

            # Build decomposition templates based on mode
            decomposition_templates = None
            if decomp_mode == "predefined":
                decomposition_templates = decomp_config.get("templates", {})

            decomp = render_task_decomposition(
                task=task or "",
                task_family=task_family,
                skills=active_skills,
                decomposition_templates=decomposition_templates,
            )
            if decomp:
                enriched_parts.append(decomp)

        # Add enriched skill guidance (step-by-step with context)
        enriched_guidance = render_enriched_skill_guidance(
            skills=active_skills,
            observation=observation,
            task=task,
            gamefile=gamefile,
            history_steps=history_steps,
            max_lookahead=self.max_lookahead,
            include_rationale=self.include_rationale,
        )

        if enriched_guidance:
            enriched_parts.append(enriched_guidance)
            _debug(
                f"[ENHANCED_DEBUG] Added enriched guidance, "
                f"length={len(enriched_guidance)}"
            )
        else:
            _debug("[ENHANCED_DEBUG] No enriched guidance generated")

        final_prompt = "\n\n".join(enriched_parts)
        _debug(
            f"[ENHANCED_DEBUG] Final prompt length: "
            f"base={len(base_prompt)}, enriched={len(final_prompt)}"
        )
        return final_prompt

    def _get_active_skills(
        self,
        actor: AgentSpec,
        active_skill_names: set[str] | None,
        injected_skills: list[Skill] | None,
    ) -> list[Skill]:
        """Get list of currently active skills for the actor."""
        if self.turn_delegation:
            # Use the selected Skill objects, not an ambiguous name lookup that
            # may resolve another revision of a skill with the same name.
            return [s for s in (injected_skills or [])
                    if actor.agent_id == self.executor.agent_id or s.skill_name in actor.assigned_skills]
        skills = []

        # Assigned skills
        if active_skill_names:
            skills.extend([
                self.skill_by_name[name]
                for name in actor.assigned_skills
                if name in self.skill_by_name and name in active_skill_names
            ])

        # Injected skills
        if injected_skills:
            skills.extend(injected_skills)

        # FALLBACK: If no skills are active but we have skills in the bank,
        # use all available skills for enrichment (for testing/bootstrapping).
        # Under strict prompt-based retrieval an empty selection is honored.
        if (
            not skills
            and self.skill_by_name
            and not getattr(self, "strict_skill_selection", False)
        ):
            _debug(
                f"[ENHANCED_DEBUG] No active/injected skills, using all "
                f"{len(self.skill_by_name)} skills from bank"
            )
            skills = list(self.skill_by_name.values())

        return skills

    def _select_step_actor(
        self,
        active_assigned_skill_names: set[str] | None,
        *,
        preferred_actor_name: str | None = None,
    ) -> AgentSpec:
        """Override to use enhanced meta dispatcher if enabled."""
        if not self.enable_meta_dispatcher or not self.meta_dispatcher:
            return super()._select_step_actor(
                active_assigned_skill_names,
                preferred_actor_name=preferred_actor_name,
            )

        # Use meta dispatcher for semantic matching
        # Note: This requires task context, which isn't available here
        # For now, fall back to parent implementation
        # TODO: Refactor to pass task context through the call chain
        return super()._select_step_actor(
            active_assigned_skill_names,
            preferred_actor_name=preferred_actor_name,
        )


def enhance_cold_start_skills(
    skills: list[Skill],
    *,
    teacher_trajectories: list[dict[str, Any]] | None = None,
    enhancement_config: EnhancementConfig | None = None,
    enhancement_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Enhance cold start skills with multi-tiered verification.

    All thresholds and parameters are loaded from config - no hardcoded values.

    Returns a dict with:
        - "accepted": List of skills to add to bank (auto-accept + credit-track)
        - "probe_needed": List of (skill, episodes_needed) to quick-probe
        - "rejected": List of rejected skills
        - "report": Verification report dict
    """
    # Load config
    if enhancement_config is None:
        if enhancement_config_path:
            enhancement_config = load_enhancement_config(enhancement_config_path)
        else:
            from sage_mas.enhancement_config_loader import get_default_config
            enhancement_config = get_default_config()

    # Extract verifier parameters from config
    verifier_params = extract_verifier_params(enhancement_config)
    verifier = EnhancedSkillVerifier(**verifier_params)

    results = batch_verify_skills(
        skills,
        teacher_trajectories=teacher_trajectories,
        verifier=verifier,
    )

    # Aggregate results
    accepted = []
    probe_needed = []
    rejected = []

    # Auto-accept tier: directly to bank
    for skill, result in results["auto_accept"]:
        accepted.append(skill)

    # Credit-based tier: also to bank (will be validated online)
    for skill, result in results["credit_based"]:
        accepted.append(skill)

    # Quick-probe tier: need testing first
    for skill, result in results["quick_probe"]:
        probe_needed.append((skill, result.probe_budget))

    # Rejected tier
    for skill, result in results["rejected"]:
        rejected.append(skill)

    # Build report
    report = {
        "total": len(skills),
        "auto_accepted": len(results["auto_accept"]),
        "credit_tracked": len(results["credit_based"]),
        "probe_needed": len(results["quick_probe"]),
        "rejected": len(results["rejected"]),
        "acceptance_rate": len(accepted) / len(skills) if skills else 0.0,
    }

    return {
        "accepted": accepted,
        "probe_needed": probe_needed,
        "rejected": rejected,
        "report": report,
    }




# Example usage patterns
def example_usage():
    """Example of how to use the enhanced components with configuration."""
    from sage_mas.enhancement_config_loader import load_enhancement_config
    from sage_mas.runtime import OpenAIChatBackend
    from sage_mas.serialization import load_agents, load_skills

    # Load enhancement configuration
    config = load_enhancement_config("sage_mas/enhancement_config.yaml")

    # Load existing data
    agents = load_agents("organization.json")
    skills = load_skills("skill_bank.json")
    backend = OpenAIChatBackend(model="gpt-4o-mini", api_key="...")

    # Use enhanced runtime with config
    runtime = EnhancedMASRuntime(
        agents=agents,
        skills=skills,
        backend=backend,
        enhancement_config=config,
        prompt_style="alfworld",
    )

    # Execute with enhanced prompts
    action = runtime.act(
        observation="You are in a kitchen...",
        task="cool some potato and put it in microwave",
        task_family="pick_cool_then_place_in_recep",
        history_steps=[],
        gamefile="trial_xxx.tw-pddl",
    )

    print(f"Action: {action.action}")
    print(f"Token cost: {action.token_cost}")

