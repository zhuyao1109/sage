"""Configuration loader for enhancement modules.

Loads all enhancement configurations from YAML files, eliminating hardcoded values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class EnhancementConfig:
    """Container for all enhancement configurations."""

    def __init__(self, config_dict: dict[str, Any]):
        self.raw = config_dict
        self.enhancement = config_dict.get("enhancement", {})

    def get_skill_context_config(self) -> dict[str, Any]:
        """Get skill context enricher configuration."""
        return self.enhancement.get("skill_context", {})

    def get_task_decomposition_config(self) -> dict[str, Any]:
        """Get task decomposition configuration."""
        return self.enhancement.get("task_decomposition", {})

    def get_verification_config(self) -> dict[str, Any]:
        """Get skill verification configuration."""
        return self.enhancement.get("skill_verification", {})

    def get_meta_dispatcher_config(self) -> dict[str, Any]:
        """Get meta dispatcher configuration."""
        return self.enhancement.get("meta_dispatcher", {})

    def get_runtime_config(self) -> dict[str, Any]:
        """Get runtime integration configuration."""
        return self.enhancement.get("runtime", {})

    def is_enabled(self, component: str) -> bool:
        """Check if a component is enabled."""
        component_map = {
            "skill_context": "skill_context",
            "task_decomposition": "task_decomposition",
            "verification": "skill_verification",
            "meta_dispatcher": "meta_dispatcher",
        }

        config_key = component_map.get(component, component)
        component_config = self.enhancement.get(config_key, {})
        return bool(component_config.get("enabled", False))


def load_enhancement_config(
    config_path: str | Path | None = None,
    base_config: dict[str, Any] | None = None,
) -> EnhancementConfig:
    """
    Load enhancement configuration from file or dict.

    Args:
        config_path: Path to enhancement config YAML
        base_config: Optional base config dict to merge with

    Returns:
        EnhancementConfig object
    """
    config_dict = {}

    # Load from file if provided
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}

    # Merge with base config if provided
    if base_config is not None:
        config_dict = _deep_merge(base_config, config_dict)

    return EnhancementConfig(config_dict)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = dict(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def get_default_config() -> EnhancementConfig:
    """Get default enhancement configuration (all features disabled by default)."""
    default_dict = {
        "enhancement": {
            "skill_context": {
                "enabled": False,
                "max_lookahead": 3,
                "include_rationale": True,
                "loop_threshold": 2,
            },
            "task_decomposition": {
                "enabled": False,
                "mode": "infer_from_skills",
            },
            "skill_verification": {
                "enabled": False,
                "auto_accept": {
                    "teacher_success_rate": 0.85,
                    "min_support_count": 3,
                },
                "quick_probe": {
                    "enabled": True,
                    "num_episodes": 5,
                    "success_rate_threshold": 0.40,
                },
                "credit_based": {
                    "enabled": True,
                    "min_wins": 1,
                },
                "structural": {
                    "require_checks": True,
                    "min_protocol_steps": 3,
                    "max_protocol_steps": 20,
                },
            },
            "meta_dispatcher": {
                "enabled": False,
                "operation_keywords": ["cool", "heat", "clean", "use"],
                "matching_scores": {
                    "task_family_match": 10.0,
                    "operation_match": 5.0,
                    "keyword_match": 0.5,
                },
            },
            "runtime": {
                "use_enhanced_runtime": False,
                "enable_enriched_context": False,
                "enable_task_decomposition": False,
                "enable_meta_dispatcher": False,
            },
        }
    }
    return EnhancementConfig(default_dict)


def merge_with_sage_config(
    sage_config: dict[str, Any],
    enhancement_config: EnhancementConfig,
) -> dict[str, Any]:
    """
    Merge enhancement config into existing SAGE config.

    This allows seamless integration with existing pipeline configs.
    """
    merged = dict(sage_config)

    # Add enhancement section
    if "sage" not in merged:
        merged["sage"] = {}

    merged["sage"]["enhancement"] = enhancement_config.enhancement

    return merged


# Helper functions for component-specific config extraction

def extract_state_hint_patterns(config: EnhancementConfig) -> dict[str, dict[str, Any]]:
    """Extract state hint patterns from config."""
    skill_context = config.get_skill_context_config()
    return skill_context.get("state_hint_patterns", {})


def extract_warning_patterns(config: EnhancementConfig) -> dict[str, Any]:
    """Extract warning patterns from config."""
    skill_context = config.get_skill_context_config()
    return skill_context.get("warning_patterns", {})


def extract_verb_categories(config: EnhancementConfig) -> dict[str, list[str]]:
    """Extract verb categories for protocol decomposition."""
    skill_context = config.get_skill_context_config()
    return skill_context.get("verb_categories", {})


def extract_phase_descriptions(config: EnhancementConfig) -> dict[str, str]:
    """Extract phase descriptions for task decomposition."""
    skill_context = config.get_skill_context_config()
    return skill_context.get("phase_descriptions", {})


def extract_verifier_params(config: EnhancementConfig) -> dict[str, Any]:
    """Extract parameters for EnhancedSkillVerifier."""
    verification = config.get_verification_config()

    auto_accept = verification.get("auto_accept", {})
    quick_probe = verification.get("quick_probe", {})
    structural = verification.get("structural", {})

    return {
        "auto_accept_teacher_sr_threshold": auto_accept.get("teacher_success_rate", 0.85),
        "quick_probe_episodes": quick_probe.get("num_episodes", 5),
        "quick_probe_sr_threshold": quick_probe.get("success_rate_threshold", 0.40),
        "min_protocol_steps": structural.get("min_protocol_steps", 3),
        "max_protocol_steps": structural.get("max_protocol_steps", 20),
        "require_structural_quality": structural.get("require_checks", True),
    }


def extract_dispatcher_params(config: EnhancementConfig) -> dict[str, Any]:
    """Extract parameters for MetaAgentDispatcher."""
    dispatcher = config.get_meta_dispatcher_config()

    return {
        "operation_keywords": dispatcher.get("operation_keywords", []),
        "verb_patterns": dispatcher.get("verb_patterns", []),
        "stop_words": set(dispatcher.get("stop_words", [])),
    }


# Example usage
def example_usage():
    """Example of how to use the config loader."""
    # Load from file
    config = load_enhancement_config("sage_mas/enhancement_config.example.yaml")

    # Check if components are enabled
    if config.is_enabled("skill_context"):
        print("Skill context enrichment is enabled")

    # Get component-specific config
    skill_context_cfg = config.get_skill_context_config()
    max_lookahead = skill_context_cfg.get("max_lookahead", 3)

    # Extract patterns
    state_hints = extract_state_hint_patterns(config)
    warnings = extract_warning_patterns(config)

    # Get verifier parameters
    verifier_params = extract_verifier_params(config)

    print(f"Verifier auto-accept SR: {verifier_params['auto_accept_teacher_sr_threshold']}")
