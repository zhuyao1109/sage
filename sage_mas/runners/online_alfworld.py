"""Run online segmented SAGE-MAS evolution on ALFWorld (play → distill → evolve)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from sage_mas.online_evolution import OnlineAlfWorldEvolution


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--sage-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-games", type=int)
    parser.add_argument("--segment-size", type=int)
    parser.add_argument(
        "--collection-dataset",
        choices=["train", "valid_seen", "valid_unseen"],
    )
    parser.add_argument(
        "--game-selection",
        choices=["first", "hard", "shuffle", "family_proportional"],
    )
    parser.add_argument("--max-skills-per-round", type=int)
    parser.add_argument("--max-advisors", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--executor-model",
        help="Override sage.online.executor_model (acting / teacher play LLM).",
    )
    parser.add_argument(
        "--specialist-model",
        help=(
            "Override sage.online.specialist_model "
            "(non-Executor agents during train play, e.g. gemini-2.5-flash)."
        ),
    )
    parser.add_argument(
        "--acceptor-model",
        help="Override sage.online.acceptor_model (mini gates + test140).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_config = _load_yaml(args.llm_config)
    sage_config = _load_yaml(args.sage_config)
    online = sage_config.setdefault("sage", {}).setdefault("online", {})
    overrides = {
        "num_games": args.num_games,
        "segment_size": args.segment_size,
        "collection_dataset": args.collection_dataset,
        "game_selection": args.game_selection,
        "max_skills_per_round": args.max_skills_per_round,
        "max_advisors": args.max_advisors,
        "seed": args.seed,
        "executor_model": args.executor_model,
        "specialist_model": args.specialist_model,
        "acceptor_model": args.acceptor_model,
    }
    for key, value in overrides.items():
        if value is not None:
            online[key] = value

    state = OnlineAlfWorldEvolution(
        llm_config=llm_config,
        sage_config=sage_config,
        output_root=args.output,
    ).run()
    summary = state.get("summary") or {}
    print(
        json.dumps(
            {
                "protocol": summary.get("protocol"),
                "success_rate": summary.get("success_rate"),
                "wins": summary.get("wins"),
                "num_games": summary.get("num_games"),
                "num_segments": summary.get("num_segments"),
                "final_agent_count": summary.get("final_agent_count"),
                "by_family": summary.get("by_family"),
                "segments": summary.get("segments"),
                "trajectories_jsonl": summary.get("trajectories_jsonl"),
                "skill_bank": summary.get("skill_bank"),
                "output": str(Path(args.output) / "online_summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
