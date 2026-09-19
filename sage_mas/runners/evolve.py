"""CLI for offline SAGE-MAS evolution and shadow decisions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from sage_mas.pipeline import SageEvolutionPipeline
from sage_mas.schemas import ShadowMetrics
from sage_mas.serialization import write_json
from sage_mas.shadow_evaluation import ShadowEvaluator


def _run_evolution(args: argparse.Namespace) -> None:
    pipeline = SageEvolutionPipeline.from_yaml(args.config)
    artifacts = pipeline.run(
        trajectory_path=args.trajectories,
        output_root=args.output,
        active_organization_path=args.active_organization,
    )
    print(json.dumps(asdict(artifacts), indent=2))


def _run_shadow(args: argparse.Namespace) -> None:
    with Path(args.metrics).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    old = ShadowMetrics(**payload["old"])
    new = ShadowMetrics(**payload["new"])
    evaluator = ShadowEvaluator(
        cost_weight=args.cost_weight,
        significance_threshold=args.significance_threshold,
    )
    decision = evaluator.decide(old, new)
    write_json(args.output, decision)
    organization_output = None
    if args.old_organization or args.candidate_organization or args.active_output:
        if not all(
            [args.old_organization, args.candidate_organization, args.active_output]
        ):
            raise ValueError(
                "--old-organization, --candidate-organization, and "
                "--active-output must be supplied together"
            )
        selected_path = (
            args.candidate_organization if decision.accepted else args.old_organization
        )
        with Path(selected_path).open("r", encoding="utf-8") as f:
            selected_organization = json.load(f)
        write_json(args.active_output, selected_organization)
        organization_output = args.active_output
    print(
        json.dumps(
            {
                "accepted": decision.accepted,
                "reason": decision.reason,
                "active_organization": organization_output,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAGE-MAS prototype utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evolve = subparsers.add_parser("evolve", help="Run offline skill and organization evolution")
    evolve.add_argument("--config", required=True)
    evolve.add_argument("--trajectories", required=True)
    evolve.add_argument("--active-organization")
    evolve.add_argument("--output", default="logs/sage_mas/evolution")
    evolve.set_defaults(func=_run_evolution)

    shadow = subparsers.add_parser("shadow", help="Accept or roll back a shadow organization")
    shadow.add_argument("--metrics", required=True)
    shadow.add_argument("--output", required=True)
    shadow.add_argument("--cost-weight", type=float, default=0.1)
    shadow.add_argument("--significance-threshold", type=float, default=0.02)
    shadow.add_argument("--old-organization")
    shadow.add_argument("--candidate-organization")
    shadow.add_argument("--active-output")
    shadow.set_defaults(func=_run_shadow)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
