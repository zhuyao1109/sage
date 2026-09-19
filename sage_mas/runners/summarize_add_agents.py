"""Scan online run segments and print an ADD_AGENT table.

Reads the latest attempt under each ``segment_NNN`` directory:
  - segment_outcome.json
  - distillation/*/run_summary.json
  - distillation/*/organization_edits.json

Example:
  python -m sage_mas.runners.summarize_add_agents \\
    --output logs/sage_mas/online600_noseed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_dir(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda item: item.name)[-1]


def _latest_attempt(segment_dir: Path) -> Path | None:
    attempts = [path for path in segment_dir.iterdir() if path.is_dir()]
    complete = [
        path for path in attempts if (path / "segment_outcome.json").exists()
    ]
    return _latest_dir(complete) or _latest_dir(attempts)


def _latest_distillation(attempt_dir: Path) -> Path | None:
    distill_root = attempt_dir / "distillation"
    if not distill_root.is_dir():
        return None
    runs = [path for path in distill_root.iterdir() if path.is_dir()]
    return _latest_dir(runs)


def _edit_type(edit: dict[str, Any]) -> str:
    return str(edit.get("edit_type") or "").strip().lower()


def _new_agent_name(edit: dict[str, Any]) -> str:
    new_agent = edit.get("new_agent")
    if isinstance(new_agent, dict):
        return str(new_agent.get("name") or "").strip()
    return ""


def _summarize_segment(segment_dir: Path) -> dict[str, Any]:
    attempt = _latest_attempt(segment_dir)
    distill = _latest_distillation(attempt) if attempt else None
    outcome_path = (attempt / "segment_outcome.json") if attempt else None
    summary_path = (distill / "run_summary.json") if distill else None
    edits_path = (distill / "organization_edits.json") if distill else None

    outcome = _read_json(outcome_path) if outcome_path and outcome_path.exists() else {}
    summary = _read_json(summary_path) if summary_path and summary_path.exists() else {}
    edits = _read_json(edits_path) if edits_path and edits_path.exists() else []
    if not isinstance(edits, list):
        edits = []

    add_edits = [edit for edit in edits if _edit_type(edit) == "add_agent"]
    add_names = [name for name in (_new_agent_name(edit) for edit in add_edits) if name]
    dispatch = (outcome.get("segment_metrics") or {}).get("dispatch") or {}
    metrics = outcome.get("segment_metrics") or {}

    return {
        "segment": segment_dir.name,
        "attempt": str(attempt) if attempt else None,
        "distillation": str(distill) if distill else None,
        "status": outcome.get("status"),
        "agent_names": list(outcome.get("agent_names") or []),
        "next_agent_names": list(outcome.get("next_agent_names") or []),
        "probation_agent_names": list(outcome.get("probation_agent_names") or []),
        "verified_skill_count": summary.get("verified_skill_count"),
        "eligible_skill_count": summary.get("eligible_skill_count"),
        "cluster_count": summary.get("cluster_count"),
        "deviated_cluster_count": summary.get("deviated_cluster_count"),
        "add_agent_count": (
            summary.get("add_agent_count")
            if summary.get("add_agent_count") is not None
            else len(add_edits)
        ),
        "organization_edit_count": summary.get("organization_edit_count", len(edits)),
        "edit_types": [_edit_type(edit) or "?" for edit in edits],
        "add_agent_names": add_names,
        "verified_skill_names": list(outcome.get("verified_skill_names") or []),
        "success_rate": metrics.get("success_rate"),
        "specialist_primary_count": dispatch.get("specialist_primary_count"),
        "dispatch_layers": dispatch.get("dispatch_count_by_layer") or {},
    }


def scan_run(output_root: Path) -> list[dict[str, Any]]:
    segments = sorted(
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.name.startswith("segment_")
    )
    return [_summarize_segment(path) for path in segments]


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return ",".join(str(item) for item in value) if value else "-"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "seg",
        "status",
        "verified",
        "eligible",
        "clusters",
        "add",
        "new_agents",
        "roster",
        "next",
        "probation",
        "SR",
        "spec_pri",
        "dispatch",
    ]
    table = [headers]
    for row in rows:
        table.append(
            [
                row["segment"].replace("segment_", ""),
                _fmt(row["status"]),
                _fmt(row["verified_skill_count"]),
                _fmt(row["eligible_skill_count"]),
                _fmt(row["cluster_count"]),
                _fmt(row["add_agent_count"]),
                _fmt(row["add_agent_names"]),
                _fmt(row["agent_names"]),
                _fmt(row["next_agent_names"]),
                _fmt(row["probation_agent_names"]),
                _fmt(row["success_rate"]),
                _fmt(row["specialist_primary_count"]),
                _fmt(row["dispatch_layers"]),
            ]
        )
    widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]
    for index, line in enumerate(table):
        rendered = "  ".join(str(cell).ljust(widths[col]) for col, cell in enumerate(line))
        print(rendered)
        if index == 0:
            print("  ".join("-" * width for width in widths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="Online run directory, e.g. logs/sage_mas/online600_noseed",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the compact table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    if not output_root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {output_root}")
    rows = scan_run(output_root)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    print_table(rows)
    added = sum(int(row.get("add_agent_count") or 0) for row in rows)
    print()
    print(f"segments={len(rows)}  total_add_agent={added}")


if __name__ == "__main__":
    main()
