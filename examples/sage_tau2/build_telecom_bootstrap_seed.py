"""Build a bootstrap seed bank for telecom from skills distilled by prior runs.

The bootstrap-seed path: instead of hand-writing seed protocols, reuse skills
that the SAGE pipeline itself distilled (probe / main runs), so the seeded
knowledge is pipeline-generated end-to-end.

What it does:
  1. From a main run bank, take the best VERIFIED ``tau2.enable_roaming`` skill
     (highest support) — a proven carrier.
  2. From a probe bank (e.g. telecom_refuel_probe_bank2), take the best skill
     whose protocol calls ``refuel_data`` and re-label its capability to
     ``tau2.refuel_data`` (distill may mislabel combo trajectories by an
     extraneous write call).
  3. Assign stable seed ids, mark ``metadata.seed=True`` /
     ``source=bootstrap_distill_v1``, and write a seed bank JSON.

Usage:
  python examples/sage_tau2/build_telecom_bootstrap_seed.py \
    --main-bank logs/sage_tau2/telecom_seg20_x5_coarse/skill_bank.json \
    --probe-bank logs/sage_tau2/telecom_refuel_probe_bank2/skill_bank.json \
    --out sage_tau2/seed_banks/telecom_bootstrap_seed.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROAMING_SEED_ID = "seed_telecom_roaming_bootstrap_v1"
REFUEL_SEED_ID = "seed_telecom_refuel_bootstrap_v1"


def _load_skills(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("skills") or []
    return [s for s in data if isinstance(s, dict)]


def _cap(skill: dict) -> str:
    meta = skill.get("metadata") or {}
    return str(meta.get("organizational_capability_key") or skill.get("capability_key") or "").strip().lower()


def _protocol_text(skill: dict) -> str:
    return " ".join(str(s) for s in (skill.get("action_protocol") or [])).lower()


def _pick_roaming(skills: list[dict]) -> dict | None:
    cands = [
        s
        for s in skills
        if _cap(s) == "tau2.enable_roaming" and s.get("status") == "verified"
    ]
    if not cands:
        cands = [s for s in skills if "enable_roaming" in _protocol_text(s)]
    if not cands:
        return None
    return max(cands, key=lambda s: int(s.get("support_count") or 0))


def _pick_refuel(skills: list[dict]) -> dict | None:
    cands = [s for s in skills if "refuel_data" in _protocol_text(s)]
    if not cands:
        return None
    return max(cands, key=lambda s: int(s.get("support_count") or 0))


def _curate_refuel_seed(skill: dict) -> dict:
    """Re-label a mislabeled combo distill to its refuel_data core.

    The source trajectory solved a pure-refuel task but called the backend
    ``enable_roaming`` extraneously, so distill labeled the capability
    ``tau2.enable_roaming``. As a seed we (a) drop the agent-side roaming
    *write* steps (state-mutating; harmful on refuel tasks where roaming must
    stay untouched), (b) keep the enumerated probe steps — they are the
    valuable line-resolution pattern, (c) keep user-side guide steps (the
    episode succeeded with them), and (d) repair all routing metadata so the
    canonical ``organizational_capability_key`` (metadata.write_capability_key
    first) resolves to ``tau2.refuel_data``.
    """
    skill["action_protocol"] = [
        step
        for step in (skill.get("action_protocol") or [])
        if "enable_roaming" not in str(step).lower()
    ]
    meta = dict(skill.get("metadata") or {})
    atp = [
        step
        for step in (meta.get("agent_tool_protocol") or [])
        if "enable_roaming" not in str(step).lower()
    ]
    if atp:
        meta["agent_tool_protocol"] = atp

    meta["write_capability_key"] = "tau2.refuel_data"
    meta["primary_write"] = "refuel_data"
    meta["primary_task_family"] = "refuel_data"
    meta["parent_capability_key"] = "tau2.refuel_data"
    families = [f for f in (meta.get("task_families") or []) if f != "enable_roaming"]
    if "refuel_data" not in families:
        families.insert(0, "refuel_data")
    meta["task_families"] = families
    meta["intent_cues"] = [
        c for c in (meta.get("intent_cues") or []) if "roaming" not in str(c).lower()
    ]
    # Stale distill bookkeeping (bucket identity): drop so dedup/clustering
    # re-derive from the curated protocol at load.
    for key in (
        "protocol_bucket",
        "protocol_bucket_key",
        "protocol_bucket_write_spine",
    ):
        meta.pop(key, None)
    skill["metadata"] = meta
    return skill


def _as_seed(
    skill: dict,
    *,
    seed_id: str,
    capability: str,
    origin: str,
    skill_name: str | None = None,
) -> dict:
    out = dict(skill)
    out["skill_id"] = seed_id
    out["capability_key"] = capability
    if skill_name:
        out["skill_name"] = skill_name
    out["status"] = "verified"
    # Seed convention (see seed_skills.py): support 2 keeps the seed
    # nominate-able (min_cluster_support=2) while credit still decides its
    # online fate from real usage.
    out["support_count"] = max(int(out.get("support_count") or 0), 2)
    meta = dict(out.get("metadata") or {})
    meta.pop("organizational_capability_key", None)
    # Reset credit: seeds start with birth-seeded credit only (derived from
    # support at load), not the source run's online history.
    for key in ("skill_credit", "utility", "credit_seeded_from_birth"):
        meta.pop(key, None)
    meta["seed"] = True
    meta["source"] = "bootstrap_distill_v1"
    meta["bootstrap_origin"] = origin
    out["metadata"] = meta
    evidence = list(out.get("evidence_ids") or [])
    if "seed:bootstrap" not in evidence:
        evidence.append("seed:bootstrap")
    out["evidence_ids"] = evidence
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-bank", required=True, help="main run skill_bank.json")
    ap.add_argument("--probe-bank", required=True, help="probe run skill_bank.json")
    ap.add_argument("--out", required=True, help="output seed bank JSON path")
    args = ap.parse_args()

    main_skills = _load_skills(Path(args.main_bank))
    probe_skills = _load_skills(Path(args.probe_bank))

    seeds: list[dict] = []
    roaming = _pick_roaming(main_skills)
    if roaming is not None:
        seeds.append(
            _as_seed(
                roaming,
                seed_id=ROAMING_SEED_ID,
                capability="tau2.enable_roaming",
                origin=str(args.main_bank),
            )
        )
        print(f"[seed] roaming  <- support={roaming.get('support_count')} ({args.main_bank})")
    else:
        print("[seed] WARNING: no roaming skill found in main bank")

    refuel = _pick_refuel(probe_skills)
    if refuel is not None:
        seeds.append(
            _as_seed(
                _curate_refuel_seed(refuel),
                seed_id=REFUEL_SEED_ID,
                capability="tau2.refuel_data",
                origin=str(args.probe_bank),
                # Distill mislabels combo trajectories by an extraneous write;
                # give the seed a name that matches its real capability.
                skill_name="enumerate_lines_then_refuel_data",
            )
        )
        print(f"[seed] refuel   <- support={refuel.get('support_count')} ({args.probe_bank})")
    else:
        print("[seed] WARNING: no refuel_data skill found in probe bank")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(seeds, indent=2, ensure_ascii=False))
    print(f"[seed] wrote {len(seeds)} seeds -> {out_path}")


if __name__ == "__main__":
    main()
