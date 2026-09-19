"""Opt-in seed skills for τ² ablation (learned-capability layer, not policy).

These are **not** distilled online and are **off by default**. They stand in for
skills that *would* be learned from specific failures (e.g. gift-card vs
certificate totals). Use only for with/without-skill A/B — keep Executor
instructions and domain ``policy.md`` unchanged.

Pass ``seed_skills: true|"auto"|"telecom"|"airline"`` in config, or call
``ensure_seed_skills_in_bank(..., skills=...)`` with an explicit list.
"""

from __future__ import annotations

from pathlib import Path

from sage_tau2.credit import initialize_credit, seed_credit_from_birth_support
from sage_tau2.schemas import SkillStatus, Tau2Skill
from sage_tau2.skill_bank import Tau2SkillBank


# Stable id so journals / ablations can refer to the same skill across runs.
AIRLINE_BALANCE_SPLIT_SKILL_ID = "seed_airline_balance_split_v1"
AIRLINE_PASSENGER_DOB_SKILL_ID = "seed_airline_passenger_dob_v1"
TELECOM_ENABLE_ROAMING_SKILL_ID = "seed_telecom_enable_roaming_v1"
TELECOM_REFUEL_DATA_SKILL_ID = "seed_telecom_refuel_data_v1"


def airline_balance_split_skill() -> Tau2Skill:
    """Report gift-card and certificate balances as two numbers, not one sum."""
    skill = Tau2Skill(
        skill_id=AIRLINE_BALANCE_SPLIT_SKILL_ID,
        skill_name="report_gift_card_and_certificate_balances_separately",
        description=(
            "When the customer asks for gift-card and certificate balances, "
            "compute each payment type separately and state both totals. "
            "Never answer with only the combined sum of gift cards + certificates."
        ),
        precondition=(
            "User asks how much they have on gift cards and/or certificates "
            "(often both in one request)."
        ),
        action_protocol=[
            "get_user_details(user_id=?)",
            "calculate(expression=sum of gift_card amounts only)",
            "calculate(expression=sum of certificate amounts only)",
            "tell_user(gift_card_total=? and certificate_total=? as two separate "
            "numbers; do not report only gift_cards+certificates)",
        ],
        expected_effect=(
            "Customer-visible reply contains both category totals (e.g. gift cards "
            "327 and certificates 1000), matching communicate checks for each figure."
        ),
        capability_key="airline.balances.split_payment_types",
        domain="airline",
        status=SkillStatus.VERIFIED,
        support_count=2,
        evidence_ids=["seed:task14_communicate_pattern"],
        metadata={
            "seed": True,
            "source": "manual_seed_v1",
            "ablation": "balance_split_vs_combined_sum",
            "primary_task_family": "balances",
            "task_families": ["balances"],
        },
    )
    initialize_credit(skill)
    seed_credit_from_birth_support(skill)
    skill.status = SkillStatus.VERIFIED
    return skill


def airline_passenger_dob_skill() -> Tau2Skill:
    """book/update passengers must use schema field ``dob``, not date_of_birth."""
    skill = Tau2Skill(
        skill_id=AIRLINE_PASSENGER_DOB_SKILL_ID,
        skill_name="use_dob_field_on_passenger_writes",
        description=(
            "When calling book_reservation or update_reservation_passengers, "
            "each passenger object must use the field name dob (YYYY-MM-DD). "
            "Do not use date_of_birth or other aliases — the tool schema rejects them."
        ),
        precondition=(
            "About to book a reservation or update passenger details; "
            "passenger birth dates are known from user profile or prior reservation."
        ),
        action_protocol=[
            "copy_dob_from_user_or_existing_reservation_passengers",
            "book_reservation|update_reservation_passengers("
            "passengers=[{first_name, last_name, dob}, ...])",
        ],
        expected_effect=(
            "Passenger write tools accept the call (no Missing required parameter: "
            "'passengers.N.dob' / unexpected date_of_birth)."
        ),
        capability_key="airline.passengers.dob_schema",
        domain="airline",
        status=SkillStatus.VERIFIED,
        support_count=2,
        evidence_ids=["seed:book_reservation_dob_schema"],
        metadata={
            "seed": True,
            "source": "manual_seed_v1",
            "ablation": "passenger_dob_field_name",
            "primary_task_family": "book_reservation",
            "task_families": ["book_reservation", "update_reservation_passengers"],
        },
    )
    initialize_credit(skill)
    seed_credit_from_birth_support(skill)
    skill.status = SkillStatus.VERIFIED
    return skill


def telecom_enable_roaming_skill() -> Tau2Skill:
    """Backend enable_roaming after customer/line lookup."""
    skill = Tau2Skill(
        skill_id=TELECOM_ENABLE_ROAMING_SKILL_ID,
        skill_name="enable_roaming_after_line_lookup",
        description=(
            "When the line must allow data roaming abroad, look up the customer "
            "and line, then call enable_roaming. Do not invent line ids."
        ),
        precondition=(
            "User is abroad / roaming-related mobile data failure; policy requires "
            "backend roaming enabled on the line."
        ),
        action_protocol=[
            "get_customer_by_phone(phone_number=?)",
            "get_details_by_id(id=?)",
            "enable_roaming(customer_id=?, line_id=?)",
        ],
        expected_effect="Line roaming enabled in DB; continue remaining device/policy steps.",
        capability_key="tau2.enable_roaming",
        domain="telecom",
        status=SkillStatus.VERIFIED,
        support_count=2,
        evidence_ids=["seed:telecom_enable_roaming"],
        metadata={
            "seed": True,
            "source": "manual_seed_v1",
            "primary_task_family": "enable_roaming",
            "task_families": ["enable_roaming", "mobile_data_issue"],
            "intent_cues": ["roaming", "abroad"],
        },
    )
    initialize_credit(skill)
    seed_credit_from_birth_support(skill)
    skill.status = SkillStatus.VERIFIED
    return skill


def telecom_refuel_data_skill() -> Tau2Skill:
    """Backend refuel_data when usage exceeds plan limit."""
    skill = Tau2Skill(
        skill_id=TELECOM_REFUEL_DATA_SKILL_ID,
        skill_name="refuel_data_when_usage_exceeded",
        description=(
            "When mobile data fails because usage exceeded the plan limit, look up "
            "the customer/line, confirm usage, then call refuel_data."
        ),
        precondition=(
            "Policy / line details indicate data usage exceeded; connectivity requires "
            "refueling data on the line."
        ),
        action_protocol=[
            "get_customer_by_phone(phone_number=?)",
            "get_details_by_id(id=?)",
            "get_data_usage(customer_id=?, line_id=?)",
            "refuel_data(customer_id=?, line_id=?)",
        ],
        expected_effect="Additional data applied on the line; continue remaining troubleshooting.",
        capability_key="tau2.refuel_data",
        domain="telecom",
        status=SkillStatus.VERIFIED,
        support_count=2,
        evidence_ids=["seed:telecom_refuel_data"],
        metadata={
            "seed": True,
            "source": "manual_seed_v1",
            "primary_task_family": "refuel_data",
            "task_families": ["refuel_data", "mobile_data_issue", "mms_issue"],
            "intent_cues": ["data_usage", "refuel"],
        },
    )
    initialize_credit(skill)
    seed_credit_from_birth_support(skill)
    skill.status = SkillStatus.VERIFIED
    return skill


def default_airline_seed_skills() -> list[Tau2Skill]:
    return [airline_balance_split_skill(), airline_passenger_dob_skill()]


def default_telecom_seed_skills() -> list[Tau2Skill]:
    return [telecom_enable_roaming_skill(), telecom_refuel_data_skill()]


def resolve_seed_skills(
    spec: bool | str | list[Tau2Skill] | None,
    *,
    domain: str = "",
) -> list[Tau2Skill]:
    """Map config ``seed_skills`` to concrete skill objects.

    ``True`` / ``"auto"`` → domain defaults; ``"telecom"`` / ``"airline"`` → that
    pack; ``False`` / ``None`` / ``""`` → none.
    """
    if spec is None or spec is False or spec == "":
        return []
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, str):
        raw = spec.strip()
        # Path form: load a seed bank JSON (e.g. a bootstrap seed distilled by
        # a previous probe run). Resolved as given (CWD-relative), then relative
        # to the repo root (parent of the sage_tau2 package).
        if raw.lower().endswith(".json") or "/" in raw:
            path = Path(raw).expanduser()
            if not path.exists():
                # Fallback: repo-root relative, tolerating leading ../ segments
                # (configs are usually referenced from the tau2-bench CWD).
                repo_root = Path(__file__).resolve().parent.parent
                stripped = raw
                while stripped.startswith("../") or stripped.startswith("./"):
                    stripped = stripped[3:] if stripped.startswith("../") else stripped[2:]
                alt = repo_root / stripped
                if alt.exists():
                    path = alt
            if not path.exists():
                raise FileNotFoundError(f"seed_skills path not found: {raw}")
            return list(Tau2SkillBank(path).skills)
        key = raw.lower()
        if key in {"0", "false", "no", "none", "off"}:
            return []
        if key in {"telecom", "telecom-workflow"}:
            return default_telecom_seed_skills()
        if key == "airline":
            return default_airline_seed_skills()
        if key in {"1", "true", "yes", "on", "auto"}:
            dom = str(domain or "").strip().lower()
            if dom.startswith("telecom"):
                return default_telecom_seed_skills()
            if dom == "airline":
                return default_airline_seed_skills()
            return []
        return []
    dom = str(domain or "").strip().lower()
    if dom.startswith("telecom"):
        return default_telecom_seed_skills()
    if dom == "airline":
        return default_airline_seed_skills()
    return []


def write_seed_bank(
    path: str | Path,
    *,
    skills: list[Tau2Skill] | None = None,
) -> Path:
    """Write a skill bank JSON containing only the given seed skills.

    ``skills`` must be provided explicitly — no implicit airline/telecom pack.
    """
    bank_path = Path(path)
    bank = Tau2SkillBank(bank_path)
    bank.skills = []
    for skill in skills or []:
        bank.add(skill)
        for existing in bank.skills:
            if existing.skill_id == skill.skill_id or existing.skill_name == skill.skill_name:
                existing.status = SkillStatus.VERIFIED
                existing.metadata = dict(existing.metadata or {})
                existing.metadata["seed"] = True
    bank.save()
    return bank_path


def ensure_seed_skills_in_bank(
    path: str | Path,
    *,
    skills: list[Tau2Skill] | None = None,
) -> Tau2SkillBank:
    """Merge seed skills into an existing bank (by skill_id / protocol).

    No-op when ``skills`` is None or empty — seeds are opt-in only.
    """
    seed_ids = {
        AIRLINE_BALANCE_SPLIT_SKILL_ID,
        AIRLINE_PASSENGER_DOB_SKILL_ID,
        TELECOM_ENABLE_ROAMING_SKILL_ID,
        TELECOM_REFUEL_DATA_SKILL_ID,
    }
    bank = Tau2SkillBank(path)
    for skill in skills or []:
        added = bank.add(skill)
        if str(added.skill_id) in seed_ids or added.metadata.get("seed"):
            added.status = SkillStatus.VERIFIED
            # Birth-seed credit for path-loaded seeds too (manual seed
            # constructors already do this): support-derived prior, otherwise
            # a fresh seed would start with a zero/empty credit snapshot.
            seed_credit_from_birth_support(added)
    bank.save()
    return bank
