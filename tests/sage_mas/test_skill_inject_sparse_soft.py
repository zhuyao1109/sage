"""Unit tests for sparse soft skill attach gates."""

from sage_mas.schemas import Skill, SkillStatus
from sage_mas.skill_inject_sparse_soft import (
    render_soft_skill_block,
    should_attach_sparse_soft,
)


def _hist(*actions: str, invalid_tail: int = 0) -> list[dict]:
    steps = [{"action": action, "is_action_valid": True} for action in actions]
    for i in range(1, invalid_tail + 1):
        if i <= len(steps):
            steps[-i]["is_action_valid"] = False
    return steps


def test_attach_at_episode_start():
    attach, reason = should_attach_sparse_soft(history_steps=[])
    assert attach and reason == "episode_start"


def test_skip_mid_search():
    hist = _hist("go to fridge 1", "open fridge 1", "go to countertop 1")
    attach, reason = should_attach_sparse_soft(history_steps=hist)
    assert not attach and reason == "skip"


def test_attach_after_take():
    hist = _hist("go to fridge 1", "open fridge 1", "take apple 1 from fridge 1")
    attach, reason = should_attach_sparse_soft(history_steps=hist)
    assert attach and reason == "core_stage"


def test_attach_on_stall_repeat():
    hist = _hist("go to fridge 1", "go to fridge 1", "go to fridge 1")
    attach, reason = should_attach_sparse_soft(history_steps=hist)
    assert attach and reason == "stall"


def test_soft_block_has_no_step_cursor():
    skill = Skill(
        skill_name="place_proto",
        description="test",
        precondition="",
        action_protocol=["go to <location>", "take <object>", "put <object>"],
        applicable_atomic_ops=[],
        applicable_task_families=["pick_and_place"],
        status=SkillStatus.VERIFIED,
    )
    text = render_soft_skill_block([skill])
    assert "Skill reference (soft" in text
    assert "Now Step" not in text
    assert "k/N" not in text
    assert "take <target>" in text or "take <object>" in text
    assert "go to <location>" in text


def test_hybrid_brief_every_step_full_at_gate():
    from sage_mas.skill_inject_sparse_soft import (
        render_hybrid_soft_block,
        render_soft_skill_brief,
    )

    skill = Skill(
        skill_name="place_proto",
        description="test",
        precondition="",
        action_protocol=["go to <location>", "take <object>", "put <object>"],
        applicable_atomic_ops=[],
        applicable_task_families=["pick_and_place"],
        status=SkillStatus.VERIFIED,
    )
    brief = render_soft_skill_brief([skill])
    assert "Skill cue (soft" in brief
    assert "→" in brief

    mid = render_hybrid_soft_block(
        [skill],
        history_steps=_hist("go to fridge 1", "open fridge 1"),
    )
    assert "Skill cue (soft" in mid
    assert "Skill reference (soft" not in mid

    gate = render_hybrid_soft_block(
        [skill],
        history_steps=_hist(
            "go to fridge 1",
            "open fridge 1",
            "take apple 1 from fridge 1",
        ),
    )
    assert "Skill cue (soft" in gate
    assert "Skill reference (soft" in gate
    assert "Now Step" not in gate


def _skill_with_prior() -> Skill:
    return Skill(
        skill_name="clean_proto",
        description="test",
        precondition="",
        action_protocol=[
            "go to <source>",
            "take <object> from <source>",
            "go to sinkbasin",
            "clean <object> with sinkbasin",
        ],
        applicable_atomic_ops=[],
        applicable_task_families=["pick_clean_then_place_in_recep"],
        status=SkillStatus.VERIFIED,
        metadata={
            "search_prior": {
                "sources": [
                    {"source": "countertop", "count": 4},
                    {"source": "drawer", "count": 1},
                    {"source": "fridge", "count": 1},
                ],
                "episodes": 6,
                "acquisitions": 6,
            }
        },
    )


def test_full_block_renders_search_prior():
    text = render_soft_skill_block([_skill_with_prior()])
    assert "Search prior" in text
    assert "countertop ×4" in text
    assert "from 6 winning episodes" in text


def test_brief_cue_renders_likely_sources():
    from sage_mas.skill_inject_sparse_soft import render_soft_skill_brief

    brief = render_soft_skill_brief([_skill_with_prior()])
    assert "likely sources: countertop, drawer" in brief


def test_no_prior_no_render_noise():
    skill = Skill(
        skill_name="place_proto",
        description="test",
        precondition="",
        action_protocol=["go to <location>", "take <object>", "put <object>"],
        applicable_atomic_ops=[],
        applicable_task_families=["pick_and_place"],
        status=SkillStatus.VERIFIED,
    )
    from sage_mas.skill_inject_sparse_soft import render_soft_skill_brief

    assert "Search prior" not in render_soft_skill_block([skill])
    assert "likely sources" not in render_soft_skill_brief([skill])


def _skill_with_anti_patterns() -> Skill:
    return Skill(
        skill_name="heat_proto",
        description="test",
        precondition="",
        action_protocol=[
            "go to <source>",
            "take <object> from <source>",
            "go to microwave",
            "heat <object> with microwave",
        ],
        applicable_atomic_ops=[],
        applicable_task_families=["pick_heat_then_place_in_recep"],
        status=SkillStatus.VERIFIED,
        metadata={
            "anti_patterns": [
                "Avoid repeating `open microwave` after no-op / invalid feedback "
                "(Nothing happens / invalid action).",
                "Do not put the object into the microwave before heating; "
                "heat the object while holding it.",
            ]
        },
    )


def test_full_block_renders_semantic_anti_patterns_only():
    text = render_soft_skill_block([_skill_with_anti_patterns()])
    assert "Avoid: Do not put the object into the microwave" in text
    # Mechanical loop-detection entries are filtered out as noise.
    assert "Avoid repeating" not in text


def test_brief_cue_skips_anti_patterns():
    from sage_mas.skill_inject_sparse_soft import render_soft_skill_brief

    brief = render_soft_skill_brief([_skill_with_anti_patterns()])
    assert "Avoid" not in brief
