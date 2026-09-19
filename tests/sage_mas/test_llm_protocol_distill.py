"""Tests for LLM-proposed skill protocols and their grounding verification.

Unit tests (parse / grounding / proposer) deliberately use made-up verbs
and entities ("blorp", "wibbox", "zapper") instead of ALFWorld vocabulary.
The module under test must stay grammar-free and environment-agnostic; if
anyone later hardcodes ALFWorld verb rules into it, these tests must fail.

Only the distiller-overlay integration tests use ALFWorld-flavored
fixtures, because ``EnrichedSkillDistiller``'s seed clustering keys off
environment capability markers — that coupling is the seed layer's, not
the proposer's.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from sage_mas.enriched_skill_distiller import EnrichedSkillDistiller
from sage_mas.llm_protocol_distill import (
    LLMProtocolProposer,
    LLMProtocolProposal,
    parse_proposal,
    verify_proposal_grounding,
)
from sage_mas.schemas import AtomicOp, AtomicStep
from sage_mas.skill_distiller import DistillationConfig
from sage_mas.trajectory_adapter import AlfWorldTrajectoryAdapter


@dataclass
class _FakeResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _FakeBackend:
    """Scripted backend: replays the same response, or raises on demand."""

    def __init__(self, responses: list[str] | None = None, error: Exception | None = None):
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt, user_prompt, max_completion_tokens=None):
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return _FakeResult(self.responses[0] if self.responses else "")


def _fake_steps(actions: list[str]):
    """Build AtomicStep trajectories from arbitrary (non-ALFWorld) actions.

    A terminal empty-action step is appended because ``trajectory_action_steps``
    drops the final step of a trajectory.
    """
    steps = [
        AtomicStep(
            node_id=f"s{index}",
            atomic_op=AtomicOp.ACT,
            agent="Executor",
            observation="effect confirmed",
            action=action,
            metadata={},
        )
        for index, action in enumerate(actions)
    ]
    steps.append(
        AtomicStep(
            node_id="end",
            atomic_op=AtomicOp.ACT,
            agent="Executor",
            observation="",
            action="",
            metadata={},
        )
    )
    return steps


def _fake_wins(count: int = 3):
    """Synthetic wins in an invented action language: fetch, zap, deposit."""
    return [
        _fake_steps(
            [
                f"go to wibbox {index}",
                f"open wibbox {index}",
                f"take blorp {index} from wibbox {index}",
                "go to zapper 1",
                f"zap blorp {index} with zapper 1",
                "go to dumpster 1",
                f"move blorp {index} to dumpster 1",
            ]
        )
        for index in range(1, count + 1)
    ]


def _proposal_payload(protocol: list[str], **overrides) -> str:
    payload = {
        "protocol": protocol,
        "search_hint": "The target was usually in a wibbox; check it first.",
        "anti_patterns": ["Do not wander between burners."],
        "object_note": "",
        "rationale": "Acquisition, transform, placement; detours removed.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _heat_trajectory(*, game_id: str) -> dict:
    """ALFWorld-flavored fixture for the distiller-integration tests only."""
    steps = [
        {
            "observation": "You arrive at the fridge.",
            "action": "go to fridge 1",
            "is_action_valid": True,
        },
        {
            "observation": "You open the fridge.",
            "action": "open fridge 1",
            "is_action_valid": True,
        },
        {
            "observation": "You pick up the potato.",
            "action": "take potato 1 from fridge 1",
            "is_action_valid": True,
        },
        {
            "observation": "You arrive at the microwave.",
            "action": "go to microwave 1",
            "is_action_valid": True,
        },
        {
            "observation": "You heat the potato using the microwave.",
            "action": "heat potato 1 with microwave 1",
            "is_action_valid": True,
        },
        {
            "observation": "You arrive at the garbagecan.",
            "action": "go to garbagecan 1",
            "is_action_valid": True,
        },
        {
            "observation": "You move the potato.",
            "action": "move potato 1 to garbagecan 1",
            "is_action_valid": True,
        },
    ]
    return {
        "gamefile": (
            f"/tmp/pick_heat_then_place_in_recep-Potato-None-GarbageCan-"
            f"{game_id}/game.tw-pddl"
        ),
        "task": "heat some potato and put it in garbagecan",
        "won": True,
        "num_steps": len(steps),
        "steps": steps,
    }


def _adapted_wins(count: int = 3):
    return AlfWorldTrajectoryAdapter().adapt_many(
        [_heat_trajectory(game_id=str(index + 1)) for index in range(count)]
    )


class ParseProposalTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        proposal = parse_proposal(_proposal_payload(["go to <source>", "take <object> from <source>"]))
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.protocol[0], "go to <source>")

    def test_fenced_json(self) -> None:
        text = "```json\n" + _proposal_payload(["look"]) + "\n```"
        proposal = parse_proposal(text)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.protocol, ["look"])

    def test_enumeration_prefix_stripped(self) -> None:
        proposal = parse_proposal(_proposal_payload(["1. go to <source>", "2) take <object> from <source>"]))
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.protocol, ["go to <source>", "take <object> from <source>"])

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(parse_proposal("sorry, I cannot help"))

    def test_missing_protocol_returns_none(self) -> None:
        self.assertIsNone(parse_proposal(json.dumps({"search_hint": "x"})))

    def test_malformed_slot_rejected(self) -> None:
        self.assertIsNone(parse_proposal(_proposal_payload(["go to <Source>"])))

    def test_too_many_steps_rejected(self) -> None:
        self.assertIsNone(parse_proposal(_proposal_payload([f"step {i}" for i in range(20)])))


class GroundingTests(unittest.TestCase):
    """Grounding must work on any action language — no ALFWorld vocabulary."""

    def setUp(self) -> None:
        self.wins = _fake_wins(3)

    def test_grounded_protocol_passes(self) -> None:
        proposal = LLMProtocolProposal(
            protocol=[
                "go to <source>",
                "open <source>",
                "take <object> from <source>",
                "go to zapper",
                "zap <object> with zapper",
                "go to <destination>",
                "move <object> to <destination>",
            ]
        )
        report = verify_proposal_grounding(proposal, self.wins)
        self.assertTrue(report["ok"])
        self.assertEqual(report["ungrounded_steps"], [])
        self.assertEqual(report["subsequence_support"], 3)

    def test_hallucinated_step_rejected(self) -> None:
        proposal = LLMProtocolProposal(
            protocol=["take <object> from <source>", "slice <object> with knife"]
        )
        report = verify_proposal_grounding(proposal, self.wins)
        self.assertFalse(report["ok"])
        self.assertIn("slice <object> with knife", report["ungrounded_steps"])

    def test_instance_ids_do_not_break_matching(self) -> None:
        proposal = LLMProtocolProposal(protocol=["zap <object> with zapper"])
        report = verify_proposal_grounding(proposal, self.wins)
        self.assertTrue(report["ok"])
        self.assertEqual(report["per_step_support"][0]["support"], 3)

    def test_subsequence_order_matters(self) -> None:
        proposal = LLMProtocolProposal(
            protocol=["zap <object> with zapper", "take <object> from <source>"]
        )
        report = verify_proposal_grounding(proposal, self.wins)
        # Each step is individually grounded, but no trace has zap before take.
        self.assertTrue(report["ok"])
        self.assertEqual(report["subsequence_support"], 0)


class ProposerTests(unittest.TestCase):
    def test_backend_error_returns_none(self) -> None:
        proposer = LLMProtocolProposer(_FakeBackend(error=RuntimeError("boom")))
        self.assertIsNone(
            proposer.propose(capability="transform.zap", family="zap", wins=_fake_wins())
        )

    def test_parse_failure_returns_none(self) -> None:
        proposer = LLMProtocolProposer(_FakeBackend(responses=["not json"]))
        self.assertIsNone(
            proposer.propose(capability="transform.zap", family="zap", wins=_fake_wins())
        )

    def test_ungrounded_proposal_returns_none(self) -> None:
        proposer = LLMProtocolProposer(
            _FakeBackend(responses=[_proposal_payload(["teleport <object> to <destination>"])])
        )
        self.assertIsNone(
            proposer.propose(capability="transform.zap", family="zap", wins=_fake_wins())
        )
        self.assertEqual(proposer.last_call_stats.get("rejected"), "grounding")

    def test_valid_proposal_returned_with_grounding(self) -> None:
        proposer = LLMProtocolProposer(
            _FakeBackend(
                responses=[
                    _proposal_payload(
                        [
                            "go to <source>",
                            "take <object> from <source>",
                            "go to zapper",
                            "zap <object> with zapper",
                            "go to <destination>",
                            "move <object> to <destination>",
                        ]
                    )
                ]
            )
        )
        proposal = proposer.propose(
            capability="transform.zap", family="zap", wins=_fake_wins()
        )
        self.assertIsNotNone(proposal)
        self.assertTrue(proposal.grounding["ok"])
        self.assertIn("wibbox", proposal.search_hint)

    def test_prompt_contains_family_and_actions(self) -> None:
        backend = _FakeBackend(responses=["not json"])
        proposer = LLMProtocolProposer(backend)
        proposer.propose(capability="transform.zap", family="zap", wins=_fake_wins())
        system_prompt, user_prompt = backend.calls[0]
        self.assertIn("GROUNDING", system_prompt)
        self.assertIn("Task family: zap", user_prompt)
        self.assertIn("take blorp 1 from wibbox 1", user_prompt)


class DistillerOverlayTests(unittest.TestCase):
    """Integration with EnrichedSkillDistiller (seed clustering is env-keyed)."""

    def _distiller(self, backend) -> EnrichedSkillDistiller:
        return EnrichedSkillDistiller(
            DistillationConfig(operation_min_support=1, min_support=1),
            use_llm_rewrite=False,
            min_wins_for_protocol=3,
            protocol_proposer=LLMProtocolProposer(backend),
        )

    def test_valid_proposal_overlays_protocol(self) -> None:
        llm_protocol = [
            "go to <source>",
            "take <object> from <source>",
            "go to microwave",
            "heat <object> with microwave",
            "go to <destination>",
            "move <object> to <destination>",
        ]
        distiller = self._distiller(_FakeBackend(responses=[_proposal_payload(llm_protocol)]))
        skills = distiller.distill(_adapted_wins(3))
        skill = skills[0]
        self.assertEqual(skill.action_protocol, llm_protocol)
        self.assertEqual(skill.metadata.get("protocol_source"), "llm_proposed_grounded_v1")
        self.assertIn("heuristic_protocol_fallback", skill.metadata)
        self.assertTrue(skill.metadata.get("search_hint"))

    def test_ungrounded_proposal_keeps_heuristic(self) -> None:
        distiller = self._distiller(
            _FakeBackend(responses=[_proposal_payload(["teleport <object> to <destination>"])])
        )
        skills = distiller.distill(_adapted_wins(3))
        skill = skills[0]
        self.assertNotEqual(skill.metadata.get("protocol_source"), "llm_proposed_grounded_v1")
        self.assertEqual(skill.metadata.get("llm_protocol"), "unavailable_or_rejected")

    def test_no_proposer_is_noop(self) -> None:
        skills = EnrichedSkillDistiller(
            DistillationConfig(operation_min_support=1, min_support=1),
            use_llm_rewrite=False,
            min_wins_for_protocol=3,
        ).distill(_adapted_wins(3))
        skill = skills[0]
        self.assertNotIn("llm_protocol", skill.metadata)


class PipelineOverlayWiringTests(unittest.TestCase):
    """SageEvolutionPipeline must forward the overlay flag to the distiller."""

    def test_overlay_flag_builds_proposer(self) -> None:
        from sage_mas.pipeline import SageEvolutionPipeline

        pipeline = SageEvolutionPipeline(
            {"sage": {"distillation": {"mode": "heuristic", "llm_protocol_overlay": True}}},
            distillation_backend=_FakeBackend(),
        )
        self.assertIsInstance(
            getattr(pipeline.distiller, "protocol_proposer", None),
            LLMProtocolProposer,
        )

    def test_default_off(self) -> None:
        from sage_mas.pipeline import SageEvolutionPipeline

        pipeline = SageEvolutionPipeline({"sage": {"distillation": {"mode": "heuristic"}}})
        self.assertIsNone(getattr(pipeline.distiller, "protocol_proposer", None))

    def test_overlay_without_backend_raises(self) -> None:
        from sage_mas.pipeline import SageEvolutionPipeline

        with self.assertRaises(ValueError):
            SageEvolutionPipeline(
                {"sage": {"distillation": {"mode": "heuristic", "llm_protocol_overlay": True}}}
            )


if __name__ == "__main__":
    unittest.main()
