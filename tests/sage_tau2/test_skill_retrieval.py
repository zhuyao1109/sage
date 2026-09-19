"""BM25 skill index: snake_case overlap and term saturation."""

from __future__ import annotations

import unittest

from sage_tau2.schemas import Tau2Skill
from sage_tau2.skill_retrieval import SkillIndex, tokenize


def _skill(name: str, protocol: list[str], **kwargs: str) -> Tau2Skill:
    return Tau2Skill(
        skill_name=name,
        description=kwargs.get("description", name),
        precondition=kwargs.get("precondition", ""),
        action_protocol=protocol,
        expected_effect=kwargs.get("expected_effect", ""),
        capability_key=kwargs.get("capability_key", ""),
    )


class SkillIndexBm25Tests(unittest.TestCase):
    def test_snake_case_tool_matches_natural_language_query(self) -> None:
        self.assertEqual(tokenize("resume_line"), ["resume", "line"])
        index = SkillIndex(
            [
                _skill("resume", ["resume_line"], capability_key="line.resume"),
                _skill("refuel", ["refuel_data"], capability_key="data.refuel"),
            ]
        )
        hits = index.search("resume the line", top_k=1)
        self.assertEqual([skill.skill_name for skill in hits], ["resume"])

    def test_repeated_tool_name_does_not_outrank_a_closer_match(self) -> None:
        repeated = ["get_customer_by_id"] * 12
        index = SkillIndex(
            [
                _skill("boilerplate", repeated, description="lookup customer id"),
                _skill(
                    "refuel",
                    ["get_customer_by_id", "refuel_data"],
                    description="refuel data for an exceeded plan",
                    capability_key="data.refuel",
                ),
            ]
        )
        hits = index.search("refuel data for exceeded plan", top_k=1)
        self.assertEqual(hits[0].skill_name, "refuel")

    def test_zero_overlap_is_dropped(self) -> None:
        index = SkillIndex([_skill("resume", ["resume_line"])])
        self.assertEqual(index.search("unrelated billing dispute", top_k=2), [])
