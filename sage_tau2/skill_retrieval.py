"""Skill retrieval index for τ² (Okapi BM25).

Each step an LLM-generated query searches the skill bank. Matching is
lexical: tool names such as ``resume_line`` are split on underscores so a
natural-language query ("resume the line") can overlap the protocol text.

This is not hardcoded routing (``skill_matches_episode``). There is no
embedding backend.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from sage_tau2.schemas import Tau2Skill


def build_skill_document(skill: Tau2Skill) -> str:
    """Build a searchable text document from a skill.

    Concatenates name, precondition, protocol, and metadata so the index
    can match on any of those fields.
    """
    parts: list[str] = []
    for field in (
        skill.skill_name,
        skill.description,
        skill.precondition,
        skill.expected_effect,
        skill.capability_key,
    ):
        if field:
            parts.append(str(field))
    parts.extend(str(step) for step in (skill.action_protocol or []) if step)
    meta = skill.metadata or {}
    for key in (
        "bug_tags_core",
        "bug_intersection",
        "primary_write",
        "bug_signature",
        "primary_task_family",
        "task_families",
        "write_capability_key",
        "parent_capability_key",
        "intent_cues",
    ):
        val = meta.get(key)
        if isinstance(val, list):
            parts.extend(str(v) for v in val if v)
        elif val:
            parts.append(str(val))
    contract = meta.get("execution_contract") or {}
    parts.extend(str(contract.get(key) or '') for key in ('condition', 'observed_conditions', 'bindings', 'verification'))
    return " ".join(parts)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, splitting snake_case.

    ``resume_line`` becomes ``resume`` and ``line``. Tokens shorter than
    two characters are dropped.
    """
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) >= 2
    ]


class _BM25:
    """Okapi BM25. Same saturation and length norm as the ALFWorld index."""

    def __init__(
        self,
        documents: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_freqs = [Counter(doc) for doc in documents]
        self.doc_len = [len(doc) for doc in documents]
        self.num_docs = len(documents)
        self.avgdl = (
            sum(self.doc_len) / self.num_docs if self.num_docs else 0.0
        )
        self.doc_frequency: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            for term in freqs:
                self.doc_frequency[term] += 1

    def _idf(self, term: str) -> float:
        df = self.doc_frequency.get(term, 0)
        # Always non-negative. A term in every document scores near 0.
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        if not self.num_docs or top_k <= 0 or not query_tokens:
            return []
        # Query-side repeats must not inflate the score.
        terms = list(dict.fromkeys(query_tokens))
        scores: list[tuple[int, float]] = []
        for index, freqs in enumerate(self.doc_freqs):
            score = 0.0
            length_norm = self.k1 * (
                1.0 - self.b + self.b * self.doc_len[index] / max(self.avgdl, 1e-9)
            )
            for term in terms:
                tf = freqs.get(term, 0)
                if tf <= 0:
                    continue
                score += self._idf(term) * (tf * (self.k1 + 1.0)) / (tf + length_norm)
            scores.append((index, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[: max(1, top_k)]


class SkillIndex:
    """BM25 retrieval over a skill bank.

    Usage::

        index = SkillIndex(skills)
        results = index.search("resume the line", top_k=2)
    """

    def __init__(self, skills: list[Tau2Skill]):
        self.skills = list(skills)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.documents = [tokenize(build_skill_document(skill)) for skill in self.skills]
        self._bm25 = _BM25(self.documents)

    def search(
        self,
        query: str,
        top_k: int = 2,
        min_score: float = 0.0,
    ) -> list[Tau2Skill]:
        """Return up to ``top_k`` skills with BM25 score at least ``min_score``.

        Scores are unbounded and not cosine similarities. The default
        ``min_score`` keeps every positive hit and drops zero-overlap
        documents. Raise it only if you have calibrated BM25 scores.
        """
        if not self.skills or not str(query or "").strip() or top_k <= 0:
            return []
        ranked = self._bm25.search(tokenize(query), top_k)
        return [
            self.skills[index]
            for index, score in ranked
            if score > 0.0 and score >= float(min_score)
        ]

    def rebuild(self, skills: list[Tau2Skill]) -> None:
        """Rebuild the index from a new skill list (e.g. after bank update)."""
        self.skills = list(skills)
        self._rebuild_index()
