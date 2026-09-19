"""On-demand skill retrieval for the ALFWorld Executor.

Each step the Executor may write a capability query from the *current*
observation. That query searches the whole skill bank with BM25 over the
learned operation (capability name and protocol steps). Shared template
words are dropped, so a query that does not name the operation scores
zero and mounts nothing. ``<query>none</query>`` mounts nothing. There is no episode-level frozen catalog and no
second selection pass over a fixed menu.

The scoring backend is pluggable; BM25 is the built-in default because it is
dependency-free and deterministic. An embedding backend can implement the
same ``search(query_tokens, top_k)`` surface and drop in.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from sage_mas.schemas import Skill, SkillStatus

# Skills that may be recalled: admitted to the bank and credit-eligible.
RECALLABLE_STATUSES = {SkillStatus.PROVISIONAL, SkillStatus.VERIFIED}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUERY_TAG_RE = re.compile(r"<query>(.*?)</query>", re.IGNORECASE | re.DOTALL)
_SKIP_QUERIES = frozenset(
    {"none", "no", "no skill", "no skills", "no query", "n/a", "skip"}
)

# Closed-class English words carry no capability signal; with short skill
# documents they would otherwise dominate BM25 scores.
_STOPWORDS = frozenset(
    "a an the and or but if then else of in on at to from for with without "
    "by is are was were be been being do does did have has had i you he she "
    "it we they them him her this that these those there here as so such "
    "into over under again once about between through during before after "
    "above below up down out off own same too very can will just should now "
    "not no nor only also than".split()
)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(str(text or "").lower())
        if token not in _STOPWORDS
    ]


def skill_document_text(skill: Skill) -> str:
    """Operation text a query may match.

    Descriptions, preconditions, and trajectory summaries repeat the same
    template ("find", "go to <source>", "task family matches"). Indexing
    them makes every query hit every skill, and the shorter document wins.
    The learned identity is the capability name plus the protocol steps.
    """
    parts = [
        str(skill.capability_key or "").replace(".", " "),
        " ".join(skill.action_protocol or []),
    ]
    return " ".join(part for part in parts if part)


def discriminative_tokens(documents: list[list[str]]) -> list[list[str]]:
    """Drop tokens shared by more than half the bank.

    ``go`` / ``take`` / ``open`` / ``object`` sit in almost every protocol.
    They are not a capability. A query that only overlaps those words must
    score zero and mount nothing, instead of the shortest skill.
    """
    if len(documents) < 2:
        return documents
    df: Counter[str] = Counter()
    for doc in documents:
        for term in set(doc):
            df[term] += 1
    cutoff = len(documents) / 2.0
    return [
        [term for term in doc if df[term] <= cutoff]
        for doc in documents
    ]


def dedup_recallable_skills(skills: list[Skill]) -> list[Skill]:
    """Filter to recallable statuses and dedup by name.

    Verified siblings win over provisional ones with the same name; ties
    break on higher support_count so the search index is stable.
    """
    status_rank = {SkillStatus.VERIFIED: 0, SkillStatus.PROVISIONAL: 1}
    best: dict[str, Skill] = {}
    for skill in skills:
        if skill.status not in RECALLABLE_STATUSES:
            continue
        name = skill.skill_name.strip()
        if not name:
            continue
        existing = best.get(name)
        if existing is None:
            best[name] = skill
            continue
        key_new = (status_rank.get(skill.status, 2), -int(skill.support_count or 0))
        key_old = (
            status_rank.get(existing.status, 2),
            -int(existing.support_count or 0),
        )
        if key_new < key_old:
            best[name] = skill
    return sorted(best.values(), key=lambda skill: skill.skill_name)


class BM25Index:
    """Okapi BM25 over tokenized skill documents."""

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
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def search(
        self,
        query_tokens: list[str],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` sorted by descending score."""
        if not self.num_docs or top_k <= 0:
            return []
        scores: list[tuple[int, float]] = []
        for index, freqs in enumerate(self.doc_freqs):
            score = 0.0
            length_norm = self.k1 * (
                1.0 - self.b + self.b * self.doc_len[index] / max(self.avgdl, 1e-9)
            )
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if tf <= 0:
                    continue
                score += self._idf(term) * (tf * (self.k1 + 1.0)) / (tf + length_norm)
            scores.append((index, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[: max(1, top_k)]


def build_recall_index(skills: list[Skill]) -> tuple[BM25Index, list[Skill]]:
    """Deduplicate the bank and index only the operations that differ."""
    indexed = dedup_recallable_skills(skills)
    documents = discriminative_tokens(
        [tokenize(skill_document_text(skill)) for skill in indexed]
    )
    return BM25Index(documents), indexed


_OBS_MARK = "Your current observation is:"
_ADM_MARK = "Your admissible actions"


def retrieval_situation(observation: str) -> str:
    """Keep the environment text; drop the action-taking template.

    ``obs['text']`` is the GiGPO prompt. It tells the model to emit
    ``<think>`` and ``<action>`` and lists admissible commands. That text
    must not be copied into the retrieval prompt, or the model treats the
    lookup as an action choice and skips it.
    """
    text = str(observation or "").strip()
    start = text.find(_OBS_MARK)
    if start < 0:
        return text
    body = text[start + len(_OBS_MARK) :].strip()
    end = body.find(_ADM_MARK)
    if end >= 0:
        body = body[:end]
    return body.strip()


def build_recall_query_prompt(task: str, observation: str) -> str:
    """Ask whether this step needs a skill lookup. Not an action prompt."""
    situation = retrieval_situation(observation)
    return "\n".join(
        [
            "Decide whether to look up a learned skill before the next "
            "environment action is chosen.",
            "You are not choosing that action. Do not write <think> or "
            "<action>.",
            "",
            f"Task: {task}",
            "",
            "Current observation:",
            situation,
            "",
            "A skill is a reusable procedure learned from earlier "
            "trajectories. It describes an operation and how that operation "
            "is done, not where a particular object is, and not the place "
            "the object will finally be put.",
            "Decide from this task and this observation whether one would "
            "help. There is no preset list of steps that must or must not "
            "be looked up.",
            "If you want to search, write one short phrase that names the "
            "operation verb and the tool that performs it.",
            "Only those words can match a skill. Do not describe searching "
            "a container, and do not name the place the object will finally "
            "be put.",
            "Do not write an object-finding phrase such as \"find <object>\".",
            "Do not replace the operation verb with a word for the object's "
            "resulting state.",
            "If you do not need a lookup, reply <query>none</query>.",
            "Format: <query>your phrase</query> or <query>none</query>",
            "Output only that one tag.",
        ]
    )


def parse_recall_query(content: str) -> str | None:
    """Extract the query phrase.

    ``None`` means the model declined to retrieve (or the reply could not
    be parsed). Callers must not substitute the task text: that would search
    the bank on steps where the model did not ask.
    """
    text = str(content or "").strip()
    if not text:
        return None
    match = _QUERY_TAG_RE.search(text)
    if match:
        query = " ".join(match.group(1).split())
        if not query or query.lower() in _SKIP_QUERIES:
            return None
        return query
    lowered = text.lower()
    if lowered in _SKIP_QUERIES:
        return None
    # Lenient fallback: a short reply is probably the phrase itself.
    if len(text) <= 120 and "\n" not in text:
        return text
    return None
