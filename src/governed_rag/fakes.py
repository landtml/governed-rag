from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from governed_rag.gates import numbers
from governed_rag.types import Citation, Draft, Evidence, Verdict

STOPWORDS = frozenset("""a an and any anyone are as at be by can do does for from has have how i in is it
must of on or our the to was we what when which who why with you""".split())

FAKE_TAU = 0.30

_TOKEN = re.compile(r"\w+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKER = re.compile(r"\[E\d+\]")


def content_words(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS}


class FakeEmbedder:
    model_id = "fake-bow-v1"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        v = [0.0] * self.dim
        for token in tokens:
            v[int(hashlib.sha1(token.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
        if not tokens:
            v[0] = 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v]


class FakeReranker:
    model_id = "fake-overlap-v1"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = content_words(query)
        if not q:
            return [0.0] * len(passages)
        return [len(q & content_words(p)) / len(q) for p in passages]


class ScriptedGenerator:
    model_id = "scripted"

    def __init__(self, drafts: list[Draft]):
        self._drafts = list(drafts)
        self.calls: list[tuple[str, tuple[Evidence, ...], str | None]] = []

    def generate(self, question: str, evidence: Sequence[Evidence],
                 feedback: str | None = None) -> Draft:
        self.calls.append((question, tuple(evidence), feedback))
        return self._drafts.pop(0)


class ExtractiveGenerator:
    model_id = "extractive-v1"

    def generate(self, question: str, evidence: Sequence[Evidence],
                 feedback: str | None = None) -> Draft:
        best_score, best_eid, best_sentence = 0.0, None, None
        reranker = FakeReranker()
        for ev in evidence:
            for sentence in _sentences(ev.text):
                score = reranker.score(question, [sentence])[0]
                if score > best_score:
                    best_score, best_eid, best_sentence = score, ev.eid, sentence
        if best_sentence is None:
            return Draft(abstain=True)
        return Draft(False, f"{best_sentence} [{best_eid}]", (Citation(best_eid, best_sentence),))


class FakeJudge:
    model_id = "fake-judge-v1"

    def judge(self, question: str, answer: str, evidence: Sequence[Evidence]) -> Verdict:
        text = _MARKER.sub(" ", answer)
        evidence_text = " ".join(e.text for e in evidence)
        words = content_words(text)
        covered = len(words & content_words(evidence_text))
        supported = bool(words) and numbers(text) <= numbers(evidence_text) and 2 * covered >= len(words)
        return Verdict(True, ()) if supported else Verdict(False, (answer,))


def _sentences(text: str) -> list[str]:
    pieces = (s.strip() for line in text.split("\n") for s in _SENTENCE_END.split(line))
    return [p for p in pieces if p]
