from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import IntEnum
from typing import Protocol


class Authority(IntEnum):
    NONE = 0
    ARCHIVE = 1
    CURATED = 2


@dataclass(frozen=True)
class Principal:
    groups: frozenset[str] = frozenset()

    def effective(self) -> list[str]:
        return sorted(self.groups | {"public"})


@dataclass(frozen=True)
class Fact:
    key: str
    value: str
    unit: str | None = None


@dataclass(frozen=True)
class CuratedDoc:
    doc_id: str
    type: str
    title: str
    owner: str
    review_by: date
    acl: tuple[str, ...]
    body: str
    facts: tuple[Fact, ...] = ()
    supersedes: tuple[str, ...] = ()
    source: str | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class Chunk:
    ord: int
    section: str
    body: str
    index_text: str


@dataclass(frozen=True)
class Candidate:
    chunk_id: int
    doc_id: str
    ord: int
    section: str
    authority: Authority
    body: str
    index_text: str
    score: float


@dataclass(frozen=True)
class Evidence:
    eid: str
    doc_id: str
    title: str
    authority: Authority
    text: str
    score: float


@dataclass(frozen=True)
class SearchResult:
    evidence: tuple[Evidence, ...]
    top_score: float
    abstain_reason: str | None = None


@dataclass(frozen=True)
class Citation:
    eid: str
    quote: str


@dataclass(frozen=True)
class Draft:
    abstain: bool
    answer: str = ""
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    authority: Authority
    evidence: tuple[Evidence, ...]
    abstain_reason: str | None = None
    gate_failures: tuple[tuple[str, ...], ...] = ()
    top_score: float = 0.0


class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return unit-norm vectors, one per input text."""
        ...


class Reranker(Protocol):
    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one score in [0, 1] per passage."""
        ...


class Generator(Protocol):
    model_id: str

    def generate(self, question: str, evidence: Sequence[Evidence],
                 feedback: str | None = None) -> Draft:
        ...


@dataclass(frozen=True)
class Verdict:
    supported: bool
    unsupported: tuple[str, ...] = ()        # the claims the judge could not find support for


class Judge(Protocol):
    model_id: str

    def judge(self, question: str, answer: str, evidence: Sequence[Evidence]) -> Verdict:
        ...
