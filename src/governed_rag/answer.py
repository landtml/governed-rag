from __future__ import annotations

import time
from datetime import date

from psycopg import Connection
from psycopg.types.json import Jsonb

from governed_rag.config import Settings
from governed_rag.evidence import search
from governed_rag.gates import authority_of, check
from governed_rag.types import Answer, Authority, Embedder, Generator, Principal, Reranker, SearchResult


def answer(conn: Connection, question: str, principal: Principal, *, embedder: Embedder,
           reranker: Reranker, generator: Generator, settings: Settings, today: date,
           min_authority: Authority = Authority.ARCHIVE) -> Answer:
    started = time.perf_counter()
    result = search(conn, question, principal, embedder=embedder, reranker=reranker,
                    settings=settings, today=today, min_authority=min_authority)

    draft, reason, gate_failures = None, result.abstain_reason, ()
    if reason is None:
        draft, reason, gate_failures = _attempts(generator, question, result, min_authority,
                                                 settings.gate_retries)

    if draft is None:
        out = Answer("", (), Authority.NONE, result.evidence, reason, gate_failures, result.top_score)
    else:
        out = Answer(draft.answer, draft.citations, authority_of(draft, result.evidence),
                     result.evidence, None, gate_failures, result.top_score)

    latency_ms = int((time.perf_counter() - started) * 1000)
    with conn.transaction():
        conn.execute(
            "INSERT INTO query_log (question, groups, min_authority, evidence, authority, "
            "abstain_reason, gate_failures, latency_ms, versions) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (question, principal.effective(), int(min_authority),
             Jsonb([{"eid": e.eid, "doc_id": e.doc_id, "authority": int(e.authority), "score": e.score}
                    for e in result.evidence]),
             int(out.authority), out.abstain_reason,
             Jsonb([list(f) for f in gate_failures]), latency_ms,
             Jsonb({"embedder": embedder.model_id, "reranker": reranker.model_id,
                    "generator": generator.model_id, "tau": settings.tau})))
    return out


def _attempts(generator: Generator, question: str, result: SearchResult,
              min_authority: Authority, retries: int):
    """Return (passing draft or None, abstain reason or None, gate failures per attempt)."""
    failures: list[tuple[str, ...]] = []
    feedback: str | None = None
    for _ in range(1 + retries):
        try:
            draft = generator.generate(question, result.evidence, feedback)
        except Exception:
            return None, "generator_error", tuple(failures)
        if draft.abstain:
            return None, "model_abstained", tuple(failures)
        problems = check(draft, result.evidence, min_authority)
        failures.append(tuple(problems))
        if not problems:
            return draft, None, tuple(failures)
        feedback = "; ".join(problems)
    return None, "unverifiable", tuple(failures)
