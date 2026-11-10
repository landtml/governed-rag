from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from psycopg import Connection

from governed_rag.config import Settings
from governed_rag.types import Authority, Candidate, Fact, Principal

# Shared by both legs and get_fact. The text is a constant; every value is a %(name)s parameter.
_FILTER = """d.superseded_by IS NULL
  AND d.acl && %(groups)s::text[]
  AND (%(min_authority)s <= 1 OR (d.tier = 'curated' AND d.review_by >= %(today)s))"""

_AUTHORITY = "CASE WHEN d.tier = 'curated' AND d.review_by >= %(today)s THEN 2 ELSE 1 END AS authority"
_COLUMNS = f"c.chunk_id, c.doc_id, c.ord, c.section, c.body, c.index_text, {_AUTHORITY}"

# The query's lexemes that are not common. The search statement ORs them, so a common word cannot
# make the lexical leg rank a large share of the table.
_RARE_LEXEMES = """
SELECT t.lexeme FROM unnest(to_tsvector('english', %(query)s)) AS t(lexeme, positions, weights)
WHERE NOT EXISTS (SELECT 1 FROM common_lexemes k WHERE k.lexeme = t.lexeme) ORDER BY t.lexeme
"""

_LEXICAL = f"""
SELECT {_COLUMNS}
FROM chunks c JOIN documents d USING (doc_id)
WHERE c.tsv @@ %(tsq)s::tsquery AND {_FILTER}
ORDER BY ts_rank_cd(c.tsv, %(tsq)s::tsquery) DESC, c.chunk_id
LIMIT %(n)s
"""

# The inner query orders by distance alone so the HNSW index can serve it; the outer query re-sorts for determinism.
_DENSE = f"""
SELECT * FROM (
  SELECT {_COLUMNS}, c.embedding <=> %(vec)s::vector AS dist
  FROM chunks c JOIN documents d USING (doc_id)
  WHERE {_FILTER}
  ORDER BY c.embedding <=> %(vec)s::vector
  LIMIT %(n)s
) t ORDER BY dist, chunk_id
"""

_FACT = f"""
SELECT f.key, f.value, f.unit
FROM facts f JOIN documents d USING (doc_id)
WHERE f.key = %(key)s AND {_FILTER}
"""


def rrf(rankings: Sequence[Sequence[int]], k: int) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    # dict order is first-seen order and sorted() is stable, so ties keep it
    return sorted(scores.items(), key=lambda pair: -pair[1])


def tsquery_or(lexemes: Sequence[str]) -> str:
    quoted = ["'" + lexeme.replace("\\", "\\\\").replace("'", "''") + "'" for lexeme in lexemes]
    return " | ".join(quoted)


def refresh_common_lexemes(conn: Connection, settings: Settings) -> int:
    with conn.transaction():
        total = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        threshold = max(settings.lexical_common_floor, int(settings.lexical_common_fraction * total))
        conn.execute("DELETE FROM common_lexemes")
        inserted = conn.execute(
            "INSERT INTO common_lexemes (lexeme, ndoc) "
            "SELECT word, ndoc FROM ts_stat('SELECT tsv FROM chunks') WHERE ndoc > %(threshold)s",
            {"threshold": threshold})
        return inserted.rowcount


def _ef_search(settings: Settings) -> int:
    return max(settings.hnsw_ef_search, settings.candidates_per_leg)


def _lexical(conn: Connection, params: dict) -> list[dict]:
    lexemes = [row["lexeme"] for row in conn.execute(_RARE_LEXEMES, params).fetchall()]
    if not lexemes:
        return []
    return conn.execute(_LEXICAL, {**params, "tsq": tsquery_or(lexemes)}).fetchall()


def candidates(conn: Connection, query: str, query_vec: Sequence[float], principal: Principal,
               min_authority: Authority, today: date, settings: Settings) -> list[Candidate]:
    params = {
        "query": query,
        "vec": list(query_vec),
        "groups": principal.effective(),
        "min_authority": int(min_authority),
        "today": today,
        "n": settings.candidates_per_leg,
    }
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        # Plan both legs per call. A cached generic plan cannot see the vector or the tsquery:
        # measured at 500,000 chunks, the dense leg dropped the HNSW index at hnsw.ef_search = 200
        # (5 ms -> 610 ms) and the lexical leg settled at ~350 ms against a 30 ms median.
        conn.execute("SET LOCAL plan_cache_mode = force_custom_plan")
        # hnsw.ef_search below the candidate count makes the scan resume, so it is kept at or above the count.
        conn.execute(f"SET LOCAL hnsw.ef_search = {_ef_search(settings)}")
        # A retrieval query must never start parallel workers, which need /dev/shm.
        conn.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        dense = conn.execute(_DENSE, params).fetchall()
        lexical = _lexical(conn, params)

    rows: dict[int, dict] = {}
    for row in lexical + dense:
        rows.setdefault(row["chunk_id"], row)
    fused = rrf([[r["chunk_id"] for r in lexical], [r["chunk_id"] for r in dense]], settings.rrf_k)

    out: list[Candidate] = []
    for chunk_id, score in fused[: settings.rerank_n]:
        row = rows[chunk_id]
        out.append(Candidate(
            chunk_id=chunk_id, doc_id=row["doc_id"], ord=row["ord"], section=row["section"],
            authority=Authority(row["authority"]), body=row["body"], index_text=row["index_text"],
            score=score))
    return out


def get_fact(conn: Connection, key: str, principal: Principal, today: date) -> Fact | None:
    row = conn.execute(_FACT, {
        "key": key,
        "groups": principal.effective(),
        "min_authority": int(Authority.CURATED),
        "today": today,
    }).fetchone()
    return None if row is None else Fact(row["key"], row["value"], row["unit"])
