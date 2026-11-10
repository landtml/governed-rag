"""Load test for the retrieval read path.

Fills `chunks`, `documents` and `facts` with synthetic clustered vectors (or, with --vectors and
--query-vectors, with the rows of two .npy files), then measures (1) recall of the filtered HNSW dense
leg against exact search and (2) open-loop latency of the full hybrid `candidates()` at a fixed arrival rate.

Destructive: truncates documents, chunks, facts and query_log in the target database.
Only databases whose name starts with rag_load or rag_test are accepted.
Pass the same --chunks to every run on the same database: the cluster centres depend on it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from governed_rag.config import Settings
from governed_rag.db import apply_schema, connect
from governed_rag.retrieve import _DENSE, _ef_search, candidates, refresh_common_lexemes
from governed_rag.types import Authority, Principal

DIM = 1024
GROUPS = 20
VOCAB = 20_000
TOKENS_PER_CHUNK = 30
NOISE = 0.02
BATCH = 10_000  # chunks per COPY transaction
PROGRESS = 50_000
REVIEW_BY = date(2099, 1, 1)
REPO = Path(__file__).resolve().parent.parent

SCENARIOS = [
    ("public_archive", frozenset(), Authority.ARCHIVE),
    ("group_archive", frozenset({"g00"}), Authority.ARCHIVE),
    ("public_curated", frozenset(), Authority.CURATED),
]

_WEIGHTS = 1.0 / (np.arange(VOCAB) + 10)
_WEIGHTS /= _WEIGHTS.sum()
_TOKENS = np.array([f"w{i:05d}" for i in range(VOCAB)])

DOC_COPY = "COPY documents (doc_id, tier, title, body, content_hash, acl, owner, review_by) FROM STDIN WITH (FORMAT BINARY)"
DOC_TYPES = ["text", "text", "text", "text", "text", "text[]", "text", "date"]
CHUNK_COPY = "COPY chunks (doc_id, ord, section, body, index_text, embedding, embed_model) FROM STDIN WITH (FORMAT BINARY)"
CHUNK_TYPES = ["text", "int4", "text", "text", "text", "vector", "text"]


def _centre_draw(n_chunks: int, seed: int):
    rng = np.random.default_rng(seed)
    k = max(64, n_chunks // 2000)
    cents = rng.standard_normal((k, DIM), dtype=np.float32)
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    return rng, cents


def centres(n_chunks: int, seed: int) -> np.ndarray:
    """Cluster centres; the first draw from default_rng(seed), so fill and queries agree."""
    return _centre_draw(n_chunks, seed)[1]


def _noisy(rng, ids, cents):
    v = cents[ids] + NOISE * rng.standard_normal((len(ids), DIM), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _doc_id(i: int) -> tuple[str, str]:
    if i % 50 == 0:
        return f"kb/load/{i:07d}", "curated"
    return f"archive/load/{i:07d}", "archive"


def _doc_batch(lo: int, hi: int, meta) -> list[tuple]:
    m = hi - lo
    public = meta.random(m) < 0.5
    group = meta.integers(GROUPS, size=m)
    rows = []
    for j in range(m):
        doc_id, tier = _doc_id(lo + j)
        curated = tier == "curated"
        acl = ["public"] if public[j] else [f"g{group[j]:02d}"]
        rows.append((doc_id, tier, doc_id, "", "load", acl,
                     "load@example.com" if curated else None,
                     REVIEW_BY if curated else None))
    return rows


def _chunk_batch(lo: int, hi: int, rng, text, cents, vectors=None) -> list[tuple]:
    m = 4 * (hi - lo)
    if vectors is None:
        vecs = _noisy(rng, rng.integers(len(cents), size=m), cents)
    else:  # chunk i of the fill is row i of the vector file; documents hold 4 chunks each
        vecs = np.asarray(vectors[4 * lo:4 * hi], dtype=np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    words = text.choice(VOCAB, size=(m, TOKENS_PER_CHUNK), p=_WEIGHTS)
    bodies = [" ".join(row) for row in _TOKENS[words]]
    return [(_doc_id(lo + r // 4)[0], r % 4, "", body, body, vecs[r], "load")
            for r, body in enumerate(bodies)]


def _copy(conn, sql: str, types: list[str], rows) -> None:
    with conn.cursor() as cur, cur.copy(sql) as copy:
        copy.set_types(types)
        for row in rows:
            copy.write_row(row)


def fill(conn, n_chunks: int, seed: int, vectors=None) -> dict:
    apply_schema(conn)
    conn.execute("TRUNCATE documents, chunks, facts, query_log RESTART IDENTITY CASCADE")
    # Load into unindexed tables; apply_schema below recreates the production indexes.
    conn.execute("DROP INDEX IF EXISTS chunks_vec, chunks_tsv")

    n_docs = n_chunks // 4
    per = BATCH // 4  # documents per batch, i.e. 4 chunks each
    rng, cents = _centre_draw(n_chunks, seed)
    meta = np.random.default_rng([seed, 2])
    text = np.random.default_rng([seed, 3])

    start = time.perf_counter()
    for lo in range(0, n_docs, per):
        with conn.transaction():
            _copy(conn, DOC_COPY, DOC_TYPES, _doc_batch(lo, min(lo + per, n_docs), meta))
    loaded = 0
    for lo in range(0, n_docs, per):
        rows = _chunk_batch(lo, min(lo + per, n_docs), rng, text, cents, vectors)
        with conn.transaction():
            _copy(conn, CHUNK_COPY, CHUNK_TYPES, rows)
        before = loaded
        loaded += len(rows)
        if loaded // PROGRESS != before // PROGRESS:
            print(f"fill: {loaded} of {n_chunks} chunks", file=sys.stderr)
    fill_seconds = time.perf_counter() - start

    start = time.perf_counter()
    conn.execute("SET maintenance_work_mem = '2GB'")
    # A parallel build sizes its shared segment from maintenance_work_mem, and the
    # pgvector container's 64 MB /dev/shm cannot hold 2 GB, so build serially.
    conn.execute("SET max_parallel_maintenance_workers = 0")
    apply_schema(conn)
    conn.execute("ANALYZE documents")
    conn.execute("ANALYZE chunks")
    index_seconds = time.perf_counter() - start

    sizes = conn.execute(
        "SELECT pg_total_relation_size('chunks') AS table_bytes, pg_relation_size('chunks_vec') AS hnsw_bytes"
    ).fetchone()
    return {
        "docs": n_docs,
        "chunks": loaded,
        "clusters": None if vectors is not None else len(cents),
        "fill_seconds": round(fill_seconds, 3),
        "index_seconds": round(index_seconds, 3),
        "table_bytes": sizes["table_bytes"],
        "hnsw_bytes": sizes["hnsw_bytes"],
    }


def make_queries(n_chunks: int, seed: int, count: int, query_vectors=None) -> list[tuple[str, list[float]]]:
    if query_vectors is not None:
        rng = np.random.default_rng(seed + 1)
        words = rng.choice(VOCAB, size=(count, 3), p=_WEIGHTS)
        return [(" ".join(_TOKENS[w]), np.asarray(query_vectors[i], dtype=np.float32).tolist())
                for i, w in enumerate(words)]
    cents = centres(n_chunks, seed)
    rng = np.random.default_rng(seed + 1)
    vecs = _noisy(rng, rng.integers(len(cents), size=count), cents)
    words = rng.choice(VOCAB, size=(count, 3), p=_WEIGHTS)
    return [(" ".join(_TOKENS[w]), vecs[i].tolist()) for i, w in enumerate(words)]


def _params(text, vec, groups, min_authority, today, settings) -> dict:
    # Same keys candidates() passes to _DENSE.
    return {
        "query": text,
        "vec": list(vec),
        "groups": Principal(groups).effective(),
        "min_authority": int(min_authority),
        "today": today,
        "n": settings.candidates_per_leg,
    }


def _set_mode(conn, exact: bool, ef_search: int) -> None:
    if exact:
        conn.execute("SET LOCAL enable_indexscan = off")
        conn.execute("SET LOCAL enable_bitmapscan = off")
        # Custom plans are planned per execution, so the enable_* settings above always apply
        # to the statement text the approximate path also uses.
        conn.execute("SET LOCAL plan_cache_mode = force_custom_plan")
    else:
        # The same two settings candidates() uses for the dense leg.
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        conn.execute("SET LOCAL plan_cache_mode = force_custom_plan")
        # ef_search is candidates()'s effective value, _ef_search(settings), so the recall check matches it.
        conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")


def _ids(conn, params, exact: bool, ef_search: int) -> list[int]:
    with conn.transaction():
        _set_mode(conn, exact, ef_search)
        return [row["chunk_id"] for row in conn.execute(_DENSE, params).fetchall()]


def _plan(conn, params, exact: bool, ef_search: int) -> str:
    with conn.transaction():
        _set_mode(conn, exact, ef_search)
        rows = conn.execute("EXPLAIN " + _DENSE, params).fetchall()
    return "\n".join(row["QUERY PLAN"] for row in rows)


def _pct(values: list[float], p: float):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(math.ceil(p / 100 * len(ordered)), 1) - 1], 3)


def _mean(values: list[float]):
    return sum(values) / len(values) if values else None


def measure_recall(url: str, queries, today: date, settings: Settings) -> dict:
    # Separate connections so the approximate statement's plan cache is not shaped by exact runs.
    approx = connect(url)
    exact = connect(url)
    try:
        _, first_groups, first_min = SCENARIOS[0]
        probe = _params(*queries[0], first_groups, first_min, today, settings)
        ef = _ef_search(settings)
        out: dict = {
            "plan_uses_hnsw": "chunks_vec" in _plan(approx, probe, False, ef),
            "exact_plan_avoids_hnsw": "chunks_vec" not in _plan(exact, probe, True, ef),
        }
        for name, groups, min_authority in SCENARIOS:
            approx_ms, exact_ms, recalls, same = [], [], [], 0
            for text, vec in queries:
                params = _params(text, vec, groups, min_authority, today, settings)
                t = time.perf_counter()
                got = _ids(approx, params, False, ef)
                approx_ms.append((time.perf_counter() - t) * 1000)
                t = time.perf_counter()
                truth = _ids(exact, params, True, ef)
                exact_ms.append((time.perf_counter() - t) * 1000)
                if truth:
                    recalls.append(len(set(truth).intersection(got)) / len(truth))
                    same += len(got) == len(truth)
            out[name] = {
                "recall_mean": round(_mean(recalls), 4) if recalls else None,
                "recall_min": round(min(recalls), 4) if recalls else None,
                "recall_p05": _pct(recalls, 5),
                "near_zero": sum(r < 0.05 for r in recalls),
                "filled_fraction": round(same / len(recalls), 4) if recalls else None,
                "approx_ms_p50": _pct(approx_ms, 50),
                "approx_ms_p95": _pct(approx_ms, 95),
                "exact_ms_p50": _pct(exact_ms, 50),
                "n": len(queries),
            }
        return out
    finally:
        approx.close()
        exact.close()


def measure_latency(url: str, queries, qps: float, duration: float, workers: int,
                    today: date, settings: Settings) -> dict:
    local = threading.local()
    opened: list = []
    lock = threading.Lock()

    def thread_conn():
        if not hasattr(local, "conn"):
            local.conn = connect(url)
            with lock:
                opened.append(local.conn)
        return local.conn

    def task(due: float, i: int):
        name, groups, min_authority = SCENARIOS[i % len(SCENARIOS)]
        text, vec = queries[i % len(queries)]
        try:
            candidates(thread_conn(), text, vec, Principal(groups), min_authority, today, settings)
        except Exception as exc:  # counted as an error, reported once below
            return name, None, time.perf_counter(), exc
        end = time.perf_counter()
        return name, (end - due) * 1000, end, None  # from the scheduled arrival, so queueing counts

    # Each worker gets its connection before the clock starts; the barrier keeps every
    # warm-up task on its own thread.
    barrier = threading.Barrier(workers, timeout=60)

    def warm():
        thread_conn()
        barrier.wait()

    arrivals = int(qps * duration)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(warm) for _ in range(workers)]:
                future.result()
            t0 = time.perf_counter()
            futures = []
            for i in range(arrivals):
                due = t0 + i / qps
                pause = due - time.perf_counter()
                if pause > 0:
                    time.sleep(pause)
                futures.append(pool.submit(task, due, i))
            results = [future.result() for future in futures]
    finally:
        for conn in opened:
            conn.close()

    errors = [r[3] for r in results if r[3] is not None]
    if errors:
        print(f"latency: {len(errors)} errors, first: {errors[0]!r}", file=sys.stderr)
    ok = [(name, ms) for name, ms, _, err in results if err is None]
    wall = max(r[2] for r in results) - t0 if results else 0.0
    all_ms = [ms for _, ms in ok]
    by_scenario = {}
    for name, _, _ in SCENARIOS:
        vals = [ms for n, ms in ok if n == name]
        by_scenario[name] = {"ms_p50": _pct(vals, 50), "ms_p95": _pct(vals, 95), "n": len(vals)}
    return {
        "target_qps": qps,
        "achieved_qps": round(len(ok) / wall, 2) if wall > 0 else 0.0,
        "completed": len(ok),
        "errors": len(errors),
        "ms_p50": _pct(all_ms, 50),
        "ms_p95": _pct(all_ms, 95),
        "ms_p99": _pct(all_ms, 99),
        "ms_max": round(max(all_ms), 3) if all_ms else None,
        "by_scenario": by_scenario,
    }


def _load_vectors(vectors_path: Path, query_path: Path, chunks: int):
    """Return (vectors, query_vectors, problem); problem is a message when the files do not fit the run."""
    try:
        vectors = np.load(vectors_path, mmap_mode="r")
        query_vectors = np.load(query_path)
    except (OSError, ValueError) as exc:
        return None, None, f"cannot read the vector files: {exc}"
    if vectors.ndim != 2 or vectors.shape[1] != DIM or vectors.shape[0] != chunks:
        return None, None, f"--vectors must hold {chunks} rows of {DIM} floats, got shape {vectors.shape}"
    if query_vectors.ndim != 2 or query_vectors.shape[1] != DIM or len(query_vectors) < 1:
        return None, None, f"--query-vectors must hold at least one row of {DIM} floats, got shape {query_vectors.shape}"
    return vectors, query_vectors, None


def _server_info(conn) -> dict:
    # hnsw.ef_search only exists once the backend has loaded pgvector; with --skip-fill
    # nothing on this connection has touched a vector yet.
    conn.execute("SELECT '[0]'::vector")
    row = conn.execute("""
        SELECT current_setting('server_version') AS version,
               (SELECT extversion FROM pg_extension WHERE extname = 'vector') AS pgvector,
               current_setting('shared_buffers') AS shared_buffers,
               current_setting('work_mem') AS work_mem,
               current_setting('hnsw.ef_search') AS hnsw_ef_search,
               (SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_vec') AS hnsw_index
    """).fetchone()
    return dict(row)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database-url", default=Settings().database_url)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--qps", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--today", type=date.fromisoformat, default="2026-11-10")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-fill", action="store_true")
    parser.add_argument("--ef-search", type=int, default=Settings().hnsw_ef_search)
    parser.add_argument("--common-fraction", type=float, default=Settings().lexical_common_fraction)
    parser.add_argument("--vectors", type=Path, default=None)
    parser.add_argument("--query-vectors", type=Path, default=None)
    args = parser.parse_args(argv)

    name = urlparse(args.database_url).path.lstrip("/")
    if not name.startswith(("rag_load", "rag_test")):
        print(f"refusing to run: database {name!r} is not rag_load* or rag_test*; "
              "this script truncates documents, chunks and facts", file=sys.stderr)
        return 2
    if args.chunks < 4 or args.queries < 1 or args.workers < 1 or args.qps <= 0 or args.duration <= 0 \
            or args.ef_search < 1 or not 0.0 <= args.common_fraction <= 1.0:
        print("invalid arguments: need --chunks >= 4, --queries >= 1, --workers >= 1, "
              "--qps > 0, --duration > 0, --ef-search >= 1, 0 <= --common-fraction <= 1", file=sys.stderr)
        return 2
    if (args.vectors is None) != (args.query_vectors is None):
        print("--vectors and --query-vectors must be given together", file=sys.stderr)
        return 2
    vectors = query_vectors = None
    if args.vectors is not None:
        vectors, query_vectors, problem = _load_vectors(args.vectors, args.query_vectors, args.chunks)
        if problem:
            print(problem, file=sys.stderr)
            return 2
    queries_count = args.queries if query_vectors is None else min(args.queries, len(query_vectors))

    settings = dataclasses.replace(Settings(), hnsw_ef_search=args.ef_search,
                                   lexical_common_fraction=args.common_fraction)
    conn = connect(args.database_url)
    try:
        fill_result = None if args.skip_fill else fill(conn, args.chunks, args.seed, vectors)
        apply_schema(conn)
        common_lexemes = refresh_common_lexemes(conn, settings)
        server = _server_info(conn)
    finally:
        conn.close()

    queries = make_queries(args.chunks, args.seed, queries_count, query_vectors)
    recall = measure_recall(args.database_url, queries, args.today, settings)
    latency = measure_latency(args.database_url, queries, args.qps, args.duration,
                              args.workers, args.today, settings)

    result = {
        "chunks": args.chunks,
        "fill": fill_result,
        "server": server,
        "lexical": {
            "common_lexemes": common_lexemes,
            "common_fraction": settings.lexical_common_fraction,
            "common_floor": settings.lexical_common_floor,
        },
        "settings": {
            "candidates_per_leg": settings.candidates_per_leg,
            "rrf_k": settings.rrf_k,
            "rerank_n": settings.rerank_n,
            "hnsw_ef_search": _ef_search(settings),
        },
        "recall": recall,
        "latency": latency,
    }
    out = args.out or REPO / "eval" / f"loadtest-{args.chunks}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
