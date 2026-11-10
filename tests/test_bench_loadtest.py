import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

from governed_rag.fakes import FakeEmbedder

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag_test")
SCENARIOS = {"public_archive", "group_archive", "public_curated"}


def _load_loadtest():
    path = Path(__file__).resolve().parent.parent / "bench" / "loadtest.py"
    spec = importlib.util.spec_from_file_location("loadtest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loadtest_runs_end_to_end(db, tmp_path):
    loadtest = _load_loadtest()
    out = tmp_path / "o.json"
    rc = loadtest.main(["--database-url", TEST_DATABASE_URL, "--chunks", "2000", "--qps", "20",
                        "--duration", "2", "--queries", "5", "--workers", "4", "--out", str(out)])
    assert rc == 0
    result = json.loads(out.read_text())
    assert result["fill"]["chunks"] == 2000
    assert SCENARIOS <= set(result["recall"])
    assert SCENARIOS <= set(result["latency"]["by_scenario"])
    assert result["recall"]["public_archive"]["recall_mean"] >= 0.9
    assert result["latency"]["errors"] == 0
    assert result["latency"]["completed"] == 40
    assert result["lexical"] == {"common_lexemes": 0, "common_fraction": 0.02, "common_floor": 1000}
    assert result["settings"]["hnsw_ef_search"] == 200
    for name in SCENARIOS:
        assert {"recall_p05", "near_zero"} <= set(result["recall"][name])
        assert result["recall"][name]["recall_p05"] <= result["recall"][name]["recall_mean"]

    # A second run on the filled database must work on a connection that has not used pgvector yet.
    rc = loadtest.main(["--database-url", TEST_DATABASE_URL, "--chunks", "2000", "--qps", "20", "--duration", "1",
                        "--queries", "2", "--workers", "2", "--out", str(out), "--skip-fill"])
    assert rc == 0
    result = json.loads(out.read_text())
    assert result["fill"] is None
    assert result["server"]["hnsw_ef_search"] == "40"


def test_refuses_a_database_that_is_not_a_test_database(tmp_path):
    loadtest = _load_loadtest()
    out = tmp_path / "o.json"
    rc = loadtest.main(["--database-url", "postgresql://rag:rag@localhost:5433/rag",
                        "--chunks", "2000", "--out", str(out)])
    assert rc == 2
    assert not out.exists()


def _unit_rows(rng, count: int, centres: np.ndarray) -> np.ndarray:
    rows = centres[rng.integers(len(centres), size=count)] + 0.02 * rng.standard_normal((count, 1024))
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def test_loadtest_fills_from_real_vectors_and_measures_recall(db, tmp_path):
    loadtest = _load_loadtest()
    rng = np.random.default_rng(7)
    centres = _unit_rows(rng, 64, np.eye(64, 1024))
    vectors, queries = _unit_rows(rng, 2000, centres), _unit_rows(rng, 10, centres)
    np.save(tmp_path / "v.npy", vectors)
    np.save(tmp_path / "q.npy", queries)
    out = tmp_path / "o.json"
    rc = loadtest.main(["--database-url", TEST_DATABASE_URL, "--chunks", "2000", "--qps", "20", "--duration", "1",
                        "--queries", "20", "--workers", "2", "--vectors", str(tmp_path / "v.npy"),
                        "--query-vectors", str(tmp_path / "q.npy"), "--out", str(out)])
    assert rc == 0
    result = json.loads(out.read_text())
    assert (result["fill"]["chunks"], result["fill"]["docs"]) == (2000, 500)
    assert result["recall"]["public_archive"]["n"] == 10          # capped at the query file's rows
    assert result["recall"]["public_archive"]["recall_mean"] >= 0.9
    assert result["latency"]["errors"] == 0
    # Chunk 4 is the first chunk of archive/load/0000001: it must hold row 4 of the vector file, normalised.
    stored = db.execute("SELECT embedding::text AS e FROM chunks WHERE doc_id = 'archive/load/0000001' AND ord = 0")
    np.testing.assert_allclose(np.array(json.loads(stored.fetchone()["e"])), vectors[4], atol=1e-3)

    rc = loadtest.main(["--database-url", TEST_DATABASE_URL, "--chunks", "2000", "--qps", "20", "--duration", "1",
                        "--queries", "2", "--workers", "2", "--skip-fill", "--vectors", str(tmp_path / "v.npy"),
                        "--query-vectors", str(tmp_path / "q.npy"), "--out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["fill"] is None


def test_real_vector_options_are_checked_before_anything_runs(tmp_path):
    loadtest = _load_loadtest()
    rng = np.random.default_rng(8)
    np.save(tmp_path / "v.npy", _unit_rows(rng, 1999, np.eye(64, 1024)))
    np.save(tmp_path / "q.npy", _unit_rows(rng, 3, np.eye(64, 1024)))
    out = tmp_path / "o.json"
    base = ["--database-url", TEST_DATABASE_URL, "--chunks", "2000", "--out", str(out)]
    assert loadtest.main(base + ["--vectors", str(tmp_path / "v.npy"), "--query-vectors", str(tmp_path / "q.npy")]) == 2
    assert loadtest.main(base + ["--vectors", str(tmp_path / "v.npy")]) == 2
    assert not out.exists()


def _load_embed_tickets():
    path = Path(__file__).resolve().parent.parent / "bench" / "embed_tickets.py"
    spec = importlib.util.spec_from_file_location("embed_tickets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embed_tickets_writes_distinct_vectors_and_queries(tmp_path, capsys):
    embed = _load_embed_tickets()
    assert embed.main(["--n", "50", "--queries", "5", "--out", str(tmp_path)], embedder=FakeEmbedder()) == 0
    vectors, queries = np.load(tmp_path / "vectors.npy"), np.load(tmp_path / "queries.npy")
    assert vectors.shape == (50, 1024) and queries.shape == (5, 1024)
    assert vectors.dtype == np.float32 and queries.dtype == np.float32
    assert len(np.unique(np.vstack([vectors, queries]), axis=0)) == 55
    summary = json.loads(capsys.readouterr().out)
    assert (summary["vectors"], summary["queries"], summary["distinct_texts"]) == (50, 5, 55)
    assert summary["seconds"] >= 0


def test_embed_tickets_stops_when_the_generator_repeats_itself():
    embed = _load_embed_tickets()
    one_text = SimpleNamespace(ISSUES=(("x",),), _ticket_text=lambda rng, issue: "the same ticket")
    with pytest.raises(RuntimeError, match="only 1 distinct ticket texts"):
        embed.distinct_texts(3, 0, one_text)
