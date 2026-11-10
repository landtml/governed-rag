from datetime import date
import pytest
from helpers import fake_settings, seed_doc
from governed_rag.config import Settings
from governed_rag.fakes import FakeEmbedder
from governed_rag.retrieve import (_ef_search, _lexical, candidates, get_fact, refresh_common_lexemes, rrf,
                                 tsquery_or)
from governed_rag.types import Authority, Fact, Principal

TODAY = date(2026, 11, 10)
PUBLIC, SUPPORT = Principal(), Principal(frozenset({"support"}))

@pytest.fixture(autouse=True)
def _no_common_lexemes_left_over(db):                             # the db fixture does not clear this table
    db.execute("DELETE FROM common_lexemes")
    yield
    db.execute("DELETE FROM common_lexemes")

def cand(db, q, principal=PUBLIC, min_authority=Authority.ARCHIVE, today=TODAY):
    return candidates(db, q, FakeEmbedder().embed([q])[0], principal, min_authority, today, Settings())

def cand_with(db, q, settings):
    return candidates(db, q, FakeEmbedder().embed([q])[0], PUBLIC, Authority.ARCHIVE, TODAY, settings)

def lexical(db, q, settings):
    params = {"query": q, "groups": PUBLIC.effective(), "min_authority": int(Authority.ARCHIVE),
              "today": TODAY, "n": settings.candidates_per_leg}
    return _lexical(db, params)

def lex_ids(rows): return [r["doc_id"] for r in rows]

def common(db): return {r["lexeme"] for r in db.execute("SELECT lexeme FROM common_lexemes")}

def ids(cs): return [c.doc_id for c in cs]

def test_rrf_fuses_ranks():
    assert [i for i, _ in rrf([[1, 2, 3], [3, 1]], k=60)] == [1, 3, 2]
    assert rrf([[7]], k=60) == [(7, 1 / 61)]

def test_acl_is_a_prefilter(db):
    seed_doc(db, "archive/secret", chunks=["duplicate charge after plan switch"], acl=("finance",))
    seed_doc(db, "archive/open", chunks=["charge questions"])
    assert ids(cand(db, "duplicate charge plan switch")) == ["archive/open"]
    assert ids(cand(db, "duplicate charge plan switch", Principal(frozenset({"finance"}))))[0] == "archive/secret"

def test_superseded_docs_are_invisible(db):
    seed_doc(db, "archive/old", chunks=["refund window 30 days"], superseded_by="kb/policy/refunds")
    assert cand(db, "refund window") == []

def test_expired_curated_is_served_as_archive(db):
    seed_doc(db, "kb/fresh", tier="curated", chunks=["refund window 14 days"])
    seed_doc(db, "kb/stale", tier="curated", chunks=["refund window 10 days"], review_by=date(2026, 11, 9))
    assert {c.doc_id: c.authority for c in cand(db, "refund window")} == {
        "kb/fresh": Authority.CURATED, "kb/stale": Authority.ARCHIVE}
    assert ids(cand(db, "refund window", min_authority=Authority.CURATED)) == ["kb/fresh"]

def test_candidate_carries_chunk_fields(db):
    seed_doc(db, "archive/h", chunks=[("Laptops", "Laptop access needs a ticket.")])
    (c,) = cand(db, "laptop access")
    assert (c.ord, c.section, c.body, c.index_text) == (0, "Laptops", "Laptop access needs a ticket.",
                                                        "Laptop access needs a ticket.")
    assert c.score > 0 and isinstance(c.chunk_id, int)

def test_lexical_leg_finds_a_rare_identifier(db):
    seed_doc(db, "archive/t", chunks=["Error ERR-7731 on invoice export"])
    for i in range(5):
        seed_doc(db, f"archive/n{i}", chunks=[f"unrelated note {i}"])
    assert ids(cand(db, "ERR-7731"))[0] == "archive/t"

def test_result_is_capped_at_rerank_n(db):
    for i in range(45):
        seed_doc(db, f"archive/d{i}", chunks=["laptop access"])
    assert len(cand(db, "laptop access")) == 40

def test_degenerate_queries_do_not_error(db):                    # Review Focus 1
    seed_doc(db, "archive/a", chunks=["hello"])
    for q in ["", "?!", "the of a"]:
        assert isinstance(cand(db, q), list)

def test_empty_group_principal_sees_only_public(db):             # Review Focus 4
    seed_doc(db, "archive/pub", chunks=["laptop access"])
    seed_doc(db, "archive/priv", chunks=["laptop access"], acl=("it",))
    assert ids(cand(db, "laptop access", Principal(frozenset()))) == ["archive/pub"]

def test_get_fact_respects_acl_and_expiry(db):
    seed_doc(db, "kb/p", tier="curated", chunks=["x"], facts={"refund_window_days": "14"}, acl=("support",))
    assert get_fact(db, "refund_window_days", SUPPORT, TODAY) == Fact("refund_window_days", "14", None)
    assert get_fact(db, "refund_window_days", PUBLIC, TODAY) is None
    assert get_fact(db, "refund_window_days", SUPPORT, date(2100, 1, 1)) is None
    assert get_fact(db, "nope", SUPPORT, TODAY) is None


def test_neither_leg_runs_on_a_generic_plan(db):
    # At 500,000 chunks the cached generic plans were far slower than per-call plans once
    # Postgres switched to them after about ten calls on one connection: the dense leg dropped
    # the HNSW index at hnsw.ef_search = 200 (5 ms -> 610 ms), the lexical leg went to ~350 ms.
    for i in range(14):
        cand(db, f"question number {i}")
    plans = db.execute("SELECT statement, generic_plans FROM pg_prepared_statements "
                       "WHERE statement LIKE '%%FROM chunks c JOIN documents d%%'").fetchall()
    assert any("c.embedding <=>" in p["statement"] for p in plans), "dense statement not prepared"
    assert any("c.tsv @@" in p["statement"] for p in plans), "lexical statement not prepared"
    assert [p["generic_plans"] for p in plans] == [0] * len(plans)


def test_retrieval_settings_defaults():
    s = Settings()
    assert (s.hnsw_ef_search, s.lexical_common_fraction, s.lexical_common_floor) == (200, 0.02, 1000)

def test_tsquery_or_quotes_each_lexeme():
    assert tsquery_or(["refund", "o'neil", "a&b"]) == "'refund' | 'o''neil' | 'a&b'"
    assert tsquery_or(["a\\b"]) == "'a\\\\b'"
    assert tsquery_or([]) == ""

def test_tsquery_or_output_is_a_valid_tsquery(db):
    (text,) = db.execute("SELECT %s::tsquery::text AS q", (tsquery_or(["o'neil", "a&b", "a\\b"]),)).fetchone().values()
    assert text == "'o''neil' | 'a&b' | 'a\\\\b'"

def test_lexeme_with_ampersand_or_quote_in_query_does_not_raise(db):
    seed_doc(db, "archive/oneil", chunks=["refund owed to O'Neil and team"])
    assert "archive/oneil" in ids(cand(db, "refund for O'Neil & co"))
    assert "archive/oneil" in ids(cand(db, "O'Neil & refund & a&b"))

def test_refresh_marks_only_lexemes_in_many_chunks_as_common(db):
    for i, word in enumerate(["nest", "perch", "roost", "eggs"]):
        seed_doc(db, f"archive/k{i}", chunks=[f"kestrel {word}"])
    seed_doc(db, "archive/z", chunks=["zeppelin hangar"])
    s = fake_settings(lexical_common_floor=2, lexical_common_fraction=0.0)
    assert refresh_common_lexemes(db, s) == 1
    assert common(db) == {"kestrel"}
    assert lex_ids(lexical(db, "kestrel zeppelin", s)) == ["archive/z"]
    assert lex_ids(lexical(db, "kestrel", s)) == []
    assert set(ids(cand_with(db, "kestrel", s))) == {"archive/k0", "archive/k1", "archive/k2", "archive/k3", "archive/z"}

def test_default_floor_keeps_small_corpora_on_todays_lexical_behaviour(db):
    for i, word in enumerate(["nest", "perch", "roost", "eggs"]):
        seed_doc(db, f"archive/k{i}", chunks=[f"kestrel {word}"])
    assert refresh_common_lexemes(db, Settings()) == 0
    assert common(db) == set()
    assert sorted(lex_ids(lexical(db, "kestrel", Settings()))) == ["archive/k0", "archive/k1", "archive/k2", "archive/k3"]

def test_ef_search_is_at_least_the_candidate_count():
    assert _ef_search(Settings()) == 200
    assert _ef_search(fake_settings(hnsw_ef_search=40)) == 50
    assert _ef_search(fake_settings(candidates_per_leg=300)) == 300
