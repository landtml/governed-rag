import json
import psycopg, pytest
from governed_rag.config import Settings, calibrated
from governed_rag.db import apply_schema
from governed_rag.text import count_tokens, normalize
from governed_rag.types import Authority, Principal

def test_authority_is_ordered():
    assert Authority.NONE < Authority.ARCHIVE < Authority.CURATED

def test_principal_always_has_public():
    assert Principal().effective() == ["public"]
    assert Principal(frozenset({"support"})).effective() == ["public", "support"]

def test_count_tokens_counts_words_and_punctuation():
    assert count_tokens("Refunds: 14 days.") == 5
    assert count_tokens("") == 0

def test_normalize_folds_case_and_whitespace():
    assert normalize("  Fourteen\n\tDAYS ") == "fourteen days"

def test_settings_defaults_match_the_spec():
    s = Settings()
    assert (s.embed_dim, s.chunk_target_tokens, s.chunk_max_tokens, s.curated_max_tokens) == (1024, 350, 500, 1500)
    assert (s.candidates_per_leg, s.rrf_k, s.rerank_n) == (50, 60, 40)
    assert (s.max_parents, s.parent_cap_tokens, s.evidence_budget_tokens, s.gate_retries) == (8, 1200, 6000, 1)
    assert (s.max_delete_fraction, s.delete_guard_floor) == (0.2, 10)
    assert 0.0 < s.tau < 1.0             # provisional: its value is deliberately not asserted

def test_calibration_file_sets_tau_for_its_reranker_only(tmp_path):
    p = tmp_path / "calibration.json"
    assert calibrated(Settings(), p, "rr-1") == Settings()       # no file: the provisional default stands
    p.write_text(json.dumps({"tau": 0.62, "reranker": "rr-1", "golden_sha256": "x", "cases": 100,
                             "date": "2026-11-10"}))
    assert calibrated(Settings(), p, "rr-1").tau == 0.62
    with pytest.raises(ValueError):
        calibrated(Settings(), p, "rr-2")                        # fitted to another reranker's scores
    p.write_text(json.dumps({"tau": 1.5, "reranker": "rr-1"}))
    with pytest.raises(ValueError):
        calibrated(Settings(), p, "rr-1")

def test_schema_is_idempotent_and_complete(db):
    apply_schema(db); apply_schema(db)
    names = {r["table_name"] for r in db.execute(
        "select table_name from information_schema.tables where table_schema = 'public'")}
    assert {"documents", "chunks", "facts", "query_log"} <= names

def test_curated_rows_require_owner_and_review_date(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute("insert into documents (doc_id, tier, title, body, content_hash, acl) "
                   "values ('kb/x', 'curated', 't', 'b', 'h', '{public}')")
