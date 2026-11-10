import dataclasses
from datetime import date
import pytest
from governed_rag.config import Settings
from governed_rag.fakes import FakeEmbedder
from governed_rag.ingest import DeleteGuard, SyncAborted, SyncReport, sync

CURATED = """---
type: policy
title: Refund policy
owner: billing@example.com
review_by: 2027-05-01
acl: [public]
facts:
  refund_window_days: {value: 14, unit: days}
supersedes: [archive/policies/refund-2023]
---
# Refund policy

Customers can request a refund within 14 days of purchase.
"""
TODAY = date(2026, 11, 10)

class Counting(FakeEmbedder):
    def __init__(self): super().__init__(); self.n = 0
    def embed(self, texts): self.n += len(texts); return super().embed(texts)

class FailingFrom(FakeEmbedder):                                 # raises on its k-th embed call and every later one
    def __init__(self, k): super().__init__(); self.k, self.calls = k, 0
    def embed(self, texts):
        self.calls += 1
        if self.calls >= self.k: raise RuntimeError("embedder down")
        return super().embed(texts)

def write(root, rel, text):
    p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text); return p

def tree(tmp_path):
    kb, ar = tmp_path / "kb", tmp_path / "archive"
    write(kb, "policy/refunds.md", CURATED)
    write(ar, "policies/refund-2023.md", "# Refunds 2023\n\nRefund window is 30 days.")
    write(ar, "tickets/T-1.md", "---\nacl: [support]\ntitle: Ticket 1\n---\nDuplicate charge.")
    return kb, ar

def run(db, kb, ar, e=None): return sync(db, kb, ar, e or FakeEmbedder(), Settings(), TODAY)

def low_floor(): return dataclasses.replace(Settings(), lexical_common_floor=1, lexical_common_fraction=0.0)
def common_of(db): return {r["lexeme"] for r in db.execute("SELECT lexeme FROM common_lexemes")}
def expected_common(db, floor): return {r["word"] for r in db.execute(
    "SELECT word FROM ts_stat('SELECT tsv FROM chunks') WHERE ndoc > %s", (floor,))}

@pytest.fixture(autouse=True)
def _no_common_lexemes_left_over(db):                             # the db fixture does not clear this table
    yield
    db.execute("DELETE FROM common_lexemes")
def count(db, sql): return db.execute(sql).fetchone()["n"]

def test_first_sync_loads_everything(db, tmp_path):
    kb, ar = tree(tmp_path)
    assert run(db, kb, ar) == SyncReport(added=3, updated=0, unchanged=0, deleted=0)
    docs = {r["doc_id"]: r for r in db.execute("select * from documents")}
    assert set(docs) == {"kb/policy/refunds", "archive/policies/refund-2023", "archive/tickets/T-1"}
    old, ticket, kbdoc = docs["archive/policies/refund-2023"], docs["archive/tickets/T-1"], docs["kb/policy/refunds"]
    assert old["superseded_by"] == "kb/policy/refunds" and (old["title"], old["acl"]) == ("Refunds 2023", ["public"])
    assert (ticket["title"], ticket["acl"], ticket["body"]) == ("Ticket 1", ["support"], "Duplicate charge.")
    assert (kbdoc["tier"], kbdoc["owner"], kbdoc["review_by"]) == ("curated", "billing@example.com", date(2027, 5, 1))
    assert db.execute("select value, unit from facts where key = 'refund_window_days'").fetchone() == {
        "value": "14", "unit": "days"}
    assert count(db, "select count(*) n from chunks where embed_model = 'fake-bow-v1'") >= 3

def test_second_sync_is_a_noop_and_embeds_nothing(db, tmp_path):
    kb, ar = tree(tmp_path); run(db, kb, ar)
    e = Counting()
    assert run(db, kb, ar, e) == SyncReport(0, 0, 3, 0) and e.n == 0

def test_edit_replaces_all_chunks_of_that_doc(db, tmp_path):
    kb, ar = tree(tmp_path); run(db, kb, ar)
    write(ar, "tickets/T-1.md", "Blank invoice PDF.")
    assert run(db, kb, ar) == SyncReport(0, 1, 2, 0)
    rows = db.execute("select c.body, d.title, d.acl from chunks c join documents d using (doc_id) "
                      "where doc_id = 'archive/tickets/T-1'").fetchall()
    assert rows == [{"body": "Blank invoice PDF.", "title": "T-1", "acl": ["public"]}]

def test_removed_file_is_deleted_and_supersession_recomputed(db, tmp_path):
    kb, ar = tree(tmp_path); run(db, kb, ar)
    (ar / "tickets/T-1.md").unlink()
    write(kb, "policy/refunds.md", CURATED.replace("supersedes: [archive/policies/refund-2023]\n", ""))
    assert run(db, kb, ar) == SyncReport(0, 1, 1, 1)
    assert count(db, "select count(*) n from documents where superseded_by is not null") == 0
    assert count(db, "select count(*) n from chunks where doc_id = 'archive/tickets/T-1'") == 0

def test_lint_error_aborts_with_nothing_written(db, tmp_path):   # Review Focus 3
    kb, ar = tree(tmp_path); run(db, kb, ar)
    write(kb, "policy/refunds.md", CURATED.replace("within 14 days", "within two weeks"))
    write(ar, "tickets/T-2.md", "new ticket")
    with pytest.raises(SyncAborted) as e:
        run(db, kb, ar)
    assert [i.rule for i in e.value.issues] == ["L4"]
    assert count(db, "select count(*) n from documents") == 3

def test_embedding_model_change_reembeds_everything(db, tmp_path):
    kb, ar = tree(tmp_path); run(db, kb, ar)
    class V2(FakeEmbedder): model_id = "fake-bow-v2"
    assert run(db, kb, ar, V2()) == SyncReport(0, 3, 0, 0)

def test_a_tree_that_is_not_given_is_left_alone(db, tmp_path):
    kb, ar = tree(tmp_path); run(db, kb, ar)
    assert sync(db, None, ar, FakeEmbedder(), Settings(), TODAY) == SyncReport(0, 0, 2, 0)
    assert count(db, "select count(*) n from documents where tier = 'curated'") == 1

def test_interrupted_sync_keeps_finished_documents_and_resumes(db, tmp_path):
    kb, ar = tree(tmp_path)
    with pytest.raises(RuntimeError):
        run(db, kb, ar, FailingFrom(2))                          # the first document commits, the second fails
    assert count(db, "select count(*) n from documents") == 1
    assert count(db, "select count(*) n from documents d where not exists "
                     "(select 1 from chunks c where c.doc_id = d.doc_id)") == 0
    assert run(db, kb, ar) == SyncReport(added=2, updated=0, unchanged=1, deleted=0)

def test_supersession_is_applied_before_any_document_is_written(db, tmp_path):
    kb, ar = tree(tmp_path)
    sync(db, None, ar, FakeEmbedder(), Settings(), TODAY)        # archive only: nothing is superseded yet
    assert count(db, "select count(*) n from documents where superseded_by is not null") == 0
    with pytest.raises(RuntimeError):
        run(db, kb, ar, FailingFrom(1))                          # the curated document never gets written
    row = db.execute("select superseded_by from documents where doc_id = 'archive/policies/refund-2023'").fetchone()
    assert row["superseded_by"] == "kb/policy/refunds"           # the stale document is already hidden
    assert count(db, "select count(*) n from documents where tier = 'curated'") == 0

def test_update_keeps_a_document_superseded(db, tmp_path):
    kb, ar = tree(tmp_path); run(db, kb, ar)
    write(ar, "policies/refund-2023.md", "# Refunds 2023\n\nRefund window is 30 days. Reworded.")
    assert sync(db, None, ar, FakeEmbedder(), Settings(), TODAY) == SyncReport(0, 1, 1, 0)
    assert count(db, "select count(*) n from documents where superseded_by = 'kb/policy/refunds'") == 1

def test_emptied_source_folder_is_refused_without_force(db, tmp_path):
    kb, ar = tree(tmp_path); run(db, kb, ar)
    for f in list(ar.rglob("*.md")):
        f.unlink()
    with pytest.raises(DeleteGuard) as e:
        sync(db, None, ar, FakeEmbedder(), Settings(), TODAY)
    assert (e.value.tier, e.value.removed, e.value.existing) == ("archive", 2, 2)
    assert count(db, "select count(*) n from documents") == 3
    assert sync(db, None, ar, FakeEmbedder(), Settings(), TODAY, force=True) == SyncReport(0, 0, 0, 2)

def test_large_partial_deletion_is_refused_and_writes_nothing(db, tmp_path):
    ar = tmp_path / "archive"
    for i in range(40):
        write(ar, f"n/{i:02}.md", f"note {i}")
    sync(db, None, ar, FakeEmbedder(), Settings(), TODAY)
    for i in range(11):
        (ar / f"n/{i:02}.md").unlink()                           # 11 of 40: above the floor and above 20 %
    write(ar, "n/new.md", "new note")
    with pytest.raises(DeleteGuard):
        sync(db, None, ar, FakeEmbedder(), Settings(), TODAY)
    assert count(db, "select count(*) n from documents") == 40
    (ar / "n/new.md").unlink(); write(ar, "n/00.md", "note 0")   # now 10 of 40: at the floor, allowed
    assert sync(db, None, ar, FakeEmbedder(), Settings(), TODAY) == SyncReport(0, 0, 30, 10)

def test_unusable_archive_acl_refuses_the_sync_with_nothing_written(db, tmp_path):
    kb, ar = tree(tmp_path); run(db, kb, ar)
    for bad in ("---\nacl: finance\n---\nx", "---\nacl: []\n---\nx",
                "---\nacl: [finance, 5]\n---\nx", "---\nacl: [finance]\nbody without closing"):
        write(ar, "tickets/T-2.md", bad)
        write(ar, "tickets/T-1.md", "Edited.")                   # a valid edit in the same sync
        with pytest.raises(SyncAborted) as e:
            run(db, kb, ar)
        assert [i.rule for i in e.value.issues] == ["A1"]
        assert count(db, "select count(*) n from documents") == 3
        assert db.execute("select body from documents where doc_id = 'archive/tickets/T-1'").fetchone() == {
            "body": "Duplicate charge."}

def test_bom_does_not_hide_archive_front_matter(db, tmp_path):
    kb, ar = tree(tmp_path)
    (ar / "secrets").mkdir(); (ar / "secrets/hr.md").write_bytes(b"\xef\xbb\xbf---\nacl: [finance]\n---\nSecret.")
    run(db, kb, ar)
    assert db.execute("select acl, body from documents where doc_id = 'archive/secrets/hr'").fetchone() == {
        "acl": ["finance"], "body": "Secret."}

def test_archive_file_without_acl_defaults_to_public(db, tmp_path):
    kb, ar = tree(tmp_path)
    write(ar, "misc/x.md", "---\ntitle: T\n---\nx")
    run(db, kb, ar)
    assert db.execute("select acl from documents where doc_id = 'archive/misc/x'").fetchone() == {"acl": ["public"]}

def test_sync_refreshes_common_lexemes_from_the_chunks(db, tmp_path):
    kb, ar = tree(tmp_path)
    sync(db, kb, ar, FakeEmbedder(), low_floor(), TODAY)
    assert "refund" in common_of(db)                             # two chunks carry it, the floor is 1
    assert common_of(db) == expected_common(db, 1)

def test_refresh_also_runs_when_a_sync_changes_nothing(db, tmp_path):
    kb, ar = tree(tmp_path)
    sync(db, kb, ar, FakeEmbedder(), low_floor(), TODAY)
    db.execute("DELETE FROM common_lexemes")
    assert sync(db, kb, ar, FakeEmbedder(), low_floor(), TODAY) == SyncReport(0, 0, 3, 0)
    assert common_of(db) == expected_common(db, 1)

def test_aborted_or_refused_sync_does_not_refresh_common_lexemes(db, tmp_path):
    kb, ar = tree(tmp_path)
    sync(db, kb, ar, FakeEmbedder(), low_floor(), TODAY)
    db.execute("DELETE FROM common_lexemes")
    db.execute("INSERT INTO common_lexemes (lexeme, ndoc) VALUES ('sentinel', 99)")
    write(kb, "policy/refunds.md", CURATED.replace("within 14 days", "within two weeks"))
    with pytest.raises(SyncAborted):
        sync(db, kb, ar, FakeEmbedder(), low_floor(), TODAY)
    assert common_of(db) == {"sentinel"}
    write(kb, "policy/refunds.md", CURATED)
    for f in list(ar.rglob("*.md")):
        f.unlink()
    with pytest.raises(DeleteGuard):
        sync(db, None, ar, FakeEmbedder(), low_floor(), TODAY)   # the archive loses everything: refused
    assert common_of(db) == {"sentinel"}
