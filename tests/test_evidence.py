from datetime import date
from helpers import fake_settings, seed_doc
from governed_rag.evidence import search
from governed_rag.fakes import FakeEmbedder, FakeReranker
from governed_rag.text import count_tokens
from governed_rag.types import Authority, Principal, SearchResult

TODAY = date(2026, 11, 10)

def go(db, q, settings=None, **kw):
    return search(db, q, Principal(), embedder=FakeEmbedder(), reranker=FakeReranker(),
                  settings=fake_settings(**(settings or {})), today=TODAY, **kw)

def test_nothing_above_tau_abstains_and_reports_top_score(db):
    seed_doc(db, "archive/a", chunks=["refund policy overview"])
    r = go(db, "office espresso machine refund")                # overlap 1/4 < τ
    assert r.evidence == () and r.abstain_reason == "no_evidence" and r.top_score == 0.25

def test_empty_index_abstains(db):
    assert go(db, "anything") == SearchResult((), 0.0, "no_evidence")

def test_curated_doc_is_delivered_whole_with_facts(db):
    seed_doc(db, "kb/policy/refunds", tier="curated", title="Refund policy",
             chunks=[("Refunds", "Refund window is 14 days."), ("Exceptions", "Gift cards are final sale.")],
             facts={"refund_window_days": "14"})
    r = go(db, "refund window")
    (e,) = r.evidence
    assert (e.eid, e.authority, e.title, e.score) == ("E1", Authority.CURATED, "Refund policy", 1.0)
    assert e.text == "FACT refund_window_days = 14\n\nRefund window is 14 days.\n\nGift cards are final sale."
    assert r.abstain_reason is None and r.top_score == 1.0

def test_archive_hit_expands_within_its_section_only(db):
    seed_doc(db, "archive/h", title="Handbook", chunks=[
        ("Laptops", "Request hardware early."), ("Laptops", "Laptop access needs a ticket."),
        ("Laptops", "IT replies in a day."), ("Parking", "Parking is limited.")])
    (e,) = go(db, "laptop access ticket").evidence
    assert e.text == "Request hardware early.\n\nLaptop access needs a ticket.\n\nIT replies in a day."
    assert e.authority == Authority.ARCHIVE

def test_parent_cap_limits_expansion(db):
    seed_doc(db, "archive/h", chunks=[("S", "w " * 100), ("S", "laptop access ticket " + "w " * 97),
                                      ("S", "v " * 100)])
    (e,) = go(db, "laptop access ticket", settings={"parent_cap_tokens": 200}).evidence
    assert count_tokens(e.text) == 200 and "v" not in e.text.split()

def test_non_adjacent_hits_in_one_section_merge_with_a_gap_marker(db):
    seed_doc(db, "archive/h", chunks=[("S", "laptop access one"), ("S", "w " * 50), ("S", "laptop access two")])
    (e,) = go(db, "laptop access", settings={"parent_cap_tokens": 6}).evidence
    assert e.text == "laptop access one\n\n[…]\n\nlaptop access two"

def test_curated_precedes_higher_scoring_archive(db):
    seed_doc(db, "archive/t", chunks=["refund window dispute escalation"])
    seed_doc(db, "kb/p", tier="curated", chunks=["Refund window is 14 days."])
    ev = go(db, "refund window dispute escalation").evidence     # archive 1.0, curated 0.5
    assert [(e.eid, e.doc_id) for e in ev] == [("E1", "kb/p"), ("E2", "archive/t")]

def test_budget_drops_whole_parents_and_never_truncates(db):
    for i in range(3):
        seed_doc(db, f"archive/d{i}", chunks=["laptop access " + "w " * 98])
    ev = go(db, "laptop access", settings={"evidence_budget_tokens": 250}).evidence
    assert len(ev) == 2 and all(count_tokens(e.text) == 100 for e in ev)

def test_max_parents(db):
    for i in range(10):
        seed_doc(db, f"archive/d{i}", chunks=["laptop access"])
    assert len(go(db, "laptop access", settings={"max_parents": 3}).evidence) == 3

def test_min_authority_curated_hides_archive(db):
    seed_doc(db, "archive/t", chunks=["duplicate charge"])
    r = go(db, "duplicate charge", min_authority=Authority.CURATED)
    assert r.evidence == () and r.abstain_reason == "no_evidence"
