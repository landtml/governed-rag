from helpers import fake_settings, seed_doc
from governed_rag.fakes import FAKE_TAU, ExtractiveGenerator, FakeEmbedder, FakeReranker, ScriptedGenerator
from governed_rag.types import Authority, Citation, Draft, Evidence

def dot(u, v): return sum(x * y for x, y in zip(u, v))

def test_fake_settings_pins_tau_and_accepts_overrides():
    assert fake_settings().tau == FAKE_TAU == 0.30
    assert fake_settings(max_parents=3).max_parents == 3
    assert FakeReranker.model_id == "fake-overlap-v1"

def test_fake_embedder_is_deterministic_and_unit_norm():
    e = FakeEmbedder()
    a, b = e.embed(["refund window", "refund window"])
    assert a == b and len(a) == 1024 == e.dim and abs(dot(a, a) - 1.0) < 1e-9
    assert e.embed([""])[0][0] == 1.0 and e.model_id == "fake-bow-v1"

def test_fake_embedder_similarity_tracks_word_overlap():
    q, near, far = FakeEmbedder().embed(["refund window", "the refund window is short", "espresso machine"])
    assert dot(q, near) > dot(q, far)

def test_fake_reranker_is_content_word_overlap():
    r = FakeReranker()
    assert r.score("What is the refund window?", ["Refund window is 14 days", "Espresso"]) == [1.0, 0.0]
    assert r.score("the of a", ["anything"]) == [0.0]

def test_scripted_generator_replays_and_records():
    g = ScriptedGenerator([Draft(abstain=True)])
    assert g.generate("q", (), None) == Draft(abstain=True) and g.calls == [("q", (), None)]

def test_extractive_generator_quotes_the_best_sentence():
    ev = (Evidence("E1", "kb/policy/refunds", "Refund policy", Authority.CURATED,
                   "Shipping is free. Customers can request a refund within 14 days.", 0.9),)
    d = ExtractiveGenerator().generate("What is the refund window?", ev)
    assert d.citations == (Citation("E1", "Customers can request a refund within 14 days."),)
    assert d.answer == "Customers can request a refund within 14 days. [E1]"
    assert ExtractiveGenerator().generate("q", ()) == Draft(abstain=True)
    assert ExtractiveGenerator().generate("espresso", ev) == Draft(abstain=True)

def test_fake_judge_accepts_an_answer_the_evidence_supports():
    from governed_rag.fakes import FakeJudge
    from governed_rag.types import Verdict
    ev = (Evidence("E1", "kb/p", "Refund policy", Authority.CURATED, "Refund window is 14 days.", 1.0),)
    assert FakeJudge.model_id == "fake-judge-v1"
    assert FakeJudge().judge("q?", "The refund window is 14 days [E1].", ev) == Verdict(True, ())

def test_fake_judge_rejects_a_wrong_number_unrelated_text_and_empty_answers():
    from governed_rag.fakes import FakeJudge
    from governed_rag.types import Verdict
    ev = (Evidence("E1", "kb/p", "Refund policy", Authority.CURATED, "Refund window is 14 days.", 1.0),)
    judge = FakeJudge()
    wrong = "The refund window is 30 days [E1]."
    assert judge.judge("q?", wrong, ev) == Verdict(False, (wrong,))
    assert judge.judge("q?", "Espresso machines are expensive.", ev) == Verdict(False, ("Espresso machines are expensive.",))
    assert judge.judge("q?", "", ev).supported is False and judge.judge("q?", "[E1]", ev).supported is False

def test_fake_judge_needs_half_the_answer_words_in_the_evidence():
    from governed_rag.fakes import FakeJudge
    ev = (Evidence("E1", "kb/p", "Refund policy", Authority.CURATED, "Refund window is 14 days.", 1.0),)
    judge = FakeJudge()
    assert judge.judge("q?", "The refund window policy is short.", ev).supported is True     # 2 of 4 content words
    assert judge.judge("q?", "The refund policy is short.", ev).supported is False           # 1 of 3 content words

def test_seed_doc_inserts_searchable_rows(db):
    seed_doc(db, "archive/a", chunks=[("Intro", "hello world")], acl=("support",))
    row = db.execute("select d.acl, d.body, c.section, c.ord, c.index_text "
                     "from chunks c join documents d using (doc_id)").fetchone()
    assert (row["acl"], row["body"], row["section"], row["ord"], row["index_text"]) == (
        ["support"], "hello world", "Intro", 0, "hello world")
