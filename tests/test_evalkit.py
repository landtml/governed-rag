import json
import threading
import time
from datetime import date
import pytest
from conftest import TEST_DATABASE_URL
from helpers import fake_settings, seed_doc
from governed_rag.db import connect
from governed_rag.evalkit import (Case, CaseResult, _suggest_tau, bootstrap_ci, calibrate_tau, calibration_stats,
                                gate, judge_answers, load_golden, run, score)
from governed_rag.fakes import ExtractiveGenerator, FakeEmbedder, FakeReranker
from governed_rag.types import Answer, Authority, Citation, Evidence, Verdict

def case(**kw):
    base = dict(id="c", question="q", groups=(), min_authority=Authority.ARCHIVE, tags=(), answerable=True,
                gold_docs=(), must_contain=(), forbidden_docs=())
    return Case(**{**base, **kw})

def nm(id, **kw):
    return case(id=id, answerable=False, tags=("near_miss",), **kw)

def res(c, text="", docs=(), reason=None, fails=((),), top=0.9):
    ev = tuple(Evidence(f"E{i + 1}", doc, doc, Authority.ARCHIVE, "t", 0.9) for i, doc in enumerate(docs))
    auth = Authority.ARCHIVE if reason is None else Authority.NONE
    return CaseResult(c, Answer(text, (), auth, ev, reason, fails, top), 10.0)

def test_bootstrap_ci():
    assert bootstrap_ci([1.0] * 20) == (1.0, 1.0, 1.0)
    m, lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert m == 0.5 and lo < 0.5 < hi and 0.15 < hi - lo < 0.25
    assert bootstrap_ci([0.0, 1.0] * 50) == (m, lo, hi)
    with pytest.raises(ValueError):
        bootstrap_ci([])

def test_calibrate_tau_maximises_youden_j():
    assert calibrate_tau([0.7, 0.8, 0.9], [0.1, 0.2]) == pytest.approx(0.45)
    assert calibrate_tau([0.4, 0.9], [0.1, 0.6]) == pytest.approx(0.75)    # J ties at 0.25 and 0.75: take the higher
    with pytest.raises(ValueError):
        calibrate_tau([], [0.1])

def test_load_golden_applies_defaults(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"id": "a", "question": "q?", "answerable": True, "groups": ["support"],
                             "min_authority": "curated", "gold_docs": ["kb/p"]}) + "\n\n"
                 + json.dumps({"id": "b", "question": "r?", "answerable": False}) + "\n")
    a, b = load_golden(p)
    assert a == case(id="a", question="q?", groups=("support",), min_authority=Authority.CURATED, gold_docs=("kb/p",))
    assert b == case(id="b", question="r?", answerable=False)

def test_score_and_hard_gates():
    results = [
        res(case(id="exact", tags=("exact", "curated"), gold_docs=("kb/p",), must_contain=("14",)), "14 days [E1]", ("kb/p",)),
        res(case(id="leak", answerable=False, forbidden_docs=("archive/secret",)), "x", ("archive/secret",)),
        res(case(id="unans", answerable=False), reason="no_evidence", fails=(), top=0.1),
        res(case(id="miss", gold_docs=("archive/a",)), "y", ("archive/b",), fails=(("G2:quote_not_found:E1",), ())),
    ]
    r = score(results)
    assert r["n"] == 4 and r["hard"] == {"forbidden_leaks": 1, "exact_accuracy": 1.0}
    soft = r["soft"]
    assert set(soft["recall_at_k"]) == {"mean", "lo", "hi", "n"} and soft["recall_at_k"]["n"] == 2
    assert soft["recall_at_k"]["mean"] == 0.5 and soft["curated_first"]["mean"] == 1.0
    assert soft["abstain_recall"]["mean"] == 0.5 and soft["abstain_precision"]["mean"] == 1.0
    assert soft["gate_pass_rate"]["mean"] == pytest.approx(2 / 3)
    assert r["suggested_tau"] == pytest.approx(0.5) and r["latency_ms"] == {"p50": 10.0, "p95": 10.0}
    assert gate(r, None) == ["hard:forbidden_leaks=1"]

def test_exact_case_fails_when_abstained_or_missing_the_value():
    c = case(tags=("exact",), must_contain=("14",))
    assert score([res(c, "thirty days")])["hard"]["exact_accuracy"] == 0.0
    assert score([res(c, reason="unverifiable")])["hard"]["exact_accuracy"] == 0.0
    assert score([res(case(), "x")])["hard"]["exact_accuracy"] == 1.0
    assert "curated_first" not in score([res(case(), "x")])["soft"]

def test_soft_gate_fires_only_outside_the_interval():
    rep = {"hard": {"forbidden_leaks": 0, "exact_accuracy": 1.0},
           "soft": {"recall_at_k": {"mean": 0.8, "lo": 0.7, "hi": 0.9, "n": 50}}}
    assert gate(rep, {"soft": {"recall_at_k": {"mean": 0.85}}}) == []
    assert gate(rep, {"soft": {"recall_at_k": {"mean": 0.95}}}) == ["soft:recall_at_k baseline=0.950 hi=0.900"]
    assert gate({**rep, "hard": {"forbidden_leaks": 0, "exact_accuracy": 0.5}}, None) == ["hard:exact_accuracy=0.500"]

def test_run_executes_each_case_as_its_principal(db):
    seed_doc(db, "archive/t", chunks=["Duplicate charge after switching plans."], acl=("support",))
    q = "duplicate charge switching plans"
    cases = [case(id="in", question=q, groups=("support",), gold_docs=("archive/t",)),
             case(id="out", question=q, answerable=False, forbidden_docs=("archive/t",))]
    out = run(db, cases, embedder=FakeEmbedder(), reranker=FakeReranker(), generator=ExtractiveGenerator(),
              settings=fake_settings(), today=date(2026, 11, 10))
    assert [r.answer.abstain_reason for r in out] == [None, "no_evidence"]
    assert score(out)["hard"]["forbidden_leaks"] == 0 and all(r.latency_ms >= 0 for r in out)

class _Probe:
    """Wraps a model and records the highest number of calls that were in flight at once."""
    def __init__(self, inner, barrier=None):
        self._inner, self._barrier = inner, barrier
        self._lock, self._active, self.peak = threading.Lock(), 0, 0
    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr
        def call(*a, **kw):
            with self._lock:
                self._active += 1
                self.peak = max(self.peak, self._active)
            try:
                if self._barrier:
                    self._barrier.wait(timeout=10)   # passes only if two calls overlap
                else:
                    time.sleep(0.01)
                return attr(*a, **kw)
            finally:
                with self._lock:
                    self._active -= 1
        return call

def test_run_with_workers_overlaps_generation_but_not_local_models(db):
    seed_doc(db, "archive/t", chunks=["Duplicate charge after switching plans."], acl=("support",))
    q = "duplicate charge switching plans"
    cases = [case(id=f"in{i}", question=q, groups=("support",), gold_docs=("archive/t",)) for i in range(4)]
    cases.insert(2, case(id="out", question=q, answerable=False))
    kw = dict(settings=fake_settings(), today=date(2026, 11, 10))
    serial = run(db, cases, embedder=FakeEmbedder(), reranker=FakeReranker(), generator=ExtractiveGenerator(), **kw)
    emb, rer = _Probe(FakeEmbedder()), _Probe(FakeReranker())
    gen = _Probe(ExtractiveGenerator(), threading.Barrier(2))
    out = run(db, cases, embedder=emb, reranker=rer, generator=gen, workers=4,
              connect=lambda: connect(TEST_DATABASE_URL), **kw)
    assert [r.case.id for r in out] == [c.id for c in cases]
    assert [r.answer.text for r in out] == [r.answer.text for r in serial]
    assert [r.answer.abstain_reason for r in out] == [None, None, "no_evidence", None, None]
    assert gen.peak >= 2 and emb.peak == 1 and rer.peak == 1
    assert db.execute("SELECT count(*) AS n FROM query_log").fetchone()["n"] == 2 * len(cases)
    assert score(out)["abstain_reasons"] == {"no_evidence": 1}

def test_run_with_workers_needs_a_connection_factory(db):
    with pytest.raises(ValueError):
        run(db, [case()], embedder=FakeEmbedder(), reranker=FakeReranker(), generator=ExtractiveGenerator(),
            settings=fake_settings(), today=date(2026, 11, 10), workers=2)

def test_load_golden_reads_split_and_rejects_unknown_values(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"id": "h", "question": "q?", "answerable": True, "split": "holdout"}) + "\n")
    (h,) = load_golden(p)
    assert h.split == "holdout" and case().split == "fit"
    p.write_text(json.dumps({"id": "bad", "question": "q?", "answerable": True, "split": "train"}) + "\n")
    with pytest.raises(ValueError, match="bad"):
        load_golden(p)

def test_suggest_tau_ignores_holdout_and_near_miss_cases():
    fit = [res(case(id="pos"), top=0.9), res(case(id="neg", answerable=False), top=0.1)]
    decoys = [res(nm("near"), top=0.3), res(case(id="hold", answerable=False, split="holdout"), top=0.3)]
    assert calibrate_tau([0.9], [0.1, 0.3]) == pytest.approx(0.6)     # the decoys would move tau from 0.5 to 0.6
    assert _suggest_tau(fit) == pytest.approx(0.5)
    assert _suggest_tau(fit + decoys) == pytest.approx(0.5)
    assert score(fit + decoys)["suggested_tau"] == pytest.approx(0.5)
    assert _suggest_tau(decoys) is None

def test_calibration_stats_by_split_and_near_miss():
    results = [
        res(case(id="f1"), top=0.9), res(case(id="f2"), top=0.4),
        res(case(id="f3", answerable=False), top=0.1), res(case(id="f4", answerable=False), top=0.6),
        res(case(id="h1", split="holdout"), top=0.8),
        res(case(id="h2", answerable=False, split="holdout"), top=0.2),
        res(case(id="h3", answerable=False, split="holdout"), top=0.3),
        res(case(id="h4", answerable=False, split="holdout"), top=0.7),
        res(nm("n1"), top=0.6), res(nm("n2", split="holdout"), top=0.2),
    ]
    cal = calibration_stats(results, 0.5)
    assert cal["tau"] == 0.5
    assert cal["fit"] == {"answerable": 2, "unanswerable": 2, "kept": 0.5, "passed": 0.5}
    assert cal["holdout"]["answerable"] == 1 and cal["holdout"]["unanswerable"] == 3
    assert cal["holdout"]["kept"] == 1.0 and cal["holdout"]["passed"] == pytest.approx(1 / 3)
    assert cal["near_miss_above_tau"] == {"n": 2, "share": 0.5}

def test_calibration_stats_none_cases():
    fit = [res(case(id="p"), top=0.9), res(case(id="n", answerable=False), top=0.1)]
    assert calibration_stats(fit, None) is None
    cal = calibration_stats(fit, 0.5)
    assert cal["holdout"] is None and cal["near_miss_above_tau"] is None
    assert calibration_stats(fit + [res(case(id="hp", split="holdout"), top=0.9)], 0.5)["holdout"] is None
    assert calibration_stats([res(case(id="p"), top=0.9)], 0.5)["fit"] == {
        "answerable": 1, "unanswerable": 0, "kept": 1.0, "passed": None}

def test_score_reports_calibration_at_the_suggested_tau():
    results = [res(case(id="p"), top=0.9), res(case(id="n", answerable=False), top=0.1)]
    r = score(results)
    assert r["suggested_tau"] == pytest.approx(0.5)
    assert r["calibration"] == calibration_stats(results, r["suggested_tau"])
    assert r["calibration"]["fit"]["kept"] == 1.0

def answered(cid, text, docs):
    ev = tuple(Evidence(f"E{i + 1}", d, d, Authority.ARCHIVE, "t", 0.9) for i, d in enumerate(docs))
    return CaseResult(case(id=cid, question=f"q-{cid}"),
                      Answer(text, tuple(Citation(e.eid, "t") for e in ev), Authority.ARCHIVE, ev), 1.0)

def about(question, answer, evidence):
    return Verdict(bool(evidence) and all(f"about {e.doc_id}" in answer for e in evidence))

class _Recorder:
    model_id = "stub-judge"

    def __init__(self, rule=about):
        self.rule, self.calls = rule, []

    def judge(self, question, answer, evidence):
        self.calls.append((question, answer, tuple(e.doc_id for e in evidence)))
        return self.rule(question, answer, evidence)

def test_judge_answers_pairs_each_answer_with_a_disjoint_control():
    results = [answered("A", "about kb/a", ["kb/a"]), answered("B", "about kb/b", ["kb/b"]),
               answered("C", "about kb/a", ["kb/a"]),
               CaseResult(case(id="D", answerable=False), Answer("", (), Authority.NONE, (), "no_evidence", (), 0.1), 1.0)]
    judge = _Recorder()
    r = judge_answers(results, judge)
    assert r["model"] == "stub-judge" and r["errors"] == 0 and r["unsupported"] == []
    assert r["faithfulness"]["mean"] == 1.0 and r["faithfulness"]["n"] == 3
    assert r["control_rejection"]["mean"] == 1.0 and r["control_rejection"]["n"] == 3
    assert len(judge.calls) == 6 and not any(q == "q-D" for q, _, _ in judge.calls)   # the abstention is skipped
    assert ("q-A", "about kb/a", ("kb/a",)) in judge.calls and ("q-A", "about kb/a", ("kb/b",)) in judge.calls
    assert ("q-B", "about kb/b", ("kb/a",)) in judge.calls                              # next answered case, C
    assert ("q-C", "about kb/a", ("kb/b",)) in judge.calls                              # A shares kb/a: skipped

def test_judge_answers_reports_values_when_the_judge_is_lenient():
    results = [answered("X", "wrong about kb/a", ["kb/a"]), answered("Y", "about kb/b", ["kb/b"])]
    judge = _Recorder(lambda q, a, ev: Verdict(False, ("wrong",)) if "wrong" in a else Verdict(True))
    r = judge_answers(results, judge)
    assert r["faithfulness"]["mean"] == 0.5 and r["faithfulness"]["n"] == 2
    assert r["control_rejection"]["mean"] == 0.5 and r["control_rejection"]["n"] == 2   # Y's control is accepted
    assert r["unsupported"] == [{"id": "X", "claims": ["wrong"]}]

@pytest.mark.parametrize("workers", [1, 2])
def test_judge_answers_counts_failed_calls_and_leaves_them_out(workers):
    def rule(q, a, ev):
        if "boom" in a and any(e.doc_id == "kb/a" for e in ev):
            raise RuntimeError("judge down")
        return Verdict("boom" not in a)
    results = [answered("A", "boom about kb/a", ["kb/a"]), answered("B", "about kb/b", ["kb/b"])]
    r = judge_answers(results, _Recorder(rule), workers=workers)
    assert r["errors"] == 1
    assert r["faithfulness"]["mean"] == 1.0 and r["faithfulness"]["n"] == 1
    assert r["control_rejection"]["mean"] == 0.5 and r["control_rejection"]["n"] == 2
    assert r["unsupported"] == []

def test_judge_answers_has_no_control_without_a_disjoint_partner():
    one = judge_answers([answered("A", "about kb/a", ["kb/a"])], _Recorder())
    assert one["control_rejection"] is None and one["faithfulness"]["n"] == 1
    shared = _Recorder()
    both = judge_answers([answered("A", "about kb/a", ["kb/a"]), answered("B", "about kb/a", ["kb/a"])], shared)
    assert both["control_rejection"] is None and len(shared.calls) == 2
    empty = _Recorder()
    none = judge_answers([CaseResult(case(), Answer("", (), Authority.NONE, (), "no_evidence", (), 0.1), 1.0)], empty)
    assert none["faithfulness"] is None and none["control_rejection"] is None and empty.calls == []

def test_gate_ignores_the_judge_report():
    report = score([answered("A", "about kb/a", ["kb/a"])])
    bad_judge = {"model": "m", "faithfulness": {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 1},
                 "control_rejection": None, "errors": 3, "unsupported": [{"id": "A", "claims": ["x"]}]}
    judged = {**report, "judge": bad_judge}
    assert gate(judged, None) == gate(report, None) == []
    assert gate(judged, {"soft": {}, "judge": {"faithfulness": {"mean": 1.0}}}) == []

def test_near_miss_abstain_counts_only_near_miss_cases():
    results = [res(nm("a"), reason="no_evidence", fails=(), top=0.8), res(nm("b"), "x", top=0.8),
               res(case(id="ok"), "y", top=0.9)]
    r = score(results)
    assert r["soft"]["near_miss_abstain"]["mean"] == 0.5 and r["soft"]["near_miss_abstain"]["n"] == 2
    assert r["soft"]["abstain_recall"]["mean"] == 0.5             # near-miss cases still count there
    assert r["suggested_tau"] is None and r["calibration"] is None  # near-miss cases never fit tau
    assert "near_miss_abstain" not in score([res(case(), "x")])["soft"]
