import json, os, shutil, subprocess, sys
import pytest

URL = os.environ.get("TEST_DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag_test")
TREES = ["--knowledge", "demo/knowledge", "--archive", "demo/archive"]

def rag(*args):
    return subprocess.run([sys.executable, "-m", "governed_rag", "--fake", "--database-url", URL,
                           "--today", "2026-11-10", *args], capture_output=True, text=True)

def ok(*args):
    p = rag(*args); assert p.returncode == 0, p.stdout + p.stderr; return p.stdout

def test_demo_passes_every_gate(db):
    ok("lint", *TREES); ok("ingest", *TREES)
    r = json.loads(ok("eval", "--golden", "demo/golden.jsonl", "--json"))
    assert r["n"] == 11 and r["hard"] == {"forbidden_leaks": 0, "exact_accuracy": 1.0}
    for metric in ("recall_at_k", "curated_first", "abstain_recall", "abstain_precision", "gate_pass_rate"):
        assert r["soft"][metric]["mean"] == 1.0, metric
    assert r["violations"] == []

def test_ask_fact_search_and_backlog(db):
    ok("ingest", *TREES)
    a = json.loads(ok("ask", "What is our refund window?"))
    assert "14" in a["text"] and a["authority"] == "curated" and a["abstain_reason"] is None
    assert a["citations"][0]["doc_id"] == "kb/policy/refunds" and a["citations"][0]["evidence_id"] == "E1"
    assert json.loads(ok("fact", "refund_window_days")) == {"key": "refund_window_days", "value": "14", "unit": "days"}
    assert json.loads(ok("fact", "sev1_page_within_minutes")) is None
    s = json.loads(ok("search", "What is our refund window?"))
    assert [e["doc_id"] for e in s["evidence"]] == ["kb/policy/refunds"]
    b = json.loads(ok("ask", "Has anyone seen a duplicate charge after switching plans?", "--groups", "support"))
    assert b["authority"] == "archive"
    strict = json.loads(ok("ask", "Has anyone seen a duplicate charge after switching plans?",
                           "--groups", "support", "--min-authority", "curated"))
    assert strict["abstain_reason"] == "no_evidence"
    backlog = json.loads(ok("backlog"))
    assert backlog[0] == {"question": "has anyone seen a duplicate charge after switching plans?", "count": 2}

def test_ingest_is_idempotent(db):
    ok("ingest", *TREES)
    assert json.loads(ok("ingest", *TREES)) == {"added": 0, "updated": 0, "unchanged": 8, "deleted": 0}

def test_lint_and_ingest_fail_on_a_broken_fact(db, tmp_path):
    kb = tmp_path / "knowledge"; shutil.copytree("demo/knowledge", kb)
    p = kb / "policy/refunds.md"; p.write_text(p.read_text().replace("14 days", "two weeks"))
    bad = rag("lint", "--knowledge", str(kb), "--archive", "demo/archive")
    assert bad.returncode == 1 and "L4" in bad.stdout
    assert rag("ingest", "--knowledge", str(kb), "--archive", "demo/archive").returncode == 1

def test_baseline_roundtrip(db, tmp_path):
    ok("ingest", *TREES)
    base = tmp_path / "baseline.json"
    ok("eval", "--golden", "demo/golden.jsonl", "--write-baseline", str(base))
    assert "recall_at_k" in json.loads(base.read_text())["soft"]
    ok("eval", "--golden", "demo/golden.jsonl", "--baseline", str(base))

def test_calibration_is_written_by_eval_used_by_real_mode_and_ignored_by_fake_mode(db, tmp_path):
    from governed_rag.cli import resolve_settings
    from governed_rag.fakes import FAKE_TAU
    cal = tmp_path / "calibration.json"
    ok("ingest", *TREES)
    ok("eval", "--golden", "demo/golden.jsonl", "--write-calibration", str(cal))
    written = json.loads(cal.read_text())
    assert set(written) == {"tau", "reranker", "golden_sha256", "cases", "date",
                            "fit_cases", "holdout", "near_miss_above_tau"}
    assert (written["reranker"], written["cases"], written["date"]) == ("fake-overlap-v1", 11, "2026-11-10")
    assert written["fit_cases"] == {"answerable": 5, "unanswerable": 6}   # demo/golden.jsonl has no holdout rows
    assert written["holdout"] is None and written["near_miss_above_tau"] is None
    assert 0.0 < written["tau"] < 1.0
    cal.write_text(json.dumps({**written, "tau": 0.99}))
    assert resolve_settings(URL, False, cal, "fake-overlap-v1").tau == 0.99
    assert resolve_settings(URL, True, cal, "fake-overlap-v1").tau == FAKE_TAU
    with pytest.raises(ValueError):
        resolve_settings(URL, False, cal, "some-other-reranker")
    a = json.loads(ok("--calibration", str(cal), "ask", "What is our refund window?"))
    assert a["abstain_reason"] is None                           # --fake ignores the 0.99 in the file

def test_eval_judge_is_reported_and_absent_by_default(db):
    ok("ingest", *TREES)
    judged = json.loads(ok("eval", "--golden", "demo/golden.jsonl", "--judge", "fake", "--json"))
    assert set(judged["judge"]) == {"model", "faithfulness", "control_rejection", "errors", "unsupported"}
    assert judged["judge"]["model"] == "fake-judge-v1" and judged["judge"]["errors"] == 0
    assert judged["judge"]["faithfulness"]["n"] >= 1 and judged["violations"] == []
    plain = json.loads(ok("eval", "--golden", "demo/golden.jsonl", "--json"))
    assert "judge" not in plain
    text = ok("eval", "--golden", "demo/golden.jsonl", "--judge", "fake")
    assert any(l.startswith("judge: model=fake-judge-v1 faithfulness=") for l in text.splitlines())

def test_eval_summary_prints_the_calibration_line(db):
    ok("ingest", *TREES)
    line = [l for l in ok("eval", "--golden", "demo/golden.jsonl").splitlines() if l.startswith("calibration:")]
    assert len(line) == 1 and line[0].startswith("calibration: fit kept=")
    assert "holdout" not in line[0] and "near_miss_above_tau" not in line[0]

def test_ingest_refuses_an_emptied_archive_without_force(db, tmp_path):
    ok("ingest", "--archive", "demo/archive")
    empty = tmp_path / "archive"; empty.mkdir()
    p = rag("ingest", "--archive", str(empty))
    assert p.returncode == 1 and "--force" in p.stdout
    assert json.loads(ok("ingest", "--archive", str(empty), "--force"))["deleted"] == 5
