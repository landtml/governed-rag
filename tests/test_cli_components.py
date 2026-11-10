import os

import pytest
from governed_rag.cli import main
from governed_rag.fakes import FakeEmbedder
from governed_rag.models import make_generator

URL = os.environ.get("TEST_DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag_test")
TREES = ["--knowledge", "demo/knowledge", "--archive", "demo/archive"]

class _MustNotBuild:
    def __init__(self, *args, **kwargs):
        raise AssertionError("this command must not build this model")

def test_real_mode_ingest_and_fact_build_no_reranker_or_generator(db, monkeypatch):
    import governed_rag.models as models
    monkeypatch.setattr(models, "STEmbedder", FakeEmbedder)
    monkeypatch.setattr(models, "STReranker", _MustNotBuild)
    monkeypatch.setattr(models, "AnthropicGenerator", _MustNotBuild)
    monkeypatch.setattr(models, "ClaudeCliGenerator", _MustNotBuild)
    base = ["--database-url", URL, "--today", "2026-11-10"]
    assert main([*base, "ingest", *TREES]) == 0
    assert main([*base, "fact", "refund_window_days"]) == 0

def test_calibration_line_omits_none_parts():
    from governed_rag.cli import _calibration_line
    cal = {"tau": 0.5, "fit": {"answerable": 2, "unanswerable": 2, "kept": 0.5, "passed": 0.5},
           "holdout": {"answerable": 1, "unanswerable": 3, "kept": 1.0, "passed": 1 / 3},
           "near_miss_above_tau": {"n": 2, "share": 0.5}}
    assert _calibration_line(cal) == ("calibration: fit kept=0.500 passed=0.500 | holdout kept=1.000 "
                                      "passed=0.333 | near_miss_above_tau=0.500")
    assert _calibration_line({**cal, "holdout": None, "near_miss_above_tau": None}) == \
        "calibration: fit kept=0.500 passed=0.500"

def test_judge_is_built_only_when_asked_for(db, monkeypatch):
    import governed_rag.models as models
    from governed_rag.cli import main
    monkeypatch.setattr(models, "make_judge", _MustNotBuild)
    base = ["--fake", "--database-url", URL, "--today", "2026-11-10", "eval", "--golden", "demo/golden.jsonl"]
    main(base)                                   # no --judge: must not build a judge
    with pytest.raises(AssertionError):
        main([*base, "--judge", "fake"])

def test_judge_option_rejects_unknown_kinds():
    from governed_rag.cli import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval", "--golden", "g.jsonl", "--judge", "gpt"])

def test_generator_option_selects_the_generator():
    assert make_generator("extractive").model_id == "extractive-v1"
    with pytest.raises(ValueError):
        make_generator("nope")


def test_serve_refuses_to_start_without_tokens_or_anonymous_access(db, tmp_path, capsys):
    base = ["--fake", "--database-url", URL]
    assert main([*base, "serve", "--port", "0"]) == 2
    assert "--tokens" in capsys.readouterr().err
    bad = tmp_path / "tokens.json"
    bad.write_text('{"short": ["finance"]}')
    assert main([*base, "serve", "--port", "0", "--tokens", str(bad)]) == 2
    assert "token file" in capsys.readouterr().err


def test_serve_starts_with_a_token_file_and_reads_the_date_per_request(db, tmp_path, monkeypatch):
    import governed_rag.service as service
    seen = {}

    def fake_serve(server):
        seen["today"], seen["tokens"] = server.service.today, len(server.service._tokens)
        server.server_close()

    monkeypatch.setattr(service, "serve", fake_serve)
    tokens = tmp_path / "tokens.json"
    tokens.write_text('{"0123456789abcdef0123": ["finance"]}')
    assert main(["--fake", "--database-url", URL, "serve", "--port", "0", "--tokens", str(tokens)]) == 0
    assert seen == {"today": None, "tokens": 1}
    assert main(["--fake", "--database-url", URL, "--today", "2026-11-10", "serve", "--port", "0",
                 "--tokens", str(tokens)]) == 0
    assert seen["today"].isoformat() == "2026-11-10"


def test_serve_shuts_down_cleanly_on_sigterm(db, tmp_path):
    import signal, subprocess, sys, time, urllib.request
    tokens = tmp_path / "tokens.json"
    tokens.write_text('{"0123456789abcdef0123": []}')
    proc = subprocess.Popen([sys.executable, "-m", "governed_rag", "--fake", "--database-url", URL, "serve",
                             "--port", "0", "--tokens", str(tokens)], stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stderr.readline()                      # "serving on http://127.0.0.1:PORT (...)"
        url = line.split()[2]
        with urllib.request.urlopen(url + "/healthz", timeout=10) as response:
            assert response.status == 200
        proc.send_signal(signal.SIGTERM)
        proc.send_signal(signal.SIGTERM)                    # process managers repeat it; the second is ignored
        assert proc.wait(timeout=10) == 0
        assert "Traceback" not in proc.stderr.read()
    finally:
        if proc.poll() is None:
            proc.kill()
