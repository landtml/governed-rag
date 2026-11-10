import pytest
from governed_rag.gates import check
from governed_rag.models import DRAFT_SCHEMA, parse_draft
from governed_rag.types import Authority, Citation, Draft, Evidence

EV = (Evidence("E1", "kb/p", "Refund policy", Authority.CURATED, "Refund window is 14 days.", 1.0),)

def test_module_imports_without_the_real_extra():
    import governed_rag.models                                     # heavy imports are deferred to constructors

def test_draft_schema_and_parser_agree():
    assert DRAFT_SCHEMA["required"] == ["abstain", "answer", "citations"]
    assert DRAFT_SCHEMA["properties"]["citations"]["items"]["required"] == ["evidence_id", "quote"]
    obj = {"abstain": False, "answer": "a [E1]", "citations": [{"evidence_id": "E1", "quote": "a"}]}
    assert parse_draft(obj) == Draft(False, "a [E1]", (Citation("E1", "a"),))
    assert parse_draft({"abstain": True, "answer": "", "citations": []}) == Draft(abstain=True)
    with pytest.raises(ValueError):
        parse_draft({"answer": "a"})

@pytest.mark.real
def test_embedder_contract():
    from governed_rag.models import STEmbedder
    e = STEmbedder(); (v,) = e.embed(["refund window"])
    assert e.dim == 1024 == len(v) and abs(sum(x * x for x in v) - 1) < 1e-3 and e.model_id == "BAAI/bge-m3"

@pytest.mark.real
def test_reranker_contract():
    from governed_rag.models import STReranker
    r = STReranker()
    hi, lo = r.score("What is the refund window?",
                     ["Refunds are accepted within 14 days.", "The office espresso machine is broken."])
    assert 0.0 <= lo < hi <= 1.0 and r.model_id == "BAAI/bge-reranker-v2-m3"

@pytest.mark.real
def test_generator_contract():
    from governed_rag.models import AnthropicGenerator
    d = AnthropicGenerator().generate("What is the refund window?", EV)
    assert check(d, EV, Authority.CURATED) == [] and "14" in d.answer

def _fake_claude(tmp_path, stdout: str, exit_code: int = 0):
    exe = tmp_path / "claude"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "Path(__file__).with_name('record.json').write_text("
        "json.dumps({'argv': sys.argv[1:], 'stdin': sys.stdin.read()}))\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.exit({exit_code})\n")
    exe.chmod(0o755)
    return exe

def test_claude_cli_generator_parses_structured_output(tmp_path):
    import json
    from governed_rag.models import ClaudeCliGenerator
    out = {"is_error": False, "structured_output": {
        "abstain": False, "answer": "a [E1]", "citations": [{"evidence_id": "E1", "quote": "Refund window is 14 days."}]}}
    exe = _fake_claude(tmp_path, json.dumps(out))
    gen = ClaudeCliGenerator(executable=str(exe))
    assert gen.model_id == "claude-sonnet-5-5@claude-cli-medium"
    d = gen.generate("q?", EV)
    assert d == Draft(False, "a [E1]", (Citation("E1", "Refund window is 14 days."),))
    rec = json.loads((tmp_path / "record.json").read_text())
    argv = rec["argv"]
    assert "-p" in argv and "--json-schema" in argv
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5-5"
    assert argv[argv.index("--effort") + 1] == "medium"
    assert rec["stdin"].endswith("Question: q?")

def test_claude_cli_generator_fails_closed(tmp_path):
    from governed_rag.models import ClaudeCliGenerator
    cases = [("", 3), ("not json", 0), ('{"is_error": true}', 0), ('{"is_error": false}', 0)]
    for i, (stdout, exit_code) in enumerate(cases):
        case_dir = tmp_path / f"case{i}"
        case_dir.mkdir()
        exe = _fake_claude(case_dir, stdout, exit_code)
        with pytest.raises(RuntimeError):
            ClaudeCliGenerator(executable=str(exe)).generate("q?", EV)

def test_claude_cli_generator_requires_an_executable(tmp_path, monkeypatch):
    from governed_rag.models import ClaudeCliGenerator
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.delenv("CLAUDE_CLI", raising=False)
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(RuntimeError):
        ClaudeCliGenerator()

@pytest.mark.real
def test_claude_cli_generator_contract():
    from governed_rag.models import ClaudeCliGenerator
    d = ClaudeCliGenerator().generate("What is the refund window?", EV)
    assert check(d, EV, Authority.CURATED) == [] and "14" in d.answer

def test_verdict_schema_and_parser_are_strict():
    from governed_rag.models import VERDICT_SCHEMA, parse_verdict
    from governed_rag.types import Verdict
    assert VERDICT_SCHEMA["required"] == ["supported", "unsupported"]
    assert VERDICT_SCHEMA["properties"]["unsupported"] == {"type": "array", "items": {"type": "string"}}
    assert VERDICT_SCHEMA["additionalProperties"] is False
    assert parse_verdict({"supported": True, "unsupported": []}) == Verdict(True, ())
    assert parse_verdict({"supported": False, "unsupported": ["30 days"]}) == Verdict(False, ("30 days",))
    assert parse_verdict({"supported": True, "unsupported": ["30 days"]}) == Verdict(False, ("30 days",))  # contradictory
    bad = ["yes", {"unsupported": []}, {"supported": True}, {"supported": 1, "unsupported": []},
           {"supported": True, "unsupported": "30 days"}, {"supported": True, "unsupported": [None]}]
    for obj in bad:
        with pytest.raises(ValueError):
            parse_verdict(obj)

def test_claude_cli_judge_returns_the_verdict(tmp_path):
    import json
    from governed_rag.models import VERDICT_SCHEMA, ClaudeCliJudge
    from governed_rag.prompt import JUDGE_PROMPT
    from governed_rag.types import Verdict
    out = {"is_error": False, "structured_output": {"supported": False, "unsupported": ["30 days"]}}
    exe = _fake_claude(tmp_path, json.dumps(out))
    judge = ClaudeCliJudge(executable=str(exe))
    assert judge.model_id == "claude-sonnet-5-5@claude-cli-medium"
    assert judge.judge("q?", "The window is 30 days [E1].", EV) == Verdict(False, ("30 days",))
    argv = json.loads((tmp_path / "record.json").read_text())["argv"]
    assert argv[argv.index("--system-prompt") + 1] == JUDGE_PROMPT
    assert argv[argv.index("--json-schema") + 1] == json.dumps(VERDICT_SCHEMA)
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5-5"
    assert json.loads((tmp_path / "record.json").read_text())["stdin"].endswith("Answer: The window is 30 days [E1].")

def test_claude_cli_judge_fails_closed(tmp_path):
    from governed_rag.models import ClaudeCliJudge
    cases = [("", 3), ("not json", 0), ('{"is_error": true}', 0), ('{"is_error": false}', 0)]
    for i, (stdout, exit_code) in enumerate(cases):
        case_dir = tmp_path / f"case{i}"
        case_dir.mkdir()
        exe = _fake_claude(case_dir, stdout, exit_code)
        with pytest.raises(RuntimeError):
            ClaudeCliJudge(executable=str(exe)).judge("q?", "a", EV)

def test_make_judge_selects_the_judge(tmp_path, monkeypatch):
    from governed_rag.fakes import FakeJudge
    from governed_rag.models import make_judge
    assert isinstance(make_judge("fake"), FakeJudge)
    with pytest.raises(ValueError):
        make_judge("nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.delenv("CLAUDE_CLI", raising=False)
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(RuntimeError):
        make_judge("claude-cli")
