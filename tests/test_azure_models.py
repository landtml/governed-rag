"""Azure OpenAI backends, tested against a stand-in client: no network, no Azure account."""
import json
from types import SimpleNamespace

import pytest

from governed_rag.models import (DRAFT_SCHEMA, VERDICT_SCHEMA, AzureOpenAIEmbedder, AzureOpenAIGenerator,
                                 AzureOpenAIJudge, OpenRouterGenerator, OpenRouterJudge, make_generator,
                                 make_judge)
from governed_rag.types import Authority, Citation, Draft, Evidence, Verdict


def _evidence():
    return [Evidence("E1", "doc-1", "Refunds", Authority.CURATED, "Refunds take 14 days.", 0.9)]


class _Chat:
    """Records each chat.completions.create call and answers with a canned message."""

    def __init__(self, content, finish_reason="stop", refusal=None, model="gpt-4o-2024-11-20", provider=None):
        self.calls = []
        self._reply = SimpleNamespace(
            model=model, provider=provider,
            choices=[SimpleNamespace(finish_reason=finish_reason,
                                     message=SimpleNamespace(content=content, refusal=refusal))])

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._reply


def _client(chat=None, embeddings=None):
    return SimpleNamespace(chat=SimpleNamespace(completions=chat), embeddings=embeddings)


def test_generator_sends_the_draft_schema_in_strict_mode_and_parses_the_reply():
    chat = _Chat(json.dumps({"abstain": False, "answer": "14 days [E1]",
                             "citations": [{"evidence_id": "E1", "quote": "Refunds take 14 days."}]}))
    gen = AzureOpenAIGenerator(deployment="generator", client=_client(chat))

    draft = gen.generate("How long do refunds take?", _evidence())

    assert draft == Draft(False, "14 days [E1]", (Citation("E1", "Refunds take 14 days."),))
    call = chat.calls[0]
    assert call["model"] == "generator"
    fmt = call["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == DRAFT_SCHEMA
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    # Reasoning models reject max_tokens and temperature.
    assert "max_completion_tokens" in call and "max_tokens" not in call and "temperature" not in call
    assert "reasoning_effort" not in call
    assert "Refunds take 14 days." in call["messages"][1]["content"]


def test_generator_reports_the_model_the_service_actually_ran():
    chat = _Chat(json.dumps({"abstain": True, "answer": "", "citations": []}), model="gpt-4o-2024-11-20")
    gen = AzureOpenAIGenerator(deployment="generator", client=_client(chat))
    assert gen.model_id == "generator@azure-openai"
    gen.generate("q", _evidence())
    assert gen.model_id == "gpt-4o-2024-11-20@azure-openai"


def test_generator_passes_gate_feedback_into_the_prompt():
    chat = _Chat(json.dumps({"abstain": True, "answer": "", "citations": []}))
    AzureOpenAIGenerator(deployment="g", client=_client(chat)).generate("q", _evidence(), feedback="quote E1 exactly")
    assert "quote E1 exactly" in chat.calls[0]["messages"][1]["content"]


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_generator_raises_when_the_reply_was_cut_off_or_filtered(finish_reason):
    chat = _Chat('{"abstain": fal', finish_reason=finish_reason)
    with pytest.raises(RuntimeError, match=finish_reason):
        AzureOpenAIGenerator(deployment="g", client=_client(chat)).generate("q", _evidence())


def test_generator_raises_on_a_refusal():
    chat = _Chat(None, refusal="I can't help with that.")
    with pytest.raises(RuntimeError, match="refus"):
        AzureOpenAIGenerator(deployment="g", client=_client(chat)).generate("q", _evidence())


def test_judge_sends_the_verdict_schema_and_parses_the_reply():
    chat = _Chat(json.dumps({"supported": False, "unsupported": ["made-up claim"]}), model="gpt-4o-mini-2024-07-18")
    judge = AzureOpenAIJudge(deployment="judge", client=_client(chat))

    verdict = judge.judge("q", "an answer [E1]", _evidence())

    assert verdict == Verdict(False, ("made-up claim",))
    assert chat.calls[0]["response_format"]["json_schema"]["schema"] == VERDICT_SCHEMA
    assert chat.calls[0]["response_format"]["json_schema"]["strict"] is True
    assert judge.model_id == "gpt-4o-mini-2024-07-18@azure-openai"


class _Embeddings:
    def __init__(self, dim):
        self.dim = dim
        self.calls = []

    def create(self, model, input, dimensions):
        self.calls.append({"model": model, "input": list(input), "dimensions": dimensions})
        # Return the items out of order: the client must sort them by index.
        data = [SimpleNamespace(index=i, embedding=[float(i)] * dimensions) for i in range(len(input))]
        return SimpleNamespace(data=list(reversed(data)))


def test_embedder_asks_for_1024_dimensions_batches_and_keeps_input_order():
    emb = _Embeddings(1024)
    embedder = AzureOpenAIEmbedder(deployment="embedder", client=_client(embeddings=emb), batch_size=2)

    vectors = embedder.embed(["a", "b", "c"])

    assert embedder.dim == 1024
    assert [len(v) for v in vectors] == [1024, 1024, 1024]
    assert [v[0] for v in vectors] == [0.0, 1.0, 0.0]  # batch [a, b] then batch [c]
    assert [c["input"] for c in emb.calls] == [["a", "b"], ["c"]]
    assert all(c["dimensions"] == 1024 and c["model"] == "embedder" for c in emb.calls)


def test_embedder_id_names_the_model_and_dimension_so_a_switch_reembeds():
    embedder = AzureOpenAIEmbedder(deployment="embedder", client=_client(embeddings=_Embeddings(1024)))
    assert embedder.model_id == "text-embedding-3-large-1024@azure-openai"


def test_embedder_returns_nothing_for_no_texts_without_calling_the_service():
    emb = _Embeddings(1024)
    assert AzureOpenAIEmbedder(deployment="e", client=_client(embeddings=emb)).embed([]) == []
    assert emb.calls == []


def test_factories_build_the_azure_backends(monkeypatch):
    import governed_rag.models as models
    monkeypatch.setattr(models, "_azure_client", lambda: _client(_Chat("{}")))
    assert isinstance(make_generator("azure-openai"), AzureOpenAIGenerator)
    assert isinstance(make_judge("azure-openai"), AzureOpenAIJudge)


def test_deployment_names_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_GENERATOR_DEPLOYMENT", "gen-x")
    monkeypatch.setenv("AZURE_OPENAI_JUDGE_DEPLOYMENT", "judge-x")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDER_DEPLOYMENT", "emb-x")
    assert AzureOpenAIGenerator(client=_client()).deployment == "gen-x"
    assert AzureOpenAIJudge(client=_client()).deployment == "judge-x"
    assert AzureOpenAIEmbedder(client=_client()).deployment == "emb-x"


def test_cli_accepts_azure_openai_for_generator_judge_and_embedder():
    from governed_rag.cli import build_parser
    args = build_parser().parse_args(["--generator", "azure-openai", "--embedder", "azure-openai",
                                      "eval", "--golden", "g.jsonl", "--judge", "azure-openai"])
    assert (args.generator, args.embedder, args.judge) == ("azure-openai", "azure-openai", "azure-openai")
    assert build_parser().parse_args(["--embedder", "bge-m3", "eval", "--golden", "g"]).embedder == "bge-m3"


def test_azure_openai_is_the_default_for_every_model_seam():
    from governed_rag.cli import build_parser
    args = build_parser().parse_args(["eval", "--golden", "g.jsonl"])
    assert (args.generator, args.embedder, args.judge) == ("azure-openai", "azure-openai", "none")
    # --judge with no value turns the judge on with the Azure backend.
    assert build_parser().parse_args(["eval", "--golden", "g.jsonl", "--judge"]).judge == "azure-openai"
    assert build_parser().parse_args(["eval", "--golden", "g.jsonl", "--judge", "openrouter"]).judge == "openrouter"


def test_factories_build_azure_backends_when_no_kind_is_given(monkeypatch):
    import governed_rag.models as models
    monkeypatch.setattr(models, "_azure_client", lambda: _client(_Chat("{}"), _Embeddings(1024)))
    assert isinstance(models.make_generator(), AzureOpenAIGenerator)
    assert isinstance(models.make_judge(), AzureOpenAIJudge)
    assert isinstance(models.make_embedder(), AzureOpenAIEmbedder)
    assert models.GENERATORS[:2] == models.JUDGES[:2] == ("azure-openai", "openrouter")


def test_reasoning_effort_is_sent_only_when_set(monkeypatch):
    chat = _Chat(json.dumps({"abstain": True, "answer": "", "citations": []}))
    AzureOpenAIGenerator(deployment="g", client=_client(chat), reasoning_effort="low").generate("q", _evidence())
    assert chat.calls[0]["reasoning_effort"] == "low"
    monkeypatch.setenv("AZURE_OPENAI_REASONING_EFFORT", "medium")
    judge_chat = _Chat(json.dumps({"supported": True, "unsupported": []}))
    AzureOpenAIJudge(deployment="j", client=_client(judge_chat)).judge("q", "a", _evidence())
    assert judge_chat.calls[0]["reasoning_effort"] == "medium"


def test_openrouter_generator_pins_strict_routing_and_records_the_provider():
    chat = _Chat(json.dumps({"abstain": True, "answer": "", "citations": []}),
                 model="openai/gpt-6-luna", provider="OpenAI")
    gen = OpenRouterGenerator(model="openai/gpt-6-luna", client=_client(chat))
    gen.generate("q", _evidence())
    call = chat.calls[0]
    assert call["model"] == "openai/gpt-6-luna"
    assert call["extra_body"] == {"provider": {"require_parameters": True}}
    assert call["response_format"]["json_schema"]["strict"] is True
    assert gen.model_id == "openai/gpt-6-luna@openrouter[OpenAI]"


def test_openrouter_lists_every_provider_that_served_answers():
    gen = OpenRouterGenerator(model="m", client=_client(_Chat(json.dumps({"abstain": True, "answer": "", "citations": []}),
                                                               model="m", provider="OpenAI")))
    gen.generate("q", _evidence())
    gen._client = _client(_Chat(json.dumps({"abstain": True, "answer": "", "citations": []}), model="m", provider="Azure"))
    gen.generate("q", _evidence())
    assert gen.model_id == "m@openrouter[Azure,OpenAI]"


def test_azure_requests_carry_no_openrouter_routing():
    chat = _Chat(json.dumps({"supported": True, "unsupported": []}))
    AzureOpenAIJudge(deployment="j", client=_client(chat)).judge("q", "a", _evidence())
    assert "extra_body" not in chat.calls[0]


def test_factories_build_the_openrouter_backends(monkeypatch):
    import governed_rag.models as models
    monkeypatch.setattr(models, "_openrouter_client", lambda: _client(_Chat("{}")))
    assert isinstance(make_generator("openrouter"), OpenRouterGenerator)
    assert isinstance(make_judge("openrouter"), OpenRouterJudge)
