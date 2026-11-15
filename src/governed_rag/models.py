from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence

from governed_rag.fakes import ExtractiveGenerator, FakeJudge
from governed_rag.prompt import render_judge_prompt, render_prompt
from governed_rag.types import Citation, Draft, Embedder, Evidence, Generator, Judge, Reranker, Verdict

DRAFT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "abstain": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["evidence_id", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["abstain", "answer", "citations"],
    "additionalProperties": False,
}


def parse_draft(obj: dict) -> Draft:
    if not isinstance(obj, dict):
        raise ValueError("draft must be a JSON object")
    for key in DRAFT_SCHEMA["required"]:
        if key not in obj:
            raise ValueError(f"draft missing {key!r}")
    if not isinstance(obj["abstain"], bool):
        raise ValueError("abstain must be a boolean")
    if not isinstance(obj["answer"], str):
        raise ValueError("answer must be a string")
    if not isinstance(obj["citations"], list):
        raise ValueError("citations must be a list")
    citations = []
    for c in obj["citations"]:
        if not isinstance(c, dict) or not isinstance(c.get("evidence_id"), str) \
                or not isinstance(c.get("quote"), str):
            raise ValueError("citation needs string evidence_id and quote")
        citations.append(Citation(c["evidence_id"], c["quote"]))
    return Draft(obj["abstain"], obj["answer"], tuple(citations))


VERDICT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "unsupported": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["supported", "unsupported"],
    "additionalProperties": False,
}


def parse_verdict(obj: dict) -> Verdict:
    if not isinstance(obj, dict):
        raise ValueError("verdict must be a JSON object")
    for key in VERDICT_SCHEMA["required"]:
        if key not in obj:
            raise ValueError(f"verdict missing {key!r}")
    if not isinstance(obj["supported"], bool):
        raise ValueError("supported must be a boolean")
    if not isinstance(obj["unsupported"], list) or not all(isinstance(c, str) for c in obj["unsupported"]):
        raise ValueError("unsupported must be a list of strings")
    claims = tuple(obj["unsupported"])
    return Verdict(obj["supported"] and not claims, claims)  # a supported verdict that lists claims is contradictory


class STEmbedder:
    dim = 1024

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.model_id = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return vectors.tolist()


class STReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder
        self.model_id = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        raw = [float(s) for s in self._model.predict([(query, p) for p in passages])]
        if any(not 0.0 <= s <= 1.0 for s in raw):
            # Raw logits: squash every score so the whole batch is on the same scale.
            raw = [1.0 / (1.0 + math.exp(-s)) for s in raw]
        return raw


class AnthropicGenerator:
    def __init__(self, model: str = "claude-sonnet-5-5"):
        import anthropic
        self.model_id = model
        self._client = anthropic.Anthropic()

    def generate(self, question: str, evidence: Sequence[Evidence],
                 feedback: str | None = None) -> Draft:
        system, user = render_prompt(question, evidence, feedback)
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": DRAFT_SCHEMA}},
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"generation stopped: {response.stop_reason}")
        text = next(b.text for b in response.content if b.type == "text")
        return parse_draft(json.loads(text))


def _find_claude(executable: str | None) -> str:
    found = executable or os.environ.get("CLAUDE_CLI") or shutil.which("claude")
    if found is None:
        raise RuntimeError("claude CLI not found: put it on PATH or set CLAUDE_CLI")
    return found


class ClaudeCliGenerator:
    def __init__(self, model: str = "claude-sonnet-5-5", effort: str = "medium",
                 executable: str | None = None, timeout: float = 180.0):
        self._executable = _find_claude(executable)
        self.model = model
        self.effort = effort
        self.model_id = f"{model}@claude-cli-{effort}"
        self._timeout = timeout

    def generate(self, question: str, evidence: Sequence[Evidence],
                 feedback: str | None = None) -> Draft:
        system, user = render_prompt(question, evidence, feedback)
        return parse_draft(_claude_json(self._executable, self.model, self.effort, system, user,
                                        DRAFT_SCHEMA, self._timeout))


class ClaudeCliJudge:
    def __init__(self, model: str = "claude-sonnet-5-5", effort: str = "medium",
                 executable: str | None = None, timeout: float = 180.0):
        self._executable = _find_claude(executable)
        self.model = model
        self.effort = effort
        self.model_id = f"{model}@claude-cli-{effort}"
        self._timeout = timeout

    def judge(self, question: str, answer: str, evidence: Sequence[Evidence]) -> Verdict:
        system, user = render_judge_prompt(question, answer, evidence)
        return parse_verdict(_claude_json(self._executable, self.model, self.effort, system, user,
                                          VERDICT_SCHEMA, self._timeout))


def _claude_json(executable: str, model: str, effort: str, system: str, user: str, schema: dict,
                 timeout: float) -> dict:
    argv = [executable, "-p", "--model", model, "--effort", effort,
            "--tools", "", "--strict-mcp-config", "--setting-sources", "",
            "--no-session-persistence", "--system-prompt", system,
            "--json-schema", json.dumps(schema), "--output-format", "json"]
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(argv, input=user, text=True, capture_output=True,
                              timeout=timeout, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr[-500:]}")
    try:
        obj = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError("claude output is not JSON") from e
    if not isinstance(obj, dict) or obj.get("is_error") or not isinstance(obj.get("structured_output"), dict):
        raise RuntimeError("claude returned no structured output")
    return obj["structured_output"]


# Data-plane API version that has structured outputs (json_schema, strict) and
# the reasoning-model parameters (max_completion_tokens, reasoning_effort).
AZURE_OPENAI_API_VERSION = "2025-04-01-preview"


def _azure_client():
    """An Azure OpenAI client that authenticates with Microsoft Entra ID, never an API key.

    DefaultAzureCredential uses the Azure CLI login on a developer machine and the
    managed identity inside Azure. The Foundry resource has key auth disabled.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI
    token = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    # The SDK retries 429s with backoff and honours Retry-After; regional quota is small.
    return AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], azure_ad_token_provider=token,
                       api_version=AZURE_OPENAI_API_VERSION, max_retries=10)


def _azure_json(client, deployment: str, system: str, user: str, name: str, schema: dict,
                max_tokens: int, reasoning_effort: str | None,
                extra_body: dict | None = None) -> tuple[dict, str, str | None]:
    """One structured-output chat call. Returns the parsed object and the model the service ran.

    Written for reasoning models (the GPT-5 family and later): they take
    max_completion_tokens, which also counts the hidden reasoning tokens, and
    reject temperature, so sampling stays at the service default.
    """
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    if extra_body:
        extra["extra_body"] = extra_body
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
        max_completion_tokens=max_tokens,
        **extra,
    )
    choice = response.choices[0]
    if choice.message.refusal:
        raise RuntimeError(f"model refused: {choice.message.refusal[:200]}")
    if choice.finish_reason != "stop":
        raise RuntimeError(f"generation stopped: {choice.finish_reason}")
    # OpenRouter adds the upstream provider to the response; Azure does not.
    return json.loads(choice.message.content), response.model, getattr(response, "provider", None)


def _served_id(backend, model: str, provider: str | None) -> str:
    """model@label, plus every upstream provider seen so far when the gateway reports one."""
    if provider:
        backend.providers = sorted(set(getattr(backend, "providers", [])) | {provider})
        return f"{model}@{backend._label}[{','.join(backend.providers)}]"
    return f"{model}@{backend._label}"


class AzureOpenAIGenerator:
    _label = "azure-openai"
    _extra_body: dict | None = None

    def __init__(self, deployment: str | None = None, client=None, max_tokens: int = 16000,
                 reasoning_effort: str | None = None):
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_GENERATOR_DEPLOYMENT", "generator")
        self._client = client if client is not None else _azure_client()
        self._max_tokens = max_tokens
        self._effort = reasoning_effort or os.environ.get("AZURE_OPENAI_REASONING_EFFORT")
        # Replaced by the model version the service reports on the first call.
        self.model_id = f"{self.deployment}@{self._label}"

    def generate(self, question: str, evidence: Sequence[Evidence],
                 feedback: str | None = None) -> Draft:
        system, user = render_prompt(question, evidence, feedback)
        obj, model, provider = _azure_json(self._client, self.deployment, system, user, "draft", DRAFT_SCHEMA,
                                           self._max_tokens, self._effort, self._extra_body)
        self.model_id = _served_id(self, model, provider)
        return parse_draft(obj)


class AzureOpenAIJudge:
    _label = "azure-openai"
    _extra_body: dict | None = None

    def __init__(self, deployment: str | None = None, client=None, max_tokens: int = 16000,
                 reasoning_effort: str | None = None):
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_JUDGE_DEPLOYMENT", "judge")
        self._client = client if client is not None else _azure_client()
        self._max_tokens = max_tokens
        self._effort = reasoning_effort or os.environ.get("AZURE_OPENAI_REASONING_EFFORT")
        self.model_id = f"{self.deployment}@{self._label}"

    def judge(self, question: str, answer: str, evidence: Sequence[Evidence]) -> Verdict:
        system, user = render_judge_prompt(question, answer, evidence)
        obj, model, provider = _azure_json(self._client, self.deployment, system, user, "verdict", VERDICT_SCHEMA,
                                           self._max_tokens, self._effort, self._extra_body)
        self.model_id = _served_id(self, model, provider)
        return parse_verdict(obj)


class AzureOpenAIEmbedder:
    """text-embedding-3-large shortened to 1,024 dimensions, the width of the chunks table."""

    dim = 1024

    def __init__(self, deployment: str | None = None, client=None,
                 model: str = "text-embedding-3-large", batch_size: int = 64):
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_EMBEDDER_DEPLOYMENT", "embedder")
        self._client = client if client is not None else _azure_client()
        self._batch_size = batch_size
        # Part of every chunk's content hash, so switching embedders re-embeds the corpus.
        self.model_id = f"{model}-{self.dim}@azure-openai"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start:start + self._batch_size]
            response = self._client.embeddings.create(model=self.deployment, input=batch, dimensions=self.dim)
            vectors += [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        return vectors


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Route only to upstream providers that honour every parameter sent (strict json_schema above all).
OPENROUTER_ROUTING = {"provider": {"require_parameters": True}}


def _openrouter_client():
    from openai import OpenAI
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], max_retries=10)


class OpenRouterGenerator(AzureOpenAIGenerator):
    """Secondary backend: the Azure structured-output generator pointed at OpenRouter, an
    OpenAI-compatible gateway. Useful for trying a model before deploying it on Azure."""

    _label = "openrouter"
    _extra_body = OPENROUTER_ROUTING

    def __init__(self, model: str | None = None, client=None, **kwargs):
        model = model or os.environ.get("OPENROUTER_GENERATOR_MODEL", "openai/gpt-6-luna")
        super().__init__(deployment=model, client=client if client is not None else _openrouter_client(), **kwargs)


class OpenRouterJudge(AzureOpenAIJudge):
    """Secondary backend: the Azure structured-output judge pointed at OpenRouter."""

    _label = "openrouter"
    _extra_body = OPENROUTER_ROUTING

    def __init__(self, model: str | None = None, client=None, **kwargs):
        model = model or os.environ.get("OPENROUTER_JUDGE_MODEL", "openai/gpt-6-luna")
        super().__init__(deployment=model, client=client if client is not None else _openrouter_client(), **kwargs)


# Backend names, Azure OpenAI first: it is the default for every model seam.
# OpenRouter is the secondary, OpenAI-compatible route; the rest run locally or on other APIs.
EMBEDDERS = ("azure-openai", "bge-m3")
GENERATORS = ("azure-openai", "openrouter", "anthropic", "claude-cli", "extractive")
JUDGES = ("azure-openai", "openrouter", "claude-cli", "fake")
DEFAULT_EMBEDDER = DEFAULT_GENERATOR = DEFAULT_JUDGE = "azure-openai"


def make_embedder(kind: str = DEFAULT_EMBEDDER) -> Embedder:
    if kind == "azure-openai":
        return AzureOpenAIEmbedder()
    if kind == "bge-m3":
        return STEmbedder()
    raise ValueError(f"unknown embedder {kind!r}")


def make_generator(kind: str = DEFAULT_GENERATOR) -> Generator:
    if kind == "azure-openai":
        return AzureOpenAIGenerator()
    if kind == "openrouter":
        return OpenRouterGenerator()
    if kind == "anthropic":
        return AnthropicGenerator()
    if kind == "claude-cli":
        return ClaudeCliGenerator()
    if kind == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"unknown generator {kind!r}")


def make_judge(kind: str = DEFAULT_JUDGE) -> Judge:
    if kind == "azure-openai":
        return AzureOpenAIJudge()
    if kind == "openrouter":
        return OpenRouterJudge()
    if kind == "claude-cli":
        return ClaudeCliJudge()
    if kind == "fake":
        return FakeJudge()
    raise ValueError(f"unknown judge {kind!r}")


def load_real() -> tuple[Embedder, Reranker, Generator]:
    """The default real stack: Azure OpenAI embedder and generator, local cross-encoder reranker."""
    return make_embedder(), STReranker(), make_generator()
