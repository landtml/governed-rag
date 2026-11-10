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


def make_generator(kind: str) -> Generator:
    if kind == "anthropic":
        return AnthropicGenerator()
    if kind == "claude-cli":
        return ClaudeCliGenerator()
    if kind == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"unknown generator {kind!r}")


def make_judge(kind: str) -> Judge:
    if kind == "claude-cli":
        return ClaudeCliJudge()
    if kind == "fake":
        return FakeJudge()
    raise ValueError(f"unknown judge {kind!r}")


def load_real() -> tuple[Embedder, Reranker, Generator]:
    return STEmbedder(), STReranker(), AnthropicGenerator()
