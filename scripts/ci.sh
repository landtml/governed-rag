#!/usr/bin/env bash
# Hermetic CI gate: tests, lint, sync and eval, all on the deterministic fakes.
set -euo pipefail

for var in TEST_DATABASE_URL DATABASE_URL; do
  if [[ -z "${!var:-}" ]]; then
    echo "error: $var must be set" >&2
    exit 2
  fi
done

cd "$(dirname "$0")/.."

# The demo corpus is pinned to this date so that the run is reproducible.
CI_TODAY="${CI_TODAY:-2026-11-10}"
RAG=(uv run rag --fake --database-url "$DATABASE_URL" --today "$CI_TODAY")

echo "== sync dependencies =="
uv sync --frozen --inexact

echo "== tests =="
uv run pytest -q

echo "== lint =="
"${RAG[@]}" lint --knowledge demo/knowledge --archive demo/archive

echo "== sync =="
"${RAG[@]}" ingest --knowledge demo/knowledge --archive demo/archive --force

echo "== eval =="
"${RAG[@]}" eval --golden demo/golden.jsonl --baseline eval/baseline-fake.json
