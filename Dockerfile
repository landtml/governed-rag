# The HTTP service (`rag serve`) for Azure Container Apps.
#
# Models: the embedder and generator are Azure OpenAI deployments called with the
# container's managed identity; the cross-encoder reranker runs in the container on
# CPU and is downloaded at build time, so a cold start needs no model download.
#
#   docker build -t governed-rag .
#   az acr build -r <registry> -t governed-rag:<tag> .     # build in Azure instead
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    HF_HOME=/app/.cache/huggingface \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# Dependencies first, so a source change does not reinstall them.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project --extra azure --extra real

COPY src ./src
RUN uv sync --frozen --no-dev --extra azure --extra real

RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

# The fitted threshold tau, tied to the reranker above.
COPY eval/calibration.json eval/calibration.json

RUN useradd --system --uid 10001 --home-dir /app rag && chown -R rag /app/.cache
USER rag

EXPOSE 8080
# The bearer-token file is mounted from a Container Apps secret (infra/app.bicep).
CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8080", "--tokens", "/secrets/tokens.json"]
