#!/usr/bin/env bash
# Deploy the service to Azure Container Apps (infra/app.bicep), on top of the data
# and model layer from infra/deploy.sh.
#
#   1. create the registry, logs, environment and managed identity
#   2. build the image in Azure with `az acr build` (no local Docker needed)
#   3. deploy the container app with that image
#
# The bearer-token file is read from TOKENS_FILE (default tokens.json, gitignored)
# and passed as a secure parameter; it is never written into the repository.
set -euo pipefail
cd "$(dirname "$0")/.."

RG="${RG:-rg-governed-rag}"
TAG="${TAG:-$(git rev-parse --short HEAD)}"
TOKENS_FILE="${TOKENS_FILE:-tokens.json}"

[[ -f .env.azure ]] || { echo "error: run infra/deploy.sh first (.env.azure is missing)" >&2; exit 2; }
[[ -f "$TOKENS_FILE" ]] || { echo "error: $TOKENS_FILE not found; see 'Serving over HTTP' in README.md" >&2; exit 2; }
# shellcheck disable=SC1091
source .env.azure

params=(pgServerName="$PG_SERVER" aiAccountName="$AI_ACCOUNT")

echo "== registry, logs, environment, identity =="
registry=$(az deployment group create -g "$RG" -n governed-rag-app -f infra/app.bicep \
  -p "${params[@]}" deployApp=false --query properties.outputs.registryName.value -o tsv)

echo "== build governed-rag:$TAG in $registry =="
az acr build -r "$registry" -t "governed-rag:$TAG" .

echo "== container app =="
url=$(az deployment group create -g "$RG" -n governed-rag-app -f infra/app.bicep \
  -p "${params[@]}" deployApp=true image="governed-rag:$TAG" tokensJson="$(cat "$TOKENS_FILE")" \
  --query properties.outputs.appUrl.value -o tsv)

echo "Serving at $url  (GET $url/healthz)"
