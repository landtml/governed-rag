#!/usr/bin/env bash
# Deploy infra/main.bicep into the resource group made by infra/bootstrap.bicep,
# then write the connection settings to .env.azure (gitignored).
#
# Identity, tenant and IP are looked up at deploy time and passed as parameters,
# so none of them is stored in the repository.
set -euo pipefail
cd "$(dirname "$0")/.."

RG="${RG:-rg-governed-rag}"

object_id=$(az ad signed-in-user show --query id -o tsv)
upn=$(az ad signed-in-user show --query userPrincipalName -o tsv)
client_ip=$(curl -fsS https://api.ipify.org)

echo "Deploying to $RG for $upn from $client_ip"
outputs=$(az deployment group create -g "$RG" -n governed-rag -f infra/main.bicep \
  -p adminObjectId="$object_id" adminPrincipalName="$upn" clientIp="$client_ip" deployAi="${DEPLOY_AI:-true}" deployPg="${DEPLOY_PG:-true}" pgZone="${PG_ZONE:-}" ${PG_LOCATION:+pgLocation="$PG_LOCATION"} \
  --query properties.outputs -o json)

pg_host=$(jq -r .postgresHost.value <<<"$outputs")
ai_endpoint=$(jq -r .aiEndpoint.value <<<"$outputs")
pg_user=$(jq -rn --arg u "$upn" '$u|@uri')

umask 077
cat > .env.azure <<EOF
# Written by infra/deploy.sh. The app fetches its own Entra token for Postgres;
# source infra/env.sh to also get one for psql.
DATABASE_URL=postgresql://${pg_user}@${pg_host}:5432/rag?sslmode=require
AZURE_OPENAI_ENDPOINT=${ai_endpoint}
AZURE_OPENAI_GENERATOR_DEPLOYMENT=generator
AZURE_OPENAI_JUDGE_DEPLOYMENT=judge
AZURE_OPENAI_EMBEDDER_DEPLOYMENT=embedder
PG_SERVER=$(jq -r .postgresServer.value <<<"$outputs")
AI_ACCOUNT=$(jq -r .aiAccount.value <<<"$outputs")
EOF
echo "Wrote .env.azure"
