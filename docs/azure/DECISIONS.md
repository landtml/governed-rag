# Azure deployment: decisions

One entry per decision. Each says what was chosen and why, in a sentence or
two, with the evidence where there is some.

## D1. Keep pgvector; do not move retrieval to Azure AI Search

Access control is applied inside the retrieval SQL (`acl` filter on both the
vector and lexical legs, `src/governed_rag/schema.sql`). Moving to AI Search
would re-implement that filter in another engine and throw away the measured
results, so the same Postgres schema runs on Azure Database for PostgreSQL.

## D2. Region: Sweden Central

It offers every model the deployment needs, including `text-embedding-3-large`
as a regional Standard deployment (Learn: "Region availability for Foundry
Models sold by Azure"). The Postgres region is a separate parameter
(`pgLocation`), so the database can sit in another region if needed.

## D3. Read the quota from the subscription, not from a table

Azure OpenAI quota is set per subscription, region, model and deployment
type. The reliable source is the usage API for the subscription in question:

```
az cognitiveservices usage list -l swedencentral
```

The deployment capacities in `infra/main.bicep` are parameters, so they can
be sized to what that command reports.

## D4. Budget and resource group are code too

`infra/bootstrap.bicep` (subscription scope) creates `rg-governed-rag` and a
monthly budget that emails at 50%, 80%, 100% actual and 100% forecast. It is
deployed before any billable resource. The alert address is a command-line
parameter so it never enters the repo. A budget alerts; it does not stop
spending.

## D5. Postgres: PostgreSQL 17, Burstable B1ms, Entra ID only

PostgreSQL 17 because Azure ships 17.11, the exact version of the local
baseline, so a difference in results cannot come from the engine. Burstable
B1ms with 32 GB is the smallest tier that runs the benchmark. Password
authentication is disabled: the only login is a Microsoft Entra token, so no
database password exists to leak or rotate. Public access is limited by
firewall rules (the developer IP, and Azure-hosted clients for the container
app, D14).

## D6. Models in a Foundry resource with key auth disabled

The models live in an Azure AI Foundry resource (`kind: AIServices`) with
`disableLocalAuth: true`. Code gets a token from `DefaultAzureCredential`:
the Azure CLI login on a laptop, a managed identity inside Azure, with no
code change between them. Callers hold only "Cognitive Services OpenAI User"
on the resource: call models, not change them.

## D7. Embedder in-region

The embedder (`text-embedding-3-large` at 1,024 dimensions, which fits
`vector(1024)` without a migration) is a regional Standard deployment in
Sweden Central, so document text is embedded in-region. The deployment type
of the chat models is covered in D11.

## D8. Pin model versions

`versionUpgradeOption: NoAutoUpgrade` on every deployment. A provider-side
model upgrade would change answers without any change in the repo; with
pinning, a new version has to come in through a redeploy that reruns the eval.

## D9. Structured outputs in strict mode

The Azure backends send the existing `DRAFT_SCHEMA` and `VERDICT_SCHEMA`
with `strict: true`, so the reply is guaranteed to parse. A refusal, a
truncated reply or a content-filter stop raises instead of returning partial
JSON, and the existing gates see an error rather than a wrong answer.

## D10. Keep the reranker next to the service

Azure OpenAI has no cross-encoder reranker. `bge-reranker-v2-m3` runs inside
the service (on CPU in the container image, or on a local GPU), which also
keeps the calibrated threshold τ valid: `calibrated()` ties τ to the reranker
and refuses a file fitted to another one.

## D11. Generator and judge: gpt-5-mini on Global Standard

gpt-5-mini is a current reasoning model with structured-output support. It
runs as a Global Standard deployment, which has the most capacity; the
trade-off is that Microsoft may process those prompts in any region. A
deployment that must keep inference in the EU uses a Data Zone or regional
deployment instead, which is a change to the `sku` in `infra/main.bicep`.

Reasoning models take `max_completion_tokens` (which also counts hidden
reasoning tokens) and reject `temperature`, so the backends send no
temperature and reruns are not token-for-token reproducible.
`AZURE_OPENAI_REASONING_EFFORT` sets the effort when needed.

## D12. Azure OpenAI is the default backend; OpenRouter is the secondary one

Every model seam (`--generator`, `--embedder`, `--judge`) defaults to
`azure-openai`. OpenRouter, an OpenAI-compatible gateway, reuses the same
structured-output code path and is selected with `--generator openrouter` or
`--judge openrouter`; it is useful for trying a model before deploying it.
Every eval report records the backend and the model that served it, so a
comparison file states its own provenance.

## D13. A fresh Entra token for every Postgres connection

An Entra access token is valid for about an hour, while `rag serve` keeps a
connection pool open for days. `db.connect` therefore asks
`DefaultAzureCredential` for a token each time it opens a connection to a
`*.postgres.database.azure.com` host whose URL has no password. The
credential caches and refreshes tokens, so this costs no extra round trip in
the common case.

## D14. Container Apps with a user-assigned managed identity

`infra/app.bicep` runs `rag serve` on Azure Container Apps (Consumption plan,
scale to zero, HTTPS ingress), with the image in Azure Container Registry
built by `az acr build`. One user-assigned identity does everything: it pulls
the image (AcrPull), calls the model deployments (OpenAI User) and logs in to
Postgres as an Entra principal. The registry has its admin user disabled.
The bearer-token file is a Container Apps secret mounted as a file, and
container logs go to a Log Analytics workspace.

## D15. The app identity is a Postgres Entra administrator

The service applies its schema at start-up, which needs owner rights, so the
identity is added as an Entra administrator by Bicep and no SQL step is part
of the deployment. A narrower setup creates a separate role with
`pgaadauth_create_principal` and runs migrations from a deploy job.
Consumption-plan apps have no fixed outbound address, so the server allows
Azure-hosted clients (`allowAzureServicesToPostgres`); login still requires a
token from the tenant.
