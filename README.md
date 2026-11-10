<p align="center">
  <img src="assets/icon.png" alt="" width="128">
</p>

<h1 align="center">Governed RAG Reference Architecture</h1>

<p align="center">
  Access-controlled, authority-tiered retrieval with cited answers that are checked before they are returned.
</p>

<p align="center">
  <a href="https://github.com/landtml/governed-rag/actions/workflows/ci.yml"><img src="https://github.com/landtml/governed-rag/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <img src="https://img.shields.io/badge/license-MIT-2B2927" alt="MIT license">
  <img src="https://img.shields.io/badge/python-3.12-2B2927" alt="Python 3.12">
  <img src="https://img.shields.io/badge/PostgreSQL%2017-pgvector%200.8-2B2927" alt="PostgreSQL 17 with pgvector 0.8">
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-a-request-is-handled">How it works</a> ·
  <a href="#serving-over-http">HTTP service</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#measured-results">Measured results</a> ·
  <a href="#known-limits">Known limits</a>
</p>

Governed RAG is a reference implementation of retrieval-augmented generation on PostgreSQL and pgvector. Documents sit in two tiers of authority: curated documents, which have an owner and a review date, rank above the archive. The implementation covers the whole path: ingest with a deletion guard, hybrid retrieval with access control in the query, reranking, generation that must quote its evidence, four verification gates, an HTTP service, and an evaluation kit that fails the build on a regression. It runs end to end on deterministic fakes, so the quickstart and the test suite need no model download and no API key.

**Status.** Version 0.1.1, maintained by one author. The code is tested and the retrieval path is load-tested, but the quality figures come from a synthetic corpus. Treat it as a base to adapt and measure again on your own documents; [Known limits](#known-limits) lists what is unsolved.

## Why I built this

I developed this to solve a problem I kept meeting in retrieval-augmented generation: the demo works, and the answers cannot be checked. The pipelines I reviewed treated a reviewed policy and a three-year-old support ticket as equally good evidence. They applied access rules after retrieval, or left them to the calling application. They returned fluent text with no way to check where a number came from, and they answered questions the documents did not cover.

I wanted one default architecture to start every project from, in which each of those gaps has a specific control and each control is measured on a golden set. I also wanted the numbers to be usable by someone who was not there when they were taken, so the raw report of every run is in [`eval/`](eval/) and the limits are listed next to the results.

The rest of this document describes the system as it is.

## What it does

The pipeline answers a question from a document collection and controls four things that most pipelines leave open.

| Concern | Control |
|---|---|
| Which documents count as authoritative | Two tiers. Curated documents have an owner, a review date and typed facts; archive documents are everything else. A curated document can supersede an archive one, and it loses its rank when its review date passes. |
| Who may see what | Every document carries an access list. The reader's groups are a filter inside both retrieval queries, so a document the reader may not see never reaches the reranker or the model. |
| Whether an answer is grounded | The model must quote its evidence. Four deterministic gates check the draft before it is returned, and the answer takes the authority of its weakest citation: one archive citation makes it an archive answer. |
| Whether to answer at all | A calibrated score threshold, the model's own abstention and the gates can each end a request with a stated reason instead of an answer. |

## How a request is handled

<p align="center">
  <img src="assets/diagram.png" alt="The request path as a flow diagram: documents, filter, retrieve, rerank, threshold, generate, verify, answer. Threshold, generate and verify each have a path to an abstention." width="100%">
</p>

Red marks what carries authority: curated documents, quoted evidence and the answer.

1. **Filter.** The dense and the lexical query both carry the reader's groups, the minimum authority asked for, supersession and the review date. Filtering happens in SQL, before ranking.
2. **Retrieve.** The dense leg takes 50 candidates from an HNSW index by cosine distance. The lexical leg takes 50 from PostgreSQL full-text search, using only the question's rarer words.
3. **Fuse and rerank.** Reciprocal rank fusion merges the two lists and a cross-encoder reranks the best 40.
4. **Decide.** If the best reranker score is below the threshold τ, the request ends with `no_evidence`. τ is a number between 0 and 1 fitted on the golden set; [Evaluation](#evaluation) describes how.
5. **Generate.** The model receives numbered evidence blocks and must return an answer with citations, each holding a verbatim quote. It may abstain.
6. **Verify.** Four gates check the draft. A failed draft gets one retry that names the failures; a second failure ends the request.

| Gate | Rejects a draft when |
|---|---|
| G1 | the answer is empty, has no citation, cites evidence that was not supplied, or uses a marker without a citation |
| G2 | a quote does not appear in the evidence it cites, compared on whole tokens |
| G3 | a number in the answer appears in none of the cited quotes |
| G4 | the answer's authority is below the minimum the caller asked for |

## Quickstart

The quickstart needs Docker and [uv](https://docs.astral.sh/uv/). It uses a deterministic embedder, reranker and generator, so it downloads no model and needs no API key.

```
docker compose up -d --wait db
uv sync
uv run rag --fake ingest --knowledge demo/knowledge --archive demo/archive
uv run rag --fake ask "What is our refund window?"
uv run rag --fake eval --golden demo/golden.jsonl
```

The other commands are `search` (evidence only), `fact` (a typed fact from a curated document), `lint` (check curated documents), `serve` and `backlog` (questions that no curated document answered). `uv run rag <command> --help` lists the options.

## Documents

Documents are Markdown files. A curated document declares its governance in front matter:

```markdown
---
title: Refund policy
type: policy
owner: billing-team
review_by: 2027-06-01
acl: [public]
supersedes: [archive/policies/refund-policy-2023]
facts:
  refund_window_days: {value: 14, unit: days}
---
# Refund policy

The refund window is 14 days from purchase.
```

An archive document needs no front matter; `acl: [support]` restricts it to a group. `rag lint` rejects a curated document without an owner or review date, and `rag ingest` synchronises both trees with the database. Ingest refuses mass deletions: it stops if a run would remove every document of a tier, or more than 20 % of them when that is more than 10 documents. `--force` confirms an intended deletion.

## Real models

```
uv sync --extra real
export ANTHROPIC_API_KEY=...
uv run rag ingest --knowledge demo/knowledge --archive demo/archive
uv run rag ask "What is our refund window?"
```

Without `--fake`, the embedder is `BAAI/bge-m3` and the reranker `BAAI/bge-reranker-v2-m3` through sentence-transformers; the first run downloads both. Generation has three backends, chosen with `--generator`:

| Backend | Use |
|---|---|
| `anthropic` (default) | The Anthropic API, with `ANTHROPIC_API_KEY`. |
| `claude-cli` | The Claude Code CLI (`claude -p`). It needs `claude` on `PATH` or `CLAUDE_CLI=/path/to/claude`. |
| `extractive` | No model call. It returns the best evidence sentence, which is enough to calibrate τ at no cost. |

On an AMD GPU, `scripts/gpu.sh rag ...` runs the same commands in a ROCm environment; the setup lines are at the top of that script.

## Serving over HTTP

```
echo '{"a-long-random-token-for-alice": ["support", "finance"]}' > tokens.json
uv run rag serve --tokens tokens.json --port 8080
curl -s localhost:8080/answer -H 'Authorization: Bearer a-long-random-token-for-alice' \
  -d '{"question": "What is our refund window?"}'
```

| Route | Body | Returns |
|---|---|---|
| `POST /search` | `question`, optional `min_authority` (`archive` or `curated`) | Evidence blocks, top score, abstain reason |
| `POST /answer` | the same | Answer text, its authority, citations with quotes, abstain reason |
| `POST /fact` | `key` | A typed fact (`key`, `value`, `unit`) or `null` |
| `GET /healthz` | | `{"status": "ok"}` when the database answers |

The reader's groups come only from the token file on the server, never from the request. A request without a valid token gets 401. `--anonymous-public` lets requests with no `Authorization` header read public documents; a malformed header is still refused. The server binds to 127.0.0.1 by default, speaks plain HTTP and stops cleanly on SIGTERM, so it belongs behind a proxy that adds TLS. `--pool-size` (default 4) is the number of database connections. One connection is held for the whole of an `/answer` call, so the pool size also caps answers in flight.

## Evaluation

A golden set is a JSON Lines file with one case per line:

```json
{"id": "sev1-paging", "question": "When must a sev1 incident be paged?", "groups": ["support"], "answerable": true, "tags": ["exact", "curated"], "gold_docs": ["kb/internal/escalation"], "must_contain": ["15"]}
```

`rag eval` runs every case and reports two kinds of metric.

- **Hard gates** fail the run outright: a document from `forbidden_docs` in the evidence, or an `exact` case whose answer lacks a `must_contain` value.
- **Soft metrics** carry a bootstrap 95 % interval and fail the run only when the upper end of that interval is below the value in a committed baseline: gold document retrieved, curated document ranked first, abstention recall and precision, near-miss abstention, and first-draft gate pass rate.

```
uv run rag eval --golden bench/golden.jsonl --baseline eval/baseline.json
uv run rag eval --golden bench/golden.jsonl --write-baseline eval/baseline.json --write-calibration eval/calibration.json
```

The second command writes a new baseline and fits τ. Cases are marked `"split": "fit"` or `"holdout"`: τ is fitted on the fit half, and the report shows how the same τ does on the other half and on the cases tagged `near_miss`. The calibration file records the reranker it was fitted with, and the pipeline refuses a file fitted with a different one.

`--judge claude-cli` adds an entailment check by a second model call: is every claim in the answer supported by the evidence it cites? As a control, each answer is also judged against the cited evidence of a different case, which a working judge must reject. The judge is reported and never fails a run.

`--workers 6` answers several cases at once. Latency in that report then includes waiting for a local model, so latency figures should come from a run with one worker.

## Measured results

All figures were measured on 2026-11-10 with real models, and every run was given `--today 2026-11-10`. The raw reports are in [`eval/`](eval/).

**Setup.** Embedder `BAAI/bge-m3` and reranker `BAAI/bge-reranker-v2-m3` on a Radeon RX 6750 XT under ROCm. Generator and judge `claude-sonnet-5-5` at medium effort through the Claude Code CLI. Intel Core Ultra 7 265KF with 30 GB RAM, with all work pinned to the 8 performance cores, the PostgreSQL container included. PostgreSQL 17.11 with pgvector 0.8.7 in Docker, `shared_buffers` 128 MB.

### Answer quality

The benchmark corpus is synthetic: 2,124 documents of a fictional software company (60 curated, 2,064 archive) in 2,172 chunks, generated by `bench/make_corpus.py`. The golden set `bench/golden.jsonl` has 330 cases, 175 answerable and 155 unanswerable. Thirty of the unanswerable cases are near misses, which ask about a covered topic for a fact its document does not state. The report is `eval/baseline.json`.

| Metric | Result | 95 % bound or interval | Cases |
|---|---|---|---|
| Forbidden-document leaks (hard gate) | 0 | at most 0.009 | 330 |
| Exact-value accuracy (hard gate) | 1.000 | at least 0.945 | 55 |
| Gold document in the evidence | 0.989 | 0.971 to 1.000 | 175 |
| Curated document ranked first | 0.988 | 0.965 to 1.000 | 85 |
| Unanswerable cases abstained | 1.000 | at least 0.981 | 155 |
| Near-miss cases abstained | 1.000 | at least 0.900 | 30 |
| Abstentions that were correct | 0.987 | 0.968 to 1.000 | 157 |
| Verification gates passed on the first draft | 0.954 | 0.919 to 0.983 | 173 |
| Answers the judge found fully supported | 0.983 | 0.960 to 1.000 | 173 |
| Judge control: answer rejected against another case's evidence | 1.000 | at least 0.983 | 173 |

Intervals are bootstrap intervals. A metric with no failure has a bootstrap interval of zero width, which says nothing, so those rows give the rule-of-three bound instead: with no failure in n cases, the true failure rate is below 3/n at 95 % confidence.

Four observations qualify these figures.

- The two wrong abstentions are answerable handbook questions on the hold-out split whose best reranker score (0.006 and 0.007) falls below τ. They are the cost of fitting τ on half the cases, and τ was left as fitted.
- The threshold does not stop near misses. Twenty-nine of the 30 score above τ, and the generator abstained on all 29. That abstention depends on the model following its prompt, with no deterministic check behind it, and 30 cases bound the failure rate only loosely, at 10 %.
- The judge flagged three answers, all on archive tickets. Two add a generalisation about the sources ("the same fix was recorded across many tickets") and one says customers confirmed a fix. The gates check quotes and numbers, so they cannot see such claims. The judge is the same model as the generator.
- The judge rejected all 173 control pairs, so its 0.983 is more than agreement by default.

With one worker, the median request took 3.1 s and the 95th percentile 4.6 s (`eval/latency-workers1.json`, 33 cases). Generation accounts for nearly all of it.

### Calibration

`eval/calibration.json` holds τ = 0.0103, fitted with `--generator extractive` on the fit half of the golden set without the near-miss cases.

| | Answerable kept (score ≥ τ) | Unanswerable passed (score ≥ τ) |
|---|---|---|
| Fit split | 88 of 88 | 0 of 63 |
| Hold-out split | 85 of 87 | 0 of 62 |
| Near-miss cases | | 29 of 30 |

τ separates off-topic and access-filtered questions from answerable ones, including on cases it was not fitted on. It does nothing against a question on a covered topic, as the last row shows.

### Retrieval at 500,000 chunks

`bench/loadtest.py` fills a database with 500,000 synthetic chunks in 125,000 documents (1,024 dimensions, 250 tight clusters). Recall is the dense leg's 50 candidates against exact search over 100 queries. Latency is the full hybrid retrieval call under open-loop load for 60 s, timed from the scheduled arrival. Reports: `eval/loadtest-500000.json` and `eval/loadtest-500000-qps10.json`.

| Hybrid retrieval | p50 | p95 | p99 | Errors |
|---|---|---|---|---|
| 10 queries/s | 30 ms | 41 ms | 51 ms | 0 of 600 |
| 50 queries/s | 30 ms | 40 ms | 50 ms | 0 of 3,000 |

Three rules produce these numbers, and each one answers a measured failure.

- **Common words are left out of the lexical query.** A word found in more than 2 % of chunks, and in more than 1,000, is dropped from the OR query; that is 185 words at this size. With the rule switched off, p95 at 50 queries/s is 329 ms (`eval/sweeps/common-fraction-1.0.json`).
- **Retrieval never starts parallel workers.** Under load they exhaust the shared memory of a default container (`/dev/shm` of 64 MB) and queries fail. `compose.yaml` also raises `shm_size` to 1 GB.
- **Both queries are planned on every call.** PostgreSQL otherwise switches to a cached generic plan after a few calls on a connection, which ignores the HNSW index for some settings and slowed retrieval by a factor of ten.

`hnsw.ef_search` is a setting with a default of 200, set on every query. In a sweep, mean dense recall was 0.852 at 50, 0.917 at 200 and 0.943 at 400, and 200 was the lowest value that lifted the 5th percentile of recall from 0.02 to 0.86. The 500,000-chunk index in this test uses pgvector's default graph (`m` 16, `ef_construction` 64); the next section explains why the schema no longer does.

### Recall on real embeddings

At every `ef_search` value a few queries had recall near zero. To see whether the synthetic clusters caused that, the test was repeated on 100,000 `bge-m3` embeddings of distinct synthetic tickets, with 200 further tickets as queries (`bench/embed_tickets.py`, then `bench/loadtest.py --vectors`). The clusters were not the cause. With the default graph, a few queries per build still returned 50 chunks at cosine distance 0.26 to 0.36 while the true neighbours lay within 0.07, which means the search never reached the right region of the graph.

A denser graph removes most of these misses, and `schema.sql` builds the index with `m = 32, ef_construction = 200`. Over four index builds per setting, the default graph lost 1 to 7 of 200 queries per build (mean recall 0.921 to 0.964). The denser graph lost none in three builds and 3 in one (mean recall 0.985 to 1.000), for 0.7 ms more per query and a build time of 66 s against 36 s. Recall depends on the build because graph construction is random.

Hybrid retrieval on this database at 10 queries/s had a p50 of 8 ms and a p95 of 12 ms. The sweep tables and the per-build figures are in [`eval/README.md`](eval/README.md).

## Known limits

- **Synthetic data.** The quality figures come from a generated corpus of 2,172 chunks. A real project needs its own golden set and its own τ before any of them apply.
- **Near misses.** When the topic is covered and the specific fact is absent, only the model's abstention prevents an answer.
- **Recall tail.** The denser HNSW graph reduces queries with near-zero recall; one build in four still had some. The ticket texts come from templates and include many near duplicates.
- **`ef_search` ceiling.** At 100,000 chunks a value of 400 made the planner abandon the index for an exact scan, 62 ms against 4 ms.
- **Common-word rule on small corpora.** The 1,000-chunk floor is low enough that five words already count as common at 2,172 chunks. The retrieval metrics did not move.
- **Throughput.** The local models serve one request at a time, and an answer holds a database connection for the length of the model call.
- **Markdown only.** Other formats need a converter in front of `rag ingest`.
- **Existing databases.** `schema.sql` creates the index only when it is absent. A database built with the older graph keeps it until `chunks_vec` is dropped and the schema is applied again.

## Configuration

Defaults are in [`src/governed_rag/config.py`](src/governed_rag/config.py). The database URL comes from `DATABASE_URL` or `--database-url`.

| Setting | Default | Meaning |
|---|---|---|
| `candidates_per_leg` | 50 | Chunks each retrieval leg returns |
| `rrf_k` | 60 | Reciprocal rank fusion constant |
| `rerank_n` | 40 | Fused candidates given to the reranker |
| `hnsw_ef_search` | 200 | HNSW search breadth, never below `candidates_per_leg` |
| `lexical_common_fraction`, `lexical_common_floor` | 0.02, 1000 | A word is left out of the lexical query when it appears in more than 2 % of chunks and in more than 1,000 chunks |
| `max_parents`, `evidence_budget_tokens` | 8, 6000 | Size of the evidence given to the model |
| `gate_retries` | 1 | Retries after a failed gate |
| `tau` | 0.30 | The score threshold τ. This default is provisional; a committed `eval/calibration.json` replaces it |
| `max_delete_fraction`, `delete_guard_floor` | 0.2, 10 | Ingest deletion guard |

## Repository layout

```
src/governed_rag/   pipeline: ingest, retrieve, evidence, prompt, gates, answer, evalkit, service, cli
tests/              243 tests on deterministic fakes; tests that need real models carry the `real` marker
bench/              benchmark corpus generator, 330-case golden set, load test, ticket embedder
demo/               an 8-document corpus and 11-case golden set for the quickstart and CI
eval/               committed baseline, calibration and raw measurement reports
scripts/            ci.sh (the CI gate) and gpu.sh (ROCm launcher)
```

## Tests and CI

```
uv run pytest -q
```

The tests run against the PostgreSQL database on port 5433 (`TEST_DATABASE_URL` overrides it) and need no model or API key. `scripts/ci.sh` is the full gate: tests, document lint, ingest and the demo evaluation against `eval/baseline-fake.json`. The GitHub workflow in `.github/workflows/ci.yml` runs the same script.

## Security

The system enforces three properties. The first two are covered by tests on forbidden documents and on token handling.

| Property | Where it is enforced |
|---|---|
| A reader sees only documents whose access list includes one of the reader's groups | In the SQL of both retrieval queries and of the fact lookup, before ranking |
| A caller cannot choose its own groups | `rag serve` maps a bearer token to groups from a file on the server and ignores any groups in the request; an unknown or malformed token gets 401 |
| Retrieved text cannot issue instructions | Evidence is passed as data in delimited blocks, and the gates reject an answer whose quotes or numbers are not in that evidence |

The deployer has to supply the rest. `rag serve` speaks plain HTTP on 127.0.0.1, with no TLS and no rate limiting, so it belongs behind a proxy that adds both. The token file holds credentials in clear text and must stay out of version control. The development database in `compose.yaml` uses the user and password `rag` and must not be exposed. Prompt injection through a document is reduced by the two measures in the table, and no test in this repository shows it is prevented.

[`SECURITY.md`](SECURITY.md) describes how to report a vulnerability.

## License

MIT. See [`LICENSE`](LICENSE).
