# Measurement reports

Every file here is the unedited output of a run on 2026-11-10. The main README gives the setup and the summary; this page indexes the files and holds the detailed tables.

| File | Produced by | Content |
|---|---|---|
| `baseline.json` | `rag eval --write-baseline`, real models, with `--judge claude-cli` | The 330-case benchmark: hard gates, soft metrics with intervals, calibration statistics, judge result |
| `calibration.json` | `rag eval --write-calibration`, `--generator extractive` | The fitted threshold, the reranker it belongs to, hold-out and near-miss statistics |
| `baseline-fake.json` | `rag --fake eval` on `demo/golden.jsonl` | The baseline the CI gate compares against |
| `latency-workers1.json` | `rag eval --workers 1` on every tenth case | End-to-end latency with one worker |
| `loadtest-500000.json`, `loadtest-500000-qps10.json` | `bench/loadtest.py`, 500,000 synthetic chunks | Dense recall against exact search, hybrid retrieval latency at 50 and 10 queries/s |
| `loadtest-real-100000.json`, `loadtest-real-100000-m16.json` | `bench/loadtest.py --vectors`, 100,000 real embeddings | The same measurements with the current index settings and with pgvector's default graph |
| `hnsw-builds-real-100000.json` | Repeated index builds on the same data | Recall per build for both graph settings |
| `sweeps/` | `bench/loadtest.py --ef-search` and `--common-fraction` | One report per setting value |

## `hnsw.ef_search` at 500,000 chunks

Dense leg, public reader, archive tier, 100 queries, 50 candidates, pgvector's default graph.

| `ef_search` | Mean recall | 5th percentile | Queries below 0.05 | Dense leg p50 |
|---|---|---|---|---|
| 50 | 0.852 | 0.02 | 5 | 13.9 ms |
| 100 | 0.864 | 0.02 | 5 | 5.6 ms |
| 200 | 0.917 | 0.86 | 3 | 6.5 ms |
| 400 | 0.943 | 0.90 | 2 | 8.3 ms |

A value of 50 equals the number of candidates asked for, so the scan has to resume to fill the list, which is the likely reason it is the slowest row.

## Common-word fraction at 500,000 chunks

Hybrid retrieval at 50 queries/s for 60 s. A word is left out of the lexical query when it appears in more than this fraction of chunks. A fraction of 1.0 switches the rule off.

| Fraction | Words left out | p50 | p95 | p99 |
|---|---|---|---|---|
| 0.005 | 775 | 20 ms | 31 ms | 32 ms |
| 0.01 | 383 | 22 ms | 33 ms | 37 ms |
| 0.02 (default) | 185 | 30 ms | 41 ms | 51 ms |
| 0.05 | 67 | 35 ms | 68 ms | 85 ms |
| 1.0 | 0 | 51 ms | 329 ms | 385 ms |

The load test's vocabulary is synthetic, so these runs measure cost only. The default of 0.02 was kept because lower values drop more of a question's words, and the effect of that on answer quality was measured only at 0.02.

## HNSW graph settings on real embeddings

100,000 `bge-m3` embeddings of distinct synthetic tickets, 200 held-out tickets as queries, `ef_search` 200. Each row is a separate index build on the same data.

| Index | Build | Mean recall | 5th percentile | Queries below 0.05 (of 200) | Dense leg p50 |
|---|---|---|---|---|---|
| `m` 16, `ef_construction` 64 | load-test fill | 0.921 | 0.16 | 7 | 3.5 ms |
| `m` 16, `ef_construction` 64 | rebuild 1 | 0.922 | 0.32 | 2 | 3.4 ms |
| `m` 16, `ef_construction` 64 | rebuild 2 | 0.964 | 0.84 | 1 | 3.5 ms |
| `m` 16, `ef_construction` 64 | rebuild 3 | 0.951 | 0.84 | 3 | 3.4 ms |
| `m` 32, `ef_construction` 200 | load-test fill | 0.985 | 1.00 | 3 | 4.1 ms |
| `m` 32, `ef_construction` 200 | rebuild 1 | 1.000 | 1.00 | 0 | 4.2 ms |
| `m` 32, `ef_construction` 200 | rebuild 2 | 1.000 | 1.00 | 0 | 4.2 ms |
| `m` 32, `ef_construction` 200 | rebuild 3 | 1.000 | 1.00 | 0 | 4.3 ms |

For the queries with recall below 0.05, the index returned 50 chunks at cosine distance 0.26 to 0.36 while the true neighbours lay within 0.07. The misses are whole neighbourhoods the search never reached, and ties between near-duplicate vectors do not explain them. In both settings the build made during the load-test fill was the worst of its group; the cause is not known.
