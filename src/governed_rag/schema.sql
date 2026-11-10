CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  doc_id        text PRIMARY KEY,
  tier          text NOT NULL CHECK (tier IN ('curated', 'archive')),
  title         text NOT NULL,
  body          text NOT NULL,
  content_hash  text NOT NULL,
  acl           text[] NOT NULL,
  owner         text,
  review_by     date,
  superseded_by text,
  CHECK (tier = 'archive' OR (owner IS NOT NULL AND review_by IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS documents_acl ON documents USING gin (acl);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id    bigserial PRIMARY KEY,
  doc_id      text NOT NULL REFERENCES documents ON DELETE CASCADE,
  ord         int  NOT NULL,
  section     text NOT NULL,
  body        text NOT NULL,
  index_text  text NOT NULL,
  tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', index_text)) STORED,
  embedding   vector(1024) NOT NULL,
  embed_model text NOT NULL,
  UNIQUE (doc_id, ord)
);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
-- m and ef_construction are above pgvector's defaults (16, 64): on 100,000 real embeddings of templated
-- tickets the default graph missed the query's neighbourhood entirely for 3.5 % of queries.
CREATE INDEX IF NOT EXISTS chunks_vec ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 32, ef_construction = 200);

CREATE TABLE IF NOT EXISTS facts (
  key    text PRIMARY KEY,
  doc_id text NOT NULL REFERENCES documents ON DELETE CASCADE,
  value  text NOT NULL,
  unit   text
);

-- Lexemes present in too many chunks to be useful as OR terms; rebuilt by refresh_common_lexemes.
CREATE TABLE IF NOT EXISTS common_lexemes (
  lexeme text PRIMARY KEY,
  ndoc   int NOT NULL
);

CREATE TABLE IF NOT EXISTS query_log (
  id             bigserial PRIMARY KEY,
  ts             timestamptz NOT NULL DEFAULT now(),
  question       text NOT NULL,
  groups         text[] NOT NULL,
  min_authority  int NOT NULL,
  evidence       jsonb NOT NULL,
  authority      int NOT NULL,
  abstain_reason text,
  gate_failures  jsonb NOT NULL,
  latency_ms     int NOT NULL,
  versions       jsonb NOT NULL
);
