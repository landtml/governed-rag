from __future__ import annotations

import dataclasses
from datetime import date

from governed_rag.config import Settings
from governed_rag.fakes import FAKE_TAU, FakeEmbedder


def fake_settings(**overrides) -> Settings:
    return dataclasses.replace(Settings(), tau=FAKE_TAU, **overrides)


def seed_doc(conn, doc_id, *, chunks, tier="archive", title=None, acl=("public",),
             owner="owner@example.com", review_by=date(2099, 1, 1), superseded_by=None,
             facts=None) -> None:
    parts = [("", c) if isinstance(c, str) else c for c in chunks]
    bodies = [body for _, body in parts]
    vectors = FakeEmbedder().embed(bodies)
    with conn.transaction():
        conn.execute(
            "INSERT INTO documents (doc_id, tier, title, body, content_hash, acl, owner, review_by, superseded_by) "
            "VALUES (%s, %s, %s, %s, 'seed', %s, %s, %s, %s)",
            (doc_id, tier, title or doc_id, "\n\n".join(bodies), list(acl), owner, review_by, superseded_by))
        for ord_, ((section, body), vec) in enumerate(zip(parts, vectors)):
            conn.execute(
                "INSERT INTO chunks (doc_id, ord, section, body, index_text, embedding, embed_model) "
                "VALUES (%s, %s, %s, %s, %s, %s, 'fake-bow-v1')",
                (doc_id, ord_, section, body, body, vec))
        for key, value in (facts or {}).items():
            conn.execute(
                "INSERT INTO facts (key, doc_id, value, unit) VALUES (%s, %s, %s, NULL)",
                (key, doc_id, value))
