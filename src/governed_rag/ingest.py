from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import psycopg
import yaml

from governed_rag.chunking import chunk_markdown
from governed_rag.config import Settings
from governed_rag.curated import LintIssue, lint_tree, load_tree, split_front_matter
from governed_rag.retrieve import refresh_common_lexemes
from governed_rag.types import Embedder, Fact

_H1 = re.compile(r"^# (.+)$", re.MULTILINE)
_PUBLIC = ("public",)


@dataclass(frozen=True)
class SyncReport:
    added: int
    updated: int
    unchanged: int
    deleted: int


class SyncAborted(Exception):
    def __init__(self, issues: list[LintIssue]):
        super().__init__("; ".join(f"{i.rule} {i.doc_id}: {i.message}" for i in issues))
        self.issues = issues


class DeleteGuard(Exception):
    def __init__(self, tier: str, removed: int, existing: int):
        super().__init__(f"would delete {removed} of {existing} {tier} documents; "
                         "pass force=True to confirm")
        self.tier, self.removed, self.existing = tier, removed, existing


@dataclass(frozen=True)
class _Doc:
    doc_id: str
    tier: str
    title: str
    body: str
    content_hash: str
    acl: tuple[str, ...]
    owner: str | None = None
    review_by: date | None = None
    facts: tuple[Fact, ...] = ()
    supersedes: tuple[str, ...] = ()


def sync(conn: psycopg.Connection, knowledge_root: Path | None, archive_root: Path | None,
         embedder: Embedder, settings: Settings, today: date, force: bool = False) -> SyncReport:
    model_id = embedder.model_id

    # 1. Lint. Everything that can refuse the sync happens before the first write.
    if knowledge_root is not None:
        errors = [i for i in lint_tree(knowledge_root, archive_root, today) if i.severity == "error"]
        if errors:
            raise SyncAborted(errors)

    # 2. Scan. An archive file whose front matter is unusable refuses the sync (fail closed).
    scanned: list[_Doc] = []
    if knowledge_root is not None:
        scanned += _curated_docs(knowledge_root, model_id)
    if archive_root is not None:
        archive, archive_issues = _archive_docs(archive_root, model_id)
        if archive_issues:
            raise SyncAborted(archive_issues)
        scanned += archive
    scanned.sort(key=lambda d: d.doc_id)
    supersession = {target: d.doc_id for d in scanned if d.tier == "curated" for target in d.supersedes}

    stored = {r["doc_id"]: r for r in conn.execute("SELECT doc_id, tier, content_hash FROM documents")}
    to_write, added, updated, unchanged = [], 0, 0, 0
    for d in scanned:
        row = stored.get(d.doc_id)
        if row is None:
            added += 1
            to_write.append(d)
        elif row["content_hash"] != d.content_hash:
            updated += 1
            to_write.append(d)
        else:
            unchanged += 1

    # 3. Guard deletions.
    removed_ids: list[str] = []
    for tier, root in (("curated", knowledge_root), ("archive", archive_root)):
        if root is None:
            continue
        present = {d.doc_id for d in scanned if d.tier == tier}
        existing = [doc_id for doc_id, r in stored.items() if r["tier"] == tier]
        gone = [doc_id for doc_id in existing if doc_id not in present]
        n, removed = len(existing), len(gone)
        if not force and (removed == n > 0 or
                          (removed > settings.delete_guard_floor and removed / n > settings.max_delete_fraction)):
            raise DeleteGuard(tier, removed, n)
        removed_ids += gone

    # 4. Supersession, before any document is written.
    if knowledge_root is not None:
        with conn.transaction():
            conn.execute("UPDATE documents SET superseded_by = NULL WHERE superseded_by IS NOT NULL")
            for target, by in supersession.items():
                conn.execute("UPDATE documents SET superseded_by = %s WHERE doc_id = %s", (by, target))

    # 5. Write added and updated documents, one transaction each. Embedding precedes the transaction.
    for d in to_write:
        chunks = chunk_markdown(d.title, d.body, settings.chunk_target_tokens, settings.chunk_max_tokens)
        vectors = embedder.embed([c.index_text for c in chunks]) if chunks else []
        with conn.transaction():
            # superseded_by is left out of the update clause so an existing row keeps its value.
            conn.execute(
                "INSERT INTO documents (doc_id, tier, title, body, content_hash, acl, owner, review_by, superseded_by) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (doc_id) DO UPDATE SET tier = EXCLUDED.tier, title = EXCLUDED.title, "
                "body = EXCLUDED.body, content_hash = EXCLUDED.content_hash, acl = EXCLUDED.acl, "
                "owner = EXCLUDED.owner, review_by = EXCLUDED.review_by",
                (d.doc_id, d.tier, d.title, d.body, d.content_hash, list(d.acl), d.owner, d.review_by,
                 supersession.get(d.doc_id)))
            conn.execute("DELETE FROM chunks WHERE doc_id = %s", (d.doc_id,))
            conn.execute("DELETE FROM facts WHERE doc_id = %s", (d.doc_id,))
            for c, vec in zip(chunks, vectors):
                conn.execute(
                    "INSERT INTO chunks (doc_id, ord, section, body, index_text, embedding, embed_model) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (d.doc_id, c.ord, c.section, c.body, c.index_text, vec, model_id))
            # A fact key that moved from another document is taken over, not duplicated.
            for f in d.facts:
                conn.execute(
                    "INSERT INTO facts (key, doc_id, value, unit) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET doc_id = EXCLUDED.doc_id, value = EXCLUDED.value, "
                    "unit = EXCLUDED.unit",
                    (f.key, d.doc_id, f.value, f.unit))

    # 6. Delete removed documents.
    if removed_ids:
        with conn.transaction():
            for doc_id in removed_ids:
                conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

    # 7. Only a sync that got this far refreshes the lexical statistics; an aborted one leaves them alone.
    refresh_common_lexemes(conn, settings)
    return SyncReport(added=added, updated=updated, unchanged=unchanged, deleted=len(removed_ids))


def _content_hash(raw: bytes, model_id: str) -> str:
    return hashlib.sha256(raw + b"\0" + model_id.encode()).hexdigest()


def _curated_docs(root: Path, model_id: str) -> list[_Doc]:
    docs = []
    for doc in load_tree(root):
        raw = (root / (doc.doc_id[3:] + ".md")).read_bytes()
        docs.append(_Doc(doc.doc_id, "curated", doc.title, doc.body, _content_hash(raw, model_id),
                         doc.acl, doc.owner, doc.review_by, doc.facts, doc.supersedes))
    return docs


def _archive_docs(root: Path, model_id: str) -> tuple[list[_Doc], list[LintIssue]]:
    paths = sorted({p for pattern in ("**/*.md", "**/*.txt") for p in root.glob(pattern) if p.is_file()})
    docs: list[_Doc] = []
    issues: list[LintIssue] = []
    for path in paths:
        doc_id = "archive/" + path.relative_to(root).with_suffix("").as_posix()
        raw = path.read_bytes()
        text = raw.decode("utf-8").removeprefix("﻿")   # a BOM must not hide the front matter
        try:
            meta, body = split_front_matter(text)
        except (ValueError, yaml.YAMLError) as e:           # a broken block is an error, never body text
            issues.append(LintIssue("A1", "error", doc_id, f"invalid front matter: {e}"))
            continue
        body = body.strip()

        # Only an absent acl key defaults to public; a present but unusable one refuses the sync.
        if "acl" in meta:
            acl = meta["acl"]
            if not (isinstance(acl, list) and acl and all(isinstance(a, str) for a in acl)):
                issues.append(LintIssue("A1", "error", doc_id, "acl must be a non-empty list of strings"))
                continue
        else:
            acl = list(_PUBLIC)

        title = meta.get("title")
        if not (isinstance(title, str) and title.strip()):
            heading = _H1.search(body)
            title = heading.group(1).strip() if heading and heading.group(1).strip() else path.stem
        docs.append(_Doc(doc_id, "archive", title.strip(), body, _content_hash(raw, model_id), tuple(acl)))
    return docs, issues
