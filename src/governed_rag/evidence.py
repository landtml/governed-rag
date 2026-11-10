from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from psycopg import Connection

from governed_rag.config import Settings
from governed_rag.retrieve import candidates
from governed_rag.text import count_tokens
from governed_rag.types import Authority, Candidate, Embedder, Evidence, Principal, Reranker, SearchResult

GAP = "[…]"


@dataclass(frozen=True)
class _Parent:
    authority: Authority
    doc_id: str
    section: str
    score: float
    title: str
    text: str
    tokens: int


def search(conn: Connection, query: str, principal: Principal, *, embedder: Embedder,
           reranker: Reranker, settings: Settings, today: date,
           min_authority: Authority = Authority.ARCHIVE) -> SearchResult:
    vec = embedder.embed([query])[0]
    cands = candidates(conn, query, vec, principal, min_authority, today, settings)
    if not cands:
        return SearchResult((), 0.0, "no_evidence")

    scores = reranker.score(query, [c.index_text for c in cands])
    top_score = max(scores)
    hits = [(c, s) for c, s in zip(cands, scores) if s >= settings.tau]
    if not hits:
        return SearchResult((), top_score, "no_evidence")

    groups: dict[tuple[Authority, str, str], list[tuple[Candidate, float]]] = {}
    for c, s in hits:
        if c.authority == Authority.CURATED:
            key = (Authority.CURATED, c.doc_id, "")
        else:
            key = (Authority.ARCHIVE, c.doc_id, c.section)
        groups.setdefault(key, []).append((c, s))

    parents = [
        _curated_parent(conn, doc_id, max(s for _, s in group))
        if authority == Authority.CURATED
        else _archive_parent(conn, doc_id, section, group, settings)
        for (authority, doc_id, section), group in groups.items()
    ]
    parents.sort(key=lambda p: (p.authority != Authority.CURATED, -p.score, p.doc_id, p.section))

    taken: list[_Parent] = []
    budget = settings.evidence_budget_tokens
    for p in parents:
        if len(taken) < settings.max_parents and p.tokens <= budget:
            taken.append(p)
            budget -= p.tokens
    if not taken:
        return SearchResult((), top_score, "no_evidence")

    evidence = tuple(
        Evidence(f"E{i}", p.doc_id, p.title, p.authority, p.text, p.score)
        for i, p in enumerate(taken, start=1))
    return SearchResult(evidence, top_score, None)


def _curated_parent(conn: Connection, doc_id: str, score: float) -> _Parent:
    doc = conn.execute("SELECT title, body FROM documents WHERE doc_id = %s", (doc_id,)).fetchone()
    facts = conn.execute(
        "SELECT key, value, unit FROM facts WHERE doc_id = %s ORDER BY key", (doc_id,)).fetchall()
    lines = [f"FACT {f['key']} = {f['value']}" + ("" if f["unit"] is None else f" {f['unit']}")
             for f in facts]
    text = doc["body"] if not lines else "\n".join(lines) + "\n\n" + doc["body"]
    return _Parent(Authority.CURATED, doc_id, "", score, doc["title"], text, count_tokens(text))


def _archive_parent(conn: Connection, doc_id: str, section: str,
                    group: list[tuple[Candidate, float]], settings: Settings) -> _Parent:
    rows = conn.execute(
        "SELECT ord, body FROM chunks WHERE doc_id = %s AND section = %s ORDER BY ord",
        (doc_id, section)).fetchall()
    title = conn.execute("SELECT title FROM documents WHERE doc_id = %s", (doc_id,)).fetchone()["title"]

    position = {row["ord"]: i for i, row in enumerate(rows)}
    hit_positions = {position[c.ord] for c, _ in group}
    anchor_cand, _ = min(group, key=lambda pair: (-pair[1], pair[0].ord))
    selected = _expand(rows, hit_positions, position[anchor_cand.ord], settings.parent_cap_tokens)

    parts: list[str] = []
    prev_ord = None
    for i in sorted(selected):
        ord_ = rows[i]["ord"]
        if prev_ord is not None and ord_ != prev_ord + 1:
            parts.append(GAP)
        parts.append(rows[i]["body"])
        prev_ord = ord_
    text = "\n\n".join(parts)
    score = max(s for _, s in group)
    return _Parent(Authority.ARCHIVE, doc_id, section, score, title, text, count_tokens(text))


def _expand(rows: list[dict], hit_positions: set[int], anchor: int, cap: int) -> set[int]:
    # Hits are always kept, so the running total may already exceed the cap.
    selected = set(hit_positions)
    total = sum(count_tokens(rows[i]["body"]) for i in selected)
    offsets = [p for k in range(1, len(rows)) for p in (anchor - k, anchor + k)]
    for p in offsets:
        if not 0 <= p < len(rows) or p in selected:
            continue
        n = count_tokens(rows[p]["body"])
        if total + n > cap:
            break
        selected.add(p)
        total += n
    return selected
