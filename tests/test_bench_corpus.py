from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from governed_rag.config import Settings
from governed_rag.curated import lint_tree, split_front_matter
from governed_rag.evalkit import load_golden
from governed_rag.fakes import FakeEmbedder
from governed_rag.ingest import sync

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 11, 10)
PREFIXES = {
    "exact-": 43, "superseded-": 12, "curated-": 30, "expired-": 5, "ticket-": 60, "handbook-": 10,
    "transcript-": 20, "strict-": 20, "denied-ticket-": 30, "denied-restricted-": 20, "unanswerable-": 50,
    "nearmiss-": 30,
}


def _load_script():
    spec = importlib.util.spec_from_file_location("make_corpus", ROOT / "bench" / "make_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prefix(case_id: str) -> str:
    return max((p for p in PREFIXES if case_id.startswith(p)), key=len)


def _doc_path(out: Path, doc_id: str) -> Path:
    if doc_id.startswith("kb/"):
        return out / "knowledge" / (doc_id[3:] + ".md")
    return out / (doc_id + ".md")


def _acl_and_body(path: Path) -> tuple[set[str], str]:
    meta, body = split_front_matter(path.read_text(encoding="utf-8"))
    return set(meta.get("acl", ["public"])), body


def _rows(golden: Path) -> list[dict]:
    return [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines()]


def _topic(module, case_id: str):
    return next(t for t in module.TOPICS if t.slug == case_id[len("nearmiss-"):])


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _fact_values(root: Path) -> dict[str, int]:
    values = {}
    for path in sorted((root / "knowledge").rglob("*.md")):
        meta, _ = split_front_matter(path.read_text(encoding="utf-8"))
        (fact,) = meta["facts"].values()
        values[path.name] = fact["value"]
    return values


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    base = tmp_path_factory.mktemp("bench")
    counts = _load_script().generate(base / "corpus", base / "golden.jsonl", seed=0)
    return base / "corpus", base / "golden.jsonl", counts


def test_counts(corpus):
    out, golden, counts = corpus
    assert len(list((out / "knowledge").rglob("*.md"))) == 60
    assert len(list((out / "archive" / "legacy").glob("*.md"))) == 72
    assert len(list((out / "archive" / "tickets").glob("*.md"))) == 1800
    assert len(list((out / "archive" / "handbook").glob("*.md"))) == 24
    assert len(list((out / "archive" / "transcripts").glob("*.md"))) == 120
    assert len(list((out / "archive" / "restricted").glob("*.md"))) == 48
    cases = load_golden(golden)
    assert len(cases) == 330
    assert Counter(_prefix(c.id) for c in cases) == Counter(PREFIXES)
    assert len({c.id for c in cases}) == 330
    assert counts["golden"] == 330


def test_nearmiss_cases(corpus):
    out, golden, _ = corpus
    module = _load_script()
    rows = [r for r in _rows(golden) if r["id"].startswith("nearmiss-")]
    assert len(rows) == 30
    assert len({r["id"] for r in rows}) == 30
    for row in rows:
        t = _topic(module, row["id"])
        assert t.slug not in module.SUPERSEDING | module.EXPIRED, row["id"]
        assert row["answerable"] is False, row["id"]
        assert row["tags"] == ["near_miss"], row["id"]
        assert not {"gold_docs", "must_contain", "forbidden_docs"} & set(row), row["id"]
        assert row["question"] in {q.format(ask=t.ask) for q in module.NEAR_MISS_Q}, row["id"]
        groups = set(row.get("groups", []))
        assert groups == (set() if t.group == "public" else {t.group}), row["id"]
        acl, body = _acl_and_body(_doc_path(out, f"kb/{t.area}/{t.slug}"))
        assert acl & (groups | {"public"}), row["id"]
        assert not re.search(r"\d{4}", body), row["id"]
        for sentence in re.split(r"[.!?]", body):
            assert not re.search(r"\b(introduced|changed)\b.*\d", sentence, re.IGNORECASE), row["id"]


def test_no_curated_body_has_a_four_digit_number(corpus):
    out, _, _ = corpus
    for path in sorted((out / "knowledge").rglob("*.md")):
        _, body = split_front_matter(path.read_text(encoding="utf-8"))
        assert not re.search(r"\d{4}", body), path.name


def test_splits_are_stratified_and_cover_both_sides(corpus):
    _, golden, _ = corpus
    rows = _rows(golden)
    assert all(list(r)[-1] == "split" for r in rows)
    assert all(r["split"] in ("fit", "holdout") for r in rows)
    for cls in PREFIXES:
        ordered = [r["split"] for r in rows if _prefix(r["id"]) == cls]
        assert ordered == ["fit" if i % 2 == 0 else "holdout" for i in range(len(ordered))], cls
    for side in ("fit", "holdout"):
        chosen = [r for r in rows if r["split"] == side]
        assert any(r["answerable"] for r in chosen), side
        assert any(not r["answerable"] for r in chosen), side


def test_same_seed_is_byte_identical_and_other_seed_changes_facts(corpus, tmp_path):
    out, golden, _ = corpus
    module = _load_script()
    again, again_golden = tmp_path / "again", tmp_path / "again.jsonl"
    module.generate(again, again_golden, seed=0)
    assert _snapshot(again) == _snapshot(out)
    assert again_golden.read_bytes() == golden.read_bytes()

    other = tmp_path / "other"
    module.generate(other, tmp_path / "other.jsonl", seed=1)
    assert _fact_values(other) != _fact_values(out)


def test_lint_has_no_errors_and_exactly_five_expiry_warnings(corpus):
    out, _, _ = corpus
    issues = lint_tree(out / "knowledge", out / "archive", TODAY)
    assert [i for i in issues if i.severity == "error"] == []
    assert [(i.rule, i.severity) for i in issues] == [("L2", "warn")] * 5


def test_golden_integrity(corpus):
    out, golden, _ = corpus
    for case in load_golden(golden):
        for doc in case.gold_docs + case.forbidden_docs:
            assert _doc_path(out, doc).is_file(), (case.id, doc)
        readable = set(case.groups) | {"public"}
        if case.answerable:
            for doc in case.gold_docs:
                assert _acl_and_body(_doc_path(out, doc))[0] & readable, (case.id, doc)
        if case.id.startswith("denied-"):
            for doc in case.forbidden_docs:
                assert not (_acl_and_body(_doc_path(out, doc))[0] & readable), (case.id, doc)
        if case.id.startswith(("exact-", "superseded-")):
            (gold,) = case.gold_docs
            body = _acl_and_body(_doc_path(out, gold))[1]
            for value in case.must_contain:
                assert value in body, (case.id, value)
                assert value not in case.question, (case.id, value)
        if case.id.startswith("superseded-"):
            legacy_body = _acl_and_body(_doc_path(out, case.forbidden_docs[0]))[1]
            assert case.must_contain[0] not in legacy_body, case.id


def test_fact_hygiene(corpus):
    out, _, _ = corpus
    values = []
    for path in sorted((out / "knowledge").rglob("*.md")):
        meta, body = split_front_matter(path.read_text(encoding="utf-8"))
        (fact,) = meta["facts"].values()
        values.append(fact["value"])
        assert len(re.findall(r"\d+", body)) == 1, path.name
    assert len(set(values)) == 60
    assert all(11 <= v <= 997 for v in values)


def test_sync_ingests_every_document_and_supersedes_twelve(db, corpus):
    out, _, _ = corpus
    report = sync(db, out / "knowledge", out / "archive", FakeEmbedder(), Settings(), TODAY)
    assert report.added == 60 + 2064
    superseded = db.execute("SELECT count(*) AS n FROM documents WHERE superseded_by IS NOT NULL").fetchone()
    assert superseded["n"] == 12


def test_committed_golden_matches_seed_zero(corpus):
    _, golden, _ = corpus
    assert (ROOT / "bench" / "golden.jsonl").read_bytes() == golden.read_bytes()
