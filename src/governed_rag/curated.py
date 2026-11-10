from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

from governed_rag.config import Settings
from governed_rag.text import count_tokens
from governed_rag.types import CuratedDoc, Fact

_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_EXTERNAL_LINK = ("http://", "https://", "mailto:", "#")
_EXPIRY_WARN_DAYS = 30


class CuratedError(Exception):
    def __init__(self, rule: str, doc_id: str, message: str):
        super().__init__(f"{rule} {doc_id}: {message}")
        self.rule = rule
        self.doc_id = doc_id
        self.message = message


@dataclass(frozen=True)
class LintIssue:
    rule: str
    severity: str
    doc_id: str
    message: str


def split_front_matter(text: str) -> tuple[dict, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return {}, text
    end = next((i for i in range(1, len(lines)) if lines[i].rstrip("\r\n") == "---"), None)
    if end is None:
        raise ValueError("front matter is not closed by '---'")
    meta = yaml.safe_load("".join(lines[1:end]))
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise ValueError("front matter is not a mapping")
    return meta, "".join(lines[end + 1:]).lstrip()


def parse_curated(path: Path, root: Path) -> CuratedDoc:
    doc_id = "kb/" + path.relative_to(root).with_suffix("").as_posix()
    try:
        meta, body = split_front_matter(path.read_text(encoding="utf-8"))
    except (ValueError, yaml.YAMLError) as e:
        raise CuratedError("L1", doc_id, f"invalid front matter: {e}") from e

    review_by = meta.get("review_by")
    if not isinstance(review_by, date) or isinstance(review_by, datetime):
        raise CuratedError("L1", doc_id, "review_by must be a date (YYYY-MM-DD)")
    acl = meta.get("acl")
    if not _str_list(acl) or not acl:
        raise CuratedError("L1", doc_id, "acl must be a non-empty list of strings")
    supersedes = meta.get("supersedes") or []
    if not _str_list(supersedes):
        raise CuratedError("L1", doc_id, "supersedes must be a list of strings")
    facts = meta.get("facts") or {}
    if not isinstance(facts, dict):
        raise CuratedError("L1", doc_id, "facts must be a mapping")
    source = meta.get("source")
    sha = meta.get("source_sha256")
    if source is not None and sha is None:
        raise CuratedError("L1", doc_id, "source requires source_sha256")

    return CuratedDoc(
        doc_id=doc_id,
        type=_str_field(meta, "type", doc_id),
        title=_str_field(meta, "title", doc_id),
        owner=_str_field(meta, "owner", doc_id),
        review_by=review_by,
        acl=tuple(acl),
        body=body,
        facts=tuple(_parse_fact(key, spec, doc_id) for key, spec in facts.items()),
        supersedes=tuple(supersedes),
        source=None if source is None else _str_field(meta, "source", doc_id),
        source_sha256=None if sha is None else str(sha),
    )


def load_tree(root: Path) -> list[CuratedDoc]:
    return sorted((parse_curated(p, root) for p in _md_files(root)), key=lambda d: d.doc_id)


def lint_tree(root: Path, archive_root: Path | None, today: date) -> list[LintIssue]:
    issues: list[LintIssue] = []
    parsed: list[tuple[Path, CuratedDoc]] = []
    for path in _md_files(root):
        try:
            parsed.append((path, parse_curated(path, root)))
        except CuratedError as e:
            issues.append(LintIssue("L1", "error", e.doc_id, e.message))
    parsed.sort(key=lambda pair: pair[1].doc_id)

    defined_in: dict[str, list[str]] = {}
    for _, doc in parsed:
        for fact in doc.facts:
            defined_in.setdefault(fact.key, []).append(doc.doc_id)
    for key, ids in defined_in.items():
        if len(ids) > 1:
            issues.append(LintIssue("L5", "error", ids[-1],
                                    f"fact key {key} is also defined in {', '.join(ids[:-1])}"))

    for path, doc in parsed:
        issues.extend(_doc_issues(path, doc, root, archive_root, today))
    return sorted(issues, key=lambda i: (i.doc_id, i.rule, i.message))


def _doc_issues(path: Path, doc: CuratedDoc, root: Path, archive_root: Path | None,
                today: date) -> list[LintIssue]:
    issues: list[LintIssue] = []

    def add(rule: str, message: str, severity: str = "error") -> None:
        issues.append(LintIssue(rule, severity, doc.doc_id, message))

    days = (doc.review_by - today).days
    if days < 0:
        add("L2", f"expired {-days} days ago", "warn")
    elif days <= _EXPIRY_WARN_DAYS:
        add("L2", f"expires in {days} days", "warn")

    limit = Settings.curated_max_tokens
    if count_tokens(doc.body) > limit:
        add("L3", f"body exceeds {limit} tokens")

    for fact in doc.facts:
        if fact.value not in doc.body:
            add("L4", f"fact {fact.key} value {fact.value!r} is not in the body")

    for target in _LINK.findall(doc.body):
        if not target.startswith(_EXTERNAL_LINK) and not (path.parent / target.split("#", 1)[0]).exists():
            add("L6", f"broken link {target}")
    if archive_root is not None:
        for target in doc.supersedes:
            if not _in_archive(archive_root, target):
                add("L6", f"superseded document {target} not found in archive")

    if doc.source is not None and not doc.source.startswith(("http://", "https://")):
        src = root / doc.source
        if not src.is_file():
            add("L7", f"source {doc.source} not found")
        elif hashlib.sha256(src.read_bytes()).hexdigest() != doc.source_sha256:
            add("L7", f"source {doc.source} changed since source_sha256 was recorded")
    return issues


def _in_archive(archive_root: Path, target: str) -> bool:
    if not target.startswith("archive/"):
        return False
    base = archive_root / target.removeprefix("archive/")
    return any(Path(f"{base}{ext}").is_file() for ext in (".md", ".txt"))


def _parse_fact(key: object, spec: object, doc_id: str) -> Fact:
    value, unit = (spec.get("value"), spec.get("unit")) if isinstance(spec, dict) else (spec, None)
    if value is None:
        raise CuratedError("L1", doc_id, f"fact {key} has no value")
    return Fact(str(key), str(value), None if unit is None else str(unit))


def _str_field(meta: dict, key: str, doc_id: str) -> str:
    value = meta.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CuratedError("L1", doc_id, f"{key} must be a non-empty string")
    return value


def _str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _md_files(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("**/*.md") if p.is_file())
