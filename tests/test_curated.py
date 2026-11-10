import hashlib
from datetime import date
import pytest
from governed_rag.curated import lint_tree, parse_curated
from governed_rag.types import Fact

GOOD = """---
type: policy
title: Refund policy
owner: billing@example.com
review_by: 2027-05-01
acl: [public]
facts:
  refund_window_days: {value: 14, unit: days}
supersedes: [archive/policies/refund-2023]
---
# Refund policy

Customers can request a refund within 14 days of purchase.
"""
TODAY = date(2026, 11, 10)

def write(root, rel, text):
    p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text); return p

def rules(root, archive=None):
    return sorted((i.rule, i.severity) for i in lint_tree(root, archive, TODAY))

def test_parse_good_doc(tmp_path):
    doc = parse_curated(write(tmp_path, "policy/refunds.md", GOOD), tmp_path)
    assert doc.doc_id == "kb/policy/refunds" and doc.review_by == date(2027, 5, 1)
    assert doc.facts == (Fact("refund_window_days", "14", "days"),)
    assert doc.supersedes == ("archive/policies/refund-2023",)
    assert doc.acl == ("public",) and doc.body.startswith("# Refund policy")

@pytest.mark.parametrize("mutate, rule", [
    (lambda t: t.replace("owner: billing@example.com\n", ""), "L1"),
    (lambda t: t.replace("acl: [public]", "acl: []"), "L1"),
    (lambda t: t.replace("acl: [public]", "acl: [public]\nsource: s.pdf"), "L1"),
    (lambda t: t + "word " * 1600, "L3"),
    (lambda t: t.replace("within 14 days", "within two weeks"), "L4"),
    (lambda t: t.replace("purchase.", "purchase. See [SLA](sla.md)."), "L6"),
    (lambda t: t.replace("acl: [public]", "acl: [public]\nsource: missing.pdf\nsource_sha256: abc"), "L7"),
])
def test_each_rule_fires(tmp_path, mutate, rule):
    write(tmp_path, "policy/refunds.md", mutate(GOOD))
    assert (rule, "error") in rules(tmp_path)

def test_clean_tree_and_expiry_only_warns(tmp_path):
    write(tmp_path, "policy/refunds.md", GOOD)
    assert rules(tmp_path) == []
    write(tmp_path, "policy/refunds.md", GOOD.replace("2027-05-01", "2026-11-01"))
    assert rules(tmp_path) == [("L2", "warn")]
    write(tmp_path, "policy/refunds.md", GOOD.replace("2027-05-01", "2026-12-01"))
    assert rules(tmp_path) == [("L2", "warn")]

def test_duplicate_fact_key_across_docs(tmp_path):
    write(tmp_path, "a.md", GOOD); write(tmp_path, "b.md", GOOD)
    assert ("L5", "error") in rules(tmp_path)

def test_supersedes_target_must_exist_in_archive(tmp_path):      # Review Focus 3
    kb, ar = tmp_path / "kb", tmp_path / "archive"; ar.mkdir()
    write(kb, "policy/refunds.md", GOOD)
    assert rules(kb, ar) == [("L6", "error")]
    write(ar, "policies/refund-2023.md", "old")
    assert rules(kb, ar) == []

def test_source_hash_drift(tmp_path):
    src = write(tmp_path, "sources/refund.txt", "v1")
    sha = hashlib.sha256(b"v1").hexdigest()
    write(tmp_path, "policy/refunds.md",
          GOOD.replace("acl: [public]", f"acl: [public]\nsource: sources/refund.txt\nsource_sha256: {sha}"))
    assert rules(tmp_path) == []
    src.write_text("v2")
    assert rules(tmp_path) == [("L7", "error")]
