from __future__ import annotations

import re
from collections.abc import Sequence

from governed_rag.text import normalize
from governed_rag.types import Authority, Draft, Evidence

_MARKER = re.compile(r"\[(E\d+)\]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    text = _MARKER.sub(" ", text)
    return {m.replace(",", "") for m in _NUMBER.findall(text)}


def _quote_found(quote: str, text: str) -> bool:
    prefix = suffix = ""
    if re.match(r"\w", quote[0]):
        prefix += r"(?<!\w)"
    if re.match(r"\d", quote[0]):
        prefix += r"(?<!\d[.,])"
    if re.match(r"\w", quote[-1]):
        suffix += r"(?!\w)"
    if re.match(r"\d", quote[-1]):
        suffix += r"(?![.,]\d)"
    return re.search(prefix + re.escape(quote) + suffix, text) is not None


def check(draft: Draft, evidence: Sequence[Evidence], min_authority: Authority) -> list[str]:
    if draft.abstain:
        return []
    by_id = {e.eid: e for e in evidence}
    cited = {c.eid for c in draft.citations}
    failures: list[str] = []

    if not draft.answer.strip():
        failures.append("G1:empty_answer")
    if not draft.citations:
        failures.append("G1:no_citation")
    failures += [f"G1:unknown_evidence:{c.eid}" for c in draft.citations if c.eid not in by_id]
    failures += [f"G1:marker_without_citation:{eid}"
                 for eid in dict.fromkeys(_MARKER.findall(draft.answer)) if eid not in cited]

    for c in draft.citations:
        if c.eid not in by_id:
            continue
        quote = normalize(c.quote)
        if not quote or not _quote_found(quote, normalize(by_id[c.eid].text)):
            failures.append(f"G2:quote_not_found:{c.eid}")

    sourced: set[str] = set()
    for c in draft.citations:
        sourced |= numbers(c.quote)
    failures += [f"G3:number_unsourced:{n}" for n in sorted(numbers(draft.answer) - sourced)]

    if authority_of(draft, evidence) < min_authority:
        failures.append("G4:authority_below_minimum")

    return list(dict.fromkeys(failures))


def authority_of(draft: Draft, evidence: Sequence[Evidence]) -> Authority:
    if draft.abstain:
        return Authority.NONE
    by_id = {e.eid: e.authority for e in evidence}
    known = [by_id[c.eid] for c in draft.citations if c.eid in by_id]
    return min(known, default=Authority.NONE)
