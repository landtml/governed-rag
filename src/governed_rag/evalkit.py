from __future__ import annotations

import json
import math
import random
import statistics
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from psycopg import Connection

from governed_rag.answer import answer
from governed_rag.config import Settings
from governed_rag.types import Answer, Authority, Embedder, Evidence, Generator, Judge, Principal, Reranker, Verdict

_AUTHORITY = {"archive": Authority.ARCHIVE, "curated": Authority.CURATED}


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    groups: tuple[str, ...]
    min_authority: Authority
    tags: tuple[str, ...]
    answerable: bool
    gold_docs: tuple[str, ...]
    must_contain: tuple[str, ...]
    forbidden_docs: tuple[str, ...]
    split: str = "fit"


@dataclass(frozen=True)
class CaseResult:
    case: Case
    answer: Answer
    latency_ms: float


def load_golden(path: Path) -> list[Case]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.get("split", "fit")
        if split not in ("fit", "holdout"):
            raise ValueError(f"case {row['id']}: split must be 'fit' or 'holdout', not {split!r}")
        cases.append(Case(
            id=row["id"], question=row["question"], groups=tuple(row.get("groups", ())),
            min_authority=_AUTHORITY[row.get("min_authority", "archive")], tags=tuple(row.get("tags", ())),
            answerable=row["answerable"], gold_docs=tuple(row.get("gold_docs", ())),
            must_contain=tuple(row.get("must_contain", ())), forbidden_docs=tuple(row.get("forbidden_docs", ())),
            split=split))
    return cases


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(n_boot))
    return statistics.fmean(values), means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def calibrate_tau(pos: Sequence[float], neg: Sequence[float]) -> float:
    if not pos or not neg:
        raise ValueError("calibrate_tau needs both positive and negative scores")
    uniq = sorted(set(pos) | set(neg))
    if len(uniq) < 2:
        raise ValueError("calibrate_tau needs at least two distinct scores")
    best_tau, best_j = None, -math.inf
    for lo, hi in zip(uniq, uniq[1:]):
        tau = (lo + hi) / 2
        j = sum(p >= tau for p in pos) / len(pos) - sum(n >= tau for n in neg) / len(neg)
        if j >= best_j:  # >= so that a tie goes to the higher threshold
            best_tau, best_j = tau, j
    return best_tau


class _Locked:
    """Serialises calls into a local model, which is not safe to share between threads."""

    def __init__(self, inner, lock: threading.Lock):
        self._inner, self._lock = inner, lock

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            with self._lock:
                return attr(*args, **kwargs)
        return call


def run(conn: Connection, cases: Sequence[Case], *, embedder: Embedder, reranker: Reranker,
        generator: Generator, settings: Settings, today: date, workers: int = 1,
        connect: Callable[[], Connection] | None = None) -> list[CaseResult]:
    """Answer every case, in order. With workers > 1 the cases run on threads, each with its
    own connection from `connect`: generation overlaps, the local models take turns. Latency
    then includes time spent waiting for a model, so measure latency with workers=1."""

    def one(c: Connection, case: Case) -> CaseResult:
        started = time.perf_counter()
        out = answer(c, case.question, Principal(frozenset(case.groups)), embedder=embedder,
                     reranker=reranker, generator=generator, settings=settings, today=today,
                     min_authority=case.min_authority)
        return CaseResult(case, out, (time.perf_counter() - started) * 1000)

    if workers <= 1:
        return [one(conn, case) for case in cases]
    if connect is None:
        raise ValueError("workers > 1 needs a connect callable")

    model_lock = threading.Lock()
    embedder, reranker = _Locked(embedder, model_lock), _Locked(reranker, model_lock)
    local = threading.local()
    opened: list[Connection] = []

    def task(case: Case) -> CaseResult:
        if not hasattr(local, "conn"):
            local.conn = connect()
            opened.append(local.conn)
        return one(local.conn, case)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(task, cases))
    finally:
        for c in opened:
            c.close()


def score(results: Sequence[CaseResult]) -> dict:
    leaks = 0
    exact: list[float] = []
    recall: list[float] = []
    curated: list[float] = []
    abstain_recall: list[float] = []
    abstain_precision: list[float] = []
    gate_pass: list[float] = []
    near_miss_abstain: list[float] = []
    for r in results:
        c, a = r.case, r.answer
        docs = {e.doc_id for e in a.evidence}
        abstained = a.abstain_reason is not None
        leaks += bool(docs & set(c.forbidden_docs))
        if "exact" in c.tags:
            exact.append(float(not abstained and all(s in a.text for s in c.must_contain)))
        if c.answerable and c.gold_docs:
            recall.append(float(bool(docs & set(c.gold_docs))))
        if "curated" in c.tags:
            curated.append(float(bool(a.evidence) and a.evidence[0].doc_id in c.gold_docs))
        if not c.answerable:
            abstain_recall.append(float(abstained))
        if abstained:
            abstain_precision.append(float(not c.answerable))
        if _is_near_miss(c):
            near_miss_abstain.append(float(abstained))
        if a.gate_failures:
            gate_pass.append(float(a.gate_failures[0] == ()))

    populations = [("recall_at_k", recall), ("curated_first", curated), ("abstain_recall", abstain_recall),
                   ("abstain_precision", abstain_precision), ("near_miss_abstain", near_miss_abstain),
                   ("gate_pass_rate", gate_pass)]
    soft = {name: _interval(values) for name, values in populations if values}
    latencies = [r.latency_ms for r in results]
    tau = _suggest_tau(results)
    return {
        "n": len(results),
        "hard": {"forbidden_leaks": leaks, "exact_accuracy": statistics.fmean(exact) if exact else 1.0},
        "soft": soft,
        "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
        "suggested_tau": tau,
        "calibration": calibration_stats(results, tau),
        "abstain_reasons": dict(sorted(Counter(
            r.answer.abstain_reason for r in results if r.answer.abstain_reason).items())),
    }


def gate(report: dict, baseline: dict | None) -> list[str]:
    violations = []
    hard = report["hard"]
    if hard["forbidden_leaks"] > 0:
        violations.append(f"hard:forbidden_leaks={hard['forbidden_leaks']}")
    if hard["exact_accuracy"] < 1.0:
        violations.append(f"hard:exact_accuracy={hard['exact_accuracy']:.3f}")
    if baseline:
        current = report.get("soft", {})
        for name, base in baseline.get("soft", {}).items():
            if name in current and base["mean"] > current[name]["hi"]:
                violations.append(f"soft:{name} baseline={base['mean']:.3f} hi={current[name]['hi']:.3f}")
    return violations


def judge_answers(results: Sequence[CaseResult], judge: Judge, workers: int = 1) -> dict:
    answered = [r for r in results if r.answer.abstain_reason is None]
    cited = [_cited(r.answer) for r in answered]
    jobs: list[tuple[str, int, tuple[str, str, tuple[Evidence, ...]]]] = []
    for i, r in enumerate(answered):
        jobs.append(("faithfulness", i, (r.case.question, r.answer.text, cited[i])))
        partner = _control_partner(cited, i)
        if partner is not None:
            jobs.append(("control", i, (r.case.question, r.answer.text, cited[partner])))
    verdicts = _judge_all(judge, [args for _, _, args in jobs], workers)

    faithful: list[float] = []
    control: list[float] = []
    unsupported: list[dict] = []
    errors = 0
    for (kind, i, _), verdict in zip(jobs, verdicts):
        if verdict is None:
            errors += 1
        elif kind == "faithfulness":
            faithful.append(float(verdict.supported))
            if not verdict.supported:
                unsupported.append({"id": answered[i].case.id, "claims": list(verdict.unsupported)})
        else:
            control.append(float(not verdict.supported))
    return {"model": judge.model_id,
            "faithfulness": _interval(faithful) if faithful else None,
            "control_rejection": _interval(control) if control else None,
            "errors": errors, "unsupported": unsupported}


def _cited(answer: Answer) -> tuple[Evidence, ...]:
    ids = {c.eid for c in answer.citations}
    return tuple(e for e in answer.evidence if e.eid in ids)


def _control_partner(cited: list[tuple[Evidence, ...]], i: int) -> int | None:
    docs = {e.doc_id for e in cited[i]}
    for step in range(1, len(cited)):
        j = (i + step) % len(cited)
        if cited[j] and not docs & {e.doc_id for e in cited[j]}:
            return j
    return None


def _judge_all(judge: Judge, calls: list[tuple[str, str, tuple[Evidence, ...]]],
               workers: int) -> list[Verdict | None]:
    def ask(args: tuple[str, str, tuple[Evidence, ...]]) -> Verdict | None:
        try:
            return judge.judge(*args)
        except Exception:
            return None

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(ask, calls))
    return [ask(args) for args in calls]


def _interval(values: list[float]) -> dict:
    mean, lo, hi = bootstrap_ci(values)
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(values)}


def _percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[math.ceil(p * len(ordered) / 100) - 1])


def _is_near_miss(case: Case) -> bool:
    return "near_miss" in case.tags


def _suggest_tau(results: Sequence[CaseResult]) -> float | None:
    fit = [r for r in results if r.case.split == "fit" and not _is_near_miss(r.case)]
    pos = [r.answer.top_score for r in fit if r.case.answerable]
    neg = [r.answer.top_score for r in fit if not r.case.answerable]
    try:
        return float(calibrate_tau(pos, neg))
    except ValueError:
        return None


def calibration_stats(results: Sequence[CaseResult], tau: float | None) -> dict | None:
    if tau is None:
        return None
    near = [r.answer.top_score for r in results if _is_near_miss(r.case)]
    holdout = _split_stats(results, "holdout", tau)
    return {
        "tau": tau,
        "fit": _split_stats(results, "fit", tau),
        "holdout": holdout if holdout["answerable"] and holdout["unanswerable"] else None,
        "near_miss_above_tau": {"n": len(near), "share": _share(near, tau)} if near else None,
    }


def _split_stats(results: Sequence[CaseResult], split: str, tau: float) -> dict:
    cases = [r for r in results if r.case.split == split and not _is_near_miss(r.case)]
    pos = [r.answer.top_score for r in cases if r.case.answerable]
    neg = [r.answer.top_score for r in cases if not r.case.answerable]
    return {"answerable": len(pos), "unanswerable": len(neg), "kept": _share(pos, tau), "passed": _share(neg, tau)}


def _share(values: list[float], tau: float) -> float | None:
    return sum(v >= tau for v in values) / len(values) if values else None
