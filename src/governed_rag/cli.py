from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

from governed_rag.answer import answer
from governed_rag.config import Settings, calibrated
from governed_rag.curated import lint_tree
from governed_rag.db import apply_schema, connect
from governed_rag.evalkit import gate, judge_answers, load_golden, run, score
from governed_rag.evidence import search
from governed_rag.fakes import FAKE_TAU, ExtractiveGenerator, FakeEmbedder, FakeReranker
from governed_rag.ingest import DeleteGuard, SyncAborted, sync
from governed_rag.retrieve import get_fact
from governed_rag.types import Authority, Evidence, Principal

_AUTHORITY = {"archive": Authority.ARCHIVE, "curated": Authority.CURATED}


def resolve_settings(database_url: str | None, fake: bool, calibration: Path, reranker_id: str) -> Settings:
    settings = Settings() if database_url is None else Settings(database_url=database_url)
    if fake:
        return dataclasses.replace(settings, tau=FAKE_TAU)
    if not calibration.exists():
        print(f"warning: no calibration file at {calibration}; tau is the provisional default",
              file=sys.stderr)
    return calibrated(settings, calibration, reranker_id)


# Each builder imports its real class only when it is needed, so --fake runs and
# commands that use fewer models never load the heavy dependencies.
def _embedder(fake: bool):
    if fake:
        return FakeEmbedder()
    from governed_rag.models import STEmbedder
    return STEmbedder()


def _reranker(fake: bool):
    if fake:
        return FakeReranker()
    from governed_rag.models import STReranker
    return STReranker()


def _generator(fake: bool, kind: str):
    if fake:
        return ExtractiveGenerator()
    from governed_rag.models import make_generator
    return make_generator(kind)


def _judge(kind: str):
    from governed_rag.models import make_judge
    return make_judge(kind)


def _principal(groups: str) -> Principal:
    return Principal(frozenset(g for g in groups.split(",") if g))


def _evidence_json(e: Evidence) -> dict:
    return {"eid": e.eid, "doc_id": e.doc_id, "title": e.title, "authority": e.authority.name.lower(),
            "score": e.score, "text": e.text}


def _print_issues(issues) -> None:
    for i in issues:
        print(f"{i.severity} {i.rule} {i.doc_id}: {i.message}")


def _lint(conn, args) -> int:
    issues = lint_tree(args.knowledge, args.archive, args.today)
    _print_issues(issues)
    return 1 if any(i.severity == "error" for i in issues) else 0


def _ingest(conn, args) -> int:
    embedder = _embedder(args.fake)
    settings = Settings(database_url=args.database_url)
    try:
        report = sync(conn, args.knowledge, args.archive, embedder, settings, args.today, force=args.force)
    except SyncAborted as e:
        _print_issues(e.issues)
        return 1
    except DeleteGuard as e:
        print(f"refusing to delete {e.removed} of {e.existing} {e.tier} documents; "
              "re-run with --force if this is intended")
        return 1
    print(json.dumps(dataclasses.asdict(report)))
    return 0


def _search(conn, args) -> int:
    embedder = _embedder(args.fake)
    reranker = _reranker(args.fake)
    settings = resolve_settings(args.database_url, args.fake, args.calibration, reranker.model_id)
    result = search(conn, args.question, _principal(args.groups), embedder=embedder, reranker=reranker,
                    settings=settings, today=args.today, min_authority=_AUTHORITY[args.min_authority])
    print(json.dumps({"evidence": [_evidence_json(e) for e in result.evidence],
                      "top_score": result.top_score, "abstain_reason": result.abstain_reason}))
    return 0


def _ask(conn, args) -> int:
    embedder = _embedder(args.fake)
    reranker = _reranker(args.fake)
    generator = _generator(args.fake, args.generator)
    settings = resolve_settings(args.database_url, args.fake, args.calibration, reranker.model_id)
    out = answer(conn, args.question, _principal(args.groups), embedder=embedder, reranker=reranker,
                 generator=generator, settings=settings, today=args.today,
                 min_authority=_AUTHORITY[args.min_authority])
    doc_of = {e.eid: e.doc_id for e in out.evidence}
    print(json.dumps({
        "text": out.text,
        "authority": out.authority.name.lower(),
        "abstain_reason": out.abstain_reason,
        "citations": [{"evidence_id": c.eid, "doc_id": doc_of.get(c.eid), "quote": c.quote}
                      for c in out.citations],
    }))
    return 0


def _fact(conn, args) -> int:
    fact = get_fact(conn, args.key, _principal(args.groups), args.today)
    print(json.dumps(None if fact is None else {"key": fact.key, "value": fact.value, "unit": fact.unit}))
    return 0


def _eval(conn, args) -> int:
    cases = load_golden(args.golden)
    embedder = _embedder(args.fake)
    reranker = _reranker(args.fake)
    generator = _generator(args.fake, args.generator)
    settings = resolve_settings(args.database_url, args.fake, args.calibration, reranker.model_id)
    results = run(conn, cases, embedder=embedder, reranker=reranker, generator=generator,
                  settings=settings, today=args.today, workers=args.workers,
                  connect=lambda: connect(args.database_url))
    report = {**score(results), "workers": args.workers}
    if args.judge != "none":
        report["judge"] = judge_answers(results, _judge(args.judge), workers=args.workers)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    violations = gate(report, baseline)

    if args.json:
        print(json.dumps({**report, "violations": violations}))
    else:
        _print_summary(report, violations)

    if args.write_baseline:
        _write_json(args.write_baseline, report)
    if args.write_calibration:
        tau = report["suggested_tau"]
        if tau is None:
            print("error: no tau to write; the golden set needs both answerable and unanswerable cases",
                  file=sys.stderr)
            return 1
        cal = report["calibration"]
        _write_json(args.write_calibration, {
            "tau": tau, "reranker": reranker.model_id,
            "golden_sha256": hashlib.sha256(args.golden.read_bytes()).hexdigest(),
            "cases": report["n"], "date": args.today.isoformat(),
            "fit_cases": {"answerable": cal["fit"]["answerable"], "unanswerable": cal["fit"]["unanswerable"]},
            "holdout": cal["holdout"], "near_miss_above_tau": cal["near_miss_above_tau"]})
    return 1 if violations else 0


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _print_summary(report: dict, violations: list[str]) -> None:
    hard = report["hard"]
    print(f"cases: {report['n']}")
    print(f"hard: forbidden_leaks={hard['forbidden_leaks']} exact_accuracy={hard['exact_accuracy']:.3f}")
    for name, s in report["soft"].items():
        print(f"soft: {name} mean={s['mean']:.3f} lo={s['lo']:.3f} hi={s['hi']:.3f} n={s['n']}")
    lat = report["latency_ms"]
    print(f"latency_ms: p50={lat['p50']} p95={lat['p95']}")
    print(f"suggested_tau: {report['suggested_tau']}")
    line = _calibration_line(report["calibration"]) if report.get("calibration") else None
    if line:
        print(line)
    print(f"abstain_reasons: {report['abstain_reasons']}")
    if "judge" in report:
        print(_judge_line(report["judge"]))
    for v in violations:
        print(f"violation: {v}")


def _judge_line(j: dict) -> str:
    def mean(interval: dict | None) -> str:
        return "None" if interval is None else f"{interval['mean']:.3f}"
    return (f"judge: model={j['model']} faithfulness={mean(j['faithfulness'])} "
            f"control_rejection={mean(j['control_rejection'])} errors={j['errors']} "
            f"unsupported={len(j['unsupported'])}")


def _calibration_line(cal: dict) -> str | None:
    parts = []
    for name in ("fit", "holdout"):
        group = cal[name]
        shares = [f"{k}={group[k]:.3f}" for k in ("kept", "passed") if group and group[k] is not None]
        if shares:
            parts.append(" ".join([name, *shares]))
    near = cal["near_miss_above_tau"]
    if near and near["share"] is not None:
        parts.append(f"near_miss_above_tau={near['share']:.3f}")
    return "calibration: " + " | ".join(parts) if parts else None


def _serve(conn, args) -> int:
    from governed_rag.service import load_tokens, make_server, serve
    if args.tokens is None and not args.anonymous_public:
        print("error: serve needs --tokens FILE, or --anonymous-public to serve public documents without one",
              file=sys.stderr)
        return 2
    try:
        tokens = load_tokens(args.tokens) if args.tokens else {}
    except (OSError, ValueError) as e:
        print(f"error: cannot use token file {args.tokens}: {e}", file=sys.stderr)
        return 2
    embedder = _embedder(args.fake)
    reranker = _reranker(args.fake)
    generator = _generator(args.fake, args.generator)
    settings = resolve_settings(args.database_url, args.fake, args.calibration, reranker.model_id)
    # Without --today the date is read per request: a server outlives the day it was started on.
    server = make_server(args.host, args.port, database_url=args.database_url, embedder=embedder,
                         reranker=reranker, generator=generator, settings=settings, tokens=tokens,
                         anonymous_public=args.anonymous_public, today=args.fixed_today,
                         pool_size=args.pool_size)
    host, port = server.server_address[:2]
    print(f"serving on http://{host}:{port} ({len(tokens)} tokens, "
          f"anonymous access {'on' if args.anonymous_public else 'off'})", file=sys.stderr)
    serve(server)
    return 0


def _backlog(conn, args) -> int:
    rows = conn.execute(
        "SELECT lower(trim(question)) AS question, count(*) AS n FROM query_log "
        "WHERE authority < 2 GROUP BY lower(trim(question)) "
        "ORDER BY n DESC, lower(trim(question)) LIMIT %(limit)s",
        {"limit": args.limit}).fetchall()
    print(json.dumps([{"question": r["question"], "count": r["n"]} for r in rows]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Governed RAG Reference Architecture: authority-tiered RAG with access control.")
    parser.add_argument("--fake", action="store_true",
                        help="use the deterministic fake embedder, reranker and generator")
    parser.add_argument("--database-url", default=None, help="default: DATABASE_URL or the local dev database")
    parser.add_argument("--today", type=date.fromisoformat, default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--calibration", type=Path, default=Path("eval/calibration.json"))
    parser.add_argument("--generator", choices=("anthropic", "claude-cli", "extractive"), default="anthropic",
                        help="generation backend for ask and eval; ignored with --fake")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("lint", help="check curated documents")
    p.add_argument("--knowledge", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    p.set_defaults(run=_lint)

    p = sub.add_parser("ingest", help="sync documents into the database")
    p.add_argument("--knowledge", type=Path)
    p.add_argument("--archive", type=Path)
    p.add_argument("--force", action="store_true", help="allow mass deletions")
    p.set_defaults(run=_ingest)

    for name, run_fn in (("search", _search), ("ask", _ask)):
        p = sub.add_parser(name)
        p.add_argument("question")
        p.add_argument("--groups", default="", help="comma-separated groups")
        p.add_argument("--min-authority", choices=sorted(_AUTHORITY), default="archive")
        p.set_defaults(run=run_fn)

    p = sub.add_parser("fact", help="look up a curated fact")
    p.add_argument("key")
    p.add_argument("--groups", default="", help="comma-separated groups")
    p.set_defaults(run=_fact)

    p = sub.add_parser("eval", help="run the golden set and gate the result")
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--write-baseline", type=Path)
    p.add_argument("--write-calibration", type=Path)
    p.add_argument("--json", action="store_true")
    p.add_argument("--workers", type=int, default=1,
                   help="cases answered at once; above 1, latency includes waiting for a local model")
    p.add_argument("--judge", choices=("none", "claude-cli", "fake"), default="none",
                   help="report entailment of answers by evidence; reported, never gated")
    p.set_defaults(run=_eval)

    p = sub.add_parser("serve", help="serve search, answer and fact over HTTP")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--tokens", type=Path, help='JSON file {"<bearer token>": ["group", ...]}; groups come only from here')
    p.add_argument("--anonymous-public", action="store_true",
                   help="requests without a token see public documents instead of getting 401")
    p.add_argument("--pool-size", type=int, default=4, help="database connections, and so requests in flight")
    p.set_defaults(run=_serve)

    p = sub.add_parser("backlog", help="list questions that were not answered from curated documents")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(run=_backlog)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.database_url = args.database_url or Settings().database_url
    args.fixed_today = args.today
    args.today = args.today or date.today()
    conn = connect(args.database_url)
    try:
        apply_schema(conn)
        return args.run(conn, args)
    finally:
        conn.close()
