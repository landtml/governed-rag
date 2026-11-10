from __future__ import annotations

import hmac
import json
import logging
import queue
import re
import signal
import threading
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from psycopg import Connection

from governed_rag.answer import answer
from governed_rag.config import Settings
from governed_rag.db import apply_schema, connect
from governed_rag.evalkit import _Locked
from governed_rag.evidence import search
from governed_rag.retrieve import get_fact
from governed_rag.types import Authority, Evidence, Principal

log = logging.getLogger("governed_rag.service")

MIN_TOKEN_CHARS = 16
MAX_BODY_BYTES = 65_536
MAX_TEXT_CHARS = 2_000
POOL_WAIT_SECONDS = 30.0
CLIENT_TIMEOUT_SECONDS = 30
_AUTHORITY = {"archive": Authority.ARCHIVE, "curated": Authority.CURATED}
_ROUTES = {"/healthz": "GET", "/search": "POST", "/answer": "POST", "/fact": "POST"}
_CONTENT_LENGTH = re.compile(r"[0-9]+")


def load_tokens(path: Path) -> dict[str, frozenset[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("token file must be a JSON object of token -> list of groups")
    tokens: dict[str, frozenset[str]] = {}
    for token, groups in data.items():
        if len(token) < MIN_TOKEN_CHARS or token != token.strip():
            raise ValueError(f"tokens must be at least {MIN_TOKEN_CHARS} characters with no surrounding spaces")
        if not isinstance(groups, list) or not all(isinstance(g, str) and g for g in groups):
            raise ValueError("each token must map to a list of non-empty group names")
        tokens[token] = frozenset(groups)
    return tokens


def make_server(host: str, port: int, *, database_url: str, embedder, reranker, generator, settings: Settings,
                tokens: Mapping[str, frozenset[str]], anonymous_public: bool = False,
                today: date | None = None, pool_size: int = 4) -> ThreadingHTTPServer:
    with closing(connect(database_url)) as conn:
        apply_schema(conn)
    service = _Service(database_url=database_url, embedder=embedder, reranker=reranker, generator=generator,
                       settings=settings, tokens=tokens, anonymous_public=anonymous_public, today=today,
                       pool_size=pool_size)
    return _Server((host, port), service)


def serve(server: ThreadingHTTPServer) -> None:
    """Serve until Ctrl-C or SIGTERM (what a process manager sends), then close the socket and the pool."""
    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a repeated SIGTERM must not interrupt the clean-up
        raise KeyboardInterrupt

    in_main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, stop) if in_main else None
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if in_main:
            signal.signal(signal.SIGTERM, previous)


class PoolBusy(Exception):
    """No pooled connection became free within POOL_WAIT_SECONDS, or the pool is closed."""


class _Reject(Exception):
    def __init__(self, status: int, error: str, headers: Mapping[str, str] | None = None):
        super().__init__(error)
        self.status, self.error, self.headers = status, error, dict(headers or {})


class _Pool:
    """Up to `size` connections, each opened on first use. A queue slot holds a connection or None (not open)."""

    def __init__(self, database_url: str, size: int):
        if size < 1:
            raise ValueError("pool_size must be at least 1")
        self._url = database_url
        self._slots: queue.LifoQueue = queue.LifoQueue()
        for _ in range(size):
            self._slots.put(None)
        self._lock = threading.Lock()
        self._open: set[Connection] = set()
        self._closed = False

    @property
    def open_connections(self) -> tuple[Connection, ...]:
        with self._lock:
            return tuple(self._open)

    @contextmanager
    def borrow(self) -> Iterator[Connection]:
        try:
            conn = self._slots.get(timeout=POOL_WAIT_SECONDS)
        except queue.Empty:
            raise PoolBusy() from None
        try:
            if conn is None:
                conn = self._connect()
            yield conn
        except BaseException:
            self._discard(conn)
            raise
        self._release(conn)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            conns = tuple(self._open)
        for conn in conns:
            conn.close()

    def _connect(self) -> Connection:
        conn = connect(self._url)
        with self._lock:
            if not self._closed:
                self._open.add(conn)
                return conn
        conn.close()
        raise PoolBusy("pool is closed")

    def _discard(self, conn: Connection | None) -> None:
        if conn is not None:
            with self._lock:
                self._open.discard(conn)
            conn.close()
        self._slots.put(None)

    def _release(self, conn: Connection) -> None:
        with self._lock:
            if not self._closed:
                self._slots.put(conn)


class _Service:
    """Everything a request needs. The handler parses HTTP; this class decides who may see what."""

    def __init__(self, *, database_url: str, embedder, reranker, generator, settings: Settings,
                 tokens: Mapping[str, frozenset[str]], anonymous_public: bool, today: date | None,
                 pool_size: int):
        model_lock = threading.Lock()
        self.embedder = _Locked(embedder, model_lock)
        self.reranker = _Locked(reranker, model_lock)
        self.generator = generator
        self.settings = settings
        self.today = today
        self.anonymous_public = anonymous_public
        self._tokens = [(token.encode("utf-8"), groups) for token, groups in tokens.items()]
        self.pool = _Pool(database_url, pool_size)

    def principal(self, authorization: str | None) -> Principal:
        if authorization is None and self.anonymous_public:
            return Principal()
        presented = _bearer_token(authorization)
        if presented is None:  # no header, or a header that is not "Bearer <token>": never treated as anonymous
            raise _unauthorized()
        needle = presented.encode("utf-8")
        # Compare against every token: a match must not change how long the loop takes.
        matches = [groups for token, groups in self._tokens if hmac.compare_digest(token, needle)]
        if not matches:
            raise _unauthorized()
        return Principal(matches[0])

    def ping(self) -> None:
        with self.pool.borrow() as conn:
            conn.execute("SELECT 1")

    def search_json(self, principal: Principal, question: str, min_authority: Authority) -> dict:
        with self.pool.borrow() as conn:
            result = search(conn, question, principal, embedder=self.embedder, reranker=self.reranker,
                            settings=self.settings, today=self._today(), min_authority=min_authority)
        return {"evidence": [_evidence_json(e) for e in result.evidence],
                "top_score": result.top_score, "abstain_reason": result.abstain_reason}

    def answer_json(self, principal: Principal, question: str, min_authority: Authority) -> dict:
        with self.pool.borrow() as conn:
            out = answer(conn, question, principal, embedder=self.embedder, reranker=self.reranker,
                         generator=self.generator, settings=self.settings, today=self._today(),
                         min_authority=min_authority)
        doc_of = {e.eid: e.doc_id for e in out.evidence}
        return {"text": out.text, "authority": out.authority.name.lower(), "abstain_reason": out.abstain_reason,
                "citations": [{"evidence_id": c.eid, "doc_id": doc_of.get(c.eid), "quote": c.quote}
                              for c in out.citations]}

    def fact_json(self, principal: Principal, key: str) -> dict | None:
        with self.pool.borrow() as conn:
            fact = get_fact(conn, key, principal, self._today())
        return None if fact is None else {"key": fact.key, "value": fact.value, "unit": fact.unit}

    def _today(self) -> date:
        return date.today() if self.today is None else self.today


class _Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], service: _Service):
        self.service = service
        super().__init__(address, _Handler)

    def server_close(self) -> None:
        super().server_close()
        self.service.pool.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = CLIENT_TIMEOUT_SECONDS

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_PATCH(self) -> None:
        self._dispatch("PATCH")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def do_HEAD(self) -> None:
        self._dispatch("HEAD")

    def do_OPTIONS(self) -> None:
        self._dispatch("OPTIONS")

    def log_message(self, format: str, *args: object) -> None:
        # The request line only: method, path, status. Headers and bodies are never logged.
        log.info("%s", format % args)

    def _dispatch(self, method: str) -> None:
        path = self.path.split("?", 1)[0]
        allowed = _ROUTES.get(path)
        if allowed is None:
            return self._send(404, {"error": "not found"})
        if method != allowed:
            return self._send(405, {"error": "method not allowed"}, {"Allow": allowed})
        if path == "/healthz":
            return self._healthz()
        try:
            status, body = self._query(path)
        except _Reject as reject:
            return self._send(reject.status, {"error": reject.error}, reject.headers)
        except PoolBusy:
            return self._send(503, {"error": "busy"})
        except Exception:
            log.exception("%s %s failed", method, path)
            return self._send(500, {"error": "internal error"})
        self._send(status, body)

    def _healthz(self) -> None:
        try:
            self.server.service.ping()
        except Exception as err:
            log.warning("healthz: database check failed: %r", err)
            return self._send(503, {"status": "unavailable"})
        self._send(200, {"status": "ok"})

    def _query(self, path: str) -> tuple[int, object]:
        service = self.server.service
        principal = service.principal(self.headers.get("Authorization"))
        data = _json_object(self._read_body())
        if path == "/fact":
            return 200, service.fact_json(principal, _text_field(data, "key"))
        question = _text_field(data, "question")
        min_authority = _min_authority(data)
        if path == "/search":
            return 200, service.search_json(principal, question, min_authority)
        return 200, service.answer_json(principal, question, min_authority)

    def _read_body(self) -> bytes:
        raw = self.headers.get("Content-Length")
        if raw is None or not _CONTENT_LENGTH.fullmatch(raw.strip()):
            raise _Reject(411, "Content-Length header required")
        length = int(raw)
        if length > MAX_BODY_BYTES:
            raise _Reject(413, "body too large")
        return self.rfile.read(length)

    def _send(self, status: int, body: object, headers: Mapping[str, str] | None = None) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)


def _unauthorized() -> _Reject:
    return _Reject(401, "unauthorized", {"WWW-Authenticate": "Bearer"})


def _bearer_token(header: str | None) -> str | None:
    if header is None:
        return None
    scheme, _, token = header.strip().partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _json_object(raw: bytes) -> dict:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError):
        raise _Reject(400, "body must be UTF-8 JSON") from None
    if not isinstance(data, dict):
        raise _Reject(400, "body must be a JSON object")
    return data


def _text_field(data: dict, name: str) -> str:
    if name not in data:
        raise _Reject(400, f"missing {name}")
    value = data[name]
    if not isinstance(value, str):
        raise _Reject(400, f"{name} must be a string")
    if len(value) > MAX_TEXT_CHARS:
        raise _Reject(400, f"{name} must be at most {MAX_TEXT_CHARS} characters")
    text = value.strip()
    if not text:
        raise _Reject(400, f"{name} must not be empty")
    return text


def _min_authority(data: dict) -> Authority:
    value = data.get("min_authority", "archive")
    if not isinstance(value, str) or value not in _AUTHORITY:
        raise _Reject(400, "min_authority must be archive or curated")
    return _AUTHORITY[value]


def _evidence_json(e: Evidence) -> dict:
    return {"eid": e.eid, "doc_id": e.doc_id, "title": e.title, "authority": e.authority.name.lower(),
            "score": e.score, "text": e.text}
