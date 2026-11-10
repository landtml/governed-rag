from __future__ import annotations

import concurrent.futures
import http.client
import json
import logging
import threading
import urllib.error
import urllib.request
from datetime import date

import psycopg
import pytest
from conftest import TEST_DATABASE_URL
from helpers import fake_settings, seed_doc

from governed_rag import service
from governed_rag.fakes import ExtractiveGenerator, FakeEmbedder, FakeReranker
from governed_rag.service import load_tokens, make_server, serve

TODAY = date(2026, 11, 10)
PUBLIC_TOKEN = "public-token-0123456789"
FINANCE_TOKEN = "finance-token-0123456789"
WRONG_TOKEN = "stranger-token-9876543210"
TOKENS = {PUBLIC_TOKEN: frozenset(), FINANCE_TOKEN: frozenset({"finance"})}
MAX_BODY = 65_536


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _parse(raw: bytes):
    return json.loads(raw) if raw else None


def call(base: str, path: str, body=None, *, headers=None, raw: bytes | None = None, method: str | None = None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.headers, _parse(resp.read())
    except urllib.error.HTTPError as err:
        with err:
            return err.code, err.headers, _parse(err.read())


def raw_post(address, headers: dict[str, str], body: bytes = b"", path: str = "/search"):
    """Send exactly the headers given; urllib would add its own Content-Length."""
    conn = http.client.HTTPConnection(address[0], address[1], timeout=10)
    try:
        conn.putrequest("POST", path)
        for name, value in headers.items():
            conn.putheader(name, value)
        conn.endheaders()
        if body:
            conn.send(body)
        resp = conn.getresponse()
        return resp.status, resp.headers, _parse(resp.read())
    finally:
        conn.close()


def _options(db, **overrides) -> dict:
    options = dict(database_url=TEST_DATABASE_URL, embedder=FakeEmbedder(), reranker=FakeReranker(),
                   generator=ExtractiveGenerator(), settings=fake_settings(), tokens=TOKENS, today=TODAY)
    options.update(overrides)
    return options


@pytest.fixture
def start(db):
    running = []

    def start(**overrides):
        server = make_server("127.0.0.1", 0, **_options(db, **overrides))
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        running.append((server, thread))
        host, port = server.server_address[:2]
        return server, f"http://{host}:{port}"

    yield start
    for server, thread in running:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


@pytest.fixture
def corpus(db):
    seed_doc(db, "kb/lobby", chunks=["The lobby opens at nine every morning."], title="Lobby hours")
    seed_doc(db, "kb/secret", tier="curated", acl=("finance",), title="Merger budget",
             chunks=["The merger budget is 4 million."], facts={"merger_budget_usd": "4000000"})
    return db


def doc_ids(body) -> list[str]:
    return [e["doc_id"] for e in body["evidence"]]


# load_tokens

GOOD_TOKENS = {"public-token-0123456789": [], "finance-token-0123456789": ["finance", "legal"]}


def test_load_tokens_accepts_a_good_file(tmp_path):
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps(GOOD_TOKENS))
    assert load_tokens(path) == {"public-token-0123456789": frozenset(),
                                 "finance-token-0123456789": frozenset({"finance", "legal"})}


def test_load_tokens_accepts_a_sixteen_character_token(tmp_path):
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({"0123456789abcdef": ["finance"]}))
    assert load_tokens(path) == {"0123456789abcdef": frozenset({"finance"})}


@pytest.mark.parametrize("text", [
    '["public-token-0123456789"]',
    '"public-token-0123456789"',
    '{"": ["public"]}',
    '{"short-token-123": ["public"]}',
    '{"public-token-0123456789": "finance"}',
    '{"public-token-0123456789": ["finance", 3]}',
    '{"public-token-0123456789": [""]}',
    '{"public-token-0123456789": null}',
    '{"  public-token-0123456789": []}',
    '{not json',
])
def test_load_tokens_rejects_bad_shapes(tmp_path, text):
    path = tmp_path / "tokens.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_tokens(path)


# server skeleton, healthz, routing

def test_healthz_is_ok_without_authentication(db, start):
    _, base = start()
    status, _, body = call(base, "/healthz")
    assert (status, body) == (200, {"status": "ok"})


def test_healthz_is_503_when_the_database_call_raises(db, start, monkeypatch):
    _, base = start()

    def down(url):
        raise psycopg.OperationalError("database is down")

    with monkeypatch.context() as m:
        m.setattr(service, "connect", down)
        status, _, body = call(base, "/healthz")
        assert (status, body) == (503, {"status": "unavailable"})
    status, _, body = call(base, "/healthz")
    assert (status, body) == (200, {"status": "ok"})


def test_unknown_path_is_404(db, start):
    _, base = start()
    assert call(base, "/nope")[0] == 404
    assert call(base, "/nope", {"question": "x"}, headers=bearer(PUBLIC_TOKEN))[0] == 404


def test_wrong_method_is_405_with_allow(db, start):
    _, base = start()
    status, headers, _ = call(base, "/search", method="GET")
    assert (status, headers["Allow"]) == (405, "POST")
    status, headers, _ = call(base, "/healthz", {}, headers=bearer(PUBLIC_TOKEN))
    assert (status, headers["Allow"]) == (405, "GET")
    status, headers, _ = call(base, "/fact", method="PUT")
    assert (status, headers["Allow"]) == (405, "POST")
    for method in ("HEAD", "OPTIONS"):
        status, headers, _ = call(base, "/search", method=method)
        assert (status, headers["Allow"]) == (405, "POST"), method


def test_responses_are_json_utf8(db, start):
    _, base = start()
    _, headers, _ = call(base, "/healthz")
    assert headers["Content-Type"] == "application/json; charset=utf-8"


# authentication

@pytest.mark.parametrize("headers, anonymous, status", [
    ({}, False, 401),
    (bearer(WRONG_TOKEN), False, 401),
    (bearer(WRONG_TOKEN), True, 401),
    ({"Authorization": "Basic dXNlcjpwYXNz"}, False, 401),
    ({"Authorization": "Basic dXNlcjpwYXNz"}, True, 401),
    ({"Authorization": "Bearer"}, False, 401),
    ({"Authorization": "Bearer"}, True, 401),
    ({}, True, 200),
    (bearer(PUBLIC_TOKEN), False, 200),
    (bearer(PUBLIC_TOKEN), True, 200),
])
def test_authentication_outcomes(corpus, start, headers, anonymous, status):
    _, base = start(anonymous_public=anonymous)
    got, resp_headers, body = call(base, "/search", {"question": "lobby opens"}, headers=headers)
    assert got == status
    if status == 401:
        assert resp_headers["WWW-Authenticate"] == "Bearer"
        assert body == {"error": "unauthorized"}


def test_missing_token_is_401_on_every_post_route(corpus, start):
    _, base = start()
    for path, body in (("/search", {"question": "merger budget"}), ("/answer", {"question": "merger budget"}),
                       ("/fact", {"key": "merger_budget_usd"})):
        status, _, _ = call(base, path, body)
        assert status == 401, path


def test_anonymous_public_sees_only_public_documents(corpus, start):
    _, base = start(anonymous_public=True)
    _, _, body = call(base, "/search", {"question": "lobby opens"})
    assert doc_ids(body) == ["kb/lobby"]
    _, _, body = call(base, "/search", {"question": "merger budget"})
    assert "kb/secret" not in doc_ids(body)


def test_a_valid_token_keeps_its_groups_with_anonymous_public_on(corpus, start):
    _, base = start(anonymous_public=True)
    _, _, body = call(base, "/search", {"question": "merger budget"}, headers=bearer(FINANCE_TOKEN))
    assert "kb/secret" in doc_ids(body)


def test_every_configured_token_is_compared_in_constant_time_loop(corpus, start, monkeypatch):
    calls = []
    real = service.hmac.compare_digest

    def counting(a, b):
        calls.append(1)
        return real(a, b)

    monkeypatch.setattr(service.hmac, "compare_digest", counting)
    _, base = start()
    assert call(base, "/search", {"question": "lobby opens"}, headers=bearer(PUBLIC_TOKEN))[0] == 200
    assert len(calls) == len(TOKENS)


def test_access_log_has_method_path_and_status_but_no_secrets(corpus, start, caplog):
    caplog.set_level(logging.INFO, logger="governed_rag.service")
    _, base = start()
    call(base, "/search", {"question": "merger budget question"}, headers=bearer(FINANCE_TOKEN))
    text = "\n".join(r.getMessage() for r in caplog.records if r.name == "governed_rag.service")
    assert "POST /search" in text and " 200" in text
    assert FINANCE_TOKEN not in text and "merger budget question" not in text and "Authorization" not in text


@pytest.mark.parametrize("token, sees_secret", [(PUBLIC_TOKEN, False), (FINANCE_TOKEN, True)])
def test_groups_come_from_the_token_only(corpus, start, token, sees_secret):
    _, base = start()
    body = {"question": "merger budget", "groups": ["finance"]}
    _, _, found = call(base, "/search", body, headers=bearer(token))
    assert ("kb/secret" in doc_ids(found)) is sees_secret


# /search

def test_search_with_a_group_sees_the_restricted_document(corpus, start):
    _, base = start()
    status, _, body = call(base, "/search", {"question": "merger budget"}, headers=bearer(FINANCE_TOKEN))
    assert status == 200
    assert set(body) == {"evidence", "top_score", "abstain_reason"}
    assert body["abstain_reason"] is None and body["top_score"] > 0
    evidence = next(e for e in body["evidence"] if e["doc_id"] == "kb/secret")
    assert set(evidence) == {"eid", "doc_id", "title", "authority", "score", "text"}
    assert evidence["authority"] == "curated" and evidence["title"] == "Merger budget"


def test_search_without_the_group_does_not_see_it(corpus, start):
    _, base = start()
    _, _, body = call(base, "/search", {"question": "merger budget"}, headers=bearer(PUBLIC_TOKEN))
    assert "kb/secret" not in doc_ids(body)
    assert body["abstain_reason"] == "no_evidence" and body["evidence"] == []


def test_min_authority_curated_excludes_archive_evidence(corpus, start):
    seed_doc(corpus, "arc/thread", chunks=["Refund window was 30 days for one customer."])
    _, base = start()
    question = {"question": "refund window days"}
    _, _, archive_and_curated = call(base, "/search", question, headers=bearer(PUBLIC_TOKEN))
    assert {e["authority"] for e in archive_and_curated["evidence"]} == {"archive"}
    _, _, curated = call(base, "/search", {**question, "min_authority": "curated"}, headers=bearer(PUBLIC_TOKEN))
    assert curated["evidence"] == [] and curated["abstain_reason"] == "no_evidence"
    seed_doc(corpus, "kb/refunds", tier="curated", chunks=["Refund window is 14 days."])
    _, _, curated = call(base, "/search", {**question, "min_authority": "curated"}, headers=bearer(PUBLIC_TOKEN))
    assert {e["authority"] for e in curated["evidence"]} == {"curated"} and doc_ids(curated) == ["kb/refunds"]


# /answer

def test_answer_cites_the_restricted_document_for_an_allowed_token(corpus, start):
    _, base = start()
    status, _, body = call(base, "/answer", {"question": "merger budget"}, headers=bearer(FINANCE_TOKEN))
    assert status == 200
    assert set(body) == {"text", "authority", "abstain_reason", "citations"}
    assert body["abstain_reason"] is None and body["authority"] == "curated"
    assert "4 million" in body["text"]
    assert body["citations"][0]["doc_id"] == "kb/secret" and body["citations"][0]["evidence_id"] == "E1"
    assert "4 million" in body["citations"][0]["quote"]


def test_answer_for_an_unallowed_token_abstains(corpus, start):
    _, base = start()
    _, _, body = call(base, "/answer", {"question": "merger budget"}, headers=bearer(PUBLIC_TOKEN))
    assert body == {"text": "", "authority": "none", "abstain_reason": "no_evidence", "citations": []}


def test_answer_for_an_unanswerable_question_gives_an_abstain_reason(corpus, start):
    _, base = start()
    _, _, body = call(base, "/answer", {"question": "What is the capital of Mars?"}, headers=bearer(PUBLIC_TOKEN))
    assert body["abstain_reason"] == "no_evidence" and body["citations"] == [] and body["text"] == ""


def test_answer_with_a_raising_generator_is_generator_error_not_500(corpus, start):
    class Boom:
        model_id = "boom"

        def generate(self, *args, **kwargs):
            raise RuntimeError("upstream down")

    _, base = start(generator=Boom())
    status, _, body = call(base, "/answer", {"question": "lobby opens"}, headers=bearer(PUBLIC_TOKEN))
    assert status == 200 and body["abstain_reason"] == "generator_error"


# /fact

def test_fact_returns_the_value_for_an_allowed_token(corpus, start):
    _, base = start()
    status, _, body = call(base, "/fact", {"key": "merger_budget_usd"}, headers=bearer(FINANCE_TOKEN))
    assert (status, body) == (200, {"key": "merger_budget_usd", "value": "4000000", "unit": None})


def test_fact_is_null_for_a_disallowed_token_and_for_an_unknown_key(corpus, start):
    _, base = start()
    status, _, body = call(base, "/fact", {"key": "merger_budget_usd"}, headers=bearer(PUBLIC_TOKEN))
    assert (status, body) == (200, None)
    _, _, body = call(base, "/fact", {"key": "no_such_key"}, headers=bearer(FINANCE_TOKEN))
    assert body is None


# validation

@pytest.mark.parametrize("path, raw", [
    ("/search", b"\xff\xfe not utf-8"),
    ("/search", b"{not json"),
    ("/search", b"[1, 2]"),
    ("/search", b"[" * 20_000),
    ("/search", b'"just a string"'),
    ("/search", b"{}"),
    ("/search", b'{"question": 7}'),
    ("/search", b'{"question": "   "}'),
    ("/search", json.dumps({"question": "x" * 2001}).encode()),
    ("/search", b'{"question": "lobby", "min_authority": "all"}'),
    ("/search", b'{"question": "lobby", "min_authority": 1}'),
    ("/search", b'{"question": "lobby", "min_authority": null}'),
    ("/answer", b'{"question": ""}'),
    ("/fact", b"{}"),
    ("/fact", b'{"key": 3}'),
    ("/fact", json.dumps({"key": "k" * 2001}).encode()),
])
def test_invalid_bodies_are_400(db, start, path, raw):
    _, base = start()
    status, _, body = call(base, path, raw=raw, headers=bearer(PUBLIC_TOKEN))
    assert status == 400
    assert list(body) == ["error"] and isinstance(body["error"], str)


def test_question_of_exactly_2000_characters_is_accepted(corpus, start):
    _, base = start()
    status, _, _ = call(base, "/search", {"question": "x" * 2000}, headers=bearer(PUBLIC_TOKEN))
    assert status == 200


def test_missing_content_length_is_411(db, start):
    server, _ = start()
    status, _, body = raw_post(server.server_address, {**bearer(PUBLIC_TOKEN), "Content-Type": "application/json"})
    assert (status, list(body)) == (411, ["error"])


@pytest.mark.parametrize("length", ["abc", "-5", "1.5"])
def test_non_integer_content_length_is_411(db, start, length):
    server, _ = start()
    headers = {**bearer(PUBLIC_TOKEN), "Content-Type": "application/json", "Content-Length": length}
    assert raw_post(server.server_address, headers)[0] == 411


def test_content_length_above_limit_is_413_without_reading_the_body(corpus, start):
    server, _ = start()
    headers = {**bearer(PUBLIC_TOKEN), "Content-Type": "application/json", "Content-Length": str(MAX_BODY + 1)}
    # The client sends no body. A server that tried to read it would stall until the client timeout.
    status, _, body = raw_post(server.server_address, headers)
    assert (status, list(body)) == (413, ["error"])


def test_body_of_exactly_the_limit_is_read(corpus, start):
    server, _ = start()
    core = b'{"question": "lobby opens"}'
    body = core + b" " * (MAX_BODY - len(core))
    headers = {**bearer(PUBLIC_TOKEN), "Content-Type": "application/json", "Content-Length": str(MAX_BODY)}
    status, _, found = raw_post(server.server_address, headers, body)
    assert status == 200 and doc_ids(found) == ["kb/lobby"]


# 500 path and recovery

class FlakyEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.fail_next = True

    def embed(self, texts):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("embedder exploded: detail-that-must-not-leak")
        return super().embed(texts)


def test_handler_exception_is_500_with_generic_body_and_the_server_recovers(corpus, start, caplog):
    caplog.set_level(logging.ERROR, logger="governed_rag.service")
    _, base = start(embedder=FlakyEmbedder(), pool_size=1)
    status, _, body = call(base, "/search", {"question": "lobby opens"}, headers=bearer(PUBLIC_TOKEN))
    assert (status, body) == (500, {"error": "internal error"})
    records = [r for r in caplog.records if r.name == "governed_rag.service"]
    assert records and records[0].exc_info is not None
    status, _, body = call(base, "/search", {"question": "lobby opens"}, headers=bearer(PUBLIC_TOKEN))
    assert status == 200 and doc_ids(body) == ["kb/lobby"]
    assert call(base, "/healthz")[0] == 200


# pool and concurrency

def test_eight_concurrent_searches_with_a_pool_of_two_all_succeed(corpus, start):
    server, base = start(pool_size=2)

    def one(_):
        return call(base, "/search", {"question": "lobby opens"}, headers=bearer(PUBLIC_TOKEN))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, range(8)))
    assert [status for status, _, _ in results] == [200] * 8
    assert all(doc_ids(body) == ["kb/lobby"] for _, _, body in results)
    assert len(server.service.pool.open_connections) <= 2


def test_busy_pool_is_503_after_the_wait(corpus, start, monkeypatch):
    monkeypatch.setattr(service, "POOL_WAIT_SECONDS", 0.2)
    server, base = start(pool_size=1)
    with server.service.pool.borrow():
        status, _, body = call(base, "/search", {"question": "lobby opens"}, headers=bearer(PUBLIC_TOKEN))
        assert (status, body) == (503, {"error": "busy"})
        status, _, body = call(base, "/healthz")
        assert (status, body) == (503, {"status": "unavailable"})
    assert call(base, "/search", {"question": "lobby opens"}, headers=bearer(PUBLIC_TOKEN))[0] == 200


# server_close and serve

def test_server_close_closes_pooled_connections_and_the_socket(corpus, start):
    server, base = start(pool_size=2)
    call(base, "/search", {"question": "lobby opens"}, headers=bearer(PUBLIC_TOKEN))
    call(base, "/healthz")
    conns = server.service.pool.open_connections
    assert conns and not any(c.closed for c in conns)
    server.shutdown()
    server.server_close()
    assert all(c.closed for c in conns)
    assert server.socket.fileno() == -1


def test_serve_closes_the_server_on_keyboard_interrupt(db):
    server = make_server("127.0.0.1", 0, **_options(db))

    def interrupted():
        raise KeyboardInterrupt

    server.serve_forever = interrupted
    serve(server)
    assert server.socket.fileno() == -1
