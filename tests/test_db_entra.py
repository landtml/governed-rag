"""Entra ID login for Azure Database for PostgreSQL, tested with a stand-in credential."""
from types import SimpleNamespace

import pytest

import governed_rag.db as db

AZURE = "postgresql://id-app@pg-x.postgres.database.azure.com:5432/rag?sslmode=require"


class _Credential:
    def __init__(self):
        self.scopes = []

    def get_token(self, scope):
        self.scopes.append(scope)
        return SimpleNamespace(token=f"token-{len(self.scopes)}")


@pytest.fixture
def credential(monkeypatch):
    cred = _Credential()
    monkeypatch.setattr(db, "_azure_credential", lambda: cred)
    return cred


def test_an_azure_host_without_a_password_gets_a_fresh_token_per_connection(credential):
    assert db.entra_password(AZURE) == "token-1"
    assert db.entra_password(AZURE) == "token-2"
    assert credential.scopes == [db.AZURE_PG_TOKEN_SCOPE] * 2


@pytest.mark.parametrize("url", [
    "postgresql://rag:rag@localhost:5433/rag",
    "postgresql://id-app:secret@pg-x.postgres.database.azure.com/rag",
    "host=pg-x.postgres.database.azure.com user=id-app password=secret dbname=rag",
    "postgresql://rag@db.example.com/rag",
])
def test_no_token_for_other_hosts_or_when_the_url_has_a_password(credential, url):
    assert db.entra_password(url) is None
    assert credential.scopes == []


def test_connect_passes_the_token_as_the_password(credential, monkeypatch):
    seen = {}

    class _Conn:
        def execute(self, sql):
            seen["sql"] = sql

    def fake_connect(url, **kwargs):
        seen.update(url=url, **kwargs)
        return _Conn()

    monkeypatch.setattr(db.psycopg, "connect", fake_connect)
    monkeypatch.setattr(db, "register_vector", lambda conn: None)
    db.connect(AZURE)
    assert seen["url"] == AZURE and seen["password"] == "token-1"
