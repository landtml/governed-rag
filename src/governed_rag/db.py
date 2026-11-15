import threading
from importlib.resources import files

import psycopg
from pgvector.psycopg import register_vector
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

# Azure Database for PostgreSQL accepts a Microsoft Entra access token as the password.
AZURE_PG_HOST_SUFFIX = ".postgres.database.azure.com"
AZURE_PG_TOKEN_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"

_credential = None
_credential_lock = threading.Lock()


def _azure_credential():
    """One DefaultAzureCredential per process: it caches tokens and refreshes them before expiry.

    It uses the Azure CLI login on a developer machine and the managed identity inside Azure
    (set AZURE_CLIENT_ID to pick a user-assigned identity).
    """
    global _credential
    with _credential_lock:
        if _credential is None:
            from azure.identity import DefaultAzureCredential
            _credential = DefaultAzureCredential()
        return _credential


def entra_password(url: str) -> str | None:
    """A fresh Entra token for an Azure PostgreSQL host whose URL carries no password, else None.

    Fetched per connection, so a long-running server keeps opening connections after the
    previous token has expired. A password written into the URL always wins.
    """
    params = conninfo_to_dict(url)
    hosts = (params.get("host") or "").split(",")
    if params.get("password") or not all(h.endswith(AZURE_PG_HOST_SUFFIX) for h in hosts):
        return None
    return _azure_credential().get_token(AZURE_PG_TOKEN_SCOPE).token


def connect(url: str) -> psycopg.Connection:
    token = entra_password(url)
    extra = {"password": token} if token else {}
    conn = psycopg.connect(url, autocommit=True, row_factory=dict_row, **extra)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(files("governed_rag").joinpath("schema.sql").read_text())
