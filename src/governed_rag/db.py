from importlib.resources import files

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row


def connect(url: str) -> psycopg.Connection:
    conn = psycopg.connect(url, autocommit=True, row_factory=dict_row)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(files("governed_rag").joinpath("schema.sql").read_text())
