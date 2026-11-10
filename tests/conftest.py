import os

import pytest

from governed_rag.db import apply_schema, connect

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag_test")


@pytest.fixture
def db():
    conn = connect(TEST_DATABASE_URL)
    apply_schema(conn)
    conn.execute("TRUNCATE documents, chunks, facts, query_log RESTART IDENTITY CASCADE")
    yield conn
    conn.close()
