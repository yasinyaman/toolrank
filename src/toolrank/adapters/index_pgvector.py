"""A ``VectorIndex`` in Postgres with pgvector: tool vectors next to the application's data, shared
by every process that can reach the database.

One table, ``<table>(id text primary key, hash text, embedding vector)``. ``apply`` is one
transaction: deletes, then the new and changed rows copied (binary ``COPY``) into a temp table and
upserted. Search is an exact scan ordered by ``<#>`` (negative inner product; score = its negation):
pgvector's HNSW index stops at 2,000 dimensions for ``vector`` and 4,000 for ``halfvec``, below
Qwen3-Embedding's 4096, so an ANN index here needs binary quantisation or a subvector index with
re-ranking (backlog). ``psycopg`` 3 and ``pgvector`` (the ``[pgvector]`` extra) are imported inside.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from toolrank.ports import IndexChanged


class PgVectorIndex:
    name = "pgvector"

    def __init__(self, dsn: str, table: str = "toolrank_tools"):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", table):
            raise ValueError(f"table name {table!r}: letters, digits and _ only")
        self.dsn, self.table = dsn, table
        self._conn: Any = None
        self._lock = threading.RLock()  # one connection, used from several threads

    def _db(self) -> Any:
        with self._lock:
            return self._connect()

    def _connect(self) -> Any:
        if self._conn is None:
            import psycopg
            from pgvector.psycopg import register_vector

            conn = psycopg.connect(self.dsn, autocommit=True)
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} "
                "(id text PRIMARY KEY, hash text NOT NULL, embedding vector NOT NULL)"
            )
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def hashes(self) -> dict[str, str]:
        with self._lock:
            return dict(self._db().execute(f"SELECT id, hash FROM {self.table}").fetchall())

    def apply(
        self,
        ids: Sequence[str],
        hashes: Sequence[str],
        vectors: np.ndarray,
        delete: Sequence[str] = (),
        expect: Mapping[str, str] | None = None,
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(ids) != len(hashes) or len(ids) != vectors.shape[0]:
            raise ValueError(f"{len(ids)} ids, {len(hashes)} hashes, {vectors.shape[0]} vectors")
        with self._lock:
            self._apply(ids, hashes, vectors, delete, expect or {})

    def _apply(
        self,
        ids: Sequence[str],
        hashes: Sequence[str],
        vectors: np.ndarray,
        delete: Sequence[str],
        expect: Mapping[str, str],
    ) -> None:
        conn = self._db()
        with conn.transaction(), conn.cursor() as cur:
            if expect:  # the rows this writer keeps, locked and checked before anything changes
                cur.execute(
                    f"SELECT id, hash FROM {self.table} WHERE id = ANY(%s) FOR UPDATE", (list(expect),)
                )
                stored = dict(cur.fetchall())
                changed = [i for i, h in expect.items() if stored.get(i) != h]
                if changed:
                    raise IndexChanged(changed)
            if delete:
                cur.execute(f"DELETE FROM {self.table} WHERE id = ANY(%s)", (list(delete),))
            if len(ids):
                cur.execute(f"CREATE TEMP TABLE staging (LIKE {self.table}) ON COMMIT DROP")
                with cur.copy("COPY staging (id, hash, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
                    copy.set_types(["text", "text", "vector"])
                    for i, h, v in zip(ids, hashes, vectors, strict=True):
                        copy.write_row([i, h, v])
                cur.execute(
                    f"INSERT INTO {self.table} SELECT * FROM staging "
                    "ON CONFLICT (id) DO UPDATE SET hash = EXCLUDED.hash, embedding = EXCLUDED.embedding"
                )

    def search(self, queries: np.ndarray, k: int) -> tuple[list[list[str]], list[list[float]]]:
        conn = self._db()
        ids: list[list[str]] = []
        scores: list[list[float]] = []
        sql = f"SELECT id, -(embedding <#> %s) FROM {self.table} ORDER BY embedding <#> %s, id LIMIT %s"
        for q in np.asarray(queries, dtype=np.float32):
            with self._lock:
                rows = conn.execute(sql, (q, q, max(k, 0))).fetchall() if k > 0 else []
            ids.append([r[0] for r in rows])
            scores.append([float(r[1]) for r in rows])
        return ids, scores
