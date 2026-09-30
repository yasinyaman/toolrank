import os

import numpy as np
import pytest

from toolrank.adapters.index_numpy import NumpyIndex
from toolrank.ports import VectorIndex


def _unit(n, d, seed):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _exercise(idx):
    """The same changes on any VectorIndex; -> the search results after them."""
    v = _unit(6, 8, 0)
    ids = [f"t{i}" for i in range(6)]
    idx.apply(ids, [f"h{i}" for i in range(6)], v)
    idx.apply(["t1", "t9"], ["h1b", "h9"], _unit(2, 8, 1), delete=["t0", "t5"])
    assert idx.hashes() == {"t1": "h1b", "t2": "h2", "t3": "h3", "t4": "h4", "t9": "h9"}
    q = _unit(3, 8, 2)
    got_ids, got_scores = idx.search(q, 10)  # k larger than the index
    assert all(len(row) == 5 for row in got_ids)
    assert idx.search(q, 0) == ([[], [], []], [[], [], []])
    return got_ids, got_scores


def _reference():
    ref = NumpyIndex()
    return _exercise(ref)


def test_faiss_flat_matches_numpy_and_hnsw_finds_the_neighbours(tmp_path):
    pytest.importorskip("faiss")
    from toolrank.adapters.index_faiss import FaissIndex

    flat = FaissIndex(kind="flat")
    assert isinstance(flat, VectorIndex) and flat.name == "faiss-flat"
    ids, scores = _exercise(flat)
    ref_ids, ref_scores = _reference()
    assert ids == ref_ids and np.allclose(scores, ref_scores, atol=1e-6)

    data, queries = _unit(2000, 32, 3), _unit(50, 32, 4)
    ids = [f"t{i}" for i in range(2000)]
    exact, hnsw = NumpyIndex(), FaissIndex(tmp_path / "faiss.npz")
    for idx in (exact, hnsw):
        idx.apply(ids, ids, data)
    want, _ = exact.search(queries, 10)
    got, _ = hnsw.search(queries, 10)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(want, got, strict=True)])
    assert recall >= 0.95
    reloaded = FaissIndex(tmp_path / "faiss.npz")  # the graph comes back from the snapshot
    assert reloaded._faiss is not None and reloaded.search(queries, 10)[0] == got


@pytest.mark.skipif(
    not os.environ.get("TOOLRANK_PG_DSN"), reason="needs TOOLRANK_PG_DSN (a Postgres with pgvector)"
)
def test_pgvector_matches_numpy():
    pytest.importorskip("psycopg")
    from toolrank.adapters.index_pgvector import PgVectorIndex

    idx = PgVectorIndex(os.environ["TOOLRANK_PG_DSN"], table="toolrank_test_tools")
    idx._db().execute("DROP TABLE IF EXISTS toolrank_test_tools")
    idx.close()
    try:
        assert isinstance(idx, VectorIndex)
        ids, scores = _exercise(idx)
        ref_ids, ref_scores = _reference()
        assert ids == ref_ids and np.allclose(scores, ref_scores, atol=1e-5)
    finally:
        idx._db().execute("DROP TABLE IF EXISTS toolrank_test_tools")
        idx.close()
    with pytest.raises(ValueError, match="table name"):
        PgVectorIndex("postgresql://x", table="drop table; --")
