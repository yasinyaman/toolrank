import hashlib

import numpy as np
import pytest

from toolrank.adapters.dense import DenseScorer, row_hash, topk_dot
from toolrank.adapters.embeddings_api import l2_normalize
from toolrank.adapters.index_numpy import NumpyIndex
from toolrank.domain import Query, Tool
from toolrank.ports import VectorIndex


class _HashEncoder:
    """Deterministic pseudo-random vectors per text; records every encode call."""

    name = "hash"

    def __init__(self, dim=16):
        self.dim, self.calls = dim, []

    def encode(self, texts, *, kind="document"):
        if not texts:
            raise ValueError("encode([]) must not be called")
        self.calls.append(list(texts))
        rows = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).standard_normal(self.dim))
        return np.asarray(rows, dtype=np.float32)


def _tools(n=30, tag=""):
    return [
        Tool(id=f"t{i}", doc={"name": f"tool{i}", "description": f"does thing {i}{tag}"}) for i in range(n)
    ]


def _queries(n=5):
    return [Query(id=f"q{i}", text=f"request number {i}", qrels={}) for i in range(n)]


def test_numpy_index_apply_replace_delete_and_search():
    idx = NumpyIndex()
    assert isinstance(idx, VectorIndex) and idx.search(np.ones((2, 3), np.float32), 5) == ([[], []], [[], []])
    v = np.eye(3, dtype=np.float32)
    idx.apply(["a", "b", "c"], ["h1", "h2", "h3"], v)
    idx.apply(["b", "d"], ["h2x", "h4"], np.array([[0, 0, 1], [0.5, 0.5, 0]], np.float32), delete=["a"])
    assert idx.hashes() == {"b": "h2x", "c": "h3", "d": "h4"}
    ids, scores = idx.search(np.array([[0, 0, 1], [1, 0, 0]], np.float32), 10)
    assert ids[0][:2] == ["b", "c"] and len(ids[0]) == 3 and scores[1][0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        idx.apply(["x"], ["h"], np.ones((1, 5), np.float32))  # other width, other rows remain


def test_numpy_index_snapshot_roundtrip_and_width_change(tmp_path):
    path = tmp_path / "index" / "index.npz"
    a = NumpyIndex(path)
    a.apply(["a", "b"], ["1", "2"], np.eye(2, dtype=np.float32))
    b = NumpyIndex(path)
    assert b.hashes() == {"a": "1", "b": "2"} and b.search(np.array([[0, 1]], np.float32), 1)[0] == [["b"]]
    b.apply(["a", "b"], ["3", "4"], np.ones((2, 4), np.float32))  # every row replaced: width may change
    assert NumpyIndex(path).hashes() == {"a": "3", "b": "4"}
    with open(path, "wb") as f:
        np.savez(f, ids=np.array(["a"]), hashes=np.array(["1", "2"]), vectors=np.eye(2))
    with pytest.raises(ValueError, match="1 ids, 2 hashes"):
        NumpyIndex(path)


def test_dense_scorer_matches_the_pre_index_computation():
    tools, queries, enc = _tools(), _queries(), _HashEncoder()
    s = DenseScorer(enc, "name_desc", "plain")
    s.index(tools)
    got = s.rank(queries, k=7)
    m = l2_normalize(enc.encode([s.tool_format(t) for t in tools]))
    q = l2_normalize(enc.encode([s.query_format(x) for x in queries]))
    idx, sc = topk_dot(q, m, 7)
    for r, row, srow in zip(got, idx, sc, strict=True):
        assert r.tool_ids == [tools[int(j)].id for j in row] and r.scores == [float(x) for x in srow]
    assert s.name == "dense/hash/name_desc/plain" and s.last_sync == {"embedded": 30, "removed": 0, "kept": 0}


def test_incremental_sync_embeds_only_new_or_changed_tools(tmp_path):
    path = tmp_path / "index.npz"
    enc = _HashEncoder()
    DenseScorer(enc, "name_desc", index=NumpyIndex(path), fingerprint="fp1").index(_tools())
    warm = DenseScorer(enc, "name_desc", index=NumpyIndex(path), fingerprint="fp1")
    warm.index(_tools())
    assert warm.last_sync == {"embedded": 0, "removed": 0, "kept": 30} and len(enc.calls) == 1
    changed = _tools()
    changed[3] = Tool(id="t3", doc={"name": "tool3", "description": "now does something else"})
    warm.index(changed[:-2])
    assert warm.last_sync == {"embedded": 1, "removed": 2, "kept": 27} and len(enc.calls[-1]) == 1
    other = DenseScorer(enc, "name_desc", index=NumpyIndex(path), fingerprint="fp2")  # new heads/encoder
    other.index(changed[:-2])
    assert other.last_sync["embedded"] == 28
    ranked = other.rank(_queries(1), k=3)[0]
    fresh = DenseScorer(_HashEncoder(), "name_desc", fingerprint="fp2")
    fresh.index(changed[:-2])
    assert fresh.rank(_queries(1), k=3)[0].tool_ids == ranked.tool_ids


def test_duplicate_tool_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate tool ids"):
        DenseScorer(_HashEncoder(), "name_desc").index(_tools(3) + _tools(1))


def test_a_writer_with_other_settings_never_leaves_its_rows_kept_as_ours(tmp_path):
    """Two processes on one DATA/index with different flags: rows the other one rewrote must be
    embedded again, whether it wrote before our index() or between our look and our write."""
    path = tmp_path / "index.npz"
    tools = _tools()
    mine = DenseScorer(_HashEncoder(), "name_desc", index=NumpyIndex(path), fingerprint="fp1")
    mine.index(tools)
    DenseScorer(_HashEncoder(), "name_desc", index=NumpyIndex(path), fingerprint="fp2").index(tools)
    mine.index(tools)  # the snapshot changed since we read it: reread, not "kept"
    assert mine.last_sync["embedded"] == 30
    want = {
        t.id: h for t, h in zip(tools, [row_hash("fp1", mine.tool_format(t)) for t in tools], strict=True)
    }
    assert NumpyIndex(path).hashes() == want

    # the other writer gets in between our look at the hashes and our apply
    late = DenseScorer(_HashEncoder(), "name_desc", index=NumpyIndex(path), fingerprint="fp1")
    look = late.vindex.hashes
    calls = []

    def racing():
        seen = look()
        if not calls:
            DenseScorer(_HashEncoder(), "name_desc", index=NumpyIndex(path), fingerprint="fp2").index(
                tools[:5]
            )
        calls.append(1)
        return seen

    late.vindex.hashes = racing
    changed = [*tools[:-1], Tool(id="new", doc={"name": "new", "description": "a new tool"})]
    late.index(changed)
    stored = NumpyIndex(path).hashes()
    assert len(calls) == 2 and all(stored[t.id] == row_hash("fp1", late.tool_format(t)) for t in changed)
