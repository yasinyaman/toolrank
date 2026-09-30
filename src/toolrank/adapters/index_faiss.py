"""A ``VectorIndex`` with approximate search: a FAISS HNSW graph over ``NumpyIndex`` storage.

For catalogues where an exact scan starts to cost; at typical tool counts the default numpy index
is exact and fast enough. Inner product on unit vectors is cosine. HNSW cannot delete, so every
``apply`` rebuilds the graph (``ef_construction`` 200), and the snapshot stores the serialised graph
next to the rows, so loading does not rebuild. ``kind="flat"`` (exact ``IndexFlatIP``) exists to
test the plumbing against numpy. ``faiss`` (the ``[faiss]`` extra) is imported inside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toolrank.adapters.index_numpy import NumpyIndex


class FaissIndex(NumpyIndex):
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        kind: str = "hnsw",
        m: int = 32,
        ef_search: int = 128,
        ef_construction: int = 200,
    ):
        if kind not in ("hnsw", "flat"):
            raise ValueError(f"unknown FAISS index kind {kind!r}")
        self.kind, self.m, self.ef_search, self.ef_construction = kind, m, ef_search, ef_construction
        self.name = f"faiss-{kind}"
        self._faiss: Any = None
        super().__init__(path)

    def _build(self) -> None:
        import faiss

        self._faiss = None
        if not self._ids:
            return
        d = self._m.shape[1]
        if self.kind == "flat":
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexHNSWFlat(d, self.m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.ef_construction
        index.add(self._m)
        if self.kind == "hnsw":
            index.hnsw.efSearch = self.ef_search
        self._faiss = index

    def _after_change(self) -> None:
        self._build()

    def _extra_arrays(self) -> dict[str, np.ndarray]:
        import faiss

        return {"faiss": faiss.serialize_index(self._faiss)} if self._faiss is not None else {}

    def _load_extra(self, z: Any) -> None:
        if "faiss" in z.files:
            import faiss

            self._faiss = faiss.deserialize_index(np.asarray(z["faiss"]))
            if self._faiss.ntotal != len(self._ids):
                self._build()
        else:
            self._build()

    def search(self, queries: np.ndarray, k: int) -> tuple[list[list[str]], list[list[float]]]:
        n = len(queries)
        if self._faiss is None or k <= 0:
            return [[] for _ in range(n)], [[] for _ in range(n)]
        kk = min(k, len(self._ids))
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if self.kind == "hnsw" and kk > self.ef_search:  # per-call parameters: no shared state written
            import faiss

            scores, idx = self._faiss.search(q, kk, params=faiss.SearchParametersHNSW(efSearch=kk))
        else:
            scores, idx = self._faiss.search(q, kk)
        ids = [[self._ids[int(j)] for j in row if j >= 0] for row in idx]
        return ids, [
            [float(s) for s, j in zip(srow, row, strict=True) if j >= 0]
            for srow, row in zip(scores, idx, strict=True)
        ]
