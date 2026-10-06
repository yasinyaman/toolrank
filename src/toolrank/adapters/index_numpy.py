"""The default ``VectorIndex``: exact inner product over a float32 matrix, optionally on disk.

Rows stay in insertion order, so an index built in one ``apply`` holds exactly the matrix the
scorer computed and ``search`` (``topk_dot``) ranks like the pre-index code did. With a ``path``
the index is one snapshot file (ids, row hashes and vectors in one ``.npz``) replaced atomically
on every ``apply`` while an exclusive ``flock`` on ``<path>.lock`` is held, so two writers (an
ingest warm-up and a server) never interleave and a reader never pairs new ids with old rows.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from toolrank.ports import IndexChanged

try:  # POSIX; elsewhere writers are not serialised
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


def topk_dot(q: np.ndarray, m: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by inner product: -> (indices [n, k], scores [n, k]), best first."""
    k = min(k, m.shape[0])
    s = q @ m.T
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    ps = np.take_along_axis(s, part, axis=1)
    order = np.argsort(-ps, axis=1)
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(ps, order, axis=1)


class NumpyIndex:
    name = "numpy"

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._ids: list[str] = []
        self._hashes: list[str] = []
        self._m = np.zeros((0, 0), dtype=np.float32)
        self._seen: tuple[int, int, int] | None = None  # the snapshot file these rows came from
        if self.path is not None and self.path.exists():
            self._load()

    def _stamp(self) -> tuple[int, int, int] | None:
        try:
            st = os.stat(self.path) if self.path is not None else None
        except OSError:
            return None
        return (st.st_ino, st.st_size, st.st_mtime_ns) if st else None

    # -- storage ----------------------------------------------------------------------------
    def _load(self) -> None:
        assert self.path is not None
        with np.load(self.path, allow_pickle=False) as z:
            ids, hashes, m = z["ids"].tolist(), z["hashes"].tolist(), np.asarray(z["vectors"], np.float32)
            if not (len(ids) == len(hashes) == m.shape[0]):
                raise ValueError(f"{self.path}: {len(ids)} ids, {len(hashes)} hashes, {m.shape[0]} rows")
            self._ids, self._hashes, self._m = ids, hashes, m
            self._load_extra(z)
        self._seen = self._stamp()

    # hooks for subclasses that keep a search structure next to the rows (FaissIndex)
    def _load_extra(self, z: Any) -> None:
        pass

    def _extra_arrays(self) -> dict[str, np.ndarray]:
        return {}

    def _after_change(self) -> None:
        pass

    def _save(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(
                f,
                ids=np.array(self._ids, dtype=str),
                hashes=np.array(self._hashes, dtype=str),
                vectors=self._m,
                **self._extra_arrays(),
            )
        os.replace(tmp, self.path)
        self._seen = self._stamp()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if self.path is None or fcntl is None:
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    # -- port -------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._ids)

    def hashes(self) -> dict[str, str]:
        """The rows' hashes, reread first when another writer replaced the snapshot."""
        if self.path is not None and self._stamp() not in (None, self._seen):
            with self._locked():
                self._load()
        return dict(zip(self._ids, self._hashes, strict=True))

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
        with self._locked():
            if self.path is not None and self.path.exists():
                self._load()  # another writer may have moved on
            if expect:
                stored = dict(zip(self._ids, self._hashes, strict=True))
                changed = [i for i, h in expect.items() if stored.get(i) != h]
                if changed:
                    raise IndexChanged(changed)
            drop = set(delete)
            keep = [n for n, i in enumerate(self._ids) if i not in drop]
            out_ids = [self._ids[n] for n in keep]
            out_hashes = [self._hashes[n] for n in keep]
            m = self._m[keep] if keep else np.zeros((0, vectors.shape[1] if len(ids) else 0), np.float32)
            if len(ids) and m.shape[0] and m.shape[1] != vectors.shape[1]:
                if set(out_ids) - set(ids):
                    raise ValueError(
                        f"new vectors have {vectors.shape[1]} dims, stored ones {m.shape[1]}: replace all rows"
                    )
                out_ids, out_hashes, m = [], [], np.zeros((0, vectors.shape[1]), np.float32)
            pos = {i: n for n, i in enumerate(out_ids)}
            fresh = [j for j, i in enumerate(ids) if i not in pos]
            for j, i in enumerate(ids):
                if i in pos:
                    m[pos[i]] = vectors[j]
                    out_hashes[pos[i]] = hashes[j]
            if fresh:
                m = vectors[fresh] if m.shape[0] == 0 else np.concatenate([m, vectors[fresh]])
                out_ids += [ids[j] for j in fresh]
                out_hashes += [hashes[j] for j in fresh]
            self._ids, self._hashes, self._m = out_ids, out_hashes, np.ascontiguousarray(m, dtype=np.float32)
            self._after_change()
            if self.path is not None:
                self._save()

    def search(self, queries: np.ndarray, k: int) -> tuple[list[list[str]], list[list[float]]]:
        n = len(queries)
        if not self._ids or k <= 0:
            return [[] for _ in range(n)], [[] for _ in range(n)]
        idx, sc = topk_dot(np.asarray(queries, dtype=np.float32), self._m, k)
        return [[self._ids[int(j)] for j in row] for row in idx], [[float(s) for s in row] for row in sc]
