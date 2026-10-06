"""Dense bi-encoder scorer: one ``TextEncoder`` for both sides, cosine / inner-product top-k.

This is the "off-the-shelf embedding" row of the competitor table (Qwen3-Embedding, bge, e5 ...)
and also CLM's ``clm-raw`` ablation (backbone embeddings without the heads).

Tool vectors live in a ``VectorIndex`` (default: an in-memory ``NumpyIndex``). Each row is keyed
by a hash of the scorer's ``fingerprint`` (encoder, heads) and the tool's text, so ``index()`` on
a persistent index embeds only new or changed tools and drops vanished ones: the action-vector
cache. A fresh index receives every tool in one batch, in order, which keeps eval results exactly
as they were before indexes existed.

Server routing (``server_weight`` > 0, off by default): every server (a tool's ``category``) is
embedded as one summary text (``formats.server_summary``) and a tool's score becomes
``cosine(query, tool) + server_weight * cosine(query, its server)``, over the index's top
``ROUTE_DEPTH`` tools. A soft "which server?" vote: picking servers first and searching only those
(MCP-Zero's pattern) lost points on every set, since the right server is first only 70-85% of the
time (``scripts/routing_sweep.py``, ``docs/reports/faz2-week5.md``). A catalogue of one server is
left as it is.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from toolrank.adapters.embeddings_api import l2_normalize
from toolrank.adapters.index_numpy import NumpyIndex, topk_dot
from toolrank.domain import Query, RankedList, Tool
from toolrank.formats import QUERY_FORMATS, TOOL_FORMATS, NamedFormatter, server_summary
from toolrank.ports import IndexChanged, TextEncoder, VectorIndex, within_visible

__all__ = ["DenseScorer", "row_hash", "topk_dot"]

ROUTE_DEPTH = 100  # tools re-scored with their server's term; a tool below this rank stays out


def row_hash(fingerprint: str, text: str) -> str:
    return hashlib.sha256((fingerprint + "\x00" + text).encode("utf-8")).hexdigest()[:16]


class DenseScorer:
    name = "dense"
    score_kind = "cosine"

    def __init__(
        self,
        encoder: TextEncoder,
        tool_format: NamedFormatter | str = "name_desc",
        query_format: NamedFormatter | str = "plain",
        *,
        project_tools: Callable[[np.ndarray], np.ndarray] | None = None,
        project_queries: Callable[[np.ndarray], np.ndarray] | None = None,
        label: str | None = None,
        index: VectorIndex | None = None,
        fingerprint: str = "",
        server_weight: float = 0.0,
    ):
        self.encoder = encoder
        self.tool_format = TOOL_FORMATS[tool_format] if isinstance(tool_format, str) else tool_format
        self.query_format = QUERY_FORMATS[query_format] if isinstance(query_format, str) else query_format
        self.project_tools, self.project_queries = project_tools, project_queries
        self.vindex: VectorIndex = index if index is not None else NumpyIndex()
        self.fingerprint = fingerprint
        self.server_weight = float(server_weight)
        self._servers: np.ndarray | None = None  # [servers, dim], set by index() when routing is on
        self._server_of: dict[str, int] = {}
        suffix = "" if self.vindex.name == "numpy" else f"@{self.vindex.name}"
        if self.server_weight:
            suffix += f"+srv{self.server_weight:g}"
        self.name = (
            f"{label or 'dense'}/{encoder.name}/{self.tool_format.name}/{self.query_format.name}{suffix}"
        )
        self.last_sync: dict[str, int] = {}

    def _vectors(
        self, texts: list[str], kind: str, project: Callable[[np.ndarray], np.ndarray] | None
    ) -> np.ndarray:
        m = self.encoder.encode(texts, kind=kind)
        if project is not None:
            m = project(m)
        return l2_normalize(np.asarray(m, dtype=np.float32))

    def index(self, tools: Iterable[Tool]) -> None:
        tools = list(tools)
        ids = [t.id for t in tools]
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        if dupes:
            raise ValueError(f"duplicate tool ids: {dupes[:5]}")
        texts = [self.tool_format(t) for t in tools]
        hashes = [row_hash(self.fingerprint, x) for x in texts]
        present = set(ids)
        embedded = 0
        for attempt in range(4):  # a persistent index another writer (other flags) changes meanwhile
            have = self.vindex.hashes()
            todo = [n for n, (i, h) in enumerate(zip(ids, hashes, strict=True)) if have.get(i) != h]
            kept = {ids[n]: hashes[n] for n in range(len(ids)) if have.get(ids[n]) == hashes[n]}
            gone = [i for i in have if i not in present]
            if todo:
                vecs = self._vectors([texts[n] for n in todo], "document", self.project_tools)
            else:
                vecs = np.zeros((0, 0), dtype=np.float32)
            embedded += len(todo)
            if not (todo or gone):
                break
            try:
                self.vindex.apply([ids[n] for n in todo], [hashes[n] for n in todo], vecs, gone, expect=kept)
                break
            except IndexChanged:
                if attempt == 3:
                    raise
        self.last_sync = {"embedded": embedded, "removed": len(gone), "kept": len(ids) - len(todo)}
        self._route(tools)

    def _route(self, tools: Sequence[Tool]) -> None:
        """The servers' summary vectors, in the tools' space (one cached embedding per server)."""
        self._servers, self._server_of = None, {}
        if not self.server_weight:
            return
        groups: dict[str, list[Tool]] = {}
        for t in tools:
            groups.setdefault(t.category, []).append(t)
        if len(groups) < 2:
            return  # one server: its term would be the same for every tool
        names = sorted(groups)
        texts = [server_summary(n, groups[n]) for n in names]
        self._servers = self._vectors(texts, "document", self.project_tools)
        self._server_of = {t.id: i for i, n in enumerate(names) for t in groups[n]}

    def score_tools(self, query: Query, tools: Sequence[Tool]) -> list[float]:
        """Cosine of ``query`` with tools that need not be in the index (``/v1/rank``)."""
        if not tools:
            return []
        m = self._vectors([self.tool_format(t) for t in tools], "document", self.project_tools)
        q = self._vectors([self.query_format(query)], "query", self.project_queries)
        return [float(x) for x in m @ q[0]]

    def rank(self, queries: Sequence[Query], k: int) -> list[RankedList]:
        if not queries:
            return []
        q = self._vectors([self.query_format(x) for x in queries], "query", self.project_queries)
        if self._servers is None:
            ids, scores = within_visible(lambda depth: self.vindex.search(q, depth), k)
            return [RankedList(x.id, ids[i], scores[i]) for i, x in enumerate(queries)]
        ids, scores = within_visible(lambda depth: self.vindex.search(q, depth), max(k, ROUTE_DEPTH))
        term = self.server_weight * (q @ self._servers.T)
        out: list[RankedList] = []
        term = np.concatenate([term, np.zeros((len(queries), 1), dtype=term.dtype)], axis=1)
        for i, x in enumerate(queries):
            server = [self._server_of.get(t, -1) for t in ids[i]]  # -1: a row index() did not see, no term
            routed = np.asarray(scores[i], dtype=np.float32) + term[i, server]
            order = np.argsort(-routed, kind="stable")[:k]
            out.append(RankedList(x.id, [ids[i][j] for j in order], [float(routed[j]) for j in order]))
        return out
