"""A second scorer over a first scorer's shortlist: the local counterpart of ``JevReranker``.

``ScorerReranker`` ranks with ``base`` to ``depth`` and reorders that head by ``second.score_tools``
(a ``DenseScorer`` or ``CLMScorer``: cosine between the query and each candidate, encoded with the
second scorer's own endpoint, formats and heads); ties keep the base order and the base list
continues below ``depth`` with negative scores, so a top-100 list stays complete. The scores of the
head are the second scorer's, and so is ``score_kind``: with a cosine scorer on top the adaptive
cut applies. Candidates are encoded through the second encoder's cache, so a corpus that was
encoded once in that setting costs nothing to rerank. While a server answers a request
(``ports.rerank_failures`` set), a second stage that fails gives way to the first stage's list.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from toolrank.domain import Query, RankedList, Tool
from toolrank.formats import NamedFormatter
from toolrank.ports import rerank_failures

T = TypeVar("T")

__all__ = ["ScorerReranker", "Slots", "cut_formatter"]


def cut_formatter(f: NamedFormatter, max_chars: int) -> NamedFormatter:
    """``f``'s text cut to ``max_chars`` characters, named ``<f>[:<n>]``: the text a Jev option gets
    (``--jev-max-chars``), so a second scorer can read exactly what Jev read."""
    return NamedFormatter(f"{f.name}[:{max_chars}]", lambda t: f(t)[:max_chars])


def first_stage(base: Any, queries: Sequence[Query], k: int) -> tuple[list[RankedList], list[RankedList]]:
    """``base``'s lists and the cosine lists an adaptive cut counts on (a hybrid's semantic arm,
    else the lists themselves), without touching any scorer's shared state."""
    if hasattr(base, "rank_pairs"):
        pairs = base.rank_pairs(queries, k)
        return [f for f, _ in pairs], [s for _, s in pairs]
    lists = base.rank(queries, k)
    return lists, lists


def or_first_stage(
    rerank: Callable[[], list[RankedList]], base: list[RankedList], k: int
) -> list[RankedList]:
    """``rerank()``; when it fails while a server answers (``ports.rerank_failures`` is a list, which
    keeps the reason), ``base`` cut to ``k``. Eval sets no list: the failure raises."""
    try:
        return rerank()
    except Exception as e:
        failures = rerank_failures.get()
        if failures is None:
            raise
        failures.append(f"{type(e).__name__}: {e}")
        return [RankedList(r.query_id, r.tool_ids[:k], r.scores[:k]) for r in base]


class Slots:
    """At most ``n`` calls to a second stage's endpoint at once (a local server with one worker
    would queue the rest); a call that waits ``timeout`` seconds for its turn fails, and a server's
    request takes the first stage's list."""

    def __init__(self, n: int, timeout: float):
        self.n, self.timeout = max(1, n), timeout
        self._sem = threading.BoundedSemaphore(self.n)

    @contextlib.contextmanager
    def __call__(self, what: str) -> Iterator[None]:
        if not self._sem.acquire(timeout=self.timeout):
            raise RuntimeError(f"{what} busy: {self.n} calls in flight for {self.timeout:g} s")
        try:
            yield
        finally:
            self._sem.release()


def pool_map(fn: Callable[[int], T], n: int, workers: int) -> list[T]:
    """``fn(0..n-1)``, in ``workers`` threads when there is more than one item, each under a copy of
    the caller's context (the request's cache scope travels with it)."""
    if workers <= 1 or n <= 1:
        return [fn(i) for i in range(n)]
    contexts = [contextvars.copy_context() for _ in range(n)]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda i: contexts[i].run(fn, i), range(n)))


class ScorerReranker:
    def __init__(
        self, base: Any, second: Any, *, depth: int = 100, max_chars: int | None = None, workers: int = 1
    ):
        if depth < 2:
            raise ValueError("--rerank-depth must be at least 2")
        if not hasattr(second, "score_tools"):
            raise ValueError(f"{second.name} cannot rerank: it has no score_tools (dense and clm can)")
        if max_chars:  # the second scorer reads cut candidate text, and says so in its name
            old = second.tool_format
            second.tool_format = cut_formatter(old, max_chars)
            tail = f"/{old.name}"
            if second.name.endswith(tail):  # the format last (a cross-encoder's name)
                second.name = second.name[: -len(old.name)] + second.tool_format.name
            else:
                second.name = second.name.replace(f"/{old.name}/", f"/{second.tool_format.name}/", 1)
        self.base, self.second, self.depth, self.workers = base, second, depth, max(1, workers)
        self.score_kind = getattr(second, "score_kind", "cosine")
        self.tool_format, self.query_format = base.tool_format, base.query_format
        self.name = f"rerank[{second.name},d{depth}]/{base.name}"
        self.tools: dict[str, Tool] = {}
        self.last_base: dict[str, RankedList] = {}

    def index(self, tools: Iterable[Tool]) -> None:
        tools = list(tools)
        self.tools = {t.id: t for t in tools}
        self.base.index(tools)

    def rank(self, queries: Sequence[Query], k: int) -> list[RankedList]:
        base = self.base.rank(queries, max(k, self.depth))
        self.last_base = {r.query_id: r for r in base}
        return self._rerank(queries, base, k)

    def rank_pairs(self, queries: Sequence[Query], k: int) -> list[tuple[RankedList, RankedList]]:
        """(reranked, the first stage's cosine list) per query, for a server: no shared state is
        written, and an adaptive K counts on the cosines (a second stage's scores have no margin)."""
        base, semantic = first_stage(self.base, queries, max(k, self.depth))
        reranked = or_first_stage(lambda: self._rerank(queries, base, k), base, k)
        return list(zip(reranked, semantic, strict=True))

    def _rerank(self, queries: Sequence[Query], base: list[RankedList], k: int) -> list[RankedList]:
        heads = [r.tool_ids[: self.depth] for r in base]

        def score(i: int) -> list[float]:
            return self.second.score_tools(queries[i], [self.tools[t] for t in heads[i]]) if heads[i] else []

        scored = pool_map(score, len(queries), self.workers)  # vLLM's score API batches concurrent requests
        out = []
        for r, head, s in zip(base, heads, scored, strict=True):
            order = sorted(range(len(head)), key=lambda i: (-s[i], i))
            ids = [head[i] for i in order] + r.tool_ids[len(head) :]
            scores = [float(s[i]) for i in order] + [
                -float(n + 1) for n in range(len(r.tool_ids) - len(head))
            ]
            out.append(RankedList(r.query_id, ids[:k], scores[:k]))
        return out

    def score_tools(self, query: Query, tools: Sequence[Tool]) -> list[float]:
        """The first stage's cosines, as ``JevReranker``'s: what ``/v1/rank`` promises and an
        adaptive cut's margin is chosen for."""
        return self.base.score_tools(query, tools)
