"""A second scorer over a first scorer's shortlist: the local counterpart of ``JevReranker``.

``ScorerReranker`` ranks with ``base`` to ``depth`` and reorders that head by ``second.score_tools``
(a ``DenseScorer`` or ``CLMScorer``: cosine between the query and each candidate, encoded with the
second scorer's own endpoint, formats and heads); ties keep the base order and the base list
continues below ``depth`` with negative scores, so a top-100 list stays complete. The scores of the
head are the second scorer's, and so is ``score_kind``: with a cosine scorer on top the adaptive
cut applies. Candidates are encoded through the second encoder's cache, so a corpus that was
encoded once in that setting costs nothing to rerank.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from toolrank.domain import Query, RankedList, Tool
from toolrank.formats import NamedFormatter

__all__ = ["ScorerReranker", "cut_formatter"]


def cut_formatter(f: NamedFormatter, max_chars: int) -> NamedFormatter:
    """``f``'s text cut to ``max_chars`` characters, named ``<f>[:<n>]``: the text a Jev option gets
    (``--jev-max-chars``), so a second scorer can read exactly what Jev read."""
    return NamedFormatter(f"{f.name}[:{max_chars}]", lambda t: f(t)[:max_chars])


class ScorerReranker:
    def __init__(self, base: Any, second: Any, *, depth: int = 100, max_chars: int | None = None):
        if depth < 2:
            raise ValueError("--rerank-depth must be at least 2")
        if not hasattr(second, "score_tools"):
            raise ValueError(f"{second.name} cannot rerank: it has no score_tools (dense and clm can)")
        if max_chars:  # the second scorer reads cut candidate text, and says so in its name
            old = second.tool_format
            second.tool_format = cut_formatter(old, max_chars)
            second.name = second.name.replace(f"/{old.name}/", f"/{second.tool_format.name}/", 1)
        self.base, self.second, self.depth = base, second, depth
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
        out = []
        for q, r in zip(queries, base, strict=True):
            head = r.tool_ids[: self.depth]
            s = self.second.score_tools(q, [self.tools[t] for t in head]) if head else []
            order = sorted(range(len(head)), key=lambda i: (-s[i], i))
            ids = [head[i] for i in order] + r.tool_ids[len(head) :]
            scores = [float(s[i]) for i in order] + [
                -float(n + 1) for n in range(len(r.tool_ids) - len(head))
            ]
            out.append(RankedList(r.query_id, ids[:k], scores[:k]))
        return out

    def score_tools(self, query: Query, tools: Sequence[Tool]) -> list[float]:
        return self.second.score_tools(query, tools)
