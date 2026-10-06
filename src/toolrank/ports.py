"""Ports: the interfaces the core talks to. Adapters live in ``toolrank.adapters``.

Keeping these as ``Protocol`` classes means a scorer can be swapped without touching the
runner, the CLI or the datasets - the whole point of the Phase 0 go/no-go on CLM.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from typing import Protocol, runtime_checkable

import numpy as np

from toolrank.domain import Query, RankedList, Tool

# Whose request is being answered: an API key's name, or None. Caches keyed by request text (query
# embeddings, second-stage scores and answers) keep a named key's entries apart, so how fast an
# answer comes tells one key nothing about another key's requests. The shared catalogue is indexed
# outside any scope.
cache_scope: ContextVar[str | None] = ContextVar("toolrank_cache_scope", default=None)


# The tool ids a key limited to some sources may see while a request runs, or None for all. The
# first-stage scorers rank within them, so fusion and a second stage only ever see those tools:
# their scores carry no trace of where hidden tools ranked, and no hidden tool's text is sent out.
visible_ids: ContextVar[frozenset[str] | None] = ContextVar("toolrank_visible_ids", default=None)


def within_visible(
    search: Callable[[int], tuple[list[list[str]], list[list[float]]]], k: int
) -> tuple[list[list[str]], list[list[float]]]:
    """``search(depth)`` (ids and scores per query, best first) cut to ``visible_ids``: deeper, four
    times at a time, until every query has ``k`` visible ids or the index has no more."""
    allowed = visible_ids.get()
    if allowed is None:
        return search(k)
    depth = k
    while True:
        ids, scores = search(depth)
        kept = [
            [(t, s) for t, s in zip(i, sc, strict=True) if t in allowed]
            for i, sc in zip(ids, scores, strict=True)
        ]
        if all(len(r) >= k for r in kept) or all(len(i) < depth for i in ids):
            break
        depth *= 4
    return [[t for t, _ in r][:k] for r in kept], [[s for _, s in r][:k] for r in kept]


def scoped(key_text: str) -> str:
    """``key_text`` within the current ``cache_scope`` (unchanged outside one)."""
    scope = cache_scope.get()
    return key_text if scope is None else f"scope={scope}\x00{key_text}"


@runtime_checkable
class ToolFormatter(Protocol):
    """Turns a Tool into the text a scorer indexes (the "action" side)."""

    name: str

    def __call__(self, tool: Tool) -> str: ...


@runtime_checkable
class QueryFormatter(Protocol):
    """Turns a Query into the text a scorer searches with (the "state" side)."""

    name: str

    def __call__(self, query: Query) -> str: ...


@runtime_checkable
class TextEncoder(Protocol):
    """Maps texts to a float32 matrix ``[n, dim]``. Rows need not be normalised."""

    name: str

    def encode(self, texts: Sequence[str], *, kind: str = "document") -> np.ndarray: ...


class IndexChanged(RuntimeError):
    """Rows a writer meant to keep were rewritten by another process meanwhile (``ids``)."""

    def __init__(self, ids: Sequence[str]):
        super().__init__(f"{len(ids)} rows changed under this writer, e.g. {list(ids)[:3]}")
        self.ids = list(ids)


@runtime_checkable
class VectorIndex(Protocol):
    """Unit-length tool vectors by id, searched by inner product.

    Every row carries a hash of what produced it (encoder, heads, tool text), so a scorer embeds
    only new or changed tools; ``apply`` is one atomic change (rows replaced or appended, ids
    dropped). ``expect`` (id -> hash) names the rows the writer keeps as it saw them: a shared index
    raises ``IndexChanged`` instead of applying when another writer changed one of them. ``search``
    returns at most ``k`` ids and scores per query, best first.
    """

    name: str

    def hashes(self) -> dict[str, str]: ...

    def apply(
        self,
        ids: Sequence[str],
        hashes: Sequence[str],
        vectors: np.ndarray,
        delete: Sequence[str] = (),
        expect: Mapping[str, str] | None = None,
    ) -> None: ...

    def search(self, queries: np.ndarray, k: int) -> tuple[list[list[str]], list[list[float]]]: ...


@runtime_checkable
class ToolSource(Protocol):
    """Where a customer tool set comes from: an MCP server (``kind`` "mcp") or an OpenAPI spec
    ("openapi"). Every tool it lists has ``category == name``; an ingest replaces exactly those."""

    name: str
    kind: str

    def list_tools(self) -> list[Tool]: ...


@runtime_checkable
class ChatModel(Protocol):
    """One chat completion: system and user message in, the assistant's text out. Generates
    benchmark queries (MCP-Zero) and synthetic tool text; never used for ranking."""

    name: str

    def complete(self, system: str, user: str) -> str: ...


@runtime_checkable
class Scorer(Protocol):
    """Indexes a tool corpus once, then ranks tools for queries.

    ``kind`` passed to encoders is ``"document"`` for tools and ``"query"`` for states, so
    asymmetric models (CLM's state/action heads, instruction-tuned embedders) can branch.
    """

    name: str

    def index(self, tools: Iterable[Tool]) -> None: ...

    def rank(self, queries: Sequence[Query], k: int) -> list[RankedList]: ...
