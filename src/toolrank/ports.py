"""Ports: the interfaces the core talks to. Adapters live in ``toolrank.adapters``.

Keeping these as ``Protocol`` classes means a scorer can be swapped without touching the
runner, the CLI or the datasets - the whole point of the Phase 0 go/no-go on CLM.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from toolrank.domain import Query, RankedList, Tool


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


@runtime_checkable
class VectorIndex(Protocol):
    """Unit-length tool vectors by id, searched by inner product.

    Every row carries a hash of what produced it (encoder, heads, tool text), so a scorer embeds
    only new or changed tools; ``apply`` is one atomic change (rows replaced or appended, ids
    dropped). ``search`` returns at most ``k`` ids and scores per query, best first.
    """

    name: str

    def hashes(self) -> dict[str, str]: ...

    def apply(
        self, ids: Sequence[str], hashes: Sequence[str], vectors: np.ndarray, delete: Sequence[str] = ()
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
