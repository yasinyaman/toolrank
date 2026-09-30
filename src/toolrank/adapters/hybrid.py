"""Hybrid retrieval: a semantic scorer and BM25 fused by reciprocal rank fusion (RRF).

A tool's score is the sum over the two lists of 1 / (k_rrf + rank), rank from 1 (Cormack et al.
2009; k = 60 is also the default of Elasticsearch, LlamaIndex and LangChain), the lexical term
times ``lexical_weight`` (1 = plain RRF). Each list is taken
to ``depth`` (at least ``k``). BM25 hits scoring 0 are dropped first: when nothing matches, bm25s
pads the list with arbitrary zero-score tools. Ties go to the better semantic rank, then the
better lexical rank, then the id. The lexical arm is built to read the request without the
instruction (Phase 0: a generic instruction halves BM25 on the MCP sets).

``last_semantic`` keeps the semantic list of each query from the last ``rank`` call: RRF scores
say nothing about how many tools a request needs, so an adaptive cut (``toolrank.cut``) takes that
number from the cosines and the tools from the fused order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from toolrank.domain import Query, RankedList, Tool


class HybridScorer:
    score_kind = "rrf"

    def __init__(
        self, semantic: Any, lexical: Any, *, k_rrf: int = 60, depth: int = 100, lexical_weight: float = 1.0
    ):
        self.semantic, self.lexical, self.k_rrf, self.depth = semantic, lexical, k_rrf, depth
        self.lexical_weight = lexical_weight
        self.tool_format, self.query_format = semantic.tool_format, semantic.query_format
        w = "" if lexical_weight == 1.0 else f",w{lexical_weight:g}"
        self.name = f"hybrid[rrf{k_rrf},d{depth}{w}]/{semantic.name}+{lexical.name}"
        self.last_semantic: dict[str, RankedList] = {}

    def index(self, tools: Iterable[Tool]) -> None:
        tools = list(tools)
        self.semantic.index(tools)
        self.lexical.index(tools)

    def fuse(self, sem: RankedList, lex: RankedList, k: int) -> RankedList:
        s_rank = {t: n for n, t in enumerate(sem.tool_ids, 1)}
        hits = [t for t, s in zip(lex.tool_ids, lex.scores, strict=True) if s > 0]
        l_rank = {t: n for n, t in enumerate(hits, 1)}
        fused: dict[str, float] = {}
        for ranks, weight in ((s_rank, 1.0), (l_rank, self.lexical_weight)):
            for t, n in ranks.items():
                fused[t] = fused.get(t, 0.0) + weight / (self.k_rrf + n)
        inf = float("inf")
        order = sorted(fused, key=lambda t: (-fused[t], s_rank.get(t, inf), l_rank.get(t, inf), t))[:k]
        return RankedList(sem.query_id, order, [fused[t] for t in order])

    def rank_pairs(self, queries: Sequence[Query], k: int) -> list[tuple[RankedList, RankedList]]:
        """(fused, semantic) per query, without touching ``last_semantic``: safe to call from
        several threads at once."""
        depth = max(k, self.depth)
        sem = self.semantic.rank(queries, depth)
        lex = self.lexical.rank(queries, depth)
        return [(self.fuse(a, b, k), a) for a, b in zip(sem, lex, strict=True)]

    def rank(self, queries: Sequence[Query], k: int) -> list[RankedList]:
        pairs = self.rank_pairs(queries, k)
        self.last_semantic = {s.query_id: s for _, s in pairs}
        return [fused for fused, _ in pairs]

    def score_tools(self, query: Query, tools: Sequence[Tool]) -> list[float]:
        return self.semantic.score_tools(query, tools)
