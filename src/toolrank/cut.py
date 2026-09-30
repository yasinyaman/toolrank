"""Adaptive K: how many tools to hand an agent for one request.

A fixed top-k either misses tools a multi-tool request needs or spends context on tools it does
not. ``AdaptiveK`` keeps, from a cosine-scored list, the leading tools within ``margin`` of the best
score (and at or above ``threshold``), at least ``min_k`` and at most ``max_k``. On a hybrid list
the count comes from the semantic cosines (RRF scores carry no absolute meaning) and the tools from
the fused order; BM25 scores have no scale to cut on, so a BM25 list is refused.

Defaults (``DEFAULT_MARGIN``, ``DEFAULT_MAX_K``): picked on ToolRet as the margin with the largest
completeness gain over fixed top-k at the same mean K (+1.5 at K 8.3, full-data heads), then checked
on the held-out MCP sets: LiveMCPBench recall +2.7 at K 7.5, MCP-Zero +1.1 at K 5.9, precision far
above fixed top-k on all three (``docs/reports/faz1-week2.md``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from toolrank.domain import RankedList

DEFAULT_MARGIN = 0.2
DEFAULT_MAX_K = 10


@dataclass(frozen=True)
class AdaptiveK:
    max_k: int = 10
    min_k: int = 1
    margin: float | None = None
    threshold: float | None = None

    def count(self, scores: Sequence[float]) -> int:
        """How many of these best-first scores to keep."""
        if not scores:
            return 0
        top, n = scores[0], 0
        for s in scores[: self.max_k]:
            if (self.margin is not None and s < top - self.margin) or (
                self.threshold is not None and s < self.threshold
            ):
                break
            n += 1
        return max(n, min(self.min_k, self.max_k, len(scores)))

    def cut(self, ranked: RankedList, semantic: RankedList | None = None) -> RankedList:
        n = self.count((semantic or ranked).scores)
        return RankedList(ranked.query_id, ranked.tool_ids[:n], ranked.scores[:n])

    def describe(self) -> str:
        bits = [f"max {self.max_k}", f"min {self.min_k}"]
        if self.margin is not None:
            bits.insert(0, f"margin {self.margin:g}")
        if self.threshold is not None:
            bits.insert(0, f"threshold {self.threshold:g}")
        return ", ".join(bits)


def rule_from_flags(
    margin: float | None,
    threshold: float | None,
    max_k: int = DEFAULT_MAX_K,
    min_k: int = 1,
    *,
    default_margin: bool = False,
) -> AdaptiveK | None:
    """The ``--cut-*`` flags as a rule: None when neither margin nor threshold is set, unless
    ``default_margin`` (``toolrank search`` / ``serve``) fills in ``DEFAULT_MARGIN``."""
    if margin is None and threshold is None:
        if not default_margin:
            return None
        margin = DEFAULT_MARGIN
    return AdaptiveK(max_k=max_k, min_k=min_k, margin=margin, threshold=threshold)


def cutter(scorer: Any, rule: AdaptiveK) -> Callable[[RankedList], RankedList]:
    """``rule`` as a function of the scorer's ranked lists; call it right after each ``rank`` (a
    hybrid scorer's semantic lists are those of its last call)."""
    kind = getattr(scorer, "score_kind", "cosine")
    if kind == "cosine":
        return rule.cut
    if kind == "rrf":
        return lambda r: rule.cut(r, semantic=scorer.last_semantic[r.query_id])
    raise ValueError(f"adaptive K needs cosine scores; {scorer.name} scores are {kind}")
