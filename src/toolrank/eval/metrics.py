"""trec_eval-compatible metrics, in plain Python, so results line up with ToolRet's
``pytrec_eval`` protocol (``ndcg_cut``, ``recall``, ``P``, ``map_cut``) plus ToolRet's
"Comprehensiveness@k" (recall@k == 1, i.e. every gold tool retrieved; also called completeness).

Conventions copied from trec_eval:

* graded relevance with LINEAR gain: ``DCG@k = sum(rel_i / log2(i + 1))`` for i = 1..k;
* the ideal DCG uses all relevant docs sorted by relevance (cut at k);
* ``recall@k`` and ``map@k`` divide by the TOTAL number of relevant docs (rel > 0);
* ``P@k`` divides by k even when fewer than k docs were returned.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

DEFAULT_KS: tuple[int, ...] = (5, 10, 20)


def evaluate_query(
    ranked: Sequence[str], qrels: dict[str, int], ks: Iterable[int] = DEFAULT_KS
) -> dict[str, float]:
    """Metrics for one query. ``ranked`` = tool ids best first; ``qrels`` = id -> relevance."""
    rel = {d: g for d, g in qrels.items() if g > 0}
    n_rel = len(rel)
    gains = [rel.get(d, 0) for d in ranked]
    ideal = sorted(rel.values(), reverse=True)
    out: dict[str, float] = {}
    for k in ks:
        top = gains[:k]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(top))
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
        hits = sum(1 for g in top if g > 0)
        ap = 0.0
        if n_rel:
            seen = 0
            for i, g in enumerate(top):
                if g > 0:
                    seen += 1
                    ap += seen / (i + 1)
            ap /= n_rel
        recall = hits / n_rel if n_rel else 0.0
        out[f"NDCG@{k}"] = dcg / idcg if idcg > 0 else 0.0
        out[f"MAP@{k}"] = ap
        out[f"Recall@{k}"] = recall
        out[f"Precision@{k}"] = hits / k
        out[f"Comprehensiveness@{k}"] = 1.0 if n_rel and hits == n_rel else 0.0
    return out


def evaluate_cut(returned: Sequence[str], qrels: dict[str, int]) -> dict[str, float]:
    """Metrics of a variable-length result (an adaptive cut): how many tools were handed over
    (``K@cut``), recall and completeness over all relevant tools, precision over what was returned."""
    rel = {d for d, g in qrels.items() if g > 0}
    hits = sum(1 for d in returned if d in rel)
    return {
        "K@cut": float(len(returned)),
        "Recall@cut": hits / len(rel) if rel else 0.0,
        "Precision@cut": hits / len(returned) if returned else 0.0,
        "Comprehensiveness@cut": 1.0 if rel and hits == len(rel) else 0.0,
    }


def aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric over queries (micro-average; ToolRet's size-weighted 'Avg')."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def category_macro(
    per_task: dict[str, dict[str, float]], task_category: dict[str, str]
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """The ToolRet paper's "Average": each category is the plain mean over its tasks, the result
    the plain mean over categories, so a 1,000-query task weighs as much as a 20-query one.

    Returns ``(per_category, overall)``. Not part of the trec_eval protocol; reported next to it.
    """
    groups: dict[str, list[dict[str, float]]] = {}
    for task, metrics in per_task.items():
        groups.setdefault(task_category[task], []).append(metrics)
    per_category = {c: aggregate(rows) for c, rows in sorted(groups.items())}
    return per_category, aggregate(list(per_category.values()))
