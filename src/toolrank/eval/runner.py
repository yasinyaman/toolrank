"""The benchmark runner: index once, rank every query, score, aggregate, time.

Only two things vary between runs: the ``Scorer`` adapter and the dataset. Everything else
(metrics, aggregation, latency accounting, the results file) is shared, so numbers across
BM25 / dense / CLM runs are comparable by construction.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toolrank.domain import EvalReport, Query, RankedList, Tool
from toolrank.eval.metrics import DEFAULT_KS, aggregate, category_macro, evaluate_cut, evaluate_query
from toolrank.ports import Scorer


@dataclass(frozen=True)
class Summary:
    overall: dict[str, float]  # the micro-average over queries: the protocol's number
    per_task: dict[str, dict[str, float]]
    per_category: dict[str, dict[str, float]]  # empty when the queries carry no category
    category_macro: dict[str, float]


def summarize(per_query: Sequence[tuple[Query, dict[str, float]]]) -> Summary:
    """Aggregate per-query metrics the way every report does: micro-average, per task (a query
    without one is task ``all``) and, when queries carry categories, per category and the category
    macro-average. ``run_eval`` and the fine-tuning curves share it."""
    by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    for q, m in per_query:
        by_task[q.task or "all"].append(m)
    per_task = {t: aggregate(ms) for t, ms in sorted(by_task.items())}
    task_category = {q.task or "all": q.category for q, _ in per_query if q.category}
    per_category, cat_macro = (
        category_macro(per_task, {t: task_category.get(t, "") for t in per_task})
        if task_category
        else ({}, {})
    )
    return Summary(aggregate([m for _, m in per_query]), per_task, per_category, cat_macro)


def run_eval(
    scorer: Scorer,
    tools: Sequence[Tool],
    queries: Sequence[Query],
    *,
    dataset: str,
    k: int = 100,
    ks: Sequence[int] = DEFAULT_KS,
    batch: int = 64,
    config: dict[str, Any] | None = None,
    cut: Callable[[RankedList], RankedList] | None = None,
    runs: list[dict[str, Any]] | None = None,
) -> EvalReport:
    """Index ``tools`` with ``scorer``, rank ``queries`` in batches of ``batch``, return the report.

    ``k`` is the retrieval depth (ToolRet uses 100); ``ks`` are the cut-offs reported.
    Latency is measured per batch on the ranking step only (indexing is a one-off) and reported
    per query, so a 64-query batch that takes 640 ms reports 10 ms/query. With ``cut`` (an adaptive
    K, ``toolrank.cut``) every query also gets the ``@cut`` metrics of the cut list; the fixed-k
    metrics still read the full list. With ``runs`` (a list to fill) every query also leaves one
    row — id, the top-20 ids, P@1, hit@5, NDCG@10 — the raw material of ``compare --paired``.
    """
    t0 = time.perf_counter()
    scorer.index(tools)
    index_s = time.perf_counter() - t0

    per_query: list[tuple[Query, dict[str, float]]] = []
    lat: list[float] = []
    for i in range(0, len(queries), batch):
        chunk = queries[i : i + batch]
        t1 = time.perf_counter()
        ranked = scorer.rank(chunk, k)
        dt = (time.perf_counter() - t1) * 1000.0 / max(1, len(chunk))
        lat.extend([dt] * len(chunk))
        for q, r in zip(chunk, ranked, strict=True):
            m = evaluate_query(r.tool_ids, q.qrels, ks)
            if cut is not None:
                m.update(evaluate_cut(cut(r).tool_ids, q.qrels))
            per_query.append((q, m))
            if runs is not None:
                runs.append(_runs_row(q, r, m))

    summary = summarize(per_query)
    latency = {
        "index_s": round(index_s, 3),
        "per_query_p50": round(statistics.median(lat), 3) if lat else 0.0,
        "per_query_p95": round(_quantile(lat, 0.95), 3) if lat else 0.0,
        "per_query_mean": round(statistics.fmean(lat), 3) if lat else 0.0,
    }
    return EvalReport(
        scorer=scorer.name,
        dataset=dataset,
        n_queries=len(queries),
        n_tools=len(tools),
        overall=summary.overall,
        per_task=summary.per_task,
        latency_ms=latency,
        config=dict(config or {}),
        per_category=summary.per_category,
        category_macro=summary.category_macro,
    )


def _runs_row(q: Query, r: RankedList, m: dict[str, float]) -> dict[str, Any]:
    """One per-query row of a runs file: the top-20 ids and the paired-test metrics."""
    rel = {d for d, g in q.qrels.items() if g > 0}
    ndcg10 = m.get("NDCG@10")
    if ndcg10 is None:  # the run reported other cut-offs; the paired test still needs @10
        ndcg10 = evaluate_query(r.tool_ids, q.qrels, (10,))["NDCG@10"]
    return {
        "id": q.id,
        "top": list(r.tool_ids[:20]),
        "P@1": float(bool(r.tool_ids) and r.tool_ids[0] in rel),
        "hit@5": float(any(t in rel for t in r.tool_ids[:5])),
        "NDCG@10": ndcg10,
    }


def _quantile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def save_report(report: EvalReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return p


def format_table(
    report: EvalReport, metrics: Sequence[str] = ("NDCG@10", "Recall@10", "Comprehensiveness@10")
) -> str:
    """A markdown table: one row per task, then the micro-average, values in percent (``K@``
    metrics, a number of tools, as they are). With categories, also one row per category and the
    category macro-average (the paper's Average)."""
    head = "| Task | n | " + " | ".join(metrics) + " |"
    sep = "| --- | ---: | " + " | ".join("---:" for _ in metrics) + " |"
    rows = [head, sep]
    counts = _task_counts(report)
    for task, m in report.per_task.items():
        rows.append(
            f"| {task} | {counts.get(task, '')} | " + " | ".join(_fmt(x, m[x]) for x in metrics) + " |"
        )
    rows.append(
        f"| **Avg** | {report.n_queries} | "
        + " | ".join(f"**{_fmt(x, report.overall[x])}**" for x in metrics)
        + " |"
    )
    if report.category_macro:
        for cat, m in report.per_category.items():
            rows.append(f"| *{cat}* | | " + " | ".join(_fmt(x, m[x]) for x in metrics) + " |")
        rows.append(
            f"| **Cat-macro** | {len(report.per_category)} cat. | "
            + " | ".join(f"**{_fmt(x, report.category_macro[x])}**" for x in metrics)
            + " |"
        )
    return "\n".join(rows)


def _fmt(metric: str, value: float) -> str:
    return f"{value:.2f}" if metric.startswith("K@") else f"{100 * value:.2f}"


def _task_counts(report: EvalReport) -> dict[str, int]:
    return dict(report.config.get("task_counts", {}))
