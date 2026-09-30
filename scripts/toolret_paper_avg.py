"""Reproduce the ToolRet paper's BM25s row (Tables 4 and 5) with the paper's own settings.

The paper's "Average" is the category macro-average (``EvalReport.category_macro``, the
``cat-macro`` column of ``toolrank compare``), and its BM25s is unstemmed and indexes
``str(doc)`` rather than the ``documentation`` JSON string. The CLI covers the aggregation and
``--no-stem``; this script adds the ``str(doc)`` tool text and prints the per-category deltas
against the paper, so the harness can be checked without changing the protocol.

    uv run python scripts/toolret_paper_avg.py --data data/toolret
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from toolrank.adapters.bm25 import BM25Scorer
from toolrank.datasets.jsonl import load_queries, load_tools
from toolrank.datasets.toolret import TASK_TO_CATEGORY, TOOLRET_CATEGORIES
from toolrank.eval.runner import run_eval
from toolrank.formats import NamedFormatter

# BM25s NDCG@10 from the paper: per category, and the printed "Average" (w/o inst, w/ inst).
PAPER_BM25S = {
    False: ({"web": 18.98, "code": 21.20, "customized": 26.76}, 22.32),
    True: ({"web": 26.33, "code": 41.90, "customized": 41.16}, 36.46),
}


def main() -> None:
    p = argparse.ArgumentParser(description="ToolRet BM25 with the paper's settings")
    p.add_argument("--data", default="data/toolret")
    p.add_argument("--stem", action="store_true", help="Snowball stemming (the paper's BM25s has none)")
    p.add_argument("--tool-text", choices=["doc_repr", "documentation"], default="doc_repr")
    a = p.parse_args()

    tools = load_tools(f"{a.data}/tools.jsonl")
    # older pulls have no per-query category; the task map gives the same grouping
    queries = [
        replace(q, category=q.category or TASK_TO_CATEGORY[q.task])
        for q in load_queries(f"{a.data}/queries.jsonl")
    ]
    fmt = NamedFormatter("doc_repr", lambda t: str(t.doc)) if a.tool_text == "doc_repr" else "documentation"
    print(f"BM25, tool text {a.tool_text}, stem={a.stem}: NDCG@10 (delta vs paper)")
    for with_inst in (False, True):
        qs = queries if with_inst else [replace(q, instruction="") for q in queries]
        scorer = BM25Scorer(fmt, "concat" if with_inst else "plain", stem=a.stem)
        report = run_eval(scorer, tools, qs, dataset="toolret", k=100, ks=(10,))
        paper_cats, paper_avg = PAPER_BM25S[with_inst]
        avg = 100 * report.category_macro["NDCG@10"]
        cells = "  ".join(
            f"{c} {100 * report.per_category[c]['NDCG@10']:.2f} "
            f"({100 * report.per_category[c]['NDCG@10'] - paper_cats[c]:+.2f})"
            for c in TOOLRET_CATEGORIES
        )
        print(
            f"{'w/ ' if with_inst else 'w/o'} inst | micro {100 * report.overall['NDCG@10']:.2f} | "
            f"cat-macro {avg:.2f} ({avg - paper_avg:+.2f}) | {cells}"
        )


if __name__ == "__main__":
    main()
