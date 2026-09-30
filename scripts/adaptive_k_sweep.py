"""Adaptive-K sweep: rank once, cut many ways (Faz 1 Hafta 2).

Ranks every query of a data dir once (k = 100) with the scorer the ``toolrank eval`` flags after
``--`` describe, then applies fixed top-k baselines and a grid of ``AdaptiveK`` rules, and prints
per rule: the mean number of tools handed over (K), Recall, Precision and Comprehensiveness of the
cut list, and the share of the catalogue that K is.

uv run python scripts/adaptive_k_sweep.py --margins 0.02,0.05,0.1 --max-k 10,20 -- \
    --data data/livemcpbench_server --scorer clm --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb \
    --truncate 8192 --tool-format documentation --query-format instruct_query --with-inst \
    --clm-ckpt data/heads/qwen_full_skip_neg0_e5.pt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from toolrank.build import build_scorer
from toolrank.cli import build_parser
from toolrank.cut import AdaptiveK
from toolrank.datasets.jsonl import load_queries, load_tools
from toolrank.eval.metrics import aggregate, evaluate_cut


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: adaptive_k_sweep.py [grid flags] -- <toolrank eval flags>")
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", default="1,3,5,10", help="fixed top-k baselines")
    ap.add_argument("--margins", default="0.02,0.03,0.05,0.08,0.1,0.15")
    ap.add_argument("--thresholds", default="", help="absolute cosine thresholds (alone, no margin)")
    ap.add_argument("--max-k", default="10,20")
    a = ap.parse_args(argv[:split])
    e = build_parser().parse_args(["eval", *argv[split + 1 :]])
    if e.cache_dir == "":
        e.cache_dir = None

    data = Path(e.data)
    tools = load_tools(data / "tools.jsonl")
    queries = load_queries(
        data / "queries.jsonl", [t.strip() for t in e.tasks.split(",")] if e.tasks else None
    )
    if not e.with_inst:
        queries = [replace(q, instruction="") for q in queries]
    scorer = build_scorer(e)
    scorer.index(tools)
    ranked, semantic = [], []
    for s in range(0, len(queries), e.batch):
        for r in scorer.rank(queries[s : s + e.batch], 100):
            ranked.append(r)
            semantic.append(scorer.last_semantic[r.query_id] if scorer.score_kind == "rrf" else None)

    rules: list[tuple[str, AdaptiveK]] = [
        (f"top-{k}", AdaptiveK(max_k=k, min_k=k)) for k in map(int, _floats(a.fixed))
    ]
    for max_k in map(int, _floats(a.max_k)):
        rules += [
            (f"margin {m:g}, max {max_k}", AdaptiveK(max_k=max_k, margin=m)) for m in _floats(a.margins)
        ]
        rules += [
            (f"threshold {t:g}, max {max_k}", AdaptiveK(max_k=max_k, threshold=t))
            for t in _floats(a.thresholds)
        ]

    print(f"{scorer.name} on {data.name}: {len(queries)} queries, {len(tools)} tools\n")
    print("| rule | K | Recall | Precision | Comprehensiveness | share of tools |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, rule in rules:
        rows = [
            evaluate_cut(rule.cut(r, s).tool_ids, q.qrels)
            for q, r, s in zip(queries, ranked, semantic, strict=True)
        ]
        m = aggregate(rows)
        print(
            f"| {label} | {m['K@cut']:.2f} | {100 * m['Recall@cut']:.2f} | {100 * m['Precision@cut']:.2f} "
            f"| {100 * m['Comprehensiveness@cut']:.2f} | {100 * m['K@cut'] / len(tools):.2f}% |"
        )


if __name__ == "__main__":
    main()
