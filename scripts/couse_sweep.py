"""Co-use sweep: do the tools agents call together complete a result list? (Faz 2 Hafta 5)

A usage log says which tools were called after the same search. From it: for a tool A, the tools B
that were called along with it in at least ``--min-count`` searches and in at least ``--min-p`` of
the searches where A was called. The script ranks the queries of a data dir with the scorer the
``toolrank eval`` flags after ``--`` describe, cuts each list with the adaptive-K rule of those
flags, and then appends up to ``--extra`` such partners of the tools shown; it prints, per setting,
the mean number of tools handed over and Recall, Precision and Comprehensiveness of the list, next
to plain adaptive K at larger ``max`` values (the same budget spent on the ranking itself).

The log is a simulated one (scripts/learn_sim.py): the data dir must be queries that were never
served, e.g. its ``heldout/``.

uv run python scripts/couse_sweep.py --log data/sim/toolret/runs/n0/usage --min-count 2,3 --min-p 0.3,0.5 -- \
    --data data/sim/toolret/heldout --scorer clm --clm-ckpt dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz \
    --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 --tool-format documentation \
    --query-format instruct_query --with-inst --cut-margin 0.2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from toolrank.build import build_scorer
from toolrank.cli import build_parser
from toolrank.couse import co_use, expand
from toolrank.cut import AdaptiveK, rule_from_flags
from toolrank.datasets.jsonl import load_queries, load_tools
from toolrank.eval.metrics import aggregate, evaluate_cut
from toolrank.learn import read_events


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: couse_sweep.py --log DIR [grid flags] -- <toolrank eval flags>")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="a usage log directory (usage-*.jsonl)")
    ap.add_argument("--min-count", default="2,3,5", help="searches a pair must have been called together in")
    ap.add_argument("--min-p", default="0.3,0.5,0.7", help="share of A's searches in which B was called too")
    ap.add_argument("--extra", default="1,2,3", help="partners appended per result at most")
    ap.add_argument("--max-k", default="12,15,20", help="plain adaptive K at these maxima, for comparison")
    a = ap.parse_args(argv[:cut])
    e = build_parser().parse_args(["eval", *argv[cut + 1 :]])
    if e.cache_dir == "":
        e.cache_dir = None
    rule = rule_from_flags(e.cut_margin, e.cut_threshold, e.cut_max, e.cut_min, default_margin=True)
    assert rule is not None

    data = Path(e.data)
    tools = load_tools(data / "tools.jsonl")
    queries = load_queries(data / "queries.jsonl")
    if not e.with_inst:
        queries = [replace(q, instruction="") for q in queries]
    events = read_events(a.log)
    scorer = build_scorer(e)
    scorer.index(tools)
    ranked = [r for s in range(0, len(queries), e.batch) for r in scorer.rank(queries[s : s + e.batch], 100)]
    multi = [n for n, q in enumerate(queries) if sum(g > 0 for g in q.qrels.values()) > 1]

    print(f"{scorer.name} on {data.name}: {len(queries)} queries ({len(multi)} with several gold tools)\n")
    print(
        "| list | K | Recall | Precision | Comprehensiveness | Comprehensiveness, multi-tool | lists changed |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    def row(label: str, lists: list[list[str]], changed: int | None = None) -> None:
        rows = [evaluate_cut(x, q.qrels) for x, q in zip(lists, queries, strict=True)]
        m, mm = aggregate(rows), aggregate([rows[n] for n in multi])
        print(
            f"| {label} | {m['K@cut']:.2f} | {100 * m['Recall@cut']:.2f} | {100 * m['Precision@cut']:.2f} | "
            f"{100 * m['Comprehensiveness@cut']:.2f} | {100 * mm['Comprehensiveness@cut']:.2f} | "
            f"{'' if changed is None else changed} |"
        )

    base = [rule.cut(r).tool_ids for r in ranked]
    row(f"adaptive K ({rule.describe()})", base)
    for max_k in map(int, _floats(a.max_k)):
        wider = AdaptiveK(max_k=max_k, min_k=rule.min_k, margin=rule.margin, threshold=rule.threshold)
        row(f"adaptive K, max {max_k}", [wider.cut(r).tool_ids for r in ranked])
    for c in map(int, _floats(a.min_count)):
        for p in _floats(a.min_p):
            table = co_use(events, min_count=c, min_p=p)
            for extra in map(int, _floats(a.extra)):
                lists = [expand(x, table, extra) for x in base]
                changed = sum(len(x) != len(y) for x, y in zip(lists, base, strict=True))
                row(f"+ co-use (count ≥ {c}, p ≥ {p:g}, ≤ {extra} more; {len(table)} tools)", lists, changed)


if __name__ == "__main__":
    main()
