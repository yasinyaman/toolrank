"""Do two head checkpoints rank alike? (Faz 1 Hafta 2: the packaged ``.npz`` vs the torch ``.pt``.)

Builds the scorer the ``toolrank eval`` flags after ``--`` describe twice, once per checkpoint,
and prints the cosine between the two projections of every tool and query (min, mean) and, per
query, the top-10 overlap and whether the top-1 agrees.

uv run python scripts/heads_parity.py --a data/heads/x.pt --b dist/heads/x.npz -- --data data/toolret \
    --scorer clm --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 \
    --tool-format documentation --query-format instruct_query --with-inst
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from toolrank.build import build_scorer
from toolrank.cli import build_parser
from toolrank.datasets.jsonl import load_queries, load_tools


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: heads_parity.py --a CKPT --b CKPT -- <toolrank eval flags>")
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    own = ap.parse_args(argv[:split])
    e = build_parser().parse_args(["eval", *argv[split + 1 :]])
    data = Path(e.data)
    tools = load_tools(data / "tools.jsonl")
    queries = load_queries(data / "queries.jsonl")
    if not e.with_inst:
        queries = [replace(q, instruction="") for q in queries]

    scorers = []
    for ck in (own.a, own.b):
        a = copy.copy(e)
        a.clm_ckpt = ck
        s = build_scorer(a)
        s.index(tools)
        scorers.append(s)
    sa, sb = scorers
    tool_cos = np.sum(sa.vindex._m * sb.vindex._m, axis=1)  # same tools, same order: fresh indexes
    texts = [sa.query_format(q) for q in queries]
    qa = sa._vectors(texts, "query", sa.project_queries)
    qb = sb._vectors(texts, "query", sb.project_queries)
    query_cos = np.sum(qa * qb, axis=1)
    ra, rb = sa.rank(queries, 10), sb.rank(queries, 10)
    overlap = np.mean(
        [
            len(set(x.tool_ids) & set(y.tool_ids)) / max(len(x.tool_ids), 1)
            for x, y in zip(ra, rb, strict=True)
        ]
    )
    top1 = np.mean([x.tool_ids[:1] == y.tool_ids[:1] for x, y in zip(ra, rb, strict=True)])
    print(
        f"{data.name}: {len(tools)} tools, {len(queries)} queries; {Path(own.a).name} vs {Path(own.b).name}"
    )
    print(f"  tool projection cosine  min {tool_cos.min():.6f}  mean {tool_cos.mean():.6f}")
    print(f"  query projection cosine min {query_cos.min():.6f}  mean {query_cos.mean():.6f}")
    print(f"  top-10 overlap {100 * overlap:.2f}%  top-1 agreement {100 * top1:.2f}%")


if __name__ == "__main__":
    main()
