"""Server -> tool routing sweep: rank once, re-score many ways (Faz 2 Hafta 5).

Does a "which server?" decision before "which tool?" help on a catalogue of many servers? Every
tool's server is its ``category``. The script scores each query against every tool once with the
scorer the ``toolrank eval`` flags after ``--`` describe, gives each server a score per query, and
prints the metrics of the tool ranking re-scored by a grid of rules:

  server score   centroid   cosine with the mean of the server's tool vectors (no new text)
                 summary    cosine with an embedded summary of the server: its name and tool names
                 top3       mean of the server's three best tool scores for the query
  rule           add W      tool + W * server
                 product    (server * tool) * max(server, tool), MCP-Zero's
                 hard M     only tools of the M best servers

With ``--gate``, a second table: can the best score tell that the catalogue has no tool for a
request? Every query is scored twice, against the whole catalogue (it has an answer) and against the
catalogue without the servers of its gold tools (it has none), and each threshold shows the share of
answerable requests it would turn away and the share of unanswerable ones it would catch.

uv run python scripts/routing_sweep.py --weights 0.1,0.2,0.3 --hard 1,3,5 --gate -- \
    --data data/mcp_zero_server --scorer clm --clm-ckpt dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz \
    --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 --tool-format documentation \
    --query-format instruct_query --with-inst
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from toolrank.adapters.embeddings_api import l2_normalize
from toolrank.build import build_scorer
from toolrank.cli import build_parser
from toolrank.datasets.jsonl import load_queries, load_tools
from toolrank.domain import Tool
from toolrank.eval.metrics import aggregate, evaluate_query

KS = (1, 5, 10)
SUMMARY_CHARS = 4000


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def server_summary(server: str, tools: list[Tool]) -> str:
    """What a server offers, in the shape of the indexed tool text: its name and its tools' names."""
    text = json.dumps({"server": server, "tools": [t.name for t in tools]}, ensure_ascii=False)
    return text[:SUMMARY_CHARS]


def server_scores(
    kind: str, s: np.ndarray, t: np.ndarray, q: np.ndarray, members: list[np.ndarray]
) -> np.ndarray:
    """[queries, servers] for ``centroid`` and ``top3`` (``summary`` needs the encoder: see main)."""
    if kind == "centroid":
        c = l2_normalize(np.stack([t[m].mean(axis=0) for m in members]))
        return q @ c.T
    if kind == "top3":
        return np.stack([np.sort(s[:, m], axis=1)[:, -3:].mean(axis=1) for m in members], axis=1)
    raise ValueError(kind)


def rescored(
    rule: str, value: float, s: np.ndarray, per_tool: np.ndarray, srv: np.ndarray, of: np.ndarray
) -> np.ndarray:
    """The tool scores under one rule; ``per_tool`` = each tool's server score, ``srv`` = [queries,
    servers], ``of`` = each tool's server index."""
    if rule == "add":
        return s + value * per_tool
    if rule == "product":
        return (per_tool * s) * np.maximum(per_tool, s)
    if rule == "hard":
        m = min(int(value), srv.shape[1])
        best = np.argpartition(-srv, m - 1, axis=1)[:, :m]
        allowed = np.zeros(srv.shape, dtype=bool)
        np.put_along_axis(allowed, best, True, axis=1)
        return np.where(allowed[:, of], s, -np.inf)
    raise ValueError(rule)


def metrics(scores: np.ndarray, ids: list[str], queries: list) -> dict[str, float]:
    k = min(max(KS), scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]  # a full sort of a 44k-tool row is wasted
    top = np.take_along_axis(part, np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1), axis=1)
    return aggregate(
        [evaluate_query([ids[j] for j in row], q.qrels, KS) for row, q in zip(top, queries, strict=True)]
    )


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: routing_sweep.py [grid flags] -- <toolrank eval flags>")
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", default="centroid,summary,top3", help="server scores to try")
    ap.add_argument("--weights", default="0.05,0.1,0.2,0.3,0.5", help="W of tool + W * server")
    ap.add_argument("--hard", default="1,3,5,10", help="M of 'only the M best servers'")
    ap.add_argument("--gate", action="store_true", help="also print the no-tool gate table")
    ap.add_argument("--thresholds", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    a = ap.parse_args(argv[:cut])
    e = build_parser().parse_args(["eval", *argv[cut + 1 :]])
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
    t = scorer._vectors([scorer.tool_format(x) for x in tools], "document", scorer.project_tools)
    q = scorer._vectors([scorer.query_format(x) for x in queries], "query", scorer.project_queries)
    s = q @ t.T
    names = sorted({x.category for x in tools})
    of = np.array([names.index(x.category) for x in tools])
    members = [np.flatnonzero(of == n) for n in range(len(names))]
    ids = [x.id for x in tools]

    print(f"{scorer.name} on {data.name}: {len(queries)} queries, {len(tools)} tools, {len(names)} servers\n")
    print(
        "| server score | rule | "
        + " | ".join(f"P@{k}" if k == 1 else f"Recall@{k}" for k in KS)
        + " | NDCG@10 | right server first |"
    )
    print("| --- | --- | " + "---: | " * (len(KS) + 2))

    def row(kind: str, rule: str, m: dict[str, float], first: float | None) -> None:
        cells = [100 * m["Precision@1"], *(100 * m[f"Recall@{k}"] for k in KS[1:]), 100 * m["NDCG@10"]]
        print(
            f"| {kind} | {rule} | "
            + " | ".join(f"{c:.2f}" for c in cells)
            + f" | {'' if first is None else f'{100 * first:.2f}'} |"
        )

    row("—", "tool score alone", metrics(s, ids, queries), None)
    gold = [{names.index(x.category) for x in tools if q_.qrels.get(x.id, 0) > 0} for q_ in queries]
    for kind in [k.strip() for k in a.servers.split(",") if k.strip()]:
        if kind == "summary":
            texts = [server_summary(n, [tools[j] for j in members[i]]) for i, n in enumerate(names)]
            srv = q @ scorer._vectors(texts, "document", scorer.project_tools).T
        else:
            srv = server_scores(kind, s, t, q, members)
        first = float(np.mean([int(np.argmax(r)) in g for r, g in zip(srv, gold, strict=True)]))
        per_tool = srv[:, of]
        for w in _floats(a.weights):
            row(kind, f"add {w:g}", metrics(rescored("add", w, s, per_tool, srv, of), ids, queries), first)
        row(kind, "product", metrics(rescored("product", 0, s, per_tool, srv, of), ids, queries), first)
        for m in _floats(a.hard):
            row(
                kind,
                f"hard {int(m)}",
                metrics(rescored("hard", m, s, per_tool, srv, of), ids, queries),
                first,
            )
    if a.gate:
        gate_table(s, of, gold, _floats(a.thresholds))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a positive scores above a negative), ties counted half (rank-sum)."""
    both = np.concatenate([pos, neg])
    order = both.argsort(kind="stable")
    ranks = np.empty(len(both))
    ranks[order] = np.arange(1, len(both) + 1)
    for v in np.unique(both[np.diff(np.sort(both), prepend=np.nan) == 0]):  # average the ranks of ties
        ranks[both == v] = ranks[both == v].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gate_table(s: np.ndarray, of: np.ndarray, gold: list[set[int]], thresholds: list[float]) -> None:
    """The best cosine with the gold tools' servers in the catalogue (answerable) and without them
    (unanswerable), and what each threshold on it would do."""
    have = s.max(axis=1)
    gone = np.array(
        [np.where(np.isin(of, list(g)), -np.inf, row).max() for row, g in zip(s, gold, strict=True)]
    )
    print(
        f"\nno-tool gate on the best cosine: answerable mean {have.mean():.3f}, unanswerable (gold servers "
        f"removed) mean {gone.mean():.3f}, AUROC {auroc(have, gone):.3f}\n"
    )
    print("| threshold | answerable turned away | unanswerable caught |\n| ---: | ---: | ---: |")
    for th in thresholds:
        print(f"| {th:g} | {100 * float((have < th).mean()):.2f} | {100 * float((gone < th).mean()):.2f} |")


if __name__ == "__main__":
    main()
