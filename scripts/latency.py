"""Batch-1 query latency at ToolRet scale: state encode + scoring against the full tool index.

The tool index is built once from the embedding cache; then a sample of queries is ranked one
at a time through an uncached encoder (cold: HTTP encode + heads + exact top-k over all tools)
and through the cached one (warm: heads + top-k only). Run on the GB10 from ~/toolrank:

    uv run python scripts/latency.py --scorer clm --emb-url http://127.0.0.1:8090/v1 \
        --emb-model qwen3-8b --truncate 2048 --tool-format example_call
    # the packaged heads on Qwen3-Embedding-8B (bf16 on 8091, FP8 on 8094)
    uv run python scripts/latency.py --scorer clm --clm-ckpt dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz \
        --emb-url http://127.0.0.1:8094/v1 --emb-model qwen3-emb-fp8 --truncate 8192 \
        --tool-format documentation --query-format instruct_query
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from toolrank.adapters.dense import DenseScorer
from toolrank.adapters.embeddings_api import OpenAIEmbeddings
from toolrank.datasets.jsonl import load_queries, load_tools


def main() -> None:
    p = argparse.ArgumentParser(description="batch-1 latency, cold and warm, at full index size")
    p.add_argument("--data", default="data/toolret")
    p.add_argument("--scorer", choices=["clm", "dense"], default="clm")
    p.add_argument("--clm-ckpt", default=None, help="heads (.pt or .npz); default: CLM's reference heads")
    p.add_argument("--emb-url", required=True)
    p.add_argument("--emb-model", required=True)
    p.add_argument("--truncate", type=int, default=None)
    p.add_argument("--tool-format", default="example_call")
    p.add_argument("--query-format", default=None, help="default: clm for CLM, instruct_query for dense")
    p.add_argument("--n", type=int, default=100, help="queries sampled across tasks (w/ inst)")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    tools = load_tools(f"{a.data}/tools.jsonl")
    sample = random.Random(a.seed).sample(load_queries(f"{a.data}/queries.jsonl"), a.n)
    kw = dict(truncate_prompt_tokens=a.truncate)
    cached = OpenAIEmbeddings(a.emb_model, a.emb_url, cache_dir=".cache/toolrank", **kw)
    cold = OpenAIEmbeddings(a.emb_model, a.emb_url, cache_dir=None, **kw)
    if a.scorer == "clm":
        from toolrank.adapters.clm import CLMScorer, default_checkpoint
        from toolrank.adapters.heads_np import load_heads

        heads = load_heads(a.clm_ckpt or default_checkpoint(), device=a.device)
        scorer = CLMScorer(cached, heads, a.tool_format, a.query_format or "clm")
    else:
        scorer = DenseScorer(cached, a.tool_format, a.query_format or "instruct_query")

    t0 = time.perf_counter()
    scorer.index(tools)
    print(f"{scorer.name}: index of {len(tools)} tools in {time.perf_counter() - t0:.1f} s")
    scorer.rank(sample[:1], a.k)  # warm up kernels and the connection

    for mode, enc in (("cold", cold), ("warm", cached)):
        if mode == "warm":
            cached.encode([scorer.query_format(q) for q in sample], kind="query")  # fill the cache
        scorer.encoder = enc
        lat = []
        for q in sample:
            t = time.perf_counter()
            scorer.rank([q], a.k)
            lat.append((time.perf_counter() - t) * 1000.0)
        lat.sort()
        p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
        print(
            f"  {mode}: p50 {statistics.median(lat):.1f} ms | p95 {p95:.1f} ms | mean {statistics.fmean(lat):.1f} ms | n {len(lat)}"
        )


if __name__ == "__main__":
    main()
