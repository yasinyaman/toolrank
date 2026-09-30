"""Two backbone endpoints on the same texts (bf16 vs FP8, or two vLLM versions): how close are their
vectors, before and after the heads, and do they rank alike? Reads both from the embedding cache,
so each endpoint must have encoded the set first (e.g. a `toolrank eval` run each, same formats).
On the GB10, from ~/toolrank:

    # Qwen3-Embedding-8B bf16 (8091) vs FP8 (8094), with the packaged heads
    uv run python scripts/fp8_agreement.py --a http://127.0.0.1:8091/v1=qwen3-emb \\
        --b http://127.0.0.1:8094/v1=qwen3-emb-fp8 --truncate 8192 --tool-format documentation \\
        --query-format instruct_query --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz
    # the defaults: the CLM backbone (Qwen3-8B, 8090 vs 8092) with CLM's reference heads
    uv run python scripts/fp8_agreement.py
"""

from __future__ import annotations

import argparse

import numpy as np

from toolrank.adapters.dense import topk_dot
from toolrank.adapters.embeddings_api import EmbeddingCache, l2_normalize
from toolrank.adapters.heads_np import load_heads
from toolrank.datasets.jsonl import load_queries, load_tools
from toolrank.formats import query_format, tool_format


def _stack(found: dict[int, np.ndarray], n: int) -> np.ndarray:
    return l2_normalize(np.stack([found[i] for i in range(n)]).astype(np.float32))


def describe(name: str, x: np.ndarray, y: np.ndarray) -> None:
    cos = np.sum(x * y, axis=1)
    print(
        f"{name:22s} cosine a~b: mean {cos.mean():.5f} | p1 {np.percentile(cos, 1):.5f} | min {cos.min():.5f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", default="data/toolret")
    p.add_argument("--a", default="http://127.0.0.1:8090/v1=qwen3-8b", help="URL=MODEL, as served")
    p.add_argument("--b", default="http://127.0.0.1:8092/v1=qwen3-8b-fp8")
    p.add_argument("--truncate", type=int, default=2048)
    p.add_argument("--tool-format", default="example_call")
    p.add_argument("--query-format", default="clm")
    p.add_argument("--heads", default=None, help=".npz or .pt heads (default: CLM's reference heads)")
    p.add_argument("--n", type=int, default=1000, help="queries compared on their top 10")
    p.add_argument("--db", default=".cache/toolrank/embeddings.sqlite")
    a = p.parse_args()

    def cache(spec: str) -> EmbeddingCache:
        url, _, model = spec.rpartition("=")
        return EmbeddingCache(a.db, f"{url.rstrip('/')}|{model}|trunc={a.truncate}|norm=True")

    side_a, side_b = cache(a.a), cache(a.b)

    def both(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        texts = [t if t else " " for t in texts]  # as OpenAIEmbeddings sends them
        x, y = side_a.get_many(texts), side_b.get_many(texts)
        assert len(x) == len(y) == len(texts), f"not all cached: a {len(x)}, b {len(y)} of {len(texts)}"
        return _stack(x, len(texts)), _stack(y, len(texts))

    if a.heads is None:
        from toolrank.adapters.clm import default_checkpoint

        a.heads = default_checkpoint()
    heads = load_heads(a.heads)
    Ta, Tb = both([tool_format(a.tool_format)(t) for t in load_tools(f"{a.data}/tools.jsonl")])
    qs = load_queries(f"{a.data}/queries.jsonl")
    Qa, Qb = both([query_format(a.query_format)(q) for q in qs])
    describe("backbone, tools", Ta, Tb)
    describe("backbone, queries", Qa, Qb)
    Aa, Ab = heads.project_actions(Ta), heads.project_actions(Tb)
    Sa, Sb = heads.project_states(Qa), heads.project_states(Qb)
    describe("after heads, tools", Aa, Ab)
    describe("after heads, queries", Sa, Sb)

    pick = np.random.default_rng(0).choice(len(qs), min(a.n, len(qs)), replace=False)
    top_a, top_b = topk_dot(Sa[pick], Aa, 10)[0], topk_dot(Sb[pick], Ab, 10)[0]
    overlap = np.mean([len(set(x) & set(y)) / 10 for x, y in zip(top_a, top_b, strict=True)])
    same = np.mean(top_a[:, 0] == top_b[:, 0])
    print(f"top-10 overlap on {len(pick)} queries: {overlap:.3f} | same top-1: {same:.3f}")


if __name__ == "__main__":
    main()
