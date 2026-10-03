"""Two backbone endpoints on the same texts (bf16 vs FP8, two vLLM versions, vLLM vs llama.cpp): how
close are their vectors, before and after the heads, and do they rank alike? Reads both from the
embedding cache, so each endpoint must have encoded the set first (e.g. a `toolrank eval` run each,
same formats), or ``--encode`` sends what a side lacks to its endpoint. A set without
``queries.jsonl`` (an ingest dir) compares tool vectors only; ``--heads none`` skips the heads.
On the GB10, from ~/toolrank:

    # Qwen3-Embedding-8B bf16 (8091) vs FP8 (8094), with the packaged heads
    uv run python scripts/fp8_agreement.py --a http://127.0.0.1:8091/v1=qwen3-emb \\
        --b http://127.0.0.1:8094/v1=qwen3-emb-fp8 --truncate 8192 --tool-format documentation \\
        --query-format instruct_query --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz
    # the defaults: the CLM backbone (Qwen3-8B, 8090 vs 8092) with CLM's reference heads
    uv run python scripts/fp8_agreement.py
    # on the Mac: vLLM bf16 vectors of data/w3 (cached) vs a llama.cpp GGUF server
    uv run python scripts/fp8_agreement.py --data data/w3 --db data/w3/cache/embeddings.sqlite \
        --a http://$GB10:8091/v1=qwen3-emb --b http://$LAPTOP:8102/v1=qwen3-emb-8b-q8_0 \
        --truncate 8192 --tool-format documentation --heads none --encode
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from toolrank.adapters.dense import topk_dot
from toolrank.adapters.embeddings_api import EmbeddingCache, OpenAIEmbeddings, l2_normalize
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
    p.add_argument(
        "--heads", default=None, help=".npz or .pt heads, or none (default: CLM's reference heads)"
    )
    p.add_argument("--n", type=int, default=1000, help="queries compared on their top 10")
    p.add_argument("--db", default=".cache/toolrank/embeddings.sqlite")
    p.add_argument("--encode", action="store_true", help="send the texts a side lacks to its endpoint")
    a = p.parse_args()

    def cache(spec: str) -> EmbeddingCache:
        url, _, model = spec.rpartition("=")
        return EmbeddingCache(a.db, f"{url.rstrip('/')}|{model}|trunc={a.truncate}|norm=True")

    side_a, side_b = cache(a.a), cache(a.b)

    def encode(spec: str, side: EmbeddingCache, texts: list[str]) -> None:
        if a.encode and side.missing(texts):
            url, _, model = spec.rpartition("=")
            enc = OpenAIEmbeddings(
                model, url, truncate_prompt_tokens=a.truncate, cache_dir=str(Path(a.db).parent)
            )
            enc.encode(texts)

    def both(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        texts = [t if t else " " for t in texts]  # as OpenAIEmbeddings sends them
        encode(a.a, side_a, texts)
        encode(a.b, side_b, texts)
        x, y = side_a.get_many(texts), side_b.get_many(texts)
        assert len(x) == len(y) == len(texts), f"not all cached: a {len(x)}, b {len(y)} of {len(texts)}"
        return _stack(x, len(texts)), _stack(y, len(texts))

    if a.heads is None:
        from toolrank.adapters.clm import default_checkpoint

        a.heads = default_checkpoint()
    Ta, Tb = both([tool_format(a.tool_format)(t) for t in load_tools(f"{a.data}/tools.jsonl")])
    describe("backbone, tools", Ta, Tb)
    if not Path(f"{a.data}/queries.jsonl").exists():
        return
    qs = load_queries(f"{a.data}/queries.jsonl")
    Qa, Qb = both([query_format(a.query_format)(q) for q in qs])
    describe("backbone, queries", Qa, Qb)
    if a.heads == "none":
        _rank(a, qs, Qa, Qb, Ta, Tb)
        return
    heads = load_heads(a.heads)
    Aa, Ab = heads.project_actions(Ta), heads.project_actions(Tb)
    Sa, Sb = heads.project_states(Qa), heads.project_states(Qb)
    describe("after heads, tools", Aa, Ab)
    describe("after heads, queries", Sa, Sb)
    _rank(a, qs, Sa, Sb, Aa, Ab)


def _rank(
    a: argparse.Namespace, qs: list, Sa: np.ndarray, Sb: np.ndarray, Aa: np.ndarray, Ab: np.ndarray
) -> None:
    pick = np.random.default_rng(0).choice(len(qs), min(a.n, len(qs)), replace=False)
    top_a, top_b = topk_dot(Sa[pick], Aa, 10)[0], topk_dot(Sb[pick], Ab, 10)[0]
    overlap = np.mean([len(set(x) & set(y)) / 10 for x, y in zip(top_a, top_b, strict=True)])
    same = np.mean(top_a[:, 0] == top_b[:, 0])
    print(f"top-10 overlap on {len(pick)} queries: {overlap:.3f} | same top-1: {same:.3f}")


if __name__ == "__main__":
    main()
