"""Which end of an over-long text does the embedding server's ``truncate_prompt_tokens`` keep?

Embeds one long text truncated to N tokens and compares it with its first N and its last N
tokens embedded whole (vLLM's /tokenize and /detokenize split it exactly). Faz 1 week 1, vLLM
0.13 + Qwen3-Embedding-8B: the first tokens (cos 0.99 vs 0.59).

uv run python scripts/truncation_side.py --tools data/mytools/tools.jsonl \
    --url http://127.0.0.1:8091 --model qwen3-emb --tokens 8192
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True, help="tools.jsonl whose texts make up the long input")
    ap.add_argument("--url", default="http://127.0.0.1:8091", help="vLLM server root (not /v1)")
    ap.add_argument("--model", default="qwen3-emb")
    ap.add_argument("--tokens", type=int, default=8192)
    a = ap.parse_args()

    def post(path: str, body: dict) -> dict:
        req = urllib.request.Request(
            a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def embed(text: str, truncate: int | None = None) -> np.ndarray:
        body: dict = {"model": a.model, "input": [text]}
        if truncate:
            body["truncate_prompt_tokens"] = truncate
        v = np.asarray(post("/v1/embeddings", body)["data"][0]["embedding"], dtype=np.float32)
        return v / np.linalg.norm(v)

    with open(a.tools, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    # head and tail from different sources (first and last category), so the two ends differ
    head = " ".join(r["documentation"] for r in rows if r["category"] == rows[0]["category"])
    tail = " ".join(r["documentation"] for r in rows if r["category"] == rows[-1]["category"])
    text = head[:40000] + " " + tail[:40000]
    tokens = post("/tokenize", {"model": a.model, "prompt": text})["tokens"]
    n = a.tokens - 16  # a margin for special tokens
    first = post("/detokenize", {"model": a.model, "tokens": tokens[:n]})["prompt"]
    last = post("/detokenize", {"model": a.model, "tokens": tokens[-n:]})["prompt"]
    truncated = embed(text, a.tokens)
    print(f"{len(tokens)} tokens, truncated to {a.tokens}")
    print(f"cos(truncated, first tokens) = {truncated @ embed(first):.4f}")
    print(f"cos(truncated, last tokens)  = {truncated @ embed(last):.4f}")


if __name__ == "__main__":
    main()
