# toolrank-emb-8b v0.2: Qwen3-Embedding-8B trained further on tool retrieval

[Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) with a LoRA trained on
request–tool pairs and merged into the weights. It is toolrank's default backbone: a request and
each tool become vectors, and the tools closest to the request come first. Use it as Qwen3-Embedding-8B
is used (last-token pooling, 4096 dimensions, L2-normalised), with the instruction below.

Code, documentation and benchmarks: [github.com/yasinyaman/toolrank](https://github.com/yasinyaman/toolrank),
[yaman.dev/toolrank](https://yaman.dev/toolrank/).

| | |
| --- | --- |
| Weights | `yasinyaman/toolrank-emb-8b` at revision `v0.2`: `model.safetensors`, bfloat16, 16.4 GB |
| sha256 | `53789cfff18f631fa7e6494bcf8d2db7b5c18a25d2a239443b1614a5eacb7f42` |
| Base | `Qwen/Qwen3-Embedding-8B` (Apache-2.0) |
| Change | LoRA rank 16, alpha 32, dropout 0.05 on the base model, merged; nothing else changes (tokenizer, pooling and config are the base model's) |
| License | Apache-2.0; see *Training data* |

## Serving

```bash
vllm serve yasinyaman/toolrank-emb-8b --revision v0.2 --served-model-name toolrank-emb-v0.2 \
  --runner pooling --max-model-len 8192 --port 8091      # add --quantization fp8 (as toolrank-emb-v0.2-fp8)
```

- Requests: `Instruct: Given an agent's request for a tool, retrieve the MCP tool that fulfills it.\nQuery: <request>`.
- Tools: toolrank's `documentation` text; for MCP and OpenAPI tools `{"server", "name", "description",
  "inputSchema"}` as JSON. 8192 tokens per text.
- No heads: toolrank's v0.1 heads, trained on the base model, cost 1–2 points here, and heads
  trained on top of this model stay at the identity. toolrank applies none on it.
- Serve it under a name with the version in it: toolrank's embedding cache is keyed by the served
  name, and a new version must not meet the old one's vectors.

## Training

- Data: 20,000 request–tool pairs drawn from ToolRet-Training-20w (`mangopy/ToolRet-Training-20w`),
  seed 0. Pairs whose request equals a request of ToolRet, LiveMCPBench or MCP-Zero were dropped (776,
  all ToolRet).
- Objective: InfoNCE with in-batch negatives, temperature 0.05; requests cut to 256 tokens, tools to
  768.
- Optimisation: one epoch, 625 steps, micro-batch 16 with 2 accumulation steps, lr 1e-4 with 50 warm-up
  steps; 8.6 hours on one GB10.
- Selection: the checkpoint (every 300 steps and the last) was picked on **MCP-Zero**
  (`mcp_zero_server`): its numbers below are on the selection set, not held out. ToolRet and
  LiveMCPBench played no part in selection.
- Before training, the training code's vectors of the base model were checked against the served
  base model (mean cosine 0.9999 on 8 tool texts), so the LoRA learned on the vectors vLLM serves.

## Training data license

ToolRet-Training-20w states no license: its dataset card has no license field and no license text.
The ToolRet code repository (`mangopy/tool-retrieval-benchmark`) is Apache-2.0, but that does not
cover the data, and the data itself is drawn from earlier benchmarks with their own terms. The
toolrank maintainers publish these weights accepting that risk. If you need clean data provenance,
use the base Qwen3-Embedding-8B.

## Results

w/ inst, each set under its own protocol ([benchmarks](https://yaman.dev/toolrank/benchmarks/)):

| | ToolRet NDCG@10 (micro / cat-macro) | LiveMCPBench NDCG@10 / Recall@5 | MCP-Zero top-1 (selection set) |
| --- | ---: | ---: | ---: |
| Qwen3-Embedding-8B | 51.11 / 46.54 | — / 50.82 | 78.19 |
| Qwen3-Embedding-8B + toolrank heads v0.1 | 54.03 / 47.13 | 53.95 / 53.03 | 79.87 |
| **this model** | **58.90 / 54.36** | **55.74** / 52.06 | 88.57 |
| this model in FP8 (`--quantization fp8`) | 59.02 / 54.53 | 55.34 / 52.06 | 87.71 |
| this model + toolrank heads v0.1 | — | 53.52 / 48.18 | 87.46 |

- ToolRet: 44,453 tools, 7,961 queries. LiveMCPBench: 525 tools, 94 tasks. MCP-Zero: 2,792 tools, one
  LLM-written request per tool. On both MCP sets the server name is in the tool text.
- With a second stage over its top 20 (Qwen3-Reranker-8B reading each tool's documentation):
  ToolRet 59.36 / 54.35, LiveMCPBench 61.24, MCP-Zero top-1 91.94 (`toolrank serve --rerank cross`).

## Where it helps, and where it does not

The gain over the base model is large where the requests look like its training data, and absent
where they do not:

| Set | What it is to this model | Base model → this model (NDCG@10) |
| --- | --- | --- |
| ToolRet | its training distribution (other pairs of the same tasks) | 51.11 → 58.90 |
| MCP-Zero, top-1 | its selection set; requests are two lines, `server: …` and `tool: …` | 78.19 → 88.57 |
| LiveMCPBench | held out; multi-step tasks in plain language | 53.74 → 55.74 (not significant: 94 tasks) |
| GitHub + Stripe APIs, 1,000 generated requests | held out; one tool per request, plain language | 92.99 → 92.49 |
| the same catalogue, 598 requests that state a problem, not the operation | held out | 89.67 → 87.71 |

On the last set the base model finds the tool in its top five for 19 requests where this model does
not, against 4 the other way (sign test p = 0.003); on the others the two do not differ beyond
noise. The two generated sets come from `toolrank data gen-queries` (requests written by Qwen3-8B,
1,862 tools) and are easy for both models (Recall@5 95–99%). So: expect the published gains on
ToolRet-like catalogues and on short, structured requests; on plain-language requests over your own
MCP servers and APIs expect about what the base model gives, and measure on your own catalogue
(`toolrank data gen-queries`, then `toolrank eval` with each backbone). With a second stage over
the top 20 the two backbones are indistinguishable there (top-1 89.4 and 89.5).
