# Benchmarks

Every number below is `toolrank eval` output, checked against the protocol before it is printed
(`scripts/readme_table.py`); the reports themselves are in
[`docs/results/`](https://github.com/yasinyaman/toolrank/tree/main/docs/results).

<!-- results:start -->
| Retriever | ToolRet NDCG@10 | ToolRet NDCG@10 cat-macro | LiveMCPBench Recall@5 | MCP-Zero top-1 |
| --- | ---: | ---: | ---: | ---: |
| BM25, without instruction | 29.01 | 22.24 | 31.68 | 80.44 |
| BM25, with instruction | 39.27 | 36.41 | 22.92 | 45.63 |
| Qwen3-Embedding-8B | 51.11 | 46.54 | 50.82 | 78.19 |
| Qwen3-Embedding-8B + toolrank heads v0.1 | 54.03 | 47.13 | 53.03 | 79.87 |
| Qwen3-Embedding-8B in FP8 + toolrank heads v0.1 | 53.94 | 47.27 | **53.48** | 79.51 |
| toolrank backbone v0.2 (Qwen3-Embedding-8B + LoRA) | 58.90 | 54.36 | 52.06 | **88.57** |
| toolrank backbone v0.2 in FP8 (the default) | **59.02** | **54.53** | 52.06 | 87.71 |
| NV-Embed-v1 ([ToolRet paper](https://arxiv.org/abs/2503.01763)) | — | 42.71 | — | — |
| gte-Qwen2-1.5B-instruct ([ToolRet paper](https://arxiv.org/abs/2503.01763)) | — | 45.96 | — | — |
| StackOne v2, a fine-tuned 109M BGE-base ([StackOne](https://www.stackone.com/blog/autoresearch-charged-action-search/)) | — | 54.40 | — | — |

- ToolRet: 7,961 queries over 44,453 tools, top 100 over the whole corpus. *NDCG@10* is the micro-average of the paper's released code; *cat-macro* is the paper's own aggregation (the mean of the web, code and customized categories) and the only column with published numbers. Our BM25 reproduces the paper's BM25s within 0.1 (22.24 / 36.41 against 22.32 / 36.46).
- LiveMCPBench (94 queries, 525 tools) and MCP-Zero (2,792 tools): the tool text includes the MCP server's name (`toolrank data server-names`). One LiveMCPBench query is about one point. MCP-Zero ships no queries: ours were written by Qwen3-8B, one per tool (`toolrank data pull mcp-zero`), so its column does not compare with the MCP-Zero paper. Top-1 is Precision@1. LiveMCPBench shows Recall@5; on NDCG@10 the v0.2 backbone leads the v0.1 heads (55.74 / 53.95), on Recall@5 it trails them (52.06 / 53.03) — the gap either way is about one query.
- *With instruction*, each query carries its task's instruction (ToolRet) or a generic one (the MCP sets), as the embedding model is served; the generic instruction costs BM25 on the MCP sets. BM25 is bm25s without stemming, the paper's setting.
- The heads (29.9M parameters, `docs/heads/MODEL_CARD.md`) were trained on ToolRet's training pairs, so ToolRet is in-domain for them and the MCP sets are not. On MCP-Zero, BM25 without instruction still wins at top-1: each generated query opens with a `server:` line that usually names the server, and exact matching rewards that.
- The toolrank backbone v0.2 (`docs/backbone/MODEL_CARD.md`) is Qwen3-Embedding-8B with a LoRA trained on 20,000 of ToolRet's training pairs, served without heads (the v0.1 heads cost it 0.2–3.9 points: ToolRet micro least, LiveMCPBench Recall@5 most). Its checkpoint was picked on MCP-Zero, so that column is its selection set; ToolRet is in-domain, LiveMCPBench is held out. Beyond these sets, on generated requests over a GitHub + Stripe catalogue it gains 12–20 NDCG@10 points on tasks that need two or three tools and ties with the base model on requests for one tool (the model card has the numbers).
- FP8: the bf16 weights quantized as vLLM loads them (`--quantization fp8`). Every column is within a point of bf16, at about half the weight memory and batch-1 latency.
- Reproduce: `bash scripts/readme_results.sh` in a clone, where the backbone is served (reports in `docs/results/`; `EMB_URL`, `EMB_MODEL`, `TAG` and `ROWS` select another endpoint and rows, `HEADS` the v0.1 heads), then `uv run python scripts/readme_table.py --write`; [benchmarks](https://yaman.dev/toolrank/benchmarks/#reproduce) has the commands.
<!-- results:end -->

## Protocol

The protocol is ToolRet's released evaluation code:

- **Ranking:** top 100 over the whole corpus, not a candidate list.
- **Cut-offs:** 5, 10 and 20.
- **Averaging:** over queries (the size-weighted "Avg" of the paper's code).
- **Comprehensiveness@k:** every gold tool is in the top k.

Reports also carry the paper's own aggregation, the mean of the three categories' task means
(*cat-macro*). It is the column to compare with published numbers.

*With instruction* means each query carries its task's instruction, as the embedding model is
served. The other setting blanks the instructions, so every scorer sees the bare request.

## Reproduce

The embedding model runs in vLLM on a GPU host; the corpus encode (44,453 ToolRet tools) takes
about 25 minutes, then everything comes from the embedding cache.

```bash
git clone https://github.com/yasinyaman/toolrank && cd toolrank   # the scripts are not in the package
uv sync --extra data
uv run toolrank data pull toolret          # once, from the Hugging Face Hub
uv run toolrank data pull livemcpbench
uv run toolrank data pull mcp-zero         # writes its queries with a chat model (see --gen-url)
uv run toolrank heads pull                 # the v0.1 heads, for the "+ heads" rows
# BM25 and the base model's rows (Qwen3-Embedding-8B served as qwen3-emb on 8091); the MCP sets'
# server-name copies are made on the way
HEADS=~/.cache/toolrank/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz UV=uv bash scripts/readme_results.sh
# the v0.2 row, with the backbone served as toolrank-emb-v0.2 on 8091
EMB_MODEL=toolrank-emb-v0.2 TAG=lora ROWS=qwen3emb UV=uv bash scripts/readme_results.sh
```

A single run:

```bash
toolrank eval --data data/toolret --scorer clm --clm-ckpt default \
  --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb --truncate 8192 \
  --tool-format documentation --query-format instruct_query --with-inst --out results/heads.json
toolrank compare results/*.json              # a markdown table of several reports
```

## Serving versions

The tables were measured with vLLM 0.13 (NVIDIA's NGC image). The official `vllm/vllm-openai:v0.30.0`
image, which the Docker setup uses, gives the same vectors: cosine 0.99993 on the same texts. It also
gives the same numbers within a query: on LiveMCPBench (Recall@5) and MCP-Zero (Precision@1) the
differences are between 0.00 and −0.26 points.
