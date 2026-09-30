# Benchmarks

Every number below is `toolrank eval` output, checked against the protocol before it is printed
(`scripts/readme_table.py`); the reports themselves are in
[`docs/results/`](https://github.com/yasinyaman/toolrank/tree/main/docs/results).

<!-- results:start -->
| Retriever | ToolRet NDCG@10 | ToolRet NDCG@10 cat-macro | LiveMCPBench Recall@5 | MCP-Zero top-1 |
| --- | ---: | ---: | ---: | ---: |
| BM25, without instruction | 29.01 | 22.24 | 31.68 | **80.44** |
| BM25, with instruction | 39.27 | 36.41 | 22.92 | 45.63 |
| Qwen3-Embedding-8B | 51.11 | 46.54 | 50.82 | 78.19 |
| Qwen3-Embedding-8B + toolrank heads v0.1 | **54.03** | 47.13 | 53.03 | 79.87 |
| Qwen3-Embedding-8B in FP8 + toolrank heads v0.1 (the Docker default) | 53.94 | 47.27 | **53.48** | 79.51 |
| NV-Embed-v1 ([ToolRet paper](https://arxiv.org/abs/2503.01763)) | — | 42.71 | — | — |
| gte-Qwen2-1.5B-instruct ([ToolRet paper](https://arxiv.org/abs/2503.01763)) | — | 45.96 | — | — |
| StackOne v2, a fine-tuned 109M BGE-base ([StackOne](https://www.stackone.com/blog/autoresearch-charged-action-search/)) | — | **54.40** | — | — |

- ToolRet: 7,961 queries over 44,453 tools, top 100 over the whole corpus. *NDCG@10* is the micro-average of the paper's released code; *cat-macro* is the paper's own aggregation (the mean of the web, code and customized categories) and the only column with published numbers. Our BM25 reproduces the paper's BM25s within 0.1 (22.24 / 36.41 against 22.32 / 36.46).
- LiveMCPBench (94 queries, 525 tools) and MCP-Zero (2,792 tools): the tool text includes the MCP server's name (`toolrank data server-names`). One LiveMCPBench query is about one point. MCP-Zero ships no queries: ours were written by Qwen3-8B, one per tool (`toolrank data pull mcp-zero`), so its column does not compare with the MCP-Zero paper. Top-1 is Precision@1.
- *With instruction*, each query carries its task's instruction (ToolRet) or a generic one (the MCP sets), as the embedding model is served; the generic instruction costs BM25 on the MCP sets. BM25 is bm25s without stemming, the paper's setting.
- The heads (29.9M parameters, `docs/heads/MODEL_CARD.md`) were trained on ToolRet's training pairs, so ToolRet is in-domain for them and the MCP sets are not. On MCP-Zero, BM25 without instruction still wins at top-1: each generated query opens with a `server:` line that usually names the server, and exact matching rewards that.
- FP8: the bf16 weights quantized as vLLM loads them (`--quantization fp8`). Every column is within a query or two of bf16, at about half the weight memory and batch-1 latency.
- Reproduce: `bash scripts/readme_results.sh` where Qwen3-Embedding-8B is served (reports in `docs/results/`), then `uv run python scripts/readme_table.py --write`.
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
pip install "toolrank[data]"
toolrank data pull toolret                 # once, from the Hugging Face Hub
toolrank data pull livemcpbench
toolrank data pull mcp-zero                # writes its queries with a chat model (see --gen-url)
toolrank data server-names data/livemcpbench && toolrank data server-names data/mcp_zero
bash scripts/readme_results.sh             # the runs behind the table, into results/
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
gives the same numbers within a query: on LiveMCPBench and MCP-Zero the differences are between
+0.07 and −0.26 points.
