# toolrank

[![PyPI](https://img.shields.io/pypi/v/toolrank.svg)](https://pypi.org/project/toolrank/)
[![CI](https://github.com/yasinyaman/toolrank/actions/workflows/ci.yml/badge.svg)](https://github.com/yasinyaman/toolrank/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/yasinyaman/toolrank/blob/main/LICENSE)
[![Heads](https://img.shields.io/badge/heads-Hugging%20Face-yellow.svg)](https://huggingface.co/yasinyaman/toolrank-heads-qwen3-emb-8b)

**Tool retrieval for LLM agents with hundreds of tools.** Instead of putting every tool definition
into the prompt, toolrank picks the few a request needs, with an embedding model
(Qwen3-Embedding-8B) and small learned heads on top of it, and serves them to your agent over MCP or
REST. Every retriever it ships is measured on the same public benchmarks.

Status: **alpha (0.1)**; interfaces may still change. Documentation: <https://yasinyaman.github.io/toolrank/>

## Quick start

Index MCP servers or OpenAPI specs, then serve them to any MCP client as two tools, `search_tools`
and `call_tool`:

```bash
pip install "toolrank[mcp]"
toolrank heads pull                                   # the packaged heads (60 MB)
toolrank ingest mcp --server time="uvx mcp-server-time" --out tools/
toolrank serve --data tools/                          # MCP at http://127.0.0.1:8765/mcp, REST at /v1
```

toolrank ranks with Qwen3-Embedding-8B served by vLLM (`--emb-url`, by default
`http://127.0.0.1:8091/v1`). On a GPU host, Docker runs both:

```bash
cd deploy/docker && cp .env.example .env              # set TOOLRANK_API_KEY
docker compose run --rm toolrank ingest mcp --config /config/toolrank.json --out /data
docker compose up -d
```

The [quick start](https://yasinyaman.github.io/toolrank/quickstart/) connects Claude Code, Claude Desktop and REST clients.

## Results

Retrieval quality on three benchmarks, every number from `toolrank eval` under ToolRet's protocol
(the reports are in [`docs/results/`](https://github.com/yasinyaman/toolrank/tree/main/docs/results);
`scripts/readme_table.py` checks each one before it prints it).

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

## What's inside

- **Ingestion** of MCP servers (stdio and streamable HTTP) and OpenAPI 3.x specs; a re-run syncs only
  what changed. [Guide](https://yasinyaman.github.io/toolrank/guides/ingest/)
- **Search and serve**: adaptive K, a persistent vector index (numpy, FAISS HNSW or pgvector), an MCP
  proxy with two tools, a REST API, API keys and a usage log. [Guide](https://yasinyaman.github.io/toolrank/guides/serve/)
- **Agent platforms**: toolrank as Claude's (`tool_reference`) and OpenAI's (client-side
  `tool_search`) tool search. [Guide](https://yasinyaman.github.io/toolrank/guides/platforms/)
- **Frameworks**: LangGraph (langgraph-bigtool), LlamaIndex agents and the LiteLLM proxy.
  [Guide](https://yasinyaman.github.io/toolrank/guides/frameworks/)
- **Fine-tuning**: heads trained on your own request-to-tool pairs, the epoch picked on a dev set.
  [Guide](https://yasinyaman.github.io/toolrank/guides/finetune/)
- **Benchmarks**: ToolRet, LiveMCPBench and MCP-Zero with BM25, dense and head scorers (and CLM, for
  comparison). [Benchmarks](https://yasinyaman.github.io/toolrank/benchmarks/)
- **Docker**: the `toolrank` image for amd64 and arm64, compose files with vLLM, and a Dockerfile
  that puts vLLM and toolrank in one container.
  [Guide](https://yasinyaman.github.io/toolrank/guides/docker/)

## Why

- **Tool definitions are expensive context.** Anthropic measured 58 tools at about 55K tokens per
  request and reports tool-selection accuracy falling past 30 to 50 tools. RAG-MCP lifted
  selection accuracy from 13.6% to 43.1% on a large MCP set by retrieving tools first.
- **Hosted tool searches are tied to one model provider or cloud, and most are lexical.** toolrank
  is model-agnostic and runs on your own hardware.
- **Small heads on a frozen embedding model** (29.9M parameters for the request and the tool side
  together). You embed your tools once and rank with one dot product. The heads start as the
  identity, so heads fine-tuned on your data start from the base model's quality, not below it.

## Feedback

Tried it? [Tell us how it went](https://github.com/yasinyaman/toolrank/issues/new?template=feedback.yml):
what you set up, what worked and what did not. Questions and ideas go to
[Discussions](https://github.com/yasinyaman/toolrank/discussions).

## Contributing

Set-up, tests and conventions are in
[`CONTRIBUTING.md`](https://github.com/yasinyaman/toolrank/blob/main/CONTRIBUTING.md). The weekly
reports behind every number, in Turkish, are in
[`docs/reports/`](https://github.com/yasinyaman/toolrank/tree/main/docs/reports).

## License

Apache-2.0 ([`LICENSE`](https://github.com/yasinyaman/toolrank/blob/main/LICENSE)). [`NOTICE`](https://github.com/yasinyaman/toolrank/blob/main/NOTICE) credits what toolrank takes from CLM (the head
architecture), ToolRet (its task metadata) and MCP-Zero (its query prompts);
[`THIRD_PARTY_NOTICES.md`](https://github.com/yasinyaman/toolrank/blob/main/THIRD_PARTY_NOTICES.md) lists the dependencies, models and container
images. Contributions are welcome: see [`CONTRIBUTING.md`](https://github.com/yasinyaman/toolrank/blob/main/CONTRIBUTING.md) and the
[code of conduct](https://github.com/yasinyaman/toolrank/blob/main/CODE_OF_CONDUCT.md); report vulnerabilities as [`SECURITY.md`](https://github.com/yasinyaman/toolrank/blob/main/SECURITY.md) says.
