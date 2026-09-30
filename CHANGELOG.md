# Changelog

Notable changes, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [semantic versioning](https://semver.org/); before 1.0, a minor version may
change behaviour.

## [0.1.0] - 2026-09-30

The first public release.

### Added

- `toolrank ingest mcp | openapi | drop`: MCP servers (stdio and streamable HTTP) and OpenAPI 3.x
  specs into an ingest directory; a re-run syncs only what changed and embeds only new or changed
  tools.
- `toolrank search`: Qwen3-Embedding-8B with the packaged heads, adaptive K, and a persistent
  vector index (numpy, FAISS HNSW or pgvector).
- `toolrank serve`: an MCP proxy with two tools, `search_tools` and `call_tool`, in front of every
  ingested tool; a REST API (`/v1/search`, `/v1/rank`, `/v1/call`, `/v1/tools`); API keys; a usage
  log that ties each call to the search that found the tool.
- Tool search for Claude's Messages API (`tool_reference`) and OpenAI's Responses API (client-side
  `tool_search`).
- Framework adapters: LangGraph (langgraph-bigtool retrieval), LlamaIndex (a tool retriever) and
  the LiteLLM proxy (a tool filter).
- `toolrank finetune`: heads trained on your own request-to-tool pairs, the epoch picked on a
  dev set.
- `toolrank eval` and `toolrank compare`: ToolRet, LiveMCPBench and MCP-Zero under ToolRet's
  protocol.
- `toolrank heads pull`; the `toolrank` Docker image (amd64, arm64), a Dockerfile that bundles vLLM,
  and compose examples for both.
