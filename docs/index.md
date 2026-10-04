# toolrank

**Tool retrieval for LLM agents with hundreds of tools.** Instead of putting every tool definition
into the prompt, toolrank picks the few a request needs, with an embedding model
([Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), trained further on tool
retrieval) and hands only those to the agent.

- **Your tools, indexed.** MCP servers (stdio or streamable HTTP) and OpenAPI specs become one
  catalogue; re-running the ingest syncs only what changed.
- **Two tools instead of hundreds.** `toolrank serve` is an MCP server with `search_tools` and
  `call_tool` in front of the whole catalogue, plus a REST API for platforms that run tools
  themselves.
- **Where agents already look for tools.** Claude's and OpenAI's tool search, LangChain agents,
  LangGraph (langgraph-bigtool), LlamaIndex agents, the LiteLLM proxy and Strands Agents.
- **Measured.** Every scorer is benchmarked on ToolRet, LiveMCPBench and MCP-Zero under the same
  protocol; see [Benchmarks](benchmarks.md).
- **Yours to run.** Apache-2.0, model-agnostic, on-prem: one GPU for the embedding model (or none:
  [it runs as a GGUF in Ollama](guides/local.md)), or any OpenAI-compatible embeddings endpoint.

```bash
pip install "toolrank[mcp]"
toolrank ingest mcp --server time="uvx mcp-server-time" --out tools/
toolrank serve --data tools/        # MCP at http://127.0.0.1:8765/mcp, REST at /v1
```

The [quick start](quickstart.md) walks through it, with the embedding model running in Docker.

## Why

Tool definitions are expensive context. Anthropic measured 58 tools at about 55K tokens per request
and reports tool-selection accuracy falling past 30 to 50 tools; RAG-MCP lifted selection accuracy
from 13.6% to 43.1% on a large MCP set by retrieving tools first. The hosted tool searches are tied
to one model provider or cloud, and most are lexical. toolrank is a retriever you can run anywhere
and measure.

The backbone is an embedding model trained further on tool retrieval, and tool vectors are computed
once. The optional heads on top of it are small (29.9M parameters for the query and the tool side
together) and start as the identity: heads trained on your own requests (`toolrank finetune`) cannot
start below the base model's quality.

## Feedback

Tried it? [Tell us how it went](https://github.com/yasinyaman/toolrank/issues/new?template=feedback.yml),
or ask in [Discussions](https://github.com/yasinyaman/toolrank/discussions).
