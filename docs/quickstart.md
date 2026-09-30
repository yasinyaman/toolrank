# Quick start

toolrank needs an embedding model to rank tools: Qwen3-Embedding-8B, served by vLLM on a GPU with
at least 16 GB of memory (or any OpenAI-compatible `/v1/embeddings` endpoint serving it). The
Docker path starts both; the pip path uses an endpoint you already run.

## With Docker (one GPU host)

```bash
git clone https://github.com/yasinyaman/toolrank && cd toolrank/deploy/docker
cp .env.example .env              # set TOOLRANK_API_KEY
```

List the MCP servers you want to put behind toolrank in `toolrank.json` (the same format as an MCP
client's config file). It starts with one:

```json
{
  "mcpServers": {
    "time": {"command": "uvx", "args": ["mcp-server-time"]}
  }
}
```

Index them, then start the stack:

```bash
docker compose run --rm toolrank ingest mcp --config /config/toolrank.json --out /data
docker compose up -d
```

The first start downloads the model (16 GB). toolrank then listens on `127.0.0.1:8765`: MCP at
`/mcp`, REST at `/v1`, both behind the key from `.env`.

```bash
curl -s -H "Authorization: Bearer $TOOLRANK_API_KEY" \
  -d '{"query": "what time is it in Tokyo?"}' http://127.0.0.1:8765/v1/search
```

[Docker](guides/docker.md) covers both images, OpenAPI specs, GPU memory and running behind a proxy.

## With pip

```bash
pip install "toolrank[mcp]"           # add ,openapi for YAML specs; ,stem for a stemmed BM25 fallback
toolrank heads pull                   # the packaged heads, into ~/.cache/toolrank (60 MB)
```

Serve the embedding model on a GPU host, as `qwen3-emb` on port 8091 (the default toolrank uses):

```bash
vllm serve Qwen/Qwen3-Embedding-8B --served-model-name qwen3-emb --runner pooling \
  --max-model-len 8192 --port 8091
```

On another machine, point toolrank at it with `--emb-url http://HOST:8091/v1` or
`TOOLRANK_EMB_URL`; an endpoint that wants a key gets `TOOLRANK_EMB_API_KEY` (toolrank sends
`OPENAI_API_KEY` only to `api.openai.com`). Then index your tools and search them:

```bash
toolrank ingest mcp --server time="uvx mcp-server-time" --out tools/
toolrank ingest openapi https://petstore3.swagger.io/api/v3/openapi.json --name petstore --out tools/
toolrank search --data tools/ "list the pets tagged dog"
```

And serve them to agents:

```bash
toolrank serve --data tools/ --config toolrank.json     # MCP at http://127.0.0.1:8765/mcp
```

`--config` tells `serve` how to reach the MCP servers when an agent calls one of their tools; the
catalogue itself comes from `ingest`. See [Serve them to agents](guides/serve.md).

## Connect an agent

Any MCP client can use toolrank as one server:

=== "Claude Code"

    ```bash
    claude mcp add --transport http toolrank http://127.0.0.1:8765/mcp \
      --header "Authorization: Bearer $TOOLRANK_API_KEY"
    ```

=== "Claude Desktop (stdio)"

    Desktop clients start servers themselves, so `serve` runs with `--stdio`; every path must be
    absolute (Claude Desktop starts servers in `/`):

    ```json
    {
      "mcpServers": {
        "toolrank": {
          "command": "/path/to/venv/bin/toolrank",
          "args": ["serve", "--stdio", "--data", "/path/to/tools",
                   "--config", "/path/to/toolrank.json", "--emb-url", "http://127.0.0.1:8091/v1"]
        }
      }
    }
    ```

=== "REST"

    Platforms that run tools themselves call `POST /v1/search` and then `POST /v1/call` (or run
    the tool on their side); see the [REST API](reference/rest.md).

The agent sees two tools, `search_tools` and `call_tool`, whatever the size of the catalogue.
