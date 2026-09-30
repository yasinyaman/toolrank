# Index your tools

`toolrank ingest` turns MCP servers and OpenAPI specs into an ingest directory (see
[Concepts](../concepts.md#the-ingest-directory)). Run it again whenever a source changes: it
prints what was added, changed and removed, and embeds only the new and changed tools.

## MCP servers

```bash
toolrank ingest mcp --server time="uvx mcp-server-time" --out tools/            # stdio: NAME=COMMAND
toolrank ingest mcp --server docs=https://mcp.example.com/mcp --out tools/       # streamable HTTP: NAME=URL
toolrank ingest mcp --config ~/.config/claude/mcp.json --out tools/              # an MCP client's config file
toolrank ingest mcp --config mcp.json --only github,jira --out tools/            # some of its servers
```

Needs the `[mcp]` extra. Servers are listed concurrently, each within `--timeout`; one that fails
keeps its previous tools and is reported. `${VAR}` in a config file is read from the environment,
and neither env vars nor headers are written to `sources.json`.

## OpenAPI specs

```bash
toolrank ingest openapi https://petstore3.swagger.io/api/v3/openapi.json --name petstore --out tools/
toolrank ingest openapi specs/billing.yaml --name billing --out tools/ --dry-run   # the diff only
```

OpenAPI 3.x, JSON or YAML (`[openapi]` for YAML), from a path or a URL. Each operation becomes a
tool: its parameters and request-body properties form one flat input schema, local `$ref`s are
inlined, and the HTTP routing is kept aside for `serve`. The tool text is shrunk schema-first to
6,000 characters, so a huge operation never loses its name or description.

## Removing a source

```bash
toolrank ingest drop billing --out tools/
```

## Embedding ahead of time

With `--emb-url` (or `TOOLRANK_EMB_URL`), ingest embeds the new and changed tools into
`tools/cache/` right away, with the model and truncation `search` and `serve` use, so their first
index is all cache hits:

```bash
toolrank ingest mcp --config mcp.json --out tools/ --emb-url http://gpu-host:8091/v1
```

Use the same URL string `serve` will use: the cache is keyed by it.
