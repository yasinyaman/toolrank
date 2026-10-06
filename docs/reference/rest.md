# REST API

`toolrank serve` serves REST next to MCP, on the same port. A running server describes itself at
`GET /openapi.json` (OpenAPI 3.1).

## Authentication

When a key is set (`--api-key`, `TOOLRANK_API_KEY`, or named keys from `--api-keys`), every `/v1`
and `/mcp` request needs `Authorization: Bearer <key>`, on loopback too; `serve` refuses to listen
on a non-loopback address without one. `/healthz` and `/openapi.json` need no key.
Requests must come with an allowed `Host` header and, from a browser, an allowed `Origin`; bodies are
limited to 1 MiB. `X-Session-Id` groups one conversation's searches and calls in the usage log.

## Endpoints

### `POST /v1/search`

```json
{"query": "refund the last payment", "instruction": null, "k": null, "full_schemas": false}
```

Searches the catalogue. `k` fixes the number of tools (default: adaptive K); `instruction` replaces
the serving instruction; `full_schemas` returns every hit's full input schema (default: the first
three, the rest shortened). The answer has `search_id`, `mode` (`semantic`, or `lexical` while the
first index builds), `took_ms`, `rule` and `tools`. Each tool has `name` (its id), `api_name`,
`server`, `kind` (`mcp` or `openapi`), `score`, `description` and `inputSchema`, with
`inputSchemaShrunk` when the schema was shortened; `annotations` when the tool has some, `method`
for OpenAPI operations; a tool appended by `serve --co-use` also has `used_with`, the returned tool
it is called together with. A `note` for the agent is added when the tools are keyword matches,
when none is close enough, or when the reranker did not answer and the tools are in the first
stage's order. With a calibration (`toolrank calibrate`) the answer also has `confidence`, the
share of answerable requests whose best score was at or below this one's.

### `POST /v1/rank`

```json
{"query": "refund the last payment", "tools": [{"name": "refund", "description": "...", "inputSchema": {}}]}
```

Scores up to 200 tools that you pass in (MCP tool objects, with an optional `server`) or catalogue
ids (`tool_ids`), best first; each result has the tool's `index` in the request and its `score`
(cosine). Needs the semantic index (503 while the first build runs).

### `POST /v1/call`

```json
{"name": "time/get_current_time", "arguments": {"timezone": "Asia/Tokyo"}, "search_id": "s-..."}
```

Runs a catalogue tool exactly as MCP `call_tool` does, with the same write policy. The answer has
`outcome` (`ok`, `tool_error`, `protocol_error`, `timeout` or `refused`), `isError`, `content` (MCP
content blocks) and, for OpenAPI
operations, `http_status`. A tool that fails is still a 200 with `isError: true`. JSON bodies only.

### `POST /v1/feedback`

```json
{"name": "stripe/PostRefunds", "outcome": "ok", "search_id": "s-...", "took_ms": 840}
```

For a client that runs the tools itself (its agent platform calls the APIs) and wants toolrank to
learn from what it called. It reports a call: the tool id, `outcome` (`ok`, or `tool_error` when the
tool answered with an error), and optionally the search that found it and how long it took. The
call goes to the usage log as if toolrank had run it, linked to that search (`ran_by: client`), so
`toolrank learn`, `ab` and co-use count it. The answer is `{name, call_id}`. JSON bodies only.

### `GET /v1/tools`, `GET /v1/tools/{id}`

The catalogue, optionally one server's (`?server=`). With `?full=true`, each tool also has its input
schema, annotations and, for OpenAPI operations, the HTTP method, plus the catalogue's hash: the
one download a platform client needs.

### `GET /v1/metrics`

Prometheus metrics in the text exposition format, behind the same key as the rest of `/v1`: searches
and calls with their latency histograms, the token estimate, embedding-cache hits, and the
catalogue's size. See [Metrics](../guides/serve.md#metrics).

### `GET /healthz`

200 with `{ready, mode}` once the semantic index is ready, 503 before. It needs no key and checks no
`Host` (load balancers and probes call it), so it says nothing more.

## Errors

Errors are JSON `{"error": "..."}`:

| Status | Meaning |
| --- | --- |
| 400 | bad input |
| 401 | missing or wrong key |
| 403 | a browser origin that is not allowed; `/v1/metrics` for a key limited to some sources; what a key's `scopes` do not allow |
| 404 | unknown tool |
| 413 | body over 1 MiB |
| 415 | a call or feedback that is not JSON |
| 421 | a Host header that is not allowed |
| 503 | index not ready, embedding endpoint down, or backends not running |

## A client

`toolrank.client.ToolrankClient` wraps these endpoints with the standard library only; see the
[Python API](python.md).
