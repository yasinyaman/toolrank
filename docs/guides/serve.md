# Serve them to agents

```bash
toolrank serve --data tools/ --config toolrank.json            # http://127.0.0.1:8765/mcp and /v1
toolrank serve --data tools/ --config toolrank.json --stdio    # for desktop clients
```

Needs the `[mcp]` extra. The agent sees two tools instead of hundreds:

- `search_tools(query, k?)` returns the matching tools with their input schemas (complete for the
  first three, shortened for the rest) and a `search_id`;
- `call_tool(name, arguments, search_id?)` forwards the call to the tool's MCP server or OpenAPI
  operation. A call that fails validation returns the tool's full schema, so the agent can retry.

## The config file

`--config` tells `serve` how to reach the sources when a tool is called: an MCP client file
(`mcpServers`) plus an optional `openapi` section with each API's base URL and headers. `${VAR}` is
read from the environment.

```json
{
  "mcpServers": {"time": {"command": "uvx", "args": ["mcp-server-time"]}},
  "openapi": {"stripe": {"base_url": "https://api.stripe.com",
                         "headers": {"Authorization": "Bearer ${STRIPE_KEY}"}}}
}
```

An API's headers go only to its configured `base_url`, and redirects are not followed. OpenAPI
calls are GET and HEAD only unless you pass `--allow-write`.

## Keys and hosts

On `127.0.0.1` (the default) no key is needed. Any other address needs one:

- `--api-key` or `TOOLRANK_API_KEY`: one bearer token for everyone;
- `--api-keys keys.json`: one named token per client or team (`{"ci-agent": "${CI_AGENT_KEY}"}`);
  the name goes into the usage log as the tenant.

The server checks the `Host` header, so a web page cannot reach it through a rebound domain. It
answers to its bind address and, when bound to `0.0.0.0`, to `localhost`. Add the names it is
reached by, such as a compose service or a proxy's host, with `--allowed-host NAME` or
`TOOLRANK_ALLOWED_HOSTS=a,b`. A name without a port also matches requests with no port, as a
reverse proxy on 80 or 443 sends them.

The server speaks plain HTTP. To reach it from another machine, put a reverse proxy that terminates
TLS in front of it: without one the bearer token, the requests and the tools' results cross the
network in the clear.

## REST

The same port serves REST for platforms that search and call tools themselves:

| Endpoint | What it does |
| --- | --- |
| `POST /v1/search` | a search over the catalogue |
| `POST /v1/rank` | scores up to 200 tools you pass in, or catalogue ids |
| `POST /v1/call` | runs a catalogue tool |
| `GET /v1/tools`, `GET /v1/tools/{id}` | the catalogue (`?full=true` with schemas) |
| `GET /openapi.json`, `GET /healthz` | the API description; readiness |

See the [REST reference](../reference/rest.md).

## The usage log

Every search and call goes to `DATA/usage/` (or `--usage-log DIR`), one JSON line per event in a
file per day, each call tied to the search that found its tool. It is what
[`toolrank learn`](learn.md) trains the heads on.

The server also follows `DATA/heads` while it runs: `current.npz` there replaces the served heads,
`candidate.npz` takes `--candidate-share` of the requests, and `tenants/<name>/` holds the same for one
API key. See [Learn from the usage log](learn.md#trying-the-new-heads-on-live-traffic).

What it holds, and what it does not:

- Requests and call arguments are keyed digests (HMAC-SHA256 under `DATA/usage/.key`, a random
  32-byte key made on first use, mode 0600): repeats are recognisable, guesses are not, and the log
  alone reconstructs nothing. The request's embedding-cache key is a digest too (`emb_hmac`), which
  is how `learn` finds its vector without its text.
- Tool names, scores, outcomes, latencies and the server's own instruction are text. An
  instruction sent with a request is a digest. The client (API key name, client app, remote address)
  is a digest; the session id and the tenant (the API key's name) are text.
- `--log-text` adds the request text and the error text of failed calls; `--mask-pii` then replaces
  e-mail addresses, phone, card and IBAN numbers in them with tags (a pattern, not an understanding).
- `--no-usage-log` turns the log off.

For data-protection purposes (GDPR, KVKK): without `--log-text` the log holds no personal data of
the requests themselves, only digests under a key you hold; the `.key` file and the embedding cache
(`DATA/cache`, which holds the requests' vectors) are the parts to protect and to delete with the
log. To forget a period, delete its daily files; to forget one API key's traffic, filter its
`tenant` lines out. An embedding vector can be matched to text only by embedding a guess and
comparing, which needs the same model and the cache.
