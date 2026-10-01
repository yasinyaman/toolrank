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
| `GET /v1/metrics` | Prometheus metrics |
| `GET /openapi.json`, `GET /healthz` | the API description; readiness |

See the [REST reference](../reference/rest.md).

## Many servers, several tools, nothing that fits

Three options for catalogues where plain ranking leaves something on the table. All are off by
default; the numbers are from the benchmarks ([Benchmarks](../benchmarks.md) explains the sets).

**`--server-weight 0.2`: a vote for the right server.** Each server is embedded once as a summary
(its name and tool names), and a tool's score becomes its own cosine plus 0.2 of the request's
cosine with its server. On catalogues of many servers this lifts the first hit: MCP-Zero (293
servers) top-1 79.9 → 81.0, LiveMCPBench NDCG@10 54.0 → 55.1, with fewer tools returned at a higher
recall. Choosing servers first and searching only those loses points everywhere (the right server
ranks first only 70–85% of the time), so the vote is soft. On a catalogue whose "servers" are a few
huge groups it does nothing useful (ToolRet's three categories: −0.1 NDCG@10, −0.8 averaged by
category), which is why it is not the default.

**`--co-use 2`: tools that are called together.** The usage log knows which tools agents called
after the same request. With `--co-use N`, a result gains up to N tools that were called along with
one of its tools in at least two requests and at least half of that tool's requests; they come
last, marked `used_with`. On a simulated log this changed one list in twenty and raised the share
of requests that got *every* tool they needed by 0.6 points for 0.05 more tools per list; making
the ranked list longer buys a seventh of that per tool. The table is rebuilt from the last 30 daily
log files every five minutes, counts all API keys together, and is not applied to a request that
names its own `k`.

**`--cut-threshold T --cut-min 0`: say so when nothing fits.** By default a search returns at least
one tool. With a threshold and a minimum of zero, a request whose best score is below T gets an
empty list and a note telling the agent to answer without a tool or rephrase (add `--cut-margin 0.2`
to keep the usual cut above the threshold). Choose T on your own traffic: the scores of requests
that have a tool and of those that do not overlap, and their scale moves with the catalogue and the
way requests are written. At a T that turns away 1% of answerable requests, MCP-Zero catches a
quarter of the unanswerable ones; on LiveMCPBench no threshold is that cheap. The log keeps what a
turned-away search would have shown (`results`, with `shown: 0`), which is what to calibrate on.

## Metrics

`GET /v1/metrics` answers in Prometheus' text format, behind the same bearer token as the rest of
`/v1`:

```yaml
scrape_configs:
  - job_name: toolrank
    metrics_path: /v1/metrics
    authorization: {credentials: "<TOOLRANK_API_KEY>"}
    static_configs: [{targets: ["127.0.0.1:8765"]}]
```

| Metric | What it tells you |
| --- | --- |
| `toolrank_searches_total{via,mode,arm}`, `toolrank_search_duration_seconds` | traffic and ranking latency (the request's embedding included) |
| `toolrank_search_tools_returned`, `toolrank_search_empty_total`, `toolrank_search_co_use_added_total` | how many tools a search hands over |
| `toolrank_search_returned_tokens_total`, `toolrank_search_saved_tokens_total`, `toolrank_catalog_tokens` | the token estimate (below) |
| `toolrank_calls_total{kind,outcome,via}`, `toolrank_call_duration_seconds` | calls forwarded and how they ended |
| `toolrank_calls_linked_total{link}`, `toolrank_called_tool_rank` | whether calls can be tied to a search, and where the called tool stood in it |
| `toolrank_embedding_texts_total{kind,source}`, `toolrank_embedding_tokens_total` | embedding-cache hits (`source="cache"`) against texts sent to the endpoint |
| `toolrank_catalog_tools`, `toolrank_catalog_sources`, `toolrank_index_ready`, `toolrank_heads{arm}`, `toolrank_build_info` | what is being served |

The token estimate answers "what did searching save over loading every tool?". A tool counts as
its name, description and input schema in JSON at four characters a token; a search *returned* the
tokens of the tools it handed over and *saved* the catalogue's total minus that. So
`saved / (saved + returned)` is the reduction: on a catalogue of 1,862 tools (910k tokens) a search
returns about 5k, a reduction above 99%. It is an estimate and a floor: schemas are counted in full
although later hits are shortened, and the first searches after a catalogue change claim nothing
while the new catalogue is being sized. The share of calls in the top five of their search is
`toolrank_called_tool_rank_bucket{le="5"} / toolrank_called_tool_rank_count`.

Labels never carry request text, tool arguments or API key names (`arm` says `tenant` for any
key's own heads), and the counters are server-wide: any valid key can read them. They count what
the usage log sees and keep counting under `--no-usage-log`; they start at zero with the process.

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
- Tool names, scores, outcomes, latencies and the server's own instruction are text; so are the
  tools a search added by co-use (`added`). An
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
