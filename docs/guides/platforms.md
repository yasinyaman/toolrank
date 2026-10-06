# Claude and OpenAI tool search

Claude's Messages API and OpenAI's Responses API can both keep tool definitions out of the model's
context and load them when a search finds them. toolrank can be that search.
`toolrank.integrations` does the protocol work without importing the vendors' SDKs; the examples
use the official ones.

```bash
pip install "toolrank[mcp,anthropic,openai]"
toolrank serve --data tools/ --config toolrank.json                       # in another shell
python examples/anthropic_tool_reference.py "What time is it in Tokyo?"   # ANTHROPIC_API_KEY
python examples/openai_client_tool_search.py "What time is it in Tokyo?"  # OPENAI_API_KEY
```

## Claude (Messages API)

Every catalogue tool is sent with `defer_loading: true`, so none takes context. Claude sees one
ordinary tool, `search_tools`; toolrank answers it with `tool_reference` blocks, which the API
expands into the tools found, and runs the calls Claude then makes.

```python
import anthropic
from toolrank.client import ToolrankClient
from toolrank.integrations import anthropic as tr

result = tr.run(anthropic.Anthropic(), ToolrankClient(), "What time is it in Tokyo?", model="claude-opus-5-5")
print(result.text)
```

- References name only tools of the catalogue snapshot sent with the request (an unknown name fails
  the whole request), and the tool list stays the same for the whole conversation.
- The deferred definitions travel with every request (1,862 tools: 3.7 MB) but stay out of the
  prompt cache. `Toolbox(..., builtin="bm25")` swaps in the API's own tool search, for comparison.
- In our runs toolrank's search used 37 to 60% fewer input tokens than the API's own BM25 search,
  at the cost of one extra turn per search (the search runs on your side).

`Toolbox(client, inline=True)` (the example's `--inline`) sends no catalogue at all. It uses the
`inline-tools-2026-09-15` beta, on Claude Opus 4.8 and later. `tools` holds `search_tools` alone.
The tools a search finds are added by value, in `tool_addition` blocks of a `role: "system"`
message right after the search's result, once per conversation. The first three come with full
input schemas and the rest are shortened, as over MCP. Nothing travels per request but the
conversation, and the 10,000-tool limit is gone. With `prefetch=True` (`--prefetch`) the task itself
is searched before the first request and its tools are added after the first user message, so
Claude can call one without a search turn. `run` sends the beta header. This matches the SDK's
types, but it has not been run against the live API yet.

## OpenAI (Responses API)

The request declares only `tool_search` with `execution: "client"`. toolrank answers each
`tool_search_call` with the full definitions of the tools found, and the loop stays stateless
(`store=False`, encrypted reasoning).

```python
from openai import OpenAI
from toolrank.client import ToolrankClient
from toolrank.integrations import openai as tr

result = tr.run(OpenAI(), ToolrankClient(), "What time is it in Tokyo?", model="gpt-5.5")
```

Loaded definitions are sent again on every later turn, so broad searches over large schemas add
up. As MCP's `search_tools` does, each search loads its first three tools with full input schemas
and the rest with schemas cut to 1,500 characters. The description says when a schema was cut, and
a call that fails returns the tool's full schema. On the Stripe refund search in
[the week-4 setup](https://github.com/yasinyaman/toolrank/blob/main/docs/reports/faz1-week4.md)
this loads 16.7 KB instead of 20.7 KB, and no later tool can bring in one of Stripe's 50 KB
schemas. `Toolbox(client, shrink=False)` loads every schema in full.

`Toolbox(client, namespaces=True)` (the example's `--namespaces`) loads the found tools grouped by
server: one `namespace` per server, with each tool under its own name (`issues__create` in
`github`), so long ids keep readable names. It matches the SDK's types; it has not been run
against the live API yet.

## On both

- The calls go through `POST /v1/call`, with the same write policy, usage log and error hints as
  over MCP.
- Tools are named by their api name (`github/issues/create` becomes `github__issues__create`).
- The examples ask before any call that could change something (neither `readOnlyHint` nor a GET)
  unless `--yes`. A tool's output reaches the model, so treat it like any other input.
- Code with its own loop can use `toolrank.client.ToolrankClient` or each integration's `Toolbox`
  directly.
