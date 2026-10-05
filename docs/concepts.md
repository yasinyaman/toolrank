# Concepts

## The problem

An agent with hundreds of tools cannot see all of them: their definitions crowd out the task, cost
tokens on every turn, and make the model pick worse. Tool retrieval puts a search in front: the
agent describes what it wants to do, and only the tools that fit are loaded.

## Backbone and heads

toolrank ranks with an embedding model, the **backbone**, served by vLLM (or any OpenAI-compatible
embeddings endpoint). A request and a tool's text become vectors, and the tools whose vectors are
closest to the request's come first. The default backbone is Qwen3-Embedding-8B with a LoRA trained
on ToolRet's training pairs (`yasinyaman/toolrank-emb-8b`, served as `toolrank-emb-v0.2`;
[model card](backbone/MODEL_CARD.md)): the best single stage we measured. Its gain is in requests
that need several tools (on generated two- and three-tool tasks over an API catalogue of our own,
NDCG@10 71 → 83 and 55 → 75); a request for one tool it finds about as well as the base model.

The base Qwen3-Embedding-8B can be served instead (as `qwen3-emb`). On it sit two small **heads**,
one for requests and one for tools (29.9M parameters together). Each is `x + MLP(x)`: it starts as
the identity, so it can only move the backbone's vectors where training showed it helps. The
packaged heads (v0.1) were trained on ToolRet's training pairs; they run in numpy, without torch,
and ship as a 60 MB `.npz` ([model card](heads/MODEL_CARD.md)). toolrank applies them only on the
base model they were trained on: on the LoRA backbone they cost 0.2–3.9 points, and heads trained on
top of it learn nothing. `toolrank finetune` and `toolrank learn` train your own.

The backbone reads an **instruction** in front of each request
(`Instruct: Given an agent's request for a tool, retrieve the MCP tool that fulfills it`), and each
tool as JSON with its server first (`{"server", "name", "description", "inputSchema"}`): the server
name alone is worth several points on MCP benchmarks.

## Adaptive K

A fixed top-k either misses tools that a multi-step request needs or wastes context on tools it
does not. By default toolrank returns the tools within **0.2 cosine of the best one, at most 10**:
a precise request gets one or two tools, a broad one more. `--k N` asks for a fixed number instead.

## The ingest directory

`toolrank ingest` writes one directory, which `search` and `serve` read:

| Path | What it holds |
| --- | --- |
| `tools.jsonl` | the catalogue: one tool per line, id `<source>/<name>` |
| `sources.json` | each source (MCP server or OpenAPI spec) and when it was listed; never secrets |
| `cache/` | embeddings, keyed by endpoint, model, truncation and text |
| `index/` | the tool vectors after the heads, rebuilt only for new or changed tools |
| `usage/` | the usage log (`serve` only) |

Re-running an ingest replaces exactly the sources it lists, keeps a source whose listing failed,
and embeds only new or changed tools. A running server notices the new `tools.jsonl` and swaps its
index without a restart.

Because the embedding cache is keyed by the endpoint's URL as written, `ingest --emb-url` must use
the same URL string as `search` and `serve` for them to find the vectors (`TOOLRANK_EMB_URL` sets it
for all three).

## Search, rank, call

- **Search**: a request against the whole catalogue (`toolrank search`, MCP `search_tools`,
  `POST /v1/search`). Each search gets a `search_id`.
- **Rank**: tools the caller brings, scored for a request, not the catalogue (`POST /v1/rank`, which
  the LiteLLM filter uses).
- **Call**: runs a catalogue tool (MCP `call_tool`, `POST /v1/call`) on its MCP server or as an
  HTTP request to its OpenAPI operation. OpenAPI calls are GET and HEAD only unless `serve
  --allow-write`.

On the agent APIs a tool goes by its **api name**: `github/issues/create` becomes
`github__issues__create`, and ids that don't fit the APIs' character rules get a short hash.

## While the index builds

At the first start, tools missing from the cache are embedded in the background (1,862 tools took
154 s with a remote backbone). Meanwhile searches get BM25 keyword matches, marked as such, and
calls work; the server reports healthy once the semantic index is up.

## The usage log

`serve` appends every search and call to `usage/usage-YYYY-MM-DD.jsonl`: what was retrieved, what
the agent then called and how it went, with each call tied to the search that found its tool.
Requests and arguments are stored as keyed digests, not text, unless `--log-text`. The log is what
the next heads can learn from: tools retrieved but not called, and tools called successfully.
