# Changelog

Notable changes, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [semantic versioning](https://semver.org/); before 1.0, a minor version may
change behaviour.

## [Unreleased]

### Added

- `toolrank eval --rerank jev`: TypeSafe AI's Jev reorders the top `--rerank-depth` tools of any
  scorer with one Choice question per query; `--scorer jev` ranks a corpus with Jev alone (chunked
  Choice questions, the chunk winners re-ranked once). The key is read from `TYPESAFE_API_KEY`;
  answers are cached in `jev.sqlite` next to the embedding cache; `scripts/jev_compare.sh` runs the
  comparison rows.
- `toolrank eval --rerank dense|clm|cross`: a second scorer over the first one's shortlist, with its
  own `--rerank-*` endpoint, formats, heads and text cut (`--rerank-max-chars`); `--scorer cross` and
  `--rerank cross` run a cross-encoder behind vLLM's score API (Qwen3-Reranker and
  bge-reranker-v2-gemma prompt formats, scores cached in `scores.sqlite`); compose profile `rerank`
  serves both on the GB10; `scripts/clm_rerank.sh` and `scripts/cross_rerank.sh` run the rows.
- `scripts/lora_train.py` (`[lora]` extra): LoRA fine-tuning of Qwen3-Embedding-8B on the
  fine-tuning data path, a parity check against the served vectors, the best adapter picked on a dev
  set and merged into weights that compose profile `lora` serves as `qwen3-emb-lora`.
- `toolrank learn`: heads trained from what `toolrank serve` logged (calls that ended `ok` as
  positives, `tool_error` as weak positives, tools shown but not called as hard negatives), with the
  requests' vectors found through the log's key and never their text; the newest requests are the
  dev set, and the heads are written only when they beat the served ones there.
- Heads that change while serving: `toolrank serve` follows `DATA/heads` (`current.npz` replaces the
  served heads, `candidate.npz` answers a sticky `--candidate-share` of the requests, `tenants/<name>/`
  the same per API key), and the usage log records which arm answered. `toolrank learn` writes its
  result as the candidate and takes `--replay pairs.jsonl` against forgetting; `toolrank ab` compares
  the arms on the log and promotes the candidate, sets it aside, or waits.
- `toolrank serve --mask-pii`: with `--log-text`, e-mail addresses, phone, card and IBAN numbers are
  replaced by tags before the request or error text is written.
- `--server-weight W` (`eval`, `search`, `serve`): each server is embedded as a summary of its tools
  and a tool's score gains W times the request's cosine with its server. Off by default; 0.2 lifts
  the first hit on catalogues of many servers (MCP-Zero top-1 79.9 → 81.0) and does nothing for a
  catalogue of a few huge groups.
- `toolrank search` / `serve --co-use N`: a result gains up to N tools that the usage log shows were
  called together with one of its tools; hits carry `used_with`, the log's search events `added`.
- `search_tools` says so when a threshold (`--cut-threshold T --cut-min 0`) turned every tool away,
  instead of returning an empty list without a word.
- `scripts/routing_sweep.py` and `scripts/couse_sweep.py`: the measurements behind the two options
  and the no-tool gate.
- `GET /v1/metrics`: Prometheus metrics of a running server (searches and calls with latency
  histograms, where called tools stood in their search, embedding-cache hits, and an estimate of
  the tokens searching saved over loading the whole catalogue).
- `scripts/learn_sim.py` and `scripts/learn_sim.sh`: the learning loop measured on simulated traffic.
  A benchmark is served as a catalogue, a share of its queries is logged by an agent that calls the
  gold tools it is shown, `toolrank learn` trains on that log and the queries never served are the
  test; the learn guide's "What to expect" carries the numbers.

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
