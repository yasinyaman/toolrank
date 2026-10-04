# Changelog

Notable changes, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [semantic versioning](https://semver.org/); before 1.0, a minor version may
change behaviour.

## [Unreleased]

### Added

- `toolrank.integrations.langchain.ToolrankToolSelector` (`[langchain]` extra): a LangChain 1.x
  `create_agent` middleware that shows each model call only the tools toolrank finds for the last
  user message. It ranks the agent's own tools through `/v1/rank`, or, with a `Toolbox`, picks the
  catalogue's tools by a search whose calls are linked in the usage log. It keeps the tools already
  called, `tool_choice`, `always_include` and provider tools, and fails open.
- `examples/skills/toolrank`: a skill for agents with only a shell (Claude Code skills, bash-only
  harnesses). It has a short `SKILL.md` and two standard-library scripts, `search.py` and `call.py`,
  around `/v1/search` and `/v1/call`. They do not follow redirects and use exit codes that tell a
  tool's error from toolrank's.
- A guide for running without a large GPU (`docs/guides/local.md`): the default backbone as GGUF in
  Ollama, at the same quality as vLLM (ToolRet NDCG@10 59.50 for Q4_K_M against 58.90), and the smaller
  Qwen3-Embedding models, measured on a 4 GB laptop GPU. `scripts/gguf_matrix.sh` runs those rows.

### Changed

- The `dev` extra is now a PEP 735 dependency group, so the package on PyPI no longer offers a `dev`
  extra. `uv sync` installs it. The images' `uv export` and the third-party notices pass
  `--no-default-groups`, so the images keep the same 35 packages.
- README: CLM here is Contrastive-LM, not Context Language Models.
- `search` and `serve` skip the packaged heads for the Ollama names of the GGUF builds and of the
  smaller Qwen3-Embedding models (`build.BACKBONES`), as they already did for the LoRA backbone.
- `scripts/fp8_agreement.py` compares any two endpoints, vLLM against Ollama or llama.cpp included;
  `--encode` sends what one side lacks, `--heads none` compares the backbones alone.
- `scripts/learn_sim.sh` runs headless with `HEADS=none`: dense scoring and `learn --init none`,
  for the v0.2 backbone.

### Fixed

- `toolrank learn` on the default (v0.2) backbone: it started from a random head, mixed the
  vectors of requests different models had answered, and never saw the promoted heads. Now a start
  without fitting heads is a fresh skip head (identity, so epoch 0 is the backbone's own score),
  the usage log records the serving model and `learn` uses only its requests (the rest are skipped
  and counted), and the start is what a server would serve: `DATA/heads/current.npz` first (a
  tenant's own with `--tenant`), the packaged heads when they fit, identity otherwise. `--init`'s
  help said "the served ones"; now it means it.
- `--clm-ckpt none` drops only the packaged heads: a learned `DATA/heads/current.npz` (and a
  candidate) still serves. Before, `none` switched them off too while `candidate.npz` kept running.
- `/v1/rank` ranks with the heads a search would use: the tenant's own, the promoted
  `current.npz`, the candidate's share. Before, it always used the base heads.
- `build.BACKBONES` knows the vLLM names of the small Qwen3-Embedding models, the GGUF names
  `scripts/gguf_matrix.sh` serves and `qwen3-emb-lora-fp8`: the packaged heads no longer count as
  fitting them.
- `toolrank search` and `serve` take `--clm-ckpt none`: the backbone alone, whatever heads are cached.
  Before, a backbone the cached heads do not fit (Qwen3-Embedding-0.6B or 4B, say) could not be
  served without deleting the heads; the width error now names the flag.
- `scripts/serve_e2e.py`, `platforms_e2e.py` and `frameworks_e2e.py` pass `--emb-model` (default
  `qwen3-emb`) to the server they start; since 0.2.0 the server's own default is the LoRA backbone's
  name, which port 8091 does not serve.

## [0.2.0] - 2026-10-02

Learning from the usage log, a second stage, tenants, metrics, a Helm chart, and a LoRA-trained
default backbone.

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
- `toolrank search` / `serve --rerank cross | jev`: a second stage reorders the top 20 tools with the
  request (a local cross-encoder behind vLLM's score API, or TypeSafe AI's hosted Jev); the number of
  tools returned still comes from the first stage's cosines.
- `toolrank data gen-queries`: a selection set of your own. A chat model writes one request per
  sampled tool of any catalogue (four styles; tools sampled evenly over the sources), giving a
  benchmark-format directory for `eval`, `finetune --dev` and `learn --dev` that shares no query
  with a benchmark. `scripts/lora_train.py --keep-all` keeps every evaluated adapter, so another
  dev set can pick again.
- Tenants: an `--api-keys` entry can be `{key, sources, headers, env}`. `sources` limits the key to
  those sources' tools (search, catalogue and calls); `headers` and `env` are its own credentials
  for a source, sent only with its calls over a connection of its own. Co-use tables are per key.
- A Helm chart (`deploy/helm/toolrank`): toolrank with its backbone as a vLLM pod, an external
  endpoint or the bundled image, FP8 or bf16 profiles, MCP sources ingested by an init container,
  keys and tenants from values or Secrets; `scripts/helm_smoke.py` installs it on a cluster without
  a GPU.
- `GET /v1/metrics`: Prometheus metrics of a running server (searches and calls with latency
  histograms, where called tools stood in their search, embedding-cache hits, and an estimate of
  the tokens searching saved over loading the whole catalogue).
- `scripts/learn_sim.py` and `scripts/learn_sim.sh`: the learning loop measured on simulated traffic.
  A benchmark is served as a catalogue, a share of its queries is logged by an agent that calls the
  gold tools it is shown, `toolrank learn` trains on that log and the queries never served are the
  test; the learn guide's "What to expect" carries the numbers.

### Changed

- The default backbone is Qwen3-Embedding-8B with a LoRA trained on ToolRet's training pairs
  (`yasinyaman/toolrank-emb-8b` at `v0.2`, served as `toolrank-emb-v0.2`, `-fp8` in FP8): ToolRet
  NDCG@10 58.90 against 54.03 for the base model with heads, in one stage and without heads. On an API
  catalogue of our own it gains on tasks that need several tools and ties with the base model on
  requests for one tool (see its model card). The
  packaged heads are applied only on the base model they were trained on (`qwen3-emb`,
  `qwen3-emb-fp8`; `TOOLRANK_HEADS` still forces them); `toolrank learn` starts from no heads on the
  new backbone. Docker, compose and the Helm chart serve it by default; `TOOLRANK_BACKBONE` /
  `embedding.backbone` select the base model again.
- The documentation lives at <https://yaman.dev/toolrank/> (the old address redirects there).

### Fixed

- Heads given vectors of another width (an embedding endpoint serving another model) now say so,
  naming both widths, instead of failing inside the index build with a bare shape error.
- OpenAPI calls no longer keep cookies: one API response's `Set-Cookie` was sent with every later
  call to that host, whoever made it.

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
