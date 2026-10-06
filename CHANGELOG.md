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
- `toolrank eval --runs-out FILE` writes one row per query (the top-20 ids, P@1, hit@5, NDCG@10),
  and `toolrank compare --paired A B` tests two such runs query by query: the exact sign test for
  P@1 and hit@5, a paired permutation test for NDCG@10.
- `toolrank finetune --data-seed` seeds the train/val split apart from the training seed, and
  `toolrank learn --data-seed` the replay sample, so the same pairs split alike across training
  seeds. `docs/reports/TEMPLATE.md` gains a pre-registration section.
- The frameworks guide covers Strands Agents: `toolrank serve` speaks streamable-HTTP MCP, so a
  Strands agent connects with no adapter (its own environment — strands-agents pins `mcp<2.2`).
  Verified with strands-agents 1.57.2: the tool list, `search_tools`, and one MCP and one OpenAPI
  `call_tool`.

### Changed

- The docs' product text now describes the v0.2 default (the headless LoRA backbone, heads
  optional): the README's "Why" bullet and `docs/index.md`, the `search` help text, and the
  package docstring. The README's "What's inside" gains learn/ab with tenants, Prometheus metrics,
  `serve --rerank` and the Helm chart; the Python reference documents the LangChain integration;
  README, index and the quick start link the GGUF (no big GPU) guide. The results table's notes
  state LiveMCPBench NDCG@10 next to Recall@5.
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

- `toolrank learn --replay`: the sample now covers the whole pairs file (a seeded reservoir; before,
  the first `max(4N, 1000)` rows in file order stood in for it), replay rows bring their positives
  only (the mined negatives such files carry cost more than they taught in Phase 0), and pairs a
  `--dev` set asks about are dropped, as `finetune` already does. Until the simulation measures
  replay on the v0.2 backbone it is out of the nightly command (CLAUDE.md, the learn guide).
- Searches a Jev second stage answered no longer feed `learn`'s pairs, `ab`'s decision or the
  co-use table (the provider's terms, MCA 2.3(b)); each of them skips such searches and counts
  them (`searches_with_jev`), `ab` prints the count, and serve's Jev line says so.
- `release_check.py` now also refuses a release whose Helm chart `appVersion` names the previous
  one (the chart's image tag defaults to it), and `docs/guides/local.md` pins the GGUF download to
  `--revision v0.2`.
- The second-stage report is `docs/reports/faz2-rerank.md` (was `faz2-jev.md`) and, like the serve
  guide and CLAUDE.md, no longer publishes Jev's results: the provider's terms do not allow it. The
  local rerankers', CLM's and the LoRA backbone's numbers stay; `scripts/rerank_report.py` leaves the
  Jev rows out.
- Jev's price and the dollar figures leave the tracked documents — they are the provider's
  confidential information (MCA §14.1). The token counts stay; `scripts/rerank_report.py` no longer
  prints a fee column.
- Stale sentences in the reports: the phase-1 gate report's "no GPU-less path" and "47.13 < 50"
  (the GGUF guide and the v0.2 backbone settled both), the 0.2.0 release week's "weights not on
  the Hub yet" and "names proposed", the second-stage report's "60K pairs" heads (206K), its bge-gemma
  number (the ToolRet paper's, not measured) and its per-query token range and spend note, the
  backbone card's two em-dashes (LiveMCPBench NDCG@10 of the base model, ToolRet of this model +
  v0.1 heads), benchmarks.md's parity range (it spans Recall@5 and Precision@1, not NDCG), and
  backlog-d1's 8B encode time.
- The Jev seat is provider-aware: `TYPESAFE_API_KEY` goes to `api.typesafe.ai` alone (never to a
  local or third-party `/systemone` endpoint, which is asked without a key), TypeSafe's answers
  stay in `jev.sqlite` with their keys unchanged while every other endpoint gets its own
  `jev-<host>.sqlite` with the URL in the key, the cross-encoder's score cache keys the URL too,
  `JevScorer` never sends a one-option Choice, and the scorer names and serve's privacy line name
  the endpoint (`jev[<model>@<host>,...]`).
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
- `toolrank search` and `serve` take `--clm-ckpt none`: no packaged heads, whatever heads are cached.
  Before, a backbone the cached heads do not fit (Qwen3-Embedding-0.6B or 4B, say) could not be
  served without deleting the heads; the width error now names the flag.
- A running server picks up heads replaced under the same name (`learn` rewriting `candidate.npz`,
  `ab` promoting it to `current.npz`). The heads file's hash was cached by its path, so the variant's
  persistent index kept tool rows projected by the old heads while requests went through the new
  ones. The hash now follows the file (inode, size, modification time) and comes from the bytes the
  heads were loaded from.
- Heads go only on the backbone their cfg names. Heads trained on Qwen3-Embedding-8B and the v0.2
  backbone have the same width, so `--clm-ckpt heads.npz` without `--emb-model` put them silently on
  `toolrank-emb-v0.2`. `search` and `serve` now stop with the fix when the model was left to its
  default (a learned `current.npz` from 0.1.x leaves an error on its variant and the base serves),
  and warn when the model was named and in `eval`. `learn`'s default start skips a promoted
  `current.npz` trained on another backbone, and `finetune --backbone` defaults to what `--emb-model`
  names (it was always `Qwen/Qwen3-Embedding-8B`).
- A search behind a second stage (`serve --rerank`) logs the model that embedded it. It logged
  `model: null`, which `learn`'s backbone filter let through; `learn` also tells searches logged
  without the field by the scorer's name.
- `search` and `serve --rerank jev` ask for `TYPESAFE_API_KEY` only when `--jev-url` is TypeSafe's.
- `scripts/learn_sim.sh` with `HEADS=none` scored the learned heads as the backbone alone; it now
  scores them, serves the simulated traffic with `--clm-ckpt none`, runs on bash 3.2, and prints
  `FAILED` when an eval fails (`scripts/gguf_matrix.sh` too). `scripts/lora_train.py --data-seed`
  draws the training pairs apart from `--seed`, as `finetune` and `learn` do.
- Docs: the REST reference's call outcomes, authentication rule and hit fields; the reranker's
  latency per search and the server vote's measured setting in the serve guide, with the second
  stage's numbers over the v0.2 backbone; the v0.1 heads cost the v0.2 backbone 0.2–3.9 points, not
  1–2; benchmarks' reproduce steps work from a clone; the `vllm serve` lines carry
  `--no-enable-chunked-prefill --max-num-batched-tokens 8192`; the heads card and the fine-tuning
  guide name `--emb-model qwen3-emb`; the DGX Spark README lists every port and says 8091 there
  serves the base model; `SECURITY.md` says what `--rerank jev` sends; the third-party notices list
  the `langchain` extra and the default backbone.
- `learn --replay` drops the pairs a `--dev` set asks about before it draws the sample, so the sample
  has `--replay-n` pairs (it had that many minus the dropped ones); `ab --tenant` counts only that
  key's Jev-served searches.
- A `--jev-url` with `user:password@` in it no longer puts them into the scorer name (the usage log,
  result files) or the cache file's name.
- `ToolrankToolSelector`: model calls that come while a selection is still being made wait for it
  instead of asking toolrank again (each timed-out call had started another ranking, and with a
  toolbox another logged search); an async timeout no longer cancels selections waiting for a worker;
  selections run in daemon threads, so one still running never holds up the interpreter's exit; and
  with a toolbox a failure shows the session's last selection rather than the whole catalogue.
- A persistent vector index shared by two writers with different settings (say `search` with other
  flags on the same `DATA/index`) no longer keeps the other writer's rows as if they were current:
  `NumpyIndex` (and FAISS) rereads a snapshot another process replaced, and `apply` takes the rows
  the writer means to keep (`expect`) and raises `IndexChanged` when one was rewritten meanwhile, so
  `DenseScorer.index` embeds those rows again (pgvector checks the same under `SELECT … FOR UPDATE`).
- A usage-log line no version of toolrank writes (a search without an id, a call without a tool) no
  longer stops `serve --co-use` from starting or the table from ever refreshing again (a `KeyError`
  killed the refresh thread), nor fails `learn` and `ab`: such lines are skipped (`learn` counts them
  as `searches_malformed`).
- `ingest openapi`: a generated name never takes a name the spec uses (operationIds `list`, `list`,
  `list_2` gave two `list_2` tools, and duplicate ids kept the semantic index from being built), and
  `$ref`s that branch at every level stop at 5,000 schema nodes an operation (12 chained schemas of
  6 properties had taken 83 s and written 347 MB for one operation; Stripe's largest has under 900);
  the summary line counts the schemas cut.
- Ingested tool text: schemas whose models sit in `$defs` / `definitions` (pydantic, FastMCP), and
  `additionalProperties` / `patternProperties` subschemas, now shrink like `properties` do; before,
  such a schema never shrank, the description was cut to nothing and the text still went over 6,000
  characters. When no schema step fits, the schema goes and the description stays. Only tools over
  the budget get a new text (and one new embedding).
- pgvector: each heads variant (`current`, `candidate`, a tenant's) keeps its vectors in a table of its
  own (`<--pg-table>_<hash of the variant>`), as its numpy snapshot already did under `index/variants/`;
  before, a candidate's rows overwrote the control arm's in the one table. `eval --pg-table` defaults
  to `toolrank_eval`, so an eval no longer deletes a served catalogue's rows, and a search skips ids a
  shared table holds that the catalogue does not (it ended in a `KeyError`).
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
