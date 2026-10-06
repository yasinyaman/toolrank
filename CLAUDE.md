# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**toolrank** — tool retrieval for LLM agents with hundreds of tools. It began (Phase 0, Sep 2026) as
a benchmark harness whose job was a fair number per scorer (BM25, Qwen3-Embedding-8B, CLM heads) on
ToolRet, then MCP-Zero and LiveMCPBench, to decide the core model. Decided on 29 Sep 2026
(`docs/reports/faz0-gate.md`): the gate did not hold; the core adapter became Qwen3-Embedding-8B +
our own skip heads, and since 0.2.0 the default is the toolrank backbone (Qwen3-Embedding-8B + a
LoRA on ToolRet pairs), served headless as `toolrank-emb-v0.2` — the v0.1 heads cost it 0.2–3.9
points. Search and serve expect it on port 8091 (the images serve it there); on the GB10, 8091 is
the base model as `qwen3-emb` and the LoRA backbone is the `lora` profile on 8097. CLM stays as a
benchmark adapter. Plans and gates live in `docs/plan/` (Turkish): a separate,
private git repository checked out in place and ignored here, so public clones do not have it; never
copy its content into tracked files. The product/market plan lives outside the repo.

## Working agreements

- Python 3.11+, `uv`, `src/` layout. Before any commit: `uv run ruff format src tests scripts examples`,
  `uv run ruff check src tests scripts examples`, `uv run pytest` — all green, no exceptions. CI
  (3.11–3.13) runs `ruff check`, `pytest` and a 50-query synthetic BM25 eval; it never installs
  `[clm]`, so the torch tests (`test_clm_heads.py`, the training tests in `test_finetune.py`) are
  skipped there and wherever torch is missing: after touching `adapters/clm.py` or `finetune.py`,
  run `uv sync --extra clm` and then those tests locally.
- Ports and adapters: `domain.py`, `ports.py`, `formats.py`, `eval/`, `ingest/`, `cut.py`,
  `calibration.py`, `retriever.py`, `usage.py` never import from `adapters/`; `build.py` is the composition root
  (flags → scorer, encoder, heads, index) that eval, search and serve share, and may import
  adapters inside functions. A new scorer = one file in `adapters/` implementing the `Scorer` Protocol plus
  `tool_format` / `query_format` attributes (`cmd_eval` records their `.name`; the Protocol
  doesn't require them), a branch in `build.build_scorer` and its name in the `--scorer`
  choices, and a test on the synthetic set (`datasets/synthetic.py`). Tests never call a live
  endpoint: encoders are faked (`_ToyEncoder`, `_RandEncoder`), CLM heads use `_fake_checkpoint`,
  MCP servers are `tests/fixtures/mcp_server.py` (in-process, stdio or local HTTP; `--mode extra`
  adds slow/crash/pid tools), HTTP APIs are `httpx2.MockTransport`, the served app runs under
  Starlette's `TestClient` (base URL `http://127.0.0.1:8765`, or the Host check answers 421), numpy heads
  check against `tests/fixtures/heads_golden.npz` (torch outputs; regenerate with
  `make_heads_golden.py`), pgvector tests run only with `TOOLRANK_PG_DSN` (the `pg` compose profile).
- Heavy dependencies stay optional and are imported inside functions: `torch` → `[clm]`,
  `datasets` → `[data]`, `PyStemmer` → `[stem]`, `mcp` → `[mcp]`, `pyyaml` → `[openapi]`,
  `faiss-cpu` → `[faiss]`, `psycopg` + `pgvector` → `[pgvector]`, `langchain-core` → `[langgraph]`,
  `langchain` → `[langchain]`, `llama-index-core` → `[llamaindex]`, the `anthropic` / `openai` SDKs →
  `[anthropic]` / `[openai]` (the examples; the integrations never import them), and torch +
  transformers + peft → `[lora]` (`scripts/lora_train.py` only). All but torch, transformers, peft and
  datasets are also in the `dev` dependency group, which `uv sync` installs; litellm is no
  extra and stays out of the venv, since it pins `openai<3`). `[mcp]` also brings the serve stack (starlette, uvicorn, httpx2, anyio);
  `test_ingest_mcp.py` checks that importing `toolrank.cli` and the serve adapters loads none of them. Packaged heads run
  in numpy (`adapters/heads_np.py`): torch is only for training and `toolrank heads export`. The base install must stay numpy + bm25s. BM25 stems only when PyStemmer is importable (otherwise the run name ends in
  `/nostem`), so the same command scores differently across installs; the paper's BM25s is
  unstemmed (`--no-stem`).
- The evaluation protocol is fixed to ToolRet's released code: top-100 over the full corpus,
  cut-offs 5/10/20, micro-average over queries (its size-weighted "Avg"), "Comprehensiveness@k"
  = every gold tool retrieved. Reports carry the paper's aggregation next to it (`cat-macro`,
  see Reference numbers), but the micro-average stays the protocol's number.
  Do not change `eval/metrics.py` semantics without a test proving trec_eval parity.
- `data/` and `results/*.json` are gitignored on purpose. The README's results table is the
  exception: `docs/results.toml` names curated reports in `docs/results/` (tracked, made by
  `scripts/readme_results.sh` on the GB10), `scripts/readme_table.py --write` renders them after
  checking each against the protocol (`eval/table.py`) into the README and `docs/benchmarks.md`, and a test keeps
  both equal. Other generated files with a `--check` test: `docs/reference/cli.md` (`scripts/cli_reference.py`,
  from the parser) and `THIRD_PARTY_NOTICES.md` (`scripts/third_party.py`, from `uv.lock`); after a
  `pyproject.toml` change run `uv lock`, and after a dependency change `third_party.py --write`. Numbers that matter go into
  `docs/reports/<faz>-<hafta>.md` (e.g. `faz0-week1.md`; gate: `faz0-gate.md`; start from
  `TEMPLATE.md`) as a `toolrank compare` table with the exact commands used — the results JSON
  records formats, `with_inst`, `tasks`, `limit`, `emb_model` / `emb_url` / `truncate`, `batch`
  and `encoder_tokens` (tokens sent on cache misses), but not the commit.
- Public steps (push, PyPI, GHCR, Pages, posts) never happen during working hours: only after 19:00
  (Europe/Istanbul) and each with the user's explicit go. Local commits are fine any time. The
  release and docs workflows stay off until the repo variables `TOOLRANK_RELEASE` / `TOOLRANK_PAGES`
  are set; `scripts/release_check.py` lists what blocks a release (launch placeholder markers, an
  empty `HEADS_URL`, a dev version, an undated CHANGELOG entry). The public history starts with one
  squashed commit by `Yasin YAMAN <99225310+yasinyaman@users.noreply.github.com>`; the history before
  it is the local branch `local-history` and is never pushed (`remote.origin.push` is `main` only;
  push tags by name).
- Language: code, docstrings, README, commit messages in English; `docs/plan` and
  `docs/reports` in Turkish.
- Commits: small, imperative subject, prefixed with the plan item, e.g.
  `faz-0/w1: reproduce BM25 baseline on ToolRet`. Tick the matching checkbox in
  `docs/plan/faz-N.md` and commit it in the plan repository (`git -C docs/plan commit`) with the
  same subject. Never move phase dates; a missed gate narrows scope.
- Do not "fix" a benchmark number by changing the protocol. If a reproduction is off by more
  than 1 point, report the gap and the suspected cause (tokenisation, stemming, truncation).
- Large changes: plan first (files + approach), then code. Long runs (encoding the 44k corpus,
  fine-tunes) go on the Spark under `nohup`/`tmux`: write the command for the user to start;
  results land in `results/`.

## Commands

```bash
uv sync                               # + the dev group; --extra data for the ToolRet pull, --extra clm for CLM heads
uv run pytest && uv run ruff check src tests scripts examples
uv run pytest tests/test_formats.py::test_query_formats        # one test; or -k <keyword>

toolrank data synth --out data/synthetic                       # offline smoke set; its scores mean nothing
toolrank eval --data data/synthetic --scorer bm25 --tool-format schema
toolrank formats                                               # list tool / query formats

# benchmark runs happen on the GB10 in ~/toolrank; git stays on the Mac (see Environment).
# Push tracked files (`git add` new ones first), run there, fetch results for compare/reports.
git ls-files -z | COPYFILE_DISABLE=1 tar --null -T - --no-xattrs --no-mac-metadata -cf - | ssh gb10 'mkdir -p ~/toolrank && tar -xf - -C ~/toolrank'
ssh gb10 'cd ~/toolrank && ~/.local/bin/uv run toolrank <args>'
scp 'gb10:toolrank/results/*.json' results/

# the eval lines below run on the GB10, from ~/toolrank
toolrank data pull toolret --out data/toolret                  # once, needs Hugging Face access
toolrank eval --data data/toolret --scorer bm25 --tool-format documentation --out results/toolret_bm25.json
toolrank eval --data data/toolret --scorer bm25 --tool-format documentation --with-inst --out results/toolret_bm25_inst.json
toolrank eval --data data/toolret --scorer bm25 --tasks apibank,toolbench --limit 50   # quick subset; --limit = first N in file order, not a sample

# CLM: Qwen3-8B pooling server (port 8090) + reference heads
toolrank eval --data data/toolret --scorer clm --emb-url http://127.0.0.1:8090/v1 --emb-model qwen3-8b \
  --truncate 2048 --tool-format name_desc --with-inst --out results/toolret_clm_name_desc_inst.json
# any embedding model (port 8091 = Qwen3-Embedding-8B; --truncate = its --max-model-len)
toolrank eval --data data/toolret --scorer dense --emb-url http://127.0.0.1:8091/v1 --emb-model qwen3-emb \
  --truncate 8192 --tool-format name_desc --with-inst --out results/toolret_qwen3emb_inst.json
# week-2 zero-shot matrix (4 tool formats x CLM / clm-raw / Qwen3-Embedding x w/o, w/ inst); restartable.
# Don't sync to the GB10 while it runs: bash reads the script as it goes and tar rewrites it.
tmux new -d -s matrix 'bash scripts/run_matrix.sh 2>&1 | tee -a data/logs/matrix.log'
# head fine-tune on cached vectors (Faz 1 week 5; Faz 0's train/ scripts are gone, their commands stay in
# docs/reports/faz0-week3.md): the epoch is picked on --dev, never reported; eval runs on every --eval
toolrank data pull toolret-train                               # -> data/toolret_train/pairs.jsonl (3.3 GB)
toolrank finetune --data data/toolret_train/pairs.jsonl --embed-only --dev data/mcp_zero_server   # fill the cache, no torch
toolrank finetune --data data/toolret_train/pairs.jsonl --n-train 60000 --dev data/mcp_zero_server \
  --eval data/toolret --eval data/livemcpbench_server --out data/heads/<name>.pt [--npz dist/heads/<name>.npz] [--init-ckpt default] [--data-seed N]
# week-4 held-out sets: LiveMCPBench from GitHub; MCP-Zero ships no queries, a chat model writes them
toolrank data pull livemcpbench                                # -> data/livemcpbench (525 tools, 94 queries)
docker compose -f deploy/spark/compose.yaml --profile gen up -d qwen3-8b-chat   # port 8093; stop it afterwards
toolrank data pull mcp-zero --gen-workers 64                   # 333 MB from Google Drive + one request per tool
toolrank eval --data data/mcp_zero --ks 1,5,10,20 <scorer flags>   # Precision@1 = the paper's top-1 accuracy
# a selection set of one's own (Faz 2 week 7): a chat model writes one request per sampled tool of a catalogue
toolrank data gen-queries --data data/devcat --out data/dev_w3 --n 1000 --exclude data/toolret [--styles situation]
toolrank data gen-queries --data data/devcat --out data/dev_w3_multi2 --n 800 --tools-per-request 2   # tasks: all tools gold

toolrank compare results/toolret_*.json                        # on the Mac: markdown table for the report
toolrank eval ... --runs-out results/a_runs.jsonl              # one row per query (top-20 ids, P@1, hit@5, NDCG@10)
toolrank compare --paired results/a_runs.jsonl results/b_runs.jsonl   # sign test (P@1, hit@5), permutation test (NDCG@10)
PYTHONUNBUFFERED=1 nohup bash scripts/readme_results.sh > data/logs/readme_results.log 2>&1 &   # GB10: the README's runs
uv run python scripts/readme_table.py --write                  # Mac, after scp 'gb10:toolrank/results/readme_*.json' docs/results/

# Faz 1: index your own tools into an ingest dir (tools.jsonl + sources.json); re-run to sync
toolrank ingest mcp --server time="uvx mcp-server-time" --config ~/mcp.json --out data/mytools
toolrank ingest openapi data/specs/stripe.json --name stripe --out data/mytools   # JSON or YAML, path or URL
toolrank ingest openapi spec.yaml --out data/mytools --dry-run                    # print the diff only
toolrank ingest drop stripe --out data/mytools
# + --emb-url http://127.0.0.1:8091/v1 (the URL serve will use): embed new/changed tools into DIR/cache now

# Faz 1 week 2: search an ingest dir (defaults: toolrank-emb-v0.2 on 8091, no heads,
# adaptive K margin 0.2 / max 10, persistent index in DIR/index; --json for machines;
# against the GB10 name what it serves: --emb-model qwen3-emb on 8091, or 8097's TOOLRANK_LORA_NAME)
# (the v0.1 heads sit on qwen3-emb, so a run with them names the base model:)
TOOLRANK_HEADS=dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz toolrank search --emb-model qwen3-emb --data data/mytools "refund this payment"
toolrank eval --data data/toolret --scorer clm ... --index faiss|pgvector --hybrid --cut-margin 0.2 --instruction "..."
toolrank eval --data data/toolret --scorer clm ... --rerank jev --rerank-depth 100   # TypeSafe AI's Jev over the top K
# (key in TYPESAFE_API_KEY, sent to api.typesafe.ai only; answers cached in .cache/toolrank/jev.sqlite, another
# --jev-url /systemone endpoint is asked without a key into jev-<host>.sqlite); --scorer jev = Jev alone, chunked, MCP sets only.
# GB10, key exported in the shell: LIMIT=50 TAG=jevsmoke bash scripts/jev_compare.sh, then the full run under nohup
toolrank eval ... --rerank clm|dense|cross --rerank-depth 20 --rerank-emb-url ... --rerank-tool-format documentation \
  --rerank-max-chars 3000 [--rerank-template qwen3|bge]   # a local second scorer over the shortlist (scripts/clm_rerank.sh, cross_rerank.sh)
docker compose -f deploy/spark/compose.yaml --profile rerank up -d qwen3-reranker bge-reranker   # 8095 / 8096, vLLM score API
uv run --extra lora python scripts/lora_train.py --pairs data/toolret_train/pairs.jsonl --dev data/mcp_zero_server \
  --eval data/toolret --eval data/livemcpbench_server --n-train 20000 --out data/lora/<name> --check-parity   # GB10, hours
TOOLRANK_LORA=$HOME/toolrank/data/lora/<name>/merged docker compose -f deploy/spark/compose.yaml --profile lora up -d qwen3-emb-lora  # 8097
toolrank heads export data/heads/<run>.pt dist/heads/<name>.npz --dtype float16 --tool-format documentation ...
uv run python scripts/adaptive_k_sweep.py --margins 0.1,0.2 -- <eval flags>     # rank once, cut many ways
uv run python scripts/heads_parity.py --a x.pt --b x.npz -- <eval flags>         # do two checkpoints rank alike
docker compose -f deploy/spark/compose.yaml --profile pg up -d toolrank-pg        # Postgres + pgvector on 127.0.0.1:55440

# Faz 1 week 3: serve an ingest dir to agents: MCP at /mcp + REST at /v1 on 127.0.0.1:8765, or --stdio.
# --config = an MCP client file (+ "openapi": {source: {base_url, headers}}, ${ENV} expanded); search flags apply
TOOLRANK_HEADS=dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz toolrank serve --emb-model qwen3-emb --data data/mytools --config toolrank.json
curl -s -H "Authorization: Bearer $TOOLRANK_API_KEY" -d '{"query": "refund this payment"}' http://127.0.0.1:8765/v1/search
curl -s -H "Authorization: Bearer $TOOLRANK_API_KEY" http://127.0.0.1:8765/v1/metrics      # Prometheus text (Faz 2 week 6)
# end to end on the Mac, embeddings from the GB10: MCP over HTTP and stdio, REST, usage log, cold start
uv run python scripts/serve_e2e.py --data data/w3 --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --out results/serve_e2e_w3.json

# Faz 1 week 4: toolrank as Claude's / OpenAI's tool search (serve running; keys from the env or .env)
uv run python examples/anthropic_tool_reference.py "What time is it in Tokyo?" [--builtin bm25] [--yes]
uv run python examples/openai_client_tool_search.py "What time is it in Tokyo?" [--model gpt-5.5]
# catalogue as the APIs get it (bytes, names, meta-schema), searches, calls; --live: both APIs, 3 tasks
uv run python scripts/platforms_e2e.py --data data/w3 --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz [--live] --out results/platforms_e2e.json

# Faz 1 week 5: the framework adapters end to end with scripted models (bigtool runs in a
# `uv run --with 'langgraph<1'` env, the LiteLLM proxy through uvx: litellm pins openai<3)
uv run python scripts/frameworks_e2e.py --data data/w3 --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --out results/frameworks_e2e.json [--only langchain]

# Faz 1 week 6: packages, images, docs (nothing is published; see Working agreements)
uv build --out-dir dist/pypi && uvx twine check --strict dist/pypi/*     # never publish dist/* (heads live there)
uv run --group docs mkdocs serve                                         # the docs site; CI builds it --strict
uv run python scripts/cli_reference.py --write && uv run python scripts/third_party.py --write
uv run python scripts/release_check.py [--tag v0.1.0]                    # what still blocks a release
# on the GB10 (arm64): both images, then a GPU-free smoke test against a fake embedding server
docker build -f deploy/docker/Dockerfile --build-context heads=dist/heads -t toolrank:dev .
docker build -f deploy/docker/Dockerfile.vllm -t toolrank-vllm:dev .    # no heads context: the build pulls them
uv run python scripts/container_smoke.py toolrank:dev --version 0.1.0
uv run python scripts/container_smoke.py toolrank-vllm:dev --bundle     # a fake vllm inside: the entrypoint, who runs what
# FP8 backbone (profile fp8, port 8094, served as qwen3-emb-fp8); readme_results.sh takes EMB_URL,
# EMB_MODEL, TAG, ROWS, SETS; fp8_agreement.py compares two endpoints, latency.py times .npz heads
docker compose -f deploy/spark/compose.yaml --profile fp8 up -d qwen3-embedding-8b-fp8
# Backlog D1.2 / D1.3: GGUF builds and small Qwen3-Embedding models served by Ollama (docs/guides/local.md), scored like
# the README's rows (restartable, results/gguf_<set>_<model>.json); run where the data is: the laptop has ToolRet + LiveMCPBench
export LAPTOP=…; export EMB_URL=http://$LAPTOP:11434/v1 UV=~/.local/bin/uv; bash scripts/gguf_matrix.sh   # SETS=toolret MODELS="…"

# Faz 2 week 1: learn from what serve logged (no request text leaves the machine; --dry-run needs no torch)
toolrank learn --data data/mytools [--dev data/livemcpbench_server] [--since 2026-10-01] [--tenant NAME] [--dry-run]
# Faz 2 week 3: learn writes DATA/heads/candidate.npz, a running serve gives it --candidate-share (0.1) of the
# requests, ab reads the log back and promotes it to current.npz or sets it aside; every night (--replay stays
# out until learn_sim measures it on the v0.2 backbone: GB10 queue 4):
toolrank ab --data data/mytools [--dry-run] && toolrank learn --data data/mytools
# Faz 2 week 4: the loop on simulated traffic (GB10): a benchmark as the served catalogue, 70% of its queries
# logged by an agent that calls the gold tools shown, learn on that log, eval on the 30% never served;
# NAME/BENCH/GUARD/EVALS/SIZES/NOISE/SEED/TAG/LEARN in the script's header; LEARN="--device cpu" when the GPU is full
bash scripts/learn_sim.sh                                               # ToolRet, sizes 100..all -> results/sim_toolret_*.json
# Faz 2 week 5: a server vote, co-use partners and the no-tool gate (all off by default)
toolrank eval --data data/mcp_zero_server --scorer clm ... --server-weight 0.2    # tool cosine + 0.2 * its server's
toolrank serve --data data/mytools --server-weight 0.2 --co-use 2 [--cut-margin 0.2 --cut-threshold T --cut-min 0]
uv run python scripts/routing_sweep.py [--gate] -- <eval flags>         # rank once: server rules (and the gate table)
uv run python scripts/couse_sweep.py --log RUN/usage -- <eval flags> --cut-margin 0.2   # co-use partners vs a longer list

# Backlog D2.14: what a best score means on this catalogue; searches then carry a confidence
toolrank data gen-queries --data data/mytools --out data/mytools_requests --n 200
toolrank calibrate --data data/mytools --requests data/mytools_requests [search flags] [--dry-run]
toolrank serve --data data/mytools --min-confidence 0.05          # turn away ~5% of answerable requests
# Faz 2 week 6: the Helm chart (deploy/helm/toolrank): render tests run where helm is installed (not CI);
# a real install without a GPU on kind (Docker on the Mac), the published image or one built here
kind create cluster --name toolrank && uv run python scripts/helm_smoke.py [--image toolrank:dev --tenants]

# Faz 1 week 7: launch (every public step after 19:00 and approved one by one)
uv run --with huggingface_hub python scripts/publish_heads.py --repo USER/NAME [--upload]   # dry run without --upload
uv run python scripts/leaderboard_table.py --row NAME W_INST.json WO_INST.json PARAMS TYPE ... [--latex]
uv run python scripts/launch_metrics.py --since YYYY-MM-DD [--discussions] [--out results/launch_metrics.json]
gh workflow run release.yml -R OWNER/REPO                               # a rehearsal: builds everything, publishes nothing
```

## Environment

- The "Spark" in the plans is the GB10 (ASUS Ascent GX10): SSH alias `gb10`, Tailscale IP
  in `$GB10` (the address and the machine's host name stay out of the repo, in the untracked
  `CLAUDE.local.md`; commands write `http://$GB10:PORT/v1`, which must expand to the same literal
  address every time, since the embedding cache is keyed by the URL). MagicDNS is off: the Tailscale
  names don't resolve from the Mac, the `.local` name does on the LAN. Ubuntu 24.04 aarch64, CUDA 13, 121 GB unified memory shared with other projects'
  containers. Git stays on the Mac; `~/toolrank` on the GB10 is a run-only copy (sync commands
  above). uv there is `~/.local/bin/uv`, not on the non-interactive SSH PATH. Never pipe data
  into `ssh gb10 'bash -s'` that also has a heredoc: zsh (MULTIOS) feeds both to stdin and the
  piped bytes run as shell commands. Sync with the one-liner above, run scripts in a separate call.
- vLLM pooling servers: port 8090 = `Qwen/Qwen3-8B` (`--runner pooling --max-model-len 2048`,
  last-token pooling; this is the CLM backbone), port 8091 = `Qwen/Qwen3-Embedding-8B` as `qwen3-emb` (not
  toolrank's default `toolrank-emb-v0.2`: search and serve against it need `--emb-model qwen3-emb`). The GB10
  has no host vLLM: they run from the NGC image via `deploy/spark/compose.yaml` (the systemd
  units are for hosts with one). `--emb-model` is the served name (`qwen3-8b`, `qwen3-emb`),
  not the HF repo id. Keep `--no-enable-chunked-prefill --max-num-batched-tokens 8192`: on vLLM
  0.13, chunked prefill hangs a pooling engine when a long prompt is split next to short ones.
- CLM heads checkpoint, on the machine that runs the eval: `~/.cache/clm/CLM_v0.1-8B.pt` or
  `CLM_CKPT=<path>` (https://huggingface.co/Contrastive-LM/CLM-v0.1-8B). Only the
  ~20M-parameter heads run in-process; CPU is fine. On the GB10 it is in place and the heads run
  on its GPU (the venv has `[clm]`; torch 2.14.0+cu130 handles sm_121).
- Embedding cache: `.cache/toolrank/embeddings.sqlite`, relative to the working directory (run
  from the repo root, i.e. `~/toolrank` on the GB10; `ingest`, `search` and `serve` use the ingest
  dir's `cache/` instead), keyed by the literal `--emb-url` + model
  + `--truncate` + text. Keep it: deleting it means re-encoding 44k tools. Different tool
  formats are different keys, and so are `http://127.0.0.1:8090/v1` and
  `http://$GB10:8090/v1`. It is also what keeps numbers reproducible: vLLM vectors are
  not bit-reproducible (the same batch re-encoded differs by up to ~3e-3 per component).
- vLLM 0.13's `truncate_prompt_tokens` keeps the first N tokens of a pooling request (measured
  on 8091 with `scripts/truncation_side.py`: cos 0.99 to the head, 0.59 to the tail).
- `OpenAIEmbeddings` retries HTTP 4xx like network errors, then raises "endpoint … unreachable:
  HTTP Error 400" without the response body; re-send one request with `curl`
  (`deploy/spark/README.md`) to see it. Usual causes: wrong `--emb-model`, input longer than
  `--max-model-len` without `--truncate`, or an old vLLM rejecting `truncate_prompt_tokens`.
- Hugging Face Hub is needed exactly once per machine (`toolrank data pull toolret`); ToolRet
  lives on the GB10 in `~/toolrank/data/toolret`.

## Architecture

- **Eval path.** `cli.cmd_eval` loads the JSONL pair, blanks every `Query.instruction` unless
  `--with-inst` (w/o inst. is enforced here, not in the scorers) and builds the scorer;
  `eval.runner.run_eval` then calls `index()` once on the full corpus and `rank(batch, k)` per
  `--batch`, and averages per-query metrics per `Query.task` and overall. When queries carry a
  `category` (`data pull toolret` writes web/code/customized), it also reports `per_category`
  (mean over each category's tasks) and `category_macro` (mean over categories). The saved
  `EvalReport` JSON is the only input to `toolrank compare`, which shows `category_macro` as the
  `cat-macro` column (`—` for runs without categories).
- **Scorers compose.** `BM25Scorer` stands alone. `DenseScorer` = one `TextEncoder` + tool and
  query formats + optional projections, exact inner-product top-k over L2-normalised rows
  (`topk_dot`, no ANN). `CLMScorer` is a `DenseScorer` whose projections are the heads (action
  head on tools, state head on queries). The plan's `clm-raw` ablation is `--scorer dense` on
  port 8090, not a new scorer: give it the CLM run's `--emb-url`/`--emb-model`/`--truncate`
  (and `--query-format clm` with `--with-inst`) so it sees the same text and hits the cache.
- **Formats are the ablation axis.** All text shaping lives in `formats.py`; encoders get the
  final string (`OpenAIEmbeddings` uses `kind` only for serve's query timeout), so instruction prefixes exist only as query
  formats. Defaults when the flags are omitted: bm25 `documentation` + `concat`, dense
  `name_desc` + `instruct_query`, clm `name_desc` + `clm`; the query side falls back to `plain`
  without `--with-inst`. The scorer `name` encodes all of it (e.g.
  `clm[CLM_v0.1-8B]/emb/qwen3-8b/name_desc/clm`) and is the row label in `toolrank compare`.
- **Latency** is the `rank()` step only, per query (batch time / batch size); indexing is
  reported as `index_s`. Query embeddings are cached too, so re-running a dense/CLM eval gives
  warm-cache latency without the state encode; `--cache-dir ''` disables the cache entirely
  (the corpus is re-encoded as well).
- **New benchmark** = a converter in `datasets/` writing the JSONL pair (`datasets/jsonl.py`)
  plus a `data pull` choice and branch in `cmd_data_pull`; eval never touches the Hub. A
  benchmark without queries (MCP-Zero) gets them from the `ChatModel` port
  (`adapters/chat_api.py`, any OpenAI-compatible chat endpoint); responses are cached in the data
  dir (`generations.jsonl`, keyed by model + prompt), so re-converting never regenerates.
  `Tool.doc` falls back to parsing `documentation` (JSON or Python repr); `Tool.category` is the
  ToolRet category or the MCP server name.
- **Ingestion** (`ingest/`, Faz 1): a `ToolSource` (`ports.py`: `name`, `kind`, `list_tools()`)
  is an MCP server (`adapters/mcp_client.py` over the `mcp` 2.x SDK's `mcp.Client`; async core
  `fetch_tools`, servers listed concurrently, one timeout each) or an OpenAPI 3.x spec
  (`ingest/openapi.py`, pure: one tool per operation, flat `inputSchema` of parameters + body
  properties, local `$ref`s inlined). Every ingested tool is indexed by `ingest.text.tool_text`:
  `{"server", "name", "description", "inputSchema"}` JSON, the Faz 0 ablation shape, shrunk
  schema-first to 6000 characters so truncation never drops the server or name (it is what the
  `documentation` format returns). `Tool.doc` keeps the rest (MCP annotations; OpenAPI routing in
  `doc["http"]`, never under `parameters`, which `formats._params` reads first). Tool ids are
  `<source>/<name>` and never parsed (GitHub names contain `/`): `category` is the source.
  `ingest/sync.py` replaces exactly the listed sources' tools, keeps failed ones, guards kind
  changes and empty listings, and writes `sources.json` (no secrets). Unchanged tools are never
  re-embedded because the embedding cache is keyed by text; `ingest --emb-url` embeds only cache
  misses, into `DIR/cache` with search's and serve's model and truncation (toolrank-emb-v0.2, 8192), so
  their first index is all cache hits (1,862 tools: 154 s cold over Tailscale otherwise).
- **Vector indexes** (Faz 1 week 2): `DenseScorer` keeps tool vectors in a `VectorIndex`
  (`ports.py`; `hashes`, one atomic `apply`, `search`): `NumpyIndex` (default; exact `topk_dot`, rows
  in insertion order, optional single-`.npz` snapshot under `flock`), `FaissIndex` (HNSW rebuilt on
  every apply, graph stored in the snapshot), `PgVectorIndex` (exact scan; pgvector's HNSW stops
  below 4096 dims). Rows are keyed by sha256(fingerprint ‖ tool text), fingerprint = endpoint +
  model + truncate + tool format + heads file: a persistent index re-embeds only new/changed tools.
  A fresh in-memory index reproduces Faz 0 numbers exactly; non-numpy indexes add `@<name>` to the
  scorer name.
- **Hybrid and adaptive K**: `HybridScorer` (RRF k 60, depth 100, optional BM25 weight; BM25 arm
  without instruction, 0-score hits dropped) is opt-in: it helps only MCP-Zero's agent-written
  requests and costs 8–12 points on ToolRet/LiveMCPBench. `cut.AdaptiveK` (default margin 0.2, max
  10, chosen on ToolRet) keeps the tools within the margin of the best cosine; on hybrid lists the
  count comes from `last_semantic`. Eval reports it as `K@cut`, `Recall@cut`, `Precision@cut`,
  `Comprehensiveness@cut` next to the untouched fixed-k metrics.
- **Server vote and co-use** (Faz 2 week 5, both opt-in): `DenseScorer(server_weight=W)` (`--server-weight`)
  embeds each server (`Tool.category`) as `formats.server_summary` (name + tool names, 4,000 characters) and
  re-scores the index's top 100 as cosine + W × the request's cosine with the tool's server; scores are then no
  longer plain cosines (the name ends in `+srv<W>`, adaptive K cuts on them as they are), one server = no-op.
  Picking servers first (MCP-Zero's pattern) loses everywhere. `couse.py`: `co_use(events)` counts, per request
  (`emb_hmac`), the tools whose calls ended `ok` together; `serve --co-use N` keeps a `CoUseTable` of the last
  30 daily log files (rebuilt in the background every 5 minutes, all API keys together) and `Retriever.search`
  appends up to N partners of the shown tools (`Hit.used_with`, `SearchResult.added`; never for a request's
  own `k`). The log keeps them apart: `shown` is what the ranking returned, `added` the partners; `learn.mine`
  counts both as shown. A "no tool fits" gate is `--cut-threshold T --cut-min 0` (an empty list plus a note in
  `search_tools`); there is no default T, the best cosine does not separate well (`docs/reports/faz2-week5.md`).
- **Default backbone** (Faz 2 week 7): `build.BACKBONES` maps served names to weights and whether the packaged
  heads belong on them; `DEFAULT_EMB_MODEL = "toolrank-emb-v0.2"` (the LoRA-merged Qwen3-Embedding-8B,
  `BACKBONE_REPO` @ `BACKBONE_REVISION`; the version is in the name because the embedding cache is keyed by it).
  `search_defaults` loads cached heads only when `packaged_heads_fit(emb_model)` (unknown names: yes, as before;
  `TOOLRANK_HEADS` always), `learn.resolve_init("default", emb_model)` follows the same rule. The Ollama names of the
  GGUF builds (`toolrank-emb-v0.2-q4_k_m`, `-q8_0`; `BACKBONE_GGUF_REPO`, not on the Hub yet) and of the small
  Qwen3-Embedding models are listed too, heads off (`docs/guides/local.md`). The weights are on the
  Hub since 2 Oct 2026 (0.2.0; `BACKBONE_PUBLISHED = True`, else `release_check` refuses a release);
  `scripts/publish_backbone.py` (run on the GB10, dry run without `--upload`) uploads a merged directory, tags the
  revision and flips the flag. New weights get a new tag and served name (the cache is keyed by the name). `entrypoint-vllm.sh`, `deploy/docker/compose.yaml` and the chart's `embedding.backbone` carry the same
  names (a test ties the entrypoint to `build`). finetune keeps `qwen3-emb` as its default.
- **Second stage while serving** (Faz 2 week 7): `search` / `serve --rerank cross|jev` (`cli._add_serve_rerank_args`:
  depth 20, documentation cut to 3,000 characters, Qwen3-Reranker as `qwen3-reranker`; Jev reads the same text)
  goes through the same `scorer_factory` wrapping as eval. Both rerankers have `rank_pairs` (reranked, first
  stage's cosine list) that writes no shared state (`last_base` is eval's), so the retriever's adaptive cut
  counts on the cosines and shows the reranked order. Jev in serve sends request text out: logged at start.
  `Retriever.search` / `rank` `tidy` the request and an own instruction (NFC, blank runs, ends), never eval;
  `retriever.second` is the second stage's client, whose `scored()` feeds `toolrank_rerank_candidates_total`.
  The retriever sets `ports.rerank_failures` around a search: a failing second stage (`--rerank-timeout` 10 s
  × 2 attempts, `rerank.Slots` = `--rerank-workers` / `--jev-workers` calls in flight, the wait counted)
  answers with the first stage's list (`SearchResult.rerank_error` → log, `note`, metric); eval sets no list
  and raises. `/v1/rank` (`score_tools`) is the first stage's cosines under either reranker.
- **Confidence** (backlog D2.14, `calibration.py`): `toolrank calibrate` ranks gen-queries requests over the
  catalogue with the search flags (`Retriever.settled`: current heads if any, never the candidate) and keeps
  their best first-stage scores (and those of "twins", the gold tools' sources hidden) in `DATA/calibration.json`,
  one entry per (first stage's name, heads sha16, serving instruction). `Retriever.calibration` re-reads the file
  when it changes; a search's `confidence` = the share of those scores ≤ its best first-stage score (`semantic or
  fused`), `--min-confidence Q` empties a list below Q unless the request named `k`. REST/MCP `confidence`, the
  log's `confidence`, `toolrank_search_confidence`.
- **Tenants** (Faz 2 week 6, `tenants.py`): `--api-keys` entries are a key string or `{key, sources, headers,
  env}`. One shared catalogue and index; `Retriever(allowed={tenant: sources})` sets `ports.visible_ids` and the
  first-stage scorers rank within it (`within_visible`: deeper, ×4 at a time), so fusion and a second stage see
  only the key's tools; a named key's request caches are its own (`ports.cache_scope`); `get(id, tenant)` and
  `catalogue(tenant)`, so a tool outside a key's sources is indistinguishable from a missing one (REST 404, MCP
  `unknown_tool`, never sent). `Backends(tenants=...)`: a key's headers join the config's on its OpenAPI calls
  (configured `base_url` only), and an MCP server it has headers or env for gets a connection of the key's own
  (`backend_for`). The OpenAPI client keeps no cookies (a cookie jar would have carried one caller's cookie to
  the next). Co-use tables and the token estimate are per tenant; `/v1/metrics` refuses keys with `sources`.
- **Metrics** (Faz 2 week 6, `GET /v1/metrics`, Prometheus text 0.0.4, under `Guard` like the rest of `/v1`):
  `UsageLog` owns a `metrics.Metrics` and feeds it from `search()` and `call()`, so MCP and REST are counted
  alike and `--no-usage-log` stops the files, not the counters; a failing counter never fails a request. The
  route adds what it reads off the retriever at scrape time (catalogue size, index readiness, heads variants,
  `retriever.encoder`'s cache-hit counts). Token estimate: `tool_tokens` = name + description + input schema as
  JSON at 4 characters a token; a search's saving is the catalogue's total minus what it returned, and the
  catalogue is sized in a background thread (searches meanwhile claim nothing). No request text, arguments or
  key names in labels (`arm_kind`); the counters are server-wide and reset with the process.
- **Jev** (TypeSafe AI's "System One" model, `adapters/jev.py`): no text, one Choice question over
  up to 255 options returns a probability per option. `--rerank jev` wraps any scorer (BM25, dense,
  clm, hybrid): the base top `--rerank-depth` becomes one Choice per query, probabilities are the
  scores, ties keep the base order, the base list continues below the depth, so top-100 metrics
  stay complete. `--scorer jev` is Jev alone: chunks of `--jev-chunk` tools, the chunk winners
  re-ranked once (feasible for the MCP sets, not ToolRet's 44k). State = `{"request"}`, the
  benchmark instruction leads the question, option text = `--jev-tool-format` cut to
  `--jev-max-chars` (32k tokens for state + longest question). `TYPESAFE_API_KEY` goes to `api.typesafe.ai`
  only (`jev.is_typesafe`); any other `--jev-url` (a self-hosted `/systemone`) is asked without a key and
  cached in its own `jev-<host>.sqlite` with the URL in the key (the cross-encoder's cache keys the URL too).
  Pinned
  `jev-1.13.0` (aliases move), answers cached by request body so a rerun ranks the same for free;
  the report's `config["jev"]` has calls, cached hits, billed tokens and per-call p50. MCA 2.3(b)
  forbids training on its output or building a competing product with it: eval and an optional
  adapter only, never a training signal (`usage.may_learn_from`: learn, ab and co-use skip and count
  Jev-served searches). The same seat for local models: `--rerank dense|clm|cross`
  (`adapters/rerank.py`, the second scorer's own `--rerank-*` flags, `--rerank-max-chars` = Jev's
  text cut) and `adapters/cross_encoder.py` (vLLM `/score`; the request is in every pair, so it is
  cut to `--rerank-query-chars` and the rerankers serve an 8192-token window). CLM in that seat
  breaks the list (ToolRet 54 → 15); the cross-encoders are the real local candidates.
- **Packaged heads**: `NumpyHeads` reads `.npz` checkpoints (`allow_pickle=False`) and runs
  `make_head`'s forward in numpy; the `.npz` `cfg` carries serving defaults (backbone, formats,
  truncate, instruction) that `build.py` applies; `build.heads_mismatch` compares the cfg's backbone with
  the served name's (same width, so nothing else notices): search and serve stop when `--emb-model` was
  defaulted, warn when it was named, eval warns. `--clm-ckpt` takes `.pt`, `.npz`, `default` or (search / serve)
  `none`, which drops the packaged heads only (a learned `DATA/heads/current.npz` and the candidate still serve)
  (`TOOLRANK_HEADS`, `~/.cache/toolrank/heads/`, else a sha256-checked download from `HEADS_URL`). The artifact and its model card: `dist/heads/` (gitignored),
  `docs/heads/MODEL_CARD.md` (training data has no license; the maintainers accepted that).
- **Serving** (Faz 1 week 3, `toolrank serve`): `retriever.Retriever` holds one immutable state
  (tools, id map, indexed scorer) and swaps it whole when `tools.jsonl` changes (mtime + size,
  checked per search; a fresh scorer from `build.scorer_factory` shares encoder and heads), so
  searches run in threads without locks; `tools.jsonl` is read once it has been unchanged for
  50 ms. The first index builds in the background with a BM25 stand-in built alongside: until the
  semantic index is up, searches get keyword matches (`mode: lexical`, a note in the MCP result),
  `call_tool` works and `/v1/rank` fails fast (503); a failed first build is retried after 30 s on
  a request. `adapters/mcp_proxy.py` is the SDK's low-level
  `Server` with two tools, `search_tools` (full `inputSchema` for the first 3 hits, shrunk after)
  and `call_tool` → `adapters/backends.py`: one `mcp.Client` per MCP server, opened lazily and
  reopened after a crash, an in-flight call never resent; OpenAPI operations over `httpx2`,
  GET/HEAD unless `--allow-write`, config headers only to the config's `base_url`, no redirects.
  The server's lifespan owns the backends, so no stdio child outlives it. `adapters/rest.py` adds
  `/v1/*` to the same Starlette app, `/v1/call` included (`mcp_proxy.dispatch_call`, shared with
  MCP `call_tool`; JSON only, a failing tool is a 200 with `isError`); `Guard` puts the bearer token
  on `/mcp` and `/v1` and the Host, Origin (403: a web page must not reach a tokenless loopback
  server) and 1 MiB body checks on `/v1` (the SDK covers `/mcp`); `--api-keys FILE` adds named keys,
  whose name becomes the request's tenant. OpenAPI path parameters `.`/`..` are refused, and so is a spec path without its leading `/` (appended to the base URL it names another host; `backends.same_origin` checks the composed URL against the base URL as well). `usage.UsageLog` appends schema-v3 JSONL to
  `DATA/usage/` (one `os.write` per event), requests, a request's own instruction, arguments and the
  embedding-cache key (`emb_hmac`; the key itself is an unkeyed hash that would confirm a guess) as HMAC digests under a
  per-install key (`usage.read_key`: written whole, refused under 32 bytes; an unknown tool name is a digest too), each
  call linked to a search of the same key: `search_id`, else the session's latest, else the same client's latest
  (`client` = key name | client app | remote host, logged as a digest; a named key's REST session is `rest@<name>:<id>`).
  `/healthz` (no token, no Host check) says `{ready, mode}` only, and client-facing errors leave out exception text
  (the `toolrank.serve` log keeps it). `/v1/rank`'s supplied tools (`ports.supplied`) get query limits, an in-memory
  cache and two worker slots of their own.
  2026-07-28 HTTP clients have no session id (`session: null`); `client` carries their links.
  Serve gives query embeddings 10 s and one retry; indexing keeps the encoder's 600 s. Under
  `--stdio` stdout is the protocol: log to stderr only. Paths are made absolute (Claude Desktop
  starts servers in `/`); cache and index default to `DATA/cache` and `DATA/index`.
- **Platform integrations** (Faz 1 week 4): `names.api_name(id)` is a tool's name on the agent APIs
  (`^[A-Za-z0-9_-]{1,64}$`; readable ids keep their shape with `/` as `__`, others are cut and get
  `___` + 12 hex of sha256; injective, and a pure function of the id), carried on REST only;
  `/v1/tools?full=true` is the platform-ready catalogue. `client.ToolrankClient` (stdlib, no
  redirects, never retries a call) talks to REST. `integrations/anthropic.py`: every tool deferred
  (`defer_loading`), `search_tools` answered with `tool_reference` blocks naming only tools of the
  snapshot sent (an unknown reference is a 400), tool list frozen per conversation, history
  append-only, every `tool_use` answered in one message, nothing run after `max_tokens`/`refusal`,
  `pause_turn` resent; `builtin="bm25"` uses the API's own search. `integrations/openai.py`:
  only `tool_search` (`execution: "client"`) declared, each `tool_search_call` answered with the new
  tools' full `function` definitions, stateless (`store=False` + encrypted reasoning, all items
  resent); `Toolbox(namespaces=True)`: one `namespace` per server, tools under `own_name` (a call names
  both; untested live). Neither imports the SDKs; tests run the real SDKs over `httpx2.MockTransport` against
  toolrank in process. Examples ask before non-read-only calls unless `--yes`.
- **Framework adapters** (Faz 1 week 5, `integrations/`): `langgraph.Toolbox` = the catalogue as
  LangChain `BaseTool`s whose `_run(**arguments)` takes nothing else (LangChain injects `config` /
  `run_manager` by signature) plus `retrieve_tools(query) -> list[str]`, registry keys only (what
  langgraph-bigtool's `retrieve_tools_function` wants; its docstring is the model's description),
  sessions from the LangGraph `thread_id`. `llamaindex.ToolrankToolRetriever` (an `ObjectRetriever`;
  LlamaIndex 0.14 agents call `aretrieve(user message)` every step and `aretrieve(tool name)` before
  a call): searches kept per query for a TTL, handed-out tools found by name, `fn_schema` a pydantic
  model serving the `inputSchema`, descriptions ≤ 1,024. `litellm.tool_filter` (PEP 562: litellm is
  imported only when the proxy loads it; no extra, litellm pins `openai<3`): a `CustomLogger` whose
  `async_pre_call_hook` must be defined on the class itself (the proxy skips inherited hooks), ranks
  the request's own function tools with `/v1/rank`, fails open, never edits `data` in place.
  `langchain.ToolrankToolSelector` (backlog D1.4; PEP 562 like litellm, `Selector` is the work without
  LangChain's class): a LangChain 1.x `AgentMiddleware` whose `wrap_model_call` / `awrap_model_call`
  narrow `request.tools` (`request.override`) for the last human message: the agent's own tools via
  `/v1/rank` + adaptive K, or with `toolbox=` the catalogue's tools via `Toolbox.retrieve_tools` (logged
  searches, calls linked; the agent's other tools pass). Keeps called / `tool_choice` / `always_include` /
  dict tools, passes deferred tools (`extras["defer_loading"]`) through, fails open after `timeout_s`; the
  work runs in a pool thread under `contextvars.copy_context()` (the LangGraph thread id is a context var)
  and its result is cached for `ttl_s` per (session, query) or (instruction, query, tool names), so a
  late answer serves the next call and one turn's model calls ask once.
- **Packaging and images** (Faz 1 week 6): the version lives only in `toolrank/__init__.py` (hatch reads
  it); PEP 639 license metadata with LICENSE and NOTICE; the sdist is `src`, `tests` and the top-level
  files. Missing extras print their install line (`build.require`, `cli._need_mcp`). `serve` on
  `0.0.0.0` also answers to the loopback names; an allowed name without a port also matches a
  portless Host (a proxy on 80/443); `TOOLRANK_ALLOWED_HOSTS`, `TOOLRANK_EMB_URL`, `TOOLRANK_EMB_MODEL`
  configure containers (search, serve and the ingest warm-up read them, eval does not).
  `deploy/docker/Dockerfile` (`toolrank`, 505 MB): the `uv.lock` versions of `[mcp,openapi,stem]` on
  python:3.12-slim with Node.js, uv and tini, non-root, heads from an optional `heads` build context
  (only `toolrank-heads-*.npz` is taken; the stage bind-mounts the context, because BuildKit reused a
  cached `COPY --from=heads` across builds with and without it). `Dockerfile.vllm` (`toolrank-vllm`,
  22.6 GB, not published: it does not fit two platforms on a free runner; compose.bundle.yaml builds
  it, and without a heads context the build runs `toolrank heads pull`): the same in a
  venv of its own on `vllm/vllm-openai:v0.30.0`; `entrypoint-vllm.sh` starts vLLM on loopback
  (`yasinyaman/toolrank-emb-8b@v0.2` in FP8 as `toolrank-emb-v0.2-fp8` by default, bf16 as
  `toolrank-emb-v0.2`; `TOOLRANK_BACKBONE=Qwen/Qwen3-Embedding-8B` gives `qwen3-emb[-fp8]`: the cache is keyed
  by the name, not the dtype),
  waits, runs toolrank, and exits when either does. vLLM stays root; `toolrank` on that image's PATH is
  `as-toolrank.sh` around the real command, so toolrank (and the npx/uvx MCP servers it starts) never
  runs as root, from the entrypoint or from `docker exec`: it runs as the owner of the directory the
  command writes to (`--out`; `--data` for serve and search; else `/data`) when that is not root (a bind
  mount stays its owner's; never the root group; homes under `/home`, which the user cannot relink), else
  as the image's `toolrank` user, and only root's files on the `/data` mount are re-owned (`chown -h`,
  `-xdev`; a failure, like a home that cannot be made, is a warning). Under a rootless engine a root-owned
  directory is the engine user's: no drop, no chown. Both images install with
  `--require-hashes` (the vLLM base's `UV_OVERRIDE` is unset for that) and name their base images once,
  by tag and digest, on `FROM` lines (Dependabot reads nothing else); the `*_IMAGE` build arguments
  default to those pinned stages. Workflow actions are pinned to commits, and a test keeps them so.
  Compose projects are `toolrank-stack` and
  `toolrank-bundled`, never `toolrank` (that name is `deploy/spark`'s benchmark servers).
  `release.yml` on a `v*` tag: checks, the `toolrank` image to GHCR, then PyPI (it cannot be taken
  back, so it waits for the image), then the GitHub release; by hand (`workflow_dispatch`) it is a
  rehearsal that publishes nothing. Endpoint keys: `TOOLRANK_EMB_API_KEY` / `TOOLRANK_CHAT_API_KEY` go
  anywhere, `OPENAI_API_KEY` only to https://api.openai.com, and none follows a redirect.
- **Learning from the usage log** (Faz 2 week 1, `toolrank learn` → `learn.run`): the log holds no request
  text, so `learn.mine` turns searches with linked calls into pairs of tool ids (`ok` → positive,
  `tool_error` → weak positive unless `--strict`, shown-but-never-called → hard negative; other outcomes say
  nothing; the same request merges), `learn.state_vectors` finds each request's backbone vector by digesting
  the embedding cache's keys with `DATA/usage/.key` and matching `emb_hmac`, and the tools' vectors come from
  `tools.jsonl` through the same cache. `mine` keeps only the served model's searches (the log's `model`,
  else the first stage's `emb/<model>/` in the scorer name; others counted as `searches_of_other_models`) and
  no Jev-served ones (`usage.may_learn_from`). Training is `finetune.train_heads` (lr 1e-5,
  5 shown negatives, `neg_filter` 0.95) from `resolve_init("default")`: what a server would serve —
  `DATA/heads/current.npz` (the tenant's own with `--tenant`) if its cfg names this backbone, else the
  packaged heads where they fit, else fresh skip heads (identity: epoch 0 is the backbone alone, the v0.2
  case); the newest 20% of requests are the dev set (`log.Recall@5`: the called
  tool in the catalogue's top 5), an optional `--dev` benchmark set is a guard, and the heads are written
  (`DATA/heads/candidate.npz`, `tenants/<name>/` with `--tenant`) only when epoch > 0 beats the start on the
  log without losing more than `--max-drop` NDCG@10 points on the benchmark; `--dry-run` mines without torch;
  `--replay pairs.jsonl` mixes general pairs into the batches against forgetting: a seeded reservoir sample of
  the whole file (`--replay-n`, 1000; `--data-seed`), positives only, requests a `--dev` set asks about dropped
  (selection stays on the log; not in the nightly command until learn_sim measures it on v0.2). `serve --mask-pii` (with `--log-text`) tags e-mail, phone, card and IBAN numbers before they are written.
- **Heads that change while serving** (Faz 2 week 3): `Retriever.pick` answers a request with a variant, a
  state built from a heads file under `DATA/heads` by `build_retriever`'s `variant` factory (same encoder
  flags, index snapshot under `index/variants/<name>`): `current.npz` replaces the flags' heads (not when
  `--clm-ckpt` named some), `candidate.npz` takes a sticky `--candidate-share` (`retriever.bucket` of the
  session, else the client), `tenants/<name>/` does both for one API key. Variants build in the background
  when the file appears or changes (mtime + size + the tools stamp), a request never waits for one, a file
  that cannot be loaded leaves an `error` in `status()["heads"]`. `SearchResult.arm` / `heads` go to the
  log. `toolrank ab` (`learn.judge`, `decide`, `apply`): per arm since the candidate appeared, searches,
  called, top-1 and `mrr` (mean 1/rank of the called tool over all the arm's searches); promote
  (`candidate` → `current`, the old one kept as `previous-<stamp>`), roll back (`rejected-<stamp>`) or wait
  (`--min-searches` 100 a side, `--margin` 0.01); renames only, which the running server follows.
- **Head fine-tuning** (`toolrank finetune` → `finetune.run`): the backbone stays frozen and training
  reads only cached vectors (one command embeds what the cache lacks, then trains). Training requests
  equal to a dev or eval query are dropped (counted per source); `split_pairs` takes a seeded
  train/val split (val pairs are a diagnostic). The epoch is picked on `--dev` (`EvalSet`, scored
  exactly like `toolrank eval`, a test pins that), which must not be an eval set nor share queries
  with one; the saved heads carry the serving cfg and `selected_on`, and `toolrank eval` then runs
  on the saved file for dev and every eval set. Faz 0 picked on val pairs; its runs are the parity
  reference (`results/finetune_qwen_60k_skip_neg0.json`). Heads are saved in the checkpoint shape `CLMHeads`
  loads (`state_head`, `action_head`, `logit_scale`, `cfg`), so `toolrank eval --clm-ckpt` scores
  them on any backbone; `cfg["skip"]` (x + MLP(x), identity at init) is for heads on top of an
  embedding model, which must not start below its zero-shot quality.

## Map

```
src/toolrank/domain.py            Tool, Query, TrainPair, RankedList, EvalReport
src/toolrank/ports.py             Scorer, TextEncoder, ToolFormatter, QueryFormatter, VectorIndex, ToolSource, ChatModel
src/toolrank/formats.py           tool formats documentation|name_desc|schema|example_call; query formats plain|concat|instruct_query|clm
src/toolrank/adapters/bm25.py     BM25Scorer (bm25s, English stopwords, Snowball stemming if PyStemmer)
src/toolrank/adapters/embeddings_api.py  OpenAIEmbeddings (+ EmbeddingCache, SQLite)
src/toolrank/adapters/dense.py    DenseScorer (+ row_hash; tool vectors in a VectorIndex), topk_dot
src/toolrank/adapters/index_numpy.py, index_faiss.py, index_pgvector.py   the VectorIndex adapters
src/toolrank/adapters/hybrid.py   HybridScorer (RRF)
src/toolrank/adapters/jev.py      JevClient (+ SQLite answer cache), JevReranker (Jev over a scorer's top K), JevScorer (Jev alone, chunked)
src/toolrank/adapters/rerank.py   ScorerReranker (a second dense/clm/cross scorer over a scorer's top K), cut_formatter
src/toolrank/adapters/cross_encoder.py  CrossEncoderScorer + ScoreClient (vLLM /score; qwen3 and bge prompt templates, SQLite cache)
src/toolrank/adapters/clm.py      CLMHeads (mirrors clm/heads.py; torch), CLMScorer
src/toolrank/adapters/heads_np.py NumpyHeads, load_heads, export_npz, default_heads, download
src/toolrank/adapters/chat_api.py OpenAIChat (query generation only, never ranking)
src/toolrank/adapters/mcp_client.py  fetch_tools / fetch_many / MCPServerSource (mcp SDK, stdio + streamable HTTP)
src/toolrank/adapters/backends.py MCPBackend, OpenAPIExecutor, Backends (call_tool's router)
src/toolrank/adapters/mcp_proxy.py  build_proxy (search_tools + call_tool), serve_stdio, Guard, http_app
src/toolrank/adapters/rest.py     rest_routes (/v1/search, /v1/rank, /v1/call, /v1/tools, /openapi.json, /healthz), platform_record, OPENAPI
src/toolrank/retriever.py         Retriever (state swap, background first index; pick: heads variants current/candidate/tenant), bucket, Hit, SearchResult
src/toolrank/usage.py             UsageLog (schema v3: search and call events, HMAC digests, client key, call → search links), read_events
src/toolrank/couse.py             co_use (log -> tool partners), partners / expand, CoUseTable (a server's table, refreshed in the background)
src/toolrank/tenants.py           Tenant, load_tenants / parse_tenants (--api-keys: sources, headers, env), check_sources
src/toolrank/metrics.py           Metrics (Prometheus counters and histograms of searches and calls, the token estimate), tool_tokens
src/toolrank/names.py             api_name (tool ids as agent-API tool names)
src/toolrank/client.py            ToolrankClient, ToolrankError (REST, stdlib)
src/toolrank/integrations/        anthropic.py, openai.py (Toolbox, run), _common.py (read_only, get);
                                  langgraph.py (Toolbox), langchain.py (ToolrankToolSelector, Selector),
                                  llamaindex.py (ToolrankToolRetriever), litellm.py (tool_filter)
src/toolrank/ingest/              text.py (the indexed text), mcp.py (server configs, MCP tool → Tool), openapi.py, sync.py
src/toolrank/datasets/jsonl.py    the on-disk format (+ pairs.jsonl); toolret.py (pull + task→category map); toolret_train.py;
                                  livemcpbench.py; mcp_zero.py (download + LLM-written queries); synthetic.py;
                                  genqueries.py (data gen-queries: sample tools over sources, LLM-written requests in styles)
src/toolrank/eval/metrics.py      trec_eval-compatible metrics; runner.py (run_eval, summarize, format_table, save_report, --runs-out rows);
                                  paired.py (compare --paired: exact sign test, paired permutation test);
                                  table.py (the README's results table: render, splice, the protocol checks)
src/toolrank/finetune.py          toolrank finetune: Job/run, EvalSet (dev curves), train_heads(select=), load_checkpoint
src/toolrank/learn.py             toolrank learn: mine (log -> pairs of tool ids), state_vectors (emb_hmac -> cache), split, run;
                                  toolrank ab: judge, decide, apply (candidate -> current | rejected), heads_home
src/toolrank/build.py             composition root: scorer_factory, build_scorer, build_retriever, build_index, fingerprint; BACKBONES
src/toolrank/cut.py               AdaptiveK (+ defaults), cutter
src/toolrank/calibration.py       Calibration (confidence = ECDF of answerable requests' best scores), Calibrations, measure
src/toolrank/cli.py               eval | compare | data (pull, server-names, synth, gen-queries) | ingest (mcp, openapi, drop) | search | calibrate | serve | finetune | learn | ab | heads (export, pull) | formats
docs/plan/                        private repo (ignored here): README.md ("Şu an"), faz-0..3.md, backlog.md, claude-code-handoff.md,
                                  and the decision, launch and review notes
docs/reports/                     weekly numbers; TEMPLATE.md
docs/results.toml, docs/results/  the README's results table: its rows and the curated eval reports behind them
docs/heads/MODEL_CARD.md          the packaged heads' card (sha256, serving, data license, numbers)
docs/backbone/MODEL_CARD.md       the default backbone's card (LoRA recipe, selection set, serving, data license, numbers; publish_backbone.py stages it)
deploy/spark/                     vLLM servers: systemd units, compose.yaml (NGC image, GB10; fp8 (8092 CLM, 8094 embedding), gen and pg profiles;
                                  rerank: Qwen3-Reranker-8B 8095 + bge-reranker-v2-gemma 8096 as vLLM score models; lora: the merged LoRA backbone 8097,
                                  served as $TOOLRANK_LORA_NAME; lora-fp8: the same in FP8 on 8098)
deploy/helm/toolrank/             the Helm chart (embedding.mode vllm | external | bundled, profile fp8 | bf16; one replica, Recreate)
deploy/docker/                    Dockerfile (toolrank), Dockerfile.vllm + entrypoint-vllm.sh + as-toolrank.sh (toolrank-vllm), compose.yaml,
                                  compose.bundle.yaml, toolrank.json, .env.example
mkdocs.yml, docs/*.md             the docs site (guides/, reference/ with the generated cli.md); plan/ and reports/ stay off it
.github/                          ci.yml; release.yml and docs.yml (off until TOOLRANK_RELEASE / TOOLRANK_PAGES); issue forms
                                  (bug, feature, benchmark result, feedback); dependabot.yml (actions)
scripts/                          run_matrix.sh; toolret_paper_avg.py; truncation_side.py; adaptive_k_sweep.py; heads_parity.py;
                                  serve_e2e.py (toolrank serve end to end); platforms_e2e.py (the agent APIs, --live);
                                  frameworks_e2e.py (bigtool, LlamaIndex, LiteLLM proxy + MCP gateway);
                                  readme_results.sh (the README's runs), readme_table.py (--write | --check);
                                  cli_reference.py, third_party.py, release_check.py, container_smoke.py, fp8_agreement.py, latency.py;
                                  per_task_diff.py (per-task diffs between two eval reports);
                                  publish_heads.py (the Hub, pinned to a tag), leaderboard_table.py (ToolRet leaderboard sheets),
                                  launch_metrics.py (the Faz 1 gate's numbers);
                                  jev_compare.sh, clm_rerank.sh, cross_rerank.sh (second-stage rows: Jev, CLM, cross-encoders),
                                  lora_train.py (LoRA on the embedding backbone, [lora] extra), rerank_report.py (faz2-rerank.md's tables, --write);
                                  learn_sim.py (split a benchmark, play its queries as logged traffic), learn_sim.sh (learn + eval per traffic size);
                                  routing_sweep.py (server -> tool rules and the no-tool gate), couse_sweep.py (co-use partners on a log);
                                  helm_smoke.py (the chart on a cluster without a GPU: fake embeddings, install, upgrade, uninstall);
                                  publish_backbone.py (the LoRA-merged backbone to the Hub, tagged; dry run by default);
                                  gguf_matrix.sh (GGUF builds and small backbones through Ollama, the README's eval flags)
examples/                         anthropic_tool_reference.py, openai_client_tool_search.py, litellm/config.yaml;
                                  skills/toolrank/ (SKILL.md + stdlib search.py / call.py / _toolrank.py over REST; tests/test_skill.py runs them
                                  as processes against the served app behind a real HTTP server)
```

## Reference numbers (sanity checks, not targets)

- ToolRet paper, NDCG@10 (w/o inst / w/ inst): BM25s 22.32 / 36.46; NV-Embed-v1 33.83 / 42.71;
  gte-Qwen2-1.5B-instruct 28.96 / 45.96; bge-reranker-v2-gemma 35.51 / 47.52. These are
  category macro-averages (plain mean over the three categories of the plain mean over each
  category's tasks): compare them with the `cat-macro` column, not the micro-average. BM25s is
  unstemmed: `--no-stem` gives 22.24 / 36.41 cat-macro; `scripts/toolret_paper_avg.py` adds the
  paper's `str(doc)` tool text and lands within 0.07 per category (`docs/reports/faz0-week1.md`).
- StackOne's ToolRet-full setting: Qwen3-Embedding-8B 0.462, StackOne v2 0.544 (a fine-tuned
  109M BGE-base). **Phase 0 gate: CLM after head fine-tuning ≥ 0.50** in that setting. Their other
  rows are the paper's w/ inst Averages (NV-Embed-v1 0.427 = 42.71, GritLM-7B 0.411 = 41.13), so
  the setting is w/ inst + `cat-macro`
  (https://www.stackone.com/blog/autoresearch-charged-action-search/; `docs/reports/faz0-week2.md`).
- CLM reference latency: 58.1 ms per single question, RTX 4090, cold cache. On the GB10, batch 1
  against all 44k ToolRet tools (w/ inst queries, `scripts/latency.py`): 85.9 ms p50 cold with the
  bf16 backbone, 49.3 ms with FP8 (port 8092, `--profile fp8`), 1.8 ms warm. A single query is
  memory-bandwidth bound, so FP8 nearly halves it; head outputs stay at cosine ≥ 0.997 to bf16.
- Text conventions the CLM heads were trained on: state = `context\n\ninstruction`
  (query format `clm`: request, blank line, instruction), candidates verbatim, backbone rows
  L2-normalised before the heads. If CLM doesn't clearly beat `clm-raw`, check these first.
  End-to-end check: the CLM README's `Engine.rank("What causes tides on Earth?", …)` example
  gives 0.994 for "The Moon's gravitational pull." here (README: 0.997).
- CLM was trained on question → answer pairs and agent (context → action taken) steps, so tool
  descriptions are off its distribution. ToolRet zero-shot, w/ inst cat-macro NDCG@10 (week 2):
  CLM best 4.99 (`example_call`), clm-raw ~0.3, Qwen3-Embedding-8B 46.54 (`documentation`;
  reproduces StackOne's 0.462), BM25 with the paper's settings 36.41. Not a bug: the README
  example reproduces; the gate rests on head fine-tuning.
- Week 3 head fine-tune (60K ToolRet-train pairs, same setting): CLM 4.99 → 17.68 (from
  CLM_v0.1-8B, lr 1e-2; fresh heads do not learn), Qwen3-Embedding-8B + skip heads 46.55 → 47.23
  (lr 1e-5, in-batch negatives only; lr ≥ 3e-4 collapses). The dataset's 15 mined negatives cost
  Qwen3 up to 10 points, and recall on held-out training pairs rises while the benchmark falls,
  so it is no proxy for the gate (`docs/reports/faz0-week3.md`). All 206K pairs: 54.03 micro but
  still 47.14 cat-macro — more data helps the big tasks only.
- LiveMCPBench (w/ inst, Recall@5 micro; the set without server names, and the 60K-pairs heads —
  the README's table uses the server-named set and the 206K heads): Qwen3-Embedding-8B 49.04,
  + ToolRet-trained skip heads 52.25, BM25 20.33 (28.95 w/o inst: the generic instruction hurts
  BM25 on MCP sets), CLM 4.56 (fine-tuned 8.57).
- MCP-Zero (2,792 tools, one Qwen3-8B-written request per tool, w/ inst, Precision@1 = the paper's
  top-1 accuracy; the paper gives no single number): Qwen3-Embedding-8B 69.73, + skip heads 71.99,
  BM25 56.34 (w/o inst), CLM 1.47 (fine-tuned 4.62). The server name in the tool text adds ~8
  points (BM25 +24; LiveMCPBench +1-3) — `docs/reports/faz0-week4.md`.
- Gate decision (29 Sep 2026, `docs/reports/faz0-gate.md`): Qwen3-Embedding-8B + own skip heads
  (29.9M parameters for both heads) as the core adapter. It sits below the gate's 50 too (47.2
  cat-macro); the levers toward it (LoRA, cleaner negatives, LLM-enriched tool text) are in
  `docs/plan/backlog.md`.
- Faz 1 week 2 (`docs/reports/faz1-week2.md`): packaged fp16 heads = torch heads (ToolRet 54.03 /
  47.13 cat-macro, LiveMCPBench 53.03, MCP-Zero 79.87 with server names); FAISS HNSW on ToolRet
  −0.08 NDCG@10 at 1.6 ms vs 8.2 ms exact (batch 1); plain RRF hybrid 46.17 ToolRet / 40.80
  LiveMCPBench / 84.31 MCP-Zero; adaptive K (margin 0.2, max 10): ToolRet completeness 54.34 at K
  8.29 (+1.49 over fixed top-k at that K).
- Faz 1 week 3 (`docs/reports/faz1-week3.md`; `toolrank serve` on the Mac, 1,862 tools, embeddings
  from the GB10 over Tailscale): `search_tools` p50 5.2 ms with the query cached, 130–195 ms when it
  is embedded; `call_tool` adds ~2 ms over the backend; first start 154 s with an empty cache (the
  BM25 stand-in answers from 1.2 s meanwhile), 0.8 s with a warm one (`ingest --emb-url`).
- Faz 1 week 4 (`docs/reports/faz1-week4.md`; Claude `claude-opus-5-5`, data/w3): toolrank as
  Claude's tool search answered all three live tasks right, with 37–60% fewer input tokens than
  the API's own BM25 search (fewer, more relevant tools loaded) but one extra turn per search
  (client-side). 1,862 deferred tools = 3.67 MB per request. `gpt-5.5`: two tasks right in 3 turns
  (6.4 s, 4.7 s; one tool loaded for "time in Tokyo"); the refund task loaded 18 Stripe schemas and
  hit the account's 10k TPM limit at 34.5k tokens — loaded schemas pile up on broad searches.

- Faz 1 week 5 (`docs/reports/faz1-week5.md`): `toolrank finetune` reproduces Faz 0's 60K runs
  epoch for epoch. Picked on MCP-Zero `_server` as dev, it keeps epoch 2 with in-batch negatives (dev
  88.50; ToolRet 51.74 / 46.68) and the untrained start with 15 mined negatives (val recall 89.45 →
  96.85 while dev fell 87.21 → 78.65); from the packaged heads, epoch 0 is v0.1's own 88.53. README
  table: BM25 with the instruction on the `_server` sets 22.92 (LiveMCPBench R@5) / 45.63 (MCP-Zero
  P@1). On data/w3, bigtool, LlamaIndex and the LiteLLM filter (120 → 3 function tools; 11.5 s the
  first time a list is seen) all pick the right tool.

- Faz 1 week 6 (`docs/reports/faz1-week6.md`): Qwen3-Embedding-8B in FP8 (vLLM `--quantization fp8`,
  8094) is within a query or two of bf16 everywhere — heads ToolRet 53.94 / 47.27, LiveMCPBench 53.48,
  MCP-Zero 79.51 — with backbone cosine 0.998, top-10 overlap 94.5%, batch-1 cold p50 54.9 ms (bf16
  98.5) and 7.6 GiB of weights (14.1): the images' default. The official vLLM v0.30.0 image runs on the
  GB10 and matches NGC 0.13 (cosine 0.99993; MCP sets within −0.26 / +0.00).
- Faz 1 week 7: ToolRet w/o inst (plain queries): heads 44.86 micro / 34.64 cat-macro, Qwen3-Embedding-8B
  42.73 / 35.28 (the heads, trained with instructions, help micro and cost cat-macro here). On the ToolRet
  leaderboard (HF Space, last updated Mar 2025; "w/ meta" = the full documentation, Avg = cat-macro) the
  heads (47.13) and Qwen3-Embedding-8B (46.54) would lead w/ inst over jina-reranker-v2 (45.73); w/o inst
  NV-Embed-v1 leads (35.50). Submissions go in as issues on mangopy/tool-retrieval-benchmark.

- Faz 2 week 1, second stage and LoRA (`docs/reports/faz2-rerank.md`, tables from `scripts/rerank_report.py`;
  w/ inst): a second stage that reads the request with each candidate gains on every set, a bi-encoder
  in that seat does not. Heads' top 20 with the documentation cut to 3000 characters: ToolRet NDCG@10
  54.03 → Qwen3-Reranker-8B 58.05 (cat-macro 52.93), LiveMCPBench 53.95 → 62.68, MCP-Zero top-1 79.87 →
  91.26; bge-reranker-v2-gemma 53.96 on ToolRet and breaks the MCP lists (36.19, 48.24); CLM_v0.1-8B in
  the seat 15.36 / 28.94 (top 100 / top 20), fine-tuned 34.20. Jev was measured in the same seat; its
  results stay out of every tracked file (the provider's terms; removed 5 Oct 2026). **LoRA on the
  backbone** (`scripts/lora_train.py`: rank 16, 20k ToolRet-train pairs, in-batch InfoNCE, 625 steps,
  8.6 h on the GB10, picked on MCP-Zero `_server`): ToolRet 58.90 / cat-macro 54.36 in one stage (the
  Faz 0 gate's 50, StackOne v2's 54.4), Recall@20 75.25 (heads 72.51), LiveMCPBench 55.74 (+2, held
  out), MCP-Zero 93.67 / top-1 88.57 (the selection set); heads trained on top keep epoch 0 (identity).
  Qwen3-Reranker-8B over the LoRA shortlist (top 20 + documentation): ToolRet 59.36 / 54.35 (the best
  row, +0.5 over LoRA alone), LiveMCPBench 61.24, MCP-Zero top-1 91.94. For the cross-encoder, top 100
  with name_desc is worse than top 20 with the documentation on every set (ToolRet 54.55 vs 58.05), and
  alone (every pair, LiveMCPBench) it gives 55.25 at 32 s a query: pointwise scoring needs a good
  shortlist.
  The merged weights live in `~/toolrank/data/lora/qwen3-emb-lora-20k/merged` on the GB10 and must
  carry the original repo's config, tokenizer and `1_Pooling` files: transformers 5.x writes a
  list-valued `extra_special_tokens` and `rope_parameters`, which the NGC image's 4.51 crashes on or
  misreads (the script copies the originals). The CLM backbone (8090) was stopped to make room;
  rerankers (48 GB) and LoRA training (20–30 GB) do not fit together.
- Faz 2 week 4, `toolrank learn` on simulated traffic (`docs/reports/faz2-week4.md`; `scripts/learn_sim.sh`,
  NDCG@10 micro / cat-macro on the 30% of queries never served, start = the v0.1 heads): ToolRet as the
  catalogue, 5,573 requests logged → 4,313 pairs: 54.20 / 48.59 → 57.23 / 52.55 (another split: 53.97 /
  48.02 → 57.49 / 53.02); by size 300 → 55.11, 1,000 → 55.27, 3,000 → 56.64, and 100 requests publish
  nothing. The whole gain is on held-out requests for tools the traffic had asked for (46.62 → 51.95);
  requests for tools never asked for do not move (63.67 → 63.83). Nothing forgotten without replay
  (LiveMCPBench 53.95 → 56.05, MCP-Zero 88.53 → 88.96). MCP-Zero as the catalogue (1,954 requests, every
  held-out request about a new tool): 89.24 → 91.96, ToolRet −0.5. The defaults stay: 10 epochs pick the
  same epoch, lr 3e-5 is a wash, `--neg 0` loses 1.5 (shown-but-not-called tools are useful negatives).
  An agent that calls a wrong tool 1 time in 5 (`tool_error`): the default publishes nothing, `--strict`
  restores the gain (57.22 / 52.45). A/B on the held-out queries at share 0.5: `mrr` 0.572 → 0.593,
  `promote`. A perfect simulated agent on benchmark queries: the shape of the curve, not real traffic.
- Faz 2 week 5 (`docs/reports/faz2-week5.md`; v0.1 heads, w/ inst): server → tool routing. Searching only the
  best servers loses on every set (MCP-Zero top-1 79.87 → 63.07 / 74.46 / 78.37 for the best 1 / 3 / 10
  servers; the right server ranks first 68–86% of the time), MCP-Zero's product formula is erratic (ToolRet
  −22). Adding 0.2 × the server summary's cosine: MCP-Zero top-1 80.98, NDCG@10 88.53 → 89.28, K@cut 5.88 →
  5.52 at a higher recall; LiveMCPBench NDCG@10 53.95 → 55.06; ToolRet (three categories as "servers") 54.03
  → 53.93, cat-macro 47.13 → 46.36: hence opt-in. Without the server name in the tool text the same rule is
  worth +2.2 top-1. No-tool gate on the best cosine (answerable vs the same request with its gold servers
  removed): AUROC 0.882 MCP-Zero, 0.761 LiveMCPBench, means 0.71 / 0.51 for answerable requests, so no
  threshold carries over (0.45 turns away 0.8% and catches 23% on MCP-Zero, turns away 30% on LiveMCPBench).
  Backlog D2.14 makes the threshold portable instead (`calibration.py`, `toolrank calibrate`): not measured yet.
  Co-use on the simulated ToolRet log (count ≥ 2, share ≥ 0.5, ≤ 2 partners): 111 of 2,388 held-out lists
  change, K 8.33 → 8.38, completeness 54.15 → 54.73 (multi-tool requests 31.48 → 32.80), the same +0.6 with
  heads learned from that log; a longer ranked list (max 12) pays +2.05 for 1.32 more tools.
- Faz 2 week 7 (`docs/reports/faz2-week7.md`): the LoRA backbone in FP8 = bf16 (ToolRet 59.02 / 54.53, LiveMCPBench
  NDCG@10 55.34, MCP-Zero top-1 87.71; cosine 0.997, top-10 overlap 95.4%); the v0.1 heads on it cost 1–2 points.
  Generated dev sets over the w3 catalogue (1,862 tools; `data gen-queries`, on the GB10 as `data/dev_w3*`),
  base → LoRA NDCG@10: one tool per request 92.99 → 92.49 (a tie; Recall@5 95–99% for every backbone),
  problem-style requests 89.67 → 87.71 (top 5: 19 to 4 for the base model, p 0.003), tasks of two tools 71.04 →
  82.95 and of three tools 55.21 → 75.40 (every tool in the top 10: 72 → 90% and 34 → 64%). So LoRA's gain is in
  multi-tool tasks (and ToolRet, MCP-Zero's two-line requests); LiveMCPBench is a tie within noise (94 tasks).
  Behind Qwen3-Reranker both backbones give top-1 89.4 / 89.5 on the one-tool set (from 84).
- Backlog D1.2 / D1.3 (`docs/reports/backlog-d1.md`; a 4 GB RTX 3050 Ti laptop, Ollama 0.35.1, `num_ctx` 8192, w/ inst):
  the LoRA backbone's GGUF Q4_K_M scores as vLLM bf16 does (ToolRet 59.50 / 54.51 cat-macro vs 58.90 / 54.36,
  LiveMCPBench NDCG@10 55.75 vs 55.74; Q8_0 55.23), and so does the base 8B's (54.42 / 53.62 vs 53.74). Qwen3-Embedding
  0.6B and 4B are 6–7 points lower on LiveMCPBench and 9–11 on ToolRet. Search latency there, one request at a time and
  not cached: 0.6B 46–93 ms (p95 ≤ 191), 4B Q4_K_M 275–326 ms, 8B Q4_K_M 571–668 ms (p95 ~0.7 s); cached 3–13 ms. The
  8B embeds ~100 tools a minute there (ToolRet 7.3 h; vLLM on the GB10 0.4 h). Ollama ignores `truncate_prompt_tokens`
  and cuts at `num_ctx`, keeping the head; llama.cpp's own `llama-server --embeddings` keeps an output row per token
  (~5 GB for 8192 tokens) and crashed on a long input. MCP-Zero and the dev sets with GGUF wait for the GB10.

## Where we are

`docs/plan/README.md` → "Şu an" (maintainers' checkout). 0.2.0 is out (2 Oct 2026); backlog wave 1 aims at
0.3.0: D1.3–D1.5, D1.7–D1.12 and the 5 Oct review's fixes (`docs/plan/yapilacaklar-5-eki.md`) are done, D1.1 (the 60k LoRA run stopped at step 690 on 3 Oct) and the GPU
parts of D1.2/D1.3 wait for the GB10, the GGUF upload for a public step. Faz 2's pilots and gate report remain,
and the review's multi-tenant findings are a wave of their own. Session prompts: `docs/plan/claude-code-handoff.md`.
