# Contributing to toolrank

Bug reports, benchmark results, new adapters and documentation fixes are all welcome. Please follow
the [code of conduct](https://github.com/yasinyaman/toolrank/blob/main/CODE_OF_CONDUCT.md).

## Set up

```bash
git clone https://github.com/yasinyaman/toolrank && cd toolrank
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"            # add ",clm" to run the torch tests too
```

## Before you open a pull request

```bash
uv run ruff format src tests scripts examples
uv run ruff check src tests scripts examples
uv run pytest
```

All three must pass; CI runs them on Python 3.11, 3.12 and 3.13. A few rules keep the suite fast and
honest:

- **Tests never call a live endpoint.** Encoders are faked, MCP servers are
  `tests/fixtures/mcp_server.py` (in process, over stdio or local HTTP), HTTP APIs go through
  `httpx2.MockTransport`, and the served app runs under Starlette's `TestClient`.
- **Heavy dependencies stay optional** and are imported inside functions. The base install is
  numpy and bm25s; `tests/test_ingest_mcp.py` checks that importing the CLI loads nothing else.
- **The benchmark protocol is fixed** to ToolRet's released code (top-100 over the whole corpus,
  cut-offs 5/10/20, micro-average over queries). Don't change a number by changing the protocol;
  `eval/metrics.py` changes need a test that shows trec_eval parity.
- **Generated files stay generated.** The results table in the README and on the benchmarks page
  comes from `docs/results.toml` (`scripts/readme_table.py --write`), the CLI reference from the
  argument parser (`scripts/cli_reference.py --write`) and `THIRD_PARTY_NOTICES.md` from `uv.lock`
  (`scripts/third_party.py --write`). Tests fail when one of them is stale.

## Adding things

- **A scorer:** one file in `src/toolrank/adapters/` implementing the `Scorer` protocol
  (`src/toolrank/ports.py`) with `tool_format` and `query_format` attributes, a branch in
  `build.build_scorer`, its name in the `--scorer` choices, and a test on the synthetic set.
- **A benchmark:** a converter in `src/toolrank/datasets/` that writes the JSONL pair
  (`datasets/jsonl.py`), plus a `toolrank data pull` choice. Evaluation never touches the network.
- **An integration:** a module in `src/toolrank/integrations/` that talks to `toolrank serve`
  through `toolrank.client.ToolrankClient`, imports its framework only when used, and has tests
  against the framework's real classes.

## Documentation

The site's pages are in `docs/` (the plans and weekly reports there, in Turkish, stay out of it):

```bash
uv run --group docs mkdocs serve        # http://127.0.0.1:8000
```

## Commits

Small commits with an imperative subject line ("Add a pgvector index", not "Added ..."), and a body
that says why when it isn't obvious.
