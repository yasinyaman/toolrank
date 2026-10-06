"""``toolrank`` command line.

toolrank data pull toolret --out data/toolret        # once, on a machine with Hub access
toolrank data pull mcp-zero --gen-url http://127.0.0.1:8093/v1   # queries come from a chat model
toolrank data synth --out data/synthetic             # tiny synthetic set for smoke tests
toolrank eval --data data/toolret --scorer bm25 --tool-format documentation --out results/bm25.json
toolrank eval --data data/toolret --scorer dense --emb-model qwen3-emb --emb-url http://spark:8091/v1
toolrank eval --data data/toolret --scorer clm --emb-model qwen3-8b --clm-ckpt ~/.cache/clm/CLM_v0.1-8B.pt
toolrank compare results/*.json                       # one markdown table across runs
toolrank ingest mcp --config ~/mcp.json --out data/mytools     # index your own MCP servers
toolrank ingest openapi spec.json --name billing --out data/mytools
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from toolrank import __version__
from toolrank.datasets.jsonl import load_queries, load_tools
from toolrank.eval.runner import format_table, run_eval, save_report
from toolrank.formats import QUERY_FORMATS, TOOL_FORMATS
from toolrank.ingest.text import MAX_CHARS


def cmd_eval(a: argparse.Namespace) -> int:
    data = Path(a.data)
    tools = load_tools(data / "tools.jsonl")
    tasks = [t.strip() for t in a.tasks.split(",")] if a.tasks else None
    queries = load_queries(data / "queries.jsonl", tasks)
    from dataclasses import replace

    if a.instruction is not None:  # one instruction for every query (e.g. a product default)
        a.with_inst = True
        queries = [replace(q, instruction=a.instruction) for q in queries]
    elif not a.with_inst:
        # strip instructions so every scorer sees the same "w/o inst." protocol
        queries = [replace(q, instruction="") for q in queries]
    if a.limit:
        queries = queries[: a.limit]
    if not queries:
        sys.exit("no queries selected")
    from toolrank.build import file_sha256, scorer_factory

    try:
        make, info = scorer_factory(a)
        scorer = make()
    except ValueError as e:
        sys.exit(str(e))
    heads = info["heads_path"]
    ks = tuple(int(x) for x in a.ks.split(","))
    counts = Counter(q.task or "all" for q in queries)
    rule = _cut_rule(a)
    if rule is not None:
        from toolrank.cut import cutter

        try:
            cut = cutter(scorer, rule)
        except ValueError as e:
            sys.exit(str(e))
    runs: list[dict] | None = [] if a.runs_out else None
    report = run_eval(
        scorer,
        tools,
        queries,
        dataset=data.name,
        k=a.k,
        ks=ks,
        batch=a.batch,
        cut=cut if rule is not None else None,
        config={
            "with_inst": a.with_inst,
            "instruction": a.instruction,
            "tool_format": scorer.tool_format.name,
            "query_format": scorer.query_format.name,
            "k": a.k,
            "tasks": tasks,
            "limit": a.limit,
            "emb_model": getattr(a, "emb_model", None) if a.scorer not in ("bm25", "jev") else None,
            "emb_url": a.emb_url if a.scorer not in ("bm25", "jev") else None,
            "truncate": a.truncate if a.scorer not in ("bm25", "jev") else None,
            "batch": a.batch,
            "clm_ckpt": str(a.clm_ckpt) if a.scorer == "clm" else None,
            "heads_sha256": file_sha256(str(heads)) if heads else None,
            "index": a.index if a.scorer not in ("bm25", "jev") else None,
            "cut": rule.describe() if rule is not None else None,
            "server_weight": a.server_weight or None,
            "hybrid": (
                {
                    "k_rrf": a.rrf_k,
                    "depth": a.rrf_depth,
                    "weight": a.rrf_weight,
                    "lexical": getattr(scorer, "base", scorer).lexical.name,
                }
                if a.hybrid
                else None
            ),
            "jev": (
                {
                    "rerank_depth": a.rerank_depth if a.rerank else None,
                    "tool_format": (a.jev_tool_format or "name_desc")
                    if a.rerank
                    else (a.tool_format or "name_desc"),
                    "max_chars": a.jev_max_chars,
                    "chunk": a.jev_chunk if a.scorer == "jev" else None,
                    "per_chunk": a.jev_per_chunk if a.scorer == "jev" else None,
                    "workers": a.jev_workers,
                }
                if "jev" in info
                else None
            ),
            "rerank": (
                {
                    "scorer": scorer.second.name,
                    "depth": a.rerank_depth,
                    "emb_url": getattr(scorer.second, "encoder", None) and scorer.second.encoder.base_url,
                    "heads_sha256": file_sha256(str(info["rerank"]["heads_path"]))
                    if info["rerank"].get("heads_path")
                    else None,
                }
                if "rerank" in info
                else None
            ),
            "task_counts": dict(counts),
            "version": __version__,
        },
        runs=runs,
    )
    encoder = getattr(scorer, "encoder", None)
    if encoder is not None:  # tokens sent to the endpoint, i.e. cache misses only
        report.config["encoder_tokens"] = encoder.tokens_spent
    jev = info.get("jev")
    if jev is not None:  # calls that reached TypeSafe (the rest came from the cache), tokens billed
        report.config["jev"].update(jev.stats())
    cross = info.get("cross") or (info.get("rerank") or {}).get("cross")
    if cross is not None:  # pairs that reached the score endpoint, the rest from the cache
        report.config["cross"] = cross.stats()
    cols = ("NDCG@10", "Recall@10", "Comprehensiveness@10")
    print(format_table(report, cols + (("K@cut", "Recall@cut", "Comprehensiveness@cut") if rule else ())))
    print(
        f"\nlatency/query: p50 {report.latency_ms['per_query_p50']} ms, p95 {report.latency_ms['per_query_p95']} ms"
        f" (batch {a.batch}); index {report.latency_ms['index_s']} s; tools {report.n_tools}; queries {report.n_queries}"
        + (f"; encoder tokens {report.config['encoder_tokens']}" if encoder is not None else "")
        + (
            f"; jev calls {jev.calls} (+{jev.cached} cached), tokens {jev.tokens_spent},"
            f" p50 {report.config['jev']['call_ms_p50']} ms"
            if jev is not None
            else ""
        )
    )
    if a.out:
        p = save_report(report, a.out)
        print(f"saved {p}")
    if runs is not None:
        p = Path(a.runs_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"dataset": data.name, "scorer": scorer.name, "rows": runs},
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"saved {p}")
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    if a.paired:
        return _compare_paired(a)
    metrics = a.metrics.split(",")
    macro = metrics[:1] if a.cat_macro is None else [m for m in a.cat_macro.split(",") if m]
    cols = metrics + [f"{m} cat-macro" for m in macro]
    rows = [
        "| Run | dataset | inst | n | " + " | ".join(cols) + " | p50 ms |",
        "| --- | --- | :-: | ---: | " + " | ".join("---:" for _ in cols) + " | ---: |",
    ]
    for f in a.files:
        r = json.loads(Path(f).read_text())
        cfg, cat = r.get("config", {}), r.get("category_macro") or {}
        rows.append(
            f"| {r['scorer']} | {r['dataset']} | {'y' if cfg.get('with_inst') else 'n'} | {r['n_queries']} | "
            + " | ".join(
                [_metric(r["overall"], m) for m in metrics]
                + [f"{100 * cat[m]:.2f}" if m in cat else "—" for m in macro]
            )
            + f" | {r['latency_ms'].get('per_query_p50', '')} |"
        )
    print("\n".join(rows))
    return 0


def _compare_paired(a: argparse.Namespace) -> int:
    """Paired tests over two runs files (``eval --runs-out``): the same queries answered by two
    scorers, so the differences are tested per query — sign test for the 0/1 metrics, a paired
    permutation test for NDCG@10."""
    from statistics import fmean

    from toolrank.eval.paired import permutation_test, sign_test

    if len(a.files) != 2:
        sys.exit("--paired compares exactly two runs files (toolrank eval --runs-out writes them)")
    runs = []
    for f in a.files:
        r = json.loads(Path(f).read_text())
        rows = r.get("rows")
        if not isinstance(rows, list):
            sys.exit(f"{f} is not a runs file: run toolrank eval with --runs-out")
        seen: set[str] = set()
        twice = {row["id"] for row in rows if row["id"] in seen or seen.add(row["id"])}
        if twice:  # pairing by id would silently keep one of them
            sys.exit(f"{f}: query ids repeat ({sorted(twice)[:3]}): a runs file names each query once")
        runs.append(r)
    sets = [r.get("dataset") for r in runs]
    if sets[0] != sets[1]:
        print(
            f"warning: the runs are of {sets[0]!r} and {sets[1]!r}: pairing by id assumes one set",
            file=sys.stderr,
        )
    by_id = [{row["id"]: row for row in r["rows"]} for r in runs]
    ids = [i for i in by_id[0] if i in by_id[1]]
    if not ids:
        sys.exit("the two runs share no query id")
    names = [r.get("scorer", f) for r, f in zip(runs, a.files, strict=True)]
    lines = [
        f"| metric | {names[0]} ↑ % | {names[1]} ↑ % | A−B | p |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, kind in (("P@1", "sign"), ("hit@5", "sign"), ("NDCG@10", "permutation")):
        xs = [by_id[0][i][metric] for i in ids]
        ys = [by_id[1][i][metric] for i in ids]
        if kind == "sign":
            wins, losses, p = sign_test(xs, ys)
            detail = f"{wins} up, {losses} down"
        else:
            p, detail = permutation_test(xs, ys), "mean diff"
        lines.append(
            f"| {metric} ({detail}) | {100 * fmean(xs):.2f} | {100 * fmean(ys):.2f} | "
            f"{100 * (fmean(xs) - fmean(ys)):+.2f} | {p:.4f} |"
        )
    note = f"n = {len(ids)} queries paired by id"
    if len(ids) < len(by_id[0]) or len(ids) < len(by_id[1]):
        note += f" (the runs answer {len(by_id[0])} and {len(by_id[1])}; only the shared ones pair)"
    lines.append(f"\n{note}; p: exact sign test (P@1, hit@5), paired permutation test (NDCG@10)")
    print("\n".join(lines))
    return 0


def cmd_data_pull(a: argparse.Namespace) -> int:
    defaults = {
        "toolret": "data/toolret",
        "toolret-train": "data/toolret_train",
        "livemcpbench": "data/livemcpbench",
        "mcp-zero": "data/mcp_zero",
    }
    out = a.out or defaults[a.dataset]
    if a.dataset == "mcp-zero":
        from toolrank.adapters.chat_api import OpenAIChat
        from toolrank.datasets.mcp_zero import pull_mcp_zero

        chat = OpenAIChat(
            a.gen_model, a.gen_url, max_tokens=a.gen_max_tokens, extra_body=json.loads(a.gen_extra)
        )
        n = pull_mcp_zero(out, chat, workers=a.gen_workers)
        print(
            f"wrote {n['tools']} tools ({n['servers']} servers, {n['duplicates']} duplicate ids dropped) "
            f"and {n['queries']} queries to {out}; {n['unparsed']} responses had no request block "
            "and are kept as plain text"
        )
        return 0
    if a.dataset == "toolret-train":
        from toolrank.datasets.toolret_train import pull_toolret_train

        print(f"wrote {pull_toolret_train(out)} training pairs to {out}")
        return 0
    if a.dataset == "livemcpbench":
        from toolrank.datasets.livemcpbench import pull_livemcpbench

        n_tools, n_q, dropped = pull_livemcpbench(out)
        print(
            f"wrote {n_tools} tools and {n_q} queries to {out} ({dropped} tasks without a corpus tool dropped)"
        )
        return 0
    from toolrank.datasets.toolret import pull_toolret

    tasks = [t.strip() for t in a.tasks.split(",")] if a.tasks else None
    n_tools, n_q = pull_toolret(out, tasks=tasks)
    print(f"wrote {n_tools} tools and {n_q} queries to {out}")
    return 0


def cmd_data_gen_queries(a: argparse.Namespace) -> int:
    from toolrank.adapters.chat_api import OpenAIChat
    from toolrank.datasets.genqueries import gen_queries

    if not (Path(a.data) / "tools.jsonl").exists():
        sys.exit(f"{Path(a.data) / 'tools.jsonl'} not found: --data is an ingest or benchmark dir")
    chat = OpenAIChat(a.gen_model, a.gen_url, max_tokens=a.gen_max_tokens, extra_body=json.loads(a.gen_extra))
    try:
        styles = [x.strip() for x in a.styles.split(",") if x.strip()]
        n = gen_queries(
            a.data,
            a.out,
            chat,
            n=a.n,
            seed=a.seed,
            exclude=a.exclude,
            styles=styles,
            per_request=a.tools_per_request,
            workers=a.gen_workers,
        )
    except ValueError as e:
        sys.exit(str(e))
    dropped = {k: v for k, v in n.items() if k not in ("tools", "sources", "sampled", "queries") and v}
    print(
        f"wrote {n['queries']} queries over {n['tools']} tools ({n['sources']} sources) to {a.out}: one per "
        f"sampled tool ({n['sampled']})" + (f"; dropped {dropped}" if dropped else "")
    )
    return 0


def cmd_data_server_names(a: argparse.Namespace) -> int:
    from toolrank.datasets.jsonl import with_server_names

    src = Path(a.src)
    out = Path(a.out) if a.out else src.with_name(src.name + "_server")
    if out.resolve() == src.resolve():
        sys.exit("--out must differ from the source set")
    n = with_server_names(src, out)
    print(f"wrote {n} tools with their server names, and the queries, to {out}")
    return 0


def cmd_data_synth(a: argparse.Namespace) -> int:
    from toolrank.datasets.synthetic import write_synthetic

    n_tools, n_q = write_synthetic(a.out, n_tools=a.n_tools, n_queries=a.n_queries, seed=a.seed)
    print(f"wrote {n_tools} tools and {n_q} queries to {a.out}")
    return 0


def cmd_search(a: argparse.Namespace) -> int:
    from toolrank.build import build_retriever

    data = Path(a.data)
    if a.index_dir is None:
        a.index_dir = str(data / "index")
    _data_cache(a, data)
    _check_rerank(a)
    _check_confidence(a)
    try:
        retriever = build_retriever(a)
    except ValueError as e:
        sys.exit(str(e))
    st = retriever.state()
    res = retriever.search(a.request)
    if res.rerank_error:
        print(f"warning: second stage failed, first-stage order: {res.rerank_error}", file=sys.stderr)
    if a.json:
        tools = [
            {
                "id": h.id,
                "score": round(h.score, 6),
                "server": h.server,
                "name": h.tool.name,
                "description": h.tool.description,
                **({"used_with": h.used_with} if h.used_with else {}),
            }
            for h in res.hits
        ]
        out = {"request": a.request, "instruction": res.instruction, "scorer": res.scorer, "tools": tools}
        if res.confidence is not None:
            out["confidence"] = res.confidence
        print(json.dumps(out, ensure_ascii=False))
        return 0
    sure = f"; confidence {res.confidence:.3f}" if res.confidence is not None else ""
    print(
        f"{res.scorer}\n{len(st.tools)} tools; index {st.index_s:.2f} s "
        f"(embedded {st.sync.get('embedded', 0)}, kept {st.sync.get('kept', 0)}); "
        f"search {res.took_ms:.0f} ms; {res.rule}{sure}"
    )
    if not res.hits:
        print("no tool is close enough to the request")
    for n, h in enumerate(res.hits, 1):
        first = (h.tool.description.strip().splitlines() or [""])[0][:90]
        with_ = f"  (used with {h.used_with})" if h.used_with else ""
        print(f"{n:>2}. {h.score:.4f}  {h.id}  {first}{with_}")
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from toolrank.build import build_retriever
    from toolrank.calibration import FILE, MIN_REQUESTS, Calibration, band_score, measure, save
    from toolrank.datasets.jsonl import load_queries
    from toolrank.retriever import LOG_TOP, innermost

    data = Path(a.data)
    if a.index_dir is None:
        a.index_dir = str(data / "index")
    _data_cache(a, data)
    _check_rerank(a)
    _check_confidence(a)
    path = Path(a.requests) / "queries.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found: --requests is a dir from toolrank data gen-queries --data {a.data}")
    queries = load_queries(path)
    try:
        retriever = build_retriever(a)
    except ValueError as e:
        sys.exit(str(e))
    st, arm, heads = retriever.settled()
    first = innermost(st.scorer)
    best, twins, recall, skipped = measure(
        first, st.tools, queries, retriever.instruction, depth=max(a.cut_max, LOG_TOP)
    )
    gone = f" ({skipped} skipped: their tools are no longer in the catalogue)" if skipped else ""
    if len(best) < MIN_REQUESTS:
        sys.exit(
            f"{len(best)} requests{gone}: a calibration needs at least {MIN_REQUESTS} "
            f"(toolrank data gen-queries --data {a.data} --n 200)"
        )
    cal = Calibration(
        scorer=first.name,
        heads=heads,
        instruction=retriever.instruction,
        best=tuple(sorted(best)),
        twins=tuple(sorted(twins)),
        recall_at_5=round(recall, 4),
        catalog=st.catalog,
        requests=Path(a.requests).name,
        made=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        extra={"arm": arm, "skipped": skipped},
    )
    print(f"{len(best)} requests{gone}; first stage {first.name} ({arm} heads); Recall@5 {100 * recall:.1f}%")
    if recall < 0.8:
        print(
            "warning: under 80% of these requests find their tool in the top 5; check the set before trusting it"
        )
    print(f"best score: p5 {band_score(cal, 0.05):.3f}, median {band_score(cal, 0.5):.3f}")
    for band in (0.01, 0.05, 0.1):
        caught = cal.caught(band)
        print(
            f"--min-confidence {band:g}: turns away requests below {band_score(cal, band):.3f}, "
            f"about {100 * band:g}% of answerable ones"
            + (f", and {100 * caught:.1f}% of the twins (their answer hidden)" if caught is not None else "")
        )
    if a.dry_run:
        return 0
    save(data / FILE, cal)
    print(f"wrote {data / FILE}: searches by this first stage carry a confidence")
    return 0


def _check_confidence(a: argparse.Namespace) -> None:
    if a.min_confidence is not None and not 0 < a.min_confidence < 1:
        sys.exit("--min-confidence: a share between 0 and 1, e.g. 0.05")


def _load_api_keys(path: str) -> dict[str, str]:
    """``--api-keys`` as {name: key} (``tenants.load_tenants`` has the sources and credentials too)."""
    from toolrank.tenants import load_tenants

    return {name: t.key for name, t in load_tenants(path).items()}


def _env_list(name: str) -> list[str]:
    """A comma-separated environment variable as a list (containers configure through these)."""
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def _need_mcp(command: str) -> None:
    """Exit with the install line when the ``[mcp]`` extra (the MCP SDK and the serve stack) is missing."""
    try:
        import mcp  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        sys.exit(f"{command} needs the MCP stack: pip install 'toolrank[mcp]'")


def cmd_serve(a: argparse.Namespace) -> int:
    _need_mcp("toolrank serve")
    from toolrank.adapters.backends import Backends
    from toolrank.adapters.mcp_proxy import build_proxy, http_app, serve_stdio
    from toolrank.build import build_retriever
    from toolrank.ingest.mcp import load_config, load_openapi, parse_server
    from toolrank.usage import UsageLog

    def log(msg: str) -> None:  # stdout is the protocol channel under --stdio
        print(msg, file=sys.stderr, flush=True)

    data = Path(a.data).resolve()
    if not (data / "tools.jsonl").exists():
        sys.exit(f"{data / 'tools.jsonl'} not found: run toolrank ingest first")
    a.data = str(data)
    a.index_dir = str(Path(a.index_dir).resolve()) if a.index_dir else str(data / "index")
    _data_cache(a, data)
    from toolrank.tenants import check_sources, load_tenants

    api_key = a.api_key or os.environ.get("TOOLRANK_API_KEY")
    try:
        tenants = load_tenants(a.api_keys) if a.api_keys else {}
    except ValueError as e:
        sys.exit(str(e))
    named = {name: t.key for name, t in tenants.items()}
    if api_key and api_key in named.values():
        sys.exit("the anonymous API key is also in --api-keys: keep it under its name only")
    if not a.stdio and a.host not in ("127.0.0.1", "localhost", "::1") and not (api_key or named):
        sys.exit(
            f"--host {a.host} is reachable from other machines: set --api-key, TOOLRANK_API_KEY or --api-keys"
        )
    try:
        servers = [c for path in a.config for c in load_config(path)] + [parse_server(s) for s in a.server]
        openapi = {k: v for path in a.config for k, v in load_openapi(path).items()}
    except ValueError as e:
        sys.exit(str(e))
    names = [c.name for c in servers]
    if len(set(names)) != len(names):
        sys.exit(f"duplicate server names: {sorted({n for n in names if names.count(n) > 1})}")
    try:
        catalogue = (
            json.loads((data / "sources.json").read_text()) if (data / "sources.json").exists() else {}
        )
        for warning in check_sources(tenants, catalogue, {c.name: c.transport for c in servers}, openapi):
            log(warning)
    except ValueError as e:
        sys.exit(str(e))
    a.allowed = {name: t.sources for name, t in tenants.items() if t.sources is not None}
    usage = UsageLog(
        None if a.no_usage_log else (a.usage_log or data / "usage"), log_text=a.log_text, mask_pii=a.mask_pii
    )
    backends = Backends(
        servers,
        openapi,
        allow_write=a.allow_write,
        call_timeout=a.timeout,
        connect_timeout=a.timeout,
        tenants=tenants,
    )
    log(
        f"toolrank serve: {data} ({len(servers)} MCP servers, {len(openapi)} OpenAPI configs"
        f"{', writes allowed' if a.allow_write else ''}); usage log "
        + ("off" if usage.dir is None else str(usage.dir))
    )
    _check_rerank(a)
    _check_confidence(a)
    if a.rerank == "jev":
        from toolrank.adapters.jev import is_typesafe

        label = "Jev (TypeSafe AI)" if is_typesafe(a.jev_url) else "a System One endpoint"
        log(f"second stage: {label} - each request's text and its top tools' text are sent to " + a.jev_url)
        log("searches served with Jev feed neither learn, ab nor co-use (the provider's terms)")
    try:
        retriever = build_retriever(a, background=True, serving_limits=True, notify=log)
    except ValueError as e:
        sys.exit(str(e))
    retriever.ready_timeout = 30.0
    server = build_proxy(retriever, backends, usage)
    if a.stdio:
        import anyio

        anyio.run(serve_stdio, server)
        return 0
    import uvicorn

    from toolrank.adapters.rest import rest_routes

    routes = rest_routes(retriever, usage, backends)
    app = http_app(
        server,
        host=a.host,
        extra_hosts=[*a.allowed_host, *_env_list("TOOLRANK_ALLOWED_HOSTS")],
        api_key=api_key,
        named_keys=named,
        routes=routes,
    )
    base = f"http://{a.host}:{a.port}"
    tokens = len(named) + bool(api_key)
    log(
        f"MCP at {base}/mcp, REST at {base}/v1 ({base}/openapi.json)"
        + (f" — bearer token required ({tokens} accepted)" if tokens else "")
    )
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


def cmd_finetune(a: argparse.Namespace) -> int:
    from toolrank.build import DEFAULT_EMB_URL, DEFAULT_SERVING, backbone_repo
    from toolrank.finetune import Job, TrainConfig, run

    if a.init_ckpt and (a.width is not None or a.depth is not None or a.no_skip):
        sys.exit("--width, --depth and --no-skip shape fresh heads; --init-ckpt brings its own")
    if not a.embed_only:
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.exit(
                "toolrank finetune trains with torch: pip install 'toolrank[clm]' "
                "(--embed-only fills the embedding cache without it)"
            )
    name = a.name or Path(a.out).stem
    head_cfg = {"width": a.width or 1536, "depth": a.depth or 3, "skip": not a.no_skip}
    job = Job(
        pairs=Path(a.data),
        dev=Path(a.dev),
        out=Path(a.out),
        evals=[Path(p) for p in a.eval],
        init=a.init_ckpt,
        npz=Path(a.npz) if a.npz else None,
        emb_url=a.emb_url or DEFAULT_EMB_URL,
        emb_model=a.emb_model or "qwen3-emb",  # the base model its cached vectors and recipes use
        emb_batch=a.emb_batch,
        truncate=a.truncate or None,
        cache_dir=a.cache_dir or None,
        tool_format=a.tool_format,
        query_format=a.query_format,
        instruction="" if a.keep_bare else (a.instruction or DEFAULT_SERVING["instruction"]),
        backbone=a.backbone or backbone_repo(a.emb_model or "qwen3-emb"),
        n_train=a.n_train,
        n_val=a.n_val,
        # the split takes the data seed; training takes --seed (same split across training seeds)
        seed=a.data_seed if a.data_seed is not None else a.seed,
        select=a.select,
        curve=a.curve,
        embed_only=a.embed_only,
        train=TrainConfig(
            epochs=a.epochs,
            batch=a.batch,
            neg_per_pair=a.neg,
            lr=a.lr,
            weight_decay=a.weight_decay,
            warmup=a.warmup,
            seed=a.seed,
            device=a.device,
            head_cfg=head_cfg,
            freeze_action=a.freeze_action,
            neg_filter=a.neg_filter,
        ),
    )
    try:  # flushed lines: a run under nohup shows its progress while it embeds for hours
        report = run(job, log=lambda line: print(line, flush=True))
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))
    report = {"name": name, "args": {k: v for k, v in vars(a).items() if k != "fn"}, **report}
    if not a.embed_only:
        # the official numbers: toolrank eval on the saved file, as anyone would run it
        official: dict[str, dict[str, object]] = {}
        for data in [job.dev, *job.evals]:
            out = Path(a.results) / f"finetune_{name}_{data.name}.json"
            flags = ["eval", "--data", str(data), "--scorer", "clm", "--clm-ckpt", str(job.out)]
            flags += [
                "--emb-url",
                job.emb_url,
                "--emb-model",
                job.emb_model,
                "--emb-batch",
                str(job.emb_batch),
            ]
            flags += ["--tool-format", job.tool_format, "--query-format", job.query_format, "--with-inst"]
            flags += ["--ks", "1,5,10,20", "--cache-dir", job.cache_dir or "", "--out", str(out)]
            flags += ["--truncate", str(job.truncate)] if job.truncate else []
            cmd_eval(build_parser().parse_args(flags))
            ev = json.loads(out.read_text())
            row = {"report": str(out), "NDCG@10": ev["overall"]["NDCG@10"]}
            if ev.get("category_macro"):
                row["NDCG@10-cat"] = ev["category_macro"]["NDCG@10"]
            official["dev" if data == job.dev else data.name] = row
        best = report["history"][report["best_epoch"]]
        report["official"] = official
        report["curve_vs_official"] = {
            f"{k}.{m}": round(float(v[m]) - best[f"{k}.{m}"], 6)
            for k, v in official.items()
            for m in ("NDCG@10", "NDCG@10-cat")
            if m in v and f"{k}.{m}" in best
        }
    path = Path(a.report) if a.report else Path(a.results) / f"finetune_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"report: {path}")
    return 0


def cmd_learn(a: argparse.Namespace) -> int:
    from toolrank.build import DEFAULT_EMB_MODEL, DEFAULT_EMB_URL, DEFAULT_SERVING, backbone_repo
    from toolrank.finetune import TrainConfig
    from toolrank.learn import CANDIDATE, Job, heads_home, run

    data = Path(a.data).resolve()
    if not (data / "tools.jsonl").exists():
        sys.exit(f"{data / 'tools.jsonl'} not found: an ingest dir that toolrank serve has served")
    _data_cache(a, data)
    # by default the result is the candidate a running server gives a share of the requests to
    out = Path(a.out).resolve() if a.out else heads_home(data, a.tenant) / CANDIDATE
    if out.name == CANDIDATE and out.exists() and not (a.replace_candidate or a.dry_run):
        print(
            f"{out} is still being judged (toolrank ab decides); --replace-candidate trains a new one over it"
        )
        return 0
    if not a.dry_run:  # after the guard above: saying "not yet" needs no torch
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.exit("toolrank learn trains with torch: pip install 'toolrank[clm]' (--dry-run needs none)")
    name = a.name or out.stem
    emb_model = a.emb_model or os.environ.get("TOOLRANK_EMB_MODEL") or DEFAULT_EMB_MODEL
    job = Job(
        data=data,
        out=out,
        dev=Path(a.dev) if a.dev else None,
        init=None if a.init == "none" else a.init,
        emb_url=a.emb_url or os.environ.get("TOOLRANK_EMB_URL") or DEFAULT_EMB_URL,
        emb_model=emb_model,
        emb_batch=a.emb_batch,
        truncate=a.truncate or int(DEFAULT_SERVING["truncate"]),
        cache_dir=a.cache_dir or None,
        tool_format=a.tool_format,
        query_format=a.query_format,
        backbone=a.backbone or backbone_repo(emb_model),
        since=a.since,
        tenant=a.tenant,
        strict=a.strict,
        min_pairs=a.min_pairs,
        dev_share=a.dev_share,
        max_drop=a.max_drop,
        dry_run=a.dry_run,
        # the replay sample takes the data seed; training takes --seed
        data_seed=a.data_seed if a.data_seed is not None else a.seed,
        replay=Path(a.replay) if a.replay else None,
        replay_n=a.replay_n if a.replay else 0,
        instruction=str(DEFAULT_SERVING["instruction"]),
        train=TrainConfig(
            epochs=a.epochs,
            batch=a.batch,
            neg_per_pair=a.neg,
            lr=a.lr,
            weight_decay=a.weight_decay,
            warmup=a.warmup,
            seed=a.seed,
            device=a.device,
            neg_filter=a.neg_filter,
        ),
    )
    try:
        report = run(job, log=lambda line: print(line, flush=True))
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))
    report = {"name": name, "args": {k: v for k, v in vars(a).items() if k != "fn"}, **report}
    path = Path(a.report) if a.report else Path(a.results) / f"learn_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"{report['decision']}; report: {path}")
    return 0


def cmd_ab(a: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from toolrank.learn import CANDIDATE, apply, decide, heads_home, judge, read_events

    data = Path(a.data).resolve()
    home = heads_home(data, a.tenant)
    candidate = home / CANDIDATE
    if not candidate.exists():
        print(f"no candidate: {candidate} does not exist (toolrank learn writes it)")
        return 0
    # the candidate's arm starts when its file appears: only searches from then on are compared
    since = a.since or datetime.fromtimestamp(candidate.stat().st_mtime, UTC).isoformat(
        timespec="milliseconds"
    )
    counts: Counter[str] = Counter()
    stats = judge(read_events(data / "usage"), since=since, tenant=a.tenant, counts=counts)
    decision = a.force or decide(stats, min_searches=a.min_searches, margin=a.margin)
    print(f"since {since}" + (f", tenant {a.tenant}" if a.tenant else ""))
    if counts["searches_with_jev"]:
        print(
            f"{counts['searches_with_jev']} Jev-served searches decide nothing "
            "(the provider's terms keep them out)"
        )
    print("| arm | searches | called | top-1 | mrr |\n| --- | ---: | ---: | ---: | ---: |")
    for name in ("control", "candidate"):
        r = stats[name]
        print(f"| {name} | {r['searches']} | {r['called']} | {r['top1']:.3f} | {r['mrr']:.3f} |")
    moved = {} if a.dry_run else apply(home, decision)
    report = {"data": str(data), "tenant": a.tenant, "since": since, "stats": stats, "decision": decision}
    report.update(moved=moved, forced=bool(a.force), min_searches=a.min_searches, margin=a.margin)
    if counts:
        report["skipped"] = dict(counts)
    path = Path(a.results) / f"ab_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    done = ", ".join(f"{k} -> {v}" for k, v in moved.items()) or (
        "nothing moved" if not a.dry_run else "dry run"
    )
    print(f"{decision}: {done}; report: {path}")
    return 0


def cmd_heads_pull(a: argparse.Namespace) -> int:
    from toolrank.adapters.heads_np import default_heads, sha256_file

    try:
        path = default_heads(url=a.url)
    except (ValueError, FileNotFoundError, OSError) as e:
        sys.exit(f"toolrank heads pull: {e}")
    print(f"{path}  sha256 {sha256_file(path)}")
    return 0


def cmd_heads_export(a: argparse.Namespace) -> int:
    from toolrank.adapters.heads_np import export_npz

    serving = {
        k: v
        for k, v in {
            "backbone": a.backbone,
            "tool_format": a.tool_format,
            "query_format": a.query_format,
            "truncate": a.truncate,
            "instruction": a.instruction,
        }.items()
        if v is not None
    }
    digest = export_npz(a.src, a.dst, dtype=a.dtype, serving=serving)
    print(f"wrote {a.dst} ({Path(a.dst).stat().st_size / 1e6:.1f} MB, {a.dtype}), sha256 {digest}")
    return 0


def cmd_formats(a: argparse.Namespace) -> int:
    print("tool formats: " + ", ".join(TOOL_FORMATS))
    print("query formats: " + ", ".join(QUERY_FORMATS))
    return 0


def _add_cut_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("adaptive K (on when --cut-margin or --cut-threshold is given)")
    g.add_argument("--cut-margin", type=float, default=None, help="keep tools within this cosine of the best")
    g.add_argument("--cut-threshold", type=float, default=None, help="keep tools at or above this cosine")
    g.add_argument("--cut-max", type=int, default=10)
    g.add_argument("--cut-min", type=int, default=1)


def _add_confidence_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        metavar="Q",
        help="turn away a request less sure than this share of answerable ones (an empty list and a note; "
        "e.g. 0.05); needs DATA/calibration.json (toolrank calibrate)",
    )


def _add_jev_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group(
        "Jev (TypeSafe AI; key in $TYPESAFE_API_KEY): --rerank jev over any scorer, or --scorer jev alone"
    )
    g.add_argument(
        "--rerank",
        choices=["jev", "dense", "clm", "cross"],
        default=None,
        help="reorder the top --rerank-depth: with Jev, or with a second dense / clm / cross-encoder scorer (--rerank-* flags)",
    )
    g.add_argument("--rerank-depth", type=int, default=100, help="tools per query reranked (Jev: max 255)")
    g.add_argument("--rerank-emb-url", default=None, help="the second scorer's endpoint (default: --emb-url)")
    g.add_argument("--rerank-emb-model", default=None, help="(default: --emb-model)")
    g.add_argument("--rerank-truncate", type=int, default=None, help="(default: --truncate)")
    g.add_argument("--rerank-clm-ckpt", default=None, help="--rerank clm: its heads (default: --clm-ckpt)")
    g.add_argument(
        "--rerank-tool-format", choices=list(TOOL_FORMATS), default=None, help="(default: --tool-format)"
    )
    g.add_argument(
        "--rerank-query-format", choices=list(QUERY_FORMATS), default=None, help="(default: --query-format)"
    )
    g.add_argument(
        "--jev-model", default="jev-1.13.0", help="a versioned id: aliases such as jev-latest move"
    )
    g.add_argument("--jev-url", default="https://api.typesafe.ai/v1")
    g.add_argument(
        "--jev-tool-format",
        choices=list(TOOL_FORMATS),
        default=None,
        help="text per option (default: name_desc)",
    )
    g.add_argument("--jev-max-chars", type=int, default=1000, help="characters kept per option")
    g.add_argument(
        "--jev-chunk", type=int, default=200, help="--scorer jev: tools per Choice question (max 255)"
    )
    g.add_argument(
        "--jev-per-chunk", type=int, default=20, help="--scorer jev: chunk winners into the final round"
    )
    g.add_argument("--jev-workers", type=int, default=8, help="concurrent requests (TypeSafe: 40/s)")
    g.add_argument(
        "--rerank-max-chars",
        type=int,
        default=None,
        help="cut each candidate's text, like --jev-max-chars does for Jev",
    )
    g.add_argument(
        "--rerank-template",
        choices=["qwen3", "bge"],
        default=None,
        help="--rerank cross: the reranker's prompt format",
    )
    g.add_argument(
        "--cross-template",
        choices=["qwen3", "bge"],
        default=None,
        help="--scorer cross: the prompt format (default qwen3)",
    )
    g.add_argument(
        "--rerank-query-chars",
        type=int,
        default=None,
        help="--rerank cross: characters of the request kept (6000)",
    )
    g.add_argument("--cross-query-chars", type=int, default=None, help="--scorer cross: the same (6000)")
    g.add_argument(
        "--rerank-workers",
        type=int,
        default=1,
        help="queries scored concurrently by the second scorer (cross: 8)",
    )


def _add_serve_rerank_args(p: argparse.ArgumentParser) -> None:
    """``search`` / ``serve``: an optional second stage over the first stage's top tools, with the
    setting that measured best (``docs/reports/faz2-rerank.md``): the top 20, each tool's full
    documentation cut to 3,000 characters. Off unless ``--rerank`` is given."""
    g = p.add_argument_group("second stage (off by default): rerank the top tools with the request")
    g.add_argument(
        "--rerank",
        choices=["cross", "jev"],
        default=None,
        help="cross: a local cross-encoder behind vLLM's score API (Qwen3-Reranker-8B); "
        "jev: TypeSafe AI's hosted Jev (key in $TYPESAFE_API_KEY; the request text leaves the machine)",
    )
    g.add_argument("--rerank-depth", type=int, default=20, help="tools reranked per request")
    g.add_argument("--rerank-emb-url", default=None, help="cross: the reranker's /v1 endpoint")
    g.add_argument("--rerank-emb-model", default="qwen3-reranker", help="cross: its served name")
    g.add_argument(
        "--rerank-template", choices=["qwen3", "bge"], default="qwen3", help="cross: prompt format"
    )
    g.add_argument(
        "--rerank-tool-format", choices=list(TOOL_FORMATS), default="documentation", help="text per tool"
    )
    g.add_argument("--rerank-max-chars", type=int, default=3000, help="characters kept per tool")
    g.add_argument(
        "--rerank-query-chars", type=int, default=None, help="cross: characters of the request (6000)"
    )
    g.add_argument(
        "--rerank-workers",
        type=int,
        default=1,
        help="cross: scoring calls in flight at once, all requests together",
    )
    g.add_argument(
        "--rerank-timeout",
        type=float,
        default=10.0,
        help="seconds per attempt (two attempts, a turn in the queue included); then the first stage's order answers",
    )
    g.add_argument("--jev-model", default="jev-1.13.0", help="jev: a versioned id (aliases move)")
    g.add_argument("--jev-url", default="https://api.typesafe.ai/v1")
    g.add_argument("--jev-workers", type=int, default=8, help="jev: calls in flight at once")
    # the eval-only knobs the shared factory reads, at their "same as the first stage" values
    p.set_defaults(
        rerank_truncate=None,
        rerank_clm_ckpt=None,
        rerank_query_format=None,
        jev_tool_format=None,
        jev_max_chars=None,
        cross_template=None,
        cross_query_chars=None,
    )


def _check_rerank(a: argparse.Namespace) -> None:
    """The second stage's flags, checked before anything is built; Jev reads the tool format and
    the cut of the cross-encoder's flags, so both rerankers see the same text."""
    if a.rerank is None:
        return
    if a.rerank == "cross" and not a.rerank_emb_url:
        sys.exit(
            "--rerank cross needs --rerank-emb-url: the reranker's /v1 endpoint (vLLM serving Qwen3-Reranker-8B)"
        )
    if a.rerank == "jev":
        from toolrank.adapters.jev import is_typesafe

        if is_typesafe(a.jev_url) and not os.environ.get("TYPESAFE_API_KEY"):
            sys.exit("--rerank jev needs TYPESAFE_API_KEY for TypeSafe (a local --jev-url needs none)")
        a.jev_tool_format, a.jev_max_chars = a.rerank_tool_format, a.rerank_max_chars
    if not 2 <= a.rerank_depth <= 255:
        sys.exit("--rerank-depth: from 2 to 255")
    if not a.rerank_timeout > 0 or a.rerank_workers < 1 or a.jev_workers < 1:
        sys.exit("--rerank-timeout must be positive, --rerank-workers and --jev-workers at least 1")


def _cut_rule(a: argparse.Namespace):
    from toolrank.cut import rule_from_flags

    return rule_from_flags(a.cut_margin, a.cut_threshold, a.cut_max, a.cut_min)


def _metric(values: dict, m: str) -> str:
    if m not in values:
        return "—"
    return f"{values[m]:.2f}" if m.startswith("K@") else f"{100 * values[m]:.2f}"


def _add_encoder_args(
    p: argparse.ArgumentParser,
    *,
    url: str | None,
    model: str | None,
    cache_dir: str | None = ".cache/toolrank",
) -> None:
    p.add_argument("--emb-url", default=url, help="OpenAI-compatible base URL")
    p.add_argument("--emb-model", default=model)
    p.add_argument("--emb-batch", type=int, default=32)
    p.add_argument(
        "--truncate", type=int, default=None, help="vLLM truncate_prompt_tokens (CLM reference: 2048)"
    )
    where = "DIR/cache, next to tools.jsonl" if cache_dir is None else cache_dir
    p.add_argument(
        "--cache-dir", default=cache_dir, help=f"embedding cache directory (default: {where}; '' = none)"
    )


def _data_cache(a: argparse.Namespace, data: Path) -> None:
    """An ingest dir keeps its own embedding cache, so ``ingest --emb-url`` warms exactly what
    ``search`` and ``serve`` read: ``DATA/cache`` unless ``--cache-dir`` names one (``''`` = none).
    Absolute, since Claude Desktop starts servers in ``/``."""
    if a.cache_dir is None:
        a.cache_dir = str(data.resolve() / "cache")
    elif a.cache_dir:
        a.cache_dir = str(Path(a.cache_dir).resolve())


def _add_retrieval_args(p: argparse.ArgumentParser) -> None:
    """``search``'s and ``serve``'s retrieval flags (product defaults filled by ``build.search_defaults``)."""
    p.add_argument("--data", required=True, help="ingest dir with tools.jsonl")
    p.add_argument("--instruction", default=None, help="default: the heads' instruction")
    p.add_argument("--k", type=int, default=0, help="a fixed top-k instead of adaptive K")
    p.add_argument("--no-cut", action="store_true", help="plain top --cut-max")
    p.add_argument(
        "--clm-ckpt",
        default=None,
        help="heads: .npz / .pt path, 'default' (downloads) or 'none' (no packaged heads; learned "
        "DATA/heads ones still serve)",
    )
    p.add_argument("--tool-format", choices=list(TOOL_FORMATS), default=None)
    p.add_argument("--query-format", choices=list(QUERY_FORMATS), default=None)
    p.add_argument("--index", default="numpy", help="numpy | faiss | pgvector")
    p.add_argument("--index-dir", default=None, help="default: DATA/index")
    p.add_argument("--pg-dsn", default=None, help="pgvector: Postgres DSN (default: $TOOLRANK_PG_DSN)")
    p.add_argument("--pg-table", default="toolrank_tools")
    p.add_argument(
        "--hybrid", action="store_true", help="fuse BM25 by RRF (helps agent-written requests only)"
    )
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--rrf-depth", type=int, default=100)
    p.add_argument("--rrf-weight", type=float, default=1.0)
    p.add_argument(
        "--server-weight",
        type=float,
        default=0.0,
        help="add this much of the request's cosine with a tool's server to the tool's score (0 = off; try 0.2)",
    )
    p.add_argument(
        "--co-use",
        type=int,
        default=0,
        metavar="N",
        help="append up to N tools the usage log shows are called together with a tool in the list (0 = off)",
    )
    p.add_argument("--no-stem", action="store_true")
    p.add_argument("--device", default=None)
    _add_cut_args(p)
    _add_confidence_args(p)
    _add_serve_rerank_args(p)
    _add_encoder_args(p, url=None, model=None, cache_dir=None)


def _warm_cache(a: argparse.Namespace, tools: list) -> None:
    """Embed the texts the cache does not have yet: new and changed tools, never unchanged ones.
    Model and truncation default to search's and serve's, so their first index is all cache hits."""
    from toolrank.adapters.embeddings_api import OpenAIEmbeddings
    from toolrank.build import DEFAULT_EMB_MODEL, DEFAULT_SERVING

    _data_cache(a, Path(a.out))
    a.emb_model = a.emb_model or os.environ.get("TOOLRANK_EMB_MODEL") or DEFAULT_EMB_MODEL
    enc = OpenAIEmbeddings(
        a.emb_model,
        a.emb_url,
        batch=a.emb_batch,
        truncate_prompt_tokens=a.truncate or DEFAULT_SERVING["truncate"],
        cache_dir=a.cache_dir,
    )
    fmt = TOOL_FORMATS[a.tool_format]
    texts = [fmt(t) for t in tools]
    todo = enc.cache.missing(texts) if enc.cache is not None else list(range(len(texts)))
    if todo:
        enc.encode([texts[i] for i in todo])
    print(
        f"embedded {len(todo)} new or changed texts ({enc.tokens_spent} tokens, {a.emb_model}, "
        f"{a.tool_format}); {len(texts) - len(todo)} already cached"
    )


def _apply(a: argparse.Namespace, listings: list) -> int:
    from toolrank.ingest.sync import load_dir, sync, write_dir

    tools, manifest = load_dir(a.out)
    tools, manifest, diffs = sync(tools, manifest, listings, replace=a.replace, allow_empty=a.allow_empty)
    for d in diffs:
        print(d.line())
    if a.dry_run:
        print(f"dry run: {a.out} not written ({len(tools)} tools after this sync)")
        return int(any(d.error for d in diffs))
    write_dir(a.out, tools, manifest)
    print(f"wrote {len(tools)} tools from {len(manifest)} sources to {a.out}")
    a.emb_url = a.emb_url or os.environ.get("TOOLRANK_EMB_URL")
    if a.emb_url:
        _warm_cache(a, tools)
    return int(any(d.error for d in diffs))


def cmd_ingest_mcp(a: argparse.Namespace) -> int:
    from toolrank.ingest.mcp import load_config, parse_server
    from toolrank.ingest.sync import Listing

    try:
        cfgs = [c for path in a.config or [] for c in load_config(path)] + [
            parse_server(s) for s in a.server or []
        ]
    except ValueError as e:
        sys.exit(str(e))
    if a.only:
        wanted = {n.strip() for n in a.only.split(",")}
        cfgs = [c for c in cfgs if c.name in wanted]
    names = [c.name for c in cfgs]
    if not cfgs:
        sys.exit("no servers: pass --server NAME=URL|COMMAND or --config FILE (and check --only)")
    if len(set(names)) != len(names):
        sys.exit(f"duplicate server names: {sorted({n for n in names if names.count(n) > 1})}")
    _need_mcp("toolrank ingest mcp")
    from toolrank.adapters.mcp_client import fetch_many

    results = asyncio.run(fetch_many(cfgs, timeout=a.timeout))
    listings = [
        Listing(c.name, "mcp", None, error=str(r))
        if isinstance(r, Exception)
        else Listing(c.name, "mcp", r, info={"transport": c.transport})
        for c, r in zip(cfgs, results.values(), strict=True)
    ]
    return _apply(a, listings)


def cmd_ingest_openapi(a: argparse.Namespace) -> int:
    from toolrank.ingest.openapi import MAX_NODES, OpenAPISource, load_spec
    from toolrank.ingest.sync import Listing

    try:
        spec = load_spec(a.spec)
    except (ValueError, RuntimeError, OSError) as e:
        sys.exit(str(e))
    src = OpenAPISource(spec, a.name, origin=a.spec, max_chars=a.max_chars)
    tools = src.list_tools()
    st = src.stats
    print(
        f"{src.name}: {st['operations']} operations, {st['capped']} texts shrunk to {a.max_chars} characters"
        + (f", {st['external_refs']} external $refs stubbed" if st["external_refs"] else "")
        + (f", {st['broken_refs']} broken $refs" if st["broken_refs"] else "")
        + (f", {st['bad_paths']} paths without a leading / skipped" if st["bad_paths"] else "")
        + (f", {st['schemas_cut']} schemas cut at {MAX_NODES:,} nodes" if st["schemas_cut"] else "")
    )
    info = {"origin": src.origin, "base_url": src.base_url}
    return _apply(a, [Listing(src.name, "openapi", tools, info=info)])


def cmd_ingest_drop(a: argparse.Namespace) -> int:
    from toolrank.ingest.sync import drop, load_dir, write_dir

    tools, manifest = load_dir(a.out)
    try:
        tools, manifest, diffs = drop(tools, manifest, a.names)
    except ValueError as e:
        sys.exit(str(e))
    for d in diffs:
        print(f"{d.name}  -{d.removed}")
    write_dir(a.out, tools, manifest)
    print(f"wrote {len(tools)} tools from {len(manifest)} sources to {a.out}")
    return 0


def _add_ingest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True, help="ingest directory (tools.jsonl + sources.json)")
    p.add_argument("--dry-run", action="store_true", help="print the diff, write nothing")
    p.add_argument("--replace", action="store_true", help="let a source replace one of another kind")
    p.add_argument("--allow-empty", action="store_true", help="accept a listing of 0 tools")
    p.add_argument("--tool-format", choices=list(TOOL_FORMATS), default="documentation", help="text to embed")
    _add_encoder_args(p, url=None, model=None, cache_dir=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="toolrank",
        description="Tool retrieval for LLM agents: ingest tools, search them, serve them over MCP and "
        "REST, and benchmark retrievers.",
    )
    p.add_argument("--version", action="version", version=f"toolrank {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="index a tool set and score a query set")
    e.add_argument("--data", required=True, help="directory with tools.jsonl and queries.jsonl")
    e.add_argument("--scorer", choices=["bm25", "dense", "clm", "jev", "cross"], default="bm25")
    e.add_argument("--tool-format", choices=list(TOOL_FORMATS), default=None)
    e.add_argument("--query-format", choices=list(QUERY_FORMATS), default=None)
    e.add_argument(
        "--with-inst", action="store_true", help="keep the task instruction (ToolRet 'w/ inst.' setting)"
    )
    e.add_argument("--tasks", default=None, help="comma-separated task filter")
    e.add_argument(
        "--instruction", default=None, help="use this instruction for every query (implies --with-inst)"
    )
    e.add_argument("--limit", type=int, default=0, help="only the first N queries (smoke tests)")
    e.add_argument("--k", type=int, default=100, help="retrieval depth (ToolRet: 100)")
    e.add_argument("--ks", default="5,10,20", help="reported cut-offs")
    e.add_argument("--batch", type=int, default=64, help="queries per rank() call (latency is per query)")
    e.add_argument("--no-stem", action="store_true", help="BM25 without Snowball stemming")
    _add_encoder_args(e, url="http://127.0.0.1:8090/v1", model="qwen3-8b")
    e.add_argument("--clm-ckpt", default=None)
    e.add_argument("--index", default="numpy", help="vector index for dense/clm: numpy | faiss | pgvector")
    e.add_argument("--index-dir", default=None, help="keep the index on disk here (default: in memory)")
    e.add_argument("--pg-dsn", default=None, help="pgvector: Postgres DSN (default: $TOOLRANK_PG_DSN)")
    e.add_argument(
        "--pg-table", default="toolrank_eval", help="pgvector: not a served catalogue's table (its rows go)"
    )
    e.add_argument("--hybrid", action="store_true", help="fuse BM25 (request without instruction) by RRF")
    e.add_argument("--rrf-k", type=int, default=60, help="RRF constant")
    e.add_argument("--rrf-depth", type=int, default=100, help="list depth taken from each arm")
    e.add_argument("--rrf-weight", type=float, default=1.0, help="weight of the BM25 term (1 = plain RRF)")
    e.add_argument(
        "--server-weight",
        type=float,
        default=0.0,
        help="dense, clm: tool score + this much of the request's cosine with the tool's server (its category)",
    )
    _add_cut_args(e)
    _add_jev_args(e)
    e.add_argument("--device", default=None, help="torch device for the CLM heads")
    e.add_argument("--out", default=None, help="results JSON path")
    e.add_argument(
        "--runs-out",
        default=None,
        help="per-query rows (top-20 ids, P@1, hit@5, NDCG@10) for compare --paired",
    )
    e.set_defaults(fn=cmd_eval)

    c = sub.add_parser("compare", help="markdown table across result files")
    c.add_argument("files", nargs="+")
    c.add_argument(
        "--paired",
        action="store_true",
        help="two runs files (eval --runs-out): sign and paired permutation tests per query",
    )
    c.add_argument("--metrics", default="NDCG@10,Recall@10,Comprehensiveness@10")
    c.add_argument(
        "--cat-macro",
        default=None,
        help="metrics to add as category macro-averages (the ToolRet paper's Average); "
        "default: the first --metrics entry, '' for none",
    )
    c.set_defaults(fn=cmd_compare)

    d = sub.add_parser("data", help="dataset commands")
    ds = d.add_subparsers(dest="data_cmd", required=True)
    pull = ds.add_parser("pull", help="download a benchmark and convert it to JSONL")
    pull.add_argument("dataset", choices=["toolret", "toolret-train", "livemcpbench", "mcp-zero"])
    pull.add_argument("--out", default=None, help="default: data/<dataset> (dashes as underscores)")
    pull.add_argument("--tasks", default=None)
    pull.add_argument(
        "--gen-url", default="http://127.0.0.1:8093/v1", help="mcp-zero: OpenAI-compatible chat endpoint"
    )
    pull.add_argument("--gen-model", default="qwen3-8b-chat", help="mcp-zero: model that writes the queries")
    pull.add_argument("--gen-workers", type=int, default=32, help="mcp-zero: concurrent requests")
    pull.add_argument("--gen-max-tokens", type=int, default=256)
    pull.add_argument(
        "--gen-extra",
        default='{"chat_template_kwargs": {"enable_thinking": false}}',
        help="JSON merged into every chat request (default turns off Qwen3 thinking on vLLM; '{}' for none)",
    )
    pull.set_defaults(fn=cmd_data_pull)
    gq = ds.add_parser(
        "gen-queries",
        help="a dev set of your own: a chat model writes one request per sampled tool of a catalogue",
        description="Make a selection set that shares no query with a benchmark: sample tools evenly over "
        "the sources of an ingest (or benchmark) dir and let a chat model write one request per tool, in "
        "three styles. The result is a benchmark-format dir for eval, finetune --dev and learn --dev.",
    )
    gq.add_argument("--data", required=True, help="the catalogue: a dir with tools.jsonl")
    gq.add_argument("--out", required=True, help="the set's directory (tools.jsonl, queries.jsonl)")
    gq.add_argument("--n", type=int, default=600, help="tools sampled, one request each")
    gq.add_argument("--seed", type=int, default=0)
    gq.add_argument(
        "--styles",
        default="task,step,goal",
        help="request styles, used in turn: task, step, goal, situation (a problem stated without the "
        "operation: harder to match)",
    )
    gq.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="DIR",
        help="a benchmark dir whose queries the set must not repeat",
    )
    gq.add_argument(
        "--tools-per-request",
        type=int,
        default=1,
        metavar="K",
        help="2-4: tasks that need K related tools of one source, all of them gold (much harder; --styles "
        "is not used)",
    )
    gq.add_argument("--gen-url", default="http://127.0.0.1:8093/v1", help="OpenAI-compatible chat endpoint")
    gq.add_argument("--gen-model", default="qwen3-8b-chat", help="the model that writes the requests")
    gq.add_argument("--gen-workers", type=int, default=32, help="concurrent requests")
    gq.add_argument("--gen-max-tokens", type=int, default=256)
    gq.add_argument(
        "--gen-extra",
        default='{"chat_template_kwargs": {"enable_thinking": false}}',
        help="JSON merged into every chat request (default turns off Qwen3 thinking on vLLM; '{}' for none)",
    )
    gq.set_defaults(fn=cmd_data_gen_queries)
    srv = ds.add_parser(
        "server-names",
        help="copy a benchmark set with each tool's server name in its text (the _server sets)",
    )
    srv.add_argument("src", help="a benchmark dir (tools.jsonl + queries.jsonl), e.g. data/mcp_zero")
    srv.add_argument("--out", default=None, help="default: <src>_server")
    srv.set_defaults(fn=cmd_data_server_names)
    syn = ds.add_parser("synth", help="write a small synthetic tool set + queries")
    syn.add_argument("--out", default="data/synthetic")
    syn.add_argument("--n-tools", type=int, default=300)
    syn.add_argument("--n-queries", type=int, default=200)
    syn.add_argument("--seed", type=int, default=7)
    syn.set_defaults(fn=cmd_data_synth)

    ing = sub.add_parser(
        "ingest",
        help="index your own tools: MCP servers and OpenAPI specs",
        description="Sync tools into an ingest directory. With --emb-url (or $TOOLRANK_EMB_URL), also "
        "embed the new and changed tools (unchanged ones are already in the embedding cache).",
    )
    ings = ing.add_subparsers(dest="ingest_cmd", required=True)
    im = ings.add_parser("mcp", help="list tools from MCP servers (stdio or streamable HTTP)")
    im.add_argument("--server", action="append", help="NAME=URL (streamable HTTP) or NAME=COMMAND (stdio)")
    im.add_argument(
        "--config", action="append", help="MCP client config: {'mcpServers': ...} or VS Code {'servers': ...}"
    )
    im.add_argument("--only", default="", help="comma-separated server names to take from the configs")
    im.add_argument("--timeout", type=float, default=60.0, help="seconds per server, start to last page")
    _add_ingest_args(im)
    im.set_defaults(fn=cmd_ingest_mcp)
    io = ings.add_parser("openapi", help="one tool per operation of an OpenAPI 3.x spec (path or URL)")
    io.add_argument("spec")
    io.add_argument("--name", default=None, help="source name (default: slug of info.title)")
    io.add_argument("--max-chars", type=int, default=MAX_CHARS, help="text budget per tool")
    _add_ingest_args(io)
    io.set_defaults(fn=cmd_ingest_openapi)
    idr = ings.add_parser("drop", help="remove whole sources")
    idr.add_argument("names", nargs="+")
    idr.add_argument("--out", required=True)
    idr.set_defaults(fn=cmd_ingest_drop)

    se = sub.add_parser(
        "search",
        help="find the tools for one request in an ingest dir",
        description="Rank the tools of an ingest dir for one request. Defaults: the toolrank "
        "backbone on 127.0.0.1:8091 (toolrank-emb-v0.2), no heads, adaptive K (margin 0.2, max 10), "
        "a persistent index in DIR/index.",
    )
    se.add_argument("request")
    se.add_argument("--json", action="store_true", help="one JSON object instead of a table")
    _add_retrieval_args(se)
    se.set_defaults(fn=cmd_search, scorer=None)

    ca = sub.add_parser(
        "calibrate",
        help="what a search's best score means on this catalogue: confidence and --min-confidence",
        description="Rank requests written for the catalogue's own tools (toolrank data gen-queries --data DIR) "
        "the way search and serve do, with the same flags, and keep their best scores in DIR/calibration.json. "
        "Searches by that first stage, heads and instruction then carry a confidence (the share of these "
        "answerable requests that scored lower), and --min-confidence turns away requests below a share. "
        "Calibrate again after the backbone, the heads (toolrank ab) or the instruction change.",
    )
    ca.add_argument("--requests", required=True, help="a toolrank data gen-queries dir over this catalogue")
    ca.add_argument("--dry-run", action="store_true", help="print the bands, write nothing")
    _add_retrieval_args(ca)
    ca.set_defaults(fn=cmd_calibrate, scorer=None)

    sv = sub.add_parser(
        "serve",
        help="serve an ingest dir to agents: MCP (search_tools, call_tool) and REST",
        description="One MCP server with two tools, search_tools and call_tool, in front of every tool "
        "of an ingest dir; calls go to the MCP servers of --config/--server and to OpenAPI operations "
        "(GET/HEAD unless --allow-write). HTTP by default (MCP at /mcp), or --stdio for desktop clients.",
    )
    _add_retrieval_args(sv)
    sv.add_argument(
        "--config", action="append", default=[], help="MCP client config (+ an optional 'openapi' section)"
    )
    sv.add_argument(
        "--server", action="append", default=[], help="NAME=URL (streamable HTTP) or NAME=COMMAND (stdio)"
    )
    sv.add_argument("--stdio", action="store_true", help="speak MCP over stdin/stdout instead of HTTP")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument(
        "--api-key", default=None, help="bearer token for /mcp and /v1 (default: $TOOLRANK_API_KEY)"
    )
    sv.add_argument(
        "--api-keys",
        default=None,
        metavar="FILE",
        help="JSON {name: key}, one bearer token per client or team; the name is logged as the tenant",
    )
    sv.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="extra Host header value to accept, e.g. the name a proxy or container network uses "
        "(also $TOOLRANK_ALLOWED_HOSTS, comma-separated)",
    )
    sv.add_argument("--allow-write", action="store_true", help="let call_tool send non-GET OpenAPI requests")
    sv.add_argument("--timeout", type=float, default=60.0, help="seconds per backend call and connection")
    sv.add_argument("--usage-log", default=None, help="usage log directory (default: DATA/usage)")
    sv.add_argument("--no-usage-log", action="store_true")
    sv.add_argument(
        "--candidate-share",
        type=float,
        default=0.1,
        help="share of requests answered with DATA/heads/candidate.npz when there is one (sticky per session)",
    )
    sv.add_argument("--log-text", action="store_true", help="also log request and error text")
    sv.add_argument(
        "--mask-pii", action="store_true", help="with --log-text: mask e-mail, phone, card and IBAN numbers"
    )
    sv.set_defaults(fn=cmd_serve, scorer=None)

    ft = sub.add_parser(
        "finetune",
        help="train heads on request -> tool pairs; the epoch is picked on a dev set",
        description="Embed the pairs once (only what the cache lacks), train heads on the frozen "
        "backbone's vectors, pick the epoch on --dev (a benchmark-format set that is not reported), "
        "then run toolrank eval with the saved heads on --dev and every --eval set. Defaults: the "
        "setting of the released heads (skip heads, lr 1e-5, batch 512, 5 epochs, in-batch negatives "
        "only) and the encoder of the released heads (the base model as qwen3-emb on 8091, "
        "documentation + instruct_query, truncate 8192), whose heads search and serve take with "
        "--emb-model qwen3-emb.",
    )
    ft.add_argument("--data", required=True, help="pairs.jsonl (data pull toolret-train, or your own)")
    ft.add_argument("--dev", required=True, help="benchmark-format dir the epoch is picked on; not reported")
    ft.add_argument("--eval", action="append", default=[], help="benchmark-format dir to evaluate")
    ft.add_argument("--out", required=True, help="the heads, a torch .pt")
    ft.add_argument("--npz", default=None, help="also export the packaged fp16 .npz")
    ft.add_argument(
        "--init-ckpt",
        default=None,
        help=".pt, .npz or 'default' (the packaged heads); default: fresh skip heads",
    )
    ft.add_argument("--name", default=None, help="the report's name (default: --out's stem)")
    ft.add_argument("--results", default="results", help="where the report and eval JSONs go")
    ft.add_argument(
        "--report", default=None, help="the report's path (default: RESULTS/finetune_<name>.json)"
    )
    ft.add_argument("--select", choices=["ndcg10", "ndcg10-cat"], default="ndcg10", help="dev metric")
    ft.add_argument(
        "--curve", action="store_true", help="score the eval sets every epoch too (reported only)"
    )
    ft.add_argument("--embed-only", action="store_true", help="fill the embedding cache and stop (no torch)")
    ft.add_argument("--tool-format", choices=list(TOOL_FORMATS), default="documentation")
    ft.add_argument("--query-format", choices=list(QUERY_FORMATS), default="instruct_query")
    ft.add_argument("--instruction", default=None, help="for pairs without one (default: the serving one)")
    ft.add_argument("--keep-bare", action="store_true", help="leave pairs without an instruction bare")
    ft.add_argument(
        "--backbone", default=None, help="recorded in the heads' cfg (default: what --emb-model names)"
    )
    ft.add_argument("--n-train", type=int, default=0, help="training pairs (0 = all)")
    ft.add_argument(
        "--n-val", type=int, default=0, help="held-out training pairs: a diagnostic, never selected on"
    )
    ft.add_argument("--seed", type=int, default=0)
    ft.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="seed of the train/val split, so the same pairs split alike across training seeds "
        "(default: --seed)",
    )
    ft.add_argument("--epochs", type=int, default=5)
    ft.add_argument("--batch", type=int, default=512)
    ft.add_argument("--lr", type=float, default=1e-5, help="skip heads collapse at 3e-4 and above")
    ft.add_argument(
        "--neg", type=int, default=0, help="mined negatives per pair (ToolRet's cost up to 10 NDCG points)"
    )
    ft.add_argument("--neg-filter", type=float, default=None)
    ft.add_argument("--weight-decay", type=float, default=0.01)
    ft.add_argument("--warmup", type=float, default=0.05)
    ft.add_argument("--width", type=int, default=None, help="fresh heads' hidden width (default 1536)")
    ft.add_argument("--depth", type=int, default=None, help="fresh heads' depth (default 3)")
    ft.add_argument("--no-skip", action="store_true", help="fresh heads without the x + MLP(x) skip")
    ft.add_argument("--freeze-action", action="store_true", help="train the state head only")
    ft.add_argument("--device", default=None)
    _add_encoder_args(ft, url=None, model=None)
    ft.set_defaults(fn=cmd_finetune, emb_batch=128, truncate=8192)

    ln = sub.add_parser(
        "learn",
        help="train the heads on what the usage log says agents called; publish if it helps",
        description="From an ingest dir toolrank serve has served: the log's searches and calls become "
        "request -> tool pairs (a call that ended ok is a positive, a tool_error a weak one, tools shown "
        "but not called are hard negatives), the requests' vectors come from DATA/cache through the log's "
        "key, never their text, and the heads train from the served ones. The newest 20% of requests "
        "are the dev set (Recall@5 of the called tool over the catalogue); the heads are written only "
        "when they beat the starting ones there, and, with --dev, do not fall on a benchmark set.",
    )
    ln.add_argument("--data", required=True, help="the ingest dir: tools.jsonl, cache/, usage/")
    ln.add_argument(
        "--out",
        default=None,
        help="the .npz to write (default: DATA/heads/candidate.npz, or tenants/<name>/ with --tenant: "
        "a running server gives it a share of the requests)",
    )
    ln.add_argument(
        "--replace-candidate", action="store_true", help="train even though a candidate is still being judged"
    )
    ln.add_argument(
        "--replay", default=None, help="general pairs.jsonl mixed into training, against forgetting"
    )
    ln.add_argument("--replay-n", type=int, default=1000, help="how many of --replay's pairs")
    ln.add_argument(
        "--dev", default=None, help="benchmark-format dir scored alongside: a guard against forgetting"
    )
    ln.add_argument(
        "--init",
        default="default",
        help="heads to start from: default (the served ones: DATA/heads/current.npz when there, else "
        "the packaged ones when they fit the backbone, else fresh identity heads), a path, or none",
    )
    ln.add_argument("--since", default=None, help="only searches from this ISO date or timestamp on")
    ln.add_argument("--tenant", default=None, help="only one API key's searches (its name)")
    ln.add_argument("--strict", action="store_true", help="a tool_error call is not a (weak) positive")
    ln.add_argument("--min-pairs", type=int, default=20, help="fewer usable requests: nothing is trained")
    ln.add_argument("--dev-share", type=float, default=0.2, help="share of the newest requests held out")
    ln.add_argument("--max-drop", type=float, default=0.5, help="NDCG@10 points the --dev set may lose")
    ln.add_argument(
        "--dry-run", action="store_true", help="mine and match the vectors, train nothing (no torch)"
    )
    ln.add_argument("--name", default=None, help="the report's name (default: the output's stem)")
    ln.add_argument("--results", default="results", help="where the report goes")
    ln.add_argument("--report", default=None, help="the report's path (default: RESULTS/learn_<name>.json)")
    ln.add_argument("--tool-format", choices=list(TOOL_FORMATS), default="documentation")
    ln.add_argument(
        "--query-format", choices=list(QUERY_FORMATS), default="instruct_query", help="for --dev's queries"
    )
    ln.add_argument(
        "--backbone", default=None, help="recorded in the heads' cfg (default: what --emb-model names)"
    )
    ln.add_argument("--epochs", type=int, default=3)
    ln.add_argument("--batch", type=int, default=256)
    ln.add_argument("--lr", type=float, default=1e-5)
    ln.add_argument("--neg", type=int, default=5, help="shown-but-not-called tools per request in a batch")
    ln.add_argument(
        "--neg-filter", type=float, default=0.95, help="drop negatives the start scores like a positive"
    )
    ln.add_argument("--weight-decay", type=float, default=0.01)
    ln.add_argument("--warmup", type=float, default=0.05)
    ln.add_argument("--seed", type=int, default=0)
    ln.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="seed of the replay sample, so replay picks alike across training seeds (default: --seed)",
    )
    ln.add_argument("--device", default=None)
    _add_encoder_args(ln, url=None, model=None, cache_dir=None)
    ln.set_defaults(fn=cmd_learn, emb_batch=128)

    ab = sub.add_parser(
        "ab",
        help="compare the candidate heads with the served ones on the usage log; promote or roll back",
        description="Since DATA/heads/candidate.npz appeared, a share of the requests was answered with "
        "it. Per arm: the searches, how many led to a call, and how high the called tool stood (mrr: "
        "the mean of 1/rank over all the arm's searches). The candidate becomes current.npz when its "
        "mrr is --margin above the control's with --min-searches on both sides, is set aside when it is "
        "that much below, and keeps running otherwise. A running server follows the files.",
    )
    ab.add_argument("--data", required=True, help="the ingest dir: usage/ and heads/")
    ab.add_argument("--tenant", default=None, help="one API key's heads (DATA/heads/tenants/<name>)")
    ab.add_argument("--min-searches", type=int, default=100, help="per arm, before anything is decided")
    ab.add_argument("--margin", type=float, default=0.01, help="the mrr difference that decides")
    ab.add_argument("--since", default=None, help="ISO timestamp (default: when the candidate appeared)")
    ab.add_argument("--dry-run", action="store_true", help="say the decision, move nothing")
    ab.add_argument("--promote", dest="force", action="store_const", const="promote", help="promote now")
    ab.add_argument("--rollback", dest="force", action="store_const", const="rollback", help="roll back now")
    ab.add_argument("--results", default="results", help="where the report goes")
    ab.set_defaults(fn=cmd_ab, force=None)

    hd = sub.add_parser("heads", help="head checkpoints")
    hds = hd.add_subparsers(dest="heads_cmd", required=True)
    ex = hds.add_parser("export", help="a torch .pt checkpoint -> the .npz that runs without torch")
    ex.add_argument("src")
    ex.add_argument("dst")
    ex.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ex.add_argument("--backbone", default=None, help="serving defaults stored in the checkpoint cfg")
    ex.add_argument("--tool-format", default=None)
    ex.add_argument("--query-format", default=None)
    ex.add_argument("--truncate", type=int, default=None)
    ex.add_argument("--instruction", default=None)
    ex.set_defaults(fn=cmd_heads_export)
    hp = hds.add_parser(
        "pull",
        help="download the packaged heads into the cache (sha256-checked); search and serve use them on "
        "the backbone they were trained on (--emb-model qwen3-emb or qwen3-emb-fp8)",
    )
    hp.add_argument(
        "--url", default=None, help="a mirror of the file (default: TOOLRANK_HEADS_URL, then the release)"
    )
    hp.set_defaults(fn=cmd_heads_pull)

    f = sub.add_parser("formats", help="list tool / query text formats")
    f.set_defaults(fn=cmd_formats)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
