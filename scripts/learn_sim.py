"""Simulated traffic for the learning loop (Faz 2 Hafta 4): does ``toolrank learn`` help?

A benchmark stands in for a served catalogue. ``split`` makes an ingest-shaped dir of it: every
tool, a seeded share of the queries as the traffic to come (``traffic.jsonl``) and the rest held out
in benchmark format (``heldout/``, what ``toolrank eval`` reads; ``heldout_new/`` is its harder
part: the queries none of whose gold tools is gold for a traffic query). ``traffic`` then answers queries
the way ``toolrank serve`` does (the same retriever, cut rule, heads dir and usage log; the flags
after ``--`` are serve's) and plays the agent: it calls the gold tools that were shown, each call
linked to its search. A gold tool that was not shown is never called, so the log can only teach what
the served ranking already surfaced, as in production. With ``--noise P`` the agent instead calls
the first shown tool that is not gold, with probability P, and that call ends ``tool_error``.

Run on the held-out queries after ``toolrank learn`` wrote a candidate, ``traffic`` is the A/B half:
``--candidate-share 0.5`` sends half of the sessions to the candidate and ``toolrank ab`` reads the
log back.

uv run python scripts/learn_sim.py split --bench data/toolret --out data/sim/toolret
uv run python scripts/learn_sim.py traffic --queries data/sim/toolret/traffic.jsonl -- \
    --data data/sim/toolret --emb-url http://127.0.0.1:8091/v1 --cache-dir .cache/toolrank
toolrank learn --data data/sim/toolret --cache-dir .cache/toolrank --dev data/mcp_zero_server
toolrank eval --data data/sim/toolret/heldout --scorer clm --clm-ckpt data/sim/toolret/heads/candidate.npz ...
uv run python scripts/learn_sim.py traffic --queries data/sim/toolret/heldout/queries.jsonl --wait-candidate -- \
    --data data/sim/toolret --emb-url http://127.0.0.1:8091/v1 --cache-dir .cache/toolrank --candidate-share 0.5
toolrank ab --data data/sim/toolret --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from toolrank.datasets.jsonl import load_queries, write_queries
from toolrank.domain import Query


def split(bench: Path, out: Path, share: float, seed: int) -> dict[str, int]:
    """``out`` as an ingest dir over the benchmark's tools (linked, not copied: the same text, the
    same cache keys), ``traffic.jsonl``, ``heldout/`` and ``heldout_new/`` (held-out queries about
    tools the traffic never asks for); queries without a gold tool are left out."""

    def gold(q: Query) -> set[str]:
        return {t for t, r in q.qrels.items() if r > 0}

    queries = [q for q in load_queries(bench / "queries.jsonl") if gold(q)]
    random.Random(seed).shuffle(queries)
    n = round(len(queries) * share)
    asked = set().union(*(gold(q) for q in queries[:n])) if n else set()
    new = [q for q in queries[n:] if not gold(q) & asked]
    tools = (bench / "tools.jsonl").resolve()
    for d in (out, out / "heldout", out / "heldout_new"):
        d.mkdir(parents=True, exist_ok=True)
        link = d / "tools.jsonl"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(tools, link)
    write_queries(out / "traffic.jsonl", queries[:n])
    write_queries(out / "heldout" / "queries.jsonl", queries[n:])
    write_queries(out / "heldout_new" / "queries.jsonl", new)
    return {"traffic": n, "heldout": len(queries) - n, "heldout_new": len(new)}


def play(
    retriever: Any, usage: Any, queries: list[Query], *, noise: float = 0.0, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Answer and log every query, call what the simulated agent would; -> per arm: searches, the
    share with a gold tool shown, with every gold tool shown (``complete``), top-1, ``mrr`` (0 for a
    search with no gold tool shown) and the mean number of tools shown."""
    rng = random.Random(seed)
    arms: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for q in queries:
        session = f"sim-{q.id}"
        res = retriever.search(q.text, instruction=q.instruction, arm_key=session)
        sid = usage.search(res, session=session, via="sim", heads=res.heads, client="learn_sim", arm=res.arm)
        gold = {t for t, r in q.qrels.items() if r > 0}
        shown = [h.id for h in res.hits]
        right = [t for t in shown if t in gold]
        wrong = [t for t in shown if t not in gold]
        m = arms[res.arm]
        m["searches"] += 1
        m["shown"] += len(shown)
        m["complete"] += gold <= set(shown)
        if right:
            rank = shown.index(right[0]) + 1
            m["gold_shown"] += 1
            m["top1"] += rank == 1
            m["mrr"] += 1.0 / rank
        calls = [(t, "ok") for t in right]
        if wrong and rng.random() < noise:  # the agent's mistake: a tool that does not do the job
            calls = [(wrong[0], "tool_error")]
        for tool, outcome in calls:
            usage.call(
                tool=tool,
                kind="sim",
                session=session,
                via="sim",
                outcome=outcome,
                took_ms=0.0,
                search_id=sid,
                client="learn_sim",
            )
    return {
        arm: {"searches": int(m["searches"])}
        | {k: round(m[k] / m["searches"], 4) for k in ("gold_shown", "complete", "top1", "mrr", "shown")}
        for arm, m in sorted(arms.items())
    }


def wait_candidate(retriever: Any, timeout: float) -> None:
    """Block until the candidate heads answer (the server builds them in the background)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        retriever.pick(arm_key="warm-up")  # a request is what makes the server look at the file
        st = retriever.status().get("heads", {}).get("candidate")
        if st and st["error"]:
            sys.exit(f"candidate heads: {st['error']}")
        if st and st["ready"]:
            return
        time.sleep(0.5)
    sys.exit(f"no candidate heads after {timeout:.0f} s: is DATA/heads/candidate.npz there?")


def main() -> None:
    argv = sys.argv[1:]
    rest: list[str] = []
    if "--" in argv:
        rest, argv = argv[argv.index("--") + 1 :], argv[: argv.index("--")]
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split", help="a benchmark as an ingest dir: traffic.jsonl and heldout/")
    sp.add_argument("--bench", required=True, help="benchmark dir (tools.jsonl, queries.jsonl)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--traffic-share", type=float, default=0.7)
    sp.add_argument("--seed", type=int, default=0)
    tp = sub.add_parser("traffic", help="serve queries, log them, call the gold tools shown")
    tp.add_argument("--queries", required=True, help="queries.jsonl to replay, in file order")
    tp.add_argument("--limit", type=int, default=0, help="only the first N queries")
    tp.add_argument("--noise", type=float, default=0.0, help="probability of a wrong call (tool_error)")
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument(
        "--wait-candidate", action="store_true", help="start once DATA/heads/candidate.npz answers"
    )
    tp.add_argument("--report", default=None, help="write the per-arm summary here as JSON")
    a = ap.parse_args(argv)
    if a.cmd == "split":
        print(json.dumps(split(Path(a.bench), Path(a.out), a.traffic_share, a.seed)))
        return
    if not rest:
        sys.exit("usage: learn_sim.py traffic --queries FILE -- <toolrank serve flags>")

    from toolrank.build import build_retriever
    from toolrank.cli import _data_cache, build_parser
    from toolrank.usage import UsageLog

    s = build_parser().parse_args(["serve", *rest])
    data = Path(s.data).resolve()
    s.data = str(data)
    s.index_dir = str(Path(s.index_dir).resolve()) if s.index_dir else str(data / "index")
    _data_cache(s, data)
    retriever = build_retriever(s, notify=lambda msg: print(msg, file=sys.stderr, flush=True))
    queries = load_queries(a.queries)
    queries = queries[: a.limit] if a.limit else queries
    if a.wait_candidate:
        wait_candidate(retriever, 600.0)
    t0 = time.perf_counter()
    arms = play(retriever, UsageLog(data / "usage"), queries, noise=a.noise, seed=a.seed)
    report = {
        "data": str(data),
        "queries": str(a.queries),
        "n": len(queries),
        "noise": a.noise,
        "seed": a.seed,
        "rule": retriever.describe(),
        "took_s": round(time.perf_counter() - t0, 1),
        "arms": arms,
    }
    print(json.dumps(report, indent=2))
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
