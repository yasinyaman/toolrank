"""Learning from the usage log: ``toolrank learn`` (``run``).

A served catalogue's usage log (``usage.UsageLog``, schema v3) says which tools each request
retrieved and which one the agent then called, but never the request's text: ``emb_hmac`` is the
keyed digest of the request's embedding-cache key. The heads learn from it all the same. The
request's backbone vector is in the ingest dir's embedding cache, found by digesting the cache's
keys with the log's key (``DATA/usage/.key``), and the tools' vectors come from the catalogue's
text through the same cache. No text leaves the machine and none is reconstructed.

Pairs (``mine``): a search with linked calls. A call that ended ``ok`` makes its tool a positive,
``tool_error`` a weak positive (kept unless ``strict``: the tool was the one to try), and the
tools the search showed that no call of that search used are hard-negative candidates; searches of
the same request merge, and the other outcomes (``refused``, ``protocol_error``, ``timeout``,
``unknown_tool``) say nothing about the tools. ``TrainConfig.neg_filter`` still drops the
negatives the starting heads score like a positive: Phase 0's lesson about equivalent tools.

Selection (``split``): the newest share of the requests, by time, is the dev set, scored as the
share of requests whose called tool is in the top 5 of the whole catalogue (``log.Recall@5``); a
benchmark-format ``dev`` set can be scored alongside as a guard against forgetting. The starting
heads compete as epoch 0, and the result is published (an ``.npz`` next to the log) only when it
beats them on the log's dev set without falling below them on the benchmark by more than
``max_drop`` NDCG@10 points. ``torch`` is imported inside functions (``toolrank[clm]``).

Forgetting (``replay``): general request -> tool pairs (``pairs.jsonl``, e.g. ToolRet's training
set) can be mixed into the training batches, so heads learned from one catalogue's traffic keep
what the released ones knew; they are embedded through the same cache, and selection stays on the
log. The mix is a seeded reservoir sample over the whole file (the head of a 3.3 GB set is not a
sample of it), its positives only — the mined negatives such files carry cost more than they
taught (Phase 0) — without the requests a ``--dev`` set asks about.

A/B (``judge``, ``decide``, ``apply``; ``toolrank ab``): published heads go to
``DATA/heads/candidate.npz`` (``tenants/<name>/`` for one API key), where a running server gives
them a sticky share of the requests (``retriever.Retriever.pick``) and logs which arm answered. The
log then says how each arm did: of its searches, how many led to a call that ended ``ok`` or
``tool_error``, and how high the called tool stood (``mrr``: the mean of 1/rank over all the arm's
searches, 0 for a search nobody acted on). The candidate is promoted to ``current.npz`` when its
``mrr`` beats the control's by ``margin`` with ``min_searches`` on both sides, rolled back when it is
that much worse, and left running otherwise. Both moves are file renames the server picks up.
"""

from __future__ import annotations

import hmac
import os
import random
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from toolrank.domain import TrainPair
from toolrank.finetune import Batches, TrainConfig, curve_metrics, project, recall_at, train_heads
from toolrank.usage import may_learn_from, read_events

POSITIVE, WEAK = "ok", "tool_error"
K = 5  # the log's own metric: the called tool among the top K of the catalogue
SCHEMA = 3  # the first log schema with emb_hmac


def _reservoir(pairs: Iterable[TrainPair], n: int, seed: int) -> list[TrainPair]:
    """``n`` pairs sampled over the whole file, every pair equally likely, without holding it in
    memory: the head of a 3.3 GB training set is not a sample of it. Pairs without a positive count
    for nothing."""
    rng = random.Random(seed)
    out: list[TrainPair] = []
    seen = 0
    for p in pairs:
        if not p.positives:
            continue
        seen += 1
        if len(out) < n:
            out.append(p)
        elif (j := rng.randrange(seen)) < n:
            out[j] = p
    return out


@dataclass(frozen=True)
class LogPair:
    """One request of the log, in tool ids: what the agent called after it and what it passed over."""

    state: str  # the request's emb_hmac
    positives: tuple[str, ...]  # called, ok
    weak: tuple[str, ...]  # called, tool_error
    negatives: tuple[str, ...]  # shown by a search of this request, never called
    ts: str  # the earliest search
    tenants: tuple[str, ...]


_WRAPPERS = ("rerank[", "jev[", "hybrid[")  # names of scorers around a first stage: <kind>[...]/<first stage>


def _served_by_other(e: dict[str, Any], model: str) -> bool:
    """Whether a backbone other than ``model`` served this search: its ``model`` field, else (logs
    before the field, or a second stage's before it was filled) the first stage's ``emb/<model>/``
    in the scorer's name; a name without one (keywords) says nothing."""
    if e.get("model") is not None:
        return bool(e["model"] != model)
    name = str(e.get("scorer") or "")
    while name.startswith(_WRAPPERS):  # skip the bracket, with the brackets inside it
        depth, end = 0, len(name)
        for n, ch in enumerate(name):
            depth += (ch == "[") - (ch == "]")
            if ch == "]" and depth == 0:
                end = n
                break
        name = name[end + 2 :]
    at = name.find("emb/")
    return at >= 0 and not name[at + 4 :].startswith(model + "/")


def mine(
    events: Iterable[dict[str, Any]],
    *,
    strict: bool = False,
    since: str | None = None,
    tenant: str | None = None,
    model: str | None = None,
) -> tuple[list[LogPair], dict[str, int]]:
    """Searches with linked calls -> one ``LogPair`` per request, oldest first, and the counts of
    what was used and what was passed over (``since``: an ISO date or timestamp, searches from it
    on; ``tenant``: one API key's searches; ``model``: only searches this backbone served — a log
    can hold requests another model answered, and their vectors do not belong with its tools).
    Searches logged without the ``model`` field are told by their scorer's name."""
    counts: Counter[str] = Counter()
    searches: dict[str, dict[str, Any]] = {}
    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        kind = e.get("event")
        if kind == "search":
            counts["searches"] += 1
            if not isinstance(e.get("id"), str):
                counts["searches_malformed"] += 1  # a line no version of the log writes
            elif e.get("v", 0) < SCHEMA or not e.get("emb_hmac"):
                counts["searches_without_vector"] += 1  # an older schema, or a keyword answer
            elif since and str(e.get("ts", "")) < since:
                counts["searches_before_since"] += 1
            elif tenant and e.get("tenant") != tenant:
                counts["searches_of_other_tenants"] += 1
            elif model and _served_by_other(e, model):
                counts["searches_of_other_models"] += 1
            elif not may_learn_from(e.get("scorer")):
                counts["searches_with_jev"] += 1  # the provider's terms keep them out of training
            else:
                searches[e["id"]] = e
        elif kind == "call":
            counts["calls"] += 1
            if e.get("search_id"):
                calls[e["search_id"]].append(e)
            else:
                counts["calls_unlinked"] += 1
    requests: dict[str, dict[str, Any]] = {}
    for sid, s in searches.items():
        linked = calls.get(sid, [])
        if not linked:
            counts["searches_without_calls"] += 1
            continue
        counts["calls_linked"] += len(linked)
        shown = [t for t, _ in (s.get("results") or [])[: int(s.get("shown") or 0)]]
        shown += [t for t in s.get("added") or [] if t not in shown]  # co-use partners were shown too
        called, ok, weak = set(), set(), set()
        for c in linked:
            called.add(c["tool"])
            if c.get("outcome") == POSITIVE:
                ok.add(c["tool"])
            elif c.get("outcome") == WEAK:
                weak.add(c["tool"])
            else:
                counts["calls_saying_nothing"] += 1
        counts["positives_not_shown"] += sum(1 for t in ok | weak if t not in shown)
        r = requests.setdefault(
            s["emb_hmac"], {"pos": set(), "weak": set(), "neg": set(), "ts": s["ts"], "tenants": set()}
        )
        r["pos"] |= ok
        r["weak"] |= weak
        r["neg"] |= {t for t in shown if t not in called}
        r["ts"] = min(r["ts"], s["ts"])
        if s.get("tenant"):
            r["tenants"].add(s["tenant"])
    counts["requests"] = len(requests)
    pairs: list[LogPair] = []
    for state, r in requests.items():
        weak = () if strict else tuple(sorted(r["weak"] - r["pos"]))
        if not r["pos"] and not weak:
            counts["requests_without_positive"] += 1
            continue
        taken = r["pos"] | r["weak"]  # called, whatever came of it: never a negative
        pairs.append(
            LogPair(
                state,
                tuple(sorted(r["pos"])),
                weak,
                tuple(sorted(r["neg"] - taken)),
                r["ts"],
                tuple(sorted(r["tenants"])),
            )
        )
    counts["pairs"] = len(pairs)
    counts["weak_positives"] = sum(len(p.weak) for p in pairs)
    counts["negatives"] = sum(len(p.negatives) for p in pairs)
    return sorted(pairs, key=lambda p: p.ts), dict(counts)


def state_vectors(cache: str | Path, key: bytes, wanted: Iterable[str]) -> dict[str, np.ndarray]:
    """The requests' backbone vectors: each key of the embedding cache digested with the log's key
    and kept when the digest is a wanted ``emb_hmac``; rows L2-normalised, as the encoder returns
    them. The cache is opened read-only, and its keys are read before any vector: a cache that has
    served a large catalogue is gigabytes of vectors, of which only the requests' are needed."""
    from toolrank.adapters.embeddings_api import l2_normalize

    want, out = set(wanted), {}
    if not want or not Path(cache).exists():
        return out
    db = sqlite3.connect(f"file:{Path(cache).resolve()}?mode=ro", uri=True)
    try:
        found: dict[str, str] = {}
        for (k,) in db.execute("SELECT key FROM emb"):
            digest = hmac.new(key, k.encode("utf-8"), sha256).hexdigest()
            if digest in want:
                found[k] = digest
        keys = list(found)
        for start in range(0, len(keys), 900):  # SQLite variable limit
            chunk = keys[start : start + 900]
            q = f"SELECT key, dim, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})"
            for k, dim, vec in db.execute(q, chunk):
                out[found[k]] = l2_normalize(np.frombuffer(vec, dtype=np.float32)[:dim].copy())
    finally:
        db.close()
    return out


def split(pairs: Sequence[LogPair], dev_share: float) -> tuple[list[LogPair], list[LogPair]]:
    """The newest ``dev_share`` of the requests (by their first search) is the dev set; at least one
    request when there are two or more."""
    ordered = sorted(pairs, key=lambda p: p.ts)
    n_dev = min(len(ordered) - 1, max(1, round(len(ordered) * dev_share))) if len(ordered) > 1 else 0
    return ordered[: len(ordered) - n_dev], ordered[len(ordered) - n_dev :]


def batches(
    pairs: Sequence[LogPair], states: dict[str, np.ndarray], index: dict[str, int], tool_vecs: np.ndarray
) -> Batches:
    """Pairs as ``finetune.Batches`` over the whole catalogue: positives are the called tools
    (weak ones included), negatives the shown-but-not-called ones; tools no longer in the catalogue
    are left out."""
    return Batches(
        np.stack([states[p.state] for p in pairs])
        if pairs
        else np.zeros((0, tool_vecs.shape[1]), np.float32),
        tool_vecs,
        [[index[t] for t in p.positives + p.weak if t in index] for p in pairs],
        [[index[t] for t in p.negatives if t in index] for p in pairs],
    )


@dataclass
class Job:
    """What ``toolrank learn`` was asked to do (``cli.cmd_learn`` fills it from the flags)."""

    data: Path  # the ingest dir: tools.jsonl, cache/, usage/
    out: Path  # the .npz to publish; its .pt goes next to it
    dev: Path | None = None  # a benchmark-format set scored alongside: a guard, never selected on
    init: str | None = "default"  # the heads to start from: the packaged/served ones, a path, or None
    emb_url: str = "http://127.0.0.1:8091/v1"
    emb_model: str = "qwen3-emb"
    emb_batch: int = 128
    truncate: int | None = 8192
    cache_dir: str | None = None  # default: DATA/cache, the one serve wrote
    tool_format: str = "documentation"
    query_format: str = "instruct_query"  # the benchmark dev set's queries only
    backbone: str = ""
    since: str | None = None
    tenant: str | None = None
    strict: bool = False
    min_pairs: int = 20
    dev_share: float = 0.2
    max_drop: float = 0.5  # NDCG@10 points the benchmark dev may lose
    dry_run: bool = False
    data_seed: int = 0  # the replay sample's seed, kept apart from the training one
    replay: Path | None = None  # general pairs.jsonl mixed into training, against forgetting
    replay_n: int = 0
    instruction: str = ""  # given to replay pairs without one (the serving instruction)
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(epochs=3, batch=256, neg_per_pair=5, lr=1e-5, neg_filter=0.95)
    )


def _fits(path: Path, emb_model: str | None) -> bool:
    """Whether heads promoted under an older default (the cfg names their backbone) belong on
    ``emb_model``; a file whose cfg cannot be read is left to the loader to report."""
    import json

    import numpy as np

    from toolrank.build import heads_mismatch

    try:
        with np.load(path, allow_pickle=False) as z:
            trained_on = json.loads(str(z["cfg"])).get("backbone")
    except (OSError, ValueError, KeyError):
        return True
    return heads_mismatch(trained_on, emb_model) is None


def resolve_init(
    init: str | None,
    emb_model: str | None = None,
    data: str | Path | None = None,
    tenant: str | None = None,
) -> str | None:
    """``default`` -> the heads a server would serve on ``emb_model``: the promoted ones
    (``DATA/heads/current.npz``, or the tenant's own) when there are some and they were trained on
    that backbone, else the packaged (or
    ``TOOLRANK_HEADS``) ones, or none on a backbone they do not belong on; a path as is; None ->
    fresh skip heads (identity at the start, so epoch 0 is the backbone alone)."""
    if init == "default":
        import os

        from toolrank.adapters.heads_np import default_heads
        from toolrank.build import packaged_heads_fit

        if data is not None:
            # what a server would serve: the tenant's own heads, else the promoted ones
            homes = [heads_home(Path(data), tenant)] if tenant else []
            homes.append(heads_home(Path(data)))
            for home in homes:
                current = home / CURRENT
                if current.exists() and _fits(current, emb_model):
                    return str(current)
        if not packaged_heads_fit(emb_model) and not os.environ.get("TOOLRANK_HEADS"):
            return None
        return str(default_heads())
    return init or None


def run(job: Job, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Mine the log, find the vectors, and (unless ``dry_run``) train, select and decide;
    -> the report. ``decision`` is ``published``, ``no improvement``, ``benchmark dropped`` or
    ``not enough pairs``; only the first writes the heads."""
    from toolrank.adapters.embeddings_api import OpenAIEmbeddings
    from toolrank.datasets.jsonl import load_queries, load_tools
    from toolrank.formats import query_format, tool_format

    usage_dir, key_path = job.data / "usage", job.data / "usage" / ".key"
    if not key_path.exists():
        raise FileNotFoundError(f"{usage_dir}: no usage log here (toolrank serve writes one, with its .key)")
    events = read_events(usage_dir)
    pairs, counts = mine(events, strict=job.strict, since=job.since, tenant=job.tenant, model=job.emb_model)
    tools = load_tools(job.data / "tools.jsonl")
    index = {t.id: n for n, t in enumerate(tools)}
    cache_dir = Path(job.cache_dir) if job.cache_dir else job.data / "cache"
    states = state_vectors(cache_dir / "embeddings.sqlite", key_path.read_bytes(), {p.state for p in pairs})
    usable: list[LogPair] = []
    for p in pairs:
        if p.state not in states:
            counts["requests_without_vector"] = counts.get("requests_without_vector", 0) + 1
        elif not any(t in index for t in p.positives + p.weak):
            counts["requests_whose_tools_left"] = counts.get("requests_whose_tools_left", 0) + 1
        else:
            usable.append(p)
    report: dict[str, Any] = {
        "data": str(job.data),
        "events": len(events),
        "counts": counts,
        "pairs": len(usable),
        "tools": len(tools),
        "since": job.since,
        "tenant": job.tenant,
        "strict": job.strict,
    }
    log(
        f"{len(events):,} events: {counts.get('searches', 0):,} searches, {counts.get('calls', 0):,} calls -> "
        f"{counts.get('requests', 0):,} requests, {len(usable):,} usable pairs "
        f"({counts.get('weak_positives', 0)} weak positives, {counts.get('negatives', 0)} negatives)"
    )
    if len(usable) < job.min_pairs:
        report["decision"] = "not enough pairs"
        log(f"not enough pairs: {len(usable)} of the {job.min_pairs} needed; nothing trained")
        return report
    train_pairs, dev_pairs = split(usable, job.dev_share)
    report["split"] = {"train": len(train_pairs), "dev": len(dev_pairs), "dev_from": dev_pairs[0].ts}
    log(f"train {len(train_pairs)} requests, dev {len(dev_pairs)} (the newest, from {dev_pairs[0].ts})")
    if len(dev_pairs) < 50:
        report.setdefault("warnings", []).append(
            f"{len(dev_pairs)} dev requests: Recall@{K} moves in steps of {1 / len(dev_pairs):.2f}"
        )
    if job.dry_run:
        report["decision"] = "dry run"
        return report

    enc = OpenAIEmbeddings(
        job.emb_model,
        job.emb_url,
        batch=job.emb_batch,
        truncate_prompt_tokens=job.truncate,
        cache_dir=cache_dir,
    )
    tf, qf = tool_format(job.tool_format), query_format(job.query_format)
    tool_vecs = enc.encode([tf(t) for t in tools])
    hidden = next(iter(states.values())).shape[0]
    if tool_vecs.shape[1] != hidden:
        raise ValueError(
            f"the requests' vectors have {hidden} dimensions and the catalogue's {tool_vecs.shape[1]}: "
            "the log and the encoder flags name different backbones"
        )
    report["tokens_spent"] = enc.tokens_spent  # 0 when every catalogue text was in the cache
    train_b, dev_b = (
        batches(train_pairs, states, index, tool_vecs),
        batches(dev_pairs, states, index, tool_vecs),
    )
    if job.replay is not None and job.replay_n > 0:
        from toolrank.datasets.jsonl import iter_pairs
        from toolrank.finetune import asks_like, index_tools, pair_texts, with_instruction

        # a pair the guard set asks about would make the guard meaningless: out before the sample,
        # so the sample still has replay_n pairs
        asked = asks_like(load_queries(job.dev / "queries.jsonl")) if job.dev is not None else None
        dropped = 0

        def clean(pairs: Iterable[TrainPair]) -> Iterator[TrainPair]:
            nonlocal dropped
            for p in pairs:
                if asked is not None and asked(p):
                    dropped += 1
                else:
                    yield p

        picked = _reservoir(clean(iter_pairs(job.replay)), job.replay_n, job.data_seed)
        picked, _ = with_instruction(picked, job.instruction)
        # positives only: the mined negatives replay files carry cost more than they taught (Phase 0)
        picked = [replace(p, negatives=()) for p in picked]
        state_texts, pos_texts, neg_texts = pair_texts(picked, tf, qf)
        tool_texts, pos, neg = index_tools(pos_texts, neg_texts)
        off = len(tools)  # replay tools follow the catalogue in the training matrix only
        train_b = Batches(
            np.vstack([train_b.states, enc.encode(state_texts, kind="query")]),
            np.vstack([tool_vecs, enc.encode(tool_texts)]),
            train_b.pos + [[off + j for j in row] for row in pos],
            train_b.neg + [[off + j for j in row] for row in neg],
        )
        report["replay"] = {"pairs": len(picked), "tools": len(tool_texts), "from": job.replay.name}
        if dropped:
            report["replay"]["against_dev"] = dropped
        report["tokens_spent"] = enc.tokens_spent
        log(f"replay: {len(picked)} general pairs over {len(tool_texts)} tools from {job.replay.name}")
    bench = None
    if job.dev is not None:
        from toolrank.finetune import EvalSet

        bench_tools, bench_queries = (
            load_tools(job.dev / "tools.jsonl"),
            load_queries(job.dev / "queries.jsonl"),
        )
        bench = EvalSet(
            "dev",
            bench_queries,
            [t.id for t in bench_tools],
            enc.encode([tf(t) for t in bench_tools]),
            enc.encode([qf(q) for q in bench_queries], kind="query"),
        )

    def on_epoch(epoch: int, state_head: Any, action_head: Any) -> dict[str, float]:
        device = str(next(action_head.parameters()).device)
        zs, za = project(state_head, dev_b.states, device), project(action_head, tool_vecs, device)
        m = {
            f"log.Recall@{K}": recall_at(zs, za, dev_b.pos, K),
            "log.Recall@1": recall_at(zs, za, dev_b.pos, 1),
        }
        if bench is not None:
            m.update(
                curve_metrics(
                    "dev",
                    bench.score(
                        lambda x: project(state_head, x, device), lambda x: project(action_head, x, device)
                    ),
                )
            )
        return m

    select = f"log.Recall@{K}"
    init = resolve_init(job.init, job.emb_model, data=job.data, tenant=job.tenant)
    if init is None:
        # fresh heads on a backbone the packaged ones do not fit: a skip head starts as the
        # identity, so epoch 0 is the backbone's own score, not noise
        job.train.head_cfg.setdefault("skip", True)
    ck, history = train_heads(train_b, None, job.train, init=init, on_epoch=on_epoch, log=log, select=select)
    best = int(ck["cfg"]["best_epoch"])
    start, chosen = history[0], history[best]
    improved = best > 0 and chosen[select] > start[select]
    dropped = bench is not None and chosen["dev.NDCG@10"] < start["dev.NDCG@10"] - job.max_drop / 100
    report.update(history=history, best_epoch=best, select=select, start=start, chosen=chosen)
    if not improved:
        report["decision"] = "no improvement"
        log(f"no epoch beat the starting heads on {select} ({start[select]:.3f}); nothing published")
        return report
    if dropped:
        report["decision"] = "benchmark dropped"
        log(
            f"epoch {best} improves {select} {start[select]:.3f} -> {chosen[select]:.3f} but the benchmark dev "
            f"falls {100 * (start['dev.NDCG@10'] - chosen['dev.NDCG@10']):.2f} points (max {job.max_drop}); nothing published"
        )
        return report
    import torch

    from toolrank.adapters.heads_np import export_npz, sha256_file

    serving = {
        "backbone": job.backbone,
        "tool_format": job.tool_format,
        "query_format": job.query_format,
        "truncate": job.truncate,
    }
    ck["cfg"].update({k: v for k, v in serving.items() if v})
    ck["cfg"]["selected_on"] = {
        "set": f"{job.data.name}/usage (newest {len(dev_pairs)} requests)",
        "queries": len(dev_pairs),
        "metric": select,
        "value": round(float(chosen[select]), 6),
        "epoch": best,
    }
    ck["cfg"]["learned_from"] = {
        "log": job.data.name,
        "requests": len(train_pairs),
        "until": train_pairs[-1].ts,
        "tenant": job.tenant,
    }
    pt = job.out.with_suffix(".pt")
    pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, pt)
    digest = export_npz(pt, job.out, dtype="float16")
    report["decision"] = "published"
    report["outputs"] = {
        "pt": {"path": str(pt), "sha256": sha256_file(pt)},
        "npz": {"path": str(job.out), "sha256": digest},
    }
    log(
        f"epoch {best}: {select} {start[select]:.3f} -> {chosen[select]:.3f}; wrote {job.out} "
        + (
            "(a running server gives it a share of the requests; toolrank ab decides)"
            if job.out.name == CANDIDATE
            else f"(serve it with TOOLRANK_HEADS={job.out})"
        )
    )
    return report


# -- A/B: the log says how each arm did ------------------------------------------------------------
CURRENT, CANDIDATE = "current.npz", "candidate.npz"


def heads_home(data: Path, tenant: str | None = None) -> Path:
    """Where a server looks for heads that change while it runs: ``DATA/heads``, or
    ``DATA/heads/tenants/<name>`` for one API key's requests."""
    return data / "heads" / "tenants" / tenant if tenant else data / "heads"


def judge(
    events: Iterable[dict[str, Any]],
    *,
    since: str | None = None,
    tenant: str | None = None,
    counts: Counter[str] | None = None,
) -> dict[str, dict[str, float]]:
    """How the control and the candidate did since ``since`` -> {"control" | "candidate": {searches,
    called, top1, mrr}}. A search belongs to the candidate when its ``arm`` ends in ``candidate``,
    with ``tenant`` only that key's searches count; a search counts as called when a linked call
    ended ``ok`` or ``tool_error``, at the best rank among those calls. Searches a Jev second stage
    answered decide nothing (the provider's terms); when ``counts`` is given they are tallied in it."""
    arms: dict[str, str] = {}
    for e in events:
        if e.get("event") != "search" or (since and str(e.get("ts", "")) < since):
            continue
        if not isinstance(e.get("id"), str):
            continue
        arm = str(e.get("arm") or "base")
        if tenant is not None and e.get("tenant") != tenant:
            continue
        if tenant is None and arm.startswith("tenant:"):
            continue  # a tenant's own heads are another experiment
        if not may_learn_from(e.get("scorer")):  # counted for the key being judged only
            if counts is not None:
                counts["searches_with_jev"] += 1
            continue
        arms[e["id"]] = "candidate" if arm.endswith("candidate") else "control"
    best: dict[str, int] = {}
    for e in events:
        sid = e.get("search_id")
        if (
            e.get("event") == "call"
            and sid in arms
            and e.get("outcome") in (POSITIVE, WEAK)
            and e.get("rank")
        ):
            best[sid] = min(best.get(sid, 10**9), int(e["rank"]))
    out: dict[str, dict[str, float]] = {}
    for name in ("control", "candidate"):
        ids = [sid for sid, arm in arms.items() if arm == name]
        ranks = [best[sid] for sid in ids if sid in best]
        n = len(ids)
        out[name] = {
            "searches": n,
            "called": len(ranks),
            "top1": sum(r == 1 for r in ranks) / n if n else 0.0,
            "mrr": sum(1.0 / r for r in ranks) / n if n else 0.0,
        }
    return out


def decide(stats: dict[str, dict[str, float]], *, min_searches: int = 100, margin: float = 0.01) -> str:
    """``promote``, ``rollback`` or ``wait``: nothing is decided before both arms have
    ``min_searches``, and the candidate's ``mrr`` has to differ from the control's by ``margin``."""
    control, candidate = stats["control"], stats["candidate"]
    if min(control["searches"], candidate["searches"]) < min_searches:
        return "wait"
    diff = candidate["mrr"] - control["mrr"]
    return "promote" if diff > margin else "rollback" if diff < -margin else "wait"


def apply(home: Path, decision: str, *, stamp: str | None = None) -> dict[str, str]:
    """Carry ``promote`` or ``rollback`` out with renames (a running server follows the files):
    the candidate becomes ``current.npz`` and the heads it replaces ``previous-<stamp>.npz``, or the
    candidate is set aside as ``rejected-<stamp>.npz``; the ``.pt`` next to each moves with it.
    -> what moved where."""
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    moved: dict[str, str] = {}

    def move(src: Path, dst: Path) -> None:
        for suffix in (".npz", ".pt"):
            a, b = src.with_suffix(suffix), dst.with_suffix(suffix)
            if a.exists():
                os.replace(a, b)
                moved[a.name] = b.name

    if decision == "promote":
        move(home / CURRENT, home / f"previous-{stamp}.npz")
        move(home / CANDIDATE, home / CURRENT)
    elif decision == "rollback":
        move(home / CANDIDATE, home / f"rejected-{stamp}.npz")
    return moved
