"""Head fine-tuning on cached backbone vectors: ``toolrank finetune`` (``run``).

The backbone stays frozen: every text is embedded once through a pooling server into the
embedding cache, and training only ever reads vectors. The heads have the CLM architecture
(``adapters.clm.make_head``) and are saved in its checkpoint format, so ``toolrank eval --scorer
clm --clm-ckpt`` loads them for any backbone; skip heads (x + MLP(x)) start as the identity, i.e.
at the embedding model's own quality.

Loss: InfoNCE over the batch's candidates (one sampled positive per request plus its hard
negatives), with the request's other positives and duplicate tools masked out. The epoch is picked
on a dev set (``EvalSet``, scored exactly like ``toolrank eval``) that is never reported: in Phase 0
recall on held-out training pairs rose while the benchmark fell. Training can start from fresh
heads, a torch ``.pt`` or the packaged ``.npz``.

``torch`` is imported inside functions (``toolrank[clm]``).
"""

from __future__ import annotations

import dataclasses
import math
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from toolrank.domain import Query, Tool, TrainPair
from toolrank.eval.metrics import evaluate_query
from toolrank.eval.runner import Summary, summarize
from toolrank.formats import NamedFormatter, parse_doc


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def drop_test_requests(
    pairs: Sequence[TrainPair], test_queries: Sequence[Query]
) -> tuple[list[TrainPair], int]:
    """Pairs whose request text equals a benchmark request are dropped -> (kept, n_dropped): a pair
    the selection set already asks about would make the guard meaningless."""
    test = {_norm(q.text) for q in test_queries if q.text.strip()}
    kept = [p for p in pairs if _norm(p.text) not in test]
    return kept, len(pairs) - len(kept)


def split_pairs(
    pairs: Sequence[TrainPair], test_queries: Sequence[Query], n_train: int, n_val: int, seed: int = 0
) -> tuple[list[TrainPair], list[TrainPair], int]:
    """Drop pairs whose request text equals a benchmark request, then a seeded split.

    -> (train, val, n_dropped). ``n_train`` 0 means every remaining pair. The same arguments give
    the same subsets, which is what lets the embed step and the train step agree.
    """
    kept, dropped = drop_test_requests(pairs, test_queries)
    order = list(range(len(kept)))
    random.Random(seed).shuffle(order)
    val = [kept[i] for i in order[:n_val]]
    rest = order[n_val:]
    train = [kept[i] for i in (rest[:n_train] if n_train else rest)]
    return train, val, dropped


def pair_texts(
    pairs: Sequence[TrainPair], tool_fmt: NamedFormatter, query_fmt: NamedFormatter
) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """-> (state text per pair, positive tool texts per pair, negative tool texts per pair), formatted
    exactly like the benchmark formats its queries and corpus tools."""
    memo: dict[str, str] = {}

    def tool_text(doc: str) -> str:
        if doc not in memo:
            memo[doc] = tool_fmt(Tool(id="", doc=parse_doc(doc), documentation=doc))
        return memo[doc]

    states = [query_fmt(Query(id=p.id, text=p.text, qrels={}, instruction=p.instruction)) for p in pairs]
    return (
        states,
        [[tool_text(d) for d in p.positives] for p in pairs],
        [[tool_text(d) for d in p.negatives] for p in pairs],
    )


@dataclass
class Batches:
    """Pairs as indices into one matrix of unique tool vectors."""

    states: np.ndarray  # [n_pairs, hidden]
    tools: np.ndarray  # [n_tools, hidden]
    pos: list[list[int]]
    neg: list[list[int]]


def index_tools(
    pos_texts: list[list[str]], neg_texts: list[list[str]]
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    """-> (unique tool texts, positives as indices, negatives as indices)."""
    ids: dict[str, int] = {}
    pos = [[ids.setdefault(t, len(ids)) for t in ts] for ts in pos_texts]
    neg = [[ids.setdefault(t, len(ids)) for t in ts] for ts in neg_texts]
    return list(ids), pos, neg


# -- data preparation (no torch) -------------------------------------------------------------------
def leaks(pairs: Sequence[TrainPair], sets: dict[str, Sequence[Query]]) -> dict[str, int]:
    """How many training requests equal a query of each set (normalised text): what
    ``split_pairs`` drops, per source."""
    out = {}
    for name, queries in sets.items():
        test = {_norm(q.text) for q in queries if q.text.strip()}
        out[name] = sum(1 for p in pairs if _norm(p.text) in test)
    return out


def with_instruction(pairs: Sequence[TrainPair], instruction: str) -> tuple[list[TrainPair], int]:
    """Pairs without an instruction get ``instruction`` — the one serve sends with every request, so
    training sees requests as they will arrive; -> (pairs, how many were filled)."""
    out, filled = [], 0
    for p in pairs:
        if not p.instruction and instruction:
            p = dataclasses.replace(p, instruction=instruction)
            filled += 1
        out.append(p)
    return out, filled


@dataclass
class EvalSet:
    """A benchmark-format set (``tools.jsonl`` + ``queries.jsonl``) as backbone vectors, scored with
    the heads while they train. The ranking is exact top-k over the whole corpus, and the metrics
    go through ``eval.runner.summarize``, so the number an epoch is selected on is the one
    ``toolrank eval`` reports for those heads (NDCG@10 only needs the top 10)."""

    name: str
    queries: list[Query]
    tool_ids: list[str]
    tool_vecs: np.ndarray
    query_vecs: np.ndarray

    def score(
        self,
        project_states: Callable[[np.ndarray], np.ndarray],
        project_actions: Callable[[np.ndarray], np.ndarray],
        ks: Sequence[int] = (10,),
    ) -> Summary:
        from toolrank.adapters.index_numpy import topk_dot

        zq, za = project_states(self.query_vecs), project_actions(self.tool_vecs)
        per_query: list[tuple[Query, dict[str, float]]] = []
        for start in range(0, len(self.queries), 1024):
            idx, _ = topk_dot(zq[start : start + 1024], za, max(ks))
            for q, row in zip(self.queries[start : start + 1024], idx, strict=True):
                per_query.append((q, evaluate_query([self.tool_ids[j] for j in row], q.qrels, ks)))
        return summarize(per_query)


def curve_metrics(name: str, summary: Summary, metric: str = "NDCG@10") -> dict[str, float]:
    """``{name}.NDCG@10`` (micro) and, for sets with categories, ``{name}.NDCG@10-cat``."""
    out = {f"{name}.{metric}": summary.overall[metric]}
    if summary.category_macro:
        out[f"{name}.{metric}-cat"] = summary.category_macro[metric]
    return out


@dataclass
class TrainConfig:
    epochs: int = 3
    batch: int = 512
    neg_per_pair: int = 15
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup: float = 0.05
    seed: int = 0
    device: str | None = None
    head_cfg: dict[str, Any] = field(
        default_factory=dict
    )  # width, depth, projection_dim, skip for a fresh init
    freeze_action: bool = False  # adapt requests only; tools keep the starting geometry
    # Drop a mined negative the starting model scores above this share of the request's best positive:
    # likely a functionally equivalent tool, not a negative (NV-Retriever's TopK-PercPos, e.g. 0.95).
    neg_filter: float | None = None


PROVENANCE = ("trained_from", "init_sha256", "best_epoch", "selected_on")


def load_checkpoint(init: str) -> tuple[dict[str, Any], Path]:
    """``init`` — a torch ``.pt``, a packaged ``.npz`` or ``default`` (the packaged heads) — as the
    checkpoint dict ``train_heads`` continues from, and the file it came from. An ``.npz`` is fp16
    on disk: training continues from exactly the weights users run."""
    import torch

    from toolrank.adapters.heads_np import default_heads, read_checkpoint

    path = default_heads() if init == "default" else Path(init)
    if path.suffix != ".npz":
        return torch.load(path, map_location="cpu", weights_only=True), path
    raw = read_checkpoint(path)
    return {
        "state_head": {k: torch.from_numpy(v.copy()) for k, v in raw["state_head"].items()},
        "action_head": {k: torch.from_numpy(v.copy()) for k, v in raw["action_head"].items()},
        "logit_scale": torch.tensor(raw["logit_scale"]),
        "cfg": raw["cfg"],
    }, path


def _heads_from(init: str | None, cfg: TrainConfig, hidden: int):
    import torch

    from toolrank.adapters.clm import PROJ_DIM, make_head
    from toolrank.adapters.heads_np import sha256_file

    origin: dict[str, str] = {"trained_from": "scratch"}
    if init:
        ck, path = load_checkpoint(init)
        head_cfg = {k: v for k, v in ck["cfg"].items() if k not in PROVENANCE}
        if int(head_cfg.get("hidden_size", hidden)) != hidden:
            raise ValueError(
                f"{path.name} takes {head_cfg['hidden_size']}-d backbone vectors, the training data has {hidden}"
            )
        proj = ck.get("projection_dim", head_cfg.get("projection_dim", PROJ_DIM))
        scale = float(torch.as_tensor(ck["logit_scale"]).float())
        origin = {"trained_from": path.name, "init_sha256": sha256_file(path)}  # no local paths
    else:
        head_cfg = {
            "width": 1536,
            "depth": 3,
            "projection_dim": PROJ_DIM,
            "activation": "gelu",
            "layernorm": True,
        }
        head_cfg.update(cfg.head_cfg)
        head_cfg["hidden_size"] = hidden
        if head_cfg.get("skip"):
            head_cfg["projection_dim"] = hidden  # x + MLP(x) keeps the width
        proj, scale = head_cfg["projection_dim"], math.log(20.0)
    kw = dict(
        width=head_cfg["width"],
        depth=head_cfg["depth"],
        proj=proj,
        activation=head_cfg.get("activation", "gelu"),
        layernorm=head_cfg.get("layernorm", False),
        residual=head_cfg.get("residual", False),
        hidden=head_cfg.get("hidden_size", hidden),
        skip=head_cfg.get("skip", False),
    )
    sh, ah = make_head(**kw), make_head(**kw)
    if init:
        sh.load_state_dict(ck["state_head"])
        ah.load_state_dict(ck["action_head"])
    elif kw["skip"]:  # start as the identity: the embedding model's own zero-shot geometry
        for h in (sh, ah):
            torch.nn.init.zeros_(h.out.weight)
            torch.nn.init.zeros_(h.out.bias)
    head_cfg.update(projection_dim=proj, hidden_size=kw["hidden"])
    return sh, ah, torch.nn.Parameter(torch.tensor(scale)), head_cfg, origin


def project(head, x: np.ndarray, device: str, batch: int = 8192) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            t = torch.from_numpy(np.ascontiguousarray(x[s : s + batch], dtype=np.float32)).to(device)
            out.append(torch.nn.functional.normalize(head(t), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def recall_at(states: np.ndarray, tools: np.ndarray, pos: list[list[int]], k: int = 10) -> float:
    """Share of pairs with a positive among the top-k of all ``tools`` (projected, normalised)."""
    from toolrank.adapters.dense import topk_dot

    hits = 0
    for s in range(0, len(states), 1024):
        idx, _ = topk_dot(states[s : s + 1024], tools, k)
        hits += sum(bool(set(row) & set(pos[s + i])) for i, row in enumerate(idx.tolist()))
    return hits / max(1, len(states))


def train_heads(
    train: Batches,
    val: Batches | None,
    cfg: TrainConfig,
    init: str | None = None,
    on_epoch: Callable[[int, Any, Any], dict[str, float]] | None = None,
    log: Callable[[str], None] = print,
    *,
    select: str = "val_recall@10",
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Train state/action heads; -> (the checkpoint dict of the best epoch, per-epoch history).

    ``on_epoch(epoch, state_head, action_head)`` adds metrics to each epoch's history (a dev set,
    benchmark curves). ``select`` names the one the best epoch is picked on, higher being better:
    ``val_recall@10`` (held-out training pairs, needs ``val``) or a key ``on_epoch`` adds, such as a
    dev set's ``dev.NDCG@10``. The starting heads (epoch 0) compete, and a later epoch must beat
    the best so far strictly: ties keep the earlier epoch, and heads that never improve come back
    as they started. Phase 0 found recall on held-out training pairs rising while the benchmark fell,
    so ``toolrank finetune`` selects on a dev set.
    """
    import torch

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sh, ah, logit_scale, head_cfg, origin = _heads_from(init, cfg, train.states.shape[1])
    sh.to(device), ah.to(device)
    logit_scale = torch.nn.Parameter(logit_scale.detach().to(device))
    if cfg.freeze_action:
        ah.requires_grad_(False)
    params = [*sh.parameters(), *(() if cfg.freeze_action else ah.parameters()), logit_scale]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train.pos) / cfg.batch)
    total, warm = cfg.epochs * steps_per_epoch, max(1, int(cfg.warmup * cfg.epochs * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    )
    S = torch.from_numpy(np.ascontiguousarray(train.states, dtype=np.float32)).to(device)
    T = torch.from_numpy(np.ascontiguousarray(train.tools, dtype=np.float32)).to(device)
    pos_sets = [set(p) for p in train.pos]
    negatives = train.neg
    if cfg.neg_filter:
        sh.eval(), ah.eval()
        zs, za = project(sh, train.states, device), project(ah, train.tools, device)
        negatives = []
        for i, (pos, neg) in enumerate(zip(train.pos, train.neg, strict=True)):
            ref = max(float(zs[i] @ za[j]) for j in pos)
            negatives.append([j for j in neg if ref <= 0 or float(zs[i] @ za[j]) < cfg.neg_filter * ref])
        n_all, n_kept = sum(map(len, train.neg)), sum(map(len, negatives))
        log(f"neg_filter {cfg.neg_filter}: kept {n_kept:,} of {n_all:,} mined negatives")
        sh.train(), ah.train()

    def evaluate(epoch: int) -> dict[str, float]:
        sh.eval(), ah.eval()
        m: dict[str, float] = {"epoch": epoch}
        if val is not None and len(val.pos):
            vt = project(ah, val.tools, device)
            m["val_recall@10"] = recall_at(project(sh, val.states, device), vt, val.pos, 10)
        if on_epoch:
            m.update(on_epoch(epoch, sh, ah))
        sh.train(), ah.train()
        return m

    def snapshot(epoch: int) -> dict[str, Any]:
        return {
            "state_head": {k: v.detach().cpu().clone() for k, v in sh.state_dict().items()},
            "action_head": {k: v.detach().cpu().clone() for k, v in ah.state_dict().items()},
            "logit_scale": logit_scale.detach().cpu().clone(),
            "cfg": {**head_cfg, **origin, "best_epoch": epoch},
        }

    history = [evaluate(0)]
    log(f"epoch 0 (init): {history[-1]}")
    if select not in history[0]:
        raise ValueError(f"select={select!r} is not an epoch metric: {sorted(history[0])}")
    best, best_ck = history[0][select], snapshot(0)  # training must beat the starting point
    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(len(train.pos))
        t0, losses = time.time(), []
        for s in range(0, len(order), cfg.batch):
            rows = order[s : s + cfg.batch]
            targets_tool = [int(rng.choice(train.pos[i])) for i in rows]
            cands: dict[int, int] = {}
            for t in targets_tool:
                cands.setdefault(t, len(cands))
            for i in rows:
                negs = negatives[i]
                picked = (
                    negs
                    if len(negs) <= cfg.neg_per_pair
                    else rng.choice(negs, cfg.neg_per_pair, replace=False)
                )
                for t in picked:
                    cands.setdefault(int(t), len(cands))
            cand_ids = list(cands)
            mask = torch.zeros(len(rows), len(cand_ids), dtype=torch.bool)
            for r, i in enumerate(rows):  # other positives of the same request are not negatives
                for t in pos_sets[i]:
                    c = cands.get(t)
                    if c is not None and t != targets_tool[r]:
                        mask[r, c] = True
            zs = torch.nn.functional.normalize(sh(S[torch.as_tensor(rows, device=device)]), dim=-1)
            za = torch.nn.functional.normalize(ah(T[torch.as_tensor(cand_ids, device=device)]), dim=-1)
            logits = logit_scale.exp().clamp(max=100.0) * zs @ za.T
            logits = logits.masked_fill(mask.to(device), float("-inf"))
            target = torch.as_tensor([cands[t] for t in targets_tool], device=device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        m = evaluate(epoch)
        m.update(train_loss=float(np.mean(losses)), seconds=round(time.time() - t0, 1))
        history.append(m)
        log(f"epoch {epoch}: {m}")
        if m[select] > best:
            best, best_ck = m[select], snapshot(epoch)
    return best_ck, history


# -- toolrank finetune ------------------------------------------------------------------------------
@dataclass
class Job:
    """What ``toolrank finetune`` was asked to do (``cli.cmd_finetune`` fills it from the flags)."""

    pairs: Path
    dev: Path
    out: Path
    evals: list[Path] = field(default_factory=list)
    init: str | None = None
    npz: Path | None = None
    emb_url: str = "http://127.0.0.1:8091/v1"
    emb_model: str = "qwen3-emb"
    emb_batch: int = 128
    truncate: int | None = 8192
    cache_dir: str | None = ".cache/toolrank"
    tool_format: str = "documentation"
    query_format: str = "instruct_query"
    instruction: str = ""  # given to pairs without one (the serving instruction)
    backbone: str = ""
    n_train: int = 0
    n_val: int = 0
    seed: int = 0
    select: str = "ndcg10"  # or "ndcg10-cat"
    curve: bool = False  # also score the eval sets each epoch (reported, never selected on)
    embed_only: bool = False
    train: TrainConfig = field(default_factory=TrainConfig)
    chunk: int = 4096


def _load_set(path: Path) -> tuple[list[Tool], list[Query]]:
    from toolrank.datasets.jsonl import load_queries, load_tools

    return load_tools(path / "tools.jsonl"), load_queries(path / "queries.jsonl")


def check_sets(
    dev: Path, dev_queries: Sequence[Query], evals: dict[Path, Sequence[Query]], select: str
) -> list[str]:
    """Refuse a dev set that is also reported (the same directory, or sharing query texts with an
    eval set, as ``mcp_zero`` and ``mcp_zero_server`` do) and ``ndcg10-cat`` without categories;
    -> warnings."""
    texts = {_norm(q.text) for q in dev_queries if q.text.strip()}
    for path, queries in evals.items():
        if path.resolve() == dev.resolve():
            raise ValueError(
                f"{dev} is both the dev set and an eval set: a set selected on cannot be reported"
            )
        shared = sum(1 for q in queries if _norm(q.text) in texts)
        if shared:
            raise ValueError(
                f"{dev} and {path} share {shared} queries: a set selected on cannot be reported; pick another dev set"
            )
    if select == "ndcg10-cat" and not any(q.category for q in dev_queries):
        raise ValueError(f"--select ndcg10-cat needs categories, and {dev} has none")
    if len(dev_queries) < 300:
        return [f"{dev} has {len(dev_queries)} queries: epochs a few tenths of a point apart may be noise"]
    return []


def run(job: Job, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Prepare, embed what the cache lacks, and (unless ``embed_only``) train and save the heads;
    -> the report ``cli.cmd_finetune`` completes with the official evaluations."""
    from toolrank.adapters.embeddings_api import OpenAIEmbeddings
    from toolrank.adapters.heads_np import export_npz, sha256_file
    from toolrank.datasets.jsonl import load_pairs
    from toolrank.formats import query_format, tool_format

    tf, qf = tool_format(job.tool_format), query_format(job.query_format)
    dev_tools, dev_queries = _load_set(job.dev)
    sets = {p: _load_set(p) for p in job.evals}
    warnings = check_sets(job.dev, dev_queries, {p: q for p, (_, q) in sets.items()}, job.select)
    for w in warnings:
        log(f"warning: {w}")

    loaded = load_pairs(job.pairs)
    pairs = [p for p in loaded if p.positives]
    by_source = {job.dev.name: dev_queries, **{p.name: q for p, (_, q) in sets.items()}}
    dropped = {"no_positives": len(loaded) - len(pairs), **leaks(pairs, by_source)}
    everything = [q for queries in by_source.values() for q in queries]
    train, val, _ = split_pairs(pairs, everything, job.n_train, job.n_val, job.seed)
    filled = 0
    if job.instruction:
        train, n_train = with_instruction(train, job.instruction)
        val, n_val = with_instruction(val, job.instruction)
        filled = n_train + n_val
    log(
        f"{len(loaded):,} pairs: train {len(train):,}, val {len(val):,}; dropped {dropped}; "
        f"instruction filled in {filled:,}"
    )

    enc = OpenAIEmbeddings(
        job.emb_model,
        job.emb_url,
        batch=job.emb_batch,
        truncate_prompt_tokens=job.truncate,
        cache_dir=job.cache_dir,
    )
    parts = [pair_texts(train, tf, qf), pair_texts(val, tf, qf)]
    texts: list[str] = []
    for states, pos, neg in parts:
        texts += states + [t for ts in pos for t in ts] + [t for ts in neg for t in ts]
    for tools, queries in [(dev_tools, dev_queries), *sets.values()]:
        texts += [tf(t) for t in tools] + [qf(q) for q in queries]
    texts = list(dict.fromkeys(texts))
    todo = [texts[i] for i in enc.cache.missing(texts)] if enc.cache is not None else texts
    log(f"{len(texts):,} texts, {len(texts) - len(todo):,} cached, {len(todo):,} to embed")
    t0 = time.time()
    for s in range(0, len(todo), job.chunk):
        enc.encode(todo[s : s + job.chunk])
        done, dt = min(len(todo), s + job.chunk), time.time() - t0
        log(f"embedded {done:,}/{len(todo):,} texts, {enc.tokens_spent:,} tokens, {dt / 60:.1f} min")
    report: dict[str, Any] = {
        "pairs": str(job.pairs),
        "dev": job.dev.name,
        "evals": [p.name for p in job.evals],
        "dropped": dropped,
        "instructions_filled": filled,
        "n_train": len(train),
        "n_val": len(val),
        "texts": {"total": len(texts), "embedded": len(todo), "tokens": enc.tokens_spent},
        "warnings": warnings,
    }
    if job.embed_only:
        return report

    def batches(states: list[str], pos: list[list[str]], neg: list[list[str]]) -> Batches:
        tools, pos_i, neg_i = index_tools(pos, neg)
        return Batches(enc.encode(states, kind="query"), enc.encode(tools), pos_i, neg_i)

    def eval_set(name: str, tools: list[Tool], queries: list[Query]) -> EvalSet:
        tool_vecs = enc.encode([tf(t) for t in tools])
        return EvalSet(
            name,
            queries,
            [t.id for t in tools],
            tool_vecs,
            enc.encode([qf(q) for q in queries], kind="query"),
        )

    train_b = batches(*parts[0])
    val_b = batches(*parts[1]) if val else None
    dev_set = eval_set("dev", dev_tools, dev_queries)
    eval_sets = [eval_set(p.name, *sets[p]) for p in job.evals] if job.curve else []
    select = "dev.NDCG@10-cat" if job.select == "ndcg10-cat" else "dev.NDCG@10"

    def on_epoch(epoch: int, state_head: Any, action_head: Any) -> dict[str, float]:
        device = str(next(action_head.parameters()).device)

        def states(x: np.ndarray) -> np.ndarray:
            return project(state_head, x, device)

        def actions(x: np.ndarray) -> np.ndarray:
            return project(action_head, x, device)

        m = curve_metrics("dev", dev_set.score(states, actions))
        for es in eval_sets:
            m.update(curve_metrics(es.name, es.score(states, actions)))
        return m

    ck, history = train_heads(
        train_b, val_b, job.train, init=job.init, on_epoch=on_epoch, log=log, select=select
    )
    import torch

    best = int(ck["cfg"]["best_epoch"])
    serving = {
        "backbone": job.backbone,
        "tool_format": job.tool_format,
        "query_format": job.query_format,
        "truncate": job.truncate,
        "instruction": job.instruction,
    }
    ck["cfg"].update({k: v for k, v in serving.items() if v})
    ck["cfg"]["selected_on"] = {
        "set": job.dev.name,
        "queries": len(dev_queries),
        "metric": "NDCG@10 cat-macro" if job.select == "ndcg10-cat" else "NDCG@10",
        "value": round(float(history[best][select]), 6),
        "epoch": best,
    }
    job.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, job.out)
    outputs = {"pt": {"path": str(job.out), "sha256": sha256_file(job.out)}}
    if job.npz is not None:
        outputs["npz"] = {"path": str(job.npz), "sha256": export_npz(job.out, job.npz, dtype="float16")}
    log(f"best epoch {best} by {select}; saved {job.out}" + (f" and {job.npz}" if job.npz else ""))
    report.update(select=select, best_epoch=best, history=history, outputs=outputs, cfg=ck["cfg"])
    return report
