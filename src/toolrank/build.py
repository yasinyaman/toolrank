"""Composition root: parsed flags -> a ready scorer (encoder, heads, vector index).

Shared by ``toolrank eval``, ``toolrank search`` and ``toolrank serve``. Besides
``cli.py`` this is the only module outside ``adapters/`` that imports adapters; everything is
imported inside the functions, so the base install still loads nothing heavy.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

INDEX_KINDS = ("numpy", "faiss", "pgvector")
# Embedding backbones by served name. The name is the embedding cache's key, so new weights get a
# new name (a version in it). "heads": whether the packaged heads were trained on its vectors; an
# unknown name keeps the old behaviour (heads when a checkpoint is at hand). The default is
# Qwen3-Embedding-8B LoRA-trained on ToolRet's training pairs (docs/backbone/MODEL_CARD.md): one
# stage, no heads (the v0.1 heads cost it 1-2 points; heads trained on it stay at identity).
BACKBONE_REPO, BACKBONE_REVISION = "yasinyaman/toolrank-emb-8b", "v0.2"
BACKBONE_PUBLISHED = True  # True once the weights are on the Hub at that revision (release_check)
BACKBONES: dict[str, dict[str, Any]] = {
    "toolrank-emb-v0.2": {"repo": BACKBONE_REPO, "revision": BACKBONE_REVISION, "heads": False},
    "toolrank-emb-v0.2-fp8": {"repo": BACKBONE_REPO, "revision": BACKBONE_REVISION, "heads": False},
    "qwen3-emb": {"repo": "Qwen/Qwen3-Embedding-8B", "heads": True},
    "qwen3-emb-fp8": {"repo": "Qwen/Qwen3-Embedding-8B", "heads": True},
    "qwen3-emb-lora": {"repo": "a local scripts/lora_train.py run", "heads": False},
}
# toolrank search's and serve's product defaults: the served backbone, its texts and the
# instruction that did best on the MCP sets
DEFAULT_EMB_URL = "http://127.0.0.1:8091/v1"
DEFAULT_EMB_MODEL = "toolrank-emb-v0.2"


def packaged_heads_fit(emb_model: str | None) -> bool:
    """Whether the packaged heads belong on this served backbone (unknown names: yes, as before)."""
    return bool(BACKBONES.get(emb_model or "", {}).get("heads", True))


def backbone_repo(emb_model: str | None) -> str:
    """What a served name is, for the cfg of heads trained on it (the name itself when unknown)."""
    return str(BACKBONES.get(emb_model or "", {}).get("repo") or emb_model or "")


DEFAULT_SERVING = {
    "tool_format": "documentation",
    "query_format": "instruct_query",
    "truncate": 8192,
    "instruction": "Given an agent's request for a tool, retrieve the MCP tool that fulfills it.",
}


@lru_cache(maxsize=8)
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(a: Any, tool_format: str, heads_path: str | Path | None = None) -> str:
    """What a stored tool vector depends on besides the tool text: endpoint, model, truncation,
    tool format and the heads file (by content)."""
    heads = file_sha256(str(heads_path))[:16] if heads_path else "-"
    return f"{a.emb_url}|{a.emb_model}|trunc={a.truncate}|{tool_format}|heads={heads}"


def require(module: str, extra: str, what: str) -> None:
    """A clear error, with the install line, instead of an ImportError deep inside an adapter."""
    import importlib.util

    if importlib.util.find_spec(module) is None:
        raise ValueError(f"{what} needs the [{extra}] extra: pip install 'toolrank[{extra}]'")


def build_index(a: Any) -> Any:
    kind = getattr(a, "index", None) or "numpy"
    index_dir = getattr(a, "index_dir", None)
    if kind == "numpy":
        from toolrank.adapters.index_numpy import NumpyIndex

        return NumpyIndex(Path(index_dir) / "index.npz" if index_dir else None)
    if kind == "faiss":
        require("faiss", "faiss", "--index faiss")
        from toolrank.adapters.index_faiss import FaissIndex

        return FaissIndex(Path(index_dir) / "faiss-hnsw.npz" if index_dir else None)
    if kind == "pgvector":
        require("psycopg", "pgvector", "--index pgvector")
        require("pgvector", "pgvector", "--index pgvector")
        from toolrank.adapters.index_pgvector import PgVectorIndex

        dsn = getattr(a, "pg_dsn", None) or os.environ.get("TOOLRANK_PG_DSN")
        if not dsn:
            raise ValueError("--index pgvector needs --pg-dsn or TOOLRANK_PG_DSN")
        return PgVectorIndex(dsn, getattr(a, "pg_table", None) or "toolrank_tools")
    raise ValueError(f"unknown index {kind!r}; choose from {INDEX_KINDS}")


def build_scorer(a: Any) -> Any:
    """The scorer ``a`` asks for (``a.scorer``: bm25 | dense | clm, plus formats and encoder flags),
    fused with BM25 when ``a.hybrid``."""
    make, _ = scorer_factory(a)
    return make()


def jev_client(a: Any) -> Any:
    """A ``JevClient`` from the ``--jev-*`` flags; the key comes from ``TYPESAFE_API_KEY`` only."""
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise ValueError("Jev needs a key: set TYPESAFE_API_KEY (https://console.typesafe.ai/keys)")
    from toolrank.adapters.jev import JevClient

    return JevClient(a.jev_model, a.jev_url, cache_dir=a.cache_dir or None, workers=a.jev_workers)


def scorer_factory(
    a: Any, *, query_timeout: float | None = None, query_attempts: int | None = None
) -> tuple[Callable[[], Any], dict[str, Any]]:
    """Resolve the flags, load the heads and open the encoder once; -> (a function that makes fresh
    scorers sharing them, info: ``serving`` cfg, ``encoder``, ``heads_path``, and ``jev``, the
    client, when Jev is in the stack). A server makes a new scorer (with a new index object) for
    every catalogue reload and swaps it in whole. ``--rerank jev`` wraps whatever the other flags
    build (BM25, dense, clm, hybrid) in a ``JevReranker``."""
    make, info = _base_factory(a, query_timeout=query_timeout, query_attempts=query_attempts)
    rerank = getattr(a, "rerank", None)
    if rerank is None:
        return make, info
    if rerank in ("dense", "clm", "cross"):  # a local second scorer with its own --rerank-* flags
        from toolrank.adapters.rerank import ScorerReranker

        second_make, second_info = _base_factory(rerank_args(a))
        info["rerank"] = second_info
        base = make

        def reranked() -> Any:
            return ScorerReranker(
                base(),
                second_make(),
                depth=a.rerank_depth,
                max_chars=getattr(a, "rerank_max_chars", None),
                workers=getattr(a, "rerank_workers", None) or 1,
            )

        return reranked, info
    if rerank != "jev":
        raise ValueError(f"unknown reranker {rerank!r}")
    from toolrank.adapters.jev import JevReranker

    client = jev_client(a)
    info["jev"] = client
    base = make

    def wrapped() -> Any:
        return JevReranker(
            base(),
            client,
            depth=a.rerank_depth,
            tool_format=a.jev_tool_format or "name_desc",
            max_chars=a.jev_max_chars,
        )

    return wrapped, info


RERANK_FLAGS = (
    "emb_url",
    "emb_model",
    "truncate",
    "clm_ckpt",
    "tool_format",
    "query_format",
    "emb_batch",
    "device",
)
RERANK_OWN = (
    "template",
    "query_chars",
)  # --rerank-<flag> with no main counterpart: the second scorer's cross_<flag>


def rerank_args(a: Any) -> Any:
    """The second scorer's flags: ``--rerank`` as its ``--scorer``, every ``--rerank-<flag>`` that
    was given over the main flag of the same name, the rest inherited (cache dir, ``--with-inst``,
    stemming); never hybrid, never a persistent index, no server term."""
    import copy

    b = copy.copy(a)
    b.scorer, b.rerank, b.hybrid, b.index, b.index_dir = a.rerank, None, False, "numpy", None
    b.server_weight = 0.0
    for f in RERANK_FLAGS:
        v = getattr(a, "rerank_" + f, None)
        if v is not None:
            setattr(b, f, v)
    for f in RERANK_OWN:
        setattr(b, "cross_" + f, getattr(a, "rerank_" + f, None))
    b.cross_max_chars = None  # the reranker cuts the text itself (cut_formatter), named in the scorer
    return b


def _base_factory(
    a: Any, *, query_timeout: float | None = None, query_attempts: int | None = None
) -> tuple[Callable[[], Any], dict[str, Any]]:
    if a.scorer not in ("dense", "clm") and getattr(a, "server_weight", 0.0):
        raise ValueError("--server-weight adds a server term to a dense or clm scorer's cosines")
    if a.scorer == "bm25":
        from toolrank.adapters.bm25 import BM25Scorer

        if getattr(a, "hybrid", False):
            raise ValueError("--hybrid fuses BM25 into a dense or clm scorer")
        qf = a.query_format or ("concat" if a.with_inst else "plain")
        tf = a.tool_format or "documentation"
        return (lambda: BM25Scorer(tf, qf, stem=not a.no_stem)), {
            "serving": {},
            "encoder": None,
            "heads_path": None,
        }
    if a.scorer == "cross":
        from toolrank.adapters.cross_encoder import CrossEncoderScorer, ScoreClient

        if getattr(a, "hybrid", False):
            raise ValueError("--hybrid fuses BM25 into a dense or clm scorer")
        client = ScoreClient(a.emb_model, a.emb_url, cache_dir=a.cache_dir or None, batch=a.emb_batch)
        return (
            lambda: CrossEncoderScorer(
                client,
                a.tool_format or "name_desc",
                template=getattr(a, "cross_template", None) or "qwen3",
                max_chars=getattr(a, "cross_max_chars", None),
                max_query_chars=getattr(a, "cross_query_chars", None) or 6000,
            )
        ), {"serving": {}, "encoder": None, "heads_path": None, "cross": client}
    if a.scorer == "jev":
        from toolrank.adapters.jev import JevScorer

        if getattr(a, "hybrid", False):
            raise ValueError("--hybrid fuses BM25 into a dense or clm scorer")
        client = jev_client(a)
        tf = a.tool_format or "name_desc"
        return (
            lambda: JevScorer(
                client, tf, chunk=a.jev_chunk, per_chunk=a.jev_per_chunk, max_chars=a.jev_max_chars
            )
        ), {"serving": {}, "encoder": None, "heads_path": None, "jev": client}
    if a.scorer not in ("dense", "clm"):
        raise ValueError(f"unknown scorer {a.scorer}")

    heads = ck = None
    serving: dict[str, Any] = {}
    if a.scorer == "clm":
        from toolrank.adapters.heads_np import load_heads

        ck = heads_path(a)
        heads = load_heads(ck, device=a.device)
        serving = dict(getattr(heads, "cfg", {}) or {})  # a packaged .npz says how it is served
    if a.truncate is None and serving.get("truncate"):
        a.truncate = int(serving["truncate"])

    from toolrank.adapters.embeddings_api import OpenAIEmbeddings

    enc = OpenAIEmbeddings(
        a.emb_model,
        a.emb_url,
        batch=a.emb_batch,
        truncate_prompt_tokens=a.truncate,
        cache_dir=a.cache_dir,
        query_timeout=query_timeout,
        query_max_retries=query_attempts,
    )
    tf = a.tool_format or serving.get("tool_format") or "name_desc"
    if a.scorer == "dense":
        qf = a.query_format or ("instruct_query" if a.with_inst else "plain")
    else:
        qf = a.query_format or ((serving.get("query_format") or "clm") if a.with_inst else "plain")
    fp = fingerprint(a, tf, ck)

    route = float(getattr(a, "server_weight", 0.0) or 0.0)

    def make() -> Any:
        if a.scorer == "dense":
            from toolrank.adapters.dense import DenseScorer

            scorer: Any = DenseScorer(enc, tf, qf, index=build_index(a), fingerprint=fp, server_weight=route)
        else:
            from toolrank.adapters.clm import CLMScorer

            scorer = CLMScorer(enc, heads, tf, qf, index=build_index(a), fingerprint=fp, server_weight=route)
            scorer.serving = serving
        if not getattr(a, "hybrid", False):
            return scorer
        from toolrank.adapters.bm25 import BM25Scorer
        from toolrank.adapters.hybrid import HybridScorer

        # no instruction: a generic one halves BM25 on MCP sets
        lexical = BM25Scorer("documentation", "plain", stem=not a.no_stem)
        return HybridScorer(scorer, lexical, k_rrf=a.rrf_k, depth=a.rrf_depth, lexical_weight=a.rrf_weight)

    return make, {"serving": serving, "encoder": enc, "heads_path": ck}


def build_retriever(
    a: Any,
    *,
    background: bool = False,
    serving_limits: bool = False,
    notify: Callable[[str], None] | None = None,
) -> Any:
    """``toolrank search`` / ``serve``: product defaults, a scorer factory, the cut rule and the
    instruction, wrapped in a ``Retriever`` over ``a.data``. ``DATA/heads`` is the retriever's heads
    dir: ``current.npz`` there replaces the heads the flags chose (unless ``--clm-ckpt`` named some),
    ``candidate.npz`` takes a share of the requests, ``tenants/<name>/`` the same per API key. ``serving_limits`` gives query embeddings
    a 10 s timeout and one retry (a server must not hang on a dead embedding endpoint); indexing the
    catalogue keeps the long ones, since a batch of long tool texts can take a while. A background
    build gets a BM25 stand-in (tool text as indexed, request without instruction) that answers
    until the semantic index is ready."""
    from toolrank.cut import rule_from_flags
    from toolrank.retriever import Retriever

    explicit_heads = a.clm_ckpt is not None  # --clm-ckpt wins over a learned DATA/heads/current.npz
    no_heads = a.clm_ckpt == "none"
    search_defaults(a)
    if a.scorer == "dense" and notify is not None and packaged_heads_fit(a.emb_model) and not no_heads:
        notify("no heads found: ranking with the embedding model alone (`toolrank heads pull` fetches them)")
    make, info = scorer_factory(
        a, query_timeout=10.0 if serving_limits else None, query_attempts=2 if serving_limits else None
    )
    instruction = a.instruction or info["serving"].get("instruction") or DEFAULT_SERVING["instruction"]
    rule = None
    if not (a.k or a.no_cut):
        rule = rule_from_flags(a.cut_margin, a.cut_threshold, a.cut_max, a.cut_min, default_margin=True)
    enc, hp = info["encoder"], info["heads_path"]
    fallback = None
    if background:

        def fallback() -> Any:
            from toolrank.adapters.bm25 import BM25Scorer

            return BM25Scorer("documentation", "plain", stem=not getattr(a, "no_stem", False))

    def variant(heads: Path, name: str) -> Any:
        """A scorer over the same encoder flags with ``heads`` (a file toolrank learn wrote), its
        index snapshot under ``index/variants/<name>``."""
        import copy

        b = copy.copy(a)
        b.clm_ckpt, b.scorer = str(heads), "clm"
        if getattr(a, "index_dir", None):
            b.index_dir = str(Path(a.index_dir) / "variants" / name.replace(":", "_"))
        return scorer_factory(
            b, query_timeout=10.0 if serving_limits else None, query_attempts=2 if serving_limits else None
        )[0]

    co_use, extra = None, int(getattr(a, "co_use", 0) or 0)
    if extra > 0:
        from toolrank.couse import CoUseTable

        if getattr(a, "no_usage_log", False):
            raise ValueError("--co-use reads the usage log: it cannot be combined with --no-usage-log")
        co_use = CoUseTable(getattr(a, "usage_log", None) or Path(a.data) / "usage")
        if notify is not None:
            notify(
                f"co-use: {len(co_use.table())} tools have partners in the usage log (up to {extra} added)"
            )
    retriever = Retriever(
        Path(a.data),
        make,
        rule=rule,
        instruction=instruction,
        fixed_k=a.k or 0,
        depth=a.cut_max,
        cache_key=enc.cache_key if enc is not None else None,
        heads_sha=file_sha256(str(hp))[:16] if hp else None,
        background=background,
        fallback=fallback,
        notify=notify,
        heads_dir=Path(a.data) / "heads",
        variants=variant,
        candidate_share=getattr(a, "candidate_share", 0.1),
        use_current=not explicit_heads,
        co_use=co_use,
        co_use_extra=extra,
        allowed=getattr(a, "allowed", None),
    )
    retriever.encoder = enc
    return retriever


def serving_of(scorer: Any) -> dict[str, Any]:
    """How the scorer's heads are served (a packaged checkpoint's cfg), {} without heads."""
    inner = getattr(scorer, "semantic", scorer)
    return dict(getattr(inner, "serving", {}) or {})


def search_defaults(a: Any) -> None:
    """Fill ``toolrank search``'s unset flags: the backbone, then the packaged heads when they belong
    on it and one is configured or cached (``TOOLRANK_HEADS`` always counts; never a download unless
    ``--clm-ckpt default``), else the backbone alone (also ``--clm-ckpt none``, for a backbone the
    cached heads do not fit); product texts and truncation."""
    from toolrank.adapters.heads_np import default_heads

    a.emb_url = a.emb_url or os.environ.get("TOOLRANK_EMB_URL") or DEFAULT_EMB_URL
    a.emb_model = a.emb_model or os.environ.get("TOOLRANK_EMB_MODEL") or DEFAULT_EMB_MODEL
    if a.clm_ckpt == "none":
        a.clm_ckpt = None
    elif a.clm_ckpt is None and (packaged_heads_fit(a.emb_model) or os.environ.get("TOOLRANK_HEADS")):
        with contextlib.suppress(FileNotFoundError):
            a.clm_ckpt = str(default_heads(url=""))
    a.scorer = "clm" if a.clm_ckpt else "dense"
    a.with_inst = True
    if a.scorer == "dense":
        a.tool_format = a.tool_format or DEFAULT_SERVING["tool_format"]
        a.query_format = a.query_format or DEFAULT_SERVING["query_format"]
        a.truncate = a.truncate or DEFAULT_SERVING["truncate"]


def heads_path(a: Any) -> Path:
    """``--clm-ckpt``: a path, ``default`` (the packaged heads), or unset (CLM's reference heads)."""
    from toolrank.adapters.clm import default_checkpoint
    from toolrank.adapters.heads_np import default_heads

    if a.clm_ckpt == "default":
        try:
            return default_heads()
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
    ck = a.clm_ckpt or default_checkpoint()
    if not ck:
        raise ValueError("no CLM checkpoint: pass --clm-ckpt or download CLM_v0.1-8B.pt (see README)")
    return Path(ck)
