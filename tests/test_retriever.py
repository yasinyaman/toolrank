import hashlib
import threading
import time

import numpy as np
import pytest

from toolrank.adapters.bm25 import BM25Scorer
from toolrank.adapters.dense import DenseScorer
from toolrank.adapters.embeddings_api import OpenAIEmbeddings
from toolrank.adapters.hybrid import HybridScorer
from toolrank.cut import AdaptiveK
from toolrank.datasets.jsonl import write_tools
from toolrank.domain import Query, Tool
from toolrank.retriever import IndexNotReady, Retriever


class _HashEncoder:
    name = "hash"

    def encode(self, texts, *, kind="document"):
        rows = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).standard_normal(16))
        return np.asarray(rows, dtype=np.float32)


def _tools(n, tag=""):
    return [
        Tool(id=f"s/t{i}", doc={"name": f"t{i}", "description": f"tool {i}{tag}"}, category="s")
        for i in range(n)
    ]


def _dir(tmp_path, n=12, tag=""):
    write_tools(tmp_path / "tools.jsonl", _tools(n, tag))
    return tmp_path


def _make():
    return DenseScorer(_HashEncoder(), "name_desc", "instruct_query")


def test_search_cuts_ranks_and_describes(tmp_path):
    r = Retriever(
        _dir(tmp_path),
        _make,
        rule=AdaptiveK(margin=10.0, max_k=4),
        instruction="find",
        cache_key=lambda t: "k:" + t,
    )
    res = r.search("tool 3")
    assert len(res.hits) == 4 and len(res.ranked) == 12 and res.rule == "adaptive K (margin 10, max 4, min 1)"
    assert res.instruction == "find" and res.emb_key == "k:Instruct: find\nQuery: tool 3"
    assert not res.own_instruction and not r.search("tool 3", instruction="find").own_instruction
    assert r.search("tool 3", instruction="look").own_instruction  # the usage log keeps its digest only
    assert [h.id for h in r.search("tool 3", k=2).hits] == [h.id for h in res.hits[:2]]
    assert (
        res.hits[0].server == "s"
        and res.hits[0].kind == "mcp"
        and res.hits[0].input_schema == {"type": "object"}
    )
    ranked = r.rank("tool 3", _tools(3, " extra"))
    assert [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True) and len(ranked) == 3


def test_reload_swaps_the_whole_state(tmp_path):
    r = Retriever(_dir(tmp_path), _make)
    old = r.state()
    assert r.reload() is False
    time.sleep(0.01)
    write_tools(tmp_path / "tools.jsonl", _tools(15))
    r.state()  # notices the change and reloads in the background
    for _ in range(200):
        if r.state() is not old:
            break
        time.sleep(0.01)
    assert len(r.tools()) == 15 and r.get("s/t14") is not None and old.scorer is not r.state().scorer


def test_first_index_in_the_background_and_failures(tmp_path):
    gate = threading.Event()

    class _Slow(DenseScorer):
        def index(self, tools):
            gate.wait(5)
            super().index(tools)

    r = Retriever(
        _dir(tmp_path), lambda: _Slow(_HashEncoder(), "name_desc"), background=True, ready_timeout=0.05
    )
    with pytest.raises(IndexNotReady, match="still being built"):
        r.search("tool 1")
    assert r.status()["ready"] is False and not r.wait_ready(0.01)
    gate.set()
    assert r.wait_ready(5) and r.status()["sync"]["embedded"] == len(r.tools())
    assert r.search("tool 1").hits

    def _broken():
        raise RuntimeError("heads file missing")

    with pytest.raises(IndexNotReady, match="heads file missing"):
        Retriever(_dir(tmp_path), _broken, background=True, ready_timeout=5).search("x")


def test_concurrent_searches_while_the_catalogue_changes(tmp_path):
    r = Retriever(_dir(tmp_path), _make, rule=AdaptiveK(margin=0.5, max_k=5))
    errors: list[BaseException] = []

    def worker():
        try:
            for i in range(25):
                assert r.search(f"tool {i % 12}").hits
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for n in (13, 14, 15, 16):
        time.sleep(0.005)
        write_tools(tmp_path / "tools.jsonl", _tools(n, f" v{n}"))
        r.reload()
    for t in threads:
        t.join()
    assert not errors and len(r.tools()) == 16


def test_hybrid_pairs_leave_eval_state_alone(tmp_path):
    tools = _tools(12)
    h = HybridScorer(DenseScorer(_HashEncoder(), "name_desc"), BM25Scorer("name_desc", "plain"))
    h.index(tools)
    q = [Query(id="q", text="tool 5", qrels={})]
    ((fused, semantic),) = h.rank_pairs(q, 5)
    assert h.last_semantic == {}
    assert h.rank(q, 5)[0].tool_ids == fused.tool_ids and h.last_semantic["q"].tool_ids == semantic.tool_ids
    scores = h.score_tools(q[0], tools[:3])
    assert len(scores) == 3 and all(-1.0001 <= s <= 1.0001 for s in scores)


def test_embedding_cache_from_several_threads(tmp_path, monkeypatch):
    def fake_post(self, texts):
        return [np.full(4, len(t), dtype=np.float32) for t in texts], len(texts)

    monkeypatch.setattr(OpenAIEmbeddings, "_post", fake_post)
    enc = OpenAIEmbeddings("m", "http://unused/v1", cache_dir=tmp_path)
    errors: list[BaseException] = []

    def worker(n):
        try:
            for i in range(30):
                enc.encode([f"text {n} {i}", "shared"])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(enc.cache) == 6 * 30 + 1
    assert enc.cache_key("shared") == enc.cache._key("shared") and enc.cache_key("") == enc.cache._key(" ")


def test_keyword_stand_in_until_the_index_is_built_and_a_failed_build_is_retried(tmp_path):
    words = ["weather forecast", "send email", "calendar event", "refund payment"]
    tools = [
        Tool(id=f"s/{w.split()[0]}", doc={"name": w.split()[0], "description": w}, category="s")
        for w in words
    ]
    write_tools(tmp_path / "tools.jsonl", tools)
    gate, builds, events = threading.Event(), [], []

    def make():
        builds.append(1)
        if len(builds) == 1:
            raise RuntimeError("endpoint down")
        gate.wait(5)
        return _make()

    r = Retriever(
        tmp_path, make, background=True, ready_timeout=5, retry_s=0.0, notify=events.append,
        fallback=lambda: BM25Scorer("name_desc", "plain", stem=False),
    )  # fmt: skip
    for _ in range(300):  # the first semantic build fails at once
        if r.status()["error"]:
            break
        time.sleep(0.01)
    res = r.search("refund this payment")  # retries the failed build; keyword matches meanwhile
    assert (res.mode, [h.id for h in res.hits], res.emb_key) == ("lexical", ["s/refund"], None)
    assert r.get("s/send") is not None and r.status()["mode"] == "lexical"
    with pytest.raises(IndexNotReady, match="endpoint down"):
        r.rank("x", tools)  # needs the semantic index
    gate.set()
    for _ in range(300):
        if r.status()["mode"] == "semantic":
            break
        time.sleep(0.01)
    assert r.search("refund this payment").mode == "semantic" and len(builds) == 2
    kinds = [e.split(":")[0] for e in events]  # the two first builds run in parallel
    assert sorted(kinds[:2]) == ["index build failed", "keyword index ready"] and kinds[2:] == ["index ready"]


def _npz_heads(path, seed, hidden=16, width=8, backbone=None):
    """Skip heads NumpyHeads can load, without torch: random weights, so each file ranks differently."""
    import json as _json

    rng = np.random.default_rng(seed)
    cfg = {"width": width, "depth": 2, "hidden_size": hidden, "projection_dim": hidden, "skip": True}
    if backbone:
        cfg["backbone"] = backbone
    arrays = {"cfg": np.array(_json.dumps(cfg)), "logit_scale": np.zeros(1, np.float32)}
    for head in ("state_head", "action_head"):
        arrays[f"{head}/inp.weight"] = rng.standard_normal((width, hidden)).astype(np.float32)
        arrays[f"{head}/inp.bias"] = np.zeros(width, np.float32)
        arrays[f"{head}/out.weight"] = rng.standard_normal((hidden, width)).astype(np.float32)
        arrays[f"{head}/out.bias"] = np.zeros(hidden, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def _with_heads(tmp_path, **kw):
    from toolrank.adapters.clm import CLMScorer
    from toolrank.adapters.heads_np import NumpyHeads

    built = []

    def variants(path, name):
        built.append(name)
        return lambda: CLMScorer(_HashEncoder(), NumpyHeads(path), "name_desc", "instruct_query")

    r = Retriever(_dir(tmp_path), _make, fixed_k=3, heads_dir=tmp_path / "heads", variants=variants, **kw)
    return r, built


def _ready(r, name, timeout=10.0):
    """Ask for a request (which starts the build) until the variant ``name`` is built."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r.pick(arm_key="k", tenant=name.split(":")[1] if name.startswith("tenant:") else None)
        v = r.status()["heads"].get(name)
        if v and (v["ready"] or v["error"]):
            return v
        time.sleep(0.02)
    raise AssertionError(f"{name} was not built: {r.status()['heads']}")


def test_heads_files_are_picked_up_while_serving(tmp_path):
    from toolrank.retriever import bucket

    r, built = _with_heads(tmp_path, candidate_share=0.5)
    first = r.search("tool 3")
    assert (first.arm, first.heads) == ("base", None) and r.status()["heads"] == {"base": None}
    heads = tmp_path / "heads"
    _npz_heads(heads / "current.npz", seed=1)
    assert _ready(r, "current")["ready"]
    cur = r.search("tool 3", arm_key="anyone")
    assert (
        cur.arm == "current" and cur.heads == r.status()["heads"]["current"]["sha"] and len(cur.heads) == 16
    )
    assert cur.scorer.startswith("clm[current]") and cur.scorer != first.scorer

    # a candidate answers a sticky share of the requests, by the session or client key
    _npz_heads(heads / "candidate.npz", seed=2)
    assert _ready(r, "candidate")["ready"]
    inside = next(k for k in (f"s{i}" for i in range(100)) if bucket(k, 0.5))
    outside = next(k for k in (f"s{i}" for i in range(100)) if not bucket(k, 0.5))
    assert [r.search("tool 3", arm_key=inside).arm for _ in range(3)] == ["candidate"] * 3
    assert [r.search("tool 3", arm_key=outside).arm for _ in range(3)] == ["current"] * 3
    assert r.search("tool 3", arm_key=inside).heads != cur.heads

    # one API key's own heads, and its own candidate; other tenants stay on the shared ones
    _npz_heads(heads / "tenants" / "acme" / "current.npz", seed=3)
    assert _ready(r, "tenant:acme")["ready"]
    assert r.search("tool 3", arm_key=outside, tenant="acme").arm == "tenant:acme"
    assert r.search("tool 3", arm_key=outside, tenant="other").arm == "current"
    assert (
        r.search("tool 3", arm_key=inside, tenant="acme").arm == "tenant:acme"
    )  # the shared candidate is not acme's
    _npz_heads(heads / "tenants" / "acme" / "candidate.npz", seed=4)
    assert _ready(r, "tenant:acme:candidate")["ready"]
    assert r.search("tool 3", arm_key=inside, tenant="acme").arm == "tenant:acme:candidate"

    # a new file under the same name is rebuilt; a file that goes takes its variant with it
    before = r.status()["heads"]["current"]["sha"]
    time.sleep(0.01)
    _npz_heads(heads / "current.npz", seed=5)
    deadline = time.time() + 10
    while r.status()["heads"]["current"]["sha"] == before and time.time() < deadline:
        r.pick(arm_key=outside)
        time.sleep(0.02)
    assert r.search("tool 3", arm_key=outside).heads not in (before, None)
    (heads / "candidate.npz").unlink()
    assert r.search("tool 3", arm_key=inside).arm == "current" and "candidate" not in r.status()["heads"]
    assert sorted(set(built)) == ["candidate", "current", "tenant:acme", "tenant:acme:candidate"]


def test_a_broken_heads_file_or_an_explicit_checkpoint_leaves_the_base_in_place(tmp_path):
    r, _ = _with_heads(tmp_path)
    (tmp_path / "heads").mkdir()
    (tmp_path / "heads" / "current.npz").write_bytes(b"not heads")
    v = _ready(r, "current")
    assert not v["ready"] and v["error"] and r.search("tool 3").arm == "base"
    # --clm-ckpt names the heads to serve: a learned current.npz does not replace them, a candidate still runs
    (tmp_path / "x").mkdir()
    explicit, _ = _with_heads(tmp_path / "x", use_current=False, candidate_share=1.0)
    _npz_heads(tmp_path / "x" / "heads" / "current.npz", seed=1)
    _npz_heads(tmp_path / "x" / "heads" / "candidate.npz", seed=2)
    assert _ready(explicit, "candidate")["ready"]
    assert (
        explicit.search("tool 3", arm_key="a").arm == "candidate"
        and "current" not in explicit.status()["heads"]
    )
    plain = Retriever(_dir(tmp_path / "x"), _make)  # no heads dir: nothing to pick
    assert plain.search("tool 3").arm == "base" and "heads" not in plain.status()


def test_a_second_stage_search_logs_the_first_stages_model(tmp_path):
    """learn keeps only the served backbone's requests: a search behind a second stage (and a hybrid)
    must still name the model that embedded the request, and keep its embedding-cache key."""
    from toolrank.adapters.rerank import ScorerReranker

    class _First(_HashEncoder):
        model = "first"

    class _Second(_HashEncoder):
        model = "second"

    def make():
        first = HybridScorer(
            DenseScorer(_First(), "name_desc", "instruct_query"), BM25Scorer("name_desc", "plain")
        )
        return ScorerReranker(first, DenseScorer(_Second(), "name_desc", "instruct_query"), depth=5)

    r = Retriever(_dir(tmp_path), make, fixed_k=3, cache_key=lambda t: "k:" + t)
    res = r.search("tool 3")
    assert res.scorer.startswith("rerank[") and res.model == "first" and res.emb_key.startswith("k:")
    assert r.status()["sync"]["embedded"] == 12  # the first stage's index sync, under both wrappers


def test_rows_of_another_catalogue_in_a_shared_index_are_skipped(tmp_path):
    """pgvector is shared by every process that reaches the database: an id the catalogue does not
    have must not end a search with a KeyError."""

    class _Ghosts:
        name, tool_format, query_format = "ghosts", None, None

        def index(self, tools):
            self.ids = [t.id for t in tools]

        def rank(self, queries, k):
            from toolrank.domain import RankedList

            ids = ["elsewhere/x", *self.ids][:k]
            return [RankedList(q.id, ids, [1.0 - n / 100 for n in range(len(ids))]) for q in queries]

    r = Retriever(_dir(tmp_path), _Ghosts, fixed_k=3)
    res = r.search("tool 3")
    assert [h.id for h in res.hits] == ["s/t0", "s/t1", "s/t2"]
