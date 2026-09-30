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
