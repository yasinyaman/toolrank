"""A second scorer over a first scorer's shortlist (``ScorerReranker``): the order the second
scorer's cosines give, the tail below the depth, the ``--rerank-*`` flags, and the CLI end to end
with a faked endpoint."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from test_bm25_runner import _ToyEncoder
from toolrank.adapters.dense import DenseScorer
from toolrank.adapters.rerank import ScorerReranker
from toolrank.build import rerank_args
from toolrank.cli import main
from toolrank.domain import Query, RankedList, Tool
from toolrank.formats import TOOL_FORMATS


class _Fixed:
    def __init__(self, ids, scores):
        self.ids, self.scores, self.name = ids, scores, "fixed"
        self.tool_format = self.query_format = TOOL_FORMATS["name_desc"]

    def index(self, tools):
        self.indexed = list(tools)

    def rank(self, queries, k):
        return [RankedList(q.id, self.ids[:k], self.scores[:k]) for q in queries]


def _tools():
    return [
        Tool(id="w", doc={"name": "get_weather", "description": "Current weather and forecast for a city."}),
        Tool(id="f", doc={"name": "search_flights", "description": "Find flights between two airports."}),
        Tool(id="m", doc={"name": "send_email", "description": "Send an email message to a recipient."}),
        Tool(id="r", doc={"name": "create_refund", "description": "Refund a payment to the customer."}),
    ]


def test_second_scorer_reorders_the_head_and_keeps_the_tail():
    base = _Fixed(["w", "f", "m", "r"], [0.9, 0.8, 0.7, 0.6])
    second = DenseScorer(_ToyEncoder(), "name_desc", "plain")
    rr = ScorerReranker(base, second, depth=3)
    rr.index(_tools())
    q = Query(id="q", text="send an email message to Ayşe", qrels={})
    (r,) = rr.rank([q], k=4)
    assert r.tool_ids[0] == "m" and r.tool_ids[3] == "r" and r.scores[3] == -1.0
    assert r.scores[0] > r.scores[1] >= r.scores[2]  # the head carries the second scorer's cosines
    assert rr.name == "rerank[dense/toy/name_desc/plain,d3]/fixed" and rr.score_kind == "cosine"
    assert rr.last_base["q"].tool_ids == ["w", "f", "m", "r"]
    assert rr.score_tools(q, _tools()[:2]) == second.score_tools(q, _tools()[:2])


def test_max_chars_cuts_what_the_second_scorer_reads_and_names_it():
    second = DenseScorer(_ToyEncoder(), "name_desc", "plain")
    rr = ScorerReranker(_Fixed(["w", "f"], [0.9, 0.8]), second, depth=2, max_chars=11)
    assert second.tool_format.name == "name_desc[:11]" and second.tool_format(_tools()[0]) == "get_weather"
    assert second.name == "dense/toy/name_desc[:11]/plain"
    assert rr.name == "rerank[dense/toy/name_desc[:11]/plain,d2]/fixed"


def test_workers_give_the_same_lists_as_one_thread():
    base = _Fixed(["w", "f", "m", "r"], [0.9, 0.8, 0.7, 0.6])
    qs = [
        Query(id=f"q{i}", text=t, qrels={})
        for i, t in enumerate(["send an email", "refund the payment", "weather"])
    ]
    one = ScorerReranker(base, DenseScorer(_ToyEncoder(), "name_desc", "plain"), depth=4)
    many = ScorerReranker(base, DenseScorer(_ToyEncoder(), "name_desc", "plain"), depth=4, workers=3)
    one.index(_tools())
    many.index(_tools())
    assert [r.tool_ids for r in many.rank(qs, 4)] == [r.tool_ids for r in one.rank(qs, 4)]
    assert many.workers == 3 and ScorerReranker(base, one.second, workers=0).workers == 1


def test_reranker_rejects_a_scorer_without_score_tools_and_a_depth_below_two():
    second = DenseScorer(_ToyEncoder(), "name_desc", "plain")
    with pytest.raises(ValueError, match="at least 2"):
        ScorerReranker(_Fixed([], []), second, depth=1)
    with pytest.raises(ValueError, match="no score_tools"):
        ScorerReranker(_Fixed([], []), _Fixed([], []))


def test_rerank_args_take_the_rerank_flags_over_the_main_ones():
    a = argparse.Namespace(
        scorer="clm", rerank="dense", hybrid=True, index="faiss", index_dir="x", with_inst=True,
        emb_url="http://a/v1", emb_model="qwen3-emb", truncate=8192, clm_ckpt="h.npz",
        tool_format="documentation", query_format=None, emb_batch=128, device=None, cache_dir="c",
        rerank_emb_url="http://b/v1", rerank_emb_model="qwen3-8b", rerank_truncate=2048, rerank_clm_ckpt=None,
        rerank_tool_format="example_call", rerank_query_format="clm", rerank_emb_batch=None, rerank_device=None,
    )  # fmt: skip
    b = rerank_args(a)
    assert (b.scorer, b.rerank, b.hybrid, b.index, b.index_dir) == ("dense", None, False, "numpy", None)
    assert (b.emb_url, b.emb_model, b.truncate, b.tool_format, b.query_format) == (
        "http://b/v1", "qwen3-8b", 2048, "example_call", "clm"
    )  # fmt: skip
    assert (b.clm_ckpt, b.emb_batch, b.cache_dir, b.with_inst) == ("h.npz", 128, "c", True)  # inherited
    assert a.scorer == "clm" and a.hybrid is True  # the original is untouched


def test_eval_cli_reranks_bm25_with_a_dense_scorer_and_records_it(tmp_path, monkeypatch):
    from toolrank.adapters.embeddings_api import OpenAIEmbeddings

    def fake_post(self, texts, **kw):
        enc = _ToyEncoder().encode(texts)
        return [np.asarray(v, dtype=np.float32) for v in enc], 0

    monkeypatch.setattr(OpenAIEmbeddings, "_post", fake_post)
    syn = tmp_path / "syn"
    assert main(["data", "synth", "--out", str(syn), "--n-tools", "30", "--n-queries", "6"]) == 0
    out = tmp_path / "r.json"
    args = [
        "eval", "--data", str(syn), "--scorer", "bm25", "--tool-format", "documentation",
        "--rerank", "dense", "--rerank-depth", "10", "--rerank-emb-url", "http://unused/v1",
        "--rerank-emb-model", "m", "--rerank-tool-format", "name_desc",
        "--cache-dir", str(tmp_path / "c"), "--cut-margin", "0.2", "--out", str(out),
    ]  # fmt: skip
    assert main(args) == 0
    rep = json.loads(out.read_text())
    assert rep["scorer"] == "rerank[dense/emb/m/name_desc/plain,d10]/bm25/documentation/plain"
    assert rep["config"]["rerank"] == {
        "scorer": "dense/emb/m/name_desc/plain", "depth": 10, "emb_url": "http://unused/v1", "heads_sha256": None
    }  # fmt: skip
    assert rep["config"]["jev"] is None and "K@cut" in rep["overall"]  # cosine scores: the cut applies
    assert main(args + ["--rerank-max-chars", "40"]) == 0
    assert json.loads(out.read_text())["scorer"].startswith("rerank[dense/emb/m/name_desc[:40]/plain,d10]/")
    with pytest.raises(SystemExit, match="at least 2"):
        main(args[:-2] + ["--rerank-depth", "1"])


def test_rank_pairs_leaves_no_shared_state_and_returns_the_first_stages_cosines():
    base = _Fixed(["w", "f", "m", "r"], [0.9, 0.8, 0.7, 0.6])
    rr = ScorerReranker(base, DenseScorer(_ToyEncoder(), "name_desc", "plain"), depth=3)
    rr.index(_tools())
    q = Query(id="q", text="send an email message to Ayşe", qrels={})
    ((reranked, first),) = rr.rank_pairs([q], k=4)
    assert reranked.tool_ids == rr.rank([q], k=4)[0].tool_ids and reranked.tool_ids[0] == "m"
    assert first.tool_ids == ["w", "f", "m", "r"] and first.scores == [0.9, 0.8, 0.7, 0.6]
    rr.last_base = {}
    rr.rank_pairs([q], k=4)
    assert rr.last_base == {}  # concurrent server searches share the scorer


def test_a_served_search_cuts_on_first_stage_cosines_and_shows_the_reranked_order(tmp_path):
    from toolrank.cut import AdaptiveK
    from toolrank.datasets.jsonl import write_tools
    from toolrank.retriever import Retriever

    write_tools(tmp_path / "tools.jsonl", _tools())
    # first stage: weather first; within 0.15 of it are three tools; the second stage prefers e-mail
    scores = [0.9, 0.8, 0.76, 0.2]
    r = Retriever(
        tmp_path,
        lambda: ScorerReranker(
            _Fixed(["w", "f", "m", "r"], scores), DenseScorer(_ToyEncoder(), "name_desc", "plain"), depth=3
        ),
        rule=AdaptiveK(margin=0.15, max_k=10),
    )
    res = r.search("send an email message to Ayşe")
    assert len(res.hits) == 3 and res.hits[0].id == "m" and {h.id for h in res.hits} == {"w", "f", "m"}
    assert res.scorer.startswith("rerank[")


def test_search_and_serve_check_the_second_stage_flags(tmp_path, monkeypatch):
    from toolrank.datasets.jsonl import write_tools

    write_tools(tmp_path / "d" / "tools.jsonl", _tools())
    base = ["search", "email", "--data", str(tmp_path / "d"), "--cache-dir", str(tmp_path / "c")]
    with pytest.raises(SystemExit, match="--rerank-emb-url"):
        main([*base, "--rerank", "cross"])
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
        main([*base, "--rerank", "jev"])
    with pytest.raises(SystemExit, match="from 2 to 255"):
        main([*base, "--rerank", "cross", "--rerank-emb-url", "http://r/v1", "--rerank-depth", "1"])


def test_search_reranks_with_a_cross_encoder_end_to_end(tmp_path, monkeypatch, capsys):
    import hashlib

    from test_cross_encoder import _overlap_post
    from toolrank.adapters.cross_encoder import ScoreClient
    from toolrank.adapters.embeddings_api import OpenAIEmbeddings
    from toolrank.datasets.jsonl import write_tools

    def fake_post(self, texts, **kw):
        rows = [
            np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).standard_normal(16)
            for t in texts
        ]
        return [r.astype(np.float32) for r in rows], len(texts)

    monkeypatch.setattr(OpenAIEmbeddings, "_post", fake_post)
    monkeypatch.setattr(ScoreClient, "_post", _overlap_post)
    monkeypatch.setattr(ScoreClient, "posted", [], raising=False)
    monkeypatch.delenv("TOOLRANK_HEADS", raising=False)
    monkeypatch.setenv("TOOLRANK_CACHE", str(tmp_path / "no-heads"))
    write_tools(tmp_path / "d" / "tools.jsonl", _tools())
    args = [
        "search",
        "send an email message",
        "--data",
        str(tmp_path / "d"),
        "--cache-dir",
        str(tmp_path / "c"),
        "--k",
        "4",
    ]
    assert main([*args, "--json"]) == 0
    plain = [t["id"] for t in json.loads(capsys.readouterr().out)["tools"]]
    assert main([*args, "--json", "--rerank", "cross", "--rerank-emb-url", "http://reranker/v1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["tools"][0]["id"] == "m" and sorted(t["id"] for t in out["tools"]) == sorted(plain)
    assert out["scorer"].startswith("rerank[cross") and ScoreClient.posted
    assert ScoreClient.posted[0]["model"] == "qwen3-reranker"
    assert all(len(d) < 3000 + 2000 for d in ScoreClient.posted[0]["text_2"])  # the cut text, in its prompt


def test_search_reranks_with_jev_end_to_end(tmp_path, monkeypatch, capsys):
    import hashlib

    from test_jev import _overlap_post as jev_post
    from toolrank.adapters.embeddings_api import OpenAIEmbeddings
    from toolrank.adapters.jev import JevClient
    from toolrank.datasets.jsonl import write_tools

    def fake_post(self, texts, **kw):
        rows = [
            np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).standard_normal(16)
            for t in texts
        ]
        return [r.astype(np.float32) for r in rows], len(texts)

    monkeypatch.setattr(OpenAIEmbeddings, "_post", fake_post)
    monkeypatch.setattr(JevClient, "_post", jev_post)
    monkeypatch.setattr(JevClient, "posted", [], raising=False)
    monkeypatch.setattr(JevClient, "answered", [], raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.delenv("TOOLRANK_HEADS", raising=False)
    monkeypatch.setenv("TOOLRANK_CACHE", str(tmp_path / "no-heads"))
    write_tools(tmp_path / "d" / "tools.jsonl", _tools())
    args = [
        "search",
        "send an email message",
        "--data",
        str(tmp_path / "d"),
        "--cache-dir",
        str(tmp_path / "c"),
    ]
    assert main([*args, "--k", "4", "--json", "--rerank", "jev"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["tools"][0]["id"] == "m" and out["scorer"].startswith("jev[jev-1.13.0,d20,documentation")
    assert JevClient.posted  # the request went to the (faked) hosted model
