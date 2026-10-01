"""A cross-encoder over vLLM's score API (``CrossEncoderScorer``): the two prompt formats, the
score cache, ranking a small corpus, and ``--rerank cross`` / ``--scorer cross`` through the CLI
against a fake endpoint."""

from __future__ import annotations

import json
import re

import pytest

from toolrank.adapters.cross_encoder import CrossEncoderScorer, ScoreClient, prompts
from toolrank.cli import main
from toolrank.domain import Query, Tool

_WORD = re.compile(r"[a-z0-9_]+")


def _overlap_post(self, body):
    """Scores from word overlap between the query prompt and each document prompt."""
    q = set(_WORD.findall(body["text_1"].lower()))
    self.posted.append(body)
    return [len(q & set(_WORD.findall(d.lower()))) / 10.0 for d in body["text_2"]]


@pytest.fixture
def fake_score(monkeypatch):
    monkeypatch.setattr(ScoreClient, "_post", _overlap_post)
    monkeypatch.setattr(ScoreClient, "posted", [], raising=False)
    return ScoreClient


def _tools():
    return [
        Tool(id="w", doc={"name": "get_weather", "description": "Current weather and forecast for a city."}),
        Tool(id="f", doc={"name": "search_flights", "description": "Find flights between two airports."}),
        Tool(id="m", doc={"name": "send_email", "description": "Send an email message to a recipient."}),
    ]


def test_prompts_follow_each_models_format():
    t1, t2 = prompts("qwen3", "send an email", "Find the tool.", ["send_email: ...", "x"])
    assert t1.startswith("<|im_start|>system\n") and t1.endswith(
        "<Instruct>: Find the tool.\n<Query>: send an email\n"
    )
    assert t2[0].startswith("<Document>: send_email: ...") and t2[0].endswith("<think>\n\n</think>\n\n")
    t1, t2 = prompts("bge", "send an email", "", ["d"])
    assert t1.startswith("A: Given an agent's request") and t1.endswith(
        "\nsend an email\n"
    )  # the serving instruction
    assert t2 == [
        "B: d\nGiven a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    ]
    with pytest.raises(ValueError, match="unknown template"):
        prompts("other", "q", "", ["d"])


def test_client_caches_pairs_and_batches_the_rest(fake_score, tmp_path):
    c = ScoreClient("m", "http://unused", cache_dir=tmp_path, batch=2)
    s = c.score("q send email", ["send email now", "weather", "email"])
    assert s == [0.2, 0.0, 0.1] and c.calls == 2 and c.pairs == 3 and c.cached == 0
    assert c.score("q send email", ["email", "new doc", "weather"]) == [0.1, 0.0, 0.0]
    assert c.calls == 3 and c.pairs == 4 and c.cached == 2  # two of the three pairs were known
    again = ScoreClient("m", "http://unused", cache_dir=tmp_path)
    assert again.score("q send email", ["weather"]) == [0.0] and again.calls == 0 and again.cached == 1
    st = c.stats()
    assert st["calls"] == 3 and st["pairs"] == 4 and st["cached_pairs"] == 2 and st["call_ms_p50"] is not None
    assert ScoreClient.posted[0]["model"] == "m" and ScoreClient.posted[0]["text_2"] == [
        "send email now",
        "weather",
    ]


def test_scorer_ranks_a_small_corpus_and_refuses_a_large_one(fake_score):
    s = CrossEncoderScorer(ScoreClient("m", "http://unused"), "name_desc", template="qwen3", max_chars=30)
    s.index(_tools())
    (r,) = s.rank([Query(id="q", text="send an email message", qrels={}, instruction="Pick the tool.")], k=2)
    assert r.tool_ids == ["m", "w"] and r.scores[0] > r.scores[1]
    assert s.name == "cross[m,qwen3]/name_desc" and s.score_kind == "cross"
    body = ScoreClient.posted[-1]
    assert "<Instruct>: Pick the tool.\n<Query>: send an email message\n" in body["text_1"]
    docs = [d[len("<Document>: ") :].split("<|im_end|>")[0] for d in body["text_2"]]
    assert all(d.startswith("<Document>: ") for d in body["text_2"]) and max(map(len, docs)) <= 30  # cut
    s.max_tools = 2
    with pytest.raises(ValueError, match="scores every pair"):
        s.rank([Query(id="q", text="x", qrels={})], k=1)
    with pytest.raises(ValueError, match="unknown template"):
        CrossEncoderScorer(ScoreClient("m", "http://unused"), template="nope")


def test_eval_cli_reranks_with_a_cross_encoder_and_runs_one_alone(fake_score, tmp_path):
    syn = tmp_path / "syn"
    assert main(["data", "synth", "--out", str(syn), "--n-tools", "40", "--n-queries", "6"]) == 0
    out = tmp_path / "r.json"
    args = [
        "eval", "--data", str(syn), "--scorer", "bm25", "--tool-format", "documentation", "--with-inst",
        "--rerank", "cross", "--rerank-depth", "10", "--rerank-emb-url", "http://unused",
        "--rerank-emb-model", "qwen3-reranker", "--rerank-template", "bge", "--rerank-tool-format", "name_desc",
        "--rerank-max-chars", "500", "--cache-dir", str(tmp_path / "c"), "--out", str(out),
    ]  # fmt: skip
    assert main(args) == 0
    rep = json.loads(out.read_text())
    assert rep["scorer"] == "rerank[cross[qwen3-reranker,bge]/name_desc[:500],d10]/bm25/documentation/concat"
    assert rep["config"]["rerank"]["scorer"] == "cross[qwen3-reranker,bge]/name_desc[:500]"
    assert rep["config"]["cross"]["pairs"] == 60 and rep["config"]["cross"]["cached_pairs"] == 0
    assert ScoreClient.posted[-1]["text_1"].startswith("A: ")
    assert main(args) == 0 and json.loads(out.read_text())["config"]["cross"]["cached_pairs"] == 60
    alone = ["eval", "--data", str(syn), "--scorer", "cross", "--emb-url", "http://unused", "--emb-model", "m",
             "--cross-template", "qwen3", "--cache-dir", str(tmp_path / "c"), "--out", str(out)]  # fmt: skip
    assert main(alone) == 0
    rep = json.loads(out.read_text())
    assert rep["scorer"] == "cross[m,qwen3]/name_desc" and rep["config"]["cross"]["pairs"] == 6 * 40
