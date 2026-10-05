"""Jev (TypeSafe AI) as a reranker and as a chunked scorer, against a fake endpoint: the request
shape, the order the probabilities give, the cache, retries, and the CLI end to end. No test
reaches api.typesafe.ai."""

from __future__ import annotations

import io
import json
import re
import urllib.error

import pytest

from toolrank.adapters.jev import (
    MAX_OPTIONS,
    JevClient,
    JevReranker,
    JevScorer,
    choice_question,
    option_key,
)
from toolrank.cli import main
from toolrank.domain import Query, RankedList, Tool
from toolrank.formats import TOOL_FORMATS

_WORD = re.compile(r"[a-z0-9_]+")


def _overlap_post(self, body):
    """A stand-in for Jev: option probabilities from word overlap with the request, normalised."""
    words = set(_WORD.findall(str(body["state"]["request"]).lower()))
    answers = {}
    for qid, q in body["questions"].items():
        assert q["type"] == "choice" and 1 <= len(q["criteria"]) <= MAX_OPTIONS
        raw = {k: len(words & set(_WORD.findall(str(v).lower()))) for k, v in q["criteria"].items()}
        total = sum(raw.values()) or 1
        probs = {k: round(v / total, 4) for k, v in raw.items()}
        best = max(probs, key=lambda k: (probs[k], -int(k[1:])))  # ties: the first option, like _order
        answers[qid] = {"type": "choice", "choice": best, "probabilities": probs, "confidence": 1.0}
    self.posted.append(body)
    self.answered.append(answers)
    tokens = sum(len(str(v)) for q in body["questions"].values() for v in q["criteria"].values()) // 4
    return {"model": body["model"], "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 1}}


@pytest.fixture
def fake_jev(monkeypatch):
    monkeypatch.setattr(JevClient, "_post", _overlap_post)
    monkeypatch.setattr(JevClient, "posted", [], raising=False)
    monkeypatch.setattr(JevClient, "answered", [], raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    return JevClient


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


def test_choice_question_shape_and_limits():
    q = choice_question("Retrieve the tool.", ["a" * 20, "b"], max_chars=5)
    assert q["type"] == "choice" and q["instructions"].startswith("Retrieve the tool.\n\n")
    assert q["criteria"] == {"t000": "aaaaa", "t001": "b"}
    assert "\n\n" not in choice_question("", ["x"], 10)["instructions"]
    with pytest.raises(ValueError, match="1..255"):
        choice_question("", [], 10)
    with pytest.raises(ValueError, match="1..255"):
        choice_question("", ["x"] * (MAX_OPTIONS + 1), 10)
    assert option_key(7) == "t007"


def test_reranker_orders_the_head_by_probability_and_keeps_the_tail(fake_jev, tmp_path):
    base = _Fixed(["w", "f", "m", "r"], [0.9, 0.8, 0.7, 0.6])
    rr = JevReranker(base, JevClient(cache_dir=tmp_path), depth=3, max_chars=200)
    rr.index(_tools())
    q = Query(id="q", text="please send an email message to Ayşe", qrels={}, instruction="Find the tool.")
    (r,) = rr.rank([q], k=4)
    # head (w, f, m): m matches "send", "email", "message"; w and f tie at 0 and keep the base order
    assert r.tool_ids == ["m", "w", "f", "r"] and r.scores[0] == 1.0 and r.scores[1:3] == [0.0, 0.0]
    assert r.scores[3] == -1.0  # below the depth: the base order, negative scores
    assert rr.name == "jev[jev-1.13.0@api.typesafe.ai,d3,name_desc]/fixed" and rr.score_kind == "jev"
    body = JevClient.posted[-1]
    assert body["model"] == "jev-1.13.0" and body["state"] == {"request": q.text}
    assert body["questions"]["tool"]["instructions"] == (
        "Find the tool.\n\nWhich of the listed tools should be called to carry out `request`?"
    )
    assert list(body["questions"]["tool"]["criteria"]) == ["t000", "t001", "t002"]
    assert body["questions"]["tool"]["criteria"]["t002"].startswith("send_email: Send an email")


def test_reranker_depth_is_one_choice_question():
    with pytest.raises(ValueError, match="2..255"):
        JevReranker(_Fixed([], []), JevClient(api_key="k"), depth=MAX_OPTIONS + 1)
    with pytest.raises(ValueError, match="2..255"):
        JevReranker(_Fixed([], []), JevClient(api_key="k"), depth=1)


def test_a_single_candidate_asks_nothing(fake_jev):
    rr = JevReranker(_Fixed(["w"], [0.9]), JevClient(), depth=5)
    rr.index(_tools())
    (r,) = rr.rank([Query(id="q", text="x", qrels={})], k=5)
    assert r.tool_ids == ["w"] and JevClient.posted == []


def test_cache_answers_a_repeated_request_without_a_call(fake_jev, tmp_path):
    c = JevClient(cache_dir=tmp_path)
    q = {"tool": choice_question("", ["send email", "weather"], 100)}
    a1 = c.ask({"request": "send an email"}, q)
    a2 = c.ask({"request": "send an email"}, q)
    assert a1 == a2 and c.calls == 1 and c.cached == 1 and c.tokens_spent > 0
    again = JevClient(cache_dir=tmp_path)  # a new process reads the same file
    assert again.ask({"request": "send an email"}, q) == a1 and again.calls == 0 and again.cached == 1
    s = c.stats()
    assert (
        s["calls"] == 1 and s["cached"] == 1 and s["call_ms_p50"] is not None and s["model"] == "jev-1.13.0"
    )


def test_standalone_scorer_chunks_the_corpus_and_reranks_the_winners(fake_jev, tmp_path):
    from toolrank.datasets.jsonl import load_queries, load_tools
    from toolrank.datasets.synthetic import write_synthetic

    write_synthetic(tmp_path, n_tools=60, n_queries=8, seed=3)
    tools, queries = load_tools(tmp_path / "tools.jsonl"), load_queries(tmp_path / "queries.jsonl")
    s = JevScorer(JevClient(workers=1), "name_desc", chunk=25, per_chunk=5)  # one thread: calls in order
    s.index(tools)
    ranked = s.rank(queries, k=10)
    assert s.name == "jev[jev-1.13.0@api.typesafe.ai,c25x5,name_desc]"
    assert all(len(r.tool_ids) == 10 and len(set(r.tool_ids)) == 10 for r in ranked)
    # round one: 3 chunks per query; round two: one final Choice over the 15 winners per query
    assert len(JevClient.posted) == 8 * 3 + 8
    finals = [
        (b, a)
        for b, a in zip(JevClient.posted, JevClient.answered, strict=True)
        if len(b["questions"]["tool"]["criteria"]) == 15
    ]
    assert len(finals) == 8
    # every list starts with the final round's choice, and its winners precede the other tools
    for r, (body, answer) in zip(ranked, finals, strict=True):
        assert (
            s.texts[s.ids.index(r.tool_ids[0])]
            == body["questions"]["tool"]["criteria"][answer["tool"]["choice"]]
        )
    full = s.rank(queries[:1], k=60)
    assert len(full[0].tool_ids) == 60 and full[0].scores[15] < 0  # the tail: round one, negative scores


def test_standalone_scorer_with_one_chunk_skips_the_final_round(fake_jev):
    s = JevScorer(JevClient(), "name_desc", chunk=10)
    s.index(_tools())
    (r,) = s.rank([Query(id="q", text="refund the payment", qrels={})], k=10)
    assert r.tool_ids[0] == "r" and len(JevClient.posted) == 1 and len(r.tool_ids) == 4


def test_post_retries_429_with_retry_after_and_shows_a_400_body(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr("time.sleep", sleeps.append)

    def flaky(req, timeout):
        calls.append(json.loads(req.data))
        assert req.get_header("Authorization") == "Bearer k" and req.full_url.endswith("/v1/systemone")
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                req.full_url, 429, "slow down", {"retry-after": "2"}, io.BytesIO(b"")
            )
        return io.BytesIO(json.dumps({"model": "m", "answers": {}, "usage": {"input_tokens": 3}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    c = JevClient("m", "https://x/v1", api_key="k")
    assert c.ask("s", {}) == {} and len(calls) == 2 and sleeps == [2.0] and c.tokens_spent == 3

    def bad(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 400, "bad", {}, io.BytesIO(b'{"error": "too many options"}')
        )

    monkeypatch.setattr("urllib.request.urlopen", bad)
    with pytest.raises(RuntimeError, match="HTTP 400 .*too many options"):
        c.ask("s", {})
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):  # TypeSafe itself always wants a key
        JevClient("m", "https://api.typesafe.ai/v1").ask("s", {})


def test_eval_cli_reranks_bm25_and_records_the_jev_block(fake_jev, tmp_path, capsys):
    from toolrank.datasets.synthetic import write_synthetic

    write_synthetic(tmp_path / "d", n_tools=40, n_queries=6, seed=1)
    out = tmp_path / "r.json"
    args = ["eval", "--data", str(tmp_path / "d"), "--scorer", "bm25", "--tool-format", "name_desc"]
    jev = ["--rerank", "jev", "--rerank-depth", "10", "--cache-dir", str(tmp_path / "c"), "--out", str(out)]
    assert main(args + jev) == 0
    rep = json.loads(out.read_text())
    assert rep["scorer"].startswith("jev[jev-1.13.0@api.typesafe.ai,d10,name_desc]/bm25")
    j = rep["config"]["jev"]
    assert j["rerank_depth"] == 10 and j["calls"] == 6 and j["cached"] == 0 and j["input_tokens"] > 0
    assert "jev calls 6 (+0 cached)" in capsys.readouterr().out
    assert main(args + jev) == 0  # the second run is all cache hits
    assert json.loads(out.read_text())["config"]["jev"]["cached"] == 6
    # standalone, and the guards
    assert (
        main(
            ["eval", "--data", str(tmp_path / "d"), "--scorer", "jev", "--jev-chunk", "16", "--out", str(out)]
        )
        == 0
    )
    rep = json.loads(out.read_text())
    assert (
        rep["scorer"] == "jev[jev-1.13.0@api.typesafe.ai,c16x20,name_desc]"
        and rep["config"]["emb_url"] is None
    )
    assert rep["config"]["jev"]["chunk"] == 16
    with pytest.raises(SystemExit, match="cosine scores"):
        main(args + jev + ["--cut-margin", "0.2"])


def test_eval_cli_refuses_to_start_without_a_key(monkeypatch, tmp_path):
    from toolrank.datasets.synthetic import write_synthetic

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    write_synthetic(tmp_path, n_tools=10, n_queries=2, seed=1)
    with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
        main(["eval", "--data", str(tmp_path), "--scorer", "bm25", "--rerank", "jev"])


def test_search_and_serve_ask_for_the_key_only_for_typesafe(monkeypatch, tmp_path):
    from toolrank.cli import _check_rerank, build_parser

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    for cmd in ("search", "serve"):
        args = [
            cmd,
            *(["send an email"] if cmd == "search" else []),
            "--data",
            str(tmp_path),
            "--rerank",
            "jev",
        ]
        with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
            _check_rerank(build_parser().parse_args(args))
        _check_rerank(build_parser().parse_args([*args, "--jev-url", "http://127.0.0.1:8093/v1"]))  # no exit


def test_the_key_goes_to_typesafe_alone(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    assert JevClient("m", "https://api.typesafe.ai/v1").api_key == "secret"
    for url in ("http://127.0.0.1:8093/v1", "https://elsewhere.example/v1"):
        assert JevClient("m", url).api_key is None  # never sent to a local or third-party endpoint

    sent = []

    def ok(req, timeout):
        sent.append(req)
        return io.BytesIO(json.dumps({"model": "m", "answers": {}, "usage": {}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", ok)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    JevClient("m", "http://127.0.0.1:8093/v1").ask("s", {})  # a local endpoint: none needed, none sent
    assert sent[0].get_header("Authorization") is None
    JevClient("m", "http://127.0.0.1:8093/v1", api_key="k").ask("s", {})  # an explicit key goes as asked
    assert sent[1].get_header("Authorization") == "Bearer k"


def test_each_endpoint_has_its_own_cache(fake_jev, tmp_path):
    q = {"tool": choice_question("", ["send email", "weather"], 100)}
    ts = JevClient(cache_dir=tmp_path)  # TypeSafe stays in jev.sqlite, its keys unchanged
    ts.ask({"request": "send an email"}, q)
    assert (tmp_path / "jev.sqlite").exists()
    local = JevClient("m", "http://127.0.0.1:8093/v1", cache_dir=tmp_path)
    local.ask({"request": "send an email"}, q)  # the same body at another endpoint is asked again
    assert (local.calls, local.cached) == (1, 0)
    assert (tmp_path / "jev-127.0.0.1-8093.sqlite").exists()
    again = JevClient("m", "http://127.0.0.1:8093/v1", cache_dir=tmp_path)
    again.ask({"request": "send an email"}, q)
    assert (again.calls, again.cached) == (0, 1)  # a new process reads that endpoint's own file


def test_standalone_scorer_never_sends_a_one_option_question(fake_jev):
    tools = _tools() + [Tool(id=f"x{i}", doc={"name": f"t{i}", "description": "d"}) for i in range(7)]
    s = JevScorer(JevClient(workers=1), "name_desc", chunk=10, per_chunk=2)
    s.index(tools)  # 11 tools: chunks of 10 and 1
    (r,) = s.rank([Query(id="q", text="refund the payment", qrels={})], k=11)
    # only the 10-option chunk is asked (the single wins by default), then one final over 3 winners
    sizes = [len(b["questions"]["tool"]["criteria"]) for b in JevClient.posted]
    assert sizes == [10, 3] and len(r.tool_ids) == 11


def test_eval_cli_reranks_through_a_local_endpoint_without_a_key(fake_jev, tmp_path, monkeypatch):
    from toolrank.datasets.synthetic import write_synthetic

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    write_synthetic(tmp_path / "d", n_tools=20, n_queries=3, seed=1)
    out = tmp_path / "r.json"
    assert (
        main(
            [
                "eval",
                "--data",
                str(tmp_path / "d"),
                "--scorer",
                "bm25",
                "--tool-format",
                "name_desc",
                "--rerank",
                "jev",
                "--rerank-depth",
                "10",
                "--jev-url",
                "http://127.0.0.1:8093/v1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    rep = json.loads(out.read_text())
    assert rep["scorer"].startswith("jev[jev-1.13.0@127.0.0.1:8093,d10,name_desc]/bm25")
