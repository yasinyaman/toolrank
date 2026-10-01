"""Server routing (``DenseScorer(server_weight=...)``, ``--server-weight``): a tool's score gains a
share of the request's cosine with its server's summary."""

import hashlib
import json

import numpy as np
import pytest

from toolrank.adapters.dense import DenseScorer
from toolrank.adapters.embeddings_api import OpenAIEmbeddings, l2_normalize
from toolrank.cli import main
from toolrank.datasets.jsonl import write_queries, write_tools
from toolrank.domain import Query, Tool
from toolrank.formats import SERVER_SUMMARY_CHARS, server_summary


class _HashEncoder:
    """Deterministic pseudo-random vectors per text; records every text it was asked for."""

    name = "hash"

    def __init__(self, dim=16):
        self.dim, self.seen = dim, []

    def encode(self, texts, *, kind="document"):
        self.seen += list(texts)
        rows = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).standard_normal(self.dim))
        return np.asarray(rows, dtype=np.float32)


def _tools(servers=("mail", "files", "maps"), per=4):
    return [
        Tool(id=f"{s}/t{i}", doc={"name": f"{s}_tool{i}", "description": f"does {s} thing {i}"}, category=s)
        for s in servers
        for i in range(per)
    ]


def _queries(n=6):
    return [Query(id=f"q{i}", text=f"request number {i}", qrels={}) for i in range(n)]


def test_server_summary_names_the_server_and_its_tools():
    tools = _tools(("mail",), 3)
    assert json.loads(server_summary("mail", tools)) == {
        "server": "mail",
        "tools": ["mail_tool0", "mail_tool1", "mail_tool2"],
    }
    assert len(server_summary("big", _tools(("big",), 2000))) == SERVER_SUMMARY_CHARS


def test_server_weight_adds_the_servers_cosine_to_its_tools():
    tools, queries, enc = _tools(), _queries(), _HashEncoder()
    plain = DenseScorer(enc, "name_desc", "plain")
    routed = DenseScorer(enc, "name_desc", "plain", server_weight=0.3)
    assert routed.name == "dense/hash/name_desc/plain+srv0.3" and plain.name == "dense/hash/name_desc/plain"
    plain.index(tools)
    routed.index(tools)

    t = l2_normalize(enc.encode([plain.tool_format(x) for x in tools]))
    q = l2_normalize(enc.encode([x.text for x in queries]))
    names = ["files", "mail", "maps"]
    s = l2_normalize(enc.encode([server_summary(n, [x for x in tools if x.category == n]) for n in names]))
    of = [names.index(x.category) for x in tools]
    want = q @ t.T + 0.3 * (q @ s.T)[:, of]
    for i, r in enumerate(routed.rank(queries, k=5)):
        order = np.argsort(-want[i])[:5]
        assert r.tool_ids == [tools[j].id for j in order]
        assert np.allclose(r.scores, want[i][order], atol=1e-5)
    # the term moves tools across servers, never within one
    for a, b in zip(plain.rank(queries, k=12), routed.rank(queries, k=12), strict=True):
        for n in names:
            assert [x for x in a.tool_ids if x.startswith(n)] == [x for x in b.tool_ids if x.startswith(n)]
    assert any(
        a.tool_ids != b.tool_ids
        for a, b in zip(plain.rank(queries, k=12), routed.rank(queries, k=12), strict=True)
    )


def test_one_server_is_left_alone_and_a_changed_server_is_summarised_again():
    enc = _HashEncoder()
    routed = DenseScorer(enc, "name_desc", "plain", server_weight=0.5)
    single = _tools(("mail",), 5)
    routed.index(single)
    assert routed._servers is None and not any('"server"' in x for x in enc.seen)  # nothing to vote on
    plain = DenseScorer(_HashEncoder(), "name_desc", "plain")
    plain.index(single)
    assert [r.tool_ids for r in routed.rank(_queries(), 3)] == [r.tool_ids for r in plain.rank(_queries(), 3)]

    tools = _tools()
    routed.index(tools)
    first = [x for x in enc.seen if x.startswith('{"server"')]
    assert len(first) == 3 and routed._servers.shape == (3, 16)
    moved = [
        *tools,
        Tool(id="maps/new", doc={"name": "maps_route", "description": "plans a route"}, category="maps"),
    ]
    routed.index(moved)
    again = [x for x in enc.seen if x.startswith('{"server"')][3:]
    assert (
        len(again) == 3 and "maps_route" in again[2] and again[:2] == first[:2]
    )  # a real encoder caches the rest
    assert routed._server_of["maps/new"] == 2
    assert len(routed.rank(_queries(1), k=200)[0].tool_ids) == 13  # every tool, once


def test_eval_takes_server_weight_for_dense_scorers_only(tmp_path, monkeypatch, capsys):
    def fake_post(self, texts):
        rows = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).standard_normal(16).astype(np.float32))
        return rows, 5 * len(texts)

    monkeypatch.setattr(OpenAIEmbeddings, "_post", fake_post)
    tools = _tools()
    write_tools(tmp_path / "d" / "tools.jsonl", tools)
    write_queries(
        tmp_path / "d" / "queries.jsonl",
        [Query(id=f"q{i}", text=f"request number {i}", qrels={tools[i].id: 1}) for i in range(6)],
    )
    base = ["eval", "--data", str(tmp_path / "d"), "--cache-dir", str(tmp_path / "c")]
    dense = [*base, "--scorer", "dense", "--emb-url", "http://unused/v1", "--emb-model", "fake"]
    out = tmp_path / "r.json"
    assert main([*dense, "--server-weight", "0.2", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["scorer"].endswith("+srv0.2") and report["config"]["server_weight"] == 0.2
    assert main([*dense, "--out", str(out)]) == 0
    assert json.loads(out.read_text())["config"]["server_weight"] is None
    with pytest.raises(SystemExit, match="dense or clm"):
        main([*base, "--scorer", "bm25", "--server-weight", "0.2"])
    # the server term belongs to the first stage: a second scorer over its shortlist gets none
    rerank = ["--rerank", "dense", "--rerank-depth", "5", "--rerank-emb-url", "http://unused/v1"]
    assert (
        main([*dense, "--server-weight", "0.2", *rerank, "--rerank-emb-model", "fake", "--out", str(out)])
        == 0
    )
    assert "+srv0.2" in json.loads(out.read_text())["scorer"]
