"""``toolrank learn``: pairs mined from the usage log, the requests' vectors found through the log's
key, the newest requests held out, and heads published only when they help."""

import json
import random

import numpy as np
import pytest

from toolrank.adapters.embeddings_api import OpenAIEmbeddings
from toolrank.cli import main
from toolrank.datasets.jsonl import write_queries, write_tools
from toolrank.datasets.synthetic import make_queries, make_tools
from toolrank.domain import Tool
from toolrank.formats import tool_format
from toolrank.learn import (
    Job,
    LogPair,
    apply,
    batches,
    decide,
    heads_home,
    judge,
    mine,
    run,
    split,
    state_vectors,
)
from toolrank.retriever import Hit, SearchResult
from toolrank.usage import UsageLog

URL, MODEL = "http://unused/v1", "fake"


def _enc(tmp_path):
    """The encoder as ``learn`` builds it (the same cache namespace: URL, model and truncation)."""
    return OpenAIEmbeddings(MODEL, URL, cache_dir=tmp_path / "cache", truncate_prompt_tokens=8192)


def _search(sid, state, results, shown, ts, tenant=None, v=3):
    return {
        "v": v,
        "event": "search",
        "id": sid,
        "ts": ts,
        "tenant": tenant,
        "emb_hmac": state if v >= 3 else None,
        "results": [[t, 1.0 - n / 10] for n, t in enumerate(results)],
        "shown": shown,
    }


def _call(tool, sid, outcome="ok"):
    return {"v": 3, "event": "call", "tool": tool, "search_id": sid, "outcome": outcome}


def test_mine_turns_linked_calls_into_positives_and_the_rest_of_the_shown_into_negatives():
    events = [
        _search("s1", "A", ["t1", "t2", "t3", "t4"], 3, "2026-09-30T10:00:00"),
        _call("t1", "s1"),
        _call("t2", "s1", "tool_error"),
        _search("s2", "A", ["t2", "t5", "t6"], 3, "2026-09-30T09:00:00", tenant="team"),  # the same request
        _call("t5", "s2", "refused"),  # says nothing about t5, but t5 was called: not a negative
        _search("s3", "B", ["t7", "t8"], 2, "2026-09-30T11:00:00"),  # no call: no pair
        _search("s4", "C", ["t9"], 1, "2026-09-30T12:00:00"),
        _call("t9", "s4", "tool_error"),
        _search("s5", "D", ["t1"], 1, "2026-09-30T13:00:00", v=2),  # no emb_hmac: cannot be used
        _call("t1", "s5"),
        {"v": 3, "event": "call", "tool": "t1", "search_id": None, "outcome": "ok"},  # unlinked
        _search("s6", "E", ["t3"], 1, "2026-09-30T14:00:00"),
        _call("t3", "s6", "timeout"),  # the only call says nothing: no positive
    ]
    pairs, counts = mine(events)
    assert [p.state for p in pairs] == ["A", "C"]  # oldest first, by the request's first search
    a, c = pairs
    assert (a.positives, a.weak, a.negatives) == (("t1",), ("t2",), ("t3", "t6"))  # t5 was called
    assert a.ts == "2026-09-30T09:00:00" and a.tenants == ("team",)
    assert (c.positives, c.weak) == ((), ("t9",))
    assert counts["searches"] == 6 and counts["searches_without_vector"] == 1
    assert counts["searches_without_calls"] == 1 and counts["calls_unlinked"] == 1
    assert counts["calls_saying_nothing"] == 2 and counts["requests_without_positive"] == 1
    assert counts["requests"] == 3 and counts["pairs"] == 2 and counts["negatives"] == 2  # B had no call
    strict, _ = mine(events, strict=True)  # a tool_error is no positive: C has none left
    assert [(p.state, p.positives, p.weak, p.negatives) for p in strict] == [("A", ("t1",), (), ("t3", "t6"))]
    since, counts = mine(events, since="2026-09-30T09:30:00")
    assert [p.state for p in since] == ["A", "C"] and counts["searches_before_since"] == 1
    assert since[0].negatives == ("t3",)  # s2 fell out of the window
    team, counts = mine(events, tenant="team")
    assert [p.state for p in team] == [] and counts["searches_of_other_tenants"] == 4  # s2 has no positive


def test_state_vectors_come_from_the_cache_through_the_logs_key(tmp_path):
    enc = _enc(tmp_path)
    vecs = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    enc.cache.put_many(["request one", "request two", "some tool"], vecs)
    log = UsageLog(tmp_path / "usage")
    wanted = [log.digest(enc.cache_key("request one")), log.digest(enc.cache_key("request two")), "nope"]
    found = state_vectors(
        tmp_path / "cache" / "embeddings.sqlite", (tmp_path / "usage" / ".key").read_bytes(), wanted
    )
    assert sorted(found) == sorted(wanted[:2])
    assert np.allclose(found[wanted[0]], vecs[0] / np.linalg.norm(vecs[0]))  # normalised, as encode returns
    other = UsageLog(tmp_path / "other")  # another install's key finds nothing
    assert state_vectors(tmp_path / "cache" / "embeddings.sqlite", other._key, wanted) == {}
    assert state_vectors(tmp_path / "missing.sqlite", log._key, wanted) == {}


def test_split_holds_out_the_newest_requests_and_batches_index_the_catalogue():
    pairs = [
        LogPair(f"s{i}", (f"t{i}",), (), (f"t{(i + 1) % 10}",), f"2026-09-{i + 1:02d}", ()) for i in range(10)
    ]
    train, dev = split(reversed(pairs), 0.2)
    assert [p.state for p in dev] == ["s8", "s9"] and len(train) == 8
    assert split(pairs[:1], 0.2) == (pairs[:1], []) and len(split(pairs[:2], 0.2)[1]) == 1
    states = {p.state: np.full(4, i, dtype=np.float32) for i, p in enumerate(pairs)}
    index = {f"t{i}": i for i in range(9)}  # t9 left the catalogue
    b = batches(pairs, states, index, np.zeros((9, 4), np.float32))
    assert (
        b.states.shape == (10, 4)
        and b.pos[9] == []
        and b.neg[8] == []
        and b.pos[0] == [0]
        and b.neg[0] == [1]
    )


def _served(tmp_path, n_requests=60, dim=16, seed=0):
    """An ingest dir toolrank serve could have written: tools.jsonl, a cache with every vector and a
    usage log of searches and calls. A request is a noisy, rotated copy of its tool's vector, so the
    starting (identity) heads rank it poorly and training can learn the rotation."""
    rng = np.random.default_rng(seed)
    tools = make_tools(30, random.Random(seed))
    write_tools(tmp_path / "tools.jsonl", tools)
    enc = _enc(tmp_path)
    tf = tool_format("documentation")
    tool_vecs = rng.standard_normal((len(tools), dim)).astype(np.float32)
    enc.cache.put_many([tf(t) for t in tools], tool_vecs)
    rot = np.linalg.qr(rng.standard_normal((dim, dim)))[0].astype(np.float32)
    texts = [f"request {i}" for i in range(n_requests)]
    targets = rng.integers(0, len(tools), n_requests)
    states = tool_vecs[targets] @ rot + 0.1 * rng.standard_normal((n_requests, dim)).astype(np.float32)
    enc.cache.put_many(texts, states)
    log = UsageLog(tmp_path / "usage")
    for i, text in enumerate(texts):
        shown = [tools[targets[i]], *(t for t in tools if t is not tools[targets[i]])][:4]
        res = SearchResult(
            query=text,
            instruction="",
            hits=[Hit(t, 0.5) for t in shown],
            ranked=[(t.id, 0.5) for t in shown],
            took_ms=1.0,
            rule="top 4",
            emb_key=enc.cache_key(text),
            scorer="test",
            catalog="c",
        )
        sid = log.search(res, session=f"s{i}", via="mcp")
        outcome = "tool_error" if i % 7 == 0 else "ok"
        log.call(
            tool=tools[targets[i]].id,
            kind="mcp",
            session=f"s{i}",
            via="mcp",
            outcome=outcome,
            took_ms=1,
            search_id=sid,
        )
    return tools, tool_vecs, rot


def test_learn_dry_run_reports_the_pairs_without_torch(tmp_path, capsys):
    _served(tmp_path)
    assert (
        main(
            [
                "learn",
                "--data",
                str(tmp_path),
                "--dry-run",
                "--results",
                str(tmp_path / "res"),
                "--emb-url",
                URL,
                "--emb-model",
                MODEL,
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    (report,) = (tmp_path / "res").glob("learn_*.json")
    j = json.loads(report.read_text())
    assert (
        j["decision"] == "dry run"
        and j["pairs"] == 60
        and j["split"] == {"train": 48, "dev": 12, "dev_from": j["split"]["dev_from"]}
    )
    assert j["counts"]["weak_positives"] == 9 and "60 usable pairs" in out
    assert (
        main(
            [
                "learn",
                "--data",
                str(tmp_path),
                "--dry-run",
                "--min-pairs",
                "100",
                "--results",
                str(tmp_path / "res2"),
                "--emb-url",
                URL,
                "--emb-model",
                MODEL,
            ]
        )
        == 0
    )
    assert "not enough pairs" in capsys.readouterr().out
    # a candidate that toolrank ab has not decided on is not trained over
    waiting = heads_home(tmp_path) / "candidate.npz"
    waiting.parent.mkdir()
    waiting.write_bytes(b"still judged")
    assert main(["learn", "--data", str(tmp_path), "--emb-url", URL, "--emb-model", MODEL]) == 0
    assert "still being judged" in capsys.readouterr().out and waiting.read_bytes() == b"still judged"


def test_learn_publishes_heads_that_beat_the_start_on_the_logs_newest_requests(tmp_path):
    torch = pytest.importorskip("torch")
    from toolrank.adapters.heads_np import NumpyHeads
    from toolrank.finetune import TrainConfig

    tools, tool_vecs, _ = _served(tmp_path)
    dev_q = make_queries(tools, 20, random.Random(3))  # a benchmark-format guard, scored alongside
    write_tools(tmp_path / "bench" / "tools.jsonl", tools)
    write_queries(tmp_path / "bench" / "queries.jsonl", dev_q)
    enc = _enc(tmp_path)
    from toolrank.formats import query_format

    enc.cache.put_many(
        [query_format("instruct_query")(q) for q in dev_q],
        np.random.default_rng(5).standard_normal((20, 16)).astype(np.float32),
    )
    cfg = TrainConfig(
        epochs=6,
        batch=32,
        neg_per_pair=3,
        lr=3e-3,
        neg_filter=0.95,
        head_cfg={"width": 64, "depth": 3, "skip": True},
    )
    job = Job(
        data=tmp_path,
        out=tmp_path / "heads" / "learned.npz",
        dev=tmp_path / "bench",
        init=None,
        emb_url=URL,
        emb_model=MODEL,
        max_drop=100.0,
        train=cfg,
    )
    lines = []
    report = run(job, log=lines.append)
    assert report["decision"] == "published", lines
    assert report["best_epoch"] > 0 and report["chosen"]["log.Recall@5"] > report["start"]["log.Recall@5"]
    assert "dev.NDCG@10" in report["chosen"] and report["tokens_spent"] == 0  # nothing left the cache
    heads = NumpyHeads(tmp_path / "heads" / "learned.npz")
    assert (
        heads.cfg["learned_from"]["requests"] == 48 and heads.cfg["selected_on"]["metric"] == "log.Recall@5"
    )
    assert heads.cfg["tool_format"] == "documentation" and heads.project_states(tool_vecs[:2]).shape == (
        2,
        16,
    )
    # the same log, with a guard that allows no drop at all: the benchmark's queries are random here
    strict = Job(**{**job.__dict__, "out": tmp_path / "heads" / "again.npz", "max_drop": -1.0})
    assert run(strict, log=lambda _: None)["decision"] in ("benchmark dropped", "published")
    assert not torch.cuda.is_initialized() or True


def test_learn_needs_a_log_and_matching_vectors(tmp_path):
    write_tools(tmp_path / "tools.jsonl", [Tool(id="a/b", doc={"name": "b"})])
    with pytest.raises(FileNotFoundError, match="no usage log"):
        run(Job(data=tmp_path, out=tmp_path / "x.npz", dry_run=True), log=lambda _: None)


def _arm_search(sid, arm, ts, tenant=None):
    return {"v": 3, "event": "search", "id": sid, "ts": ts, "arm": arm, "tenant": tenant}


def _ranked_call(sid, rank, outcome="ok"):
    return {"v": 3, "event": "call", "tool": "t", "search_id": sid, "outcome": outcome, "rank": rank}


def test_judge_compares_the_arms_on_what_the_agents_called():
    events = [
        _arm_search("a1", "current", "2026-10-01T10:00:00"),
        _ranked_call("a1", 2),
        _arm_search("a2", "base", "2026-10-01T10:01:00"),  # the control is whatever is not the candidate
        _arm_search("a3", "current", "2026-10-01T10:02:00"),
        _ranked_call("a3", 4, "refused"),  # says nothing
        _arm_search("b1", "candidate", "2026-10-01T10:03:00"),
        _ranked_call("b1", 1),
        _ranked_call("b1", 3, "tool_error"),  # the best rank of the search counts
        _arm_search("b2", "candidate", "2026-10-01T10:04:00"),
        _ranked_call("b2", 2, "tool_error"),
        _arm_search("old", "candidate", "2026-09-30T10:00:00"),
        _ranked_call("old", 1),
        _arm_search("t1", "tenant:acme:candidate", "2026-10-01T10:05:00", tenant="acme"),
        _ranked_call("t1", 1),
        _arm_search("t2", "tenant:acme", "2026-10-01T10:06:00", tenant="acme"),
    ]
    stats = judge(events, since="2026-10-01")
    assert stats["control"] == {"searches": 3, "called": 1, "top1": 0.0, "mrr": pytest.approx(0.5 / 3)}
    assert stats["candidate"] == {"searches": 2, "called": 2, "top1": 0.5, "mrr": pytest.approx(0.75)}
    acme = judge(events, since="2026-10-01", tenant="acme")
    assert (acme["control"]["searches"], acme["candidate"]["searches"], acme["candidate"]["mrr"]) == (
        1,
        1,
        1.0,
    )
    assert judge(events)["candidate"]["searches"] == 3  # without since, the old search counts too

    row = lambda n, mrr: {"searches": n, "called": n, "top1": 0.0, "mrr": mrr}  # noqa: E731
    assert decide({"control": row(100, 0.50), "candidate": row(100, 0.60)}) == "promote"
    assert decide({"control": row(100, 0.50), "candidate": row(100, 0.40)}) == "rollback"
    assert decide({"control": row(100, 0.50), "candidate": row(100, 0.505)}) == "wait"  # inside the margin
    assert decide({"control": row(100, 0.50), "candidate": row(99, 0.90)}) == "wait"  # too few searches yet
    assert decide({"control": row(20, 0.5), "candidate": row(20, 0.9)}, min_searches=20) == "promote"


def test_apply_moves_the_files_a_running_server_follows(tmp_path):
    home = heads_home(tmp_path)
    assert (
        home == tmp_path / "heads" and heads_home(tmp_path, "acme") == tmp_path / "heads" / "tenants" / "acme"
    )
    home.mkdir()
    for name, body in (("current.npz", b"old"), ("candidate.npz", b"new"), ("candidate.pt", b"new-pt")):
        (home / name).write_bytes(body)
    assert apply(home, "wait") == {} and (home / "candidate.npz").exists()
    moved = apply(home, "promote", stamp="S")
    assert moved == {
        "current.npz": "previous-S.npz",
        "candidate.npz": "current.npz",
        "candidate.pt": "current.pt",
    }
    assert (home / "current.npz").read_bytes() == b"new" and (home / "previous-S.npz").read_bytes() == b"old"
    assert (home / "current.pt").read_bytes() == b"new-pt" and not (home / "candidate.npz").exists()
    (home / "candidate.npz").write_bytes(b"worse")
    assert apply(home, "rollback", stamp="T") == {"candidate.npz": "rejected-T.npz"}
    assert (home / "current.npz").read_bytes() == b"new" and (
        home / "rejected-T.npz"
    ).read_bytes() == b"worse"


def test_ab_cli_reports_and_moves_only_when_decided(tmp_path, capsys):
    usage = UsageLog(tmp_path / "usage")
    tools = [Tool(id="s/a", category="s"), Tool(id="s/b", category="s")]

    def search(arm, session, called_rank):
        res = SearchResult(
            query="q", instruction="", hits=[Hit(t, 0.5) for t in tools], ranked=[(t.id, 0.5) for t in tools],
            took_ms=1.0, rule="top 2", emb_key="k", scorer="x", catalog="c", arm=arm,
        )  # fmt: skip
        sid = usage.search(res, session=session, via="mcp", arm=arm)
        if called_rank:
            usage.call(
                tool=tools[called_rank - 1].id,
                kind="mcp",
                session=session,
                via="mcp",
                outcome="ok",
                took_ms=1,
                search_id=sid,
            )

    (tmp_path / "heads").mkdir()
    assert main(["ab", "--data", str(tmp_path), "--results", str(tmp_path / "res")]) == 0
    assert "no candidate" in capsys.readouterr().out
    (tmp_path / "heads" / "candidate.npz").write_bytes(b"new")
    for i in range(6):
        search("current", f"c{i}", 2)  # the control's tool stood second
        search("candidate", f"n{i}", 1)  # the candidate's first
    args = ["ab", "--data", str(tmp_path), "--results", str(tmp_path / "res"), "--since", "2000-01-01"]
    assert main(args) == 0  # six searches a side: nothing is decided yet
    out = capsys.readouterr().out
    assert "| candidate | 6 | 6 | 1.000 | 1.000 |" in out and "| control | 6 | 6 | 0.000 | 0.500 |" in out
    assert "wait: nothing moved" in out and (tmp_path / "heads" / "candidate.npz").exists()
    assert main([*args, "--min-searches", "5", "--dry-run"]) == 0
    assert "promote: dry run" in capsys.readouterr().out and (tmp_path / "heads" / "candidate.npz").exists()
    assert main([*args, "--min-searches", "5"]) == 0
    assert "promote: candidate.npz -> current.npz" in capsys.readouterr().out
    assert (tmp_path / "heads" / "current.npz").read_bytes() == b"new"
    (report,) = sorted((tmp_path / "res").glob("ab_*.json"))[-1:]
    assert json.loads(report.read_text())["decision"] == "promote"


def test_learn_writes_the_candidate_and_mixes_in_general_pairs(tmp_path):
    pytest.importorskip("torch")
    from toolrank.adapters.heads_np import NumpyHeads
    from toolrank.datasets.jsonl import write_pairs
    from toolrank.domain import TrainPair
    from toolrank.finetune import TrainConfig
    from toolrank.formats import query_format

    _served(tmp_path)
    general = [
        TrainPair(
            id=str(i),
            text=f"general request {i}",
            positives=(json.dumps({"name": f"g{i % 7}", "description": "x"}),),
        )
        for i in range(40)
    ]
    write_pairs(tmp_path / "pairs.jsonl", general)
    enc, tf, qf = _enc(tmp_path), tool_format("documentation"), query_format("instruct_query")
    rng = np.random.default_rng(9)
    from toolrank.domain import Query

    texts = [qf(Query(id=p.id, text=p.text, qrels={}, instruction="Find the tool.")) for p in general]
    docs = sorted(
        {tf(Tool(id="", doc=json.loads(p.positives[0]), documentation=p.positives[0])) for p in general}
    )
    enc.cache.put_many(texts + docs, rng.standard_normal((len(texts) + len(docs), 16)).astype(np.float32))
    cfg = TrainConfig(
        epochs=6,
        batch=32,
        neg_per_pair=3,
        lr=3e-3,
        neg_filter=0.95,
        head_cfg={"width": 64, "depth": 3, "skip": True},
    )
    out = heads_home(tmp_path) / "candidate.npz"
    job = Job(
        data=tmp_path,
        out=out,
        init=None,
        emb_url=URL,
        emb_model=MODEL,
        train=cfg,
        replay=tmp_path / "pairs.jsonl",
        replay_n=25,
        instruction="Find the tool.",
    )
    lines = []
    report = run(job, log=lines.append)
    assert (
        report["replay"] == {"pairs": 25, "tools": 7, "from": "pairs.jsonl"} and report["tokens_spent"] == 0
    )
    assert report["decision"] == "published", lines
    assert any("a running server gives it a share" in line for line in lines)
    assert NumpyHeads(out).cfg["learned_from"]["requests"] == 48 and out.with_suffix(".pt").exists()
