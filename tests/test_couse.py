"""Co-use: partners counted from the usage log and appended to result lists (``serve --co-use``)."""

import hashlib
import json
import time

import numpy as np

from toolrank.adapters.dense import DenseScorer
from toolrank.couse import CoUseTable, co_use, expand, partners
from toolrank.cut import AdaptiveK
from toolrank.datasets.jsonl import write_tools
from toolrank.domain import Tool
from toolrank.learn import mine
from toolrank.retriever import Hit, Retriever, SearchResult
from toolrank.usage import UsageLog, read_events


def _search(sid, request, ts="2026-10-01T10:00:00", tenant=None):
    return {"v": 3, "event": "search", "id": sid, "ts": ts, "tenant": tenant, "emb_hmac": request}


def _call(tool, sid, outcome="ok"):
    return {"v": 3, "event": "call", "tool": tool, "search_id": sid, "outcome": outcome}


def _events():
    return [
        _search("s1", "A"),
        _call("list", "s1"),
        _call("comment", "s1"),
        _search("s2", "B", tenant="team"),
        _call("list", "s2"),
        _call("comment", "s2"),
        _call("close", "s2", "tool_error"),  # a failed call is no evidence of use
        _search("s3", "C", ts="2026-10-02T10:00:00"),
        _call("list", "s3"),
        _search("s4", "C", ts="2026-10-02T11:00:00"),  # the same request again: one request
        _call("list", "s4"),
        _call("label", "s4"),
        _search("s5", None),  # a keyword answer has no emb_hmac: it counts alone
        _call("list", "s5"),
        _call("label", "s5"),
        _call("list", "gone"),  # no such search
    ]


def test_co_use_counts_requests_whose_ok_calls_share_a_search():
    table = co_use(
        _events(), min_count=2, min_p=0.5
    )  # list was called in 4 requests (A, B, C, s5), comment in 2 of them, label in 2
    assert table == {
        "list": [("comment", 0.5, 2), ("label", 0.5, 2)],
        "comment": [("list", 1.0, 2)],
        "label": [("list", 1.0, 2)],
    }
    assert co_use(_events(), min_count=3) == {}
    assert "list" not in co_use(_events(), min_p=0.6)  # 2 of list's 4 requests is too small a share
    assert co_use(_events(), min_count=1, since="2026-10-02") == {
        "list": [("label", 1.0, 1)],
        "label": [("list", 1.0, 1)],
    }
    assert co_use(_events(), min_count=1, tenant="team") == {
        "list": [("comment", 1.0, 1)],
        "comment": [("list", 1.0, 1)],
    }


def test_partners_follow_the_shown_tools_best_share_first():
    table = {
        "a": [("x", 0.9, 9), ("y", 0.6, 6)],
        "b": [("y", 0.8, 8), ("a", 1.0, 5), ("z", 0.6, 3)],
    }
    assert partners(["a", "b"], table, 5) == [("x", "a"), ("y", "b"), ("z", "b")]  # y with its best share
    assert expand(["a", "b"], table, 2) == ["a", "b", "x", "y"]
    assert expand(["b"], table, 1) == ["b", "a"] and expand(["a"], table, 0) == ["a"]
    assert expand(["q"], table, 3) == ["q"] and partners([], table, 3) == []


def test_searches_a_jev_second_stage_answered_shape_no_table():
    from collections import Counter

    jev = "jev[jev-1.13.0@api.typesafe.ai,d20,name_desc]/dense/emb/x"
    events = [
        {**_search("s1", "A"), "scorer": jev},
        _call("list", "s1"),
        _call("comment", "s1"),
        {**_search("s2", "B"), "scorer": jev},
        _call("list", "s2"),
        _call("comment", "s2"),
    ]
    counts: Counter[str] = Counter()
    assert co_use(events, min_count=1, counts=counts) == {}  # the provider's terms: no evidence
    assert counts["searches_with_jev"] == 2
    events.append(_search("s3", "C"))
    events += [_call("list", "s3"), _call("comment", "s3")]
    counts = Counter()
    assert co_use(events, min_count=1, counts=counts) == {  # one clean request still counts
        "list": [("comment", 1.0, 1)],
        "comment": [("list", 1.0, 1)],
    }
    assert counts["searches_with_jev"] == 2


def test_the_servers_table_is_rebuilt_from_its_log_in_the_background(tmp_path):
    assert CoUseTable(tmp_path / "missing").table() == {}
    log = tmp_path / "usage"
    log.mkdir()
    old, new = _events()[:6], _events()[6:]
    (log / "usage-2026-10-01.jsonl").write_text("\n".join(json.dumps(e) for e in old) + "\nnot json\n")
    table = CoUseTable(log, min_count=1, every=0.0)
    # each key counts only its own requests: one request each here, never two together
    assert (
        table.table() == table.table("team") == {"list": [("comment", 1.0, 1)], "comment": [("list", 1.0, 1)]}
    )
    assert table.table("other") == {} and CoUseTable(log, every=1e9).table() == {}  # min_count 2: not yet
    (log / "usage-2026-10-02.jsonl").write_text("\n".join(json.dumps(e) for e in new) + "\n")
    for _ in range(200):  # every 0 s: each look starts a rebuild, the next one sees it
        if "label" in table.table():
            break
        time.sleep(0.01)
    assert table.table()["label"] == [("list", 1.0, 2)] and "label" not in table.table("team")
    assert table.table()["list"] == [("label", 2 / 3, 2)]  # comment: 1 of the 3 requests without a key
    assert len(read_events(log, newest=1)) == len(new) and len(read_events(log)) == len(_events())
    # the newest day alone never saw list with comment
    assert CoUseTable(log, days=1).table() == {"list": [("label", 1.0, 2)], "label": [("list", 1.0, 2)]}


class _HashEncoder:
    name = "hash"

    def encode(self, texts, *, kind="document"):
        rows = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).standard_normal(16))
        return np.asarray(rows, dtype=np.float32)


class _Table:
    def __init__(self, table):
        self._table = table

    def table(self, tenant=None):
        return self._table


def test_search_appends_partners_and_the_log_keeps_them_apart(tmp_path):
    tools = [
        Tool(id=f"s/t{i}", doc={"name": f"t{i}", "description": f"tool {i}"}, category="s") for i in range(12)
    ]
    write_tools(tmp_path / "tools.jsonl", tools)

    def make():
        return DenseScorer(_HashEncoder(), "name_desc", "instruct_query")

    plain = Retriever(tmp_path, make, rule=AdaptiveK(margin=10.0, max_k=3), cache_key=lambda t: "k:" + t)
    base = plain.search("tool 3")
    top, rest = [h.id for h in base.hits], [t for t, _ in base.ranked if t not in {h.id for h in base.hits}]
    far = rest[-1]  # a tool the ranking would not have shown
    table = _Table(
        {top[0]: [(far, 0.9, 4), (top[1], 0.8, 3), ("s/left", 0.7, 2)], top[2]: [(rest[0], 0.6, 2)]}
    )
    r = Retriever(
        tmp_path,
        make,
        rule=AdaptiveK(margin=10.0, max_k=3),
        cache_key=lambda t: "k:" + t,
        co_use=table,
        co_use_extra=2,
    )
    res = r.search("tool 3")
    assert [h.id for h in res.hits] == [
        *top,
        far,
        rest[0],
    ] and res.added == 2  # a shown or vanished tool is no partner
    assert [h.used_with for h in res.hits] == [None, None, None, top[0], top[2]]
    assert res.hits[3].score == dict(base.ranked)[far]  # its own cosine, though it was not ranked in
    assert r.search("tool 3", k=1).added == 0 and len(r.search("tool 3", k=1).hits) == 1  # it asked for one
    off = Retriever(tmp_path, make, rule=AdaptiveK(margin=10.0, max_k=3), co_use=table, co_use_extra=0)
    assert [h.id for h in off.search("tool 3").hits] == top

    log = UsageLog(tmp_path / "usage")
    sid = log.search(res, session="x", via="mcp")
    log.call(tool=far, kind="mcp", session="x", via="mcp", outcome="ok", took_ms=1, search_id=sid)
    search, call = read_events(tmp_path / "usage")
    assert search["shown"] == 3 and search["added"] == [far, rest[0]]
    assert call["rank"] == 4 and call["link"] == "search_id"
    log.search(base, session="y", via="mcp")
    assert "added" not in read_events(tmp_path / "usage")[-1]
    (pair,), _ = mine(read_events(tmp_path / "usage"))
    assert pair.positives == (far,) and set(pair.negatives) == {*top, rest[0]}  # shown, by rank or by co-use


def test_hits_say_which_tool_they_are_used_with():
    from toolrank.adapters.mcp_proxy import hit_json

    tool = Tool(id="s/t", doc={"name": "t", "description": "d"}, category="s")
    assert hit_json(Hit(tool, 0.3, used_with="s/other"), full=True)["used_with"] == "s/other"
    assert "used_with" not in hit_json(Hit(tool, 0.3), full=True)
    result = SearchResult("q", "", [Hit(tool, 0.3)], [("s/t", 0.3)], 1.0, "top 1", None, "x", "c")
    assert result.added == 0
