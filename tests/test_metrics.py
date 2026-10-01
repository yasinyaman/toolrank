"""Prometheus metrics (``toolrank.metrics``, ``GET /v1/metrics``): what the usage log sees, counted."""

import hashlib
import time

import numpy as np
import pytest

from toolrank.domain import Tool
from toolrank.metrics import CHARS_PER_TOKEN, Metrics, arm_kind, tool_tokens
from toolrank.retriever import Hit, SearchResult
from toolrank.usage import UsageLog


def _tool(n, schema=None):
    doc = {"name": f"t{n}", "description": f"tool {n}", **({"inputSchema": schema} if schema else {})}
    return Tool(id=f"s/t{n}", doc=doc, category="s")


def _result(tools, catalog="cat", **kw):
    hits = [Hit(t, 0.5) for t in tools]
    return SearchResult("q", "", hits, [(h.id, 0.5) for h in hits], 12.0, "top", None, "x", catalog, **kw)


def _value(text, line_start):
    (line,) = [x for x in text.splitlines() if x.startswith(line_start)]
    return float(line.rsplit(" ", 1)[1])


def test_tool_tokens_and_arm_kinds():
    plain, big = (
        _tool(1),
        _tool(2, {"type": "object", "properties": {"a": {"type": "string", "description": "x" * 400}}}),
    )
    assert tool_tokens(plain) == -(
        -len('{"name": "t1", "description": "tool 1", "input_schema": {}}') // CHARS_PER_TOKEN
    )
    assert tool_tokens(big) > tool_tokens(plain) + 100
    assert [
        arm_kind(a) for a in (None, "base", "current", "candidate", "tenant:acme", "tenant:acme:candidate")
    ] == [
        "base",
        "base",
        "current",
        "candidate",
        "tenant",
        "tenant-candidate",
    ]


def test_searches_and_calls_are_counted_and_rendered_as_prometheus_text():
    catalogue = [_tool(n) for n in range(10)]
    m = Metrics(catalogue=lambda: (catalogue, "cat"))
    assert m.catalog_tokens() == 10 * tool_tokens(catalogue[0])  # sized once, before the searches
    m.search(_result(catalogue[:2]), via="mcp", arm="tenant:acme:candidate")
    m.search(_result(catalogue[:3], added=1), via="mcp", arm="tenant:acme:candidate")
    m.search(_result([], mode="lexical"), via="rest")
    m.call(kind="mcp", outcome="ok", via="mcp", took_ms=40.0, link="search_id", rank=2)
    m.call(kind=None, outcome="unknown_tool", via="rest", took_ms=1.0, link="none", rank=None)
    text = m.render([("toolrank_catalog_tools", "gauge", "Tools.", {}, 10)])
    assert text.endswith("\n") and "acme" not in text  # a key's name never reaches a label
    assert 'toolrank_searches_total{arm="tenant-candidate",mode="semantic",via="mcp"} 2' in text
    assert 'toolrank_searches_total{arm="base",mode="lexical",via="rest"} 1' in text
    assert "# TYPE toolrank_search_duration_seconds histogram" in text
    assert 'toolrank_search_duration_seconds_bucket{le="0.01"} 0' in text
    assert 'toolrank_search_duration_seconds_bucket{le="0.025"} 3' in text  # cumulative
    assert 'toolrank_search_duration_seconds_bucket{le="+Inf"} 3' in text
    assert _value(text, "toolrank_search_duration_seconds_sum") == pytest.approx(0.036)
    assert (
        'toolrank_search_tools_returned_bucket{le="2"} 2' in text
        and "toolrank_search_tools_returned_sum 5" in text
    )
    assert "toolrank_search_empty_total 1" in text and "toolrank_search_co_use_added_total 1" in text
    one = tool_tokens(catalogue[0])
    assert _value(text, "toolrank_search_returned_tokens_total") == 5 * one
    assert _value(text, "toolrank_search_saved_tokens_total") == (10 - 2) * one + (10 - 3) * one + 10 * one
    assert 'toolrank_calls_total{kind="mcp",outcome="ok",via="mcp"} 1' in text
    assert 'toolrank_calls_total{kind="unknown",outcome="unknown_tool",via="rest"} 1' in text
    assert 'toolrank_calls_linked_total{link="none"} 1' in text
    assert (
        'toolrank_called_tool_rank_bucket{le="1"} 0' in text
        and 'toolrank_called_tool_rank_bucket{le="2"} 1' in text
    )
    assert text.rstrip().endswith(
        "# HELP toolrank_catalog_tools Tools.\n# TYPE toolrank_catalog_tools gauge\ntoolrank_catalog_tools 10"
    )

    fresh = Metrics().render()
    assert (
        "toolrank_searches_total 0" in fresh and "_bucket" not in fresh
    )  # counters start at 0, histograms empty
    assert Metrics().catalog_tokens() is None
    odd = Metrics()
    odd.inc("toolrank_calls_linked_total", link='a"b\\c\nd')
    assert 'link="a\\"b\\\\c\\nd"' in odd.render()


def test_the_catalogue_estimate_follows_the_catalogue():
    state = {"tools": [_tool(1), _tool(2)], "hash": "one"}
    m = Metrics(catalogue=lambda: (state["tools"], state["hash"]))
    assert m.catalog_tokens() == 2 * tool_tokens(_tool(1))
    state.update(tools=[_tool(1)], hash="two")
    assert m.catalog_tokens("two") == tool_tokens(_tool(1)) == m.catalog_tokens()
    assert m.catalog_tokens("one") is None  # a search of the catalogue that was: no saving is claimed

    def gone():
        raise RuntimeError("index not ready")

    assert Metrics(catalogue=gone).catalog_tokens() is None

    # a search never waits for the sizing: the first one after a change claims nothing and starts it
    lazy = Metrics(catalogue=lambda: (state["tools"], state["hash"]))
    lazy.search(_result([_tool(1)], catalog="two"), via="mcp")
    assert "toolrank_search_saved_tokens_total 0" in lazy.render()
    for _ in range(200):
        if lazy.catalog_tokens("two", wait=False) is not None:
            break
        time.sleep(0.01)
    lazy.search(_result([], catalog="two"), via="mcp")
    assert f"toolrank_search_saved_tokens_total {tool_tokens(_tool(1))}" in lazy.render()


def test_the_usage_log_counts_what_it_logs_even_with_the_files_off(tmp_path):
    for directory in (tmp_path / "usage", None):
        log = UsageLog(directory)
        res = _result([_tool(1), _tool(2)])
        sid = log.search(res, session="s", via="mcp", arm="candidate")
        log.call(tool="s/t2", kind="mcp", session="s", via="mcp", outcome="ok", took_ms=5, search_id=sid)
        log.call(tool="s/zz", kind=None, session="s", via="mcp", outcome="unknown_tool", took_ms=1)
        text = log.metrics.render()
        assert 'toolrank_searches_total{arm="candidate",mode="semantic",via="mcp"} 1' in text
        assert 'toolrank_calls_linked_total{link="search_id"} 1' in text
        assert (
            'toolrank_called_tool_rank_bucket{le="2"} 1' in text
            and "toolrank_called_tool_rank_count 1" in text
        )
    broken = UsageLog(None)
    broken.metrics.search = None  # a counter that fails must not fail the search
    assert broken.search(_result([_tool(1)]), session="s", via="mcp").startswith("s-")


def test_the_encoder_counts_cache_hits_and_endpoint_texts(tmp_path, monkeypatch):
    from toolrank.adapters.embeddings_api import OpenAIEmbeddings

    def fake_post(self, texts, **kw):
        rows = [
            np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).standard_normal(8)
            for t in texts
        ]
        return [r.astype(np.float32) for r in rows], 3 * len(texts)

    monkeypatch.setattr(OpenAIEmbeddings, "_post", fake_post)
    enc = OpenAIEmbeddings("m", "http://unused/v1", cache_dir=tmp_path)
    enc.encode(["a", "b", "c"])
    enc.encode(["a", "d"], kind="query")
    enc.encode(["d"], kind="query")
    assert enc.texts == {
        ("document", "cache"): 0,
        ("document", "endpoint"): 3,
        ("query", "cache"): 2,
        ("query", "endpoint"): 1,
    }
    assert enc.tokens_spent == 12
