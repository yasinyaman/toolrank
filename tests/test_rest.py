import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcp")

import httpx2  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from toolrank.adapters.backends import Backends  # noqa: E402
from toolrank.adapters.bm25 import BM25Scorer  # noqa: E402
from toolrank.adapters.dense import DenseScorer  # noqa: E402
from toolrank.adapters.mcp_proxy import MAX_BODY, build_proxy, http_app  # noqa: E402
from toolrank.adapters.rest import MAX_RANK, platform_record, rest_routes  # noqa: E402
from toolrank.datasets.jsonl import write_tools  # noqa: E402
from toolrank.domain import Tool  # noqa: E402
from toolrank.ingest.mcp import ServerConfig  # noqa: E402
from toolrank.retriever import Retriever  # noqa: E402
from toolrank.usage import UsageLog  # noqa: E402

BASE = "http://127.0.0.1:8765"
FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_server.py")
ADD_SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}


class _HashEncoder:
    name = "hash"
    down = False

    def encode(self, texts, *, kind="document"):
        if self.down and kind == "query":
            raise RuntimeError("endpoint down")
        return np.asarray(
            [
                np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).standard_normal(16)
                for t in texts
            ],
            dtype=np.float32,
        )


def _tools():
    http = {"method": "GET", "path": "/things/{id}", "base_url": "https://api.example.com", "args": {}}
    return [
        Tool(
            id="fx/add",
            doc={
                "server": "fx",
                "name": "add",
                "description": "Add two integers.",
                "inputSchema": ADD_SCHEMA,
            },
            category="fx",
        ),
        Tool(
            id="fx/search_issues",
            doc={"server": "fx", "name": "search_issues", "description": "Search the issue tracker.\nMore."},
            category="fx",
        ),
        Tool(
            id="api/getThing",
            doc={"server": "api", "name": "getThing", "description": "Get a thing.", "http": http},
            category="api",
        ),
    ]


def _app(tmp_path, retriever=None, *, api_key=None, named_keys=None):
    write_tools(tmp_path / "tools.jsonl", _tools())
    retriever = retriever or Retriever(tmp_path, lambda: DenseScorer(_HashEncoder(), "name_desc"))
    usage = UsageLog(tmp_path / "usage")
    server = build_proxy(retriever, Backends(), usage)
    app = http_app(server, api_key=api_key, named_keys=named_keys, routes=rest_routes(retriever, usage))
    return app, retriever


def _events(tmp_path):
    return [
        json.loads(line)
        for f in sorted((tmp_path / "usage").glob("usage-*.jsonl"))
        for line in f.read_text().splitlines()
    ]


def test_search_rank_and_catalogue_routes(tmp_path):
    app, retriever = _app(tmp_path)
    with TestClient(app, base_url=BASE) as c:
        r = c.post("/v1/search", json={"query": "add two integers"}, headers={"X-Session-Id": "abc"})
        found = r.json()
        assert r.status_code == 200 and found["search_id"].startswith("s-") and found["rule"] == "top 10"
        assert sorted(t["name"] for t in found["tools"]) == ["api/getThing", "fx/add", "fx/search_issues"]
        assert next(t for t in found["tools"] if t["name"] == "fx/add")["inputSchema"] == ADD_SCHEMA
        assert {(t["name"], t["api_name"], t["kind"]) for t in found["tools"]} == {
            ("fx/add", "fx__add", "mcp"),
            ("fx/search_issues", "fx__search_issues", "mcp"),
            ("api/getThing", "api__getThing", "openapi"),
        }
        assert len(c.post("/v1/search", json={"query": "add", "k": 1}).json()["tools"]) == 1

        given = [
            {"name": "search", "server": "web", "description": "Search the web."},
            {"name": "search", "server": "files", "description": "Search files.", "inputSchema": {}},
            {"name": "add", "description": "Add numbers."},
        ]
        ranked = c.post("/v1/rank", json={"query": "find a file", "tools": given}).json()["tools"]
        assert [t["score"] for t in ranked] == sorted((t["score"] for t in ranked), reverse=True)
        assert {(t["index"], t["name"], t.get("server")) for t in ranked} == {
            (0, "search", "web"),
            (1, "search", "files"),
            (2, "add", None),
        }
        by_id = c.post("/v1/rank", json={"query": "add", "tool_ids": ["fx/add", "api/getThing"]}).json()
        alone = retriever.rank("add", [retriever.get("fx/add")])[0][1]
        assert next(t for t in by_id["tools"] if t["name"] == "fx/add")["score"] == pytest.approx(
            alone, abs=1e-5
        )

        listed = c.get("/v1/tools").json()
        assert listed["count"] == 3 and c.get("/v1/tools", params={"server": "fx"}).json()["count"] == 2
        issues = next(t for t in listed["tools"] if t["name"] == "fx/search_issues")
        assert (issues["kind"], issues["description"]) == ("mcp", "Search the issue tracker.")
        assert (
            listed["catalog"] == retriever.status()["catalog"] and issues["api_name"] == "fx__search_issues"
        )
        full = {t["name"]: t for t in c.get("/v1/tools", params={"full": "true"}).json()["tools"]}
        assert full["fx/add"]["inputSchema"] == ADD_SCHEMA and full["fx/add"]["api_name"] == "fx__add"
        assert full["fx/search_issues"]["inputSchema"] == {"type": "object"}  # none given
        assert full["fx/search_issues"]["description"] == "Search the issue tracker.\nMore."
        assert full["api/getThing"]["method"] == "GET" and "method" not in full["fx/add"]
        thing = c.get("/v1/tools/api/getThing").json()
        assert (thing["kind"], thing["http"]["method"], thing["api_name"]) == (
            "openapi",
            "GET",
            "api__getThing",
        )
        assert c.get("/v1/tools/fx/nope").status_code == 404
        assert c.get("/healthz").json() == {"ready": True, "mode": "semantic"}  # no token: nothing more
        spec = c.get("/openapi.json").json()
        assert spec["openapi"] == "3.1.0" and {"/v1/search", "/v1/rank", "/v1/tools"} <= set(spec["paths"])
    assert [(e["event"], e["via"], e["session"]) for e in _events(tmp_path)] == [
        ("search", "rest", "rest:abc"),
        ("search", "rest", None),
    ]


def test_token_host_bad_input_and_body_limit(tmp_path):
    app, _ = _app(tmp_path, api_key="k")
    auth = {"Authorization": "Bearer k"}
    with TestClient(app, base_url=BASE) as c:
        assert c.post("/v1/search", json={"query": "x"}).status_code == 401
        assert c.get("/v1/tools", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/tools", headers={**auth, "Host": "evil.example.com"}).status_code == 421
        page = {**auth, "Origin": "https://evil.example.com", "Content-Type": "text/plain"}
        assert c.post("/v1/search", content=b'{"query": "x"}', headers=page).status_code == 403
        assert c.get("/v1/tools", headers={**auth, "Origin": "http://localhost:3000"}).status_code == 200
        assert c.get("/healthz").status_code == 200 and c.get("/openapi.json").status_code == 200
        cases = [
            ("/v1/search", {"query": " "}, 400, "query is required"),
            ("/v1/search", {"query": 5}, 400, "query must be a string"),
            ("/v1/search", {"query": "x", "k": 0}, 400, "k must be"),
            ("/v1/search", {"query": "x", "k": True}, 400, "k must be"),
            ("/v1/search", {"query": "x", "instruction": ["a"]}, 400, "instruction must be"),
            ("/v1/rank", {"query": "x"}, 400, "either tools"),
            ("/v1/rank", {"query": "x", "tools": [{"name": "a"}], "tool_ids": ["fx/add"]}, 400, "either"),
            ("/v1/rank", {"query": "x", "tools": []}, 400, "a list of 1 to"),
            ("/v1/rank", {"query": "x", "tools": [{"name": "a"}] * (MAX_RANK + 1)}, 400, "a list of 1 to"),
            (
                "/v1/rank",
                {"query": "x", "tools": [{"description": "d"}]},
                400,
                "tools[0] needs a string name",
            ),
            ("/v1/rank", {"query": "x", "tools": [{"name": "a", "inputSchema": "{}"}]}, 400, "an object"),
            ("/v1/rank", {"query": "x", "tool_ids": ["fx/add", "fx/nope"]}, 404, "fx/nope"),
        ]
        for path, body, status, message in cases:
            r = c.post(path, json=body, headers=auth)
            assert (r.status_code, message in r.json()["error"]) == (status, True), body
        not_object = c.post("/v1/search", content=b"[1]", headers=auth)
        assert (not_object.status_code, not_object.json()["error"]) == (400, "the body must be a JSON object")
        assert c.post("/v1/search", content=b"{nope", headers=auth).json()["error"] == "the body is not JSON"
        big = b'{"query": "' + b"x" * MAX_BODY + b'"}'
        assert c.post("/v1/search", content=big, headers=auth).status_code == 413  # Content-Length

        def chunked():  # no Content-Length: the route counts what it reads
            yield big[:1000]
            yield big[1000:]

        assert c.post("/v1/search", content=chunked(), headers=auth).status_code == 413


def test_first_index_and_endpoint_failures_are_503(tmp_path):
    gate, encoder = threading.Event(), _HashEncoder()

    def make():
        gate.wait(10)
        return DenseScorer(encoder, "name_desc")

    write_tools(tmp_path / "tools.jsonl", _tools())
    retriever = Retriever(tmp_path, make, background=True, ready_timeout=0.1)
    app, _ = _app(tmp_path, retriever)
    with TestClient(app, base_url=BASE) as c:
        assert c.get("/healthz").status_code == 503
        r = c.post("/v1/search", json={"query": "x"})
        assert r.status_code == 503 and "still being built" in r.json()["error"]
        gate.set()
        for _ in range(100):
            if retriever.status()["ready"]:
                break
            time.sleep(0.05)
        assert c.get("/healthz").status_code == 200
        encoder.down = True
        r = c.post("/v1/search", json={"query": "x"})
        assert (
            r.status_code == 503
            and r.json()["error"] == "search failed (RuntimeError); the server's log says why"
        )


def test_keyword_matches_while_the_index_builds_and_named_keys(tmp_path):
    gate = threading.Event()

    def make():
        gate.wait(10)
        return DenseScorer(_HashEncoder(), "name_desc")

    write_tools(tmp_path / "tools.jsonl", _tools())
    retriever = Retriever(
        tmp_path, make, background=True, ready_timeout=5, fallback=lambda: BM25Scorer(stem=False)
    )
    app, _ = _app(tmp_path, retriever, named_keys={"team-a": "ka", "team-b": "kb"})
    team_a, team_b = {"Authorization": "Bearer ka"}, {"Authorization": "Bearer kb"}
    with TestClient(app, base_url=BASE) as c:
        r = c.post("/v1/search", json={"query": "search the issue tracker"}, headers=team_a)
        assert (r.status_code, r.json()["mode"], r.json()["tools"][0]["name"]) == (
            200,
            "lexical",
            "fx/search_issues",
        )
        health = c.get("/healthz")
        assert health.status_code == 503 and health.json() == {"ready": False, "mode": "lexical"}
        assert (
            c.post("/v1/rank", json={"query": "x", "tool_ids": ["fx/add"]}, headers=team_a).status_code == 503
        )
        assert c.get("/v1/tools/fx/add", headers=team_a).status_code == 200  # the catalogue is there already
        gate.set()
        assert retriever.wait_ready(5)
        r = c.post("/v1/search", json={"query": "search the issue tracker"}, headers=team_b)
        assert r.json()["mode"] == "semantic" and c.get("/healthz").status_code == 200
        assert (
            c.post("/v1/search", json={"query": "x"}, headers={"Authorization": "Bearer kc"}).status_code
            == 401
        )
    events = _events(tmp_path)
    assert [(e["tenant"], e["scorer"].split("/")[0]) for e in events] == [
        ("team-a", "bm25"),
        ("team-b", "dense"),
    ]
    assert events[0]["client"] != events[1]["client"]  # the key name is part of the client key


def _callable(tmp_path):
    """A catalogue with an MCP tool (the stdio fixture), an OpenAPI GET and an OpenAPI POST."""
    api = {"base_url": "https://api.example.com", "body_media_type": "application/json"}
    tools = [
        Tool(
            id="fx/add",
            doc={
                "server": "fx",
                "name": "add",
                "description": "Add two integers.",
                "inputSchema": ADD_SCHEMA,
            },
            category="fx",
        ),
        Tool(
            id="api/getThing",
            doc={
                "server": "api",
                "name": "getThing",
                "description": "Get a thing by id.",
                "http": {
                    **api,
                    "method": "GET",
                    "path": "/things/{id}",
                    "args": {"id": {"in": "path", "name": "id"}},
                },
            },
            category="api",
        ),
        Tool(
            id="api/createThing",
            doc={
                "server": "api",
                "name": "createThing",
                "description": "Create a thing.",
                "http": {
                    **api,
                    "method": "POST",
                    "path": "/things",
                    "args": {"name": {"in": "body", "name": "name"}},
                },
            },
            category="api",
        ),
    ]
    write_tools(tmp_path / "tools.jsonl", tools)
    retriever = Retriever(tmp_path, lambda: DenseScorer(_HashEncoder(), "name_desc"))
    usage = UsageLog(tmp_path / "usage")
    thing = httpx2.MockTransport(
        lambda req: httpx2.Response(200, json={"id": req.url.path.rsplit("/", 1)[-1]})
    )
    fx = ServerConfig("fx", "stdio", command=sys.executable, args=(FIXTURE,))
    return retriever, usage, Backends([fx], transport=thing, call_timeout=20)


def test_calls_over_rest_run_like_mcp_call_tool(tmp_path):
    retriever, usage, backends = _callable(tmp_path)
    app = http_app(build_proxy(retriever, backends, usage), routes=rest_routes(retriever, usage, backends))
    with TestClient(app, base_url=BASE, headers={"X-Session-Id": "t1"}) as c:
        sid = c.post("/v1/search", json={"query": "add two integers"}).json()["search_id"]
        added = c.post("/v1/call", json={"name": "fx/add", "arguments": {"a": 2, "b": 3}, "search_id": sid})
        body = added.json()
        assert added.status_code == 200 and (body["outcome"], body["isError"]) == ("ok", False)
        assert body["content"] == [{"type": "text", "text": "5"}] and body["call_id"].startswith("c-")
        got = c.post("/v1/call", json={"name": "api/getThing", "arguments": {"id": 7}}).json()
        assert (got["outcome"], got["http_status"]) == ("ok", 200) and '"7"' in got["content"][0]["text"]
        refused = c.post("/v1/call", json={"name": "api/createThing", "arguments": {"name": "x"}}).json()
        assert (refused["outcome"], refused["isError"]) == ("refused", True)
        assert "--allow-write" in refused["content"][0]["text"]
        bad = c.post("/v1/call", json={"name": "fx/add", "arguments": {"a": "x"}}).json()
        assert bad["isError"] and bad["content"][-1]["text"].startswith("inputSchema of fx/add")
        assert c.post("/v1/call", json={"name": "fx/nope"}).status_code == 404
        assert c.post("/v1/call", json={"name": "fx/" + "n" * 5000}).status_code == 404
        assert c.post("/v1/call", json={"name": "fx/add", "arguments": [1]}).status_code == 400
        form = c.post("/v1/call", content=b'{"name": "fx/add"}', headers={"Content-Type": "text/plain"})
        assert form.status_code == 415  # what a web page can send without a preflight
    calls = [e for e in _events(tmp_path) if e["event"] == "call"]
    assert [(e["tool"], e["via"], e["outcome"]) for e in calls[:4]] == [
        ("fx/add", "rest", "ok"),
        ("api/getThing", "rest", "ok"),
        ("api/createThing", "rest", "refused"),
        ("fx/add", "rest", "tool_error"),
    ]
    assert (calls[0]["search_id"], calls[0]["link"], calls[0]["session"]) == (sid, "search_id", "rest:t1")
    unknown = [e["tool"] for e in calls if e["outcome"] == "unknown_tool"]
    assert unknown == ["fx/nope", "fx/" + "n" * 197]  # what the agent typed, cut at 200 characters


def test_calls_need_running_backends_and_the_route_needs_them(tmp_path):
    retriever, usage, backends = _callable(tmp_path)
    bare = Starlette(routes=rest_routes(retriever, usage, backends))  # no MCP lifespan: backends idle
    with TestClient(bare, base_url=BASE) as c:
        r = c.post("/v1/call", json={"name": "fx/add", "arguments": {"a": 1, "b": 2}})
        assert r.status_code == 503 and "running" in r.json()["error"]
    without = Starlette(routes=rest_routes(retriever, usage))
    with TestClient(without, base_url=BASE) as c:
        assert c.post("/v1/call", json={"name": "fx/add"}).status_code == 404


def test_platform_records_are_what_the_agent_apis_accept():
    schema = {"$schema": "http://json-schema.org/draft-07/schema#", "properties": {"q": {"type": "string"}}}
    tool = Tool(
        id="docs/find pages",
        doc={
            "name": "find pages",
            "description": "d" * 2000,
            "inputSchema": schema,
            "annotations": {"readOnlyHint": True},
        },
        category="docs",
    )
    rec = platform_record(tool)
    assert rec["inputSchema"] == {"type": "object", "properties": {"q": {"type": "string"}}}
    assert len(rec["description"]) == 1501 and rec["annotations"] == {"readOnlyHint": True}
    assert rec["api_name"].startswith("docs__find_pages___") and "$schema" in schema  # the tool is untouched


def test_a_container_bind_answers_its_names_with_or_without_a_port(tmp_path):
    write_tools(tmp_path / "tools.jsonl", _tools())
    retriever = Retriever(tmp_path, lambda: DenseScorer(_HashEncoder(), "name_desc"))
    usage = UsageLog(tmp_path / "usage")
    proxy = build_proxy(retriever, Backends(), usage)
    # docker: bound to 0.0.0.0, reached as the compose service name or through a proxy on 443
    app = http_app(
        proxy, host="0.0.0.0", extra_hosts=["toolrank"], api_key="k", routes=rest_routes(retriever, usage)
    )
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "t", "version": "0"},
        },
    }
    auth = {"Authorization": "Bearer k", "Accept": "application/json, text/event-stream"}
    with TestClient(app, base_url="http://toolrank:8765") as c:
        # the compose service name, a proxy without a port, and the published port on the host
        for host in ("toolrank:8765", "toolrank", "localhost:8765", "127.0.0.1:18765"):
            assert c.get("/v1/tools", headers={**auth, "Host": host}).status_code == 200, host
            assert c.post("/mcp", json=init, headers={**auth, "Host": host}).status_code == 200, host
        assert c.get("/v1/tools", headers={**auth, "Host": "evil.example.com"}).status_code == 421
        assert c.post("/mcp", json=init, headers={**auth, "Host": "evil.example.com"}).status_code == 421


def test_metrics_route_counts_searches_and_sits_behind_the_token(tmp_path):
    from toolrank.metrics import tool_tokens

    app, retriever = _app(tmp_path, api_key="sekrit", named_keys={"acme": "acme-key"})
    auth = {"Authorization": "Bearer sekrit"}
    with TestClient(app, base_url=BASE) as c:
        assert c.get("/v1/metrics").status_code == 401
        first = c.get("/v1/metrics", headers=auth)
        assert first.status_code == 200 and first.headers["content-type"].startswith(
            "text/plain; version=0.0.4"
        )
        assert "toolrank_searches_total 0" in first.text and "toolrank_index_ready 1" in first.text
        assert "toolrank_catalog_tools 3" in first.text and "toolrank_catalog_sources 2" in first.text
        whole = sum(tool_tokens(t) for t in _tools())
        assert f"toolrank_catalog_tokens {whole}" in first.text
        assert c.post("/v1/search", json={"query": "add", "k": 1}, headers=auth).status_code == 200
        assert (
            c.post(
                "/v1/search", json={"query": "issues"}, headers={"Authorization": "Bearer acme-key"}
            ).status_code
            == 200
        )
        text = c.get("/v1/metrics", headers=auth).text
        status = retriever.status()  # the same server, had it been following a heads directory
        variants = {"sha": "x", "ready": True, "error": None}
        heads = {"base": "abc", "candidate": variants, "tenant:acme:candidate": {**variants, "ready": False}}
        retriever.status = lambda: {**status, "heads": heads}
        followed = c.get("/v1/metrics", headers=auth).text
    assert 'toolrank_searches_total{arm="base",mode="semantic",via="rest"} 2' in text
    assert "toolrank_search_tools_returned_sum 4" in text and "acme" not in text
    returned = float(
        next(x for x in text.splitlines() if x.startswith("toolrank_search_returned_tokens_total")).split()[1]
    )
    saved = float(
        next(x for x in text.splitlines() if x.startswith("toolrank_search_saved_tokens_total")).split()[1]
    )
    assert returned + saved == 2 * whole and 0 < returned <= whole + max(tool_tokens(t) for t in _tools())
    assert f'toolrank_build_info{{version="{__import__("toolrank").__version__}"}} 1' in text
    assert "toolrank_heads" not in text  # this server follows no heads directory
    assert 'toolrank_heads{arm="base"} 1' in followed and 'toolrank_heads{arm="candidate"} 1' in followed
    assert 'toolrank_heads{arm="tenant-candidate"} 0' in followed and "acme" not in followed


def test_a_key_limited_to_some_sources_sees_only_those_over_rest(tmp_path):
    write_tools(tmp_path / "tools.jsonl", _tools())
    retriever = Retriever(
        tmp_path, lambda: DenseScorer(_HashEncoder(), "name_desc"), allowed={"team": frozenset({"api"})}
    )
    app, _ = _app(tmp_path, retriever, named_keys={"team": "team-key", "ops": "ops-key"})
    team, ops = {"Authorization": "Bearer team-key"}, {"Authorization": "Bearer ops-key"}
    with TestClient(app, base_url=BASE) as c:
        found = c.post("/v1/search", json={"query": "add two integers"}, headers=team).json()
        assert [t["name"] for t in found["tools"]] == ["api/getThing"]
        assert len(c.post("/v1/search", json={"query": "add two integers"}, headers=ops).json()["tools"]) == 3
        assert [t["name"] for t in c.get("/v1/tools", headers=team).json()["tools"]] == ["api/getThing"]
        assert c.get("/v1/tools", headers=ops).json()["count"] == 3
        hidden = c.get("/v1/tools/fx/add", headers=team)
        assert hidden.status_code == 404 and hidden.json() == c.get("/v1/tools/fx/nope", headers=team).json()
        assert c.get("/v1/tools/fx/add", headers=ops).status_code == 200
        ranked = c.post("/v1/rank", json={"query": "add", "tool_ids": ["fx/add"]}, headers=team)
        assert ranked.status_code == 404
        assert c.get("/v1/metrics", headers=team).status_code == 403
        assert c.get("/v1/metrics", headers=ops).status_code == 200
