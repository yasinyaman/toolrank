import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
import httpx2  # noqa: E402

from toolrank.adapters.backends import Backends  # noqa: E402
from toolrank.adapters.dense import DenseScorer  # noqa: E402
from toolrank.adapters.mcp_proxy import (  # noqa: E402
    SHRUNK_CHARS,
    TENANT_KEY,
    Guard,
    allowed_hosts,
    build_proxy,
    hit_json,
    identity,
)
from toolrank.datasets.jsonl import write_tools  # noqa: E402
from toolrank.domain import Tool  # noqa: E402
from toolrank.ingest.mcp import ServerConfig  # noqa: E402
from toolrank.retriever import Hit, Retriever  # noqa: E402
from toolrank.usage import UsageLog  # noqa: E402

FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_server.py")
ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}


class _HashEncoder:
    name = "hash"

    def encode(self, texts, *, kind="document"):
        return np.asarray(
            [
                np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).standard_normal(16)
                for t in texts
            ],
            dtype=np.float32,
        )


def _catalogue(tmp_path):
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
            id="fx/search_issues",
            doc={"server": "fx", "name": "search_issues", "description": "Search the issue tracker."},
            category="fx",
        ),
        Tool(
            id="api/getThing",
            doc={
                "server": "api",
                "name": "getThing",
                "description": "Get a thing by id.",
                "inputSchema": {"type": "object", "properties": {"id": {"type": "integer"}}},
                "http": {
                    "method": "GET",
                    "path": "/things/{id}",
                    "base_url": "https://api.example.com",
                    "args": {"id": {"in": "path", "name": "id"}},
                },
            },
            category="api",
        ),
    ]
    write_tools(tmp_path / "tools.jsonl", tools)
    return tmp_path


def _text(result):
    return result.content[0].text


def test_proxy_search_then_call_mcp_and_openapi_tools(tmp_path):
    from mcp import Client

    retriever = Retriever(_catalogue(tmp_path), lambda: DenseScorer(_HashEncoder(), "name_desc"), fixed_k=3)
    calls = []  # the paths the OpenAPI backend was asked for

    def answer(req):
        calls.append(req.url.path)
        return httpx2.Response(200, json={"id": req.url.path.rsplit("/", 1)[-1]})

    transport = httpx2.MockTransport(answer)
    backends = Backends(
        [ServerConfig("fx", "stdio", command=sys.executable, args=(FIXTURE,))],
        transport=transport,
        call_timeout=20,
    )
    usage = UsageLog(tmp_path / "usage")
    server = build_proxy(retriever, backends, usage)

    async def main():
        async with Client(server) as client:
            listed = await client.list_tools()
            assert [t.name for t in listed.tools] == ["search_tools", "call_tool"]
            assert "3 tools of 2 servers" in listed.tools[0].description
            found = json.loads(_text(await client.call_tool("search_tools", {"query": "add two integers"})))
            names = [t["name"] for t in found["tools"]]
            assert sorted(names) == ["api/getThing", "fx/add", "fx/search_issues"]
            assert next(t for t in found["tools"] if t["name"] == "fx/add")["inputSchema"] == ADD_SCHEMA
            sid = found["search_id"]
            added = await client.call_tool(
                "call_tool", {"name": "fx/add", "arguments": {"a": 2, "b": 3}, "search_id": sid}
            )
            assert not added.is_error and _text(added) == "5"
            got = await client.call_tool("call_tool", {"name": "api/getThing", "arguments": {"id": 7}})
            assert not got.is_error and _text(got).startswith("HTTP 200") and '"7"' in _text(got)
            unknown = await client.call_tool("call_tool", {"name": "fx/nope"})
            assert unknown.is_error and "search_tools" in _text(unknown)
            odd = await client.call_tool(
                "call_tool", {"name": "api/getThing", "arguments": {"id": 9}, "search_id": [sid]}
            )
            assert odd.is_error and _text(odd) == "search_id must be a string" and calls == ["/things/7"]
            bad = await client.call_tool("call_tool", {"name": "fx/add", "arguments": {"a": "x"}})
            assert bad.is_error and "inputSchema of fx/add" in bad.content[-1].text
            empty = await client.call_tool("search_tools", {"query": " "})
            assert empty.is_error

    anyio.run(main)
    events = [
        json.loads(line)
        for f in (tmp_path / "usage").glob("usage-*.jsonl")
        for line in f.read_text().splitlines()
    ]
    calls = [e for e in events if e["event"] == "call"]
    assert [e["event"] for e in events].count("search") == 1
    assert [(c["tool"], c["outcome"], c["link"]) for c in calls][:2] == [
        ("fx/add", "ok", "search_id"),
        ("api/getThing", "ok", "session"),  # in process = one session, like stdio
    ]
    # a name that is no tool is what the agent typed: a digest, as requests are, without --log-text
    assert calls[2]["tool"].startswith("unknown:") and calls[2]["outcome"] == "unknown_tool"
    assert calls[1]["http_status"] == 200 and calls[3]["outcome"] in ("tool_error", "protocol_error")


def test_a_search_that_no_tool_passes_says_so(tmp_path):
    from mcp import Client

    from toolrank.adapters.mcp_proxy import EMPTY_NOTE
    from toolrank.cut import AdaptiveK

    gate = AdaptiveK(margin=0.2, threshold=2.0, min_k=0)  # no cosine reaches 2: every request is turned away
    retriever = Retriever(_catalogue(tmp_path), lambda: DenseScorer(_HashEncoder(), "name_desc"), rule=gate)
    usage = UsageLog(tmp_path / "usage")
    server = build_proxy(retriever, Backends([]), usage)

    async def main():
        async with Client(server) as client:
            found = json.loads(_text(await client.call_tool("search_tools", {"query": "add two integers"})))
            assert found["tools"] == [] and found["note"] == EMPTY_NOTE and found["search_id"]

    anyio.run(main)
    (event,) = [
        json.loads(x) for f in (tmp_path / "usage").glob("usage-*.jsonl") for x in f.read_text().splitlines()
    ]
    assert event["shown"] == 0 and len(event["results"]) == 3  # what it would have shown stays in the log


def test_later_hits_get_shrunk_schemas():
    big = {
        "type": "object",
        "properties": {f"p{i}": {"type": "string", "description": "x" * 200} for i in range(40)},
    }
    hit = Hit(
        Tool(id="s/t", doc={"name": "t", "description": "d" * 5000, "inputSchema": big}, category="s"), 0.5
    )
    full, short = hit_json(hit, full=True), hit_json(hit, full=False)
    assert full["inputSchema"] == big and "inputSchemaShrunk" not in full
    assert short["inputSchemaShrunk"] is True and len(json.dumps(short["inputSchema"])) <= SHRUNK_CHARS
    assert len(full["description"]) == 1501


def test_guard_token_host_and_body_limit():
    seen = []

    async def inner(scope, receive, send):
        seen.append((scope["path"], scope.get(TENANT_KEY)))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    guard = Guard(
        inner,
        api_key="k",
        named_keys={"team-a": "ka"},
        hosts=allowed_hosts("127.0.0.1"),
        origins=["http://127.0.0.1:*"],
        max_body=10,
    )

    def call(path, headers):
        out = {}

        async def send(message):
            if message["type"] == "http.response.start":
                out["status"] = message["status"]

        async def receive():
            return {"type": "http.request", "body": b""}

        scope = {
            "type": "http",
            "path": path,
            "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        }
        anyio.run(guard, scope, receive, send)
        return out["status"]

    ok = {"authorization": "Bearer k", "host": "127.0.0.1:8765"}
    assert call("/mcp", {"host": "127.0.0.1:8765"}) == 401
    assert call("/v1/search", {**ok, "authorization": "Bearer wrong"}) == 401
    assert call("/v1/search", {**ok, "host": "evil.example.com"}) == 421
    assert call("/v1/search", {**ok, "content-length": "11"}) == 413
    assert call("/v1/search", ok) == 200 and call("/healthz", {"host": "x"}) == 200
    assert call("/mcp", {**ok, "authorization": "Bearer ka"}) == 200
    assert call("/v1/search", {**ok, "origin": "http://evil.example.com"}) == 403  # a web page
    assert call("/v1/search", {**ok, "origin": "null"}) == 403
    assert call("/v1/search", {**ok, "origin": "http://127.0.0.1:8765"}) == 200
    assert seen == [("/v1/search", None), ("/healthz", None), ("/mcp", "team-a"), ("/v1/search", None)]
    assert allowed_hosts("0.0.0.0", ["tools.internal", "tools.example.com:8443"]) == [
        *("127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*", "[::1]", "[::1]:*"),
        "0.0.0.0",
        "0.0.0.0:*",
        "tools.internal",
        "tools.internal:*",
        "tools.example.com:8443",
    ]


def test_identity_of_stdio_and_http_callers():
    from types import SimpleNamespace as NS

    assert identity(NS(request=None)) == ("stdio", "stdio", None)
    request = NS(scope={TENANT_KEY: "team-a"}, headers={}, client=NS(host="10.0.0.5"))
    info = NS(client_params=NS(client_info=NS(name="claude-desktop")))
    assert identity(NS(request=request, session=info)) == (None, "team-a|claude-desktop|10.0.0.5", "team-a")
    legacy = NS(scope={}, headers={"mcp-session-id": "abc"}, client=None)
    assert identity(NS(request=legacy, session=None)) == ("abc", "-|-|-", None)
