"""Tenants (``serve --api-keys``): each named key reaches only its sources, and its credentials
travel only with its own calls."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from toolrank.adapters.dense import DenseScorer
from toolrank.datasets.jsonl import write_tools
from toolrank.domain import Tool
from toolrank.retriever import Retriever
from toolrank.tenants import Tenant, check_sources, load_tenants, parse_tenants

FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_server.py")


def test_the_keys_file_takes_plain_keys_and_tenant_objects(tmp_path, monkeypatch):
    monkeypatch.setenv("TEAM_KEY", "team-secret")
    monkeypatch.setenv("TEAM_GH", "gh-token")
    raw = {
        "ops": "ops-key",
        "team": {
            "key": "${TEAM_KEY}",
            "sources": ["github", "time"],
            "headers": {"github": {"Authorization": "Bearer ${TEAM_GH}"}},
            "env": {"time": {"TZ": "Europe/Istanbul"}},
        },
    }
    (tmp_path / "keys.json").write_text(json.dumps(raw))
    tenants = load_tenants(tmp_path / "keys.json")
    ops, team = tenants["ops"], tenants["team"]
    assert (ops.key, ops.sources, ops.allows("anything")) == ("ops-key", None, True)
    assert team.key == "team-secret" and team.sources == {"github", "time"}
    assert team.allows("github") and not team.allows("stripe")
    assert team.credentials("github") == ({"Authorization": "Bearer gh-token"}, {})
    assert team.credentials("time") == ({}, {"TZ": "Europe/Istanbul"}) and team.credentials("stripe") == (
        {},
        {},
    )
    assert "gh-token" not in repr(team) and "team-secret" not in repr(team)

    for bad, message in [
        ({}, "expected a JSON object"),
        ({"a": ""}, "non-empty string key"),
        ({"a": 3}, "key string or an object"),
        ({"a/b": "k"}, "without '/'"),
        ({"a": {"key": "k", "scopes": []}}, "unknown field"),
        ({"a": {"key": "k", "sources": "github"}}, "list of source names"),
        ({"a": {"key": "k", "headers": {"github": {"X": 1}}}}, "object of strings"),
        (
            {"a": {"key": "k", "sources": ["time"], "headers": {"github": {"X": "y"}}}},
            "not among its sources",
        ),
        ({"a": "k", "b": {"key": "k"}}, "share a key"),
    ]:
        with pytest.raises(ValueError, match=message):
            parse_tenants(bad)
    with pytest.raises(ValueError, match="UNSET_VAR"):
        parse_tenants({"a": {"key": "${UNSET_VAR}"}})


def test_credentials_must_fit_how_the_source_is_called():
    def t(**kw):
        return {"t": Tenant("t", "k", **kw)}

    servers = {"local": "stdio", "remote": "http"}
    assert check_sources(t(headers={"remote": {"X": "y"}}), ["remote"], servers, []) == []
    assert check_sources(t(headers={"stripe": {"X": "y"}}), ["stripe"], servers, ["stripe"]) == []
    assert check_sources(t(env={"local": {"A": "b"}}), ["local"], servers, []) == []
    for tenants, message in [
        (t(headers={"local": {"X": "y"}}), "neither an OpenAPI source"),
        (t(headers={"nowhere": {"X": "y"}}), "neither an OpenAPI source"),
        (t(env={"remote": {"A": "b"}}), "not a stdio MCP server"),
    ]:
        with pytest.raises(ValueError, match=message):
            check_sources(tenants, [], servers, [])
    warnings = check_sources(t(sources=frozenset({"local", "later"})), ["local"], servers, [])
    assert warnings == ["--api-keys t: sources not in the catalogue (yet): later"]


class _HashEncoder:
    name = "hash"

    def encode(self, texts, *, kind="document"):
        rows = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).standard_normal(16))
        return np.asarray(rows, dtype=np.float32)


def _catalogue(tmp_path, n_big=200):
    tools = [
        Tool(id=f"big/t{i}", doc={"name": f"t{i}", "description": f"thing {i}"}, category="big")
        for i in range(n_big)
    ]
    tools += [
        Tool(id=f"small/s{i}", doc={"name": f"s{i}", "description": f"other {i}"}, category="small")
        for i in range(3)
    ]
    write_tools(tmp_path / "tools.jsonl", tools)
    return tools


def test_a_limited_key_searches_lists_and_finds_only_its_sources(tmp_path):
    tools = _catalogue(tmp_path)

    def make():
        return DenseScorer(_HashEncoder(), "name_desc", "plain")

    r = Retriever(tmp_path, make, fixed_k=5, allowed={"team": frozenset({"small"})})
    everyone = r.search("thing 7", k=50)
    team = r.search("thing 7", tenant="team")
    # the 3 small tools rank far down among 203; the search goes deeper until it has them
    assert sorted(h.id for h in team.hits) == ["small/s0", "small/s1", "small/s2"]
    assert all(t.startswith("small/") for t, _ in team.ranked)
    assert [h.id for h in r.search("thing 7", tenant="other", k=3).hits] == [h.id for h in everyone.hits[:3]]
    assert (
        r.get("big/t1") is not None
        and r.get("big/t1", "team") is None
        and r.get("small/s1", "team") is not None
    )
    listed, catalog = r.catalogue("team")
    assert [t.id for t in listed] == ["small/s0", "small/s1", "small/s2"] and catalog == r.catalogue()[1]
    assert len(r.tools()) == len(tools) and len(r.tools("other")) == len(tools)
    nothing = Retriever(tmp_path, make, fixed_k=5, allowed={"ghost": frozenset({"absent"})})
    assert nothing.search("thing", tenant="ghost").hits == []


def test_tenant_credentials_ride_only_on_their_own_openapi_calls():
    pytest.importorskip("mcp")
    import anyio
    import httpx2

    from toolrank.adapters.backends import Backends
    from toolrank.ingest.mcp import OpenAPIBackend

    seen = []

    def answer(req):
        seen.append({k: req.headers.get(k) for k in ("authorization", "x-team", "cookie")})
        return httpx2.Response(200, headers={"set-cookie": "sid=whoever-came-first; Path=/"}, text="{}")

    http = {"method": "GET", "path": "/things", "base_url": "https://api.example.com", "args": {}}
    tool = Tool(id="api/list", doc={"name": "list", "http": http}, category="api")
    team = Tenant("team", "k1", headers={"api": {"Authorization": "Bearer team-token", "X-Team": "yes"}})
    backends = Backends(
        openapi={"api": OpenAPIBackend("api", "https://api.example.com", {"Authorization": "Bearer shared"})},
        transport=httpx2.MockTransport(answer),
        tenants={"team": team, "ops": Tenant("ops", "k2")},
    )

    async def main():
        async with backends.running():
            for tenant in ("team", "ops", None, "team"):
                assert (await backends.call(tool, {}, tenant)).outcome == "ok"

    anyio.run(main)
    assert seen == [
        {"authorization": "Bearer team-token", "x-team": "yes", "cookie": None},
        {"authorization": "Bearer shared", "x-team": None, "cookie": None},  # no cookie from the first answer
        {"authorization": "Bearer shared", "x-team": None, "cookie": None},
        {"authorization": "Bearer team-token", "x-team": "yes", "cookie": None},
    ]


def test_a_tenant_with_env_gets_a_server_process_of_its_own():
    pytest.importorskip("mcp")
    import anyio

    from toolrank.adapters.backends import Backends
    from toolrank.ingest.mcp import ServerConfig

    cfg = ServerConfig("fx", "stdio", command=sys.executable, args=(FIXTURE, "--mode", "extra"))
    team = Tenant("team", "k1", env={"fx": {"TOOLRANK_TENANT_SECRET": "team-only"}})
    backends = Backends([cfg], tenants={"team": team, "ops": Tenant("ops", "k2")}, call_timeout=20)
    assert backends.backend_for("fx", "ops") is backends.mcp["fx"] is backends.backend_for("fx", None)
    own = backends.backend_for("fx", "team")
    assert own is not backends.mcp["fx"] and own is backends.backend_for("fx", "team")
    assert own.cfg.env["TOOLRANK_TENANT_SECRET"] == "team-only" and "TOOLRANK_TENANT_SECRET" not in cfg.env

    def tool(name):
        return Tool(id=f"fx/{name}", doc={"server": "fx", "name": name}, category="fx")

    async def main():
        async with backends.running():
            got = {}
            for tenant in ("team", "ops", None):
                secret = await backends.call(tool("env"), {"name": "TOOLRANK_TENANT_SECRET"}, tenant)
                pid = await backends.call(tool("pid"), {}, tenant)
                got[tenant] = (
                    secret.result.content[0].text if secret.result.content else "",
                    pid.result.content[0].text,
                )
            return got

    got = anyio.run(main)
    assert got["team"][0] == "team-only" and got["ops"][0] == "" and got[None][0] == ""
    assert got["team"][1] != got["ops"][1] == got[None][1]  # two processes: the team's and the shared one


def test_a_limited_key_cannot_call_another_sources_tool(tmp_path):
    pytest.importorskip("mcp")
    import anyio
    import httpx2

    from toolrank.adapters.backends import Backends
    from toolrank.adapters.mcp_proxy import dispatch_call
    from toolrank.ingest.mcp import OpenAPIBackend
    from toolrank.usage import UsageLog, read_events

    _catalogue(tmp_path, n_big=2)
    http = {"method": "GET", "path": "/x", "base_url": "https://api.example.com", "args": {}}
    tools = [
        Tool(id="big/t0", doc={"name": "t0", "http": http}, category="big"),
        Tool(id="small/s0", doc={"name": "s0", "http": http}, category="small"),
    ]
    write_tools(tmp_path / "tools.jsonl", tools)
    sent = []
    backends = Backends(
        openapi={s: OpenAPIBackend(s, "https://api.example.com") for s in ("big", "small")},
        transport=httpx2.MockTransport(
            lambda req: sent.append(req.url.path) or httpx2.Response(200, text="ok")
        ),
    )
    r = Retriever(
        tmp_path,
        lambda: DenseScorer(_HashEncoder(), "name_desc", "plain"),
        allowed={"team": frozenset({"small"})},
    )
    usage = UsageLog(tmp_path / "usage")

    async def in_thread(fn, *args, **kw):
        return fn(*args, **kw)

    async def main():
        async with backends.running():
            out = {}
            for name in ("big/t0", "small/s0"):
                done = await dispatch_call(
                    r, backends, usage, in_thread, name=name, arguments={}, search_id=None,
                    who=(None, None, "team"), via="rest",
                )  # fmt: skip
                out[name] = done.outcome
            return out

    assert anyio.run(main) == {"big/t0": "unknown_tool", "small/s0": "ok"}
    assert sent == ["/x"]  # only the allowed tool reached its API
    calls = [e for e in read_events(tmp_path / "usage") if e["event"] == "call"]
    assert [(c["tool"], c["outcome"], c["tenant"]) for c in calls] == [
        ("big/t0", "unknown_tool", "team"),
        ("small/s0", "ok", "team"),
    ]
