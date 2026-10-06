import os
import sys
import time
import urllib.parse
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
import httpx2  # noqa: E402

from toolrank.adapters.backends import Backends, OpenAPIExecutor, Refused, same_origin  # noqa: E402
from toolrank.domain import Tool  # noqa: E402
from toolrank.ingest.mcp import (  # noqa: E402
    OpenAPIBackend,
    ServerConfig,
    expand_env,
    load_config,
    load_openapi,
)

FIXTURE = str(Path(__file__).parent / "fixtures" / "mcp_server.py")


def _mcp(server, name):
    return Tool(id=f"{server}/{name}", doc={"server": server, "name": name}, category=server)


def _text(outcome):
    return outcome.result.content[0].text


def test_mcp_backend_calls_times_out_reconnects_and_leaves_no_process():
    cfg = ServerConfig("fx", "stdio", command=sys.executable, args=(FIXTURE, "--mode", "extra"))

    async def main():
        b = Backends([cfg], call_timeout=1.0, connect_timeout=30)
        async with b.running():
            add = await b.call(_mcp("fx", "add"), {"a": 2, "b": 3})
            assert (add.outcome, _text(add)) == ("ok", "5")
            first = int(_text(await b.call(_mcp("fx", "pid"), {})))
            slow = await b.call(_mcp("fx", "slow"), {"seconds": 5})
            assert slow.outcome == "timeout" and slow.result.is_error
            crash = await b.call(_mcp("fx", "crash"), {})
            assert crash.outcome == "protocol_error" and "not retried" in _text(crash)
            again = await b.call(_mcp("fx", "add"), {"a": 1, "b": 1})  # a new process
            assert (again.outcome, _text(again)) == ("ok", "2")
            second = int(_text(await b.call(_mcp("fx", "pid"), {})))
            assert second != first
            missing = await b.call(_mcp("nope", "x"), {})
            assert missing.outcome == "refused"
        return second

    pid = anyio.run(main)
    for _ in range(50):  # the stdio client terminates its process when the task group closes
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"backend process {pid} outlived the proxy")


def test_unreachable_server_is_an_error_result_not_an_exception():
    cfg = ServerConfig("gone", "stdio", command="no-such-command-xyz")

    async def main():
        async with Backends([cfg], connect_timeout=5).running() as b:
            return await b.call(_mcp("gone", "x"), {})

    out = anyio.run(main)
    assert out.outcome == "protocol_error" and "command not found" in _text(out)


def _op(
    method="GET", path="/items/{id}", args=None, media=None, base="https://api.example.com/v1/", source="api"
):
    http = {"method": method, "path": path, "base_url": base, "args": args or {}}
    if media:
        http["body_media_type"] = media
    return Tool(id=f"{source}/{method.lower()}", doc={"name": "op", "http": http}, category=source)


QUERY = {
    "id": {"in": "path", "name": "id"},
    "tags": {"in": "query", "name": "tags", "style": "form", "explode": True},
    "created": {"in": "query", "name": "created", "style": "deepObject", "explode": True},
    "flag": {"in": "query", "name": "flag", "style": "form", "explode": True},
}


def test_openapi_request_building_and_refusals(monkeypatch):
    ex = OpenAPIExecutor({})
    req = ex.build(
        _op(args=QUERY), {"id": "a/b", "tags": ["x", "y"], "created": {"gte": 1, "lt": 2}, "flag": True}
    )
    assert req["url"] == "https://api.example.com/v1/items/a%2Fb" and req["headers"] == {}
    assert req["params"] == [
        ("tags", "x"),
        ("tags", "y"),
        ("created[gte]", "1"),
        ("created[lt]", "2"),
        ("flag", "true"),
    ]
    body = {"metadata": {"k": "v"}, "items": [{"price": "p"}]}
    form = _op(
        "POST",
        "/customers",
        {"metadata": {"in": "body", "name": "metadata"}, "items": {"in": "body", "name": "items"}},
        "application/x-www-form-urlencoded",
    )
    with pytest.raises(Refused, match="--allow-write"):
        ex.build(form, body)
    req = OpenAPIExecutor({}, allow_write=True).build(form, body)
    assert urllib.parse.parse_qsl(req["content"].decode()) == [("metadata[k]", "v"), ("items[0][price]", "p")]
    json_op = _op("POST", "/x", {"a": {"in": "body", "name": "a"}}, "application/json")
    assert OpenAPIExecutor({}, allow_write=True).build(json_op, {"a": [1]})["json"] == {"a": [1]}
    with pytest.raises(Refused, match="re-ingest"):  # a row ingested before styles were recorded
        ex.build(_op(args={"id": QUERY["id"], "t": {"in": "query", "name": "t"}}), {"id": 1, "t": [1]})
    with pytest.raises(ValueError, match="missing path parameter"):
        ex.build(_op(args=QUERY), {})
    for dots in (".", ".."):  # httpx2 would resolve them to a parent path on the credentialed host
        with pytest.raises(ValueError, match="cannot be"):
            ex.build(_op(args=QUERY), {"id": dots})
    with pytest.raises(ValueError, match="no argument"):
        ex.build(_op(args=QUERY), {"id": 1, "zzz": 2})
    with pytest.raises(Refused, match="base_url"):
        ex.build(_op(args=QUERY, base="/v1"), {"id": 1})
    creds = OpenAPIExecutor(
        {"api": OpenAPIBackend("api", "https://real.example.com/v2", {"Authorization": "Bearer t"})}
    )
    req = creds.build(_op(args=QUERY), {"id": 7})
    assert req["url"] == "https://real.example.com/v2/items/7" and req["headers"] == {
        "Authorization": "Bearer t"
    }
    assert OpenAPIExecutor({}).build(_op(args=QUERY, source="other"), {"id": 7})["headers"] == {}
    # a spec's path is appended to the base URL: without its slash it would name the host, and the
    # config's headers would go there (https://real.example.com/v2@evil.example.com/x)
    for path in ("@evil.example.com/x", ".evil.example.com/x", ":443@evil.example.com/x", ""):
        with pytest.raises(Refused, match="does not start with"):
            creds.build(_op(path=path), {})
    for path in ("//evil.example.com/x", "/@evil.example.com/x", "/\\evil.example.com/x"):
        req = creds.build(_op(path=path), {})
        assert httpx2.URL(req["url"]).host == "real.example.com"
    # the guard is applied to the URL that is sent, after the path parameters are in it
    hostless = OpenAPIExecutor({"api": OpenAPIBackend("api", "https://real.example.com", {"X-Key": "k"})})
    sent = hostless.build(_op(args=QUERY), {"id": "@evil.example.com/x"})
    assert sent["url"] == "https://real.example.com/items/%40evil.example.com%2Fx"
    asked = []
    monkeypatch.setattr(
        "toolrank.adapters.backends.same_origin", lambda url, base: asked.append((url, base)) and False
    )
    with pytest.raises(Refused, match="is not a URL on https://real.example.com"):
        hostless.build(_op(args=QUERY), {"id": 7})
    assert asked == [("https://real.example.com/items/7", "https://real.example.com")]
    monkeypatch.undo()
    # the invariant behind it, as the HTTP client parses the two URLs
    base = "https://real.example.com"  # no path of its own: what follows it can still be the host
    assert same_origin(base + "/items/7", base) and same_origin("https://real.example.com:443/x", base)
    for url in (base + "@evil.example.com/x", base + ".evil.example.com/x", "http://real.example.com/x"):
        assert not same_origin(url, base)
    assert not same_origin("https://real.example.com:8443/x", base) and not same_origin("https://[", base)


def test_openapi_calls_statuses_redirects_and_truncation():
    seen = []

    def handler(request):
        seen.append(request)
        path = request.url.path
        if path.endswith("/missing"):
            return httpx2.Response(404, text="not found")
        if path.endswith("/moved"):
            return httpx2.Response(302, headers={"location": "https://evil.example.com/steal"})
        if path.endswith("/big"):
            return httpx2.Response(200, text="x" * 5000)
        return httpx2.Response(200, json={"ok": True})

    creds = {"api": OpenAPIBackend("api", "https://api.example.com", {"Authorization": "Bearer t"})}
    ex = OpenAPIExecutor(creds, transport=httpx2.MockTransport(handler), max_chars=1000)

    async def main():
        out = {}
        for p in ("/ok", "/missing", "/moved", "/big"):
            out[p] = await ex.call(_op(path=p), {})
        out["post"] = await ex.call(_op("POST", "/ok"), {})
        await ex.aclose()
        return out

    out = anyio.run(main)
    assert (out["/ok"].outcome, out["/ok"].http_status) == ("ok", 200) and '"ok"' in _text(out["/ok"])
    assert (out["/missing"].outcome, out["/missing"].http_status) == ("tool_error", 404)
    assert "not followed" in _text(out["/moved"]) and len(seen) == 4  # no request to the Location
    assert "truncated: 5000 bytes in total" in _text(out["/big"]) and len(_text(out["/big"])) < 1100
    assert out["post"].outcome == "refused"
    assert all(r.headers["authorization"] == "Bearer t" and r.url.host == "api.example.com" for r in seen)


def test_config_env_expansion_and_openapi_section(tmp_path, monkeypatch):
    monkeypatch.setenv("TR_TOKEN", "s3cret")
    monkeypatch.delenv("TR_MISSING", raising=False)
    assert expand_env({"a": ["${TR_TOKEN}", "x-${env:TR_TOKEN}"]}) == {"a": ["s3cret", "x-s3cret"]}
    with pytest.raises(ValueError, match="TR_MISSING is not set"):
        expand_env("${TR_MISSING}", "remote")
    cfg = tmp_path / "toolrank.json"
    cfg.write_text(
        '{"mcpServers": {"r": {"url": "https://h/mcp", "headers": {"Authorization": "Bearer ${TR_TOKEN}"}}},'
        ' "openapi": {"stripe": {"base_url": "https://api.stripe.com", "headers": {"Authorization": "Bearer ${TR_TOKEN}"}}}}'
    )
    (server,) = load_config(cfg)
    assert server.headers == {"Authorization": "Bearer s3cret"} and "s3cret" not in repr(server)
    api = load_openapi(cfg)["stripe"]
    assert api.base_url == "https://api.stripe.com" and api.headers["Authorization"] == "Bearer s3cret"
    with pytest.raises(ValueError, match="absolute base_url"):
        OpenAPIBackend("x", "", {"Authorization": "t"})
    # a config that cannot be read or parsed is a message naming the file, not a traceback
    for load in (load_config, load_openapi):
        with pytest.raises(ValueError, match="missing.json: No such file"):
            load(tmp_path / "missing.json")
    (tmp_path / "broken.json").write_text("{")
    with pytest.raises(ValueError, match="broken.json: not JSON"):
        load_config(tmp_path / "broken.json")
    (tmp_path / "utf16.json").write_text('{"mcpServers": {}}', encoding="utf-16")
    with pytest.raises(ValueError, match="utf16.json: not UTF-8"):
        load_openapi(tmp_path / "utf16.json")
    cfg.chmod(0)
    if not os.access(cfg, os.R_OK):  # root reads anything
        with pytest.raises(
            ValueError, match=rf"toolrank.json: Permission denied: toolrank runs as uid {os.getuid()}"
        ):
            load_config(cfg)


def test_a_spec_cannot_send_calls_to_metadata_or_climb_the_path():
    """Without a configured base_url the spec's own servers choose the host: never link-local or
    cloud metadata (credentials for whoever asks). "../x" climbs once an upstream decodes %2F."""
    ex = OpenAPIExecutor({})
    for base in (
        "http://169.254.169.254/latest/",
        "http://metadata.google.internal/computeMetadata/",
        "http://[fe80::1]/",
    ):
        with pytest.raises(Refused, match="link-local or metadata"):
            ex.build(_op(args=QUERY, base=base), {"id": 1})
    configured = OpenAPIExecutor({"api": OpenAPIBackend("api", "http://169.254.169.254/latest", {})})
    assert configured.build(_op(args=QUERY, base="ignored"), {"id": 1})["url"].startswith(
        "http://169.254.169.254/"
    )
    for climb in ("../admin", "x/../../admin", "..%2Fadmin", "..\\admin"):
        with pytest.raises(ValueError, match="dot segment"):
            ex.build(_op(args=QUERY), {"id": climb})
    assert ex.build(_op(args=QUERY), {"id": "a..b"})["url"].endswith("/items/a..b")
