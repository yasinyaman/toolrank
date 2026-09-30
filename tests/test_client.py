import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("mcp")

from starlette.testclient import TestClient  # noqa: E402

from test_rest import BASE, _callable, _events  # noqa: E402
from toolrank.adapters.mcp_proxy import build_proxy, http_app  # noqa: E402
from toolrank.adapters.rest import rest_routes  # noqa: E402
from toolrank.client import ToolrankClient, ToolrankError  # noqa: E402


def _through(tc):
    """The client's transport, routed into Starlette's TestClient."""

    def send(method, url, headers, body, timeout):
        r = tc.request(method, url, headers=headers, content=body)
        return r.status_code, r.content

    return send


@contextmanager
def _toolrank(tmp_path):
    """A client of the served app (MCP proxy + REST) over ``_callable``'s catalogue."""
    retriever, usage, backends = _callable(tmp_path)
    app = http_app(build_proxy(retriever, backends, usage), routes=rest_routes(retriever, usage, backends))
    with TestClient(app, base_url=BASE) as tc:
        yield ToolrankClient(BASE, transport=_through(tc))


def test_client_against_the_served_app(tmp_path):
    retriever, usage, backends = _callable(tmp_path)
    proxy = build_proxy(retriever, backends, usage)
    app = http_app(proxy, api_key="k", routes=rest_routes(retriever, usage, backends))
    with TestClient(app, base_url=BASE) as tc:
        tr = ToolrankClient(BASE, api_key="k", session="conv-1", transport=_through(tc))
        assert tr.health()["ready"] is True
        catalog = tr.catalog()
        assert {t["api_name"] for t in catalog["tools"]} == {"fx__add", "api__getThing", "api__createThing"}
        assert catalog["catalog"] and tr.catalog(server="fx")["count"] == 1
        found = tr.search("add two integers", k=2)
        assert len(found["tools"]) == 2 and found["mode"] == "semantic"
        out = tr.call("fx/add", {"a": 1, "b": 2}, search_id=found["search_id"])
        assert (out["outcome"], out["content"]) == ("ok", [{"type": "text", "text": "3"}])
        with pytest.raises(ToolrankError) as e:
            tr.call("fx/nope")
        assert e.value.status == 404 and "no tool named" in e.value.message
        with pytest.raises(ToolrankError) as e:
            ToolrankClient(BASE, api_key="wrong", transport=_through(tc)).search("x")
        assert e.value.status == 401
        mine = [{"name": f"t{n}", "description": f"tool number {n}", "inputSchema": {}} for n in range(450)]
        ranked = tr.rank("tool number 7", mine)  # three requests: 200 + 200 + 50 tools
        assert sorted(r["index"] for r in ranked) == list(range(450))
        assert [r["score"] for r in ranked] == sorted((r["score"] for r in ranked), reverse=True)
        assert all(r["name"] == f"t{r['index']}" for r in ranked)
    calls = [ev for ev in _events(tmp_path) if ev["event"] == "call"]
    assert (calls[0]["session"], calls[0]["link"]) == ("rest:conv-1", "search_id")


class _Stub(BaseHTTPRequestHandler):
    seen: list = []

    def do_GET(self):
        self._answer()

    def do_POST(self):
        self._answer()

    def _answer(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        _Stub.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if self.path.startswith("/v1/tools"):  # would hand the token to another host if followed
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/steal")
            self.end_headers()
            return
        if self.path == "/v1/call":
            time.sleep(1.0)
        status, data = {
            "/healthz": (503, b'{"ready": false}'),
            "/v1/search": (502, b"upstream exploded"),
        }.get(self.path, (200, b"{}"))
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class _Quiet(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # the timed-out call's broken pipe
        pass


def test_urllib_transport_headers_errors_redirects_and_timeouts():
    _Stub.seen = []
    server = _Quiet(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        tr = ToolrankClient(base, api_key="k", session="s1", call_timeout=0.2)
        assert tr.health() == {"ready": False}  # a server still building its index is an answer
        with pytest.raises(ToolrankError) as e:
            tr.catalog()
        assert e.value.status == 302  # not followed: the token stays on this host
        with pytest.raises(ToolrankError) as e:
            tr.search("x")
        assert (e.value.status, e.value.message) == (502, "upstream exploded")
        with pytest.raises(ToolrankError, match="timed out"):
            tr.call("a/b")
        path, headers, body = _Stub.seen[2]
        assert path == "/v1/search" and json.loads(body) == {"query": "x"}
        assert (headers["authorization"], headers["x-session-id"], headers["content-type"]) == (
            "Bearer k",
            "s1",
            "application/json",
        )
        assert headers["user-agent"].startswith("toolrank-client/") and len(_Stub.seen) == 4
    finally:
        server.shutdown()
        server.server_close()
    with pytest.raises(ToolrankError) as e:
        ToolrankClient(base, timeout=2).health()
    assert e.value.status == 0 and "unreachable" in e.value.message


def test_rank_chunks_respect_the_tool_count_and_the_body_size():
    from toolrank.client import RANK_BYTES, RANK_CHUNK, _chunks

    big = [{"name": f"b{n}", "description": "x" * 400_000} for n in range(5)]  # ~400 KB each
    assert [(start, len(c)) for start, c in _chunks(big)] == [(0, 2), (2, 2), (4, 1)]
    small = [{"name": str(n)} for n in range(RANK_CHUNK + 1)]
    assert [len(c) for _, c in _chunks(small)] == [RANK_CHUNK, 1] and RANK_BYTES < 1 << 20
    assert _chunks([]) == []
