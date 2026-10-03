"""The toolrank skill (examples/skills/toolrank): its scripts run as separate processes against the
served app behind a real HTTP server, as an agent with only a shell would run them."""

import json
import os
import re
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from starlette.testclient import TestClient  # noqa: E402

from test_rest import BASE, _callable, _events  # noqa: E402
from toolrank.adapters.mcp_proxy import build_proxy, http_app  # noqa: E402
from toolrank.adapters.rest import rest_routes  # noqa: E402

SKILL = Path(__file__).resolve().parents[1] / "examples" / "skills" / "toolrank"


@contextmanager
def _http(handle):
    """A real HTTP server on a free port: ``handle(path, headers, body) -> (status, headers, body)``."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            status, headers, data = handle(self.path, dict(self.headers), body)
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def _served(tmp_path, api_key):
    """The served app (``_callable``'s catalogue), reached over HTTP like a proxy in front of it."""
    retriever, usage, backends = _callable(tmp_path)
    proxy = build_proxy(retriever, backends, usage)
    app = http_app(proxy, api_key=api_key, routes=rest_routes(retriever, usage, backends))
    with TestClient(app, base_url=BASE) as tc:

        def handle(path, headers, body):
            sent = {k: v for k, v in headers.items() if k.lower() not in ("host", "content-length")}
            r = tc.post(path, headers=sent, content=body)
            return r.status_code, {"Content-Type": r.headers.get("content-type", "")}, r.content

        with _http(handle) as url:
            yield url


def _run(script, *args, url, key=None, stdin=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TOOLRANK_")}
    env["TOOLRANK_URL"] = url
    if key:
        env["TOOLRANK_API_KEY"] = key
    cmd = [sys.executable, str(SKILL / "scripts" / script), *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, input=stdin, timeout=60)


def test_skill_card_names_the_skill_and_its_scripts():
    text = (SKILL / "SKILL.md").read_text()
    front = re.match(r"---\n(.*?)\n---\n", text, re.S).group(1)
    fields = dict(line.split(": ", 1) for line in front.splitlines())
    assert fields["name"] == SKILL.name and len(fields["description"]) <= 1024  # the Agent Skills limits
    assert "scripts/search.py" in text and "scripts/call.py" in text and len(text) < 3000


def test_skill_searches_then_calls_through_toolrank(tmp_path):
    with _served(tmp_path, api_key="k") as url:
        found = _run("search.py", "add", "two", "integers", "--k", "2", url=url, key="k")
        assert found.returncode == 0, found.stderr
        out = json.loads(found.stdout)
        assert [t["name"] for t in out["tools"]].count("fx/add") == 1 and len(out["tools"]) == 2
        assert all(
            {"name", "description", "inputSchema"} <= set(t) and "score" not in t for t in out["tools"]
        )

        added = _run(
            "call.py", "fx/add", '{"a": 2, "b": 3}', "--search-id", out["search_id"], url=url, key="k"
        )
        assert (added.returncode, added.stdout.strip()) == (0, "5")
        piped = _run("call.py", "fx/add", "-", url=url, key="k", stdin='{"a": 1, "b": 1}')
        assert (piped.returncode, piped.stdout.strip()) == (0, "2")

        refused = _run("call.py", "api/createThing", '{"name": "x"}', url=url, key="k")
        assert refused.returncode == 1 and "--allow-write" in refused.stdout  # the tool's error: read it
        missing = _run("call.py", "fx/nope", url=url, key="k")
        assert missing.returncode == 2 and "toolrank 404" in missing.stderr
        locked = _run("search.py", "add", url=url)  # no key
        assert locked.returncode == 2 and "toolrank 401" in locked.stderr
        odd = _run("call.py", "fx/add", "[1]", url=url, key="k")
        assert odd.returncode == 2 and "one JSON object" in odd.stderr
    calls = [e for e in _events(tmp_path) if e["event"] == "call"]
    # without --search-id the call goes to this client's latest search
    assert [(c["tool"], c["link"]) for c in calls[:2]] == [("fx/add", "search_id"), ("fx/add", "client")]


def test_skill_keeps_its_token_on_the_host_and_says_when_toolrank_is_away():
    def redirect(path, headers, body):
        return 302, {"Location": "http://127.0.0.1:1/steal"}, b""

    with _http(redirect) as url:
        moved = _run("search.py", "x", url=url, key="secret")
    assert moved.returncode == 2 and "toolrank 302" in moved.stderr
    away = _run("search.py", "x", url="http://127.0.0.1:1")
    assert away.returncode == 2 and "unreachable" in away.stderr
