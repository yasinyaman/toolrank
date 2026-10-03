"""End-to-end check of ``toolrank serve`` (Faz 1 week 3): MCP over streamable HTTP and stdio, the
REST routes, the usage log, and the first start with an empty index.

Needs an ingest dir with the week-1 sources (time, everything, swagger-petstore, stripe), ``uvx``
and ``npx`` on PATH, and the embedding endpoint the heads were trained on. The petstore spec's base
URL is a demo host with other paths, so the script serves a local stand-in and points the config at
it, with an ``X-Api-Key`` header from ``${PETSTORE_KEY}``: no tool call leaves the machine. Stripe
and GitHub are only searched; the one Stripe call is a POST the proxy refuses. The bearer token and
the petstore key are random per run; the token is a named key (``--api-keys``), so the usage log
records its name as the tenant.

    uv run python scripts/serve_e2e.py --data data/w3 --emb-url http://$GB10:8091/v1 \
        --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --out results/serve_e2e.json

Delete ``DATA/index`` and ``DATA/cache`` first to measure a cold start (every tool embedded).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from collections import Counter
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import anyio
import httpx2

from toolrank.ingest.mcp import ServerConfig

QUERIES = [
    "what time is it in Tokyo right now",
    "echo back this message",
    "list the pets tagged dog",
    "open an issue in my repository about the login bug",
    "refund the last payment of this customer",
    "cancel the subscription at the end of the billing period",
]
STDIO_QUERY = "convert 3pm New York time to Istanbul time"
CALLS = [  # (tool, arguments, expected)
    ("time/get_current_time", {"timezone": "Asia/Tokyo"}, "ok"),
    ("everything/echo", {"message": "hello from toolrank"}, "ok"),
    ("swagger-petstore/findPets", {"tags": ["dog", "cat"], "limit": 2}, "ok (stand-in, key sent)"),
    ("swagger-petstore/find pet by id", {"id": 404}, "HTTP 404: error + schema"),
    ("stripe/PostRefunds", {"charge": "ch_123"}, "refused: POST"),
    ("time/get_current_time", {"tz": "Asia/Tokyo"}, "bad argument: error + schema"),
    ("nope/missing", {}, "unknown tool"),
]
STEADY = [
    ("time/get_current_time", {"timezone": "Europe/Istanbul"}),
    ("everything/echo", {"message": "ping"}),
    ("swagger-petstore/findPets", {"limit": 1}),
]
REPEAT = 20


def since(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def petstore(key: str) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urllib.parse.urlsplit(self.path)
            if url.path.rstrip("/").endswith("/404"):
                status, body = 404, {"error": "no such pet"}
            else:
                status = 200
                body = {
                    "path": url.path,
                    "query": urllib.parse.parse_qsl(url.query),
                    "api_key_ok": self.headers.get("X-Api-Key") == key,
                    "pets": [{"id": 1, "name": "Rex", "tag": "dog"}, {"id": 2, "name": "Tom", "tag": "cat"}],
                }
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def write_config(directory: Path, pet_port: int) -> Path:
    cfg = {
        "mcpServers": {
            "time": {"command": "uvx", "args": ["mcp-server-time"]},
            "everything": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-everything"]},
        },
        "openapi": {
            "swagger-petstore": {
                "base_url": f"http://127.0.0.1:{pet_port}/v2",
                "headers": {"X-Api-Key": "${PETSTORE_KEY}"},
            }
        },
    }
    path = directory / "toolrank.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def backend_pids() -> set[int]:
    out = subprocess.run(
        ["pgrep", "-f", "mcp-server-time|server-everything"], capture_output=True, text=True
    ).stdout
    return {int(x) for x in out.split()}


def text(result: Any) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


def client_for(cfg: ServerConfig, errlog: Any, http: Any) -> Any:
    from mcp import Client

    from toolrank.adapters.mcp_client import _transport

    return Client(_transport(cfg, errlog, http), cache=None)


async def timed(client: Any, name: str, args: dict[str, Any]) -> tuple[Any, float]:
    t = time.perf_counter()
    result = await client.call_tool(name, args)
    return result, (time.perf_counter() - t) * 1000.0


def p50_p90(ms: list[float]) -> dict[str, float]:
    s = sorted(ms)
    return {"p50": round(statistics.median(s), 1), "p90": round(s[max(int(0.9 * len(s)) - 1, 0)], 1)}


async def searches(client: Any, queries: list[str], warm: int = 3) -> list[dict[str, Any]]:
    out = []
    for q in queries:
        r, cold = await timed(client, "search_tools", {"query": q})
        if r.is_error:
            out.append({"query": q, "error": text(r), "ms": round(cold, 1)})
            continue
        payload = json.loads(text(r))
        again = [(await timed(client, "search_tools", {"query": q}))[1] for _ in range(warm)]
        out.append(
            {
                "query": q,
                "mode": "lexical" if "note" in payload else "semantic",
                "tools": [t["name"] for t in payload["tools"]],
                "search_id": payload["search_id"],
                "cold_ms": round(cold, 1),
                "warm_ms": round(statistics.median(again), 1) if again else None,
            }
        )
    return out


async def calls(client: Any, search_id: str | None) -> list[dict[str, Any]]:
    out = []
    for n, (name, args, expected) in enumerate(CALLS):
        body: dict[str, Any] = {"name": name, "arguments": args}
        if n == 0 and search_id:
            body["search_id"] = search_id
        r, ms = await timed(client, "call_tool", body)
        out.append(
            {
                "tool": name,
                "expected": expected,
                "is_error": r.is_error,
                "ms": round(ms, 1),
                "text": text(r)[:400],
            }
        )
    return out


async def steady(client: Any) -> dict[str, Any]:
    out = {}
    for name, args in STEADY:
        ms = [(await timed(client, "call_tool", {"name": name, "arguments": args}))[1] for _ in range(REPEAT)]
        out[name] = p50_p90(ms)
    ms = [(await timed(client, "search_tools", {"query": QUERIES[0]}))[1] for _ in range(REPEAT)]
    out["search_tools (warm)"] = p50_p90(ms)
    return out


async def rest(http: Any, base: str, key: str) -> dict[str, Any]:
    auth = {"Authorization": f"Bearer {key}"}
    out: dict[str, Any] = {}
    out["search_without_token"] = (await http.post(base + "/v1/search", json={"query": "x"})).status_code
    evil = {**auth, "Host": "evil.example.com"}
    out["search_wrong_host"] = (
        await http.post(base + "/v1/search", json={"query": "x"}, headers=evil)
    ).status_code
    page = {**auth, "Origin": "https://evil.example.com", "Content-Type": "text/plain"}
    out["search_from_a_web_page"] = (
        await http.post(base + "/v1/search", content=b'{"query": "x"}', headers=page)
    ).status_code
    t = time.perf_counter()
    r = await http.post(
        base + "/v1/search",
        json={"query": "list the pets tagged dog"},
        headers={**auth, "X-Session-Id": "e2e"},
    )
    j = r.json()
    out["search"] = {
        "status": r.status_code,
        "wall_ms": round((time.perf_counter() - t) * 1000.0, 1),
        "took_ms": j["took_ms"],
        "rule": j["rule"],
        "tools": [x["name"] for x in j["tools"]],
    }
    given = [
        {
            "name": "get_weather",
            "description": "Get the weather forecast for a city.",
            "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
        {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "inputSchema": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            },
        },
        {"name": "create_event", "description": "Create a calendar event."},
    ]
    r = await http.post(
        base + "/v1/rank", json={"query": "email my boss that I will be late", "tools": given}, headers=auth
    )
    out["rank_given"] = [[x["name"], x["score"]] for x in r.json()["tools"]]
    ids = ["everything/echo", "time/convert_time", "time/get_current_time"]
    r = await http.post(
        base + "/v1/rank", json={"query": "what time is it in London", "tool_ids": ids}, headers=auth
    )
    out["rank_ids"] = [[x["name"], x["score"]] for x in r.json()["tools"]]
    r = await http.post(base + "/v1/rank", json={"query": "x", "tool_ids": ["nope/x"]}, headers=auth)
    out["rank_unknown_id"] = [r.status_code, r.json()["error"]]
    r = await http.get(base + "/v1/tools", params={"server": "time"}, headers=auth)
    out["tools_time"] = [r.json()["count"], [x["name"] for x in r.json()["tools"]]]
    r = await http.get(base + "/v1/tools", headers=auth)
    out["tools_all"] = r.json()["count"]
    tool = urllib.parse.quote("swagger-petstore/find pet by id")
    r = await http.get(base + "/v1/tools/" + tool, headers=auth)
    out["tool_record"] = [r.status_code, r.json().get("kind"), (r.json().get("http") or {}).get("path")]
    r = await http.get(base + "/openapi.json")
    out["openapi"] = [r.status_code, r.json()["openapi"], sorted(r.json()["paths"])]
    r = await http.get(base + "/healthz")
    out["healthz"] = [r.status_code, r.json()]
    calls = {}
    for label, body in [
        ("pets", {"name": "swagger-petstore/findPets", "arguments": {"tags": ["dog"], "limit": 1}}),
        ("stripe_post", {"name": "stripe/PostRefunds", "arguments": {"charge": "ch_123"}}),
        ("unknown", {"name": "nope/missing"}),
    ]:
        t = time.perf_counter()
        r = await http.post(base + "/v1/call", json=body, headers=auth)
        j = r.json()
        calls[label] = [
            r.status_code,
            j.get("outcome") or j.get("error"),
            round((time.perf_counter() - t) * 1000, 1),
        ]
    r = await http.post(
        base + "/v1/call", content=b'{"name": "x"}', headers={**auth, "Content-Type": "text/plain"}
    )
    calls["not_json"] = [r.status_code]
    out["call"] = calls
    return out


async def http_phase(
    a: argparse.Namespace, env: dict[str, str], cfg_path: Path, key: str, logdir: Path
) -> dict:
    port = a.port or free_port()
    base = f"http://127.0.0.1:{port}"
    exe = str(Path(sys.executable).with_name("toolrank"))
    cmd = [
        exe,
        "serve",
        "--data",
        a.data,
        "--config",
        str(cfg_path),
        "--emb-url",
        a.emb_url,
        "--emb-model",
        a.emb_model,
        "--port",
        str(port),
        "--api-keys",
        str(logdir / "keys.json"),
    ]
    (logdir / "keys.json").write_text(json.dumps({"e2e": key}))
    before = backend_pids()
    out: dict[str, Any] = {}
    log_path = logdir / "serve_http.log"
    with log_path.open("w") as log:
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=log)
        try:
            async with httpx2.AsyncClient(timeout=httpx2.Timeout(30, read=120)) as plain:
                while True:
                    try:
                        r = await plain.get(base + "/healthz")
                        break
                    except httpx2.TransportError:
                        if proc.poll() is not None:
                            raise SystemExit(f"serve exited: {log_path.read_text()[-2000:]}") from None
                        await anyio.sleep(0.05)
                out["http_up_s"] = since(t0)
                out["healthz_at_start"] = [r.status_code, r.json()]
                cfg = ServerConfig(
                    "toolrank", "http", url=base + "/mcp", headers={"Authorization": f"Bearer {key}"}
                )
                async with AsyncExitStack() as stack:
                    http = await stack.enter_async_context(
                        httpx2.AsyncClient(headers=cfg.headers, timeout=httpx2.Timeout(30, read=120))
                    )
                    client = await stack.enter_async_context(client_for(cfg, None, http))
                    listed = await client.list_tools()
                    out["mcp_listed_s"] = since(t0)
                    out["search_tool_description_at_start"] = listed.tools[0].description[:90]
                    r, ms = await timed(client, "search_tools", {"query": QUERIES[0]})
                    out["first_search"] = {"done_s": since(t0), "ms": round(ms), "is_error": r.is_error}
                    out["first_search"]["text"] = text(r)[:200]
                    health = (await plain.get(base + "/healthz")).json()
                    if health.get("mode") == "lexical":  # still building: keyword matches, calls work
                        found = await searches(client, QUERIES, warm=0)
                        body = {"name": "swagger-petstore/findPets", "arguments": {"limit": 1}}
                        r, ms = await timed(client, "call_tool", body)  # HTTP stand-in: starts no process
                        out["while_building"] = {
                            "searches": found,
                            "call": {"tool": body["name"], "is_error": r.is_error, "ms": round(ms, 1)},
                            "at_s": since(t0),
                        }
                    while (await plain.get(base + "/healthz")).status_code != 200:
                        if proc.poll() is not None:
                            raise SystemExit(f"serve exited: {log_path.read_text()[-2000:]}")
                        await anyio.sleep(0.5)
                    out["ready_s"] = since(t0)
                    listed = await client.list_tools()
                    out["search_tool_description_ready"] = listed.tools[0].description[:90]
                    out["searches"] = await searches(client, QUERIES)
                    out["calls"] = await calls(client, out["searches"][0].get("search_id"))
                    out["steady"] = await steady(client)
                out["rest"] = await rest(plain, base, key)
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    await anyio.sleep(1.0)
    out["backends_left_running"] = sorted(backend_pids() - before)
    out["server_log"] = log_path.read_text().splitlines()[-8:]
    return out


async def stdio_phase(a: argparse.Namespace, env: dict[str, str], cfg_path: Path) -> dict:
    exe = str(Path(sys.executable).with_name("toolrank"))
    args = ("serve", "--stdio", "--data", a.data, "--config", str(cfg_path), "--emb-url", a.emb_url)
    args += ("--emb-model", a.emb_model)
    cfg = ServerConfig("toolrank", "stdio", command=exe, args=args, env=env, cwd="/")  # like Claude Desktop
    before = backend_pids()
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    with tempfile.TemporaryFile("w+") as errlog:
        async with client_for(cfg, errlog, None) as client:
            out["initialized_s"] = since(t0)
            listed = await client.list_tools()
            out["tools"] = [t.name for t in listed.tools]
            out["search_tool_description"] = listed.tools[0].description[:90]
            r, ms = await timed(client, "search_tools", {"query": STDIO_QUERY})
            payload = json.loads(text(r)) if not r.is_error else {}
            out["first_search"] = {
                "done_s": since(t0),
                "ms": round(ms, 1),
                "tools": [t["name"] for t in payload.get("tools", [])] or text(r)[:200],
            }
            out["repeat_search_ms"] = round(
                (await timed(client, "search_tools", {"query": STDIO_QUERY}))[1], 1
            )
            body = {
                "name": "time/convert_time",
                "arguments": {
                    "source_timezone": "America/New_York",
                    "time": "15:00",
                    "target_timezone": "Europe/Istanbul",
                },
                "search_id": payload.get("search_id"),
            }
            r, ms = await timed(client, "call_tool", body)
            out["convert_time"] = {"is_error": r.is_error, "ms": round(ms, 1), "text": text(r)[:300]}
            r, ms = await timed(
                client, "call_tool", {"name": "everything/echo", "arguments": {"message": "stdio"}}
            )
            out["echo"] = {"is_error": r.is_error, "ms": round(ms, 1), "text": text(r)[:100]}
        errlog.seek(0)
        out["stderr"] = errlog.read().splitlines()[-6:]
    out["closed_s"] = since(t0)
    await anyio.sleep(1.0)
    out["backends_left_running"] = sorted(backend_pids() - before)
    return out


def usage_summary(data: Path, start: str) -> dict[str, Any]:
    events = [
        json.loads(line)
        for f in sorted((data / "usage").glob("usage-*.jsonl"))
        for line in f.read_text().splitlines()
    ]
    events = [e for e in events if e["ts"] >= start]
    calls_ = [e for e in events if e["event"] == "call"]
    linked = next((e for e in calls_ if e["link"] == "search_id"), None)
    return {
        "events": len(events),
        "by_event": dict(Counter(f"{e['event']}/{e['via']}" for e in events)),
        "sessions": dict(Counter("null" if e["session"] is None else e["session"][:16] for e in events)),
        "tenants": dict(Counter(str(e["tenant"]) for e in events)),
        "clients": len({e["client"] for e in events}),
        "links": dict(Counter(e["link"] for e in calls_)),
        "outcomes": dict(Counter(e["outcome"] for e in calls_)),
        "query_text_logged": any(e.get("query") for e in events if e["event"] == "search"),
        "linked_call": linked,
        "key_mode": oct((data / "usage" / ".key").stat().st_mode & 0o777),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="ingest dir with the week-1 sources")
    p.add_argument("--emb-url", required=True)
    p.add_argument(
        "--emb-model",
        default="qwen3-emb",
        help="served name at --emb-url; 8091 serves qwen3-emb, the base the v0.1 heads belong to",
    )
    p.add_argument("--heads", default=None, help="packaged heads (.npz); default: $TOOLRANK_HEADS")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--skip-stdio", action="store_true")
    p.add_argument("--out", default=None, help="write the summary JSON here")
    a = p.parse_args()
    a.data = str(Path(a.data).resolve())
    start = datetime.now(UTC).isoformat(timespec="milliseconds")
    pet_key, api_key = secrets.token_hex(8), secrets.token_hex(16)
    stand_in = petstore(pet_key)
    env = {"PETSTORE_KEY": pet_key}
    heads = a.heads or os.environ.get("TOOLRANK_HEADS")
    if heads:
        env["TOOLRANK_HEADS"] = str(Path(heads).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = write_config(Path(tmp), stand_in.server_address[1])
        report: dict[str, Any] = {"data": a.data, "emb_url": a.emb_url, "heads": env.get("TOOLRANK_HEADS")}
        report["http"] = anyio.run(http_phase, a, env, cfg_path, api_key, Path(tmp))
        if not a.skip_stdio:
            report["stdio"] = anyio.run(stdio_phase, a, env, cfg_path)
    stand_in.shutdown()
    report["usage"] = usage_summary(Path(a.data), start)
    dump = json.dumps(report, indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(dump + "\n")
    print(dump)


if __name__ == "__main__":
    main()
