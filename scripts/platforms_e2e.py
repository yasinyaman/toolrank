"""End-to-end check of the platform integrations (Faz 1 week 4) on a real catalogue.

Starts ``toolrank serve`` on ``--data`` with the week-1 MCP servers (time, everything) and a local
stand-in for every OpenAPI source (the petstore answers ``/v2/pets``, anything else is a 404), so no
tool call leaves the machine; Stripe writes are refused by the proxy as usual. Then:

- always: the catalogue as the platforms get it — Anthropic's deferred tool list (bytes, api names,
  hashed names), every input schema against the JSON Schema 2020-12 meta-schema, each task's search
  with what Claude would get referenced and what OpenAI would get loaded, and two calls;
- ``--live``: each task through Claude with toolrank's search, Claude with the API's own BM25 tool
  search, and an OpenAI model with toolrank's search, against the real APIs (keys from the
  environment or ``.env``, never printed); then the usage log's call links.

    uv run python scripts/platforms_e2e.py --data data/w3 --emb-url http://$GB10:8091/v1 \
        --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz [--live] --out results/platforms_e2e.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from toolrank.client import ToolrankClient, ToolrankError
from toolrank.integrations import anthropic as claude
from toolrank.integrations import openai as gpt

TASKS = [
    "What time is it in Tokyo right now?",
    "List the pets tagged dog in the pet store.",
    "Refund the last payment of customer cus_123.",
]
PETS = [
    {"id": 1, "name": "Rex", "tag": "dog"},
    {"id": 2, "name": "Tom", "tag": "cat"},
    {"id": 3, "name": "Ace", "tag": "dog"},
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def stand_in() -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path, _, query = self.path.partition("?")
            if path == "/petstore/v2/pets":
                tags = [v for k, _, v in (p.partition("=") for p in query.split("&")) if k == "tags"]
                status, body = 200, [p for p in PETS if not tags or p["tag"] in tags]
            else:
                status, body = 404, {"error": f"{path}: not in this stand-in"}
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


def write_config(directory: Path, port: int) -> Path:
    base = f"http://127.0.0.1:{port}"
    cfg = {
        "mcpServers": {
            "time": {"command": "uvx", "args": ["mcp-server-time"]},
            "everything": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-everything"]},
        },
        "openapi": {
            "swagger-petstore": {"base_url": f"{base}/petstore/v2"},
            "stripe": {"base_url": f"{base}/stripe"},
            "github": {"base_url": f"{base}/github"},
        },
    }
    path = directory / "toolrank.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


@contextmanager
def served(data: Path, emb_url: str, heads: str | None, stand_in_port: int) -> Iterator[str]:
    """``toolrank serve`` on ``data`` with the week-1 servers and the stand-in; -> its URL, ready."""
    env = dict(os.environ)
    if heads:
        env["TOOLRANK_HEADS"] = str(Path(heads).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        cfg = write_config(Path(tmp), stand_in_port)
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        cmd = [str(Path(sys.executable).with_name("toolrank")), "serve", "--data", str(data)]
        cmd += ["--config", str(cfg), "--emb-url", emb_url, "--port", str(port)]
        log = Path(tmp) / "serve.log"
        with log.open("w") as f:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=f)
            try:
                tr = ToolrankClient(url)
                for _ in range(600):
                    try:
                        if tr.health().get("ready"):
                            break
                    except ToolrankError:
                        pass
                    if proc.poll() is not None:
                        sys.exit(f"serve exited: {log.read_text()[-2000:]}")
                    time.sleep(0.5)
                yield url
            finally:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.kill()


def load_env(path: str = ".env") -> None:
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return
    for line in lines:
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def offline(tr: ToolrankClient) -> dict[str, Any]:
    import jsonschema

    t0 = time.perf_counter()
    catalog = tr.catalog()
    out: dict[str, Any] = {"catalog_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
    tools = catalog["tools"]
    box = claude.Toolbox(tr)
    names = [t["api_name"] for t in tools]
    bad = []
    for t in tools:
        try:
            jsonschema.Draft202012Validator.check_schema(t["inputSchema"])
        except jsonschema.SchemaError as e:
            bad.append([t["name"], e.message[:120]])
    out["catalogue"] = {
        "tools": len(tools),
        "api_names_unique": len(set(names)) == len(names),
        "hashed_names": sum("___" in n for n in names),
        "anthropic_tools_bytes": len(json.dumps(box.tools).encode()),
        "schemas_failing_2020_12": bad,
    }
    searches = []
    for task in TASKS:
        found = tr.search(task, full_schemas=True)
        hits = found["tools"]
        loaded = [gpt.function_tool(h) for h in hits]
        searches.append(
            {
                "task": task,
                "took_ms": found["took_ms"],
                "tools": [h["name"] for h in hits],
                "anthropic_references": [h["api_name"] for h in hits if h["api_name"] in box.entries][:10],
                "openai_loaded_bytes": len(json.dumps(loaded).encode()),
            }
        )
    out["searches"] = searches
    calls = {}
    for label, name, args in [
        ("pets", "swagger-petstore/findPets", {"tags": ["dog"]}),
        ("stripe_refund", "stripe/PostRefunds", {"charge": "ch_123"}),
    ]:
        r = tr.call(name, args)
        calls[label] = [r["outcome"], r["isError"], (r["content"] or [{}])[0].get("text", "")[:120]]
    out["calls"] = calls
    return out


def live(url: str, a: argparse.Namespace) -> list[dict[str, Any]]:
    import anthropic
    import openai

    claude_llm, gpt_llm = anthropic.Anthropic(), openai.OpenAI()

    def approve_all(entry: dict[str, Any], arguments: dict[str, Any]) -> bool:
        return True  # the proxy still refuses writes: this run has no --allow-write

    def claude_run(task: str, note: Any, session: str, builtin: str | None = None) -> Any:
        tr = ToolrankClient(url)
        box = claude.Toolbox(tr, builtin=builtin, approve=approve_all, on_event=note, session=session)
        return claude.run(claude_llm, box, task, model=a.claude_model, max_tokens=2048, max_turns=8)

    def claude_bm25(task: str, note: Any, session: str) -> Any:
        return claude_run(task, note, session, builtin="bm25")

    def gpt_run(task: str, note: Any, session: str) -> Any:
        box = gpt.Toolbox(ToolrankClient(url), approve=approve_all, on_event=note, session=session)
        return gpt.run(gpt_llm, box, task, model=a.gpt_model, max_turns=8)

    runs: list[dict[str, Any]] = []
    wanted = set(a.setups.split(","))
    setups = [
        (label, fn)
        for label, fn in [
            ("claude+toolrank", claude_run),
            ("claude+bm25", claude_bm25),
            ("gpt+toolrank", gpt_run),
        ]
        if label in wanted
    ]
    for n, task in enumerate(TASKS):
        for label, fn in setups:
            events: list[list[Any]] = []
            row: dict[str, Any] = {"setup": label, "task": task}
            t0 = time.perf_counter()
            try:
                result = fn(
                    task,
                    lambda kind, d, ev=events: kind != "turn" and ev.append([kind, d]),
                    f"e2e-{n}-{label}",
                )
                row.update(
                    text=result.text,
                    turns=result.turns,
                    stop=getattr(result, "stop_reason", None) or getattr(result, "status", None),
                    usage=result.usage,
                )
            except Exception as e:  # an API error is a result here
                row["error"] = f"{type(e).__name__}: {e}"[:500]
            row["seconds"] = round(time.perf_counter() - t0, 1)
            row["events"] = events
            runs.append(row)
            said = row.get("error") or (row.get("text") or "")[:80]
            print(f"{label:>16} | {task[:40]:40} | {row.get('turns')} turns {row['seconds']} s | {said!r}")
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--emb-url", required=True)
    p.add_argument("--heads", default=None)
    p.add_argument("--live", action="store_true")
    p.add_argument("--claude-model", default="claude-opus-5-5")
    p.add_argument("--gpt-model", default="gpt-5.5")
    p.add_argument(
        "--setups", default="claude+toolrank,claude+bm25,gpt+toolrank", help="which live setups to run"
    )
    p.add_argument("--out", default=None)
    a = p.parse_args()
    load_env()
    needed = {"ANTHROPIC_API_KEY": "claude", "OPENAI_API_KEY": "gpt"}
    missing = [k for k, prefix in needed.items() if prefix in a.setups and not os.environ.get(k)]
    if a.live and missing:
        sys.exit(f"--live needs {', '.join(missing)} (environment or .env)")
    data = Path(a.data).resolve()
    start = datetime.now(UTC).isoformat(timespec="milliseconds")
    stub = stand_in()
    report: dict[str, Any] = {"data": str(data), "emb_url": a.emb_url}
    with served(data, a.emb_url, a.heads, stub.server_address[1]) as url:
        report["offline"] = offline(ToolrankClient(url))
        print(json.dumps(report["offline"], indent=2, ensure_ascii=False))
        if a.live:
            report["live"] = live(url, a)
    stub.shutdown()
    events = [
        json.loads(line)
        for f in sorted((data / "usage").glob("usage-*.jsonl"))
        for line in f.read_text().splitlines()
    ]
    calls = [e for e in events if e["ts"] >= start and e["event"] == "call"]
    report["usage_calls"] = {
        "count": len(calls),
        "links": dict(Counter(e["link"] for e in calls)),
        "outcomes": dict(Counter(e["outcome"] for e in calls)),
    }
    print(json.dumps(report["usage_calls"]))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
