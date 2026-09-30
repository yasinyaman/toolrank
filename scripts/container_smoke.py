"""Smoke-test the toolrank image without a GPU: a fake embedding server on this machine
(tests/fixtures/fake_embeddings.py), the image ingesting a small OpenAPI spec and serving it, then
the checks a user would make: healthy, a search through the published port, the Host check, an MCP
initialize, the runtimes stdio MCP servers need (node, npx, uvx) and the version. Everything it
starts is removed at the end.

``--bundle`` is for the toolrank-vllm image: the fake server goes inside the container as its
``vllm`` (first on the PATH), so the image's own entrypoint starts it, waits for it and runs
toolrank, and four more checks see who runs what: a bind-mounted directory keeps its owner, vLLM
runs as root, toolrank and its files as the ``toolrank`` user, and a file root left on the volume is
handed over; the runtimes are checked as that user.

    uv run python scripts/container_smoke.py toolrank:dev [--version 0.1.0]
    uv run python scripts/container_smoke.py toolrank-vllm:dev --bundle
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
from fake_embeddings import serve  # noqa: E402

SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Billing API", "version": "1"},
    "servers": [{"url": "https://billing.example.com"}],
    "paths": {
        "/invoices": {"get": {"operationId": "listInvoices", "summary": "List the invoices of a customer"}},
        "/refunds": {"post": {"operationId": "createRefund", "summary": "Refund a payment"}},
    },
}
KEY = "smoke-test-key"
# `vllm serve MODEL ... --port N`, as the entrypoint calls it -> the fake embedding server on that port
FAKE_VLLM = """#!/bin/sh
port=8091
while [ $# -gt 0 ]; do [ "$1" = --port ] && port=$2; shift; done
exec python3 /fake/fake_embeddings.py --host 127.0.0.1 --port "$port"
"""
# the runtimes stdio MCP servers need, and a uv cache their user can write
RUNTIMES = 'd=$(uv cache dir) && mkdir -p "$d" && touch "$d/.smoke" && node --version && npx --version && uvx --version'
# user and command line of every process in the container (the image may have no ps)
PROCESSES = 'for p in /proc/[0-9]*; do echo "$(stat -c %U $p) $(tr "\\0" " " < $p/cmdline)"; done'


def docker(*args: str, check: bool = True) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check).stdout.strip()


def request(url: str, body: dict | None = None, host: str | None = None) -> tuple[int, dict | str]:
    headers = {"Authorization": f"Bearer {KEY}", "Accept": "application/json, text/event-stream"}
    if host:
        headers["Host"] = host
    data = None
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
            return r.status, json.loads(raw) if raw.startswith("{") else raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("image")
    p.add_argument("--version", default=None, help="the toolrank version the image must report")
    p.add_argument("--port", type=int, default=18765, help="published on 127.0.0.1")
    p.add_argument("--bundle", action="store_true", help="the toolrank-vllm image, with a fake vllm inside")
    a = p.parse_args()
    tag = uuid.uuid4().hex[:8]
    volume, name = f"toolrank-smoke-{tag}", f"toolrank-smoke-{tag}"
    fake, logs = None, ""
    # what the container's user (not this one) must read: 0755 / 0644 whatever the umask
    shared = Path(tempfile.mkdtemp(prefix="toolrank-smoke-"))
    checks: list[tuple[str, bool, str]] = []
    try:
        (shared / "specs").mkdir()
        (shared / "data").mkdir()
        (shared / "specs" / "billing.json").write_text(json.dumps(SPEC))
        if a.bundle:
            (shared / "fake").mkdir()
            shutil.copy(ROOT / "tests" / "fixtures" / "fake_embeddings.py", shared / "fake")
            (shared / "fake" / "vllm").write_text(FAKE_VLLM)
            path = docker("run", "--rm", "--entrypoint", "sh", a.image, "-c", "echo $PATH")
            run = ["-e", f"PATH=/fake:{path}", "-v", f"{shared / 'fake'}:/fake:ro"]
        else:
            fake, _ = serve("0.0.0.0")
            emb = f"http://host.docker.internal:{fake.server_address[1]}/v1"
            env = ["-e", f"TOOLRANK_EMB_URL={emb}", "-e", "TOOLRANK_EMB_MODEL=fake"]
            run = ["--add-host", "host.docker.internal:host-gateway", *env]
        run += ["-e", f"TOOLRANK_API_KEY={KEY}", "-v", f"{shared / 'specs'}:/specs:ro"]
        data = ["-v", f"{volume}:/data"]
        for f in shared.rglob("*"):
            f.chmod(0o755 if f.is_dir() or f.name == "vllm" else 0o644)
        shared.chmod(0o755)
        version = docker("run", "--rm", a.image, "--version")
        checks.append(("version", a.version is None or version == f"toolrank {a.version}", version))
        ingest = ["ingest", "openapi", "/specs/billing.json", "--name", "billing", "--out", "/data"]
        out = docker("run", "--rm", *run, *data, a.image, *ingest)
        checks.append(("ingest + warm-up", "embedded 2 new or changed texts" in out, out.splitlines()[-1]))
        if a.bundle:  # this user's directory, bind-mounted: toolrank runs as its owner and it stays theirs
            docker("run", "--rm", *run, "-v", f"{shared / 'data'}:/data", a.image, *ingest)
            mine = [shared / "data", *(shared / "data").rglob("*")]
            owners = sorted({f.stat().st_uid for f in mine})
            kept = owners == [os.getuid()] and (shared / "data" / "tools.jsonl").exists()
            checks.append(("a bind mount keeps its owner", kept, f"{len(mine)} entries, uids {owners}"))
        docker("run", "-d", "--name", name, *run, *data, "-p", f"127.0.0.1:{a.port}:8765", a.image)
        base = f"http://127.0.0.1:{a.port}"
        for _ in range(120):
            status = docker("inspect", "-f", "{{.State.Health.Status}}", name, check=False)
            if status in ("healthy", "unhealthy"):
                break
            time.sleep(1)
        checks.append(("healthy", status == "healthy", status))
        code, found = request(f"{base}/v1/search", {"query": "refund a payment"})
        tools = [t["name"] for t in found["tools"]] if isinstance(found, dict) else found
        checks.append(("search on localhost", code == 200 and "billing/createRefund" in tools, str(tools)))
        code, _ = request(f"{base}/v1/tools", host="evil.example.com")
        checks.append(("foreign Host refused", code == 421, str(code)))
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "smoke", "version": "0"},
            },
        }
        code, _ = request(f"{base}/mcp", init)
        checks.append(("MCP initialize", code == 200, str(code)))
        # as the user toolrank runs as (the bundle's `docker exec` is root), with a cache uv can write
        runtimes = docker(
            "exec", name, *(["as-toolrank"] if a.bundle else []), "sh", "-c", RUNTIMES, check=False
        )
        checks.append(("node, npx, uvx", len(runtimes.splitlines()) == 3, " | ".join(runtimes.splitlines())))
        health = request(f"{base}/healthz")[1]
        checks.append(("scorer", isinstance(health, dict) and health.get("mode") == "semantic", str(health)))
        if a.bundle:
            procs = docker("exec", name, "sh", "-c", PROCESSES, check=False).splitlines()
            who = {
                label: sorted({line.split()[0] for line in procs if marker in line})
                for label, marker in (("vllm", "fake_embeddings.py"), ("toolrank", "toolrank serve"))
            }
            checks.append(
                (
                    "vLLM as root, toolrank as its user",
                    who == {"vllm": ["root"], "toolrank": ["toolrank"]},
                    str(who),
                )
            )
            # what a root `docker exec` leaves on the volume goes to that user with the next toolrank command
            docker("exec", name, "sh", "-c", "echo x > /data/by-root", check=False)
            docker("exec", name, "toolrank", "--version", check=False)
            stray = docker("exec", name, "stat", "-c", "%U", "/data/by-root", check=False)
            checks.append(("root's stray file is handed over", stray == "toolrank", stray))
            owners = docker(
                "exec", name, "sh", "-c", "find /data -printf '%u\\n' | sort | uniq -c", check=False
            )
            checks.append(
                ("/data belongs to toolrank", owners.split()[1:] == ["toolrank"], " ".join(owners.split()))
            )
    finally:
        if docker("ps", "-aq", "-f", f"name={name}", check=False):
            logs = docker("logs", name, check=False)
        docker("rm", "-f", name, check=False)
        docker("volume", "rm", "-f", volume, check=False)
        if fake is not None:
            fake.shutdown()
        shutil.rmtree(shared, ignore_errors=True)
    for label, ok, detail in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {detail[:160]}")
    if not all(ok for _, ok, _ in checks):
        print(logs[-3000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
