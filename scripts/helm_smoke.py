"""Smoke-test the Helm chart (deploy/helm/toolrank) on a real cluster without a GPU.

The embedding backbone is the fake server of the container smoke test (tests/fixtures/
fake_embeddings.py), run in the cluster with the toolrank image's Python; the chart runs in
``embedding.mode=external`` against it. The script installs the chart, waits for the pod to be
ready (the ingest init container has listed the config's MCP server, the index is built), then
through a port-forward: a search, a call through ``call_tool``'s REST twin, the metrics, a
rejected token; ``helm upgrade`` with a second MCP server in the config, which must show up in the
catalogue; and ``helm uninstall``, after which the data volume must still be there.

Needs kubectl and helm pointed at a cluster (``kind create cluster --name toolrank`` gives one on a
laptop: Docker only, removed with ``kind delete cluster --name toolrank``). Nothing is published.

uv run python scripts/helm_smoke.py [--image ghcr.io/yasinyaman/toolrank:0.1.0] [--namespace toolrank-smoke] [--keep]
"""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "deploy" / "helm" / "toolrank"
FAKE = ROOT / "tests" / "fixtures" / "fake_embeddings.py"
RELEASE = "smoke"


def run(*args: str, check: bool = True, quiet: bool = False) -> str:
    if not quiet:
        print("+", " ".join(args), flush=True)
    done = subprocess.run(args, capture_output=True, text=True)
    if check and done.returncode != 0:
        sys.exit(f"failed: {' '.join(args)}\n{done.stdout}\n{done.stderr}")
    return done.stdout


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def request(url: str, key: str | None, body: dict | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def fake_embedding(ns: str, image: str) -> str:
    """The fake server as a Deployment + Service; -> its /v1 URL."""
    manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata: {{name: fake-embedding, namespace: {ns}}}
spec:
  replicas: 1
  selector: {{matchLabels: {{app: fake-embedding}}}}
  template:
    metadata: {{labels: {{app: fake-embedding}}}}
    spec:
      containers:
        - name: fake
          image: {image}
          command: [python, /fake/fake_embeddings.py, --host, 0.0.0.0, --port, "8000"]
          ports: [{{containerPort: 8000}}]
          readinessProbe: {{tcpSocket: {{port: 8000}}, periodSeconds: 2}}
          volumeMounts: [{{name: fake, mountPath: /fake}}]
      volumes: [{{name: fake, configMap: {{name: fake-embedding}}}}]
---
apiVersion: v1
kind: Service
metadata: {{name: fake-embedding, namespace: {ns}}}
spec:
  selector: {{app: fake-embedding}}
  ports: [{{port: 8000, targetPort: 8000}}]
"""
    run("kubectl", "-n", ns, "create", "configmap", "fake-embedding", f"--from-file={FAKE}")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(manifest)
    run("kubectl", "apply", "-f", f.name)
    run("kubectl", "-n", ns, "rollout", "status", "deploy/fake-embedding", "--timeout=300s")
    return f"http://fake-embedding.{ns}.svc:8000/v1"


def helm_values(path: Path, url: str, key: str, image: str, servers: dict, team: str | None = None) -> None:
    repo, _, tag = image.rpartition(":")
    auth: dict = {"apiKey": key}
    if team:  # a tenant limited to the time server
        auth["apiKeys"] = {"team": {"key": team, "sources": ["time"]}}
    path.write_text(
        json.dumps(
            {
                "image": {"repository": repo, "tag": tag, "pullPolicy": "IfNotPresent"},
                "embedding": {"mode": "external", "url": url, "model": "fake"},
                "config": {"mcpServers": servers},
                "auth": auth,
                "persistence": {"size": "1Gi"},
                "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}},
            }
        )
    )


def wait_ready(ns: str, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = run(
            "kubectl", "-n", ns, "get", "pods", "-l", "app.kubernetes.io/component=server", "-o", "json",
            quiet=True,
        )  # fmt: skip
        pods = json.loads(out)["items"]
        live = [p for p in pods if not p["metadata"].get("deletionTimestamp")]
        if len(live) == 1:
            conds = {c["type"]: c["status"] for c in live[0]["status"].get("conditions", [])}
            if conds.get("Ready") == "True":
                return
        time.sleep(5)
    print(run("kubectl", "-n", ns, "describe", "pods", check=False))
    print(
        run(
            "kubectl",
            "-n",
            ns,
            "logs",
            "-l",
            "app.kubernetes.io/component=server",
            "--all-containers",
            check=False,
        )
    )
    sys.exit("the toolrank pod did not become ready")


def port_forward(ns: str) -> tuple[subprocess.Popen, str]:
    port = free_port()
    proc = subprocess.Popen(
        ["kubectl", "-n", ns, "port-forward", f"svc/{RELEASE}-toolrank", f"{port}:8765"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/healthz", timeout=2)
            return proc, base
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    proc.kill()
    sys.exit("port-forward did not come up")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--image", default="ghcr.io/yasinyaman/toolrank:0.1.0")
    ap.add_argument("--namespace", default="toolrank-smoke")
    ap.add_argument("--keep", action="store_true", help="leave the namespace and release in place")
    ap.add_argument(
        "--tenants", action="store_true", help="also a key limited to one source (images after 0.1.0)"
    )
    a = ap.parse_args()
    ns, key, checks = a.namespace, secrets.token_hex(16), []
    team = secrets.token_hex(16) if a.tenants else None

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok))
        print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""), flush=True)

    run("kubectl", "create", "namespace", ns)
    try:
        url = fake_embedding(ns, a.image)
        values = Path(tempfile.mkdtemp()) / "values.json"
        time_server = {"time": {"command": "uvx", "args": ["mcp-server-time"]}}
        helm_values(values, url, key, a.image, time_server, team)
        t0 = time.time()
        run("helm", "install", RELEASE, str(CHART), "-n", ns, "-f", str(values))
        wait_ready(ns)
        check("install: the pod is ready (ingest, index)", True)
        print(f"  ready after {time.time() - t0:.0f} s")
        proc, base = port_forward(ns)
        try:
            status, body = request(base + "/v1/search", key, {"query": "what time is it in Tokyo"})
            names = [t["name"] for t in json.loads(body).get("tools", [])] if status == 200 else []
            check(
                "search answers with the time server's tools",
                status == 200 and any(n.startswith("time/") for n in names),
                body[:300],
            )
            status, body = request(
                base + "/v1/call",
                key,
                {"name": "time/get_current_time", "arguments": {"timezone": "Asia/Tokyo"}},
            )
            check(
                "a call reaches the stdio MCP server inside the pod",
                status == 200 and json.loads(body).get("outcome") == "ok",
                body[:300],
            )
            status, body = request(base + "/v1/search", "wrong-key", {"query": "time"})
            check("a wrong token is refused", status == 401, str(status))
            status, body = request(base + "/v1/metrics", key)
            if status == 404:  # an image older than the metrics route
                print("  (no /v1/metrics in this image)")
            else:
                check(
                    "metrics count the search",
                    status == 200 and "toolrank_searches_total{" in body,
                    body[:200],
                )
        finally:
            proc.kill()

        fetch = {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}
        helm_values(values, url, key, a.image, {**time_server, **fetch}, team)
        run("helm", "upgrade", RELEASE, str(CHART), "-n", ns, "-f", str(values))
        time.sleep(5)  # the config checksum rolls the pod (Recreate)
        wait_ready(ns)
        proc, base = port_forward(ns)
        try:
            status, body = request(base + "/v1/tools", key)
            servers = {t["server"] for t in json.loads(body).get("tools", [])} if status == 200 else set()
            check("upgrade: the new server is in the catalogue", servers == {"time", "fetch"}, str(servers))
            if team:
                status, body = request(base + "/v1/tools", team)
                mine = {t["server"] for t in json.loads(body).get("tools", [])} if status == 200 else set()
                check("a key limited to one source sees only that source", mine == {"time"}, str(mine))
                status, body = request(
                    base + "/v1/call",
                    team,
                    {"name": "fetch/fetch", "arguments": {"url": "https://example.com"}},
                )
                check("... and cannot call another source's tool", status == 404, f"{status} {body[:200]}")
                status, _ = request(base + "/v1/metrics", team)
                check("... nor read the server-wide metrics", status == 403, str(status))
        finally:
            proc.kill()

        run("helm", "uninstall", RELEASE, "-n", ns)
        pvcs = run("kubectl", "-n", ns, "get", "pvc", "-o", "name")
        check(
            "uninstall keeps the data volume", f"persistentvolumeclaim/{RELEASE}-toolrank-data" in pvcs, pvcs
        )
    finally:
        if not a.keep:
            run("kubectl", "delete", "namespace", ns, "--wait=false", check=False)
    failed = [n for n, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
