"""An OpenAI-compatible embedding server for tests without a GPU: each text gets a fixed random unit
vector seeded by its sha256 (4096-d, the packaged heads' input width), so rankings are stable but
mean nothing. Standard library only; `serve()` runs it in a thread and returns (server, url).

    python tests/fixtures/fake_embeddings.py --port 8099      # or run it by itself
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

DIM = 4096


def vector(text: str, dim: int = DIM) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("/health", "/v1/models"):
            self._send(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/embeddings":
            self._send(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        texts = body.get("input") or []
        texts = [texts] if isinstance(texts, str) else texts
        rows = []
        for n, text in enumerate(texts):
            v = vector(text)
            emb = (
                base64.b64encode(v.tobytes()).decode()
                if body.get("encoding_format") == "base64"
                else v.tolist()
            )
            rows.append({"object": "embedding", "index": n, "embedding": emb})
        tokens = sum(len(t.split()) for t in texts)
        self._send(
            200,
            {"object": "list", "data": rows, "model": body.get("model"), "usage": {"prompt_tokens": tokens}},
        )

    def log_message(self, *args: Any) -> None:
        pass


def serve(host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://{host}:{server.server_address[1]}/v1"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    a = p.parse_args()
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"fake embeddings at http://{a.host}:{a.port}/v1")
    server.serve_forever()
