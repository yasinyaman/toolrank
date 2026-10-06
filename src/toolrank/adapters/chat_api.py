"""A ``ChatModel`` over any OpenAI-compatible ``/v1/chat/completions`` endpoint.

Used to generate benchmark queries (MCP-Zero) with a local model, e.g. ``vllm serve
Qwen/Qwen3-8B`` in the default (generate) runner. Greedy by default so a rerun asks the same
questions. ``extra_body`` is merged into every request for server-specific switches, such as
``{"chat_template_kwargs": {"enable_thinking": False}}`` to turn off Qwen3's thinking on vLLM.

Standard library only (``urllib``), like the embeddings adapter.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from typing import Any

from toolrank.adapters.embeddings_api import env_api_key


class OpenAIChat:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8093/v1",
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or env_api_key(self.base_url, "TOOLRANK_CHAT_API_KEY")
        self.temperature, self.max_tokens = temperature, max_tokens
        self.extra_body = dict(extra_body or {})
        self.timeout, self.max_retries = timeout, max_retries
        self.name = f"chat/{model}"

    def complete(self, system: str, user: str) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:  # never copied onto a redirect, which may point anywhere
            req.add_unredirected_header("Authorization", f"Bearer {self.api_key}")
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    j = json.loads(r.read().decode("utf-8"))
                break
            # OSError: URLError, HTTPError, timeouts, and a keep-alive the server dropped (RemoteDisconnected)
            except (OSError, http.client.HTTPException) as e:  # noqa: PERF203
                last = e
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"chat endpoint {self.base_url} unreachable: {last}") from last
        return str(j["choices"][0]["message"].get("content") or "")
