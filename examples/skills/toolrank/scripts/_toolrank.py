"""What the toolrank skill's scripts share: one JSON request to ``toolrank serve``'s REST API.

Standard library only, so the skill's folder works wherever it is copied. Redirects are not
followed (the bearer token never reaches another host); a failure prints why on stderr and exits 2.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, NoReturn

USER_AGENT = "toolrank-skill/1"  # the client key that links a call to this client's latest search


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # the 3xx comes back as an HTTPError


_OPENER = urllib.request.build_opener(_NoRedirect)


def fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def post(path: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    url = (os.environ.get("TOOLRANK_URL") or "http://127.0.0.1:8765").rstrip("/") + path
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
    if os.environ.get("TOOLRANK_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["TOOLRANK_API_KEY"]
    if os.environ.get("TOOLRANK_SESSION"):
        headers["X-Session-Id"] = os.environ["TOOLRANK_SESSION"]
    req = urllib.request.Request(url, json.dumps(body).encode("utf-8"), headers, method="POST")
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            payload = json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            message = json.loads(raw).get("error")
        except (ValueError, AttributeError):
            message = None
        fail(f"toolrank {e.code}: {message or raw[:300].decode('utf-8', 'replace') or e.reason}")
    except (urllib.error.URLError, OSError) as e:  # refused, timed out, no such host
        fail(f"toolrank at {url} unreachable: {getattr(e, 'reason', e)} (is `toolrank serve` running?)")
    except ValueError:
        fail(f"toolrank at {url} answered with something that is not JSON")
    if not isinstance(payload, dict):
        fail(f"toolrank at {url} answered with something that is not a JSON object")
    return payload
