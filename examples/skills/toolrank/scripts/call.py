#!/usr/bin/env python3
"""Run a tool through toolrank: call.py NAME '{"arg": "value"}' [--search-id ID].

The arguments are a JSON object, or ``-`` to read it from stdin. Prints the tool's output; exits 1
when the tool reported an error, 2 when toolrank refused the call or could not be reached.
"""

from __future__ import annotations

import argparse
import json
import sys

from _toolrank import fail, post

OUTPUT_CHARS = 25_000  # printed per call, like toolrank's own integrations


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("name", help="the tool's name, as search.py printed it")
    p.add_argument("arguments", nargs="?", default="{}", help="a JSON object, or - for stdin")
    p.add_argument("--search-id", default=None, help="the search that found the tool")
    a = p.parse_args(argv)
    raw = sys.stdin.read() if a.arguments == "-" else a.arguments
    try:
        arguments = json.loads(raw)
    except ValueError as e:
        fail(f"the arguments are not JSON: {e}")
    if not isinstance(arguments, dict):
        fail("the arguments must be one JSON object")
    body: dict = {"name": a.name, "arguments": arguments}
    if a.search_id:
        body["search_id"] = a.search_id
    out = post("/v1/call", body, timeout=150.0)
    parts = [
        c["text"] if c.get("type") == "text" else json.dumps(c, ensure_ascii=False)
        for c in out.get("content") or []
        if isinstance(c, dict)
    ]
    text = "\n".join(p for p in parts if p)
    if not text and out.get("structuredContent") is not None:
        text = json.dumps(out["structuredContent"], ensure_ascii=False)
    if len(text) > OUTPUT_CHARS:
        text = text[:OUTPUT_CHARS] + f"\n[truncated: {len(text)} characters, {OUTPUT_CHARS} shown]"
    print(text or "(the tool returned nothing)")
    if out.get("isError"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
