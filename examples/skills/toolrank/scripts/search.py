#!/usr/bin/env python3
"""Find the tools a step needs in toolrank's catalogue: search.py "what to do" [--k N] [--full].

Prints JSON: the search id and the matching tools, best first, each with its name, description and
input schema (shortened after the first few unless ``--full``).
"""

from __future__ import annotations

import argparse
import json

from _toolrank import post

KEEP = ("name", "description", "inputSchema", "inputSchemaShrunk", "annotations", "method", "used_with")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task", nargs="+", help="the step, in plain words")
    p.add_argument("--k", type=int, default=None, help="this many tools (default: as many as fit)")
    p.add_argument("--full", action="store_true", help="every input schema in full")
    a = p.parse_args(argv)
    body: dict = {"query": " ".join(a.task)}
    if a.k is not None:
        body["k"] = a.k
    if a.full:
        body["full_schemas"] = True
    found = post("/v1/search", body, timeout=60.0)
    out: dict = {"search_id": found.get("search_id")}
    if found.get("mode") == "lexical":
        out["note"] = "toolrank is still building its index: these are keyword matches"
    out["tools"] = [{k: t[k] for k in KEEP if k in t} for t in found.get("tools") or []]
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
