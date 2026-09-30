"""An OpenAI model with toolrank as its tool search (Responses API, ``tool_search`` run by the client).

The request declares one tool, ``tool_search`` with ``execution: "client"``. When the model emits a
``tool_search_call``, toolrank searches and the answer (``tool_search_output``) carries the found
tools' full definitions, so the catalogue never travels with the request. Calls to them run through
toolrank (``POST /v1/call``). See ``toolrank.integrations.openai`` for the rules the loop keeps.

    toolrank serve --data data/mytools --config toolrank.json          # in another shell
    uv run --extra openai python examples/openai_client_tool_search.py "What time is it in Tokyo?"

``OPENAI_API_KEY`` (and ``TOOLRANK_API_KEY`` when the server has one) come from the environment or
from a ``.env`` file in the working directory, and are never printed. A call that could change
something — no ``readOnlyHint``, or an OpenAPI operation other than GET/HEAD — is asked about first
unless ``--yes``: a tool's output can steer the model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_env(path: str = ".env") -> None:
    """``KEY=value`` lines into the environment, for keys not set already."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return
    for line in lines:
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    p = argparse.ArgumentParser(description="OpenAI + toolrank client-executed tool search")
    p.add_argument("task")
    p.add_argument("--toolrank", default=os.environ.get("TOOLRANK_URL", "http://127.0.0.1:8765"))
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--yes", action="store_true", help="run every call without asking")
    a = p.parse_args()
    load_env()

    import openai

    from toolrank.client import ToolrankClient
    from toolrank.integrations import openai as gpt
    from toolrank.integrations import read_only

    def approve(entry: dict, arguments: dict) -> bool:
        if a.yes or read_only(entry):
            return True
        answer = input(f"  run {entry['name']} {json.dumps(arguments, ensure_ascii=False)}? [y/N] ")
        return answer.strip().lower() in ("y", "yes")

    def show(kind: str, d: dict) -> None:
        if kind == "turn":
            print(f"turn {d['turn']}: {d['stop_reason']}")
        elif kind == "search":
            found = ", ".join(d.get("tools") or []) or d.get("error") or "nothing"
            print(
                f"  search {d.get('query')!r} -> {found}"
                + (f"  ({d['loaded']} new)" if "loaded" in d else "")
            )
        elif kind == "call":
            print(f"  call {d['tool']}: {d.get('outcome')}" + (f" ({d['ms']} ms)" if "ms" in d else ""))

    tr = ToolrankClient(a.toolrank, api_key=os.environ.get("TOOLRANK_API_KEY"))
    box = gpt.Toolbox(tr, approve=approve, on_event=show)
    print(f"toolrank at {a.toolrank}: {tr.health().get('tools')} tools; model {a.model}")
    try:
        result = gpt.run(openai.OpenAI(), box, a.task, model=a.model, max_turns=a.max_turns)
    except openai.APIError as e:
        print(f"Responses API error: {e}", file=sys.stderr)
        return 1
    print("\n" + (result.text or "(no text)"))
    print(f"\n{result.turns} requests, status: {result.status}, usage: {result.usage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
