"""toolrank as the tool search of OpenAI's Responses API (``tool_search``, client execution).

The request declares a single tool, ``{"type": "tool_search", "execution": "client"}``, with
toolrank's description and argument schema. When the model needs tools it emits a
``tool_search_call`` and stops; toolrank searches, and the client answers with a
``tool_search_output`` carrying the found tools' full definitions (``function`` tools,
``defer_loading: true``). Nothing is declared up front, so the catalogue never travels with the
request. The model then calls the tools (``function_call``) and toolrank runs them
(``/v1/call``). Client-executed tool search is GA (March 2026) on gpt-5.4 and later, except
gpt-5.4-nano and gpt-5.5-pro.

The loop keeps the API's rules:
- stateless: ``store=False`` with ``include=["reasoning.encrypted_content"]``, the whole item
  list sent every turn (``items += response.output``; reasoning items stay next to their calls)
  with the same ``tools``; loaded tools live in the ``tool_search_output`` items, so each is sent
  once per conversation (a later search that finds it again leaves it out);
- every ``tool_search_call`` and ``function_call`` of a response is answered, by ``call_id``, in
  order; a ``failed`` response raises, an ``incomplete`` one ends the run without running anything;
- function names are toolrank's api names, which match the API's ``^[a-zA-Z0-9_-]+$``.

    import openai
    from toolrank.client import ToolrankClient
    from toolrank.integrations import openai as tr

    result = tr.run(openai.OpenAI(), ToolrankClient(), "What time is it in Tokyo?", model="gpt-5.5")
    print(result.text)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from toolrank.client import ToolrankClient, ToolrankError
from toolrank.integrations._common import Approve, OnEvent, as_text, get

MAX_TOOLS = 10  # per search


def search_tool() -> dict[str, Any]:
    return {
        "type": "tool_search",
        "execution": "client",
        "description": (
            "Search the tool catalogue for the tools a task needs. Describe the task, or the step you "
            "are about to take, in plain words; the matching tools are then loaded for you to call. "
            "If none fits, search again in other words."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you want to do, in plain words."},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TOOLS,
                    "description": "How many tools at most; default: as many as the request needs.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }


def function_tool(hit: dict[str, Any]) -> dict[str, Any]:
    """A toolrank search hit (``full_schemas``) as a deferred ``function`` tool."""
    parameters = dict(hit.get("inputSchema") or {})
    parameters.pop("$schema", None)
    parameters.setdefault("type", "object")
    return {
        "type": "function",
        "name": hit["api_name"],
        "description": hit.get("description") or "",
        "parameters": parameters,
        "strict": False,
        "defer_loading": True,
    }


class Toolbox:
    """One conversation's tools: what toolrank found and loaded, and the calls it ran.

    ``respond(response)`` answers a response's tool searches and calls; ``approve(entry,
    arguments)`` may veto a call; ``on_event(kind, details)`` sees every search, call and turn."""

    def __init__(
        self,
        toolrank: ToolrankClient,
        *,
        approve: Approve | None = None,
        on_event: OnEvent | None = None,
        session: str | None = None,
    ):
        self.toolrank, self.approve = toolrank, approve
        self.on_event: OnEvent = on_event or (lambda kind, details: None)
        self.session = session or f"openai-{uuid.uuid4().hex[:12]}"
        self.tools: list[dict[str, Any]] = [search_tool()]
        self.loaded: dict[str, dict[str, Any]] = {}  # api name -> the search hit sent for it
        self.found: dict[str, str] = {}  # api name -> the latest search that returned it

    def respond(self, response: Any) -> list[dict[str, Any]]:
        """The items that answer ``response``'s tool searches and calls, in order; [] when it asks
        for none or is ``incomplete``."""
        status = get(response, "status")
        if status == "failed":
            raise RuntimeError(f"the response failed: {get(response, 'error')}")
        if status == "incomplete":
            return []
        out: list[dict[str, Any]] = []
        for item in get(response, "output") or []:
            kind = get(item, "type")
            if kind == "tool_search_call" and get(item, "execution") == "client":
                out.append(self._search(item))
            elif kind == "function_call":
                out.append(self._call(item))
        return out

    def _search(self, item: Any) -> dict[str, Any]:
        args = get(item, "arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        query = str(args.get("query") or "").strip() if isinstance(args, dict) else ""
        limit = args.get("limit") if isinstance(args, dict) else None
        k = None
        if isinstance(limit, int | float) and not isinstance(limit, bool):
            k = max(1, min(int(limit), MAX_TOOLS))
        new: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        try:
            found = self.toolrank.search(query, k=k, full_schemas=True, session=self.session)
        except ToolrankError as e:  # the output item has no error channel: nothing is loaded
            self.on_event("search", {"query": query, "error": e.message})
        else:
            for hit in found.get("tools", [])[:MAX_TOOLS]:
                name = hit["api_name"]
                self.found[name] = found["search_id"]
                if name not in self.loaded:
                    self.loaded[name] = hit
                    new.append(function_tool(hit))
            self.on_event(
                "search",
                {
                    "query": query,
                    "tools": [h["name"] for h in found.get("tools", [])[:MAX_TOOLS]],
                    "loaded": len(new),
                    "mode": found.get("mode"),
                    "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                },
            )
        return {
            "type": "tool_search_output",
            "execution": "client",
            "call_id": get(item, "call_id"),
            "status": "completed",
            "tools": new,
        }

    def _call(self, item: Any) -> dict[str, Any]:
        call_id, name = get(item, "call_id"), get(item, "name")
        entry = self.loaded.get(name)
        if entry is None:
            return _output(call_id, f"Error: no loaded tool is named {name!r}; search for it first.")
        raw = get(item, "arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError as e:
            return _output(call_id, f"Error: the arguments are not valid JSON ({e}).")
        if not isinstance(args, dict):
            return _output(call_id, "Error: the arguments must be a JSON object.")
        if self.approve is not None and not self.approve(entry, args):
            self.on_event("call", {"tool": entry["name"], "outcome": "declined"})
            return _output(call_id, "Error: the user declined this call.")
        t0 = time.perf_counter()
        try:
            out = self.toolrank.call(
                entry["name"], args, search_id=self.found.get(name), session=self.session
            )
        except ToolrankError as e:
            self.on_event("call", {"tool": entry["name"], "outcome": "error", "error": e.message})
            return _output(call_id, f"Error: toolrank could not run {entry['name']}: {e.message}")
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self.on_event("call", {"tool": entry["name"], "outcome": out.get("outcome"), "ms": ms})
        text = as_text(out.get("content") or []) or "(the tool returned nothing)"
        return _output(call_id, f"Error: {text}" if out.get("isError") else text)


def _output(call_id: str, text: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": text}


@dataclass
class Result:
    text: str  # the last response's output text
    status: str | None
    turns: int
    items: list[Any] = field(repr=False)
    usage: dict[str, int] = field(default_factory=dict)


def run(
    llm: Any, toolrank: ToolrankClient | Toolbox, task: str, *, model: str, max_turns: int = 12, **create: Any
) -> Result:
    """A reference loop: ``llm`` is an ``openai.OpenAI()``; ``create`` goes to every
    ``responses.create``. Stops when a response has nothing to answer, is incomplete, or after
    ``max_turns`` requests."""
    box = toolrank if isinstance(toolrank, Toolbox) else Toolbox(toolrank)
    items: list[Any] = [{"role": "user", "content": task}]
    usage: dict[str, int] = {}
    response: Any = None
    turn = 0
    for turn in range(1, max_turns + 1):
        response = llm.responses.create(
            model=model,
            input=items,
            tools=box.tools,
            store=False,
            include=["reasoning.encrypted_content"],
            **create,
        )
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = get(get(response, "usage"), key)
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
        items += list(get(response, "output") or [])
        box.on_event("turn", {"turn": turn, "stop_reason": get(response, "status")})
        replies = box.respond(response)
        if not replies:
            break
        items += replies
    return Result(_text(response), get(response, "status"), turn, items, usage)


def _text(response: Any) -> str:
    text = get(response, "output_text")
    if isinstance(text, str):
        return text
    parts = []
    for item in get(response, "output") or []:
        if get(item, "type") == "message":
            parts += [
                get(c, "text") or "" for c in get(item, "content") or [] if get(c, "type") == "output_text"
            ]
    return "".join(parts)
