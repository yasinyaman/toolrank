"""toolrank as the tool search of Claude's Messages API.

Every catalogue tool goes to the API with ``defer_loading: true``, so none of them takes context;
one ordinary tool, ``search_tools``, does not. When Claude calls it, toolrank searches, and the
``tool_result`` names the tools found in ``tool_reference`` blocks; the API expands those into the
full definitions, which Claude can call in this turn and later ones, and toolrank runs the calls
(``/v1/call``). This is the Messages API's tool search, GA with no beta header, on Claude Opus 4.5
and later, Sonnet 4.5 and later, and Haiku 4.5.

The API's rules the ``Toolbox`` keeps:
- at most 10,000 deferred tools per request, names ``^[a-zA-Z0-9_-]{1,128}$`` (toolrank's api
  names fit), and at least one tool not deferred (the search tool);
- a reference to a name the request does not declare fails the whole request, so references only
  name tools of the catalogue snapshot sent;
- the tool list is taken once and sent unchanged every turn: loaded references and thinking
  blocks belong to the exact history, which is only ever appended to;
- every ``tool_use`` of a response is answered, in order, in one user message; server tool uses
  are the API's to answer; nothing runs after ``max_tokens`` or ``refusal``.

Deferred definitions stay out of the prompt cache's prefix but still travel with every request:
data/w3's 1,862 tools are 3.7 MB. ``builtin="bm25"`` swaps ``search_tools`` for the API's own
``tool_search_tool_bm25`` over the same deferred tools, for comparison.

``Toolbox(inline=True)`` sends no catalogue at all (beta ``inline-tools-2026-09-15``, Claude Opus
4.8 and later): ``tools`` is ``search_tools`` alone, and the tools a search finds come by value, in
``tool_addition`` blocks of a ``role: "system"`` message appended right after the tool results,
once per conversation (the search's first three with full input schemas, the rest shortened, as
over MCP). No 10,000-tool limit, nothing per request but the conversation. With ``prefetch=True``
the task itself is searched before the first request, and its tools are added after the first user
message, so Claude can call them without a search turn. ``run`` sends the beta header.

    import anthropic
    from toolrank.client import ToolrankClient
    from toolrank.integrations import anthropic as tr

    result = tr.run(anthropic.Anthropic(), ToolrankClient(), "What time is it in Tokyo?",
                    model="claude-opus-5-5")
    print(result.text)
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from toolrank.client import ToolrankClient, ToolrankError
from toolrank.integrations._common import SHRUNK_NOTE, Approve, OnEvent, capped, get

SEARCH_TOOL = "search_tools"
INLINE_BETA = "inline-tools-2026-09-15"
BUILTIN = {"bm25": "tool_search_tool_bm25_20251119", "regex": "tool_search_tool_regex_20251119"}
MAX_DEFERRED = 10_000
MAX_REFERENCES = 10
IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")


def _result(tool_use_id: str, content: Any, *, is_error: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        out["is_error"] = True
    return out


def blocks(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MCP content as Messages API blocks: text, images as base64 image blocks, the rest as JSON
    text; empty text dropped (the API rejects it) and the text capped."""
    out: list[dict[str, Any]] = []
    for c in content:
        if c.get("type") == "image" and c.get("mimeType") in IMAGE_TYPES and c.get("data"):
            source = {"type": "base64", "media_type": c["mimeType"], "data": c["data"]}
            out.append({"type": "image", "source": source})
            continue
        text = c.get("text") if c.get("type") == "text" else json.dumps(c, ensure_ascii=False)
        if text:
            out.append({"type": "text", "text": capped(text)})
    return out or [{"type": "text", "text": "(the tool returned nothing)"}]


class Toolbox:
    """One conversation's tools: the frozen tool list for the API, and what toolrank found and ran.

    ``respond(response)`` answers a response's tool calls; ``approve(entry, arguments)`` may veto a
    call; ``on_event(kind, details)`` sees every search, call and turn. With ``inline`` (see the
    module's docstring) ``opening(task)`` gives the first messages and ``additions()`` the system
    message to append after each answer."""

    def __init__(
        self,
        toolrank: ToolrankClient,
        *,
        servers: Sequence[str] | None = None,
        builtin: str | None = None,
        approve: Approve | None = None,
        on_event: OnEvent | None = None,
        session: str | None = None,
        inline: bool = False,
        prefetch: bool = False,
    ):
        if builtin is not None and builtin not in BUILTIN:
            raise ValueError(f"builtin must be one of {sorted(BUILTIN)}")
        if inline and builtin is not None:
            raise ValueError("builtin searches the deferred catalogue, which inline=True does not send")
        if prefetch and not inline:
            raise ValueError("prefetch adds tools by value: it needs inline=True")
        self.inline, self.prefetch = inline, prefetch
        self.servers = frozenset(servers) if servers else None
        catalog = {"tools": [], "catalog": None} if inline else toolrank.catalog()
        entries = [t for t in catalog["tools"] if self.servers is None or t["server"] in self.servers]
        if len(entries) > MAX_DEFERRED:
            raise ValueError(
                f"{len(entries)} tools, but a request defers at most {MAX_DEFERRED}: pass servers=[...]"
            )
        self.toolrank, self.builtin, self.approve = toolrank, builtin, approve
        self.on_event: OnEvent = on_event or (lambda kind, details: None)
        self.session = session or f"anthropic-{uuid.uuid4().hex[:12]}"
        self.catalog = catalog.get("catalog")
        # api name -> its record: the catalogue's, or (inline) the search hit it was added from
        self.entries: dict[str, dict[str, Any]] = {t["api_name"]: t for t in entries}
        self.found: dict[str, str] = {}  # api name -> the search that returned it
        self._added: list[dict[str, Any]] = []  # inline: definitions for the next system message
        if builtin is not None:
            search: dict[str, Any] = {"type": BUILTIN[builtin], "name": f"tool_search_tool_{builtin}"}
        else:
            if SEARCH_TOOL in self.entries:
                raise ValueError(f"a catalogue tool is named {SEARCH_TOOL!r}")
            sources = ", ".join(sorted({t["server"] for t in entries}))
            what = "the tool catalogue" if inline else f"{len(entries)} tools ({sources})"
            search = {
                "name": SEARCH_TOOL,
                "description": (
                    f"Search {what} for the ones a task needs. Describe the "
                    "task, or the step you are about to take, in plain words; the matching tools then "
                    "become available to call. If none fits, search again in other words."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What you want to do, in plain words."},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_REFERENCES,
                            "description": "How many tools at most; default: as many as the request needs.",
                        },
                    },
                    "required": ["query"],
                },
            }
        deferred = [
            {
                "name": n,
                "description": t["description"],
                "input_schema": t["inputSchema"],
                "defer_loading": True,
            }
            for n, t in self.entries.items()
        ]
        self.tools: list[dict[str, Any]] = [search, *deferred]

    # -- inline: tools by value ---------------------------------------------------------------------
    def opening(self, task: str) -> list[dict[str, Any]]:
        """The first messages: the task, and (``prefetch``) the tools a search for it finds."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        if self.prefetch:
            self._find(task, None)
            if (added := self.additions()) is not None:
                messages.append(added)
        return messages

    def additions(self) -> dict[str, Any] | None:
        """The ``role: "system"`` message that adds the tools found since the last one (``inline``),
        to append right after the message that answered their search; None when there are none."""
        if not self._added:
            return None
        content = [
            {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": d}}
            for d in self._added
        ]
        self._added = []
        return {"role": "system", "content": content}

    def _definition(self, hit: dict[str, Any]) -> dict[str, Any]:
        description = hit.get("description") or ""
        if hit.get("inputSchemaShrunk"):
            description += SHRUNK_NOTE
        schema = dict(hit.get("inputSchema") or {})
        schema.pop("$schema", None)
        schema.setdefault("type", "object")
        return {"name": hit["api_name"], "description": description, "input_schema": schema}

    # -- one response ---------------------------------------------------------------------------
    def respond(self, response: Any) -> dict[str, Any] | None:
        """The user message that answers ``response``'s tool calls; None when there are none to
        answer (end of turn, ``max_tokens``, ``refusal``, ``pause_turn``: see ``run``)."""
        content = get(response, "content") or []
        self._builtin_searches(content)
        if get(response, "stop_reason") != "tool_use":
            return None
        results = [self._answer(b) for b in content if get(b, "type") == "tool_use"]
        return {"role": "user", "content": results} if results else None

    def _answer(self, block: Any) -> dict[str, Any]:
        tool_use_id, name, args = get(block, "id"), get(block, "name"), get(block, "input") or {}
        if name == SEARCH_TOOL and self.builtin is None:
            return self._search(tool_use_id, args)
        return self._call(tool_use_id, name, args)

    def _search(self, tool_use_id: str, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        limit = args.get("limit")
        k = None
        if isinstance(limit, int | float) and not isinstance(limit, bool):
            k = max(1, min(int(limit), MAX_REFERENCES))
        try:
            names = self._find(query, k)
        except ToolrankError as e:
            return _result(tool_use_id, f"The tool search failed: {e.message}", is_error=True)
        if not names:
            return _result(tool_use_id, "No tool matches that. Describe the task in other words.")
        if self.inline:  # their definitions follow in a system message (``additions``)
            listed = "\n".join(f"- {n}: {(self.entries[n].get('description') or '')[:120]}" for n in names)
            return _result(tool_use_id, f"These tools are now available to call:\n{listed}")
        return _result(tool_use_id, [{"type": "tool_reference", "tool_name": n} for n in names])

    def _find(self, query: str, k: int | None) -> list[str]:
        """Search, and -> the api names found: catalogue tools, or (``inline``) any hit, queued for
        the next ``additions`` the first time it is found. Raises ``ToolrankError``."""
        t0 = time.perf_counter()
        try:
            found = self.toolrank.search(query, k=k, session=self.session)
        except ToolrankError as e:
            self.on_event("search", {"query": query, "error": e.message})
            raise
        names: list[str] = []
        for hit in found.get("tools", []):
            name = hit.get("api_name")
            if self.inline and name not in self.entries and name != SEARCH_TOOL:
                if self.servers is not None and hit.get("server") not in self.servers:
                    continue
                self.entries[name] = hit
                self._added.append(self._definition(hit))
            if name in self.entries and name not in names and len(names) < MAX_REFERENCES:
                names.append(name)
                self.found[name] = found["search_id"]
        self.on_event(
            "search",
            {
                "query": query,
                "tools": [self.entries[n]["name"] for n in names],
                "mode": found.get("mode"),
                "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            },
        )
        return names

    def _call(self, tool_use_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        entry = self.entries.get(name)
        if entry is None:
            return _result(tool_use_id, f"There is no tool named {name!r}.", is_error=True)
        if self.approve is not None and not self.approve(entry, args):
            self.on_event("call", {"tool": entry["name"], "outcome": "declined"})
            return _result(tool_use_id, "The user declined this call.", is_error=True)
        t0 = time.perf_counter()
        try:
            out = self.toolrank.call(
                entry["name"], args, search_id=self.found.get(name), session=self.session
            )
        except ToolrankError as e:
            self.on_event("call", {"tool": entry["name"], "outcome": "error", "error": e.message})
            return _result(tool_use_id, f"toolrank could not run {entry['name']}: {e.message}", is_error=True)
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self.on_event("call", {"tool": entry["name"], "outcome": out.get("outcome"), "ms": ms})
        return _result(tool_use_id, blocks(out.get("content") or []), is_error=bool(out.get("isError")))

    def _builtin_searches(self, content: list[Any]) -> None:
        """Report the API's own tool searches (``builtin``), which it runs and answers itself."""
        queries = {
            get(b, "id"): get(b, "input") or {} for b in content if get(b, "type") == "server_tool_use"
        }
        for b in content:
            if get(b, "type") != "tool_search_tool_result":
                continue
            refs = get(get(b, "content"), "tool_references") or []
            names = [get(r, "tool_name") for r in refs]
            query = queries.get(get(b, "tool_use_id"), {})
            self.on_event(
                "search",
                {
                    "query": query.get("query") or query.get("pattern"),
                    "tools": [self.entries[n]["name"] if n in self.entries else n for n in names],
                    "mode": f"anthropic {self.builtin}",
                },
            )


@dataclass
class Result:
    text: str  # the last response's text
    stop_reason: str | None
    turns: int
    messages: list[Any] = field(repr=False)
    usage: dict[str, int] = field(default_factory=dict)


def run(
    llm: Any,
    toolrank: ToolrankClient | Toolbox,
    task: str,
    *,
    model: str,
    max_tokens: int = 4096,
    max_turns: int = 12,
    **create: Any,
) -> Result:
    """A reference loop: ``llm`` is an ``anthropic.Anthropic()`` (or its ``.beta``); ``create``
    goes to every ``messages.create``. Stops when a response asks for no tool, on ``max_tokens``,
    ``refusal`` and unknown stop reasons, or after ``max_turns`` requests; ``pause_turn`` is sent
    back as is."""
    box = toolrank if isinstance(toolrank, Toolbox) else Toolbox(toolrank)
    messages: list[Any] = box.opening(task)
    usage: dict[str, int] = {}
    response: Any = None
    turn = 0
    send = llm.messages.create
    if box.inline:  # the beta's header, through the SDK's beta namespace (llm may already be it)
        send = getattr(getattr(llm, "beta", None), "messages", llm.messages).create
        create = {
            **create,
            "betas": [*(b for b in create.get("betas") or [] if b != INLINE_BETA), INLINE_BETA],
        }
    for turn in range(1, max_turns + 1):
        response = send(model=model, max_tokens=max_tokens, tools=box.tools, messages=messages, **create)
        for key, value in dict(_fields(get(response, "usage"))).items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
        messages.append({"role": "assistant", "content": get(response, "content")})
        stop = get(response, "stop_reason")
        box.on_event("turn", {"turn": turn, "stop_reason": stop})
        if stop == "pause_turn":
            continue
        reply = box.respond(response)
        if reply is None:
            break
        messages.append(reply)
        if (added := box.additions()) is not None:  # inline: the tools that search found, by value
            messages.append(added)
    content = get(response, "content") or []
    text = "".join(get(b, "text") or "" for b in content if get(b, "type") == "text")
    return Result(text, get(response, "stop_reason"), turn, messages, usage)


def _fields(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    return dump() if callable(dump) else dict(vars(obj))
