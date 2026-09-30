"""toolrank's catalogue as LangChain tools, and its search as the tool retrieval of a LangGraph
agent such as langgraph-bigtool's.

``Toolbox.registry()`` is ``{api_name: tool}`` for every catalogue tool: LangChain tools whose
schema is the tool's ``inputSchema`` and whose calls ``toolrank serve`` runs (``/v1/call``).
``retrieve_tools(query)`` searches (``/v1/search``) and returns registry keys only, as many as
toolrank's adaptive K keeps (or ``k``); bigtool raises on a key its registry lacks.

    from langgraph_bigtool import create_agent
    from toolrank.client import ToolrankClient
    from toolrank.integrations.langgraph import Toolbox

    box = Toolbox(ToolrankClient())
    agent = create_agent(llm, box.registry(), retrieve_tools_function=box.retrieve_tools,
                         retrieve_tools_coroutine=box.aretrieve_tools).compile()

bigtool turns ``retrieve_tools`` into a tool the model calls, so its ``__doc__`` is the model's
description of the search. langgraph-bigtool 0.0.3, the release on PyPI, needs ``langgraph<1``;
this module needs only ``langchain-core`` (``toolrank[langgraph]``), imported when the registry is
built.

A tool that fails, a call toolrank refuses and a call ``approve`` declines raise
``ToolException``, which the tools turn into an error result the model reads. A tool takes the
model's arguments and nothing else, so arguments named ``config`` or ``run_manager`` reach it.
Searches and calls carry a session for the usage log: the LangGraph thread
(``configurable.thread_id``) when the run has one, else the toolbox's own.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from toolrank.client import ToolrankClient, ToolrankError
from toolrank.integrations._common import Approve, OnEvent, as_text

FOUND = 10_000  # (session, tool) -> search id pairs kept to link calls to their search
MAX_SERVERS = 20  # named in the search tool's description


def _thread_session() -> str | None:
    """``langgraph:<thread id>`` for code running inside a LangGraph thread, else None; a thread
    id that is not a short printable header value is hashed."""
    try:
        from langchain_core.runnables.config import ensure_config
    except ImportError:
        return None
    thread = (ensure_config().get("configurable") or {}).get("thread_id")
    if thread is None:
        return None
    thread = str(thread)
    if len(thread) > 100 or not all(" " <= c <= "~" for c in thread):
        thread = hashlib.sha256(thread.encode("utf-8")).hexdigest()[:32]
    return f"langgraph:{thread}"


@functools.cache
def _tool_class() -> type:
    from langchain_core.tools import BaseTool

    class ToolrankTool(BaseTool):
        """One catalogue tool, run by ``toolrank serve``."""

        toolbox: Any = None

        # no ``config`` / ``run_manager`` parameters: LangChain injects those by signature
        def _run(self, **arguments: Any) -> str:
            return self.toolbox.run(self.name, arguments)

        async def _arun(self, **arguments: Any) -> str:
            return await asyncio.to_thread(self.toolbox.run, self.name, arguments)

    return ToolrankTool


class Toolbox:
    """The catalogue (taken once) as LangChain tools, and the searches and calls made with it.

    ``retrieve_tools(query) -> list[str]`` and ``aretrieve_tools`` search; ``k`` and
    ``instruction`` go to every search (default: the server's); hits outside ``servers`` are
    dropped. ``approve(entry, arguments)`` may veto a call; ``on_event(kind, details)`` sees every
    search and call."""

    def __init__(
        self,
        toolrank: ToolrankClient,
        *,
        servers: Sequence[str] | None = None,
        k: int | None = None,
        instruction: str | None = None,
        approve: Approve | None = None,
        on_event: OnEvent | None = None,
        session: str | None = None,
    ):
        catalog = toolrank.catalog()
        entries = [t for t in catalog["tools"] if not servers or t["server"] in set(servers)]
        self.toolrank, self.k, self.instruction, self.approve = toolrank, k, instruction, approve
        self.on_event: OnEvent = on_event or (lambda kind, details: None)
        self.session = session or f"langgraph-{uuid.uuid4().hex[:12]}"
        self.catalog = catalog.get("catalog")
        self.entries: dict[str, dict[str, Any]] = {t["api_name"]: t for t in entries}
        self._found: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._lock = threading.Lock()
        self._registry: dict[str, Any] | None = None

        names = sorted({t["server"] for t in entries})
        shown = ", ".join(names[:MAX_SERVERS])
        if len(names) > MAX_SERVERS:
            shown += f" and {len(names) - MAX_SERVERS} more"
        description = (
            f"Search {len(entries)} tools ({shown}) for the ones a task needs. Describe the task, or "
            "the step you are about to take, in plain words; the matching tools then become available "
            "to call. If none fits, search again in other words."
        )

        def retrieve_tools(query: str) -> list[str]:
            return self._retrieve(query)

        async def aretrieve_tools(query: str) -> list[str]:
            return await asyncio.to_thread(self._retrieve, query)

        retrieve_tools.__doc__ = aretrieve_tools.__doc__ = description
        self.retrieve_tools, self.aretrieve_tools = retrieve_tools, aretrieve_tools

    def registry(self) -> dict[str, Any]:
        """``{api_name: tool}`` for every catalogue tool (``langchain_core.tools.BaseTool``s)."""
        if self._registry is None:
            cls = _tool_class()
            self._registry = {
                name: cls(
                    name=name,
                    description=entry["description"] or entry["name"],
                    args_schema=entry["inputSchema"],
                    handle_tool_error=True,
                    toolbox=self,
                )
                for name, entry in self.entries.items()
            }
        return self._registry

    def _session(self) -> str:
        return _thread_session() or self.session

    def _retrieve(self, query: str) -> list[str]:
        session, t0 = self._session(), time.perf_counter()
        try:
            found = self.toolrank.search(query, k=self.k, instruction=self.instruction, session=session)
        except ToolrankError as e:
            self.on_event("search", {"query": query, "error": e.message})
            raise
        names: list[str] = []
        for hit in found.get("tools", []):
            name = hit.get("api_name")
            if name in self.entries and name not in names:
                names.append(name)
        with self._lock:
            for name in names:
                self._found[(session, name)] = found["search_id"]
                self._found.move_to_end((session, name))
            while len(self._found) > FOUND:
                self._found.popitem(last=False)
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

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a catalogue tool (by api name) through toolrank; -> its output as text. A failed,
        refused or declined call raises ``ToolException``."""
        from langchain_core.tools import ToolException

        entry = self.entries[name]
        if self.approve is not None and not self.approve(entry, arguments):
            self.on_event("call", {"tool": entry["name"], "outcome": "declined"})
            raise ToolException("The user declined this call.")
        session, t0 = self._session(), time.perf_counter()
        with self._lock:
            search_id = self._found.get((session, name))
        try:
            out = self.toolrank.call(entry["name"], arguments, search_id=search_id, session=session)
        except ToolrankError as e:
            self.on_event("call", {"tool": entry["name"], "outcome": "error", "error": e.message})
            raise ToolException(f"toolrank could not run {entry['name']}: {e.message}") from e
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self.on_event("call", {"tool": entry["name"], "outcome": out.get("outcome"), "ms": ms})
        text = as_text(out.get("content") or [])
        if out.get("isError"):
            raise ToolException(text or f"{entry['name']} failed")
        return text or "(the tool returned nothing)"
