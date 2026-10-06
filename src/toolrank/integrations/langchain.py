"""toolrank as the tool selection of a LangChain 1.x agent (``langchain.agents.create_agent``).

``ToolrankToolSelector`` is an ``AgentMiddleware``: before each model call it narrows the tools the
model is shown to the ones toolrank finds for the conversation's last user message. Two ways:

- The agent's own tools, any LangChain tools: they go to ``toolrank serve``'s ``/v1/rank`` (any tool
  list works, not only toolrank's catalogue), and the model sees what toolrank's adaptive K keeps.
  It is LangChain's ``LLMToolSelectorMiddleware`` without the extra model call.

      from langchain.agents import create_agent
      from toolrank.integrations.langchain import ToolrankToolSelector

      agent = create_agent(model, tools=my_tools, middleware=[ToolrankToolSelector()])

- toolrank's catalogue: with ``toolbox=``, the agent has ``Toolbox.registry()`` among its tools and
  the catalogue's tools are picked by a toolrank search (``/v1/search``, the server's index, with the
  toolbox's ``k`` and ``instruction``). ``toolrank serve`` runs their calls and links each to that
  search in its usage log, which ``toolrank learn`` reads. The agent's other tools are shown as they
  are.

      from toolrank.client import ToolrankClient
      from toolrank.integrations.langgraph import Toolbox

      box = Toolbox(ToolrankClient())
      agent = create_agent(model, tools=[*box.registry().values(), *my_tools],
                           middleware=[ToolrankToolSelector(toolbox=box)])

Either way the model keeps the tools the conversation already called, any tool ``tool_choice``
names, ``always_include`` and provider tools (dicts, such as web search); shown tools keep the
agent's order. A model call goes on with every tool when there are at most ``min_tools`` to choose
from, when a tool is deferred to the provider's own tool search (``extras["defer_loading"]``), when
there is no user text, and when toolrank fails or does not answer within ``timeout_s``: the
selection then carries on in the background and is kept, so a later call is fast. With a toolbox a
failure shows the catalogue's tools of the session's last selection (and the fixed ones), not the
whole catalogue, which no model takes in one request. An agent makes several model calls per user
message; a selection is kept for ``ttl_s`` seconds, and a call that comes while the same selection
is still being made waits for that one, so they ask toolrank once. Selections run in daemon threads,
four at a time: one still running never holds up the interpreter's exit.

Needs ``langchain`` 1.x (``toolrank[langchain]``), imported when ``ToolrankToolSelector`` is first
used. ``Selector`` is the same work without LangChain's class.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from toolrank.client import ToolrankClient
from toolrank.cut import DEFAULT_MARGIN, DEFAULT_MAX_K, AdaptiveK
from toolrank.integrations._common import names_in
from toolrank.integrations.langgraph import Toolbox

log = logging.getLogger("toolrank.integrations.langchain")

CACHED_SELECTIONS = 256
WORKERS = 4  # selections asked of toolrank at a time


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            b if isinstance(b, str) else b.get("text")
            for b in content
            if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")
        ]
        return "\n".join(p for p in parts if isinstance(p, str)).strip()
    return ""


def user_text(messages: Sequence[Any]) -> str:
    """The text of the last user message that has any."""
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            text = _text(message.content)
            if text:
                return text
    return ""


def used_names(messages: Sequence[Any], tool_choice: Any) -> set[str]:
    """The tools the conversation already called, and any tool ``tool_choice`` names."""
    names = {tool_choice} if isinstance(tool_choice, str) else names_in(tool_choice)
    for message in messages:
        if getattr(message, "type", None) == "ai":
            names |= {
                c["name"]
                for c in getattr(message, "tool_calls", None) or []
                if isinstance(c.get("name"), str)
            }
    return names


def deferred(tool: Any) -> bool:
    """Whether the tool waits for the provider's own tool search."""
    if isinstance(tool, dict):
        return tool.get("defer_loading") is True or str(tool.get("type", "")).startswith("tool_search")
    return (getattr(tool, "extras", None) or {}).get("defer_loading") is True


def record(tool: Any) -> dict[str, Any]:
    """A LangChain tool as the MCP record ``/v1/rank`` takes: what the model would be sent."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    spec = convert_to_openai_tool(tool)["function"]
    return {
        "name": spec["name"],
        "description": spec.get("description") or "",
        "inputSchema": spec.get("parameters") or {"type": "object"},
    }


@dataclass(frozen=True)
class _Job:
    query: str
    pool: tuple[str, ...]  # the names to choose from, in the agent's order
    tools: tuple[Any, ...]  # those tools
    fixed: frozenset[str]  # shown whatever the choice
    key: tuple[Any, ...]  # the cache key of the choice


class Selector:
    """The middleware's work: ``select(request) -> request`` (``aselect``), the request a model call
    should get. ``on_select(details)`` sees every selection made or reused and every failure
    (``outcome`` ``selected`` or ``skipped``), not the calls with nothing to choose from (few tools,
    deferred ones, no user text)."""

    def __init__(
        self,
        toolrank: ToolrankClient | None = None,
        *,
        toolbox: Toolbox | None = None,
        url: str = "http://127.0.0.1:8765",
        api_key: str | None = None,
        min_tools: int = 20,
        max_tools: int = DEFAULT_MAX_K,
        margin: float = DEFAULT_MARGIN,
        always_include: Sequence[str] = (),
        instruction: str | None = None,
        timeout_s: float = 5.0,
        ttl_s: float = 300.0,
        on_select: Callable[[dict[str, Any]], None] | None = None,
    ):
        # the client's own timeout is long: a ranking that outlives timeout_s still fills the cache
        self.toolbox = toolbox
        self.toolrank = (
            toolbox.toolrank if toolbox else toolrank or ToolrankClient(url, api_key, timeout=60.0)
        )
        self.min_tools, self.max_tools, self.instruction = min_tools, max_tools, instruction
        self.always_include = frozenset(always_include)
        self.timeout_s, self.ttl_s = timeout_s, ttl_s
        self.rule = AdaptiveK(max_k=max_tools, margin=margin)
        self.on_select = on_select or (lambda details: None)
        self._slots = threading.BoundedSemaphore(WORKERS)
        self._lock = threading.Lock()
        self._chosen: OrderedDict[tuple[Any, ...], tuple[float, frozenset[str]]] = OrderedDict()
        self._running: dict[tuple[Any, ...], Future[frozenset[str]]] = {}
        self._last: OrderedDict[Any, frozenset[str]] = OrderedDict()  # toolbox: a session's last choice

    def _job(self, request: Any) -> _Job | None:
        """What to choose from, or None to show every tool."""
        tools = list(request.tools or [])
        if any(deferred(t) for t in tools):
            return None
        query = user_text(request.messages)
        own = [t for t in tools if not isinstance(t, dict)]
        if self.toolbox is not None:
            own = [t for t in own if t.name in self.toolbox.entries]
        if not query or len(own) <= self.min_tools:
            return None
        pool = tuple(t.name for t in own)
        fixed = frozenset(used_names(request.messages, request.tool_choice) | self.always_include)
        if self.toolbox is not None:  # searches are logged per session
            key: tuple[Any, ...] = ("search", self.toolbox.current_session(), query)
        else:
            key = ("rank", self.instruction, query, pool)
        return _Job(query, pool, tuple(own), fixed, key)

    def _cached(self, job: _Job) -> frozenset[str] | None:
        with self._lock:
            hit = self._chosen.get(job.key)
        return hit[1] if hit is not None and time.monotonic() - hit[0] < self.ttl_s else None

    def _choose(self, job: _Job) -> frozenset[str]:
        """Ask toolrank (in a pool thread, with the caller's context: a LangGraph thread's session)."""
        if self.toolbox is not None:
            found = self.toolbox.retrieve_tools(job.query)
            chosen = frozenset([n for n in found if n in job.pool][: self.max_tools])
        else:
            ranked = self.toolrank.rank(
                job.query, [record(t) for t in job.tools], instruction=self.instruction
            )
            n = self.rule.count([r["score"] for r in ranked])
            chosen = frozenset(job.pool[int(r["index"])] for r in ranked[:n])
        with self._lock:
            self._chosen[job.key] = (time.monotonic(), chosen)
            self._chosen.move_to_end(job.key)
            while len(self._chosen) > CACHED_SELECTIONS:
                self._chosen.popitem(last=False)
            if self.toolbox is not None:
                self._last[job.key[1]] = chosen
                self._last.move_to_end(job.key[1])
                while len(self._last) > CACHED_SELECTIONS:
                    self._last.popitem(last=False)
        return chosen

    def _submit(self, job: _Job) -> Future[frozenset[str]]:
        """The selection for ``job``: the one already being made, or a new one in a daemon thread
        (with the caller's context: a LangGraph thread's session)."""
        with self._lock:
            running = self._running.get(job.key)
            if running is not None:
                return running
            fut: Future[frozenset[str]] = Future()
            self._running[job.key] = fut
        ctx = contextvars.copy_context()

        def work() -> None:
            try:
                with self._slots:
                    result = ctx.run(self._choose, job)
            except BaseException as e:  # noqa: BLE001 - handed to whoever waits
                fut.set_exception(e)
            else:
                fut.set_result(result)
            finally:
                with self._lock:
                    if self._running.get(job.key) is fut:
                        del self._running[job.key]

        threading.Thread(target=work, name="toolrank-select", daemon=True).start()
        return fut

    def _skipped(self, request: Any, job: _Job, e: BaseException) -> Any:
        log.warning("toolrank tool selection skipped: %s: %s", type(e).__name__, e)
        details: dict[str, Any] = {
            "outcome": "skipped",
            "error": f"{type(e).__name__}: {e}",
            "tools": len(request.tools),
        }
        if self.toolbox is None:  # the agent's own tools: all of them, as it built the list
            self.on_select(details)
            return request
        # a toolbox's pool is the catalogue: the session's last choice and the fixed tools, not all
        with self._lock:
            last = self._last.get(job.key[1], frozenset())
        keep, pool = last | job.fixed, set(job.pool)
        kept = [t for t in request.tools if isinstance(t, dict) or t.name not in pool or t.name in keep]
        self.on_select({**details, "kept": len(kept), "fallback": "last selection" if last else "fixed only"})
        return request.override(tools=kept)

    def _apply(self, request: Any, job: _Job, chosen: frozenset[str], cached: bool) -> Any:
        keep, pool = chosen | job.fixed, set(job.pool)
        kept = [t for t in request.tools if isinstance(t, dict) or t.name not in pool or t.name in keep]
        self.on_select(
            {
                "outcome": "selected",
                "tools": len(request.tools),
                "kept": len(kept),
                "query": job.query[:200],
                "cached": cached,
            }
        )
        log.info("toolrank kept %d of %d tools", len(kept), len(request.tools))
        return request.override(tools=kept)

    def select(self, request: Any) -> Any:
        job = self._job(request)
        if job is None:
            return request
        chosen = self._cached(job)
        if chosen is not None:
            return self._apply(request, job, chosen, True)
        try:
            chosen = self._submit(job).result(timeout=self.timeout_s)
        except Exception as e:  # toolrank down, slow or odd: the model gets every tool (see _skipped)
            return self._skipped(request, job, e)
        return self._apply(request, job, chosen, False)

    async def aselect(self, request: Any) -> Any:
        job = self._job(request)
        if job is None:
            return request
        chosen = self._cached(job)
        if chosen is not None:
            return self._apply(request, job, chosen, True)
        try:  # shielded: a timeout here must not cancel the selection that goes on in the background
            chosen = await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(self._submit(job))), self.timeout_s
            )
        except Exception as e:
            return self._skipped(request, job, e)
        return self._apply(request, job, chosen, False)


@functools.cache
def _middleware_class() -> type:
    from langchain.agents.middleware import AgentMiddleware

    class ToolrankToolSelector(AgentMiddleware):
        """The tools a model call is shown, chosen by toolrank (see the module docstring); takes
        ``Selector``'s arguments."""

        def __init__(self, toolrank: ToolrankClient | None = None, **options: Any):
            super().__init__()
            self.selector = Selector(toolrank, **options)

        def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
            return handler(self.selector.select(request))

        async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
            return await handler(await self.selector.aselect(request))

    return ToolrankToolSelector


def __getattr__(name: str) -> Any:  # PEP 562: langchain is imported when the class is asked for
    if name == "ToolrankToolSelector":
        return _middleware_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
