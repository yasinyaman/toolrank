"""toolrank as the tool retriever of a LlamaIndex agent.

``ToolrankToolRetriever`` is an ``ObjectRetriever``. Given to a workflow agent (``FunctionAgent``,
``ReActAgent``, ``CodeActAgent``) as ``tool_retriever``, it answers each step with the tools a
toolrank search finds (``/v1/search``) instead of a fixed list; ``toolrank serve`` runs the calls
(``/v1/call``).

    from llama_index.core.agent.workflow import FunctionAgent
    from toolrank.client import ToolrankClient
    from toolrank.integrations.llamaindex import ToolrankToolRetriever

    agent = FunctionAgent(llm=llm, tool_retriever=ToolrankToolRetriever(ToolrankClient()))
    print(await agent.run("What time is it in Tokyo?"))

How LlamaIndex 0.14's agents use a tool retriever, and what this one does about it:
- Every step retrieves with the run's user message, so the same search would repeat: a query's
  tools are kept for ``ttl_s`` seconds.
- To run a call, the agent retrieves with the tool's name as the query and picks the tool by name.
  The name of a tool this retriever has handed out returns that tool without a search; any other
  name is searched like a query, and the agent finds the tool only if the search returns it.
- A tool's name is its api name. Its description is cut to the 1,024 characters
  ``ToolMetadata.to_openai_tool`` accepts, and its ``fn_schema`` is a pydantic model whose JSON
  schema is the tool's ``inputSchema``. A failed, refused or declined call is a ``ToolOutput``
  with ``is_error``, which the model reads.

Needs ``llama-index-core`` (``toolrank[llamaindex]``).
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any

try:
    from llama_index.core.objects import ObjectRetriever
    from llama_index.core.schema import QueryBundle
    from llama_index.core.tools import AsyncBaseTool, ToolMetadata, ToolOutput
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "toolrank.integrations.llamaindex needs llama-index-core: pip install 'toolrank[llamaindex]'"
    ) from e

from toolrank.client import ToolrankClient, ToolrankError
from toolrank.integrations._common import Approve, OnEvent, as_text

MAX_DESCRIPTION = 1024  # ToolMetadata.to_openai_tool refuses longer descriptions
CACHED_QUERIES = 256


def _schema_model(schema: dict[str, Any]) -> type[BaseModel]:
    """A pydantic model whose JSON schema is ``schema``, as ``ToolMetadata.fn_schema`` wants."""

    class Arguments(BaseModel):
        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return copy.deepcopy(schema)  # callers may edit what they get

    return Arguments


class ToolrankTool(AsyncBaseTool):
    """A tool a toolrank search found; ``toolrank serve`` runs it."""

    def __init__(self, retriever: ToolrankToolRetriever, hit: dict[str, Any]):
        self.retriever, self.hit = retriever, hit
        desc = hit.get("description") or hit["name"]
        if len(desc) > MAX_DESCRIPTION:
            desc = desc[: MAX_DESCRIPTION - 1] + "…"
        schema = hit.get("inputSchema") or {"type": "object", "properties": {}}
        self._metadata = ToolMetadata(description=desc, name=hit["api_name"], fn_schema=_schema_model(schema))

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Agents pass the model's arguments as keywords; one dict of them works too."""
        if args and not (len(args) == 1 and isinstance(args[0], dict) and not kwargs):
            raise TypeError(f"{self._metadata.name} takes keyword arguments or one dict of them")
        return self.retriever.run(self, dict(args[0]) if args else kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        return await asyncio.to_thread(self.call, *args, **kwargs)


class ToolrankToolRetriever(ObjectRetriever):
    """An agent's tools from toolrank searches (see the module docstring).

    ``k`` and ``instruction`` go to every search (default: the server's). ``approve(hit,
    arguments)`` may veto a call; ``on_event(kind, details)`` sees every search and call."""

    def __init__(
        self,
        toolrank: ToolrankClient,
        *,
        k: int | None = None,
        instruction: str | None = None,
        approve: Approve | None = None,
        on_event: OnEvent | None = None,
        session: str | None = None,
        ttl_s: float = 300.0,
    ):
        # no index and no node mapping: the tools come from toolrank
        super().__init__(retriever=None, object_node_mapping=None)  # type: ignore[arg-type]
        self.toolrank, self.k, self.instruction, self.approve = toolrank, k, instruction, approve
        self.on_event: OnEvent = on_event or (lambda kind, details: None)
        self.session = session or f"llamaindex-{uuid.uuid4().hex[:12]}"
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._queries: OrderedDict[str, tuple[float, list[ToolrankTool]]] = OrderedDict()
        self._by_name: dict[str, ToolrankTool] = {}  # every tool handed out
        self._found: dict[str, str] = {}  # api name -> the search that found it

    def retrieve(self, str_or_query_bundle: str | QueryBundle) -> list[ToolrankTool]:
        query = str_or_query_bundle if isinstance(str_or_query_bundle, str) else str_or_query_bundle.query_str
        with self._lock:
            named, cached = self._by_name.get(query), self._queries.get(query)
        if named is not None:  # the agent looking up a tool it is about to call
            return [named]
        if not query.strip():
            return []
        if cached is not None and time.monotonic() - cached[0] < self.ttl_s:
            return list(cached[1])
        return self._search(query)

    async def aretrieve(self, str_or_query_bundle: str | QueryBundle) -> list[ToolrankTool]:
        return await asyncio.to_thread(self.retrieve, str_or_query_bundle)

    def _search(self, query: str) -> list[ToolrankTool]:
        t0 = time.perf_counter()
        try:
            found = self.toolrank.search(
                query, k=self.k, instruction=self.instruction, full_schemas=True, session=self.session
            )
        except ToolrankError as e:
            self.on_event("search", {"query": query, "error": e.message})
            raise
        tools = [ToolrankTool(self, hit) for hit in found.get("tools", [])]
        with self._lock:
            for tool in tools:
                self._by_name[tool.metadata.get_name()] = tool
                self._found[tool.metadata.get_name()] = found["search_id"]
            self._queries[query] = (time.monotonic(), tools)
            self._queries.move_to_end(query)
            while len(self._queries) > CACHED_QUERIES:
                self._queries.popitem(last=False)
        self.on_event(
            "search",
            {
                "query": query,
                "tools": [t.hit["name"] for t in tools],
                "mode": found.get("mode"),
                "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            },
        )
        return list(tools)

    def run(self, tool: ToolrankTool, arguments: dict[str, Any]) -> ToolOutput:
        """Run ``tool`` through toolrank; -> its ``ToolOutput``."""
        hit, name = tool.hit, tool.metadata.get_name()

        def output(text: str, is_error: bool, raw: Any = None) -> ToolOutput:
            return ToolOutput(
                content=text, tool_name=name, raw_input=arguments, raw_output=raw, is_error=is_error
            )

        if self.approve is not None and not self.approve(hit, arguments):
            self.on_event("call", {"tool": hit["name"], "outcome": "declined"})
            return output("The user declined this call.", True)
        t0 = time.perf_counter()
        try:
            out = self.toolrank.call(
                hit["name"], arguments, search_id=self._found.get(name), session=self.session
            )
        except ToolrankError as e:
            self.on_event("call", {"tool": hit["name"], "outcome": "error", "error": e.message})
            return output(f"toolrank could not run {hit['name']}: {e.message}", True)
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self.on_event("call", {"tool": hit["name"], "outcome": out.get("outcome"), "ms": ms})
        failed = bool(out.get("isError"))
        text = as_text(out.get("content") or [])
        return output(
            text or (f"{hit['name']} failed" if failed else "(the tool returned nothing)"), failed, out
        )
