"""toolrank as a tool filter in the LiteLLM proxy: a request that carries many tools reaches the
model with the ones toolrank finds relevant to the conversation.

``tool_filter`` is a LiteLLM ``CustomLogger`` whose pre-call hook sends the request's own function
tools to ``toolrank serve``'s ``/v1/rank`` (any tool list works, not only toolrank's catalogue) and
keeps what toolrank's adaptive K keeps for the last user message. It always keeps the tools the
conversation already called, any tool ``tool_choice`` names, and tools that are not plain
functions (provider tools such as web search or code execution). It is LiteLLM's
``mcp_semantic_tool_filter`` with toolrank's ranking, for Chat Completions, Responses and
Anthropic Messages requests alike.

A request stays as it came when it has at most ``min_tools`` function tools, when it uses the
provider's own tool search (``defer_loading``, ``tool_search``), when it has no user text, and
when toolrank fails or does not answer within ``timeout_s``: the filter never blocks a request and
never edits one in place. The first sight of a large tool list may time out while toolrank embeds
it; the ranking carries on in the background, and later requests are fast. With
``previous_response_id`` the earlier calls are not in the request, so only ``tool_choice`` and the
calls in ``input`` are kept for sure.

In the proxy's ``config.yaml``, with toolrank installed in the proxy's environment (no extra: it
needs nothing but litellm, and litellm pins ``openai<3``):

    litellm_settings:
      callbacks: toolrank.integrations.litellm.tool_filter

Configuration comes from the environment: ``TOOLRANK_URL`` (default http://127.0.0.1:8765),
``TOOLRANK_API_KEY``, ``TOOLRANK_FILTER_MIN_TOOLS`` (20), ``TOOLRANK_FILTER_MAX_TOOLS`` (10),
``TOOLRANK_FILTER_MARGIN`` (0.2) and ``TOOLRANK_FILTER_TIMEOUT`` (2 seconds). Tools that LiteLLM's
MCP gateway adds itself are added after this hook; for those, put toolrank behind the gateway as
an MCP server (``examples/litellm/config.yaml`` shows both).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from collections.abc import Callable, Mapping
from typing import Any

from toolrank.client import ToolrankClient
from toolrank.cut import DEFAULT_MARGIN, DEFAULT_MAX_K, AdaptiveK

log = logging.getLogger("toolrank.integrations.litellm")

# LiteLLM's call types -> the request shape
SHAPES = {
    "acompletion": "chat",
    "completion": "chat",
    "aresponses": "responses",
    "responses": "responses",
    "anthropic_messages": "messages",
    "aanthropic_messages": "messages",
}
TEXT_BLOCKS = ("text", "input_text")


def _record(shape: str, tool: Any) -> dict[str, Any] | None:
    """A plain function tool as the MCP record ``/v1/rank`` takes; None for anything else."""
    if not isinstance(tool, dict):
        return None
    if shape == "chat":
        spec = tool.get("function") if tool.get("type", "function") == "function" else None
        schema_key = "parameters"
    elif shape == "responses":
        spec, schema_key = (tool if tool.get("type") == "function" else None), "parameters"
    else:  # Anthropic: client tools have no type or "custom"; the typed ones are the API's own
        spec, schema_key = (tool if tool.get("type") in (None, "custom") else None), "input_schema"
    if not isinstance(spec, dict) or not isinstance(spec.get("name"), str) or not spec["name"]:
        return None
    schema = spec.get(schema_key)
    description = spec.get("description")
    return {
        "name": spec["name"],
        "description": description if isinstance(description, str) else "",
        "inputSchema": schema if isinstance(schema, dict) else {"type": "object"},
    }


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") in TEXT_BLOCKS and isinstance(b.get("text"), str)
        ]
        return "\n".join(parts).strip()
    return ""


def user_text(shape: str, data: Mapping[str, Any]) -> str:
    """The text of the last user message that has any (a message of tool results has none)."""
    if shape == "responses":
        items = data.get("input")
        if isinstance(items, str):
            return items.strip()
    else:
        items = data.get("messages")
    for item in reversed(items if isinstance(items, list) else []):
        if isinstance(item, dict) and item.get("role") == "user" and item.get("type", "message") == "message":
            text = _text(item.get("content"))
            if text:
                return text
    return ""


def _names_in(value: Any) -> set[str]:
    """Every string under a ``name`` key, however deep (tool choices, allowed-tool lists)."""
    if isinstance(value, dict):
        out = {value["name"]} if isinstance(value.get("name"), str) else set()
        for v in value.values():
            out |= _names_in(v)
        return out
    if isinstance(value, list):
        return set().union(*(_names_in(v) for v in value))
    return set()


def used_names(shape: str, data: Mapping[str, Any]) -> set[str]:
    """The tools the conversation already called, and any tool ``tool_choice`` names."""
    names = _names_in(data.get("tool_choice"))
    if shape == "responses":
        items = data.get("input")
        for item in items if isinstance(items, list) else []:
            if (
                isinstance(item, dict)
                and item.get("type") == "function_call"
                and isinstance(item.get("name"), str)
            ):
                names.add(item["name"])
        return names
    for message in data.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if shape == "chat":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and isinstance((call.get("function") or {}).get("name"), str):
                    names.add(call["function"]["name"])
            if isinstance(message.get("function_call"), dict):
                names |= _names_in(message["function_call"])
        elif isinstance(message.get("content"), list):
            for block in message["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and isinstance(block.get("name"), str)
                ):
                    names.add(block["name"])
    return names


def provider_search(tools: list[Any]) -> bool:
    """Whether the request leaves tool loading to the provider's own tool search."""
    return any(
        isinstance(t, dict)
        and (t.get("defer_loading") is True or str(t.get("type", "")).startswith("tool_search"))
        for t in tools
    )


def options_from_env(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    return {
        "url": env.get("TOOLRANK_URL") or "http://127.0.0.1:8765",
        "api_key": env.get("TOOLRANK_API_KEY") or None,
        "min_tools": int(env.get("TOOLRANK_FILTER_MIN_TOOLS") or 20),
        "max_tools": int(env.get("TOOLRANK_FILTER_MAX_TOOLS") or DEFAULT_MAX_K),
        "margin": float(env.get("TOOLRANK_FILTER_MARGIN") or DEFAULT_MARGIN),
        "timeout_s": float(env.get("TOOLRANK_FILTER_TIMEOUT") or 2.0),
    }


class ToolFilter:
    """The filter without LiteLLM: ``await ToolFilter(...)(data, call_type)`` -> a new request
    dict with fewer tools, or None to send the request as it came."""

    def __init__(
        self,
        toolrank: ToolrankClient | None = None,
        *,
        url: str = "http://127.0.0.1:8765",
        api_key: str | None = None,
        min_tools: int = 20,
        max_tools: int = DEFAULT_MAX_K,
        margin: float = DEFAULT_MARGIN,
        timeout_s: float = 2.0,
        on_filter: Callable[[dict[str, Any]], None] | None = None,
    ):
        # the client's own timeout is long: a ranking that outlives timeout_s still warms the cache
        self.toolrank = toolrank or ToolrankClient(url, api_key, timeout=60.0)
        self.min_tools, self.timeout_s = min_tools, timeout_s
        self.rule = AdaptiveK(max_k=max_tools, margin=margin)
        self.on_filter = on_filter or (lambda details: None)

    async def __call__(self, data: Mapping[str, Any], call_type: str) -> dict[str, Any] | None:
        shape = SHAPES.get(call_type)
        tools = data.get("tools") if shape else None
        if not isinstance(tools, list) or provider_search(tools):
            return None
        functions = {i: r for i, t in enumerate(tools) if (r := _record(shape, t))}  # position -> record
        query = user_text(shape, data)
        if len(functions) <= self.min_tools or not query:
            return None
        positions = list(functions)
        try:
            ranked = await asyncio.wait_for(
                asyncio.to_thread(self.toolrank.rank, query, list(functions.values())), timeout=self.timeout_s
            )
            n = self.rule.count([r["score"] for r in ranked])
            chosen = {positions[int(r["index"])] for r in ranked[:n]}
        except Exception as e:  # toolrank down, slow or odd: the request goes on unfiltered
            log.warning("toolrank tool filter skipped: %s: %s", type(e).__name__, e)
            self.on_filter({"outcome": "skipped", "error": f"{type(e).__name__}: {e}", "tools": len(tools)})
            return None
        keep = used_names(shape, data)
        kept = [
            t
            for i, t in enumerate(tools)
            if i not in functions or i in chosen or functions[i]["name"] in keep
        ]
        self.on_filter({"outcome": "filtered", "tools": len(tools), "kept": len(kept), "query": query[:200]})
        log.info("toolrank kept %d of %d tools", len(kept), len(tools))
        return {**data, "tools": kept}


@functools.cache
def _filter_class() -> type:
    from litellm.integrations.custom_logger import CustomLogger

    class ToolrankToolFilter(CustomLogger):
        """``ToolFilter`` as a LiteLLM callback."""

        def __init__(self, **options: Any):
            super().__init__()
            self.filter = ToolFilter(**options)

        # defined on this class itself: the proxy only calls a callback's own pre-call hook
        async def async_pre_call_hook(
            self, user_api_key_dict: Any, cache: Any, data: dict, call_type: str
        ) -> dict | None:
            return await self.filter(data, call_type)

    return ToolrankToolFilter


@functools.cache
def _tool_filter() -> Any:
    return _filter_class()(**options_from_env())


def __getattr__(name: str) -> Any:  # PEP 562: litellm is imported when the proxy asks for these
    if name == "ToolrankToolFilter":
        return _filter_class()
    if name == "tool_filter":
        return _tool_filter()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
