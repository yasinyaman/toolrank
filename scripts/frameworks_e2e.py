"""End-to-end check of the framework adapters (Faz 1 week 5) on a real catalogue, with scripted
models: no model API is called and no tool call leaves the machine.

Starts ``toolrank serve`` as ``platforms_e2e.py`` does (the week-1 MCP servers, a local stand-in for
every OpenAPI source), then runs each task through:

- langgraph-bigtool 0.0.3's agent, in its own environment (the release needs ``langgraph<1``): the
  model calls ``retrieve_tools``, then the task's tool if it was retrieved, then answers;
- a LangChain 1.x ``create_agent`` with ``ToolrankToolSelector``, two ways: the whole catalogue as
  its tools with a ``Toolbox`` (a search per turn), and N_TOOLS of them as its own tools (``/v1/rank``);
  the model calls the task's tool if it was shown, then answers;
- a LlamaIndex ``FunctionAgent`` with ``ToolrankToolRetriever`` and ``MockFunctionCallingLLM``;
- the LiteLLM proxy (``uvx``, its own environment) with ``tool_filter`` in front of a stub upstream
  that records the tools each request brings: 120 catalogue tools plus a provider tool, in Chat
  Completions, Responses and Anthropic Messages shape, with an earlier call in the history;
- toolrank behind LiteLLM's MCP gateway: list, search, call.

    uv run python scripts/frameworks_e2e.py --data data/w3 --emb-url http://$GB10:8091/v1 \
        --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --out results/frameworks_e2e.json

``--only NAME`` (repeatable) runs some of the parts: langgraph_bigtool, langchain, llamaindex, litellm.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from platforms_e2e import free_port, served, stand_in  # noqa: E402

from toolrank.client import ToolrankClient  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BIGTOOL = ["--with", "langgraph<1", "--with", "langgraph-bigtool==0.0.3"]
LITELLM = "litellm[proxy]==1.103.1"
MASTER_KEY = "sk-e2e-local"  # the throwaway proxy's own admin key
N_TOOLS = 120  # tools per LiteLLM request, under OpenAI's 128
PARTS = ["langgraph_bigtool", "langchain", "llamaindex", "litellm"]
TASKS = [
    {
        "task": "What time is it in Tokyo right now?",
        "tool": "time__get_current_time",
        "args": {"timezone": "Asia/Tokyo"},
    },
    {
        "task": "List the pets tagged dog in the pet store.",
        "tool": "swagger-petstore__findPets",
        "args": {"tags": ["dog"]},
    },
]


# -- langgraph-bigtool (runs in its own environment: --part bigtool) ------------------------------


def scripted_model(task: dict[str, Any], bound: list[list[str]]) -> Any:
    """A chat model for bigtool that calls ``retrieve_tools``, then the task's tool if it was
    retrieved, then answers with the tool's output; ``bound`` gets the tool names of each turn."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.runnables import RunnableLambda

    def respond(messages: list[Any], names: list[str]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            search = {"name": "retrieve_tools", "args": {"query": task["task"]}, "id": "r1"}
            return AIMessage("", tool_calls=[search])
        if isinstance(last, ToolMessage) and last.tool_call_id == "r1":
            if task["tool"] not in names:
                return AIMessage("The tool was not retrieved.")
            return AIMessage("", tool_calls=[{"name": task["tool"], "args": task["args"], "id": "c1"}])
        return AIMessage(f"Answer: {str(last.content)[:300]}")

    class Model:  # bigtool binds the retrieval tool and every tool retrieved so far, each turn
        def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
            names = [getattr(x, "name", None) or x.__name__ for x in tools]
            bound.append(names)
            return RunnableLambda(lambda messages: respond(messages, names))

    return Model()


def bigtool(url: str) -> list[dict[str, Any]]:
    from langchain_core.messages import HumanMessage
    from langgraph.store.memory import InMemoryStore
    from langgraph_bigtool import create_agent

    from toolrank.integrations.langgraph import Toolbox

    events: list[tuple[str, dict[str, Any]]] = []
    box = Toolbox(ToolrankClient(url), on_event=lambda kind, details: events.append((kind, details)))
    t0 = time.perf_counter()
    registry = box.registry()
    registry_s = time.perf_counter() - t0
    rows = []
    for n, t in enumerate(TASKS):
        bound: list[list[str]] = []
        agent = create_agent(
            scripted_model(t, bound),
            registry,
            retrieve_tools_function=box.retrieve_tools,
            retrieve_tools_coroutine=box.aretrieve_tools,
        ).compile(store=InMemoryStore())
        config = {"configurable": {"thread_id": f"e2e-bigtool-{n}"}}
        inputs = {"messages": [HumanMessage(t["task"])]}
        events.clear()
        t0 = time.perf_counter()
        state = asyncio.run(agent.ainvoke(inputs, config)) if n % 2 else agent.invoke(inputs, config)
        calls = [d["tool"] + ":" + str(d.get("outcome")) for k, d in events if k == "call"]
        rows.append(
            {
                "task": t["task"],
                "mode": "async" if n % 2 else "sync",
                "retrieved": state["selected_tool_ids"],
                "bound_on_turn_2": bound[1] if len(bound) > 1 else None,
                "calls": calls,
                "answer": str(state["messages"][-1].content)[:300],
                "ok": t["tool"] in state["selected_tool_ids"] and any(c.endswith(":ok") for c in calls),
                "seconds": round(time.perf_counter() - t0, 2),
            }
        )
    return [{"registry_tools": len(registry), "registry_s": round(registry_s, 2)}, *rows]


def run_bigtool(url: str) -> Any:
    cmd = ["uv", "run", "--quiet", *BIGTOOL, "python", __file__, "--part", "bigtool", "--url", url]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=900)
    if out.returncode:
        return {"error": out.stderr[-3000:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


# -- LangChain ------------------------------------------------------------------------------------


def langchain_agent(url: str) -> list[dict[str, Any]]:
    from langchain.agents import create_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import Field

    from toolrank.integrations.langchain import ToolrankToolSelector
    from toolrank.integrations.langgraph import Toolbox

    class Scripted(BaseChatModel):
        """Calls the task's tool if it was shown, then answers with the tool's output."""

        task: dict[str, Any]
        shown: list[list[str]] = Field(default_factory=list)

        def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
            names = [t.get("name") or t["type"] if isinstance(t, dict) else t.name for t in tools]
            return self.bind(tool_names=names)

        def _generate(
            self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> Any:
            names = list(kwargs.get("tool_names") or [])
            self.shown.append(names)
            if isinstance(messages[-1], ToolMessage):
                message = AIMessage(f"Answer: {str(messages[-1].content)[:300]}")
            elif self.task["tool"] in names:
                call = {"name": self.task["tool"], "args": self.task["args"], "id": "c1"}
                message = AIMessage("", tool_calls=[call])
            else:
                message = AIMessage("The tool was not shown.")
            return ChatResult(generations=[ChatGeneration(message=message)])

        @property
        def _llm_type(self) -> str:
            return "scripted"

    events: list[tuple[str, dict[str, Any]]] = []
    box = Toolbox(ToolrankClient(url), on_event=lambda kind, details: events.append((kind, details)))
    registry = box.registry()
    own = [registry[t["api_name"]] for t in request_tools(list(box.entries.values()))]
    ways = {"toolbox": (list(registry.values()), {"toolbox": box}), "own_tools": (own, {})}
    rows = []
    for way, (tools, options) in ways.items():
        for n, t in enumerate(TASKS):
            model, selections = Scripted(task=t), []
            # a long timeout: the first ranking of a tool list embeds it, and the report should show the choice
            selector = ToolrankToolSelector(
                ToolrankClient(url), timeout_s=60.0, on_select=selections.append, **options
            )
            t0 = time.perf_counter()
            agent = create_agent(model, tools=tools, middleware=[selector])
            agent_s = time.perf_counter() - t0
            config = {"configurable": {"thread_id": f"e2e-langchain-{way}-{n}"}}
            inputs = {"messages": [HumanMessage(t["task"])]}
            events.clear()
            t0 = time.perf_counter()
            state = asyncio.run(agent.ainvoke(inputs, config)) if n % 2 else agent.invoke(inputs, config)
            calls = [d["tool"] + ":" + str(d.get("outcome")) for k, d in events if k == "call"]
            shown = model.shown[0] if model.shown else []
            rows.append(
                {
                    "way": way,
                    "task": t["task"],
                    "mode": "async" if n % 2 else "sync",
                    "agent_tools": len(tools),
                    "agent_s": round(agent_s, 2),
                    "shown": shown,
                    "model_calls": len(model.shown),
                    "selections": [
                        {k: s.get(k) for k in ("outcome", "kept", "cached", "error") if k in s}
                        for s in selections
                    ],
                    "searches": sum(k == "search" for k, _ in events),
                    "calls": calls,
                    "answer": str(state["messages"][-1].content)[:300],
                    "ok": t["tool"] in shown and any(c.endswith(":ok") for c in calls),
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            )
    return rows


# -- LlamaIndex -----------------------------------------------------------------------------------


async def llamaindex(url: str) -> list[dict[str, Any]]:
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
    from llama_index.core.llms.mock import MockFunctionCallingLLM

    from toolrank.integrations.llamaindex import ToolrankToolRetriever

    events: list[tuple[str, dict[str, Any]]] = []
    retriever = ToolrankToolRetriever(
        ToolrankClient(url), on_event=lambda kind, details: events.append((kind, details))
    )

    def scripted(t: dict[str, Any], offered: list[list[str]]) -> Any:
        def model(messages: list[Any], **kwargs: Any) -> Any:
            names = [x.metadata.name for x in kwargs.get("tools", [])]
            offered.append(names)
            if any(m.role == MessageRole.TOOL for m in messages):
                return ChatMessage(
                    role=MessageRole.ASSISTANT, content=f"Answer: {messages[-1].content[:300]}"
                )
            if t["tool"] not in names:
                return ChatMessage(role=MessageRole.ASSISTANT, content="The tool was not retrieved.")
            call = ToolCallBlock(tool_call_id="c1", tool_name=t["tool"], tool_kwargs=t["args"])
            return ChatMessage(role=MessageRole.ASSISTANT, blocks=[call])

        return model

    rows = []
    for t in TASKS:
        offered: list[list[str]] = []
        agent = FunctionAgent(
            llm=MockFunctionCallingLLM(response_generator=scripted(t, offered)), tool_retriever=retriever
        )
        events.clear()
        t0 = time.perf_counter()
        answer = str(await agent.run(user_msg=t["task"]))
        calls = [d["tool"] + ":" + str(d.get("outcome")) for k, d in events if k == "call"]
        rows.append(
            {
                "task": t["task"],
                "offered": offered[0] if offered else [],
                "searches": sum(k == "search" for k, _ in events),
                "calls": calls,
                "answer": answer[:300],
                "ok": bool(offered) and t["tool"] in offered[0] and any(c.endswith(":ok") for c in calls),
                "seconds": round(time.perf_counter() - t0, 2),
            }
        )
    return rows


# -- LiteLLM --------------------------------------------------------------------------------------

LITELLM_CONFIG = """
model_list:
  - model_name: gpt-stub
    litellm_params: {{model: openai/gpt-stub, api_base: "{upstream}/v1", api_key: sk-stub}}
  - model_name: claude-stub
    litellm_params: {{model: anthropic/claude-stub, api_base: "{upstream}", api_key: sk-stub}}
litellm_settings:
  callbacks: toolrank.integrations.litellm.tool_filter
general_settings:
  master_key: {master_key}
mcp_servers:
  toolrank:
    url: "{toolrank}/mcp"
    transport: http
"""

ANSWERS = {
    "/v1/chat/completions": {
        "id": "chatcmpl-e2e",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-stub",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "/v1/responses": {
        "id": "resp_e2e",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-stub",
        "output": [
            {
                "type": "message",
                "id": "msg_e2e",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    },
    "/v1/messages": {
        "id": "msg_e2e",
        "type": "message",
        "role": "assistant",
        "model": "claude-stub",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
}


def upstream() -> tuple[ThreadingHTTPServer, list[dict[str, Any]]]:
    """A model API stand-in: records each request's path and tools, answers "ok"."""
    got: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            path = self.path.split("?")[0]
            got.append({"path": path, "tools": body.get("tools") or []})
            data = json.dumps(ANSWERS.get(path, {"error": f"{path}: not in this stand-in"})).encode()
            self.send_response(200 if path in ANSWERS else 404)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, got


def request_tools(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """N_TOOLS catalogue entries: the small servers whole, then Stripe and GitHub in turn."""
    small = [t for t in catalog if t["server"] not in ("stripe", "github")]
    stripe, github = ([t for t in catalog if t["server"] == s] for s in ("stripe", "github"))
    return small + list(
        itertools.islice(
            itertools.chain.from_iterable(zip(stripe, github, strict=False)), N_TOOLS - len(small)
        )
    )


def shaped(
    shape: str, task: str, tools: list[dict[str, Any]], used: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """(path, body): one conversation — the task, an earlier call of ``used``, its result."""
    name, desc, schema = "api_name", "description", "inputSchema"
    if shape == "chat":
        body = {
            "model": "gpt-stub",
            "tools": [
                {
                    "type": "function",
                    "function": {"name": t[name], "description": t[desc], "parameters": t[schema]},
                }
                for t in tools
            ],
            "messages": [
                {"role": "user", "content": task},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c0", "type": "function", "function": {"name": used[name], "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c0", "content": "{}"},
            ],
        }
        return "/v1/chat/completions", body
    if shape == "responses":
        body = {
            "model": "gpt-stub",
            "tools": [
                {"type": "web_search"},
                *(
                    {"type": "function", "name": t[name], "description": t[desc], "parameters": t[schema]}
                    for t in tools
                ),
            ],
            "input": [
                {"role": "user", "content": task},
                {"type": "function_call", "call_id": "c0", "name": used[name], "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c0", "output": "{}"},
            ],
        }
        return "/v1/responses", body
    body = {
        "model": "claude-stub",
        "max_tokens": 64,
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 1},
            *({"name": t[name], "description": t[desc], "input_schema": t[schema]} for t in tools),
        ],
        "messages": [
            {"role": "user", "content": task},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "c0", "name": used[name], "input": {}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c0", "content": "{}"}]},
        ],
    }
    return "/v1/messages", body


def _post(url: str, body: dict[str, Any]) -> tuple[int, Any]:
    data = json.dumps(body).encode()
    headers = {"content-type": "application/json", "authorization": f"Bearer {MASTER_KEY}"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:500].decode("utf-8", "replace")


async def gateway(url: str) -> dict[str, Any]:
    """toolrank behind LiteLLM's MCP gateway: its tools listed, one search, one call."""
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    from toolrank.adapters.mcp_client import list_all

    headers = {"Authorization": f"Bearer {MASTER_KEY}"}
    async with (
        httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30, read=120)) as http,
        Client(streamable_http_client(url, http_client=http), cache=None) as client,
    ):
        names = [t["name"] for t in await list_all(client)]
        search = next(n for n in names if n.endswith("search_tools"))
        call = next(n for n in names if n.endswith("call_tool"))
        found = await client.call_tool(search, {"query": TASKS[0]["task"]})
        payload = json.loads(found.content[0].text)
        ran = await client.call_tool(
            call,
            {
                "name": "time/get_current_time",
                "arguments": TASKS[0]["args"],
                "search_id": payload["search_id"],
            },
        )
    return {
        "tools": names,
        "search_top3": [t["name"] for t in payload["tools"][:3]],
        "call": ran.content[0].text[:200],
        "call_error": bool(ran.is_error),
    }


def litellm_proxy(url: str) -> dict[str, Any]:
    catalog = ToolrankClient(url).catalog()["tools"]
    tools = request_tools(catalog)
    used = next(t for t in tools if t["server"] == "stripe")
    stub, got = upstream()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    out: dict[str, Any] = {"tools_per_request": len(tools), "used": used["api_name"], "requests": []}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "litellm.yaml"
        cfg.write_text(
            LITELLM_CONFIG.format(
                upstream=f"http://127.0.0.1:{stub.server_address[1]}", toolrank=url, master_key=MASTER_KEY
            )
        )
        # a long timeout here: the first request embeds the 120 tool texts, which is what it measures
        env = {**os.environ, "TOOLRANK_URL": url, "TOOLRANK_FILTER_TIMEOUT": "600"}
        env.pop("VIRTUAL_ENV", None)
        cmd = ["uvx", "--quiet", "--from", LITELLM, "--with-editable", str(REPO), "litellm"]
        cmd += ["--config", str(cfg), "--host", "127.0.0.1", "--port", str(port)]
        log = Path(tmp) / "litellm.log"
        with log.open("w") as f:
            proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            try:
                for _ in range(600):
                    try:
                        with urllib.request.urlopen(f"{base}/health/liveliness", timeout=2) as r:
                            if r.status == 200:
                                break
                    except OSError:
                        pass
                    if proc.poll() is not None:
                        return {"error": log.read_text()[-3000:]}
                    time.sleep(0.5)
                for shape in ("chat", "responses", "messages"):
                    for t in TASKS:
                        path, body = shaped(shape, t["task"], tools, used)
                        t0 = time.perf_counter()
                        status, answer = _post(base + path, body)
                        seconds = round(time.perf_counter() - t0, 2)
                        received = got[-1]["tools"] if got else []
                        names = [
                            x.get("name") or (x.get("function") or {}).get("name") or x.get("type")
                            for x in received
                        ]
                        out["requests"].append(
                            {
                                "shape": shape,
                                "task": t["task"],
                                "status": status,
                                "sent": len(body["tools"]),
                                "received": names,
                                "has_tool": t["tool"] in names,
                                "kept_used": used["api_name"] in names,
                                "seconds": seconds,
                            }
                        )
                        if status != 200:
                            out["requests"][-1]["answer"] = answer
                out["mcp_gateway"] = asyncio.run(gateway(f"{base}/mcp"))
            except Exception as e:  # a failure here is a result
                out["error"] = f"{type(e).__name__}: {e}"
                out["log"] = log.read_text()[-3000:]
            finally:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(20)
                except subprocess.TimeoutExpired:
                    proc.kill()
    stub.shutdown()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data")
    p.add_argument("--emb-url")
    p.add_argument(
        "--emb-model",
        default="qwen3-emb",
        help="served name at --emb-url; 8091 serves qwen3-emb, the base the v0.1 heads belong to",
    )
    p.add_argument("--heads", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--only", action="append", choices=PARTS, help="run this part (repeatable); default: all")
    p.add_argument("--part", choices=["bigtool"], help=argparse.SUPPRESS)
    p.add_argument("--url", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.part == "bigtool":
        print(json.dumps(bigtool(a.url), ensure_ascii=False))
        return
    if not a.data or not a.emb_url:
        p.error("--data and --emb-url are required")
    data = Path(a.data).resolve()
    start = datetime.now(UTC).isoformat(timespec="milliseconds")
    stub = stand_in()
    report: dict[str, Any] = {"data": str(data), "emb_url": a.emb_url}
    with served(data, a.emb_url, a.emb_model, a.heads, stub.server_address[1]) as url:
        parts = {
            "langgraph_bigtool": lambda: run_bigtool(url),
            "langchain": lambda: langchain_agent(url),
            "llamaindex": lambda: asyncio.run(llamaindex(url)),
            "litellm": lambda: litellm_proxy(url),
        }
        for name, part in parts.items():
            if a.only and name not in a.only:
                continue
            t0 = time.perf_counter()
            report[name] = part()
            print(f"--- {name} ({time.perf_counter() - t0:.1f} s)")
            print(json.dumps(report[name], indent=2, ensure_ascii=False)[:4000])
    stub.shutdown()
    events = [
        json.loads(line)
        for f in sorted((data / "usage").glob("usage-*.jsonl"))
        for line in f.read_text().splitlines()
    ]
    calls = [e for e in events if e["ts"] >= start and e["event"] == "call"]
    report["usage_calls"] = [
        {"tool": e["tool"], "session": e.get("session"), "link": e["link"], "outcome": e["outcome"]}
        for e in calls
    ]
    print(json.dumps({"calls": len(calls), "links": dict(Counter(c["link"] for c in calls))}))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
