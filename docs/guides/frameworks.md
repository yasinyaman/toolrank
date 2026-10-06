# LangGraph, LangChain, LlamaIndex, LiteLLM and Strands

With `toolrank serve` running, the adapters reach it through `toolrank.client.ToolrankClient`, and
`serve` runs the tools. (Strands needs no adapter; it connects over MCP, see the last section.)

## LangGraph

`pip install "toolrank[langgraph]"` (langchain-core). `Toolbox.registry()` is every catalogue tool as
a LangChain tool that toolrank runs; `retrieve_tools` is the retrieval step of
[langgraph-bigtool](https://github.com/langchain-ai/langgraph-bigtool). It returns registry keys only,
as many as toolrank's adaptive K keeps.

```python
from langgraph_bigtool import create_agent
from toolrank.client import ToolrankClient
from toolrank.integrations.langgraph import Toolbox

box = Toolbox(ToolrankClient())
agent = create_agent(llm, box.registry(), retrieve_tools_function=box.retrieve_tools,
                     retrieve_tools_coroutine=box.aretrieve_tools).compile()
```

- langgraph-bigtool 0.0.3, the release on PyPI, needs `langgraph<1`; the adapter itself needs only
  langchain-core.
- A tool that fails, a call toolrank refuses and a call your `approve` callback declines become
  error results the model reads.
- Each LangGraph thread (`configurable.thread_id`) is its own session in the usage log.

## LangChain agents

`pip install "toolrank[langchain]"` (LangChain 1.x). `ToolrankToolSelector` is a middleware for
`create_agent`: before each model call, it narrows the tools the model sees to the ones toolrank finds
for the last user message.

**Your agent's own tools.** Any LangChain tools are ranked through `/v1/rank`, and the model sees what
toolrank's adaptive K keeps. This does the job of LangChain's `LLMToolSelectorMiddleware` without the
extra model call.

```python
from langchain.agents import create_agent
from toolrank.integrations.langchain import ToolrankToolSelector

agent = create_agent(model, tools=my_tools, middleware=[ToolrankToolSelector()])
```

**toolrank's catalogue.** Give the agent `Toolbox.registry()` and the middleware the toolbox. The
catalogue tools are then picked by a toolrank search over the server's index. Each call is linked to
that search in the usage log, which [`toolrank learn`](learn.md) reads. The agent's other tools are
shown as they are.

```python
from toolrank.client import ToolrankClient
from toolrank.integrations.langgraph import Toolbox

box = Toolbox(ToolrankClient())
agent = create_agent(model, tools=[*box.registry().values(), *my_tools],
                     middleware=[ToolrankToolSelector(toolbox=box)])
```

- The model always keeps the tools the conversation already called, any tool `tool_choice` names,
  `always_include` and provider tools such as web search. Shown tools keep the agent's order.
- Every tool is shown when there are 20 or fewer to choose from (`min_tools`), when a tool is deferred
  to the provider's own tool search, or when there is no user text.
- If toolrank fails or takes longer than `timeout_s` (5 seconds), the model gets every tool of its
  own; with a `toolbox`, the catalogue tools of that session's last selection instead (a catalogue of
  hundreds is more than a model takes in one request). The selection still finishes in the
  background, so a later call can use it.
- An agent makes several model calls per user message. A selection is kept for `ttl_s` seconds, and
  calls that come while it is being made wait for it, so toolrank is asked once.
- In our end-to-end run on 1,862 catalogue tools, each task's model call saw 2 tools, the right one
  among them.

## LlamaIndex

`pip install "toolrank[llamaindex]"` (llama-index-core). `ToolrankToolRetriever` is an
`ObjectRetriever`: as a workflow agent's `tool_retriever`, it gives each step the tools a toolrank
search finds.

```python
from llama_index.core.agent.workflow import FunctionAgent
from toolrank.client import ToolrankClient
from toolrank.integrations.llamaindex import ToolrankToolRetriever

agent = FunctionAgent(llm=llm, tool_retriever=ToolrankToolRetriever(ToolrankClient()))
print(await agent.run("What time is it in Tokyo?"))
```

LlamaIndex agents retrieve with the user message at every step and with the tool's name before
each call. The retriever keeps a query's tools for `ttl_s` seconds, so the steps don't repeat the
search, and answers a name it handed out with that tool.

## LiteLLM proxy

Two ways in; `examples/litellm/config.yaml` shows both.

**A tool filter.** Install toolrank into the proxy's environment and add the callback:

```yaml
litellm_settings:
  callbacks: toolrank.integrations.litellm.tool_filter
```

A request that carries many tools (Chat Completions, Responses or Anthropic Messages) reaches the
model with the ones toolrank ranks relevant to the last user message. It always keeps the tools the
conversation already called, any tool `tool_choice` names, and provider tools such as web search.

It leaves a request alone when it has 20 or fewer function tools, uses the provider's own tool
search, or has no user text. If toolrank is slow or down (2 seconds), the request goes on
unfiltered.

Settings come from `TOOLRANK_URL`, `TOOLRANK_API_KEY` and `TOOLRANK_FILTER_MIN_TOOLS`, `_MAX_TOOLS`,
`_MARGIN` and `_TIMEOUT`. In our end-to-end run, 120-tool requests reached the model with 3 function
tools in all three API shapes.

The first request with a new 120-tool list took 11.5 s while toolrank embedded the tools, so it went
on unfiltered. `TOOLRANK_FILTER_WARM` names a JSON file with the lists you know: one tools array or
a list of them, in any of the three shapes. The filter ranks each of them once in the background
when the proxy starts, so their first request is filtered as well. toolrank keeps such tools in
memory, so after toolrank restarts, restart the proxy too, or call `ToolFilter.warm_up()`.

**toolrank behind LiteLLM's MCP gateway.** List `toolrank serve` as an MCP server; clients of the
gateway get `toolrank-search_tools` and `toolrank-call_tool`:

```yaml
mcp_servers:
  toolrank:
    url: http://127.0.0.1:8765/mcp
    transport: http
```

Tools that LiteLLM's MCP gateway adds on its own are added after the filter runs; for those, this
second way is the one that works.

## Strands Agents

No adapter: `toolrank serve` speaks streamable-HTTP MCP, and a [Strands](https://strandsagents.com)
agent connects to it like to any MCP server. The agent gets `search_tools` and `call_tool`; the
model searches the catalogue and calls what the search found, and the calls land in the usage log
[`toolrank learn`](learn.md) reads.

```python
import os

from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp import MCPClient

client = MCPClient(
    lambda: streamablehttp_client(
        "http://127.0.0.1:8765/mcp",
        headers={"Authorization": f"Bearer {os.environ['TOOLRANK_API_KEY']}"},
    )
)

with client:
    agent = Agent(tools=client.list_tools_sync())
    agent("Refund the last payment of customer 42")
```

- strands-agents pins `mcp<2.2` and `toolrank[mcp]` needs `mcp>=2.2`: run the agent in its own
  environment and process; toolrank serve runs unchanged in its own.
- Verified with strands-agents 1.57.2: the tool list, `search_tools`, and one MCP and one OpenAPI
  `call_tool` end to end, no change on toolrank's side.
- A strands-harness deployment takes the same server as `create_harness(mcp_servers=[...])`.
