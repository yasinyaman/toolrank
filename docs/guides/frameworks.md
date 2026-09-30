# LangGraph, LlamaIndex and LiteLLM

With `toolrank serve` running, the adapters reach it through `toolrank.client.ToolrankClient`, and
`serve` runs the tools.

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
