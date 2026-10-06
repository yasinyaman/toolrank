# Python API

The client and the integrations are the stable Python surface. Everything else is reached through
the CLI.

::: toolrank.client.ToolrankClient

::: toolrank.client.AsyncToolrankClient

::: toolrank.client.ToolrankError

## Claude tool search

::: toolrank.integrations.anthropic.Toolbox

::: toolrank.integrations.anthropic.run

## OpenAI tool search

::: toolrank.integrations.openai.Toolbox

::: toolrank.integrations.openai.run

## LangChain

`ToolrankToolSelector` is built lazily, on first use (`langchain` is an optional extra), so the
module carries its documentation:

::: toolrank.integrations.langchain
    options:
      members:
        - Selector

## LangGraph

::: toolrank.integrations.langgraph.Toolbox

## LlamaIndex

::: toolrank.integrations.llamaindex.ToolrankToolRetriever

## LiteLLM

::: toolrank.integrations.litellm.ToolFilter
