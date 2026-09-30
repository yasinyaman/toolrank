"""toolrank inside other agent stacks, all talking to ``toolrank serve`` through
``toolrank.client``:

- the tool-search hooks of Claude's Messages API (``anthropic``) and OpenAI's Responses API
  (``openai``), which never import the vendors' SDKs: responses are read as SDK objects or dicts
  alike, and what goes back is plain dicts;
- framework adapters: LangChain tools and langgraph-bigtool retrieval (``langgraph``, which
  imports langchain-core only when used), a LlamaIndex tool retriever (``llamaindex``, which needs
  llama-index-core to import) and a tool filter for the LiteLLM proxy (``litellm``, which imports
  litellm when the proxy loads it).
"""

from toolrank.integrations._common import read_only

__all__ = ["read_only"]
