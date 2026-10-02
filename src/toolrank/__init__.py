"""Toolrank: tool retrieval for LLM agents with hundreds of tools.

The package is organised as ports and adapters:

* ``toolrank.domain``       - plain dataclasses (Tool, Query, RankedList, EvalReport).
* ``toolrank.ports``        - Protocols every adapter implements (Scorer, TextEncoder, VectorIndex, ...).
* ``toolrank.adapters``     - BM25, dense (any OpenAI-compatible embeddings endpoint), heads, indexes,
  the MCP proxy and REST API, MCP and OpenAPI backends.
* ``toolrank.ingest``       - MCP servers and OpenAPI specs -> an ingest dir (``tools.jsonl``).
* ``toolrank.retriever``    - query-time retrieval over an ingest dir, shared by search and serve.
* ``toolrank.client``       - a standard-library client for ``toolrank serve``'s REST API.
* ``toolrank.integrations`` - Claude's and OpenAI's tool search, LangGraph, LlamaIndex, LiteLLM.
* ``toolrank.datasets``     - benchmark converters and the on-disk JSONL format.
* ``toolrank.eval``         - trec_eval-compatible metrics and the benchmark runner.
"""

from toolrank.domain import EvalReport, Query, RankedList, Tool

__all__ = ["EvalReport", "Query", "RankedList", "Tool", "__version__"]
__version__ = "0.2.0"
