"""Domain objects shared by every port and adapter. No I/O, no framework imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Tool:
    """One tool (an MCP tool, an OpenAPI operation, a Python function...).

    ``doc`` is the structured description when one is known (``name``, ``description``,
    ``parameters`` ...); ``documentation`` is the raw text the source dataset ships. A
    ``ToolFormatter`` turns either into the text a scorer sees.
    """

    id: str
    doc: dict[str, Any] = field(default_factory=dict)
    documentation: str = ""
    category: str = ""  # e.g. web / code / customized (ToolRet), or an MCP server name

    @property
    def name(self) -> str:
        return str(self.doc.get("name") or self.id)

    @property
    def description(self) -> str:
        return str(self.doc.get("description") or "")


@dataclass(frozen=True, slots=True)
class Query:
    """A retrieval task: the user request (state), an optional instruction and the gold tools.

    ``qrels`` maps tool id -> graded relevance (>0 means relevant), exactly what trec_eval expects.
    """

    id: str
    text: str
    qrels: dict[str, int]
    instruction: str = ""
    task: str = ""  # source task / sub-benchmark name, used for per-task reporting
    category: str = ""  # group of tasks (ToolRet: web / code / customized) for category_macro


@dataclass(frozen=True, slots=True)
class TrainPair:
    """A training example: a request (state), the tools it should retrieve and hard negatives.

    Tools are raw documentation strings (as a dataset ships them); they are parsed and formatted
    like corpus tools when the pair is embedded, so training and evaluation see the same text.
    """

    id: str
    text: str
    positives: tuple[str, ...]
    negatives: tuple[str, ...] = ()
    instruction: str = ""


@dataclass(slots=True)
class RankedList:
    """Top-k result for one query, best first. ``scores`` are the scorer's own units."""

    query_id: str
    tool_ids: list[str]
    scores: list[float]

    def top(self, k: int) -> list[str]:
        return self.tool_ids[:k]


@dataclass(slots=True)
class EvalReport:
    """Metrics for one run: overall (micro-average over queries) and per task.

    When queries carry a ``category``, also ``per_category`` (plain mean over each category's
    tasks) and ``category_macro`` (plain mean over categories): the ToolRet paper's "Average",
    which ignores how many queries a task has. Empty otherwise.
    """

    scorer: str
    dataset: str
    n_queries: int
    n_tools: int
    overall: dict[str, float]
    per_task: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    config: dict[str, Any] = field(default_factory=dict)
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    category_macro: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorer": self.scorer,
            "dataset": self.dataset,
            "n_queries": self.n_queries,
            "n_tools": self.n_tools,
            "overall": self.overall,
            "category_macro": self.category_macro,
            "per_task": self.per_task,
            "per_category": self.per_category,
            "latency_ms": self.latency_ms,
            "config": self.config,
        }
