"""Text formats for the two sides of retrieval.

Tool side ("action" in CLM terms) - the Phase 0 ablation:

* ``documentation`` - the raw text the dataset ships (ToolRet protocol, used for the BM25 baseline).
* ``name_desc``     - ``name: description``.
* ``schema``        - name, description and every parameter with its type and description.
* ``example_call``  - a deterministic pseudo-invocation ``name(param=<type>, ...)`` followed by the
                      description, or the tool's own concrete call when it ships one (Gorilla's
                      ``api_call``). CLM's action head was post-trained on agentic trajectories, i.e.
                      concrete calls, so this is the format closest to its training distribution.

Query side ("state"):

* ``plain``          - the user request only.
* ``concat``         - ``instruction`` + request, for lexical scorers (ToolRet "w/ inst." for BM25).
* ``instruct_query`` - ``Instruct: <instruction>\\nQuery: <request>`` (e5-mistral / Qwen3-Embedding style).
* ``clm``            - request, blank line, instruction: the ``context + question`` layout the CLM
                       heads were trained on (``clm.schema.state_text``).
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from toolrank.domain import Query, Tool


def parse_doc(text: str) -> dict[str, Any]:
    """Best-effort parse of a tool document string (JSON or Python-repr dict); ``{}`` if neither."""
    s = (text or "").strip()
    if not s or s[0] not in "{[":
        return {}
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(s)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if isinstance(v, dict):
            return v
    return {}


# Where tool docs keep their parameters, first non-empty wins (MCP, OpenAI, ToolRet's UltraTool /
# GTA / Gorilla shapes); ToolBench and T-Eval split them into required and optional lists, AppBench
# into two "name (type)" dicts, and those pairs are merged.
_PARAM_KEYS = ("parameters", "input_schema", "inputSchema", "doc_arguments", "inputs", "api_arguments")
_SPLIT_PARAM_KEYS = (
    ("required_parameters", "optional_parameters"),
    ("additional_required_arguments", "optional_arguments"),
)
_NAME_TYPE = re.compile(r"^\s*([^\s(]+)\s*\(([^)]*)\)\s*$")  # AppBench keys: "where_to (str)"


def _param_entries(p: Any) -> list[tuple[str, str, str]]:
    if isinstance(p, dict) and ("properties" in p or p.get("type") in ("object", "dict")):
        p = p.get("properties") or {}  # a JSON schema; without properties it has no parameters
    out: list[tuple[str, str, str]] = []
    if isinstance(p, dict):
        for name, spec in p.items():
            name, ptype = str(name), ""
            if m := _NAME_TYPE.match(name):
                name, ptype = m.group(1), m.group(2)
            if isinstance(spec, dict):
                out.append((name, str(spec.get("type") or ptype), str(spec.get("description") or "")))
            else:
                out.append((name, ptype, "" if spec is None else str(spec)))
    elif isinstance(p, list):
        for spec in p:
            if isinstance(spec, dict) and spec.get("name"):
                out.append(
                    (str(spec["name"]), str(spec.get("type") or ""), str(spec.get("description") or ""))
                )
            elif isinstance(spec, str) and spec:
                out.append((spec, "", ""))
    return out  # anything else ("N/A", None, a number) means no parameters


def _params(doc: dict[str, Any]) -> list[tuple[str, str, str]]:
    """-> [(name, type, description)] from the shapes tool docs use: a dict of specs, a JSON-schema
    ``properties`` object, a list of ``{name, type, description}`` entries or of bare names, and
    split required / optional lists (ToolBench, T-Eval, AppBench)."""
    for key in _PARAM_KEYS:
        if doc.get(key):
            return _param_entries(doc[key])
    return [e for pair in _SPLIT_PARAM_KEYS for key in pair for e in _param_entries(doc.get(key))]


@dataclass(frozen=True)
class NamedFormatter:
    name: str
    fn: Callable[[Any], str]

    def __call__(self, x: Any) -> str:
        return self.fn(x)


def _doc_of(tool: Tool) -> dict[str, Any]:
    return tool.doc or parse_doc(tool.documentation)


def fmt_documentation(tool: Tool) -> str:
    return tool.documentation or json.dumps(tool.doc, ensure_ascii=False)


def fmt_name_desc(tool: Tool) -> str:
    d = _doc_of(tool)
    name, desc = d.get("name") or tool.id, d.get("description") or ""
    return f"{name}: {desc}".strip(": ") if desc else str(name)


def fmt_schema(tool: Tool) -> str:
    d = _doc_of(tool)
    lines = [fmt_name_desc(tool)]
    for pname, ptype, pdesc in _params(d):
        bits = [pname]
        if ptype:
            bits.append(f"({ptype})")
        if pdesc:
            bits.append(f"- {pdesc}")
        lines.append("  " + " ".join(bits))
    return "\n".join(lines)


def fmt_example_call(tool: Tool) -> str:
    d = _doc_of(tool)
    call = d.get("api_call")  # Gorilla ships a concrete call; use it rather than a placeholder one
    if not isinstance(call, str) or not call.strip():
        args = ", ".join(f"{p}=<{t or 'value'}>" for p, t, _ in _params(d))
        call = f"{d.get('name') or tool.id}({args})"
    desc = d.get("description") or ""
    return f"{call}\n{desc}".strip()


TOOL_FORMATS: dict[str, NamedFormatter] = {
    f.name: f
    for f in (
        NamedFormatter("documentation", fmt_documentation),
        NamedFormatter("name_desc", fmt_name_desc),
        NamedFormatter("schema", fmt_schema),
        NamedFormatter("example_call", fmt_example_call),
    )
}


def q_plain(q: Query) -> str:
    return q.text


def q_concat(q: Query) -> str:
    return f"{q.instruction} {q.text}".strip() if q.instruction else q.text


def q_instruct_query(q: Query) -> str:
    return f"Instruct: {q.instruction}\nQuery: {q.text}" if q.instruction else q.text


def q_clm(q: Query) -> str:
    return f"{q.text}\n\n{q.instruction}" if q.instruction else q.text


QUERY_FORMATS: dict[str, NamedFormatter] = {
    f.name: f
    for f in (
        NamedFormatter("plain", q_plain),
        NamedFormatter("concat", q_concat),
        NamedFormatter("instruct_query", q_instruct_query),
        NamedFormatter("clm", q_clm),
    )
}


def tool_format(name: str) -> NamedFormatter:
    try:
        return TOOL_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown tool format {name!r}; choose from {sorted(TOOL_FORMATS)}") from None


def query_format(name: str) -> NamedFormatter:
    try:
        return QUERY_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown query format {name!r}; choose from {sorted(QUERY_FORMATS)}") from None
