"""A dev set of one's own (``toolrank data gen-queries``): requests written by a chat model for the
tools of any catalogue.

A model picked on a benchmark cannot be reported on it (the v0.1 fine-tunes and the LoRA backbone
were picked on MCP-Zero, so its column is their selection set). A set made here shares no query
with any benchmark: take a catalogue the benchmarks do not have (an ingest dir of your own tools),
sample tools evenly over its sources, and let a ``ChatModel`` write one request per sampled tool
whose answer is that tool. The whole catalogue is the corpus, so the result is a benchmark-format
dir (``tools.jsonl`` + ``queries.jsonl``) for ``toolrank eval``, ``finetune --dev``, ``learn --dev``
and ``scripts/lora_train.py --dev``.

Requests come in styles, used in turn (``STYLES``; by default a person's task with specifics, an
agent's terse note for its next step, and a goal in everyday words; ``situation`` states a problem
without the operation and is the hardest to match). The style is in the query's id
(``gen-<style>/<tool id>``) and the tool's source is its ``task``, so reports break down by source.
The model is told not to use the tool's name; a request that names it anyway is dropped, as are
empty ones and those equal to a query of an ``exclude`` set. One gold tool per request: a catalogue
with near-identical tools (two operations that list the same thing) makes some requests ambiguous,
which lowers every model's top-1 alike. A selection set, not a benchmark: its numbers order models,
they are not reported next to published ones.

What it can and cannot tell (``docs/reports/faz2-week7.md``): on a catalogue of two large APIs,
requests written this way are easy for an embedding model (Recall@5 98-99% for every backbone we
have), so it separates models only where one is clearly worse; it is a guard against a model that
got worse on your catalogue, not a ruler for small gains.

Responses are cached in ``<out>/generations.jsonl`` (``mcp_zero.generate``: keyed by model and
prompt), so a rerun generates only what is new.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from toolrank.datasets.jsonl import load_queries, load_tools, write_queries, write_tools
from toolrank.datasets.mcp_zero import generate
from toolrank.domain import Query, Tool
from toolrank.formats import tool_format
from toolrank.ports import ChatModel

INSTRUCTION = "Given an agent's request for a tool, retrieve the MCP tool that fulfills it."
SYSTEM = (
    "You write the requests that people and AI agents make when they need a tool. "
    "Answer with the request only: no quotes, no label, no explanation."
)
STYLES = {
    "task": "Write it as a person asking an assistant to get something done: one or two sentences, with "
    "made-up but plausible specifics (names, ids, dates, amounts) where the tool needs them.",
    "step": "Write it as the short note an agent writes to itself for its next step: one terse line saying "
    "what has to be done now, with plausible specifics.",
    "goal": "Write it as a person who does not know the system describing what they want to achieve, in "
    "everyday words, in one or two sentences.",
    "situation": "Write what a person says when they have a problem or a need that this tool resolves: "
    "describe the situation and what they are after in one or two sentences, with plausible specifics, "
    "without saying which operation to perform.",
}
DEFAULT_STYLES = ("task", "step", "goal")
RULES = (
    "The request must be one that this tool fulfills and that is specific enough to tell it from similar "
    "tools. Do not use the tool's name and do not copy phrases of its description; you may name the "
    "product or service when a person naturally would."
)
TOOL_CHARS = 3500  # of the tool's text in the prompt: the chat model's window is small
MAX_CHARS = 600  # a longer answer is an explanation, not a request
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_LABEL = re.compile(r"^(request|note|next step|goal|task)\s*:\s*", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")


def sample(tools: Sequence[Tool], n: int, seed: int) -> list[Tool]:
    """``n`` tools spread evenly over the sources (``Tool.category``): every source gives an equal
    share, a source smaller than its share gives all it has and the rest is shared again, so two
    huge APIs do not drown the small servers. Seeded; the order is source by source."""
    by_source: dict[str, list[Tool]] = defaultdict(list)
    for t in tools:
        by_source[t.category].append(t)
    rng = random.Random(seed)
    left, out = min(n, len(tools)), []
    sources = sorted(by_source, key=lambda s: (len(by_source[s]), s))
    for i, source in enumerate(sources):
        share = left // (len(sources) - i)
        take = (
            min(share, len(by_source[source])) if i < len(sources) - 1 else min(left, len(by_source[source]))
        )
        out += rng.sample(by_source[source], take)
        left -= take
    return out


def user_prompt(tool: Tool, style: str) -> str:
    text = tool_format("documentation")(tool) or tool_format("name_desc")(tool)
    return f"{STYLES[style]}\n{RULES}\n\nThe tool:\n{text[:TOOL_CHARS]}"


def clean(response: str) -> str:
    """The request in a response: thinking, quotes and a leading label removed."""
    text = _THINK.sub("", response or "").strip().strip("\"'`“”").strip()
    return _LABEL.sub("", text).strip()


def names_tool(text: str, tool: Tool) -> bool:
    """Whether the request spells the tool's name out (its words, run together or apart): the
    model was told not to, and such a request is found by name, not by meaning."""
    name = "".join(_WORD.findall(tool.name.lower()))
    return len(name) >= 6 and name in "".join(_WORD.findall(text.lower()))


def build(
    picked: Sequence[Tool], styles: Sequence[str], responses: dict[str, str], exclude: Iterable[str] = ()
) -> tuple[list[Query], dict[str, int]]:
    """Queries from the responses -> (queries, counts of what was dropped and why)."""
    taken = {" ".join(x.lower().split()) for x in exclude}
    counts = {"empty": 0, "too_long": 0, "names_the_tool": 0, "in_a_benchmark": 0, "duplicate": 0}
    queries: list[Query] = []
    for tool, style in zip(picked, styles, strict=True):
        if tool.id not in responses:
            continue
        text = clean(responses[tool.id])
        key = " ".join(text.lower().split())
        reason = (
            "empty"
            if not text
            else "too_long"
            if len(text) > MAX_CHARS
            else "names_the_tool"
            if names_tool(text, tool)
            else "in_a_benchmark"
            if key in taken
            else None
        )
        if reason is None and any(q.text == text for q in queries):
            reason = "duplicate"
        if reason is not None:
            counts[reason] += 1
            continue
        queries.append(
            Query(
                id=f"gen-{style}/{tool.id}",
                text=text,
                qrels={tool.id: 1},
                instruction=INSTRUCTION,
                task=tool.category,
            )
        )
    return queries, counts


def gen_queries(
    data_dir: str | Path,
    out_dir: str | Path,
    chat: ChatModel,
    *,
    n: int = 600,
    seed: int = 0,
    exclude: Sequence[str | Path] = (),
    styles: Sequence[str] = DEFAULT_STYLES,
    workers: int = 32,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Write ``out_dir`` (the catalogue of ``data_dir`` as ``tools.jsonl``, the generated
    ``queries.jsonl``, ``SOURCE.md``); -> counts. ``exclude``: benchmark dirs whose queries the
    set must not repeat."""
    data, out = Path(data_dir), Path(out_dir)
    if data.resolve() == out.resolve():
        raise ValueError("--out must differ from --data: the set gets its own directory")
    unknown = [x for x in styles if x not in STYLES]
    if unknown or not styles:
        raise ValueError(
            f"--styles: choose from {', '.join(STYLES)}" + (f" (not {unknown})" if unknown else "")
        )
    raw = (data / "tools.jsonl").read_bytes()
    tools = load_tools(data / "tools.jsonl")
    picked = sample(tools, n, seed)
    turn = [styles[i % len(styles)] for i in range(len(picked))]
    prompts = {t.id: user_prompt(t, s) for t, s in zip(picked, turn, strict=True)}
    responses = generate(chat, prompts, out / "generations.jsonl", workers=workers, log=log, system=SYSTEM)
    taken = [q.text for d in exclude for q in load_queries(Path(d) / "queries.jsonl")]
    queries, dropped = build(picked, turn, responses, taken)
    write_tools(out / "tools.jsonl", tools)
    write_queries(out / "queries.jsonl", queries)
    sources = sorted({t.category for t in tools})
    (out / "SOURCE.md").write_text(
        f"A generated dev set (toolrank data gen-queries): a selection set, not a benchmark.\n"
        f"catalogue: {data} ({len(tools)} tools of {len(sources)} sources; tools.jsonl sha256 "
        f"{hashlib.sha256(raw).hexdigest()})\n"
        f"queries: {len(queries)} of {len(picked)} sampled tools (seed {seed}), one per tool, written by "
        f"{chat.name} (generations.jsonl), styles {', '.join(styles)} in turn; dropped: {dropped}\n"
        f"checked against the queries of: {', '.join(str(d) for d in exclude) or 'nothing'}\n"
    )
    return {
        "tools": len(tools),
        "sources": len(sources),
        "sampled": len(picked),
        "queries": len(queries),
        **dropped,
    }
