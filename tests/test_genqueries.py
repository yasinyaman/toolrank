"""``toolrank data gen-queries``: a selection set written by a chat model for one's own catalogue."""

import json

import pytest

from toolrank.cli import main
from toolrank.datasets.genqueries import (
    INSTRUCTION,
    MAX_CHARS,
    STYLES,
    SYSTEM,
    build,
    clean,
    gen_queries,
    names_tool,
    sample,
    user_prompt,
)
from toolrank.datasets.jsonl import load_queries, load_tools, write_queries, write_tools
from toolrank.domain import Query, Tool


def _tools(sizes):
    return [
        Tool(
            id=f"{s}/op{i}",
            doc={"server": s, "name": f"{s}_operation_{i}", "description": f"does {s} thing {i}"},
            documentation=json.dumps(
                {"server": s, "name": f"{s}_operation_{i}", "description": f"does {s} thing {i}"}
            ),
            category=s,
        )
        for s, n in sizes.items()
        for i in range(n)
    ]


def test_sample_spreads_over_the_sources():
    tools = _tools({"big": 400, "mid": 50, "tiny": 3})
    picked = sample(tools, 60, seed=0)
    by = {s: sum(t.category == s for t in picked) for s in ("big", "mid", "tiny")}
    assert by == {"tiny": 3, "mid": 28, "big": 29} and len({t.id for t in picked}) == 60
    assert [t.id for t in sample(tools, 60, seed=0)] == [t.id for t in picked]  # seeded
    assert [t.id for t in sample(tools, 60, seed=1)] != [t.id for t in picked]
    assert len(sample(tools, 10_000, seed=0)) == len(tools)  # never more than there is
    assert sample([], 5, seed=0) == []


def test_prompt_and_cleaning():
    (tool,) = _tools({"mail": 1})
    prompt = user_prompt(tool, "step")
    assert prompt.startswith(STYLES["step"]) and "Do not use the tool's name" in prompt
    assert prompt.endswith(tool.documentation)
    long = Tool(id="x/y", doc={"name": "y"}, documentation="d" * 9000, category="x")
    assert len(user_prompt(long, "task")) < 4500  # the tool's text is cut for the chat model's window
    assert clean('<think>hmm</think>\n"Request: Send the invoice to Ada."') == "Send the invoice to Ada."
    assert clean("  Next step: fetch order 17 ") == "fetch order 17" and clean("") == ""
    assert names_tool("please run mail operation 0 now", tool) and names_tool("call mail_operation_0", tool)
    assert not names_tool("email Ada the invoice", tool)
    assert not names_tool("get it", Tool(id="a/get", doc={"name": "get"}, category="a"))  # too short to judge


def test_build_drops_what_is_not_a_usable_request():
    tools = _tools({"mail": 6})
    styles = ["task", "step", "goal", "task", "step", "goal"]
    responses = {
        "mail/op0": "Email Ada the March invoice.",
        "mail/op1": "",
        "mail/op2": "x" * (MAX_CHARS + 1),
        "mail/op3": "Use mail_operation_3 to send it.",
        "mail/op4": "List Open Pull requests",  # a benchmark has it, in other case and spacing
        "mail/op5": "Email Ada the March invoice.",  # the same request twice
    }
    queries, dropped = build(tools, styles, responses, exclude=["list  open pull requests"])
    assert dropped == {"empty": 1, "too_long": 1, "names_the_tool": 1, "in_a_benchmark": 1, "duplicate": 1}
    (q,) = queries
    assert (q.id, q.qrels, q.task, q.category) == ("gen-task/mail/op0", {"mail/op0": 1}, "mail", "")
    assert q.instruction == INSTRUCTION


class _Chat:
    name = "chat/fake"

    def __init__(self):
        self.calls = []

    def complete(self, system, user):
        assert system == SYSTEM
        self.calls.append(user)
        n = user.rsplit("thing ", 1)[1].split('"')[0]
        return f"Please take care of matter number {n} for the {user.split(chr(34) + 'server' + chr(34) + ': ' + chr(34))[1].split(chr(34))[0]} team."


def test_gen_queries_writes_a_benchmark_dir_and_resumes(tmp_path):
    tools = _tools({"big": 40, "tiny": 2})
    write_tools(tmp_path / "cat" / "tools.jsonl", tools)
    write_queries(
        tmp_path / "bench" / "queries.jsonl",
        [Query(id="b", text="Please take care of matter number 0 for the tiny team.", qrels={})],
    )
    chat = _Chat()
    n = gen_queries(
        tmp_path / "cat",
        tmp_path / "dev",
        chat,
        n=12,
        seed=0,
        exclude=[tmp_path / "bench"],
        workers=2,
        log=lambda _: None,
    )
    assert n["tools"] == 42 and n["sampled"] == 12 and n["queries"] == 11 and n["in_a_benchmark"] == 1
    assert load_tools(tmp_path / "dev" / "tools.jsonl") == tools  # the whole catalogue is the corpus
    queries = load_queries(tmp_path / "dev" / "queries.jsonl")
    assert {q.task for q in queries} == {"big", "tiny"}
    assert {q.id.split("/", 1)[0] for q in queries} == {"gen-task", "gen-step", "gen-goal"}  # in turn
    assert all(list(q.qrels) == [q.id.split("/", 1)[1]] for q in queries)
    assert "not a benchmark" in (tmp_path / "dev" / "SOURCE.md").read_text()
    again = _Chat()
    gen_queries(tmp_path / "cat", tmp_path / "dev", again, n=12, seed=0, workers=2, log=lambda _: None)
    assert again.calls == []  # everything came from generations.jsonl
    with pytest.raises(ValueError, match="must differ"):
        gen_queries(tmp_path / "cat", tmp_path / "cat", chat)
    hard = gen_queries(
        tmp_path / "cat", tmp_path / "hard", _Chat(), n=4, styles=["situation"], workers=2, log=lambda _: None
    )
    assert hard["queries"] == 4
    assert all(q.id.startswith("gen-situation/") for q in load_queries(tmp_path / "hard" / "queries.jsonl"))
    with pytest.raises(ValueError, match="choose from task, step, goal, situation"):
        gen_queries(tmp_path / "cat", tmp_path / "x", chat, styles=["riddle"])


def test_the_cli_command_and_the_set_scores(tmp_path, monkeypatch, capsys):
    from toolrank.adapters.chat_api import OpenAIChat

    write_tools(tmp_path / "cat" / "tools.jsonl", _tools({"big": 20, "tiny": 2}))
    monkeypatch.setattr(OpenAIChat, "complete", lambda self, system, user: _Chat().complete(system, user))
    assert (
        main(
            [
                "data",
                "gen-queries",
                "--data",
                str(tmp_path / "cat"),
                "--out",
                str(tmp_path / "dev"),
                "--n",
                "9",
            ]
        )
        == 0
    )
    assert "wrote 9 queries over 22 tools (2 sources)" in capsys.readouterr().out
    assert main(["eval", "--data", str(tmp_path / "dev"), "--scorer", "bm25", "--with-inst"]) == 0
    out = capsys.readouterr().out
    assert "| big |" in out and "| tiny |" in out  # a row per source: the queries' task
    with pytest.raises(SystemExit, match="tools.jsonl not found"):
        main(["data", "gen-queries", "--data", str(tmp_path / "nope"), "--out", str(tmp_path / "x")])
