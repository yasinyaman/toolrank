import json

import pytest

from toolrank.datasets.livemcpbench import INSTRUCTION, convert, tool_names
from toolrank.formats import tool_format


def test_livemcpbench_tool_names():
    assert tool_names("1. get-weread-rank\n2. generate_word_cloud_chart\n") == [
        "get-weread-rank",
        "generate_word_cloud_chart",
    ]
    assert tool_names("") == [] and tool_names("`search`") == ["search"]


def test_livemcpbench_convert_maps_names_to_every_server_and_drops_unmatched_tasks():
    schema = {"type": "object", "properties": {"symbol": {"type": "string", "description": "ticker"}}}
    servers = [
        {
            "name": "Yahoo",
            "tools": {
                "yahoo": {"tools": [{"name": "price", "description": "Stock price.", "inputSchema": schema}]}
            },
        },
        {"name": "Other", "tools": {"other": {"tools": [{"name": "price", "description": "Also a price."}]}}},
    ]
    tasks = [
        {
            "task_id": "t1",
            "Question": "AAPL price?",
            "category": "Finance",
            "Annotator Metadata": {"Tools": "1. price"},
        },
        {
            "task_id": "t2",
            "Question": "?",
            "category": "Office",
            "Annotator Metadata": {"Tools": "1. missing_tool"},
        },
    ]
    tools, queries, dropped = convert(servers, tasks)
    assert [t.id for t in tools] == ["yahoo/price", "other/price"] and tools[0].category == "Yahoo"
    assert tool_format("schema")(tools[0]) == "price: Stock price.\n  symbol (string) - ticker"
    assert dropped == 1 and len(queries) == 1
    q = queries[0]
    assert q.qrels == {"yahoo/price": 1, "other/price": 1}
    assert (q.task, q.category, q.instruction) == ("Finance", "Finance", INSTRUCTION)


def test_mcp_zero_input_schema_expands_condensed_parameters():
    from toolrank.datasets.mcp_zero import input_schema

    s = input_schema(
        {
            "q": "(string) Search terms",
            "n": "(Optional, integer) Max results",
            "raw": "free text",
            "a": "(string, required) A",
            "b": "(optional number, default: 10) B",
            "c": "(Required) C",
        }
    )
    assert s == {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "Search terms"},
            "n": {"type": "integer", "description": "Max results"},
            "raw": {"description": "free text"},
            "a": {"type": "string", "description": "A"},
            "b": {"type": "number, default: 10", "description": "B"},
            "c": {"description": "C"},
        },
        "required": ["q", "raw", "a", "c"],
    }
    assert input_schema({}) is None and input_schema(None) is None


def _mcp_servers():
    tool = {"name": "send", "description": "Send a message", "parameter": {"text": "(string) Body"}}
    return [
        {
            "name": "Slack",
            "readme_file": "reference/slack.md",
            "description": "Slack {workspace}",
            "tools": [tool, {"name": "send", "description": "same server, same name"}],
        },
        {"name": "Slack", "readme_file": "community/Slack.md", "description": "Slack", "tools": [tool]},
        {"name": "Mail", "readme_file": "community/Mail.md", "description": "Email", "tools": [tool]},
        {"name": "Empty", "readme_file": "community/Empty.md", "description": "no tools", "tools": []},
    ]


def test_mcp_zero_convert_keys_tools_by_readme_and_builds_prompts():
    from toolrank.datasets.mcp_zero import convert

    tools, prompts, duplicates = convert(_mcp_servers())
    assert [t.id for t in tools] == ["reference/slack/send", "community/Slack/send", "community/Mail/send"]
    assert duplicates == 1 and [t.category for t in tools] == ["Slack", "Slack", "Mail"]
    assert tool_format("schema")(tools[0]) == "send: Send a message\n  text (string) - Body"
    assert (
        prompts["reference/slack/send"] == "I need to Send a message with a MCP server of Slack {workspace}."
    )


def test_mcp_zero_queries_count_same_server_and_tool_name_as_relevant():
    from toolrank.datasets.mcp_zero import INSTRUCTION, convert, make_queries, parse_request

    ok = "<think>\n\n</think>\n<tool_assistant>\nserver: Slack chat\ntool: send a message\n</tool_assistant>"
    assert parse_request(ok) == ("Slack chat", "send a message")
    assert parse_request("I would use Slack.") is None
    tools, _, _ = convert(_mcp_servers())
    responses = {"reference/slack/send": ok, "community/Mail/send": "  just email it  "}
    queries, unparsed = make_queries(tools, responses)
    assert unparsed == 1
    assert [(q.id, q.text, q.qrels) for q in queries] == [
        (
            "reference/slack/send",
            "server: Slack chat\ntool: send a message",
            {"reference/slack/send": 1, "community/Slack/send": 1},
        ),
        ("community/Mail/send", "just email it", {"community/Mail/send": 1}),
    ]
    assert queries[0].instruction == INSTRUCTION and queries[0].category == ""


def test_mcp_zero_generate_resumes_from_its_cache(tmp_path):
    from toolrank.datasets.mcp_zero import SYSTEM_PROMPT, generate

    class FakeChat:
        name = "chat/fake"

        def __init__(self):
            self.calls = []

        def complete(self, system, user):
            assert system == SYSTEM_PROMPT
            self.calls.append(user)
            return f"<tool_assistant>server: s\ntool: {user}</tool_assistant>"

    cache = tmp_path / "generations.jsonl"
    first = FakeChat()
    assert generate(first, {"a": "pa", "b": "pb"}, cache, workers=2, log=lambda _: None) == {
        "a": "<tool_assistant>server: s\ntool: pa</tool_assistant>",
        "b": "<tool_assistant>server: s\ntool: pb</tool_assistant>",
    }
    again = FakeChat()  # same model: only the new prompt and the changed one are asked
    out = generate(again, {"a": "pa", "b": "pb2", "c": "pc"}, cache, workers=2, log=lambda _: None)
    assert sorted(again.calls) == ["pb2", "pc"] and set(out) == {"a", "b", "c"}


def test_server_name_sets_are_faz0s_byte_for_byte(tmp_path):
    import random

    from toolrank.cli import main
    from toolrank.datasets.jsonl import write_queries, write_tools
    from toolrank.datasets.synthetic import make_queries, make_tools

    tools = make_tools(12, random.Random(1))
    src = tmp_path / "bench"
    write_tools(src / "tools.jsonl", tools)
    write_queries(src / "queries.jsonl", make_queries(tools, 5, random.Random(2)))
    assert main(["data", "server-names", str(src)]) == 0

    want = []  # the snippet in docs/reports/faz0-week4.md that made data/*_server on the GB10
    for line in (src / "tools.jsonl").read_text().splitlines(keepends=True):
        r = json.loads(line)
        doc = {"server": r["category"], **r["doc"]}
        r["doc"], r["documentation"] = doc, json.dumps(doc, ensure_ascii=False)
        want.append(json.dumps(r, ensure_ascii=False) + "\n")
    out = tmp_path / "bench_server"
    assert (out / "tools.jsonl").read_text() == "".join(want)
    assert (out / "queries.jsonl").read_bytes() == (src / "queries.jsonl").read_bytes()
    assert json.loads(want[0])["documentation"].startswith('{"server": ')
    with pytest.raises(SystemExit, match="must differ"):
        main(["data", "server-names", str(src), "--out", str(src)])
