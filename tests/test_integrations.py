"""The platform integrations against the vendors' real SDKs, whose HTTP goes to scripted answers,
and a real toolrank (the served app in process, the stdio fixture MCP server behind it)."""

import json

import pytest

pytest.importorskip("mcp")
anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")

import httpx2  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

from test_client import _toolrank  # noqa: E402
from test_rest import _events  # noqa: E402
from toolrank.integrations import anthropic as claude  # noqa: E402
from toolrank.integrations import openai as gpt  # noqa: E402
from toolrank.integrations._common import read_only  # noqa: E402


class _Script:
    """Canned API answers, one per request, and the JSON bodies the SDK sent."""

    def __init__(self, *answers):
        self.answers, self.sent = list(answers), []

    def __call__(self, request):
        self.sent.append(json.loads(request.content))
        return httpx2.Response(200, json=self.answers.pop(0))


def _message(stop, *content):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": list(content),
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _tool_use(tool_use_id, name, arguments):
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": arguments}


def _claude(script):
    return anthropic.Anthropic(
        api_key="test", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(script))
    )


def _calls(tmp_path):
    return [e for e in _events(tmp_path) if e["event"] == "call"]


def test_claude_searches_gets_references_and_calls_through_toolrank(tmp_path):
    script = _Script(
        _message(
            "tool_use",
            {"type": "text", "text": "Let me find a tool."},
            _tool_use("tu_1", "search_tools", {"query": "add two integers"}),
        ),
        _message("tool_use", _tool_use("tu_2", "fx__add", {"a": 2, "b": 3})),
        _message("end_turn", {"type": "text", "text": "2 + 3 = 5"}),
    )
    events = []
    with _toolrank(tmp_path) as tr:
        box = claude.Toolbox(tr, on_event=lambda kind, details: events.append(kind))
        result = claude.run(_claude(script), box, "What is 2 + 3?", model="claude-test")
    assert (result.text, result.stop_reason, result.turns) == ("2 + 3 = 5", "end_turn", 3)
    assert result.usage["input_tokens"] == 30 and events == ["turn", "search", "turn", "call", "turn"]
    first, second, third = script.sent
    search, *deferred = first["tools"]
    assert search["name"] == "search_tools" and "defer_loading" not in search
    assert {t["name"] for t in deferred} == {"fx__add", "api__getThing", "api__createThing"}
    assert all(t["defer_loading"] is True for t in deferred)
    assert first["tools"] == second["tools"] == third["tools"]  # frozen for the conversation
    assert second["messages"][:1] == first["messages"] and third["messages"][:3] == second["messages"]
    assert [b["type"] for b in second["messages"][1]["content"]] == ["text", "tool_use"]  # sent back as is
    found = second["messages"][2]["content"][0]
    assert (
        found["tool_use_id"] == "tu_1"
        and {"type": "tool_reference", "tool_name": "fx__add"} in found["content"]
    )
    assert all(block["type"] == "tool_reference" for block in found["content"])
    ran = third["messages"][4]["content"][0]
    assert (ran["tool_use_id"], ran["content"], "is_error" in ran) == (
        "tu_2",
        [{"type": "text", "text": "5"}],
        False,
    )
    assert [(c["tool"], c["link"]) for c in _calls(tmp_path)] == [("fx/add", "search_id")]
    TypeAdapter(anthropic.types.ToolParam).validate_python(deferred[0])  # what the SDK declares
    TypeAdapter(anthropic.types.ToolParam).validate_python(search)
    TypeAdapter(anthropic.types.ToolResultBlockParam).validate_python(found)


def test_nothing_runs_after_max_tokens_and_pause_turn_is_sent_back(tmp_path):
    script = _Script(
        _message("pause_turn", {"type": "text", "text": "Still working."}),
        _message("max_tokens", _tool_use("tu_1", "fx__add", {"a": 1})),  # its input may be cut off
    )
    with _toolrank(tmp_path) as tr:
        result = claude.run(_claude(script), tr, "Add 1 and 2.", model="claude-test")
    assert (result.stop_reason, result.turns) == ("max_tokens", 2)
    assert [m["role"] for m in script.sent[1]["messages"]] == ["user", "assistant"]  # no new user message
    assert _calls(tmp_path) == []


def test_declined_and_unknown_calls_are_errors_claude_reads(tmp_path):
    script = _Script(
        _message(
            "tool_use",
            _tool_use("tu_1", "api__createThing", {"name": "x"}),
            _tool_use("tu_2", "nope", {}),
            _tool_use("tu_3", "api__getThing", {"id": 7}),
        ),
        _message("end_turn", {"type": "text", "text": "Done."}),
    )
    with _toolrank(tmp_path) as tr:
        box = claude.Toolbox(tr, approve=lambda entry, arguments: read_only(entry))
        claude.run(_claude(script), box, "Make a thing, then show thing 7.", model="claude-test")
    results = script.sent[1]["messages"][-1]["content"]
    assert [(r["tool_use_id"], r.get("is_error", False)) for r in results] == [
        ("tu_1", True),
        ("tu_2", True),
        ("tu_3", False),
    ]
    assert "declined" in results[0]["content"] and "no tool named 'nope'" in results[1]["content"]
    assert [c["tool"] for c in _calls(tmp_path)] == ["api/getThing"]  # the POST never left


def test_builtin_bm25_search_is_reported_and_blocks_are_what_the_api_takes(tmp_path):
    with _toolrank(tmp_path) as tr:
        box = claude.Toolbox(tr, builtin="bm25")
    assert box.tools[0] == {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
    seen = []
    box.on_event = lambda kind, details: seen.append((kind, details))
    server_search = [
        {
            "type": "server_tool_use",
            "id": "srv_1",
            "name": "tool_search_tool_bm25",
            "input": {"query": "add"},
        },
        {
            "type": "tool_search_tool_result",
            "tool_use_id": "srv_1",
            "content": {
                "type": "tool_search_tool_search_result",
                "tool_references": [{"type": "tool_reference", "tool_name": "fx__add"}],
            },
        },
    ]
    assert box.respond({"stop_reason": "end_turn", "content": server_search}) is None
    assert seen == [("search", {"query": "add", "tools": ["fx/add"], "mode": "anthropic bm25"})]
    image = {"type": "image", "mimeType": "image/png", "data": "iVBORw0KGgo="}
    out = claude.blocks([{"type": "text", "text": ""}, image, {"type": "resource_link", "uri": "file:///x"}])
    assert [b["type"] for b in out] == ["image", "text"] and out[0]["source"]["media_type"] == "image/png"
    assert claude.blocks([]) == [{"type": "text", "text": "(the tool returned nothing)"}]
    assert len(claude.blocks([{"type": "text", "text": "x" * 30_000}])[0]["text"]) < 25_100


# -- OpenAI Responses API ---------------------------------------------------------------------------
def _response(status, *output):
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 3},
    }
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-test",
        "status": status,
        "output": list(output),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    }


def _search_call(call_id, query):
    return {
        "type": "tool_search_call",
        "id": f"ts_{call_id}",
        "call_id": call_id,
        "execution": "client",
        "status": "completed",
        "arguments": {"query": query},
    }


def _function_call(call_id, name, arguments):
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _said(text):
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _gpt(script):
    return openai.OpenAI(
        api_key="test", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(script))
    )


def test_gpt_searches_loads_tools_and_calls_through_toolrank(tmp_path):
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"}
    script = _Script(
        _response("completed", reasoning, _search_call("call_s1", "add two integers")),
        _response("completed", _function_call("call_f1", "fx__add", '{"a": 2, "b": 3}')),
        _response("completed", _search_call("call_s2", "add numbers")),  # finds what is loaded already
        _response("completed", _said("2 + 3 = 5")),
    )
    events = []
    with _toolrank(tmp_path) as tr:
        box = gpt.Toolbox(tr, on_event=lambda kind, details: events.append(kind))
        result = gpt.run(_gpt(script), box, "What is 2 + 3?", model="gpt-test")
    assert (result.text, result.status, result.turns) == ("2 + 3 = 5", "completed", 4)
    assert result.usage == {"input_tokens": 40, "output_tokens": 20, "total_tokens": 60}
    first, second, third, fourth = script.sent
    assert first["tools"] == [gpt.search_tool()] and first["store"] is False
    assert first["include"] == ["reasoning.encrypted_content"] and first["input"] == [
        {"role": "user", "content": "What is 2 + 3?"}
    ]
    assert first["tools"] == second["tools"] == third["tools"] == fourth["tools"]
    for before, after in ((first, second), (second, third), (third, fourth)):
        assert after["input"][: len(before["input"])] == before["input"]  # append-only
    assert [i.get("type") for i in second["input"][1:]] == [
        "reasoning",
        "tool_search_call",
        "tool_search_output",
    ]
    output = second["input"][3]
    assert (output["call_id"], output["execution"], output["status"]) == ("call_s1", "client", "completed")
    names = {t["name"] for t in output["tools"]}
    assert names == {"fx__add", "api__getThing", "api__createThing"}
    add = next(t for t in output["tools"] if t["name"] == "fx__add")
    assert (add["type"], add["strict"], add["defer_loading"]) == ("function", False, True)
    assert add["parameters"]["properties"] == {"a": {"type": "integer"}, "b": {"type": "integer"}}
    assert third["input"][-1] == {"type": "function_call_output", "call_id": "call_f1", "output": "5"}
    assert fourth["input"][-1]["tools"] == []  # a tool is loaded once per conversation
    assert events.count("search") == 2 and events.count("call") == 1
    assert [(c["tool"], c["link"]) for c in _calls(tmp_path)] == [("fx/add", "search_id")]
    TypeAdapter(openai.types.responses.ToolSearchToolParam).validate_python(first["tools"][0])
    TypeAdapter(openai.types.responses.FunctionToolParam).validate_python(add)
    TypeAdapter(openai.types.responses.ResponseToolSearchOutputItemParamParam).validate_python(output)


def test_gpt_gets_the_found_tools_in_one_namespace_per_server(tmp_path):
    script = _Script(
        _response("completed", _search_call("call_s1", "add two integers")),
        _response("completed", {**_function_call("call_f1", "add", '{"a": 2, "b": 3}'), "namespace": "fx"}),
        _response("completed", _function_call("call_f2", "fx__add", "{}")),  # the flat name is not loaded
        _response("completed", _said("5")),
    )
    with _toolrank(tmp_path) as tr:
        result = gpt.run(_gpt(script), gpt.Toolbox(tr, namespaces=True), "What is 2 + 3?", model="gpt-test")
    assert result.text == "5"
    output = script.sent[1]["input"][-1]
    spaces = {ns["name"]: ns for ns in output["tools"]}
    assert set(spaces) == {"fx", "api"} and all(ns["type"] == "namespace" for ns in spaces.values())
    assert [t["name"] for t in spaces["fx"]["tools"]] == ["add"]
    assert sorted(t["name"] for t in spaces["api"]["tools"]) == ["createThing", "getThing"]
    assert spaces["fx"]["description"] == "Tools of fx (an MCP server)."
    assert spaces["api"]["description"] == "Tools of api (an HTTP API)."
    assert script.sent[2]["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_f1",
        "output": "5",
    }
    assert "no loaded tool is named 'fx__add'" in script.sent[3]["input"][-1]["output"]
    assert [(c["tool"], c["link"]) for c in _calls(tmp_path)] == [("fx/add", "search_id")]
    TypeAdapter(openai.types.responses.NamespaceToolParam).validate_python(spaces["api"])
    TypeAdapter(openai.types.responses.ResponseToolSearchOutputItemParamParam).validate_python(output)
    assert (
        gpt.own_name({"name": "github/issues/create", "server": "github", "api_name": "x"})
        == "issues__create"
    )
    assert gpt.own_name({"name": "other", "server": "github", "api_name": "other"}) == "other"


def test_gpt_bad_calls_are_outputs_and_incomplete_runs_nothing(tmp_path):
    script = _Script(
        _response("completed", _search_call("call_s1", "things")),
        _response(
            "completed",
            _function_call("call_1", "fx__add", "{not json"),
            _function_call("call_2", "api__createThing", '{"name": "x"}'),
            _function_call("call_3", "fx__unloaded", "{}"),
        ),
        _response("incomplete", _function_call("call_4", "api__getThing", '{"id": 7}')),
    )
    with _toolrank(tmp_path) as tr:
        box = gpt.Toolbox(tr, approve=lambda entry, arguments: read_only(entry))
        result = gpt.run(_gpt(script), box, "Make a thing.", model="gpt-test")
    assert (result.status, result.turns) == ("incomplete", 3)
    outputs = [i for i in script.sent[2]["input"] if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outputs] == ["call_1", "call_2", "call_3"]
    assert "not valid JSON" in outputs[0]["output"] and "declined" in outputs[1]["output"]
    assert "no loaded tool" in outputs[2]["output"]
    assert _calls(tmp_path) == []  # nothing ran, not even the GET of the incomplete response
    with pytest.raises(RuntimeError, match="failed"):
        box.respond({"status": "failed", "error": {"code": "server_error"}, "output": []})
