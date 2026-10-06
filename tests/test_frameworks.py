"""The framework adapters against the frameworks' real classes: a real toolrank (the served app in
process, the stdio fixture MCP server behind it), or a recording stand-in for the edge cases."""

import asyncio

import pytest

pytest.importorskip("mcp")

from test_client import _toolrank  # noqa: E402
from test_rest import _events  # noqa: E402
from toolrank.client import ToolrankError  # noqa: E402

ADD = {"type": "object", "properties": {"a": {"type": "integer"}, "config": {"type": "object"}}}


class _Stub:
    """A ``ToolrankClient`` stand-in: a fixed catalogue and search, recorded requests."""

    def __init__(self, hits=("fx__add",), result=None, fail=None):
        self.hits, self.fail, self.sent = list(hits), fail, []
        self.result = result or {
            "outcome": "ok",
            "isError": False,
            "content": [{"type": "text", "text": "5"}],
        }

    def catalog(self, server=None):
        tools = [
            {"name": "fx/add", "api_name": "fx__add", "server": "fx", "kind": "mcp", "inputSchema": ADD},
            {"name": "api/get", "api_name": "api__get", "server": "api", "kind": "openapi", "method": "GET"},
        ]
        for t in tools:
            t.setdefault("inputSchema", {"type": "object", "properties": {}})
            t["description"] = "Add two integers." if t["server"] == "fx" else ""
        return {"catalog": "c1", "count": len(tools), "tools": tools}

    def search(self, query, *, k=None, instruction=None, full_schemas=False, session=None):
        self.sent.append({"search": query, "k": k, "instruction": instruction, "session": session})
        if self.fail:
            raise ToolrankError(503, self.fail)
        known = {t["api_name"]: t for t in self.catalog()["tools"]}
        new = {"server": "zz", "kind": "mcp", "description": "New.", "inputSchema": {"type": "object"}}
        hits = [known.get(n) or {"name": n.replace("__", "/"), "api_name": n, **new} for n in self.hits]
        return {"search_id": "s-1", "mode": "semantic", "tools": hits}

    def call(self, name, arguments=None, *, search_id=None, session=None):
        self.sent.append({"call": name, "arguments": arguments, "search_id": search_id, "session": session})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _tool_call(name, args):
    return {"type": "tool_call", "id": "c1", "name": name, "args": args}


# -- LangGraph ------------------------------------------------------------------------------------


def test_langgraph_registry_and_bigtool_retrieval_run_through_toolrank(tmp_path):
    pytest.importorskip("langchain_core")
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import BaseTool, StructuredTool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from toolrank.integrations import langgraph as lg

    thread = {"configurable": {"thread_id": "t1"}}
    events = []
    with _toolrank(tmp_path) as tr:
        box = lg.Toolbox(tr, on_event=lambda kind, details: events.append(kind))
        registry = box.registry()
        assert set(registry) == {"fx__add", "api__getThing", "api__createThing"}
        assert all(isinstance(t, BaseTool) for t in registry.values())
        spec = convert_to_openai_tool(registry["fx__add"])["function"]  # what the model is sent
        assert (spec["name"], spec["description"]) == ("fx__add", "Add two integers.")
        assert spec["parameters"]["properties"] == {"a": {"type": "integer"}, "b": {"type": "integer"}}

        # langgraph-bigtool wraps the retrieval function like this, then looks each id up
        retrieve = StructuredTool.from_function(func=box.retrieve_tools, coroutine=box.aretrieve_tools)
        assert retrieve.name == "retrieve_tools" and retrieve.description.startswith(
            "Search 3 tools (api, fx)"
        )
        assert convert_to_openai_tool(retrieve)["function"]["parameters"]["required"] == ["query"]
        found = retrieve.invoke({"query": "add two integers"}, config=thread)
        assert "fx__add" in found and all(registry[i] for i in found)
        assert asyncio.run(retrieve.ainvoke({"query": "add two integers"})) == found

        added = registry["fx__add"].invoke(_tool_call("fx__add", {"a": 2, "b": 3}), config=thread)
        assert isinstance(added, ToolMessage) and (added.content, added.status) == ("5", "success")
        refused = registry["api__createThing"].invoke(_tool_call("api__createThing", {"name": "x"}))
        assert refused.status == "error" and "--allow-write" in refused.content  # the model reads why
        assert '"7"' in asyncio.run(registry["api__getThing"].ainvoke({"id": 7}))
    assert events == ["search", "search", "call", "call", "call"]
    calls = [e for e in _events(tmp_path) if e["event"] == "call"]
    assert (calls[0]["tool"], calls[0]["link"], calls[0]["session"]) == (
        "fx/add",
        "search_id",
        "rest:langgraph:t1",
    )
    assert calls[1]["session"] == calls[2]["session"] == f"rest:{box.session}"  # no thread: the toolbox's


def test_langgraph_tools_pass_every_argument_and_turn_failures_into_error_results():
    pytest.importorskip("langchain_core")
    from toolrank.integrations import langgraph as lg

    stub = _Stub(hits=["zz__new", "fx__add", "fx__add"])  # a tool newer than the catalogue snapshot
    box = lg.Toolbox(stub, k=3, instruction="Given a task", servers=["fx"], session="s0")
    assert set(box.registry()) == {"fx__add"} and box.retrieve_tools("add") == ["fx__add"]
    assert stub.sent[-1] == {"search": "add", "k": 3, "instruction": "Given a task", "session": "s0"}
    add = box.registry()["fx__add"]
    # LangChain injects run-time values into parameters named config / run_manager; ours have none
    assert add.invoke({"a": 1, "config": {"x": 1}, "run_manager": 2}) == "5"
    assert stub.sent[-1] == {
        "call": "fx/add",
        "arguments": {"a": 1, "config": {"x": 1}, "run_manager": 2},
        "search_id": "s-1",
        "session": "s0",
    }
    odd = {"configurable": {"thread_id": "t\n" + "x" * 200}}  # not a header value: hashed
    add.invoke({"a": 1}, config=odd)
    assert stub.sent[-1]["session"].startswith("langgraph:") and len(stub.sent[-1]["session"]) == 42

    failing = {"outcome": "error", "isError": True, "content": [{"type": "text", "text": "boom"}]}
    for result, expected in (
        (failing, "boom"),
        (ToolrankError(0, "server unreachable"), "toolrank could not run fx/add: server unreachable"),
        ({"outcome": "ok", "isError": False, "content": []}, "(the tool returned nothing)"),
    ):
        stub.result = result
        out = add.invoke(_tool_call("fx__add", {"a": 1}))
        assert out.content == expected and out.status == ("success" if "nothing" in expected else "error")

    declined = lg.Toolbox(stub, approve=lambda entry, args: False).registry()["fx__add"]
    n = len(stub.sent)
    assert declined.invoke(_tool_call("fx__add", {"a": 1})).content == "The user declined this call."
    assert len(stub.sent) == n  # nothing reached toolrank
    with pytest.raises(ToolrankError, match="index not ready"):  # bigtool has no way to tell the model
        lg.Toolbox(_Stub(fail="index not ready")).retrieve_tools("add")


# -- LangChain ------------------------------------------------------------------------------------


def _scripted_model(reply):
    """A chat model for ``create_agent``: ``reply(messages, tool names bound) -> AIMessage``; its
    ``seen`` lists the tool names of each call."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import Field

    class Scripted(BaseChatModel):
        seen: list = Field(default_factory=list)

        def bind_tools(self, tools, **kwargs):
            names = [t.get("name") or t["type"] if isinstance(t, dict) else t.name for t in tools]
            return self.bind(tool_names=names)

        def _generate(self, messages, stop=None, run_manager=None, tool_names=(), **kwargs):
            self.seen.append(list(tool_names))
            return ChatResult(generations=[ChatGeneration(message=reply(messages, list(tool_names)))])

        @property
        def _llm_type(self):
            return "scripted"

    return Scripted()


def _lc_request(n_tools=25, **overrides):
    """A model request: ``t0``..``t24`` and a provider tool; t20 was called already, tool_choice
    names t21, and the last message is a tool result."""
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import StructuredTool

    def tool(n):
        return StructuredTool.from_function(
            func=lambda x: f"t{n}({x})", name=f"t{n}", description=f"Tool number {n}."
        )

    request = {
        "model": _scripted_model(lambda messages, names: AIMessage("")),
        "tools": [*(tool(n) for n in range(n_tools)), {"type": "web_search"}],
        "tool_choice": "t21",
        "messages": [
            HumanMessage([{"type": "text", "text": "Refund the last payment"}]),
            AIMessage("", tool_calls=[{"name": "t20", "args": {"x": 1}, "id": "c1"}]),
            ToolMessage("ok", tool_call_id="c1"),
        ],
    }
    return ModelRequest(**{**request, **overrides})


def _names(request):
    return [t.get("type") if isinstance(t, dict) else t.name for t in request.tools]


def test_langchain_selector_keeps_the_ranked_used_chosen_and_provider_tools():
    pytest.importorskip("langchain")
    from toolrank.integrations.langchain import Selector, record

    request, ranker, seen = _lc_request(), _Ranker(), []
    sel = Selector(ranker, always_include=["t3"], on_select=seen.append)
    out = sel.select(request)
    assert _names(out) == ["t3", "t5", "t7", "t9", "t20", "t21", "web_search"]  # the agent's order
    assert len(request.tools) == 26  # the request itself is left as it was
    assert ranker.asked == [("Refund the last payment", [f"t{n}" for n in range(25)])]
    assert record(request.tools[4]) == {
        "name": "t4",
        "description": "Tool number 4.",
        "inputSchema": {"type": "object", "properties": {"x": {}}, "required": ["x"]},
    }
    # the agent's next model call in the same turn: the same choice, toolrank not asked again
    assert _names(asyncio.run(sel.aselect(request))) == _names(out)
    assert len(ranker.asked) == 1
    assert [e["cached"] for e in seen] == [False, True]
    assert seen[0] == {
        "outcome": "selected",
        "tools": 26,
        "kept": 7,
        "query": "Refund the last payment",
        "cached": False,
    }
    with_inst = Selector(ranker, instruction="Given a task")
    with_inst.select(request)  # another instruction is another choice
    assert len(ranker.asked) == 2


def test_langchain_selector_shows_every_tool_when_it_should_and_fails_open():
    pytest.importorskip("langchain")
    import time

    from langchain_core.messages import AIMessage

    from toolrank.client import ToolrankError as Down
    from toolrank.integrations.langchain import Selector

    request, ranker = _lc_request(), _Ranker()
    assert Selector(ranker, min_tools=25).select(request) is request  # few enough already
    request.tools[7].extras = {"defer_loading": True}
    assert Selector(ranker).select(request) is request  # the provider's own tool search
    request = _lc_request()
    assert Selector(ranker).select(request.override(messages=[AIMessage("hi")])).tools == request.tools
    assert ranker.asked == []  # no user text: nothing to rank for

    seen = []
    down = Selector(_Ranker(fail=Down(0, "server unreachable")), on_select=seen.append)
    assert down.select(request) is request
    assert seen[0]["outcome"] == "skipped" and "server unreachable" in seen[0]["error"]

    slow = Selector(_Ranker(delay=0.5), timeout_s=0.05)
    for select in (slow.select, lambda r: asyncio.run(slow.aselect(r))):
        t0 = time.perf_counter()
        assert select(request) is request and time.perf_counter() - t0 < 0.4  # sent as it came
    time.sleep(0.6)  # the ranking went on in the background; the next call uses it
    assert _names(slow.select(request)) == ["t5", "t7", "t9", "t20", "t21", "web_search"]


def test_langchain_selector_asks_once_per_selection_and_never_drops_one_that_waits():
    """Model calls that come while a slow selection is running wait for it instead of asking again,
    and an async timeout does not cancel selections still waiting for a worker."""
    pytest.importorskip("langchain")
    import time

    from langchain_core.messages import HumanMessage

    from toolrank.integrations.langchain import Selector

    request, ranker = _lc_request(), _Ranker(delay=0.3)
    sel = Selector(ranker, timeout_s=0.02)
    assert all(sel.select(request) is request for _ in range(3))  # all three time out, one ranking
    time.sleep(0.5)
    assert len(ranker.asked) == 1 and _names(sel.select(request))[-1] == "web_search"

    ranker = _Ranker(delay=0.2)
    sel = Selector(ranker, timeout_s=0.02)

    async def six():
        reqs = [request.override(messages=[HumanMessage(f"task {n}")]) for n in range(6)]
        return await asyncio.gather(*(sel.aselect(r) for r in reqs))

    asyncio.run(six())
    deadline = time.time() + 3
    while len(ranker.asked) < 6 and time.time() < deadline:  # four workers, then the other two
        time.sleep(0.05)
    assert sorted(q for q, _ in ranker.asked) == [f"task {n}" for n in range(6)]


def test_langchain_selector_with_a_toolbox_falls_back_to_the_last_choice_not_the_catalogue():
    pytest.importorskip("langchain")
    from langchain_core.messages import HumanMessage

    from toolrank.integrations.langchain import Selector

    class _Box:  # a Toolbox stand-in: the request's t0..t24 are its catalogue
        toolrank, fail = None, False

        def __init__(self):
            self.entries = {f"t{n}": None for n in range(25)}

        def current_session(self):
            return "thread-1"

        def retrieve_tools(self, query):
            if self.fail:
                raise RuntimeError("toolrank is down")
            return ["t4", "t8"]

    box, seen = _Box(), []
    sel = Selector(toolbox=box, on_select=seen.append)
    request = _lc_request()
    assert _names(sel.select(request)) == ["t4", "t8", "t20", "t21", "web_search"]
    box.fail = True
    later = sel.select(request.override(messages=[HumanMessage("Something else now")]))
    assert _names(later) == ["t4", "t8", "t21", "web_search"]  # last choice + tool_choice, not 25 tools
    assert seen[-1]["outcome"] == "skipped" and seen[-1]["fallback"] == "last selection"


def test_langchain_selector_ranks_through_a_served_toolrank(tmp_path):
    pytest.importorskip("langchain")
    from toolrank.integrations.langchain import Selector

    request = _lc_request()
    with _toolrank(tmp_path) as tr:
        out = Selector(tr).select(request)
    kept = _names(out)
    assert {"t20", "t21", "web_search"} <= set(kept) and 4 <= len(kept) <= 13 and len(kept) < 26


def test_langchain_agent_shows_the_catalogue_tools_a_toolbox_search_finds(tmp_path):
    pytest.importorskip("langchain")
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool

    from toolrank.integrations import langgraph as lg
    from toolrank.integrations.langchain import ToolrankToolSelector

    @tool
    def note(text: str) -> str:
        """Write a note."""
        return "noted"

    def reply(messages, names):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(f"Answer: {messages[-1].content}")
        assert "fx__add" in names
        return AIMessage("", tool_calls=[{"name": "fx__add", "args": {"a": 2, "b": 3}, "id": "c1"}])

    events, selections = [], []
    with _toolrank(tmp_path) as tr:
        box = lg.Toolbox(tr, on_event=lambda kind, details: events.append(kind))
        model = _scripted_model(reply)
        selector = ToolrankToolSelector(toolbox=box, min_tools=0, on_select=selections.append)
        agent = create_agent(model, tools=[*box.registry().values(), note], middleware=[selector])
        state = agent.invoke(
            {"messages": [HumanMessage("add two integers")]}, {"configurable": {"thread_id": "t1"}}
        )
        assert state["messages"][-1].content == "Answer: 5"
        again = {"messages": [HumanMessage("add two integers")]}
        assert asyncio.run(agent.ainvoke(again, {"configurable": {"thread_id": "t2"}}))["messages"][
            -1
        ].content == ("Answer: 5")
    shown = set(model.seen[0]) - {"note"}
    assert "note" in model.seen[0] and "fx__add" in shown and shown <= set(box.registry())
    assert all(names == model.seen[0] for names in model.seen)  # four model calls, one choice
    assert [s["cached"] for s in selections] == [False, True, False, True]  # one search per thread
    assert events == ["search", "call", "search", "call"]
    calls = [e for e in _events(tmp_path) if e["event"] == "call"]
    assert [(c["tool"], c["link"], c["session"]) for c in calls] == [
        ("fx/add", "search_id", "rest:langgraph:t1"),
        ("fx/add", "search_id", "rest:langgraph:t2"),
    ]


# -- LlamaIndex -----------------------------------------------------------------------------------


# LlamaIndex's workflows list the agent's members, a pydantic model's deprecated ones included
@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince211")
def test_llamaindex_agent_gets_its_tools_from_toolrank_and_runs_them(tmp_path):
    pytest.importorskip("llama_index.core")
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
    from llama_index.core.llms.mock import MockFunctionCallingLLM

    from toolrank.integrations.llamaindex import ToolrankToolRetriever

    offered = []

    def model(messages, **kwargs):  # calls fx__add once, then answers
        offered.append({t.metadata.name: t.metadata.to_openai_tool() for t in kwargs.get("tools", [])})
        if any(m.role == MessageRole.TOOL for m in messages):
            return ChatMessage(role=MessageRole.ASSISTANT, content=f"2 + 3 = {messages[-1].content}")
        call = ToolCallBlock(tool_call_id="c1", tool_name="fx__add", tool_kwargs={"a": 2, "b": 3})
        return ChatMessage(role=MessageRole.ASSISTANT, blocks=[call])

    async def ask(agent, question):
        return str(await agent.run(user_msg=question))

    events = []
    with _toolrank(tmp_path) as tr:
        retriever = ToolrankToolRetriever(tr, on_event=lambda kind, details: events.append((kind, details)))
        agent = FunctionAgent(llm=MockFunctionCallingLLM(response_generator=model), tool_retriever=retriever)
        assert asyncio.run(ask(agent, "What is 2 + 3?")) == "2 + 3 = 5"
    assert len(offered) == 2 and offered[0] == offered[1]  # both steps: the same toolrank search
    add = offered[0]["fx__add"]["function"]
    assert (add["description"], add["parameters"]["properties"]) == (
        "Add two integers.",
        {"a": {"type": "integer"}, "b": {"type": "integer"}},
    )
    # one search for the question (the second step reuses it), none for the lookup by name
    assert [(kind, d.get("query") or d.get("tool")) for kind, d in events] == [
        ("search", "What is 2 + 3?"),
        ("call", "fx/add"),
    ]
    (call,) = [e for e in _events(tmp_path) if e["event"] == "call"]
    assert (call["tool"], call["link"], call["session"]) == (
        "fx/add",
        "search_id",
        f"rest:{retriever.session}",
    )


def test_llamaindex_tools_keep_names_schemas_and_errors_straight():
    pytest.importorskip("llama_index.core")
    from toolrank.integrations.llamaindex import MAX_DESCRIPTION, ToolrankToolRetriever

    stub = _Stub(hits=["fx__add", "api__get"])
    retriever = ToolrankToolRetriever(stub, k=4, instruction="Given a task", session="s0")
    assert retriever.retrieve("  ") == [] and stub.sent == []
    add, get = retriever.retrieve("add numbers")
    assert stub.sent[-1] == {"search": "add numbers", "k": 4, "instruction": "Given a task", "session": "s0"}
    assert get.metadata.description == "api/get"  # no description: the tool id
    assert asyncio.run(retriever.aretrieve("add numbers")) == [add, get] and len(stub.sent) == 1  # cached
    assert retriever.retrieve("fx__add") == [add] and len(stub.sent) == 1  # a name handed out before
    retriever.retrieve("zz__new")  # an unknown name is a query like any other
    assert stub.sent[-1]["search"] == "zz__new"
    assert ToolrankToolRetriever(stub, ttl_s=0).retrieve("add numbers") and len(stub.sent) == 3

    # every argument reaches the tool, "input" too; one dict of them works as well
    out = asyncio.run(add.acall(a=1, input="x"))
    assert (out.content, out.is_error) == ("5", False)
    assert stub.sent[-1] == {
        "call": "fx/add",
        "arguments": {"a": 1, "input": "x"},
        "search_id": "s-1",
        "session": "s0",
    }
    assert add.call({"a": 2}).content == "5" and stub.sent[-1]["arguments"] == {"a": 2}
    with pytest.raises(TypeError):
        add.call(1, 2)

    failing = {"outcome": "error", "isError": True, "content": [{"type": "text", "text": "boom"}]}
    for result, expected, is_error in (
        (failing, "boom", True),
        (ToolrankError(0, "server unreachable"), "toolrank could not run fx/add: server unreachable", True),
        ({"outcome": "ok", "isError": False, "content": []}, "(the tool returned nothing)", False),
    ):
        stub.result = result
        out = add.call(a=1)
        assert (out.content, out.is_error) == (expected, is_error)

    declined = ToolrankToolRetriever(stub, approve=lambda hit, args: False).retrieve("add numbers")[0]
    n = len(stub.sent)
    assert declined.call(a=1).content == "The user declined this call." and len(stub.sent) == n

    long = _Stub(hits=["fx__add"])
    long.catalog = lambda server=None: {
        "tools": [{**_Stub().catalog()["tools"][0], "description": "x" * 5000}]
    }
    (tool,) = ToolrankToolRetriever(long).retrieve("add")
    spec = tool.metadata.to_openai_tool()["function"]  # raises above 1,024 characters
    assert (
        len(spec["description"]) == MAX_DESCRIPTION and spec["parameters"]["properties"] == ADD["properties"]
    )
    with pytest.raises(ToolrankError, match="index not ready"):
        ToolrankToolRetriever(_Stub(fail="index not ready")).retrieve("add")


# -- LiteLLM --------------------------------------------------------------------------------------

LITELLM_SCORES = {"t5": 0.9, "t7": 0.85, "t9": 0.8}  # the rest 0.3: adaptive K (margin 0.2) keeps 3


class _Ranker:
    """``ToolrankClient.rank`` stand-in: fixed scores by tool name, best first."""

    def __init__(self, fail=None, delay=0.0):
        self.fail, self.delay, self.asked = fail, delay, []

    def rank(self, query, tools, *, instruction=None, session=None):
        import time

        self.asked.append((query, [t["name"] for t in tools]))
        time.sleep(self.delay)
        if self.fail:
            raise self.fail
        rows = [
            {"index": n, "name": t["name"], "score": LITELLM_SCORES.get(t["name"], 0.3)}
            for n, t in enumerate(tools)
        ]
        return sorted(rows, key=lambda r: -r["score"])


def _requests():
    """The same conversation in the three shapes: 25 function tools and one provider tool; t20
    was called already, tool_choice names t21, and the last message is a tool result."""
    chat_tools = [
        {"type": "function", "function": {"name": f"t{n}", "parameters": {"type": "object"}}}
        for n in range(25)
    ]
    chat = {
        "model": "m",
        "tools": [*chat_tools[:3], {"type": "code_interpreter"}, *chat_tools[3:]],
        "tool_choice": {
            "type": "allowed_tools",
            "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": "t21"}}]},
        },
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Refund the last payment"}]},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "t20", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
    }
    responses = {
        "model": "m",
        "tools": [
            {"type": "web_search"},
            *({"type": "function", "name": f"t{n}", "parameters": {}} for n in range(25)),
        ],
        "tool_choice": {"type": "function", "name": "t21"},
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Refund the last payment"}]},
            {"type": "function_call", "call_id": "c1", "name": "t20", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ],
    }
    messages = {
        "model": "m",
        "max_tokens": 100,
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
            *({"name": f"t{n}", "input_schema": {"type": "object"}} for n in range(25)),
        ],
        "tool_choice": {"type": "tool", "name": "t21"},
        "messages": [
            {"role": "user", "content": "Refund the last payment"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "t20", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}]},
        ],
    }
    return {"acompletion": chat, "aresponses": responses, "anthropic_messages": messages}


def test_litellm_filter_keeps_the_ranked_used_chosen_and_provider_tools_of_every_shape():
    import copy

    from toolrank.integrations.litellm import SHAPES, ToolFilter, _record

    provider = {
        "acompletion": "code_interpreter",
        "aresponses": "web_search",
        "anthropic_messages": "web_search",
    }
    for call_type, data in _requests().items():
        ranker, seen = _Ranker(), []
        before = copy.deepcopy(data)
        out = asyncio.run(ToolFilter(ranker, on_filter=seen.append)(data, call_type))
        assert data == before  # never edited in place
        assert {k: v for k, v in out.items() if k != "tools"} == {
            k: v for k, v in data.items() if k != "tools"
        }

        def label(tool, shape=SHAPES[call_type]):
            record = _record(shape, tool)
            return record["name"] if record else tool.get("name") or tool["type"]

        kept = [label(t) for t in out["tools"]]
        assert kept == [label(t) for t in data["tools"] if label(t) in kept]  # the request's own order
        assert sorted(kept) == sorted(["t5", "t7", "t9", "t20", "t21", provider[call_type]])
        assert ranker.asked == [("Refund the last payment", [f"t{n}" for n in range(25)])]
        assert seen == [{"outcome": "filtered", "tools": 26, "kept": 6, "query": "Refund the last payment"}]


def test_litellm_filter_leaves_requests_alone_when_it_should_and_fails_open():
    import time

    from toolrank.client import ToolrankError as Down
    from toolrank.integrations.litellm import ToolFilter

    chat = _requests()["acompletion"]
    ranker = _Ranker()

    def run(data, call_type="acompletion", f=None):
        return asyncio.run((f or ToolFilter(ranker))(data, call_type))

    assert run(chat, "aembedding") is None  # not a request with tools
    assert run(chat, f=ToolFilter(ranker, min_tools=25)) is None  # few enough already
    deferred = {**chat, "tools": [*chat["tools"][:-1], {**chat["tools"][-1], "defer_loading": True}]}
    assert run(deferred) is None  # the provider's own tool search
    assert run({**chat, "messages": chat["messages"][:1]}) is None  # no user text
    assert ranker.asked == []
    seen = []
    assert run(chat, f=ToolFilter(_Ranker(fail=Down(0, "server unreachable")), on_filter=seen.append)) is None
    assert seen[0]["outcome"] == "skipped" and "server unreachable" in seen[0]["error"]

    async def timed(f):
        t0 = time.perf_counter()
        return await f(chat, "acompletion"), time.perf_counter() - t0

    out, took = asyncio.run(timed(ToolFilter(_Ranker(delay=0.5), timeout_s=0.05)))
    assert out is None and took < 0.4  # slow: sent unfiltered, without waiting for toolrank


def test_litellm_filter_ranks_through_a_served_toolrank(tmp_path):
    from toolrank.integrations.litellm import ToolFilter

    chat = _requests()["acompletion"]
    with _toolrank(tmp_path) as tr:
        out = asyncio.run(ToolFilter(tr)(chat, "acompletion"))
    kept = [t["function"]["name"] for t in out["tools"] if t["type"] == "function"]
    assert {"t20", "t21"} <= set(kept) and 3 <= len(kept) <= 12 and len(out["tools"]) == len(kept) + 1


def test_litellm_callback_comes_from_the_environment_and_defines_its_own_hook(monkeypatch):
    import sys
    import types

    from toolrank.integrations import litellm as tl

    class CustomLogger:  # the part of litellm's class the proxy relies on
        def __init__(self, *args, **kwargs):
            pass

        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            pass

    fake = {
        name: types.ModuleType(name)
        for name in ("litellm", "litellm.integrations", "litellm.integrations.custom_logger")
    }
    fake["litellm.integrations.custom_logger"].CustomLogger = CustomLogger
    for name, module in fake.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("TOOLRANK_URL", "http://toolrank.test:9000")
    monkeypatch.setenv("TOOLRANK_FILTER_MIN_TOOLS", "3")
    tl._filter_class.cache_clear()
    tl._tool_filter.cache_clear()
    try:
        hook = tl.tool_filter  # what `callbacks: toolrank.integrations.litellm.tool_filter` loads
        assert isinstance(hook, CustomLogger) and hook is tl.tool_filter
        assert "async_pre_call_hook" in vars(type(hook))  # LiteLLM skips inherited hooks
        assert hook.filter.toolrank.base_url == "http://toolrank.test:9000" and hook.filter.min_tools == 3
        small = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
        assert asyncio.run(hook.async_pre_call_hook(None, None, small, "acompletion")) is None
        with pytest.raises(AttributeError):
            tl.nothing_here  # noqa: B018
    finally:
        tl._filter_class.cache_clear()
        tl._tool_filter.cache_clear()
