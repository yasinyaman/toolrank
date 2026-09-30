from toolrank.domain import Query, Tool
from toolrank.formats import QUERY_FORMATS, TOOL_FORMATS, _params, parse_doc, query_format, tool_format

DOC = {
    "name": "bacterial_growth",
    "description": "Calculates the bacterial population after a given time.",
    "parameters": {
        "initial_population": {
            "description": "The initial bacterial population.",
            "type": "int",
            "default": 20,
        },
        "time": {"description": "The time elapsed.", "type": "float"},
    },
}


def test_parse_doc_json_and_python_repr():
    assert parse_doc('{"name": "x", "description": "y"}') == {"name": "x", "description": "y"}
    assert (
        parse_doc("{'name': 'x', 'parameters': {'a': {'type': 'str'}}}")["parameters"]["a"]["type"] == "str"
    )
    assert parse_doc("plain text tool description") == {}
    assert parse_doc("") == {}


def test_tool_formats_from_structured_doc():
    t = Tool(id="apigen_tool_272", doc=DOC, documentation="")
    assert (
        tool_format("name_desc")(t)
        == "bacterial_growth: Calculates the bacterial population after a given time."
    )
    schema = tool_format("schema")(t)
    assert schema.splitlines()[0].startswith("bacterial_growth:")
    assert "initial_population (int) - The initial bacterial population." in schema
    call = tool_format("example_call")(t)
    assert call.startswith("bacterial_growth(initial_population=<int>, time=<float>)")
    assert call.endswith("Calculates the bacterial population after a given time.")


def test_tool_formats_fall_back_to_documentation_string():
    t = Tool(
        id="t1",
        doc={},
        documentation="{'name': 'availability', 'description': 'Checks a URL.', 'parameters': {}}",
    )
    assert tool_format("documentation")(t) == t.documentation
    assert tool_format("name_desc")(t) == "availability: Checks a URL."
    assert tool_format("example_call")(t) == "availability()\nChecks a URL."


def test_json_schema_properties_shape():
    t = Tool(
        id="mcp_1",
        doc={
            "name": "search",
            "description": "Search docs",
            "inputSchema": {
                "type": "object",
                "properties": {"q": {"type": "string", "description": "query"}},
            },
        },
    )
    assert "q (string) - query" in tool_format("schema")(t)
    assert tool_format("example_call")(t).startswith("search(q=<string>)")


def test_params_cover_the_toolret_shapes():
    # ToolBench / T-Eval: split required / optional lists
    split = {
        "required_parameters": [{"name": "q", "type": "STRING", "description": "query", "default": "x"}],
        "optional_parameters": [{"name": "limit", "type": "NUMBER", "description": ""}],
    }
    assert _params(split) == [("q", "STRING", "query"), ("limit", "NUMBER", "")]
    # UltraTool: a JSON schema under doc_arguments
    ultra = {
        "doc_arguments": {"type": "object", "properties": {"path": {"type": "string", "description": "p"}}}
    }
    assert _params(ultra) == [("path", "string", "p")]
    # AppBench: "name (type)" keys in two dicts
    app = {
        "additional_required_arguments": {"where_to (str)": "location"},
        "optional_arguments": {"rating (float)": "r"},
    }
    assert _params(app) == [("where_to", "str", "location"), ("rating", "float", "r")]
    # Gorilla: api_arguments as names, as a dict, or "N/A"; GTA: inputs with null descriptions
    assert _params({"api_arguments": ["model_id"]}) == [("model_id", "", "")]
    assert _params({"api_arguments": {"pretrained": "True"}}) == [("pretrained", "", "True")]
    assert _params({"api_arguments": "N/A"}) == []
    assert _params({"inputs": [{"name": "expression", "type": "text", "description": None}]}) == [
        ("expression", "text", "")
    ]
    # an empty schema has no parameters (not a parameter called "type")
    assert _params({"parameters": {"type": "dict", "properties": {}, "required": []}}) == []


def test_example_call_prefers_a_shipped_call():
    doc = {
        "name": "model_id",
        "api_call": "hub.load('https://tfhub.dev/x/1')",
        "description": "Detect objects.",
    }
    assert (
        tool_format("example_call")(Tool(id="g", doc=doc))
        == "hub.load('https://tfhub.dev/x/1')\nDetect objects."
    )


def test_query_formats():
    q = Query(
        id="1", text="What is the weather?", qrels={}, instruction="Given a weather task, retrieve tools."
    )
    assert query_format("plain")(q) == "What is the weather?"
    assert query_format("concat")(q) == "Given a weather task, retrieve tools. What is the weather?"
    assert (
        query_format("instruct_query")(q)
        == "Instruct: Given a weather task, retrieve tools.\nQuery: What is the weather?"
    )
    assert query_format("clm")(q) == "What is the weather?\n\nGiven a weather task, retrieve tools."
    bare = Query(id="2", text="hi", qrels={})
    assert all(f(bare) == "hi" for f in QUERY_FORMATS.values())
    assert set(TOOL_FORMATS) == {"documentation", "name_desc", "schema", "example_call"}


def test_toolret_labels_and_task_map():
    from toolrank.datasets.toolret import TASK_TO_CATEGORY, TOOLRET_CATEGORIES, TOOLRET_TASKS, _labels

    raw = '[{"id": "apigen_tool_272", "doc": {"name": "bacterial_growth"}, "relevance": 1}, {"id": "x", "relevance": 2}]'
    assert _labels(raw) == {"apigen_tool_272": 1, "x": 2}
    assert _labels(None) == {} and _labels([{"id": "a"}]) == {"a": 1}
    assert len(TOOLRET_TASKS) == 35 and set(TASK_TO_CATEGORY.values()) == set(TOOLRET_CATEGORIES)
