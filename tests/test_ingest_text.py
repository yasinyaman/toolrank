import json

from toolrank.ingest.text import fit_text, tool_text


def test_tool_text_is_the_faz0_ablation_shape():
    # data/mcp_zero_server (docs/reports/faz0-week4.md): {"server": category, **doc}, doc = name,
    # description and, when the tool has parameters, inputSchema; ensure_ascii=False
    schema = {"type": "object", "properties": {"q": {"type": "string", "description": "Suchtext"}}}
    doc = {"name": "search", "description": "Search the wiki", "inputSchema": schema}
    assert tool_text("Wiki", "search", "Search the wiki", schema) == json.dumps(
        {"server": "Wiki", **doc}, ensure_ascii=False
    )
    with_meta = {"$schema": "http://json-schema.org/draft-07/schema#", **schema}
    assert tool_text("Wiki", "search", "Search the wiki", with_meta) == tool_text(
        "Wiki", "search", "Search the wiki", schema
    )
    no_params = {"name": "ping", "description": ""}
    assert tool_text("Wiki", "ping", "") == json.dumps({"server": "Wiki", **no_params}, ensure_ascii=False)


def _deep_schema(n_props=40, depth=5):
    def nested(d):
        if d == 0:
            return {"type": "string", "description": "leaf " * 20}
        return {"type": "object", "description": "level " * 10, "properties": {"child": nested(d - 1)}}

    return {
        "type": "object",
        "properties": {f"p{i}": nested(depth) for i in range(n_props)},
        "required": ["p0", "p39"],
    }


def test_fit_text_shrinks_the_schema_before_the_description_and_keeps_server_and_name():
    schema = _deep_schema()
    text, capped = fit_text("Stripe", "PostCharges", "Create a charge.", schema, max_chars=3000)
    doc = json.loads(text)
    assert capped and len(text) <= 3000
    assert (doc["server"], doc["name"], doc["description"]) == ("Stripe", "PostCharges", "Create a charge.")
    assert "p0" in doc["inputSchema"]["properties"]
    small, capped_small = fit_text("Stripe", "PostCharges", "Create a charge.", {"type": "object"})
    assert not capped_small and json.loads(small)["inputSchema"] == {"type": "object"}


def test_fit_text_keeps_the_first_properties_then_cuts_the_description():
    schema = _deep_schema(n_props=200, depth=1)
    text, capped = fit_text("S", "op", "d", schema, max_chars=1200)
    props = json.loads(text)["inputSchema"]
    assert capped and len(text) <= 1200 and list(props["properties"]) == [f"p{i}" for i in range(8)]
    assert props["…"] == "192 more properties" and props["required"] == ["p0"]
    text, _ = fit_text("S", "op", "word " * 500, None, max_chars=300)
    doc = json.loads(text)
    assert len(text) <= 300 and doc["name"] == "op" and doc["description"].endswith("…")


def _pydantic_schema(n_models=20):
    """What FastMCP and pydantic write: the models in $defs, the properties as $refs to them."""
    model = {
        "type": "object",
        "description": "A model with a long docstring. " * 10,
        "properties": {f"f{i}": {"type": "string", "description": "field " * 12} for i in range(6)},
    }
    return {
        "type": "object",
        "properties": {f"m{i}": {"$ref": f"#/$defs/Model{i}"} for i in range(n_models)},
        "$defs": {f"Model{i}": model for i in range(n_models)},
        "additionalProperties": {"type": "object", "properties": {"x": {"description": "extra " * 50}}},
    }


def test_fit_text_shrinks_defs_and_keeps_the_description():
    text, capped = fit_text("Docs", "create", "Create a document in a workspace.", _pydantic_schema())
    doc = json.loads(text)
    assert capped and len(text) <= 6000 and doc["description"] == "Create a document in a workspace."
    assert (doc["server"], doc["name"]) == ("Docs", "create") and "m0" in doc["inputSchema"]["properties"]
    # a schema no step brings under the budget goes before the description is cut
    text, _ = fit_text("Docs", "create", "Create a document.", _pydantic_schema(), max_chars=120)
    doc = json.loads(text)
    assert "inputSchema" not in doc and doc["description"] == "Create a document."
