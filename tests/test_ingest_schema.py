"""Input schemas an agent API would refuse (backlog D2.7): what ``problems`` finds with and without
``jsonschema``, what ingest records, and what the platform records carry instead."""

from __future__ import annotations

import builtins

import pytest

from toolrank.ingest.mcp import tool_from_mcp
from toolrank.ingest.schema import _validator, problems
from toolrank.ingest.sync import Listing, checked, sync

GOOD = {
    "type": "object",
    "properties": {  # property names that are keywords elsewhere are just names here
        "items": {"type": "array", "items": {"type": "string"}},
        "required": {"type": "boolean", "default": {"items": [1]}},  # a default is a value, not a schema
        "ref": {"$ref": "#/$defs/id"},
    },
    "required": ["items"],
    "$defs": {"id": {"type": "string"}},
    "additionalProperties": False,
}
DRAFT_04 = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {"n": {"type": "number", "minimum": 0, "exclusiveMinimum": True}},
}


@pytest.fixture(params=["jsonschema", "by hand"])
def checker(request, monkeypatch):
    if request.param == "by hand":
        real = builtins.__import__

        def no_jsonschema(name, *args, **kw):
            if name.startswith("jsonschema"):
                raise ImportError(name)
            return real(name, *args, **kw)

        monkeypatch.setattr(builtins, "__import__", no_jsonschema)
        _validator.cache_clear()
    yield request.param
    _validator.cache_clear()


def test_problems_find_what_an_api_refuses(checker):
    assert problems(GOOD) == [] and problems(None) == [] and problems({}) == []
    (draft,) = problems(DRAFT_04)
    assert draft.startswith("not JSON Schema 2020-12 at #/properties/n") and "exclusiveMinimum" in draft
    tuple_items = {"type": "object", "properties": {"a": {"type": "array", "items": [{"type": "string"}]}}}
    assert problems(tuple_items)[0].startswith("not JSON Schema 2020-12 at #/properties/a")
    assert problems({"type": "array"}) == ["the root is 'array', not an object"]
    assert problems("x") == ["the schema is a str, not an object"]
    refs = problems(
        {"type": "object", "properties": {"a": {"$ref": "#/$defs/gone"}, "b": {"$ref": "other.json#/x"}}}
    )
    assert refs == ["$ref '#/$defs/gone' points nowhere", "$ref 'other.json#/x' points outside the schema"]


def test_ingest_keeps_the_tool_records_why_and_forgets_it_once_fixed():
    bad = tool_from_mcp("s", {"name": "n", "description": "d", "inputSchema": DRAFT_04})
    good = tool_from_mcp("s", {"name": "g", "description": "d", "inputSchema": GOOD})
    tools, _, (diff,) = sync([], {}, [Listing("s", "mcp", [bad, good])], now="t")
    assert [t.id for t in tools] == ["s/n", "s/g"] and "schema_problems" not in tools[1].doc
    assert (
        list(diff.schema_problems) == ["s/n"]
        and tools[0].doc["schema_problems"] == diff.schema_problems["s/n"]
    )
    assert diff.line() == "s  +2 ~0 -0 =0  (2 tools; 1 input schema an agent API would refuse)"
    fixed = tool_from_mcp("s", {"name": "n", "description": "d", "inputSchema": GOOD})
    tools, _, (diff,) = sync(tools, {}, [Listing("s", "mcp", [fixed, good])], now="t")
    assert "schema_problems" not in tools[0].doc and diff.changed == 1 and diff.schema_problems == {}
    assert checked(good) is good  # nothing to record: the same tool
