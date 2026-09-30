import json

import pytest

from toolrank.formats import tool_format
from toolrank.ingest.openapi import OpenAPISource, load_spec

SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Pet Store API", "version": "1"},
    "servers": [{"url": "https://{host}/v1", "variables": {"host": {"default": "pets.example.com"}}}],
    "paths": {
        "/pets/{petId}": {
            "parameters": [{"$ref": "#/components/parameters/PetId"}],
            "get": {
                "operationId": "getPet",
                "summary": "Get a pet",
                "description": "<p>Returns one pet &amp; its owner.</p>",
                "parameters": [
                    {"name": "fields", "in": "query", "schema": {"type": "string"}, "x-internal": True},
                    {"name": "Accept", "in": "header", "schema": {"type": "string"}},
                    {"name": "X-Trace", "in": "header", "schema": {"type": "string"}},
                ],
            },
            "patch": {
                "summary": "Update a pet",
                "parameters": [{"name": "name", "in": "query", "schema": {"type": "string"}}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "text/plain": {"schema": {"type": "string"}},
                        "application/json": {"schema": {"$ref": "#/components/schemas/PetUpdate"}},
                    },
                },
            },
        },
        "/pets": {"$ref": "#/components/pathItems/Pets"},
        "/stores": {
            "post": {
                "operationId": "getPet",
                "requestBody": {
                    "content": {
                        "application/x-www-form-urlencoded": {
                            "schema": {
                                "oneOf": [
                                    {"properties": {"city": {"type": "string"}}},
                                    {"properties": {"zip": {"type": "string"}}},
                                ]
                            }
                        }
                    }
                },
            },
            "get": {
                "operationId": "listStores",
                "requestBody": {
                    "content": {
                        "application/x-www-form-urlencoded": {"schema": {"type": "object", "properties": {}}}
                    }
                },
            },
        },
    },
    "components": {
        "parameters": {
            "PetId": {"name": "petId", "in": "path", "required": True, "schema": {"type": "integer"}}
        },
        "schemas": {
            "PetUpdate": {
                "allOf": [
                    {"$ref": "#/components/schemas/Named"},
                    {"required": ["tag"], "properties": {"tag": {"type": "string", "example": "cat"}}},
                ]
            },
            "Named": {
                "required": ["name"],
                "properties": {"name": {"type": "string"}, "parent": {"$ref": "#/components/schemas/Node"}},
            },
            "Node": {"type": "object", "properties": {"child": {"$ref": "#/components/schemas/Node"}}},
        },
        "pathItems": {
            "Pets": {
                "get": {
                    "operationId": "listPets",
                    "parameters": [{"name": "owner", "in": "query", "schema": {"$ref": "other.yaml#/Owner"}}],
                }
            }
        },
    },
}


def _tools(spec=SPEC, **kw):
    src = OpenAPISource(spec, **kw)
    return src, {t.doc["name"]: t for t in src.list_tools()}


def test_operation_to_tool_name_text_and_routing():
    src, tools = _tools()
    assert src.name == "pet-store-api" and src.base_url == "https://pets.example.com/v1"
    get = tools["getPet"]
    assert (get.id, get.category) == ("pet-store-api/getPet", "pet-store-api")
    text = json.loads(get.documentation)
    assert list(text) == ["server", "name", "description", "inputSchema"]
    assert text["description"] == "Get a pet\n\nReturns one pet & its owner."
    schema = text["inputSchema"]
    assert list(schema["properties"]) == [
        "petId",
        "fields",
        "X-Trace",
    ]  # path-item param first, Accept skipped
    assert schema["required"] == ["petId"] and "x-internal" not in schema["properties"]["fields"]
    assert get.doc["http"] == {
        "method": "GET",
        "path": "/pets/{petId}",
        "base_url": "https://pets.example.com/v1",
        "args": {
            "petId": {"in": "path", "name": "petId"},
            "fields": {"in": "query", "name": "fields", "style": "form", "explode": True},
            "X-Trace": {"in": "header", "name": "X-Trace"},
        },
    }
    # the schema format reads inputSchema, not the routing
    assert tool_format("schema")(get).splitlines()[-3:] == [
        "  petId (integer)",
        "  fields (string)",
        "  X-Trace (string)",
    ]


def test_body_is_flattened_json_first_with_allof_refs_cycles_and_clashes():
    _, tools = _tools()
    patch = tools["patch_pets_petId"]  # no operationId
    schema = patch.doc["inputSchema"]
    assert list(schema["properties"]) == ["petId", "name__query", "name", "parent", "tag"]
    assert schema["required"] == ["petId", "name", "tag"]
    assert patch.doc["http"]["body_media_type"] == "application/json"
    assert patch.doc["http"]["args"]["name__query"] == {
        "in": "query",
        "name": "name",
        "style": "form",
        "explode": True,
    }
    assert "example" not in schema["properties"]["tag"]
    parent = schema["properties"]["parent"]  # Node -> child -> Node is cut
    assert parent["properties"]["child"] == {"type": "object"}


def test_form_body_oneof_union_duplicate_names_empty_bodies_path_item_and_external_refs():
    src, tools = _tools()
    post = tools["getPet_2"]  # a second "getPet" operationId
    assert list(post.doc["inputSchema"]["properties"]) == ["city", "zip"]
    assert post.doc["http"]["body_media_type"] == "application/x-www-form-urlencoded"
    assert tools["listStores"].doc["inputSchema"] == {"type": "object", "properties": {}}
    assert "body_media_type" not in tools["listStores"].doc["http"]
    owner = tools["listPets"].doc["inputSchema"]["properties"]["owner"]
    assert owner == {"description": "external schema other.yaml#/Owner"}
    assert src.stats["external_refs"] == 1 and src.stats["operations"] == 5


def test_paths_without_a_leading_slash_are_skipped():
    spec = json.loads(json.dumps(SPEC))
    get = {"get": {"operationId": "steal"}}
    spec["paths"].update({"@evil.example.com/x": get, ".evil.example.com/x": get, "x-internal": get})
    src, tools = _tools(spec)
    assert "steal" not in tools and len(tools) == 5
    assert src.stats["bad_paths"] == 2 and src.stats["operations"] == 5  # the extension is not counted


def test_long_operations_are_capped_but_keep_their_names():
    spec = json.loads(json.dumps(SPEC))
    spec["components"]["schemas"]["Named"]["properties"].update(
        {f"field{i}": {"type": "string", "description": "a long field description " * 5} for i in range(60)}
    )
    src, tools = _tools(spec, name="pets", max_chars=2000)
    text = json.loads(tools["patch_pets_petId"].documentation)
    assert len(tools["patch_pets_petId"].documentation) <= 2000 and text["name"] == "patch_pets_petId"
    assert src.stats["capped"] == 1 and len(tools["patch_pets_petId"].doc["inputSchema"]["properties"]) == 65


def test_load_spec_json_yaml_and_refusals(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps(SPEC))
    assert load_spec(str(tmp_path / "s.json"))["info"]["title"] == "Pet Store API"
    (tmp_path / "s2.json").write_text(json.dumps({"swagger": "2.0", "paths": {}}))
    with pytest.raises(ValueError, match="convert it to OpenAPI 3"):
        load_spec(str(tmp_path / "s2.json"))
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(ValueError, match="OpenAPI 3"):
        load_spec(str(tmp_path / "x.json"))
    pytest.importorskip("yaml")
    (tmp_path / "s.yaml").write_text(
        "openapi: 3.1.0\ninfo: {title: Dated, version: '1'}\npaths:\n  /v:\n    get:\n      operationId: v\n"
        "      parameters:\n        - {name: api-version, in: header, schema: {enum: [2022-11-28]}}\n"
    )
    spec = load_spec(str(tmp_path / "s.yaml"))
    tool = OpenAPISource(spec).list_tools()[0]
    assert tool.doc["inputSchema"]["properties"]["api-version"]["enum"] == ["2022-11-28"]  # not a date
