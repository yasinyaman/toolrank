---
name: toolrank
description: Find and run tools from a large catalogue of MCP servers and HTTP APIs served by toolrank. Use it when a step needs an external tool or API you do not already have; search for the step in plain words, then call a tool the search returns.
---

# toolrank

toolrank serves a catalogue of tools (MCP servers and OpenAPI operations) behind two scripts. The
catalogue may hold thousands of tools: you never list them, you search for the ones a step needs.
Paths below are relative to this skill's folder.

## Find tools

    python3 scripts/search.py "refund the last payment of customer cus_123"

Describe the step in plain words, not a keyword. The answer is JSON: a `search_id` and the matching
tools, best first, each with `name`, `description` and `inputSchema`. A tool marked
`inputSchemaShrunk` has an abbreviated schema: search again with `--full` before calling it. If no
tool fits, search again in other words.

## Run a tool

    python3 scripts/call.py stripe/PostRefunds '{"payment_intent": "pi_123"}' --search-id <search_id>

The arguments are one JSON object that matches the tool's `inputSchema` (or `-` to read it from
stdin). Pass the `search_id` of the search that found the tool. The tool's output is printed. Exit
status 1 means the call did not succeed: the tool reported an error, or toolrank's write policy
refused it (read the output); 2 means toolrank rejected the request or could not be reached (read
stderr).

A tool without `"readOnlyHint": true` in its `annotations`, or with an HTTP `method` other than GET,
can change things: ask the user before calling it unless they asked for exactly that.

## Setup

`TOOLRANK_URL` (default `http://127.0.0.1:8765`) points at a running `toolrank serve`;
`TOOLRANK_API_KEY` is its bearer token when it has one. `TOOLRANK_SESSION` (optional) groups one
conversation's searches and calls in toolrank's usage log.
