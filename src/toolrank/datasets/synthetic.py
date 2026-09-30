"""A small deterministic synthetic tool set, for smoke tests and for trying the CLI offline.

Tools are ``verb_object`` functions across a dozen domains; queries paraphrase one or two tools
with synonyms, so lexical and semantic scorers separate a little (BM25 lands well under 100%).
Numbers from this set mean nothing outside the test-suite: real evaluation is ToolRet & co.
"""

from __future__ import annotations

import random
from pathlib import Path

from toolrank.datasets.jsonl import write_queries, write_tools
from toolrank.domain import Query, Tool

DOMAINS: dict[str, dict[str, list[str]]] = {
    "weather": {
        "objects": ["forecast", "current_conditions", "air_quality", "uv_index", "storm_alerts"],
        "params": ["city", "date", "units"],
        "asks": [
            "What's the {obj} for {city} {when}?",
            "Tell me the {obj} in {city} {when}",
            "I need {city}'s {obj} {when}",
        ],
    },
    "flights": {
        "objects": ["flight_status", "flight_prices", "seat_map", "baggage_rules", "airport_delays"],
        "params": ["flight_number", "origin", "destination", "date"],
        "asks": [
            "Check {obj} for my trip from {city} {when}",
            "Find the {obj} to {city} {when}",
            "Is there any {obj} info for {city}?",
        ],
    },
    "hotels": {
        "objects": ["hotel_availability", "room_prices", "guest_reviews", "cancellation_policy"],
        "params": ["city", "checkin", "checkout", "guests"],
        "asks": [
            "Look up {obj} in {city} {when}",
            "Any {obj} for a stay in {city}?",
            "Show {obj} near {city} {when}",
        ],
    },
    "payments": {
        "objects": ["invoice", "refund", "payment_status", "subscription", "payout"],
        "params": ["customer_id", "amount", "currency", "invoice_id"],
        "asks": [
            "Handle the {obj} for customer {n}",
            "I want to see the {obj} of order {n}",
            "Process a {obj} of {n} euros",
        ],
    },
    "email": {
        "objects": ["inbox_messages", "draft", "attachment", "contact", "mail_thread"],
        "params": ["query", "recipient", "subject", "max_results"],
        "asks": [
            "Search my {obj} about {topic}",
            "Open the {obj} regarding {topic}",
            "Find {obj} from last week on {topic}",
        ],
    },
    "calendar": {
        "objects": ["meeting", "availability", "reminder", "event_invite"],
        "params": ["date", "time", "attendees", "title"],
        "asks": [
            "Set up a {obj} {when} about {topic}",
            "Check {obj} for {when}",
            "Create a {obj} titled {topic} {when}",
        ],
    },
    "files": {
        "objects": ["document", "folder", "file_version", "shared_link", "file_metadata"],
        "params": ["path", "name", "owner"],
        "asks": [
            "Get the {obj} named {topic}",
            "Where is the {obj} for {topic}?",
            "List every {obj} under {topic}",
        ],
    },
    "git": {
        "objects": ["pull_request", "commit_history", "branch", "code_review", "ci_pipeline"],
        "params": ["repo", "branch", "number"],
        "asks": [
            "Show the {obj} on repo {topic}",
            "What is the state of the {obj} for {topic}?",
            "Open the {obj} number {n} in {topic}",
        ],
    },
    "database": {
        "objects": ["table_schema", "sql_query", "row_count", "index_stats", "backup"],
        "params": ["database", "table", "sql"],
        "asks": [
            "Run the {obj} on table {topic}",
            "Give me the {obj} of {topic}",
            "I need a {obj} for the {topic} database",
        ],
    },
    "maps": {
        "objects": ["route", "distance", "nearby_places", "geocode", "traffic"],
        "params": ["origin", "destination", "mode"],
        "asks": [
            "Find the {obj} from {city} to {city2}",
            "How is the {obj} around {city} {when}?",
            "Compute the {obj} between {city} and {city2}",
        ],
    },
    "translation": {
        "objects": ["translation", "language_detection", "transcript", "summary"],
        "params": ["text", "source_lang", "target_lang"],
        "asks": [
            "Produce a {obj} of this {topic} text",
            "I need the {obj} for a {topic} paragraph",
            "Get a {obj} of the {topic} notes",
        ],
    },
    "crm": {
        "objects": ["lead", "deal_stage", "account_notes", "ticket", "customer_profile"],
        "params": ["account_id", "owner", "status"],
        "asks": [
            "Update the {obj} for account {n}",
            "Pull the {obj} of client {topic}",
            "Create a {obj} about {topic}",
        ],
    },
}
VERBS = ["get", "list", "create", "update", "search", "delete"]
CITIES = ["Istanbul", "Ankara", "Berlin", "London", "Tokyo", "Boston", "Madrid", "Cairo"]
WHEN = ["tomorrow", "next Monday", "this weekend", "on the 14th", "tonight", "in March"]
TOPICS = [
    "quarterly report",
    "onboarding",
    "billing",
    "the roadmap",
    "security audit",
    "launch",
    "customer churn",
    "the migration",
]
SYNONYMS = {
    "get": ["fetch", "retrieve", "pull", "look up"],
    "list": ["enumerate", "show all", "list"],
    "create": ["make", "add", "set up"],
    "update": ["change", "modify", "edit"],
    "search": ["find", "look for", "locate"],
    "delete": ["remove", "drop", "cancel"],
}


def make_tools(n_tools: int, rng: random.Random) -> list[Tool]:
    tools: list[Tool] = []
    combos = [(d, v, o) for d, spec in DOMAINS.items() for o in spec["objects"] for v in VERBS]
    rng.shuffle(combos)
    for i, (d, v, o) in enumerate(combos[:n_tools]):
        spec = DOMAINS[d]
        params = {
            p: {
                "type": rng.choice(["string", "integer", "date"]),
                "description": f"The {p.replace('_', ' ')} to use.",
            }
            for p in rng.sample(spec["params"], k=min(len(spec["params"]), rng.randint(1, 3)))
        }
        doc = {
            "name": f"{v}_{o}",
            "description": f"{v.capitalize()} the {o.replace('_', ' ')} in the {d} service.",
            "parameters": params,
        }
        tools.append(Tool(id=f"syn_{i:04d}", doc=doc, documentation=str(doc), category=d))
    return tools


def make_queries(tools: list[Tool], n_queries: int, rng: random.Random) -> list[Query]:
    by_domain: dict[str, list[Tool]] = {}
    for t in tools:
        by_domain.setdefault(t.category, []).append(t)
    out: list[Query] = []
    for i in range(n_queries):
        t = rng.choice(tools)
        verb, obj = t.name.split("_", 1)
        spec = DOMAINS[t.category]
        ask = rng.choice(spec["asks"]).format(
            obj=obj.replace("_", " "),
            city=rng.choice(CITIES),
            city2=rng.choice(CITIES),
            when=rng.choice(WHEN),
            topic=rng.choice(TOPICS),
            n=rng.randint(100, 9999),
        )
        text = f"{rng.choice(SYNONYMS[verb])}: {ask}" if rng.random() < 0.5 else ask
        qrels = {t.id: 1}
        if rng.random() < 0.2 and len(by_domain[t.category]) > 1:  # multi-tool query
            t2 = rng.choice([x for x in by_domain[t.category] if x.id != t.id])
            text += f" and then {t2.name.split('_', 1)[0]} the {t2.name.split('_', 1)[1].replace('_', ' ')}"
            qrels[t2.id] = 1
        out.append(
            Query(
                id=f"synq_{i:04d}",
                text=text,
                qrels=qrels,
                instruction=f"Given a `{t.category}` task, retrieve tools that {t.description.lower()}",
                task=t.category,
            )
        )
    return out


def write_synthetic(
    out_dir: str | Path, n_tools: int = 300, n_queries: int = 200, seed: int = 7
) -> tuple[int, int]:
    rng = random.Random(seed)
    tools = make_tools(n_tools, rng)
    queries = make_queries(tools, n_queries, rng)
    out = Path(out_dir)
    return write_tools(out / "tools.jsonl", tools), write_queries(out / "queries.jsonl", queries)
