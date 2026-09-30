import json
import os
import stat
import threading
from hashlib import sha256

import pytest

from toolrank.domain import Tool
from toolrank.retriever import Hit, SearchResult
from toolrank.usage import UsageLog, mask_pii


def _result(tools=("gh/a", "gh/b"), query="open an issue", **kw):
    hits = [Hit(Tool(id=t, category="gh"), 0.9 - n / 10) for n, t in enumerate(tools)]
    return SearchResult(
        query=query,
        instruction="Find tools.",
        hits=hits,
        ranked=[(h.id, h.score) for h in hits] + [("gh/c", 0.1)],
        took_ms=12.345,
        rule="adaptive K (margin 0.2, max 10, min 1)",
        emb_key="abc",
        scorer="dense/emb/x",
        catalog="cafe",
        **kw,
    )


def _events(d):
    return [json.loads(line) for f in sorted(d.glob("usage-*.jsonl")) for line in f.read_text().splitlines()]


def test_search_and_call_events_keep_hashes_not_text(tmp_path):
    log = UsageLog(tmp_path / "usage")
    sid = log.search(_result(own_instruction=False), session="stdio", via="mcp", heads="f3c1")
    log.call(
        tool="gh/b", kind="mcp", session="stdio", via="mcp", outcome="ok", took_ms=40.2, arguments={"x": 1}
    )
    search, call = _events(tmp_path / "usage")
    assert (
        search["event"] == "search" and search["id"] == sid and search["v"] == 3 and search["client"] is None
    )
    assert search["query"] is None and search["query_hmac"] != sha256(b"open an issue").hexdigest()
    # the cache key is an unkeyed hash of the request: in the log it would confirm a guess at it
    assert search["emb_hmac"] == log.digest("abc") and "abc" not in json.dumps(search)
    assert (search["instruction"], search["instruction_hmac"]) == ("Find tools.", log.digest("Find tools."))
    assert search["results"][-1] == ["gh/c", 0.1] and search["shown"] == 2 and search["heads"] == "f3c1"
    assert (call["search_id"], call["rank"], call["link"], call["error"]) == (sid, 2, "session", None)
    assert call["args_hmac"] == log.digest(json.dumps({"x": 1}, sort_keys=True))
    key = tmp_path / "usage" / ".key"
    assert stat.S_IMODE(os.stat(key).st_mode) == 0o600 and len(key.read_bytes()) == 32
    assert UsageLog(tmp_path / "usage").digest("q") == log.digest("q")  # the key persists


def test_an_instruction_sent_with_the_request_is_request_text(tmp_path):
    log, text = UsageLog(tmp_path / "a"), UsageLog(tmp_path / "b", log_text=True)
    own = _result()  # a result that does not say whose instruction it is: treated as the request's
    assert own.own_instruction
    log.search(own, session="s", via="rest")
    text.search(own, session="s", via="rest")
    (hidden,), (shown,) = _events(tmp_path / "a"), _events(tmp_path / "b")
    assert hidden["instruction"] is None and hidden["instruction_hmac"] == log.digest("Find tools.")
    assert "Find tools." not in json.dumps(hidden) and shown["instruction"] == "Find tools."


def test_links_prefer_the_agents_search_id_then_session_then_client(tmp_path):
    log = UsageLog(tmp_path, log_text=True)
    desk, cursor = "-|claude-desktop|10.0.0.1", "-|cursor|10.0.0.2"  # key name | client app | host
    first = log.search(_result(("gh/a",)), session="s1", via="mcp", client=desk)
    second = log.search(_result(("gh/a", "gh/b")), session="s2", via="mcp", client=cursor)
    third = log.search(_result(("gh/c",)), session=None, via="mcp", client=desk)  # 2026-07-28: no session
    fourth = log.search(_result(("gh/d",)), session=None, via="rest", client=desk)
    assert log.link("gh/a", "s2", search_id=first) == (first, 1, "search_id")
    assert log.link("gh/a", "s1", via="mcp", client=cursor) == (
        first,
        1,
        "session",
    )  # beats newer client match
    assert log.link("gh/b", None, via="mcp", client=cursor) == (second, 2, "client")
    assert log.link("gh/b", None, via="mcp", client=desk) == (None, None, "none")  # another client's search
    assert log.link("gh/b", "s9", via="mcp", client=cursor) == (None, None, "none")  # s9 never searched
    assert log.link("gh/c", None, via="mcp", client=desk) == (third, 1, "client")
    assert log.link("gh/d", None, via="mcp", client=desk) == (None, None, "none")  # REST is another interface
    assert log.link("gh/d", None, search_id=fourth, via="mcp") == (fourth, 1, "search_id")
    assert log.link("gh/zzz", "s1") == (None, None, "none")
    for odd in (["s-1"], {"id": first}, 7):  # an agent's search_id is not always a string
        assert log.link("gh/a", "s1", search_id=odd) == (first, 1, "session")
    log.call(
        tool="gh/zzz", kind=None, session="s1", via="rest", outcome="unknown_tool", took_ms=0, error="boom",
        client=desk, tenant="team-a",
    )  # fmt: skip
    events = _events(tmp_path)
    assert events[-1]["error"] == "boom" and events[0]["query"] == "open an issue"
    assert (events[-1]["tenant"], events[-1]["client"]) == ("team-a", log.digest(desk)[:16])
    assert desk not in json.dumps(events)  # the remote address is never written in clear
    with pytest.raises(ValueError):
        log.call(tool="x", kind=None, session="s", via="mcp", outcome="weird", took_ms=0)
    log.call(tool="n" * 5000, kind=None, session="s", via="mcp", outcome="unknown_tool", took_ms=0)
    log.call(tool="gh/" + "n" * 300, kind="mcp", session="s", via="mcp", outcome="ok", took_ms=0)
    typed, real = _events(tmp_path)[-2:]  # only a name that is no catalogue id is cut
    assert (len(typed["tool"]), len(real["tool"])) == (200, 303)


def test_disabled_log_writes_nothing_and_threads_do_not_interleave(tmp_path):
    off = UsageLog(None)
    off.search(_result(), session="s", via="mcp")
    assert off.link("gh/a", "s")[2] == "session" and not list(tmp_path.iterdir())

    log = UsageLog(tmp_path)

    def worker(n):
        for i in range(50):
            log.call(tool=f"t{n}", kind="mcp", session=str(n), via="mcp", outcome="ok", took_ms=i)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(_events(tmp_path)) == 400


def test_mask_pii_replaces_addresses_and_numbers_but_not_dates(tmp_path):
    assert mask_pii("mail ali.veli+x@example.co.uk, call +90 532 123 45 67") == "mail <email>, call <phone>"
    assert (
        mask_pii("card 4111 1111 1111 1111, IBAN TR33 0006 1005 1978 6457 8413 26")
        == "card <card>, IBAN <iban>"
    )
    assert (
        mask_pii("on 2026-09-30 at 12:30 order #4521 of 1,862 tools")
        == "on 2026-09-30 at 12:30 order #4521 of 1,862 tools"
    )
    # only with --log-text is there text to mask; then the request and the error are masked
    plain, masked = (
        UsageLog(tmp_path / "a", log_text=True),
        UsageLog(tmp_path / "b", log_text=True, mask_pii=True),
    )
    for log in (plain, masked):
        log.search(
            _result(query="refund card 4111 1111 1111 1111", own_instruction=False), session="s", via="rest"
        )
        log.call(
            tool="gh/a",
            kind="mcp",
            session="s",
            via="rest",
            outcome="tool_error",
            took_ms=1,
            error="write to a@b.co",
        )
    (ps, pc), (ms, mc) = _events(tmp_path / "a"), _events(tmp_path / "b")
    assert "4111" in ps["query"] and pc["error"] == "write to a@b.co"
    assert (ms["query"], mc["error"]) == ("refund card <card>", "write to <email>")
    assert ms["query_hmac"] == masked.digest(
        "refund card 4111 1111 1111 1111"
    )  # the digest is of the request as it came
    hidden = UsageLog(tmp_path / "c", mask_pii=True)
    hidden.search(_result(query="x@y.zz"), session="s", via="rest")
    assert _events(tmp_path / "c")[0]["query"] is None
