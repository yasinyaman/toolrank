import io
import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from toolrank.adapters.embeddings_api import OpenAIEmbeddings


def _encoder(cache_dir, calls):
    def fake_post(texts):  # one-hot by text length, so rows are distinguishable after normalising
        calls.append(list(texts))
        out = []
        for t in texts:
            v = np.zeros(4, dtype=np.float32)
            v[len(t)] = 1.0
            out.append(v)
        return out, len(texts)

    enc = OpenAIEmbeddings("m", "http://unused/v1", cache_dir=cache_dir)
    enc._post = fake_post
    return enc


def test_encode_fills_an_empty_cache_and_a_new_process_reuses_it(tmp_path):
    calls: list[list[str]] = []
    enc = _encoder(tmp_path, calls)
    first = enc.encode(["a", "bb", "a"])
    assert calls == [["a", "bb"]]  # duplicates sent once
    assert len(enc.cache) == 2  # regression: an empty cache is falsy, it must still be written
    again = _encoder(tmp_path, calls)  # same endpoint + model -> same keys
    assert np.allclose(again.encode(["bb", "a"]), first[[1, 0]])
    assert len(calls) == 1  # served from SQLite, no request


def test_empty_text_is_sent_as_a_space(tmp_path):
    calls: list[list[str]] = []
    m = _encoder(None, calls).encode(["", "a"])  # vLLM rejects "", ToolRet has one empty query
    assert calls == [[" ", "a"]] and m.shape == (2, 4)


def test_a_servers_short_timeout_applies_to_queries_only(monkeypatch):
    timeouts, sleeps = [], []

    def down(req, timeout):
        timeouts.append(timeout)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", down)
    monkeypatch.setattr("time.sleep", sleeps.append)
    enc = OpenAIEmbeddings("m", "http://unused/v1", query_timeout=10.0, query_max_retries=2)
    with pytest.raises(RuntimeError, match="unreachable"):
        enc.encode(["what time is it"], kind="query")
    assert timeouts == [10.0, 10.0] and sleeps == [1.5]  # one retry, no wait after the last attempt
    timeouts.clear()
    with pytest.raises(RuntimeError, match="unreachable"):
        enc.encode(["a long tool description"], kind="document")
    assert timeouts == [600.0, 600.0, 600.0]  # indexing a catalogue keeps the long leash


@pytest.mark.parametrize(
    ("url", "env", "sent"),
    [
        ("http://127.0.0.1:8091/v1", {"OPENAI_API_KEY": "sk-agent"}, None),  # a local vLLM
        ("https://embeddings.example.com/v1", {"OPENAI_API_KEY": "sk-agent"}, None),  # a third party
        ("http://api.openai.com/v1", {"OPENAI_API_KEY": "sk-agent"}, None),  # not in clear text either
        ("https://api.openai.com/v1", {"OPENAI_API_KEY": "sk-agent"}, "Bearer sk-agent"),
        (
            "http://127.0.0.1:8091/v1",
            {"OPENAI_API_KEY": "sk-agent", "TOOLRANK_EMB_API_KEY": "own"},
            "Bearer own",
        ),
    ],
)
def test_the_openai_key_goes_to_openai_only(monkeypatch, url, env, sent):
    for k in ("OPENAI_API_KEY", "TOOLRANK_EMB_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    got = []

    def fake_urlopen(req, timeout):
        got.append(req.get_header("Authorization"))
        return io.BytesIO(json.dumps({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    OpenAIEmbeddings("m", url).encode(["a"])
    assert got == [sent]


def test_the_key_does_not_follow_a_redirect(monkeypatch):
    monkeypatch.setenv("TOOLRANK_EMB_API_KEY", "own")
    sent = []

    def fake_urlopen(req, timeout):
        sent.append(req)
        return io.BytesIO(json.dumps({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    OpenAIEmbeddings("m", "https://embeddings.example.com/v1").encode(["a"])
    assert sent[0].get_header("Authorization") == "Bearer own"
    for code in (301, 302, 303):  # the ones urllib follows for a POST, to wherever they point
        moved = urllib.request.HTTPRedirectHandler().redirect_request(
            sent[0], None, code, "Moved", {}, "https://elsewhere.example.org/v1/embeddings"
        )
        assert moved is not None and moved.get_header("Authorization") is None


def test_a_refusal_without_a_key_names_the_variable_to_set(monkeypatch):
    monkeypatch.delenv("TOOLRANK_EMB_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-agent")  # not sent to a third party, so nothing was sent

    def refused(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", refused)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="set TOOLRANK_EMB_API_KEY"):
        OpenAIEmbeddings("m", "https://embeddings.example.com/v1").encode(["a"])


def test_the_fake_server_speaks_what_the_client_expects():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from fake_embeddings import DIM, serve

    server, url = serve()
    try:
        enc = OpenAIEmbeddings("fake", url, cache_dir=None, truncate_prompt_tokens=8192)
        m = enc.encode(["refund a payment", "refund a payment", "list invoices"])
        assert m.shape == (3, DIM) and np.allclose(m[0], m[1]) and abs(float(m[0] @ m[2])) < 0.2
    finally:
        server.shutdown()


def test_a_dropped_keep_alive_is_retried_like_a_network_error(monkeypatch):
    """urllib leaves RemoteDisconnected / ConnectionResetError from getresponse() unwrapped: one
    dropped connection ended a long encode instead of being tried again."""
    import http.client

    tries = []

    def flaky(req, timeout):
        tries.append(1)
        if len(tries) == 1:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return io.BytesIO(json.dumps({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    rows = OpenAIEmbeddings("m", "http://127.0.0.1:9/v1", cache_dir=None).encode(["a"])
    assert len(tries) == 2 and rows.shape == (1, 2)


def test_the_rest_client_turns_a_dropped_connection_into_its_own_error(monkeypatch):
    import http.client

    from toolrank.client import ToolrankClient, ToolrankError

    def dropped(self, req, timeout=None):
        raise http.client.RemoteDisconnected("closed")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", dropped)
    with pytest.raises(ToolrankError, match="connection lost"):
        ToolrankClient("http://127.0.0.1:9").search("x")
