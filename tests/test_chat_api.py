import io
import json

from toolrank.adapters.chat_api import OpenAIChat
from toolrank.ports import ChatModel


def test_complete_posts_both_messages_and_the_extra_body(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    chat = OpenAIChat("m", "http://x/v1/", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    assert isinstance(chat, ChatModel)
    assert chat.complete("sys", "user") == "hi"
    assert sent["url"] == "http://x/v1/chat/completions"
    assert sent["body"] == {
        "model": "m",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
        "temperature": 0.0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_the_openai_key_goes_to_openai_only(monkeypatch):
    monkeypatch.delenv("TOOLRANK_CHAT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-agent")
    assert OpenAIChat("m", "http://127.0.0.1:8093/v1").api_key is None
    assert OpenAIChat("m", "https://api.openai.com/v1").api_key == "sk-agent"
    monkeypatch.setenv("TOOLRANK_CHAT_API_KEY", "own")
    assert OpenAIChat("m", "http://127.0.0.1:8093/v1").api_key == "own"
