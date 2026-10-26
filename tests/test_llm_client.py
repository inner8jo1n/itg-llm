from types import SimpleNamespace

import pytest
from openai import omit

from itg_llm import llm_client
from itg_llm.llm_client import DEFAULT_MODEL, LLMClient


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    # Tests must not depend on a real .env or real environment
    monkeypatch.setattr(llm_client, "load_dotenv", lambda: None)
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)


def make_client(monkeypatch, content, **kwargs):
    client = LLMClient(api_key="test-key", **kwargs)
    fake = FakeCompletions(content)
    api = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    monkeypatch.setattr(client, "_client", api)
    return client, fake


def test_missing_key_raises():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        LLMClient()


def test_model_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_MODEL", "some-model")
    assert LLMClient().model == "some-model"


def test_default_model():
    assert LLMClient(api_key="test-key").model == DEFAULT_MODEL


def test_complete_sends_messages_and_returns_text(monkeypatch):
    client, fake = make_client(monkeypatch, "hello", model="m")

    assert client.complete("sys", "usr") == "hello"
    call = fake.calls[0]
    assert call["model"] == "m"
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_complete_sends_no_response_format_by_default(monkeypatch):
    client, fake = make_client(monkeypatch, "hello")
    client.complete("sys", "usr")
    assert fake.calls[0]["response_format"] is omit


def test_complete_json_mode_asks_for_json_object(monkeypatch):
    client, fake = make_client(monkeypatch, "{}")
    client.complete("sys", "usr", json_mode=True)
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_complete_turns_none_into_empty_string(monkeypatch):
    client, _ = make_client(monkeypatch, None)
    assert client.complete("sys", "usr") == ""
