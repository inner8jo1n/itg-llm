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


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
def test_temporary_api_error_recovers(monkeypatch, status, caplog):
    import httpx2 as httpx
    from openai import APIStatusError

    client, fake = make_client(monkeypatch, "ok")
    success = fake.create
    attempts = []
    sleeps = []

    def create(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            response = httpx.Response(
                status, request=httpx.Request("POST", "https://test.invalid")
            )
            raise APIStatusError("temporary", response=response, body=None)
        return success(**kwargs)

    monkeypatch.setattr(fake, "create", create)
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    assert client.complete("sys", "usr") == "ok"
    assert len(attempts) == 2
    assert sleeps == [0.5]
    assert "retry 1/2" in caplog.text


@pytest.mark.parametrize("kind", ["connection", "timeout", "server"])
def test_exhausted_api_retries_are_bounded(monkeypatch, kind, caplog):
    import httpx2 as httpx
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    client, fake = make_client(monkeypatch, "ok")
    attempts = []
    sleeps = []
    request = httpx.Request("POST", "https://test.invalid")
    errors = {
        "connection": APIConnectionError(request=request),
        "timeout": APITimeoutError(request=request),
        "server": APIStatusError(
            "private response",
            response=httpx.Response(503, request=request),
            body=None,
        ),
    }

    def create(**kwargs):
        attempts.append(kwargs)
        raise errors[kind]

    monkeypatch.setattr(fake, "create", create)
    monkeypatch.setattr(llm_client.time, "sleep", sleeps.append)
    with pytest.raises(llm_client.LLMAPIError, match="after 3 attempt"):
        client.complete("sys", "usr")
    assert len(attempts) == 3
    assert sleeps == [0.5, 1.0]
    assert "private response" not in caplog.text


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_api_error_does_not_retry(monkeypatch, status):
    import httpx2 as httpx
    from openai import APIStatusError

    client, fake = make_client(monkeypatch, "ok")
    attempts = []

    def create(**kwargs):
        attempts.append(kwargs)
        raise APIStatusError(
            "private response",
            response=httpx.Response(
                status, request=httpx.Request("POST", "https://test.invalid")
            ),
            body=None,
        )

    monkeypatch.setattr(fake, "create", create)
    with pytest.raises(llm_client.LLMAPIError, match=f"HTTP {status}"):
        client.complete("sys", "usr")
    assert len(attempts) == 1


def test_empty_choices_returns_empty_reply_for_validation(monkeypatch):
    client, fake = make_client(monkeypatch, "ok")
    monkeypatch.setattr(
        fake, "create", lambda **kwargs: SimpleNamespace(choices=[])
    )
    assert client.complete("sys", "usr") == ""


def test_transport_and_generation_limits():
    client = LLMClient(api_key="test-key")
    assert client._client.max_retries == 0
    assert client._client.timeout == 30.0
