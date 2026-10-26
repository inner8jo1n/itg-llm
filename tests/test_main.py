import json

import pytest

from itg_llm import main as main_module
from itg_llm.main import (
    format_record,
    load_inputs,
    process_text,
    run,
)
from itg_llm.prompts import DEFAULT_PROMPT, PROMPTS, PromptVariant
from itg_llm.schemas import AnalysisResult, Category, LLMOutputError

GOOD_REPLY = {
    "summary": " A short summary. ",
    "category": "question",
    "sentiment": "neutral",
    "key_points": ["First", "Second", "Third"],
    "final_answer": "A helpful answer.",
}
GOOD_JSON = json.dumps(GOOD_REPLY)
GOOD_RESULT = AnalysisResult.model_validate(GOOD_REPLY).model_dump(mode="json")


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, user, json_mode=False):
        self.calls.append((system, user))
        self.json_mode = json_mode
        return self.reply


OTHER_PROMPT = PromptVariant(
    name="other", description="", system="other system", template="{text}"
)


def test_process_text_returns_validated_result():
    client = FakeClient(json.dumps(GOOD_REPLY))

    result = process_text("Some text", client)

    assert isinstance(result, AnalysisResult)
    assert result.summary == "A short summary."
    assert result.category is Category.QUESTION
    assert result.final_answer == "A helpful answer."
    system, user = client.calls[0]
    assert system == PROMPTS[DEFAULT_PROMPT].system
    assert "Some text" in user
    assert client.json_mode is True


def test_process_text_uses_given_prompt():
    client = FakeClient(GOOD_JSON)
    process_text("Some text", client, OTHER_PROMPT)

    assert client.calls[0] == ("other system", "Some text")


def test_process_text_rejects_empty_text():
    client = FakeClient(json.dumps(GOOD_REPLY))
    with pytest.raises(ValueError, match="empty"):
        process_text("   ", client)
    assert client.calls == []


def test_process_text_raises_on_broken_reply():
    with pytest.raises(LLMOutputError, match="invalid JSON"):
        process_text("Some text", FakeClient('{"summary": "cut'))


def test_load_inputs_prefers_text(tmp_path):
    assert load_inputs("hi", None, tmp_path) == [("cli", "hi")]


def test_load_inputs_reads_file(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("from file", encoding="utf-8")
    assert load_inputs(None, path, tmp_path) == [("note", "from file")]


def test_load_inputs_reads_sorted_examples(tmp_path):
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "skip.md").write_text("no", encoding="utf-8")
    assert load_inputs(None, None, tmp_path) == [("a", "A"), ("b", "B")]


def test_load_inputs_without_examples_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_inputs(None, None, tmp_path)


def test_run_records_failures_and_keeps_going():
    records = run([("ok", "text"), ("bad", " ")], FakeClient(GOOD_JSON))

    assert records[0]["result"]["summary"] == "A short summary."
    assert "error" in records[1]
    assert {record["prompt"] for record in records} == {DEFAULT_PROMPT}
    assert records[0]["issues"] == []
    assert records[0]["priority"] == "medium"
    assert "issues" not in records[1]
    assert "priority" not in records[1]


def test_run_saves_result_as_plain_json():
    [record] = run([("ok", "text")], FakeClient(GOOD_JSON))
    assert record["result"] == GOOD_RESULT
    assert record["result"]["category"] == "question"


def test_run_records_schema_errors_and_keeps_going(caplog):
    broken = json.dumps({**GOOD_REPLY, "sentiment": "angry"})
    client = FakeClient(broken)

    records = run([("bad", "text")], client)

    assert "field 'sentiment' must be one of" in records[0]["error"]
    assert "result" not in records[0]
    assert "Failed to process bad" in caplog.text


def test_run_records_quality_issues(caplog):
    reply = {**GOOD_REPLY, "key_points": ["Only one"]}

    records = run([("short", "text")], FakeClient(json.dumps(reply)))

    assert records[0]["issues"] == ["1 key points (expected 3)"]
    assert "short: 1 key points (expected 3)" in caplog.text


def test_run_passes_prompt_to_every_text():
    client = FakeClient(GOOD_JSON)

    records = run([("a", "one"), ("b", "two")], client, OTHER_PROMPT)

    assert [record["prompt"] for record in records] == ["other"] * 2
    assert [call[0] for call in client.calls] == ["other system"] * 2


def test_format_record_shows_numbered_points():
    record = {
        "name": "x",
        "prompt": "v5_structured",
        "text": "t",
        "result": GOOD_RESULT,
        "priority": "medium",
    }
    text = format_record(record)
    assert "=== x (v5_structured) ===" in text
    assert "Category: question | Sentiment: neutral | Priority: medium" in text
    assert "  3. Third" in text
    assert "Final answer: A helpful answer." in text


def test_format_record_shows_issues():
    record = {
        "name": "x",
        "prompt": "v5_structured",
        "result": GOOD_RESULT,
        "priority": "medium",
        "issues": ["a", "b"],
    }
    assert "Issues: a; b" in format_record(record)


def test_format_record_shows_error():
    record = {"name": "x", "prompt": "v5_structured", "error": "boom"}
    assert "ERROR: boom" in format_record(record)


def test_main_saves_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        main_module, "LLMClient", lambda: FakeClient(GOOD_JSON)
    )
    output = tmp_path / "out" / "result.json"

    code = main_module.main(["--text", "Привет", "--output", str(output)])

    assert code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    [record] = saved["records"]
    assert record["text"] == "Привет"
    assert record["result"]["final_answer"] == "A helpful answer."
    assert record["prompt"] == DEFAULT_PROMPT
    assert saved["summary"]["by_category"]["question"] == 1
    assert saved["summary"]["by_priority"]["medium"] == ["cli"]
    out = capsys.readouterr().out
    assert "Summary: A short summary." in out
    assert "=== Summary: 1 of 1 texts processed ===" in out


def test_main_uses_prompt_option(tmp_path, monkeypatch):
    client = FakeClient(GOOD_JSON)
    monkeypatch.setattr(main_module, "LLMClient", lambda: client)
    monkeypatch.setitem(main_module.PROMPTS, "other", OTHER_PROMPT)
    output = tmp_path / "result.json"

    argv = ["--text", "hi", "--prompt", "other", "--output", str(output)]
    assert main_module.main(argv) == 0

    assert client.calls[0][0] == "other system"
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["records"][0]["prompt"] == "other"


def test_main_fails_when_a_reply_is_broken(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        main_module, "LLMClient", lambda: FakeClient('{"summary": "cut')
    )
    output = tmp_path / "result.json"

    assert main_module.main(["--text", "hi", "--output", str(output)]) == 1

    out = capsys.readouterr().out
    assert "ERROR: Model returned invalid JSON" in out
    assert "Failed: cli" in out
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["summary"]["failed"] == ["cli"]


def test_main_rejects_unknown_prompt():
    with pytest.raises(SystemExit):
        main_module.main(["--text", "hi", "--prompt", "nope"])


def test_main_without_key_exits_with_error(monkeypatch, capsys):
    def no_key():
        raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(main_module, "LLMClient", no_key)

    assert main_module.main(["--text", "hi"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
