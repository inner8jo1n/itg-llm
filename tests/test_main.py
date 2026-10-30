import json
from pathlib import Path

import pytest

from itg_llm import main as main_module
from itg_llm.main import format_record, load_inputs, load_labels
from itg_llm.pipeline import run
from itg_llm.schemas import Category

TEXT = "Не могу войти в кабинет"


@pytest.fixture
def record(make_llm, check_failed):
    llm = make_llm(check=[check_failed, check_failed])
    [record] = run([("x", TEXT)], llm)
    return record


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


def test_load_labels(tmp_path):
    assert load_labels(tmp_path) == {}
    (tmp_path / "labels.json").write_text('{"a": "news"}', encoding="utf-8")
    assert load_labels(tmp_path) == {"a": "news"}


def test_bundled_examples_are_labelled_with_known_categories():
    examples = Path(__file__).parent.parent / "examples"
    labels = load_labels(examples)

    assert set(labels) == {path.stem for path in examples.glob("*.txt")}
    assert len(labels) >= 10
    assert set(labels.values()) <= {category.value for category in Category}
    assert {"support", "complaint", "sales", "feedback"} <= set(
        labels.values()
    )


def test_format_record_shows_every_step_of_the_chain(record):
    lines = format_record(record).splitlines()

    assert lines[0] == "=== x (v7_extract) ==="
    assert lines[1] == (
        "[extract]     Summary: Пользователь не может войти в личный кабинет."
    )
    assert lines[2] == "              Goal: Войти в личный кабинет."
    assert lines[4].startswith("              Facts: Ошибка «Сессия истекла»;")
    assert lines[5] == (
        "[classify]    support / fix_login / negative -> priority high"
    )
    assert lines[6] == (
        "[route]       support -> technical_steps (by category)"
    )
    assert lines[7] == "[fields]      Subject: Личный кабинет"
    assert lines[10] == "              Missing: Версия браузера"
    assert lines[11].startswith("[answer]      1. Очистите cookies.")
    assert lines[12] == (
        "[check]       contradiction: Ответ говорит от лица компании; "
        "missing: Пароль сбрасывали дважды"
    )
    assert lines[13].startswith("[answer r1]   1.")
    assert lines[14].startswith("[check r1]    contradiction:")
    assert lines[15].startswith("Final answer: 1. Очистите cookies.")
    assert lines[16] == "Self-check: FAILED after 1 revision(s)"


def test_format_record_shows_passed_check(make_llm):
    [record] = run([("x", TEXT)], make_llm())
    text = format_record(record)
    assert "[check]       passed: Верно." in text
    assert "Self-check: passed" in text
    assert "Issues" not in text


def test_format_record_shows_issues(record):
    record = {**record, "issues": ["a", "b"]}
    assert "Issues: a; b" in format_record(record)


def test_format_record_shows_error():
    record = {"name": "x", "prompt": "v7_extract", "error": "boom"}
    assert "ERROR: boom" in format_record(record)


def use_llm(monkeypatch, llm):
    monkeypatch.setattr(main_module, "LLMClient", lambda: llm)


def test_main_saves_records_and_summary(
    tmp_path, monkeypatch, capsys, make_llm
):
    use_llm(monkeypatch, make_llm())
    output = tmp_path / "out" / "result.json"

    code = main_module.main(["--text", "Привет", "--output", str(output)])

    assert code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    [record] = saved["records"]
    assert record["text"] == "Привет"
    assert record["route"] == "technical_steps"
    assert saved["summary"]["by_route"] == {"technical_steps": 1}
    assert "accuracy" not in saved["summary"]
    assert [step["step"] for step in record["steps"]] == [
        "extract",
        "classify",
        "route",
        "fields",
        "answer",
        "check",
    ]
    assert saved["summary"]["self_check"]["passed"] == ["cli"]
    out = capsys.readouterr().out
    assert "[route]       support -> technical_steps (by category)" in out
    assert "=== Summary: 1 of 1 texts processed ===" in out
    assert "Self-check: 1 passed" in out


def test_main_on_examples_reports_accuracy(tmp_path, monkeypatch, make_llm):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "a.txt").write_text("Не могу войти", encoding="utf-8")
    (examples / "b.txt").write_text("Сколько стоит?", encoding="utf-8")
    labels = {"a": "support", "b": "sales"}
    (examples / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    use_llm(monkeypatch, make_llm())
    output = tmp_path / "result.json"

    argv = ["--examples", str(examples), "--output", str(output)]
    assert main_module.main(argv) == 0

    saved = json.loads(output.read_text(encoding="utf-8"))
    accuracy = saved["summary"]["accuracy"]
    assert accuracy["correct"] == 1
    assert accuracy["mismatches"] == [
        {"name": "b", "expected": "sales", "got": "support"}
    ]


def test_main_route_option_forces_the_route(tmp_path, monkeypatch, make_llm):
    use_llm(monkeypatch, make_llm())
    output = tmp_path / "result.json"

    argv = ["--text", "hi", "--route", "sales", "--output", str(output)]
    assert main_module.main(argv) == 0

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["records"][0]["route"] == "short_sales"
    assert saved["records"][0]["result"]["category"] == "support"


def test_main_rejects_unknown_prompt_and_route():
    with pytest.raises(SystemExit):
        main_module.main(["--text", "hi", "--prompt", "nope"])
    with pytest.raises(SystemExit):
        main_module.main(["--text", "hi", "--route", "nope"])


def test_main_fails_when_a_reply_is_broken(
    tmp_path, monkeypatch, capsys, make_llm
):
    use_llm(monkeypatch, make_llm(extract=['{"summary": "cut']))
    output = tmp_path / "result.json"

    assert main_module.main(["--text", "hi", "--output", str(output)]) == 1

    out = capsys.readouterr().out
    assert "ERROR: Model returned invalid JSON" in out
    assert "Failed: cli" in out
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["summary"]["failed"] == ["cli"]


def test_main_without_key_exits_with_error(monkeypatch, capsys):
    def no_key():
        raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(main_module, "LLMClient", no_key)

    assert main_module.main(["--text", "hi"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["[]", '{"a": "unknown"}', '{"a": []}', "bad"])
def test_invalid_labels_are_readable(tmp_path, capsys, bad):
    (tmp_path / "a.txt").write_text("Текст")
    (tmp_path / "labels.json").write_text(bad)
    assert main_module.main(["--examples", str(tmp_path)]) == 1
    assert "ERROR:" in capsys.readouterr().err


def test_missing_input_is_readable(tmp_path, capsys):
    assert main_module.main(["--file", str(tmp_path / "missing.txt")]) == 1
    assert "ERROR:" in capsys.readouterr().err


def test_output_error_is_readable(tmp_path, monkeypatch, capsys, make_llm):
    use_llm(monkeypatch, make_llm())
    output = tmp_path / "directory"
    output.mkdir()
    assert (
        main_module.main(
            [
                "--text",
                TEXT,
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "Cannot save results" in capsys.readouterr().err
