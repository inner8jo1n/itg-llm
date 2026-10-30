import json

import pytest

from itg_llm import compare_routes as module
from itg_llm.compare_routes import answer_with_every_route, format_routes
from itg_llm.routing import ROUTES
from itg_llm.schemas import Category


def test_runs_steps_1_to_3_once_and_answers_with_every_route(make_llm):
    llm = make_llm()
    categories = [Category.SUPPORT, Category.SALES, Category.FEEDBACK]

    data = answer_with_every_route("Не могу войти", llm, categories)

    steps = [call["step"] for call in llm.calls]
    assert steps == ["extract", "classify", "fields", *["answer"] * 3]
    for call, category in zip(llm.calls[3:], categories, strict=True):
        assert ROUTES[category].instructions in call["user"]
    assert data["meaning"]["author_goal"] == "Войти в личный кабинет."
    assert data["classification"]["category"] == "support"
    assert data["fields"]["subject"] == "Личный кабинет"
    assert [item["route"] for item in data["answers"]] == [
        "technical_steps",
        "short_sales",
        "thanks",
    ]
    assert data["answers"][0]["chars"] == len(
        data["answers"][0]["final_answer"]
    )


def test_a_broken_answer_does_not_stop_the_other_routes(make_llm):
    llm = make_llm(answer=["oops", "oops", {"final_answer": "Ок"}])

    data = answer_with_every_route(
        "text", llm, [Category.SUPPORT, Category.SALES]
    )

    assert "invalid JSON" in data["answers"][0]["error"]
    assert data["answers"][1]["final_answer"] == "Ок"


def test_format_routes_marks_the_classified_route(make_llm):
    data = answer_with_every_route(
        "text", make_llm(), [Category.SUPPORT, Category.SALES]
    )
    lines = format_routes(data).splitlines()

    assert lines[0] == "Classified as support (intent fix_login)"
    assert lines[2].startswith("* [technical_steps, ")
    assert lines[4].startswith("  [short_sales, ")


def test_main_saves_json(tmp_path, monkeypatch, capsys, make_llm):
    monkeypatch.setattr(module, "LLMClient", lambda: make_llm())
    output = tmp_path / "routes.json"

    argv = [
        "--text",
        "hi",
        "--routes",
        "sales",
        "news",
        "--output",
        str(output),
    ]
    assert module.main(argv) == 0

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert [item["category"] for item in saved["answers"]] == ["sales", "news"]
    assert "[explain_news, " in capsys.readouterr().out


def test_main_needs_a_text():
    with pytest.raises(SystemExit):
        module.main([])


def test_api_failure_does_not_stop_other_routes(make_llm):
    from itg_llm.llm_client import LLMAPIError

    delegate = make_llm()

    class PartialOutage:
        failed = False

        def complete(self, system, user, json_mode=False):
            if user.startswith("Write the answer") and not self.failed:
                self.failed = True
                raise LLMAPIError("LLM API failed: HTTP 503")
            return delegate.complete(system, user, json_mode)

    data = answer_with_every_route(
        "Не могу войти", PartialOutage(), [Category.SUPPORT, Category.SALES]
    )
    assert "HTTP 503" in data["answers"][0]["error"]
    assert "final_answer" in data["answers"][1]


def test_missing_file_has_readable_error(tmp_path, capsys):
    assert module.main(["--file", str(tmp_path / "missing")]) == 1
    assert "ERROR:" in capsys.readouterr().err
