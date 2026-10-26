import json

import pytest

from itg_llm import compare as compare_module
from itg_llm.compare import compare, summarize, to_markdown
from itg_llm.prompts import PromptVariant


def reply(answer="Короткий ответ.", points=3):
    return json.dumps(
        {
            "summary": "Краткое содержание.",
            "key_points": [f"Мысль {n}" for n in range(points)],
            "category": "question",
            "sentiment": "neutral",
            "final_answer": answer,
        }
    )


class FakeClient:
    """Replies with the next item of `replies` on every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.systems = []

    def complete(self, system, user, json_mode=False):
        self.systems.append(system)
        return self.replies.pop(0)


INPUTS = [("a", "Первый текст"), ("b", "Второй текст")]


def variant(name):
    return PromptVariant(
        name=name, description="", system=f"{name} system", template="{text}"
    )


FIRST, SECOND, THIRD = variant("first"), variant("second"), variant("third")


def test_compare_runs_every_prompt_on_every_text():
    prompts = [FIRST, THIRD]
    client = FakeClient([reply()] * 8)

    records = compare(INPUTS, client, prompts, runs=2)

    assert [(r["prompt"], r["run"], r["name"]) for r in records] == [
        ("first", 1, "a"),
        ("first", 1, "b"),
        ("first", 2, "a"),
        ("first", 2, "b"),
        ("third", 1, "a"),
        ("third", 1, "b"),
        ("third", 2, "a"),
        ("third", 2, "b"),
    ]
    assert client.systems[0] == FIRST.system
    assert client.systems[-1] == THIRD.system


def test_summarize_counts_failures_issues_and_spread():
    client = FakeClient(
        [
            reply("Ответ."),  # a, run 1
            "not json",  # b, run 1
            reply("Ответ длиннее.", points=2),  # a, run 2
            reply("Ответ."),  # b, run 2
        ]
    )
    records = compare(INPUTS, client, [SECOND], runs=2)

    [row] = summarize(records)

    assert row == {
        "prompt": "second",
        "answers": 4,
        "failed": 1,
        "with_issues": 1,
        "issues": 1,
        "avg_summary_chars": len("Краткое содержание."),
        "avg_answer_chars": round((6 + 14 + 6) / 3),
        "answer_spread": round((14 - 6 + 0) / 2),
    }


def test_summarize_all_failed():
    records = compare(INPUTS, FakeClient(["bad", "bad"]), [FIRST], runs=1)
    [row] = summarize(records)
    assert row["failed"] == 2
    assert row["avg_answer_chars"] == 0
    assert row["answer_spread"] == 0


def test_to_markdown_has_header_and_one_line_per_prompt():
    records = compare(
        INPUTS, FakeClient([reply()] * 4), [FIRST, SECOND], runs=1
    )
    lines = to_markdown(summarize(records)).splitlines()

    assert lines[0].startswith("| Prompt | Answers | Failed |")
    assert lines[1].startswith("|---|")
    assert lines[2].startswith("| first | 2 | 0 |")
    assert lines[3].startswith("| second | 2 | 0 |")


def test_main_saves_json_and_table(tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "one.txt").write_text("Текст", encoding="utf-8")
    monkeypatch.setattr(
        compare_module, "LLMClient", lambda: FakeClient([reply()])
    )
    output = tmp_path / "out" / "cmp.json"

    code = compare_module.main(
        [
            "--prompts",
            "v5_structured",
            "--runs",
            "1",
            "--examples",
            str(examples),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert [record["prompt"] for record in saved] == ["v5_structured"]
    table = (tmp_path / "out" / "cmp.md").read_text(encoding="utf-8")
    assert "| v5_structured | 1 | 0 |" in table


def test_main_rejects_zero_runs():
    with pytest.raises(SystemExit):
        compare_module.main(["--runs", "0"])
