import logging

import pytest

from itg_llm.pipeline import answer as answer_step
from itg_llm.pipeline import (
    build_fields,
    classify,
    extract,
    process_text,
    run,
    self_check,
)
from itg_llm.prompts import CHECKER_ROLE, NEUTRAL_ROLE, PROMPTS, PromptVariant
from itg_llm.routing import ROUTES
from itg_llm.schemas import (
    AnalysisResult,
    Category,
    Classification,
    LLMOutputError,
    Meaning,
    RequestFields,
)

OTHER_PROMPT = PromptVariant(
    name="other", description="", system="other system", template="{text}"
)
TEXT = "Не могу войти в кабинет: пишет «Сессия истекла»."
CHAIN = ["extract", "classify", "fields", "answer", "check"]


def test_extract(make_llm):
    llm = make_llm()

    meaning = extract(TEXT, llm)

    assert isinstance(meaning, Meaning)
    assert meaning.author_goal == "Войти в личный кабинет."
    [call] = llm.calls
    assert call["system"] == PROMPTS["v7_extract"].system
    assert TEXT in call["user"]
    assert call["json_mode"] is True


def test_classify_uses_the_meaning(make_llm, meaning):
    llm = make_llm()

    result = classify(TEXT, Meaning(**meaning), llm)

    assert result.category is Category.SUPPORT
    assert meaning["author_goal"] in llm.calls[0]["user"]


def test_build_fields_uses_meaning_and_classification(
    make_llm, meaning, classification
):
    llm = make_llm()

    fields = build_fields(
        TEXT, Meaning(**meaning), Classification(**classification), llm
    )

    assert fields.missing_info == ["Версия браузера"]
    user = llm.calls[0]["user"]
    assert "Category: support, intent: fix_login" in user
    assert "- Пароль сбрасывал два раза" in user


def test_answer_uses_every_earlier_step_and_the_route(
    make_llm, meaning, classification, fields
):
    llm = make_llm()
    route = ROUTES[Category.SALES]

    reply = answer_step(
        TEXT,
        Meaning(**meaning),
        Classification(**classification),
        RequestFields(**fields),
        route,
        llm,
    )

    assert reply.final_answer.startswith("1.")
    [call] = llm.calls
    assert call["system"] == NEUTRAL_ROLE
    for part in (
        route.instructions,
        meaning["author_goal"],
        "intent: fix_login",
        fields["requested_action"],
        "- Версия браузера",
    ):
        assert part in call["user"]


def test_self_check_sees_facts_details_route_and_answer(
    make_llm, meaning, fields
):
    llm = make_llm()
    route = ROUTES[Category.FEEDBACK]

    check = self_check(
        TEXT,
        Meaning(**meaning),
        RequestFields(**fields),
        route,
        "Мой ответ",
        llm,
    )

    assert check.passed
    [call] = llm.calls
    assert call["system"] == CHECKER_ROLE
    for part in (
        '"""\nМой ответ\n"""',
        "- Пароль сбрасывал два раза",
        "Details the answer needs:\n- Ошибка «Сессия истекла»",
        route.instructions,
        f"at most {route.max_chars} characters",
        TEXT,
    ):
        assert part in call["user"]


def test_process_text_runs_all_five_steps_in_order(make_llm):
    llm = make_llm()

    chain = process_text(TEXT, llm)

    assert llm.steps() == CHAIN
    assert all(call["json_mode"] for call in llm.calls)
    assert isinstance(chain.result, AnalysisResult)
    assert chain.result.category is Category.SUPPORT
    assert chain.result.author_goal == "Войти в личный кабинет."
    assert chain.result.subject == "Личный кабинет"
    assert chain.result.final_answer.startswith("1.")
    assert chain.route is ROUTES[Category.SUPPORT]
    assert chain.check.passed
    assert chain.revisions == 0


def test_process_text_traces_every_step_and_the_route(make_llm):
    chain = process_text(TEXT, make_llm())

    assert [step["step"] for step in chain.steps] == [
        "extract",
        "classify",
        "route",
        "fields",
        "answer",
        "check",
    ]
    route = chain.steps[2]["output"]
    assert route == {
        "category": "support",
        "route": "technical_steps",
        "forced": False,
    }
    assert chain.steps[0]["output"]["facts"][0] == "Ошибка «Сессия истекла»"
    assert all(isinstance(step["ms"], int) for step in chain.steps)


def test_process_text_logs_how_the_decision_is_made(make_llm, caplog):
    caplog.set_level(logging.INFO)

    process_text(TEXT, make_llm(), name="login")

    log = caplog.text
    assert "login | 1/5 extract: goal: Войти в личный кабинет." in log
    assert "login | 2/5 classify: support / fix_login / negative" in log
    assert "login | route: support -> technical_steps (by category)" in log
    assert "login | 3/5 fields: Личный кабинет" in log
    assert "login | 5/5 check: passed" in log


@pytest.mark.parametrize("category", list(Category))
def test_route_follows_the_classified_category(
    make_llm, classification, category
):
    llm = make_llm(classify=[{**classification, "category": category}])

    chain = process_text("text", llm)

    assert chain.route is ROUTES[category]
    answer_call = llm.calls[CHAIN.index("answer")]
    assert ROUTES[category].instructions in answer_call["user"]


def test_process_text_can_force_a_route(make_llm):
    llm = make_llm()

    chain = process_text(TEXT, llm, route_category=Category.COMPLAINT)

    assert chain.result.category is Category.SUPPORT
    assert chain.route is ROUTES[Category.COMPLAINT]
    assert chain.steps[2]["output"]["forced"] is True


def test_failed_check_revises_the_answer_once(make_llm, check_failed):
    llm = make_llm(
        answer=[
            {"final_answer": "Мы всё исправим."},
            {"final_answer": "1. Ок"},
        ],
        check=[check_failed, check_failed],
    )

    chain = process_text(TEXT, llm)

    assert llm.steps() == [*CHAIN, "answer", "check"]
    revision_prompt = llm.calls[5]["user"]
    assert "contradiction: Ответ говорит от лица компании" in revision_prompt
    assert "missing detail: Пароль сбрасывали дважды" in revision_prompt
    assert chain.result.final_answer == "1. Ок"
    assert chain.revisions == 1
    assert not chain.check.passed
    assert [s["revision"] for s in chain.steps][-2:] == [1, 1]


def test_passed_revision_stops_the_loop(make_llm, check_failed):
    llm = make_llm(
        check=[check_failed, {**check_failed, "contradictions": []}]
    )

    chain = process_text(TEXT, llm)

    assert chain.check.passed
    assert chain.revisions == 1


def test_missing_details_alone_do_not_revise(make_llm, check_failed):
    llm = make_llm(check=[{**check_failed, "contradictions": []}])

    chain = process_text(TEXT, llm)

    assert llm.steps() == CHAIN
    assert chain.check.passed
    assert chain.revisions == 0


def test_missing_details_become_issues(make_llm, check_failed, caplog):
    llm = make_llm(check=[{**check_failed, "contradictions": []}])

    [record] = run([("login", TEXT)], llm)

    assert record["check"]["passed"] is True
    assert record["issues"] == [
        "self-check: missing detail: Пароль сбрасывали дважды"
    ]
    assert "login: self-check: missing detail" in caplog.text


def test_no_revisions_when_disabled(make_llm, check_failed):
    llm = make_llm(check=[check_failed])

    chain = process_text(TEXT, llm, max_revisions=0)

    assert llm.steps() == CHAIN
    assert not chain.check.passed


def test_first_answer_prompt_has_no_problems_block(make_llm):
    llm = make_llm()
    process_text(TEXT, llm)
    assert "A reviewer found problems" not in llm.calls[3]["user"]


def test_process_text_uses_given_prompt_for_extraction(make_llm):
    llm = make_llm()
    process_text("Some text", llm, OTHER_PROMPT)
    assert llm.calls[0]["system"] == "other system"
    assert llm.calls[0]["user"] == "Some text"


def test_process_text_rejects_empty_text(make_llm):
    llm = make_llm()
    with pytest.raises(ValueError, match="empty"):
        process_text("   ", llm)
    assert llm.calls == []


@pytest.mark.parametrize("step", CHAIN)
def test_a_broken_step_stops_the_chain(make_llm, step):
    llm = make_llm(**{step: ['{"oops": ']})

    with pytest.raises(LLMOutputError, match="invalid JSON"):
        process_text(TEXT, llm)

    assert llm.steps() == [*CHAIN[: CHAIN.index(step) + 1], step]


def test_run_records_every_part_of_the_chain(make_llm):
    [record] = run([("login", TEXT)], make_llm())

    assert record["result"]["category"] == "support"
    assert record["result"]["facts"][0] == "Ошибка «Сессия истекла»"
    assert record["route"] == "technical_steps"
    assert record["priority"] == "high"
    assert record["check"] == {
        "passed": True,
        "revisions": 0,
        "contradictions": [],
        "missing_details": [],
        "comment": "Верно.",
    }
    assert len(record["steps"]) == 6
    assert record["issues"] == []
    assert record["prompt"] == "v7_extract"


def test_run_checks_quality_against_the_route(make_llm, caplog):
    llm = make_llm(answer=[{"final_answer": "Очистите cookies."}])

    [record] = run([("login", TEXT)], llm)

    assert record["issues"] == [
        "final_answer has no numbered steps (technical_steps)"
    ]
    assert "login: final_answer has no numbered steps" in caplog.text


def test_run_records_failures_and_keeps_going(make_llm, classification):
    llm = make_llm(
        classify=[
            {**classification, "sentiment": "angry"},
            {**classification, "sentiment": "angry"},
            classification,
        ]
    )

    records = run([("bad", TEXT), ("empty", " "), ("ok", TEXT)], llm)

    assert "field 'sentiment' must be one of" in records[0]["error"]
    assert records[1]["error"] == "Text is empty"
    assert "result" in records[2]
    for failed in records[:2]:
        assert not {"result", "route", "check", "steps"} & set(failed)


def test_run_passes_prompt_and_route_to_every_text(make_llm):
    records = run(
        [("a", "one"), ("b", "two")],
        make_llm(),
        OTHER_PROMPT,
        Category.FEEDBACK,
    )

    assert [record["prompt"] for record in records] == ["other"] * 2
    assert [record["route"] for record in records] == ["thanks"] * 2
