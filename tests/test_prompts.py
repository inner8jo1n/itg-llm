import pytest

from itg_llm.prompts import (
    CATEGORY_HINTS,
    DEFAULT_PROMPT,
    KEY_POINTS_COUNT,
    MAX_SUMMARY_CHARS,
    PROMPTS,
    V7_EXTRACT,
    answer_language,
    build_answer_prompt,
    build_check_prompt,
    build_classify_prompt,
    build_fields_prompt,
)
from itg_llm.routing import ROUTES
from itg_llm.schemas import (
    Category,
    Classification,
    Meaning,
    RequestFields,
    Sentiment,
)

TEXT = "Не могу войти {в кабинет}"


@pytest.fixture
def steps(meaning, classification, fields):
    return (
        Meaning(**meaning),
        Classification(**classification),
        RequestFields(**fields),
    )


def test_registry_has_extraction_variant_as_default():
    assert list(PROMPTS) == ["v7_extract"]
    assert DEFAULT_PROMPT == "v7_extract"
    assert PROMPTS[DEFAULT_PROMPT] is V7_EXTRACT


def test_extract_prompt_asks_for_every_meaning_field():
    prompt = V7_EXTRACT.build_user_prompt("  My printer stopped.  \n")

    assert prompt.startswith("Extract the meaning")
    assert '"""\nMy printer stopped.\n"""' in prompt
    for field in Meaning.model_fields:
        assert f'"{field}"' in prompt
    assert f"exactly {KEY_POINTS_COUNT} strings" in prompt
    assert f"at most {MAX_SUMMARY_CHARS} characters" in prompt
    assert "Write every value in English." in prompt


def test_classify_prompt_uses_the_meaning(steps):
    meaning, _, _ = steps

    prompt = build_classify_prompt(TEXT, meaning)

    assert prompt.startswith("Classify the request")
    assert f"Author's goal: {meaning.author_goal}" in prompt
    assert f"Summary: {meaning.summary}" in prompt
    for field in Classification.model_fields:
        assert f'"{field}"' in prompt
    for category in Category:
        assert f'"{category.value}": {CATEGORY_HINTS[category]}' in prompt
    for sentiment in Sentiment:
        assert f'"{sentiment.value}"' in prompt
    assert TEXT in prompt


def test_every_category_has_a_hint():
    assert set(CATEGORY_HINTS) == set(Category)


def test_fields_prompt_uses_meaning_and_classification(steps):
    meaning, classification, _ = steps

    prompt = build_fields_prompt(TEXT, meaning, classification)

    assert prompt.startswith("Build the structured fields")
    assert "Category: support, intent: fix_login, sentiment: negative" in (
        prompt
    )
    for fact in meaning.facts:
        assert f"- {fact}" in prompt
    for field in RequestFields.model_fields:
        assert f'"{field}"' in prompt
    assert "Write every value in Russian." in prompt


def test_answer_prompt_uses_every_earlier_step_and_the_route(steps):
    meaning, classification, fields = steps
    route = ROUTES[Category.SUPPORT]

    prompt = build_answer_prompt(TEXT, meaning, classification, fields, route)

    assert prompt.startswith("Write the answer")
    for part in (
        meaning.summary,
        meaning.author_goal,
        "intent: fix_login",
        f"Subject: {fields.subject}",
        f"Requested action: {fields.requested_action}",
        "- Ошибка «Сессия истекла»",
        "- Версия браузера",
        route.instructions,
        f"at most {route.max_chars} characters",
        "write in Russian",
        '{"final_answer": "..."}',
        TEXT,
    ):
        assert part in prompt
    assert "A reviewer found problems" not in prompt


def test_answer_prompt_lists_problems_to_fix(steps):
    meaning, classification, fields = steps
    prompt = build_answer_prompt(
        TEXT,
        meaning,
        classification,
        fields,
        ROUTES[Category.SUPPORT],
        ["contradiction: x"],
    )
    assert "A reviewer found problems" in prompt
    assert "- contradiction: x" in prompt


def test_empty_lists_are_shown_as_none(steps):
    meaning, classification, fields = steps
    empty = fields.model_copy(update={"details": [], "missing_info": []})

    prompt = build_answer_prompt(
        TEXT, meaning, classification, empty, ROUTES[Category.NEWS]
    )

    assert prompt.count("- (none)") == 2


def test_answer_prompt_differs_by_route(steps):
    meaning, classification, fields = steps
    prompts = {
        build_answer_prompt("text", meaning, classification, fields, route)
        for route in ROUTES.values()
    }
    assert len(prompts) == len(ROUTES)


def test_check_prompt_has_text_facts_details_route_and_answer(steps):
    meaning, _, fields = steps
    route = ROUTES[Category.SALES]

    prompt = build_check_prompt(
        TEXT, meaning, fields, route, "1. Очистите cookies."
    )

    assert prompt.startswith("Check the answer")
    assert "Details the answer needs:\n- Ошибка «Сессия истекла»" in prompt
    assert route.instructions in prompt
    assert f"at most {route.max_chars} characters" in prompt
    assert "Never list other facts" in prompt
    assert '"""\n1. Очистите cookies.\n"""' in prompt
    for fact in meaning.facts:
        assert f"- {fact}" in prompt
    assert '"contradictions"' in prompt
    assert '"missing_details"' in prompt
    assert '"we", "our"' in prompt
    assert TEXT in prompt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Мой заказ опаздывает", "Russian"),
        ("My order is late", "English"),
        ("12345", "the same language as the text"),
    ],
)
def test_answer_language(text, expected):
    assert answer_language(text) == expected


def test_prompts_keep_braces_in_text(steps):
    meaning, _, _ = steps
    assert 'a = {"b": 1}' in V7_EXTRACT.build_user_prompt('a = {"b": 1}')
    assert 'a = {"b": 1}' in build_classify_prompt('a = {"b": 1}', meaning)
