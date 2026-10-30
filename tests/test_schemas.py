import json

import pytest
from pydantic import ValidationError

from itg_llm.schemas import (
    AnalysisResult,
    Answer,
    Category,
    Classification,
    LLMOutputError,
    Meaning,
    RequestFields,
    SelfCheck,
    Sentiment,
    parse_response,
)

MEANING = {
    "summary": "  Заказ задерживается.  ",
    "author_goal": "Получить заказ или деньги.",
    "key_points": ["Заказ оплачен", " Срок прошёл ", "Ответа нет"],
    "facts": ["Заказ №48213", "Оплачен 2 октября"],
}
CLASSIFICATION = {
    "category": "complaint",
    "intent": " get_order ",
    "sentiment": "negative",
}


def error_fields(data, schema):
    with pytest.raises(ValidationError) as info:
        schema.model_validate(data)
    return {error["loc"][0] for error in info.value.errors()}


def test_meaning_is_parsed_and_stripped():
    meaning = Meaning.model_validate(MEANING)
    assert meaning.summary == "Заказ задерживается."
    assert meaning.key_points == ["Заказ оплачен", "Срок прошёл", "Ответа нет"]
    assert meaning.facts == ["Заказ №48213", "Оплачен 2 октября"]


def test_meaning_allows_no_facts_but_needs_key_points():
    assert Meaning.model_validate({**MEANING, "facts": []}).facts == []
    assert error_fields({**MEANING, "key_points": []}, Meaning) == {
        "key_points"
    }


def test_classification_is_parsed():
    result = Classification.model_validate(CLASSIFICATION)
    assert result.category is Category.COMPLAINT
    assert result.intent == "get_order"
    assert result.sentiment is Sentiment.NEGATIVE


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Complaint ", Category.COMPLAINT),
        ("GENERAL_QUESTION", Category.GENERAL_QUESTION),
        ("general question", Category.GENERAL_QUESTION),
        ("general-question", Category.GENERAL_QUESTION),
    ],
)
def test_category_spelling_is_normalized(raw, expected):
    data = {**CLASSIFICATION, "category": raw}
    assert Classification.model_validate(data).category is expected


def test_request_fields_allow_empty_lists():
    fields = RequestFields(
        subject="Заказ",
        requested_action="Вернуть деньги",
        details=[],
        missing_info=[],
    )
    assert fields.details == fields.missing_info == []


@pytest.mark.parametrize(
    ("contradictions", "missing", "passed"),
    [
        ([], [], True),
        (["x"], [], False),
        ([], ["y"], True),
        (["x"], ["y"], False),
    ],
)
def test_self_check_passed(contradictions, missing, passed):
    check = SelfCheck(
        contradictions=contradictions, missing_details=missing, comment="c"
    )
    assert check.passed is passed


def test_analysis_result_holds_every_step_in_chain_order():
    assert list(AnalysisResult.model_fields) == [
        *Meaning.model_fields,
        *Classification.model_fields,
        *RequestFields.model_fields,
        *Answer.model_fields,
    ]


def test_extra_fields_are_ignored():
    result = Classification.model_validate({**CLASSIFICATION, "x": 1})
    assert not hasattr(result, "x")


@pytest.mark.parametrize(
    ("schema", "data"),
    [(Meaning, MEANING), (Classification, CLASSIFICATION)],
)
def test_missing_field(schema, data):
    for field in data:
        partial = {key: value for key, value in data.items() if key != field}
        assert error_fields(partial, schema) == {field}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", "rant"),
        ("category", 1),
        ("intent", None),
        ("intent", " "),
        ("sentiment", "angry"),
    ],
)
def test_wrong_classification_value(field, value):
    data = {**CLASSIFICATION, field: value}
    assert error_fields(data, Classification) == {field}


def test_enums_list_allowed_values():
    assert [c.value for c in Category] == [
        "support",
        "complaint",
        "sales",
        "feedback",
        "general_question",
        "news",
        "other",
    ]
    assert [s.value for s in Sentiment] == ["positive", "neutral", "negative"]


def test_parse_response_returns_the_given_schema():
    result = parse_response(json.dumps(CLASSIFICATION), Classification)
    assert result.category is Category.COMPLAINT
    answer = parse_response('{"final_answer": "Да"}', Answer)
    assert answer.final_answer == "Да"


def test_parse_response_strips_code_fences():
    raw = json.dumps(CLASSIFICATION)
    for fenced in ("```json\n" + raw + "\n```", "```" + raw + "```"):
        assert parse_response(fenced, Classification).intent == "get_order"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "Model returned an empty reply"),
        ("```json\n```", "Model returned an empty reply"),
        (
            '{"summary": "cut',
            "Model returned invalid JSON: Unterminated string starting at "
            "(line 1, column 13)",
        ),
        ("Sure! Here it is.", "Model returned invalid JSON: Expecting value"),
        ("[1, 2]", "Model returned a JSON list instead of an object"),
        ('"text"', "Model returned a JSON string instead of an object"),
    ],
)
def test_parse_response_explains_broken_json(raw, message):
    with pytest.raises(LLMOutputError) as info:
        parse_response(raw, Classification)
    assert str(info.value).startswith(message)


@pytest.mark.parametrize(
    ("schema", "data", "changes", "message"),
    [
        (
            Classification,
            CLASSIFICATION,
            {"intent": None},
            "field 'intent' must be a string, got null",
        ),
        (
            Classification,
            CLASSIFICATION,
            {"category": "rant"},
            "field 'category' must be one of support, complaint, sales, "
            "feedback, general_question, news or other, got 'rant'",
        ),
        (
            Classification,
            CLASSIFICATION,
            {"sentiment": "angry"},
            "field 'sentiment' must be one of positive, neutral or negative, "
            "got 'angry'",
        ),
        (
            Meaning,
            MEANING,
            {"summary": 42},
            "field 'summary' must be a string, got number",
        ),
        (
            Meaning,
            MEANING,
            {"summary": " "},
            "field 'summary' must not be empty",
        ),
        (
            Meaning,
            MEANING,
            {"key_points": "a, b"},
            "field 'key_points' must be a list, got string",
        ),
        (
            Meaning,
            MEANING,
            {"key_points": []},
            "field 'key_points' must not be an empty list",
        ),
        (
            Meaning,
            MEANING,
            {"facts": ["a", 2]},
            "field 'facts[1]' must be a string, got number",
        ),
    ],
)
def test_parse_response_explains_schema_errors(schema, data, changes, message):
    with pytest.raises(LLMOutputError) as info:
        parse_response(json.dumps({**data, **changes}), schema)
    assert str(info.value) == (
        "Model JSON does not match the schema: " + message
    )


def test_parse_response_lists_every_problem():
    data = {"category": "rant"}

    with pytest.raises(LLMOutputError) as info:
        parse_response(json.dumps(data), Classification)

    message = str(info.value)
    assert "field 'category' must be one of" in message
    assert "field 'intent' is missing" in message
    assert "field 'sentiment' is missing" in message


def test_llm_output_error_is_a_value_error():
    assert issubclass(LLMOutputError, ValueError)
