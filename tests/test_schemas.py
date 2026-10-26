import json

import pytest
from pydantic import ValidationError

from itg_llm.schemas import (
    AnalysisResult,
    Category,
    LLMOutputError,
    Sentiment,
    parse_response,
)

VALID = {
    "summary": "  Заказ задерживается.  ",
    "category": "complaint",
    "sentiment": "negative",
    "key_points": ["Заказ оплачен", " Срок прошёл ", "Ответа нет"],
    "final_answer": "Напишите в поддержку ещё раз.",
}


def error_fields(data):
    with pytest.raises(ValidationError) as info:
        AnalysisResult.model_validate(data)
    return {error["loc"][0] for error in info.value.errors()}


def test_valid_data_is_parsed_and_stripped():
    result = AnalysisResult.model_validate(VALID)

    assert result.summary == "Заказ задерживается."
    assert result.category is Category.COMPLAINT
    assert result.sentiment is Sentiment.NEGATIVE
    assert result.key_points == ["Заказ оплачен", "Срок прошёл", "Ответа нет"]
    assert result.final_answer == "Напишите в поддержку ещё раз."


def test_labels_are_case_and_space_insensitive():
    data = {**VALID, "category": " Complaint ", "sentiment": "NEGATIVE"}
    result = AnalysisResult.model_validate(data)
    assert result.category is Category.COMPLAINT
    assert result.sentiment is Sentiment.NEGATIVE


def test_extra_fields_are_ignored():
    result = AnalysisResult.model_validate({**VALID, "confidence": 0.9})
    assert not hasattr(result, "confidence")


def test_dump_gives_plain_json_values():
    dumped = AnalysisResult.model_validate(VALID).model_dump(mode="json")
    assert dumped["category"] == "complaint"
    assert dumped["sentiment"] == "negative"


@pytest.mark.parametrize("field", list(VALID))
def test_missing_field(field):
    data = {key: value for key, value in VALID.items() if key != field}
    assert error_fields(data) == {field}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", ""),
        ("summary", "   "),
        ("summary", 42),
        ("category", "rant"),
        ("category", 1),
        ("sentiment", "angry"),
        ("key_points", "one, two, three"),
        ("key_points", []),
        ("key_points", ["ok", ""]),
        ("key_points", ["ok", 2]),
        ("final_answer", None),
    ],
)
def test_wrong_value_or_type(field, value):
    assert error_fields({**VALID, field: value}) == {field}


def test_enums_list_allowed_values():
    assert [c.value for c in Category] == [
        "question",
        "complaint",
        "news",
        "feedback",
        "other",
    ]
    assert [s.value for s in Sentiment] == ["positive", "neutral", "negative"]


def test_parse_response_returns_result():
    result = parse_response(json.dumps(VALID))
    assert result.category is Category.COMPLAINT


def test_parse_response_strips_code_fence():
    raw = "```json\n" + json.dumps(VALID) + "\n```"
    assert parse_response(raw).sentiment is Sentiment.NEGATIVE


def test_parse_response_strips_bare_code_fence():
    assert parse_response("```" + json.dumps(VALID) + "```").summary


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
        parse_response(raw)
    assert str(info.value).startswith(message)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"final_answer": None},
            "field 'final_answer' must be a string, got null",
        ),
        ({"summary": 42}, "field 'summary' must be a string, got number"),
        ({"summary": " "}, "field 'summary' must not be empty"),
        (
            {"category": "rant"},
            "field 'category' must be one of question, complaint, news, "
            "feedback or other, got 'rant'",
        ),
        (
            {"sentiment": "angry"},
            "field 'sentiment' must be one of positive, neutral or negative, "
            "got 'angry'",
        ),
        (
            {"key_points": "a, b"},
            "field 'key_points' must be a list, got string",
        ),
        ({"key_points": []}, "field 'key_points' must not be an empty list"),
        (
            {"key_points": ["a", 2]},
            "field 'key_points[1]' must be a string, got number",
        ),
    ],
)
def test_parse_response_explains_schema_errors(changes, message):
    with pytest.raises(LLMOutputError) as info:
        parse_response(json.dumps({**VALID, **changes}))
    assert str(info.value) == (
        "Model JSON does not match the schema: " + message
    )


def test_parse_response_lists_every_problem():
    data = {key: value for key, value in VALID.items() if key != "summary"}
    data["category"] = "rant"

    with pytest.raises(LLMOutputError) as info:
        parse_response(json.dumps(data))

    message = str(info.value)
    assert "field 'summary' is missing" in message
    assert "field 'category' must be one of" in message


def test_llm_output_error_is_a_value_error():
    assert issubclass(LLMOutputError, ValueError)
