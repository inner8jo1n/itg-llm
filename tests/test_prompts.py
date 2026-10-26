import json
import re

import pytest

from itg_llm.prompts import (
    CATEGORY_HINTS,
    DEFAULT_PROMPT,
    KEY_POINTS_COUNT,
    MAX_ANSWER_CHARS,
    MAX_SUMMARY_CHARS,
    PROMPTS,
    V5_STRUCTURED,
)
from itg_llm.schemas import AnalysisResult, Category, Sentiment


def test_registry_has_structured_variant_as_default():
    assert list(PROMPTS) == ["v5_structured"]
    assert DEFAULT_PROMPT == "v5_structured"
    assert PROMPTS[DEFAULT_PROMPT] is V5_STRUCTURED


def test_prompt_contains_text_and_every_schema_field():
    prompt = V5_STRUCTURED.build_user_prompt("  My printer stopped.  \n")

    assert '"""\nMy printer stopped.\n"""' in prompt
    for field in AnalysisResult.model_fields:
        assert f'"{field}"' in prompt
    assert f"exactly {KEY_POINTS_COUNT} strings" in prompt
    assert f"at most {MAX_SUMMARY_CHARS} characters" in prompt
    assert f"at most {MAX_ANSWER_CHARS} characters" in prompt


def test_prompt_lists_labels_from_the_schema():
    prompt = V5_STRUCTURED.build_user_prompt("text")
    for category in Category:
        assert f'"{category.value}": {CATEGORY_HINTS[category]}' in prompt
    for sentiment in Sentiment:
        assert f'"{sentiment.value}"' in prompt


def test_every_category_has_a_hint():
    assert set(CATEGORY_HINTS) == set(Category)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Мой заказ опаздывает", "in Russian"),
        ("My order is late", "in English"),
        ("12345", "in the same language as the text"),
    ],
)
def test_prompt_names_the_answer_language(text, expected):
    prompt = V5_STRUCTURED.build_user_prompt(text)
    assert f'"final_answer" {expected}.' in prompt


def test_prompt_keeps_braces_in_text():
    prompt = V5_STRUCTURED.build_user_prompt('config = {"a": 1}')
    assert 'config = {"a": 1}' in prompt


def test_example_passes_the_schema_and_the_limits():
    prompt = V5_STRUCTURED.build_user_prompt("text")
    match = re.search(r"JSON:\n(\{.*?\})\n", prompt, re.S)
    assert match is not None

    example = AnalysisResult.model_validate(json.loads(match[1]))

    assert len(example.summary) <= MAX_SUMMARY_CHARS
    assert len(example.key_points) == KEY_POINTS_COUNT
    assert len(example.final_answer) <= MAX_ANSWER_CHARS


def test_system_prompt_sets_neutral_role_and_json_only():
    assert "never speak on its behalf" in V5_STRUCTURED.system
    assert "single JSON object" in V5_STRUCTURED.system
