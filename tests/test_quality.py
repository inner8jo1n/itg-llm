import pytest

from itg_llm.prompts import (
    MAX_ANSWER_CHARS,
    MAX_KEY_POINT_WORDS,
    MAX_SUMMARY_CHARS,
)
from itg_llm.quality import check_quality
from itg_llm.schemas import AnalysisResult

GOOD = {
    "summary": "Заказ задерживается на две недели.",
    "category": "complaint",
    "sentiment": "negative",
    "key_points": ["Заказ оплачен", "Срок доставки прошёл", "Ответа нет"],
    "final_answer": "Напишите в поддержку ещё раз и приложите номер заказа.",
}
TEXT = "Мой заказ опаздывает уже две недели, поддержка молчит."


def check(data):
    return check_quality(AnalysisResult.model_validate(data), TEXT)


def test_good_result_has_no_issues():
    assert check(GOOD) == []


def test_long_summary():
    result = {**GOOD, "summary": "а" * (MAX_SUMMARY_CHARS + 1)}
    assert check(result) == [
        f"summary is {MAX_SUMMARY_CHARS + 1} chars (max {MAX_SUMMARY_CHARS})"
    ]


def test_wrong_number_of_key_points():
    result = {**GOOD, "key_points": ["Один", "Два"]}
    assert check(result) == ["2 key points (expected 3)"]


def test_long_key_point():
    long_point = " ".join(["слово"] * (MAX_KEY_POINT_WORDS + 1))
    result = {**GOOD, "key_points": ["Один", long_point, "Три"]}
    assert check(result) == [
        f"key point 2 is {MAX_KEY_POINT_WORDS + 1} words "
        f"(max {MAX_KEY_POINT_WORDS})"
    ]


def test_long_final_answer():
    result = {**GOOD, "final_answer": "д" * (MAX_ANSWER_CHARS + 1)}
    assert check(result) == [
        f"final_answer is {MAX_ANSWER_CHARS + 1} chars "
        f"(max {MAX_ANSWER_CHARS})"
    ]


def test_wrong_language():
    result = {**GOOD, "final_answer": "Please contact support again."}
    assert check(result) == ["final_answer is not in the text's language"]


@pytest.mark.parametrize(
    "answer",
    [
        "Мы постараемся решить проблему.",
        "Наша служба поддержки свяжется с вами.",
    ],
)
def test_answer_speaking_as_we(answer):
    result = {**GOOD, "final_answer": answer}
    assert check(result) == ["final_answer speaks as 'we'"]


def test_we_inside_a_word_is_fine():
    result = {**GOOD, "final_answer": "Мысль верная, напишите в поддержку."}
    assert check(result) == []
