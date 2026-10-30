import pytest

from itg_llm.prompts import MAX_KEY_POINT_WORDS, MAX_SUMMARY_CHARS
from itg_llm.quality import check_quality
from itg_llm.routing import ROUTES
from itg_llm.schemas import AnalysisResult, Category

GOOD = {
    "summary": "Заказ задерживается на две недели.",
    "author_goal": "Получить заказ.",
    "key_points": ["Заказ оплачен", "Срок доставки прошёл", "Ответа нет"],
    "facts": [],
    "category": "complaint",
    "intent": "get_order",
    "sentiment": "negative",
    "subject": "Заказ",
    "requested_action": "Доставить заказ",
    "details": [],
    "missing_info": [],
    "final_answer": "Напишите в поддержку ещё раз и приложите номер заказа.",
}
TEXT = "Мой заказ опаздывает уже две недели, поддержка молчит."
COMPLAINT = ROUTES[Category.COMPLAINT]
SUPPORT = ROUTES[Category.SUPPORT]


def check(data, route=COMPLAINT):
    return check_quality(AnalysisResult.model_validate(data), TEXT, route)


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


def test_answer_longer_than_the_route_allows():
    sales = ROUTES[Category.SALES]
    result = {**GOOD, "final_answer": "д" * (sales.max_chars + 1)}

    assert check(result, COMPLAINT) == []
    assert check(result, sales) == [
        f"final_answer is {sales.max_chars + 1} chars "
        f"(max {sales.max_chars} for short_sales)"
    ]


@pytest.mark.parametrize(
    "answer",
    [
        "1. Очистите кэш. 2. Перезапустите приложение.",
        "Попробуйте так:\n1) очистите кэш;\n2) перезапустите.",
    ],
)
def test_support_answer_with_numbered_steps(answer):
    assert check({**GOOD, "final_answer": answer}, SUPPORT) == []


def test_support_answer_without_numbered_steps():
    result = {**GOOD, "final_answer": "Очистите кэш и перезапустите."}
    assert check(result, SUPPORT) == [
        "final_answer has no numbered steps (technical_steps)"
    ]
    assert check(result, COMPLAINT) == []


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
