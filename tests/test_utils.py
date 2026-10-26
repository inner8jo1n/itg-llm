import pytest

from itg_llm.utils import language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Привет, как дела?", "ru"),
        ("Hello, how are you?", "en"),
        ("Ошибка в OpenAI API при вызове модели", "ru"),
        ("12345 !!!", "unknown"),
        ("", "unknown"),
    ],
)
def test_language(text, expected):
    assert language(text) == expected
