import pytest

from itg_llm.report import Priority, build_summary, format_summary, priority
from itg_llm.schemas import AnalysisResult


def result(category, sentiment):
    return AnalysisResult(
        summary="s",
        category=category,
        sentiment=sentiment,
        key_points=["a"],
        final_answer="f",
    )


@pytest.mark.parametrize(
    ("category", "sentiment", "expected"),
    [
        ("complaint", "negative", Priority.HIGH),
        ("complaint", "neutral", Priority.MEDIUM),
        ("question", "positive", Priority.MEDIUM),
        ("question", "negative", Priority.MEDIUM),
        ("news", "negative", Priority.MEDIUM),
        ("news", "neutral", Priority.LOW),
        ("feedback", "positive", Priority.LOW),
        ("feedback", "negative", Priority.MEDIUM),
        ("other", "neutral", Priority.LOW),
    ],
)
def test_priority(category, sentiment, expected):
    assert priority(result(category, sentiment)) is expected


def record(name, category, sentiment):
    parsed = result(category, sentiment)
    return {
        "name": name,
        "result": parsed.model_dump(mode="json"),
        "priority": priority(parsed).value,
    }


RECORDS = [
    record("complaint", "complaint", "negative"),
    record("question", "question", "neutral"),
    record("thanks", "feedback", "positive"),
    {"name": "broken", "error": "Model returned invalid JSON"},
]


def test_build_summary_counts_labels_and_groups_priorities():
    summary = build_summary(RECORDS)

    assert summary == {
        "total": 4,
        "failed": ["broken"],
        "by_category": {
            "question": 1,
            "complaint": 1,
            "news": 0,
            "feedback": 1,
            "other": 0,
        },
        "by_sentiment": {"positive": 1, "neutral": 1, "negative": 1},
        "by_priority": {
            "high": ["complaint"],
            "medium": ["question"],
            "low": ["thanks"],
        },
    }


def test_format_summary():
    text = format_summary(build_summary(RECORDS))

    assert text.splitlines() == [
        "=== Summary: 3 of 4 texts processed ===",
        "Failed: broken",
        "Categories: question 1, complaint 1, feedback 1",
        "Sentiment: positive 1, neutral 1, negative 1",
        "Priority high: complaint",
        "Priority medium: question",
        "Priority low: thanks",
    ]


def test_format_summary_when_everything_failed():
    text = format_summary(build_summary([RECORDS[-1]]))
    assert "Categories: -" in text
    assert "Priority" not in text
