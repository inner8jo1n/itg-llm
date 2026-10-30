import pytest

from itg_llm.report import Priority, build_summary, format_summary, priority
from itg_llm.schemas import AnalysisResult


def result(category, sentiment):
    return AnalysisResult(
        summary="s",
        author_goal="g",
        key_points=["a"],
        facts=[],
        category=category,
        intent="i",
        sentiment=sentiment,
        subject="s",
        requested_action="r",
        details=[],
        missing_info=[],
        final_answer="f",
    )


@pytest.mark.parametrize(
    ("category", "sentiment", "expected"),
    [
        ("complaint", "negative", Priority.HIGH),
        ("complaint", "neutral", Priority.MEDIUM),
        ("support", "negative", Priority.HIGH),
        ("support", "neutral", Priority.MEDIUM),
        ("sales", "positive", Priority.MEDIUM),
        ("general_question", "neutral", Priority.MEDIUM),
        ("news", "negative", Priority.MEDIUM),
        ("news", "neutral", Priority.LOW),
        ("feedback", "positive", Priority.LOW),
        ("feedback", "negative", Priority.MEDIUM),
        ("other", "neutral", Priority.LOW),
    ],
)
def test_priority(category, sentiment, expected):
    assert priority(result(category, sentiment)) is expected


def record(name, category, sentiment, route, passed=True, revisions=0):
    parsed = result(category, sentiment)
    return {
        "name": name,
        "result": parsed.model_dump(mode="json"),
        "route": route,
        "priority": priority(parsed).value,
        "check": {"passed": passed, "revisions": revisions},
    }


RECORDS = [
    record("login", "support", "negative", "technical_steps", revisions=1),
    record("price", "sales", "neutral", "short_sales", passed=False),
    record("thanks", "feedback", "positive", "thanks"),
    {"name": "broken", "error": "Model returned invalid JSON"},
]


def test_build_summary_counts_labels_routes_and_priorities():
    summary = build_summary(RECORDS)

    assert summary == {
        "total": 4,
        "failed": ["broken"],
        "by_category": {
            "support": 1,
            "complaint": 0,
            "sales": 1,
            "feedback": 1,
            "general_question": 0,
            "news": 0,
            "other": 0,
        },
        "by_sentiment": {"positive": 1, "neutral": 1, "negative": 1},
        "by_route": {"technical_steps": 1, "short_sales": 1, "thanks": 1},
        "by_priority": {
            "high": ["login"],
            "medium": ["price"],
            "low": ["thanks"],
        },
        "self_check": {
            "passed": ["login", "thanks"],
            "revised": ["login"],
            "failed": ["price"],
        },
    }


def test_build_summary_with_labels_counts_accuracy():
    labels = {
        "login": "support",
        "price": "general_question",
        "broken": "complaint",
        "unknown": "news",
    }

    accuracy = build_summary(RECORDS, labels)["accuracy"]

    assert accuracy == {
        "correct": 1,
        "total": 2,
        "mismatches": [
            {"name": "price", "expected": "general_question", "got": "sales"}
        ],
    }


def test_format_summary():
    labels = {"login": "support", "price": "general_question"}
    text = format_summary(build_summary(RECORDS, labels))

    assert text.splitlines() == [
        "=== Summary: 3 of 4 texts processed ===",
        "Failed: broken",
        "Categories: support 1, sales 1, feedback 1",
        "Sentiment: positive 1, neutral 1, negative 1",
        "Routes: technical_steps 1, short_sales 1, thanks 1",
        "Priority high: login",
        "Priority medium: price",
        "Priority low: thanks",
        "Self-check: 2 passed; revised: login; failed: price",
        "Category accuracy: 1 of 2",
        "  price: expected general_question, got sales",
    ]


def test_format_summary_when_everything_failed():
    text = format_summary(build_summary([RECORDS[-1]]))
    assert "Categories: -" in text
    assert "Routes: -" in text
    assert "Priority" not in text
    assert "accuracy" not in text
