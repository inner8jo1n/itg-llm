from collections import Counter
from enum import StrEnum

from itg_llm.schemas import AnalysisResult, Category, Sentiment


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def priority(result: AnalysisResult) -> Priority:
    """How soon a person should look at the text, from category + sentiment.

    An unhappy complaint comes first, questions wait for an answer,
    news and feedback can be read later unless they are negative.
    """
    if result.category is Category.COMPLAINT:
        if result.sentiment is Sentiment.NEGATIVE:
            return Priority.HIGH
        return Priority.MEDIUM
    if result.category is Category.QUESTION:
        return Priority.MEDIUM
    if result.sentiment is Sentiment.NEGATIVE:
        return Priority.MEDIUM
    return Priority.LOW


def build_summary(records: list[dict]) -> dict:
    """Totals of one run by category, sentiment and priority."""
    ok = [record for record in records if "result" in record]
    by_priority = {level.value: [] for level in Priority}
    for record in ok:
        by_priority[record["priority"]].append(record["name"])
    return {
        "total": len(records),
        "failed": [r["name"] for r in records if "error" in r],
        "by_category": _count(ok, "category", Category),
        "by_sentiment": _count(ok, "sentiment", Sentiment),
        "by_priority": by_priority,
    }


def _count(records: list[dict], field: str, labels: type[StrEnum]) -> dict:
    counts = Counter(record["result"][field] for record in records)
    return {label.value: counts[label.value] for label in labels}


def format_summary(summary: dict) -> str:
    """Render the run summary as readable console text."""
    ok = summary["total"] - len(summary["failed"])
    lines = [f"=== Summary: {ok} of {summary['total']} texts processed ==="]
    if summary["failed"]:
        lines.append("Failed: " + ", ".join(summary["failed"]))
    lines.append("Categories: " + _non_zero(summary["by_category"]))
    lines.append("Sentiment: " + _non_zero(summary["by_sentiment"]))
    for level, names in summary["by_priority"].items():
        if names:
            lines.append(f"Priority {level}: " + ", ".join(names))
    return "\n".join(lines)


def _non_zero(counts: dict[str, int]) -> str:
    parts = [f"{label} {count}" for label, count in counts.items() if count]
    return ", ".join(parts) or "-"
