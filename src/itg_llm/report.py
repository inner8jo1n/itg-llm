from collections import Counter
from enum import StrEnum

from itg_llm.schemas import AnalysisResult, Category, Sentiment


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


NEEDS_REPLY = {Category.SALES, Category.GENERAL_QUESTION}
NEEDS_FIX = {Category.SUPPORT, Category.COMPLAINT}


def priority(result: AnalysisResult) -> Priority:
    """How soon a person should look at the text, from category + sentiment.

    An unhappy customer with a broken product or a complaint comes first,
    buyers and questions wait for an answer, the rest can be read later
    unless it is negative.
    """
    negative = result.sentiment is Sentiment.NEGATIVE
    if result.category in NEEDS_FIX:
        return Priority.HIGH if negative else Priority.MEDIUM
    if result.category in NEEDS_REPLY or negative:
        return Priority.MEDIUM
    return Priority.LOW


def build_summary(
    records: list[dict], labels: dict[str, str] | None = None
) -> dict:
    """Totals of one run by category, sentiment, route, priority, self-check.

    With labels (expected category per text name) also the accuracy of
    the classification.
    """
    ok = [record for record in records if "result" in record]
    by_priority = {level.value: [] for level in Priority}
    for record in ok:
        by_priority[record["priority"]].append(record["name"])
    summary = {
        "total": len(records),
        "failed": [r["name"] for r in records if "error" in r],
        "by_category": _count(ok, "category", Category),
        "by_sentiment": _count(ok, "sentiment", Sentiment),
        "by_route": dict(Counter(record["route"] for record in ok)),
        "by_priority": by_priority,
        "self_check": {
            "passed": [r["name"] for r in ok if r["check"]["passed"]],
            "revised": [r["name"] for r in ok if r["check"]["revisions"]],
            "failed": [r["name"] for r in ok if not r["check"]["passed"]],
        },
    }
    if labels:
        summary["accuracy"] = _accuracy(ok, labels)
    return summary


def _count(records: list[dict], field: str, labels: type[StrEnum]) -> dict:
    counts = Counter(record["result"][field] for record in records)
    return {label.value: counts[label.value] for label in labels}


def _accuracy(records: list[dict], labels: dict[str, str]) -> dict:
    checked = [record for record in records if record["name"] in labels]
    mismatches = [
        {
            "name": record["name"],
            "expected": labels[record["name"]],
            "got": record["result"]["category"],
        }
        for record in checked
        if record["result"]["category"] != labels[record["name"]]
    ]
    return {
        "correct": len(checked) - len(mismatches),
        "total": len(checked),
        "mismatches": mismatches,
    }


def format_summary(summary: dict) -> str:
    """Render the run summary as readable console text."""
    ok = summary["total"] - len(summary["failed"])
    lines = [f"=== Summary: {ok} of {summary['total']} texts processed ==="]
    if summary["failed"]:
        lines.append("Failed: " + ", ".join(summary["failed"]))
    lines.append("Categories: " + _non_zero(summary["by_category"]))
    lines.append("Sentiment: " + _non_zero(summary["by_sentiment"]))
    lines.append("Routes: " + _non_zero(summary["by_route"]))
    for level, names in summary["by_priority"].items():
        if names:
            lines.append(f"Priority {level}: " + ", ".join(names))
    check = summary["self_check"]
    parts = [f"{len(check['passed'])} passed"]
    for outcome in ("revised", "failed"):
        if check[outcome]:
            parts.append(f"{outcome}: " + ", ".join(check[outcome]))
    lines.append("Self-check: " + "; ".join(parts))
    accuracy = summary.get("accuracy")
    if accuracy:
        lines.append(
            f"Category accuracy: {accuracy['correct']} of {accuracy['total']}"
        )
        lines.extend(
            f"  {m['name']}: expected {m['expected']}, got {m['got']}"
            for m in accuracy["mismatches"]
        )
    return "\n".join(lines)


def _non_zero(counts: dict[str, int]) -> str:
    parts = [f"{label} {count}" for label, count in counts.items() if count]
    return ", ".join(parts) or "-"
