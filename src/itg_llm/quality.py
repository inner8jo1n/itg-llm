import re

from itg_llm.prompts import (
    KEY_POINTS_COUNT,
    MAX_ANSWER_CHARS,
    MAX_KEY_POINT_WORDS,
    MAX_SUMMARY_CHARS,
)
from itg_llm.schemas import AnalysisResult
from itg_llm.utils import language

# Speaking for the company the author complains about ("we will check")
FIRST_PERSON_PLURAL = re.compile(
    r"\b(мы|нам|нас|наш\w*|we|us|our|ours)\b", re.IGNORECASE
)


def check_quality(result: AnalysisResult, text: str) -> list[str]:
    """Return the format and style problems of one validated result."""
    issues = []
    summary = result.summary
    key_points = result.key_points
    answer = result.final_answer

    if len(summary) > MAX_SUMMARY_CHARS:
        issues.append(
            f"summary is {len(summary)} chars (max {MAX_SUMMARY_CHARS})"
        )
    if len(key_points) != KEY_POINTS_COUNT:
        issues.append(
            f"{len(key_points)} key points (expected {KEY_POINTS_COUNT})"
        )
    for number, point in enumerate(key_points, start=1):
        words = len(point.split())
        if words > MAX_KEY_POINT_WORDS:
            issues.append(
                f"key point {number} is {words} words "
                f"(max {MAX_KEY_POINT_WORDS})"
            )
    if len(answer) > MAX_ANSWER_CHARS:
        issues.append(
            f"final_answer is {len(answer)} chars (max {MAX_ANSWER_CHARS})"
        )

    expected = language(text)
    for field, value in (("summary", summary), ("final_answer", answer)):
        if expected != "unknown" and language(value) != expected:
            issues.append(f"{field} is not in the text's language")

    if FIRST_PERSON_PLURAL.search(answer):
        issues.append("final_answer speaks as 'we'")
    return issues
