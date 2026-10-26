import json
import re
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

NonEmptyStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class Category(StrEnum):
    QUESTION = "question"
    COMPLAINT = "complaint"
    NEWS = "news"
    FEEDBACK = "feedback"
    OTHER = "other"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class AnalysisResult(BaseModel):
    """The structured answer the model must return for one text."""

    summary: NonEmptyStr
    category: Category
    sentiment: Sentiment
    key_points: list[NonEmptyStr] = Field(min_length=1)
    final_answer: NonEmptyStr

    @field_validator("category", "sentiment", mode="before")
    @classmethod
    def normalize_label(cls, value: object) -> object:
        # Models often write "Complaint" or " negative"
        if isinstance(value, str):
            return value.strip().lower()
        return value


class LLMOutputError(ValueError):
    """The model's reply cannot be turned into an AnalysisResult."""


CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_response(raw: str) -> AnalysisResult:
    """Parse and validate the model's reply, explaining what is wrong."""
    text = raw.strip()
    fenced = CODE_FENCE.match(text)
    if fenced:
        text = fenced[1]
    if not text:
        raise LLMOutputError("Model returned an empty reply")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMOutputError(
            f"Model returned invalid JSON: {error.msg} "
            f"(line {error.lineno}, column {error.colno})"
        ) from None
    if not isinstance(data, dict):
        raise LLMOutputError(
            f"Model returned a JSON {_json_type(data)} instead of an object"
        )

    try:
        return AnalysisResult.model_validate(data)
    except ValidationError as error:
        problems = [_describe(item) for item in error.errors()]
        raise LLMOutputError(
            "Model JSON does not match the schema: " + "; ".join(problems)
        ) from None


def _describe(error: Any) -> str:
    """Turn one pydantic error into a short human-readable sentence."""
    field = _field_path(error["loc"])
    kind = error["type"]
    value = error.get("input")
    if kind == "missing":
        return f"field '{field}' is missing"
    if kind == "enum":
        allowed = error["ctx"]["expected"].replace("'", "")
        return f"field '{field}' must be one of {allowed}, got {value!r}"
    if kind == "string_too_short":
        return f"field '{field}' must not be empty"
    if kind == "too_short":
        return f"field '{field}' must not be an empty list"
    if kind.endswith("_type"):
        expected = kind.removesuffix("_type")
        return (
            f"field '{field}' must be {_article(expected)} {expected}, "
            f"got {_json_type(value)}"
        )
    return f"field '{field}': {error['msg']}"


def _field_path(loc: tuple) -> str:
    path = str(loc[0])
    for part in loc[1:]:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _json_type(value: object) -> str:
    names = {
        dict: "object",
        list: "list",
        str: "string",
        bool: "boolean",
        int: "number",
        float: "number",
        type(None): "null",
    }
    return names.get(type(value), type(value).__name__)


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"
