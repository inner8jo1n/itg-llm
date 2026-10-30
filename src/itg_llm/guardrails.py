import json
import logging

from pydantic import BaseModel

from itg_llm.schemas import LLMOutputError, parse_response

logger = logging.getLogger(__name__)
MAX_RAW_CHARS = 32_000


def validate_reply[Model: BaseModel](
    raw: str, schema: type[Model], max_chars: int | None = None
) -> Model:
    if len(raw) > MAX_RAW_CHARS:
        raise LLMOutputError(f"Model reply exceeds {MAX_RAW_CHARS} characters")
    result = parse_response(raw, schema)
    if max_chars is not None:
        answer = getattr(result, "final_answer", "")
        if len(answer) > max_chars:
            raise LLMOutputError(
                f"final_answer is {len(answer)} chars (max {max_chars})"
            )
    return result


def complete_validated[Model: BaseModel](
    client,
    system: str,
    user: str,
    schema: type[Model],
    max_chars: int | None = None,
) -> Model:
    """Try the original prompt, then one schema-guided fallback prompt.

    Keep the original task and context; never fabricate a successful result.
    API failures are handled by the client, not retried as format errors.
    """
    for attempt in range(2):
        raw = client.complete(system, user, json_mode=True)
        try:
            result = validate_reply(raw, schema, max_chars)
        except LLMOutputError as error:
            logger.warning(
                "%s validation failed (attempt %d/2): %s",
                schema.__name__,
                attempt + 1,
                error,
            )
            if attempt:
                raise
            user += (
                "\n\nFallback: the previous response failed validation: "
                f"{error}\nGenerate a fresh JSON object for the original "
                "task. Include every required key and use the exact types "
                "and enum values in this JSON Schema:\n"
                + json.dumps(schema.model_json_schema(), ensure_ascii=False)
            )
            if max_chars is not None:
                user += f"\nfinal_answer: at most {max_chars} characters."
            logger.info("%s: using fallback prompt", schema.__name__)
        else:
            if attempt:
                logger.info(
                    "%s recovered with fallback prompt", schema.__name__
                )
            return result
    raise AssertionError("Unreachable")
