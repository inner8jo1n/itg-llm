import logging
import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, omit

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 2


class LLMAPIError(RuntimeError):
    """Readable API failure without request bodies or credentials."""


class LLMClient:
    """Thin wrapper over an OpenAI-compatible chat completions API.

    The key, base URL and model come from the environment (or .env):
    OPENAI_API_KEY, OPENAI_BASE_URL (optional), OPENAI_MODEL (optional).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        load_dotenv()
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set: copy .env.example to .env "
                "and put your key there"
            )
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, max_retries=0, timeout=30.0
        )

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """Send one system + user message pair and return the reply text.

        With json_mode the API itself makes the model return a JSON object.
        """
        logger.info("Calling %s (json_mode=%s)", self.model, json_mode)
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._complete(system, user, json_mode)
            except (APIConnectionError, APIStatusError) as error:
                status = getattr(error, "status_code", None)
                temporary = (
                    isinstance(error, APIConnectionError)
                    or status in (408, 409, 429)
                    or (status is not None and status >= 500)
                )
                description = (
                    f"HTTP {status}" if status else type(error).__name__
                )
                if not temporary or attempt == MAX_RETRIES:
                    logger.error("LLM API failed: %s", description)
                    raise LLMAPIError(
                        f"LLM API failed after {attempt + 1} attempt(s): "
                        f"{description}"
                    ) from None
                delay = 0.5 * 2**attempt
                logger.warning(
                    "LLM API temporary error %s; retry %d/%d in %.1fs",
                    description,
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
        raise AssertionError("Unreachable")

    def _complete(self, system: str, user: str, json_mode: bool) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"} if json_mode else omit,
        )
        content = (
            completion.choices[0].message.content or ""
            if completion.choices
            else ""
        )
        logger.info("Got %d characters from %s", len(content), self.model)
        return content
