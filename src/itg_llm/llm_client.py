import logging
import os

from dotenv import load_dotenv
from openai import OpenAI, omit

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


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
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """Send one system + user message pair and return the reply text.

        With json_mode the API itself makes the model return a JSON object.
        """
        logger.info("Calling %s (json_mode=%s)", self.model, json_mode)
        completion = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"} if json_mode else omit,
        )
        content = completion.choices[0].message.content or ""
        logger.info("Got %d characters from %s", len(content), self.model)
        return content
