from dataclasses import dataclass

from itg_llm.schemas import Category, Sentiment
from itg_llm.utils import language

KEY_POINTS_COUNT = 3
MAX_SUMMARY_CHARS = 200
MAX_KEY_POINT_WORDS = 15
MAX_ANSWER_CHARS = 350

CATEGORY_HINTS = {
    Category.QUESTION: "the author asks something or wants advice",
    Category.COMPLAINT: "the author reports a problem or is unhappy with "
    "a product, service or company",
    Category.NEWS: "news or information that asks nothing of the reader",
    Category.FEEDBACK: "the author shares an opinion, praise or thanks",
    Category.OTHER: "anything else",
}

LANGUAGE_NAMES = {"ru": "Russian", "en": "English"}


@dataclass(frozen=True)
class PromptVariant:
    name: str
    description: str
    system: str
    template: str

    def build_user_prompt(self, text: str) -> str:
        """Fill the template with the user's text, limits and labels."""
        text = text.strip()
        return self.template.format(
            count=KEY_POINTS_COUNT,
            max_summary=MAX_SUMMARY_CHARS,
            max_point_words=MAX_KEY_POINT_WORDS,
            max_answer=MAX_ANSWER_CHARS,
            categories=_bullets(CATEGORY_HINTS),
            sentiments=", ".join(f'"{s.value}"' for s in Sentiment),
            language=LANGUAGE_NAMES.get(
                language(text), "the same language as the text"
            ),
            text=text,
        )


def _bullets(hints: dict[Category, str]) -> str:
    return "\n".join(
        f'  - "{label.value}": {hint}' for label, hint in hints.items()
    )


V5_STRUCTURED = PromptVariant(
    name="v5_structured",
    description=(
        "The day 2 winner (v4_role_ru) extended to the structured output "
        "schema; the answer language is detected in code and named explicitly"
    ),
    system="""\
You are a neutral assistant that reads a text written by a user and helps \
its author.
You are not the company, shop or organisation mentioned in the text and \
never speak on its behalf: do not write "we", "our" or promise actions \
you cannot take.
Answer with a single JSON object and nothing else: no markdown, no code \
fences, no comments.""",
    template="""\
Return a JSON object with exactly these keys and no others:
- "summary": one sentence, at most {max_summary} characters, stating \
what the text is about;
- "category": exactly one of
{categories}
- "sentiment": the author's attitude, exactly one of {sentiments};
- "key_points": exactly {count} strings, each at most {max_point_words} \
words, the most important facts or ideas;
- "final_answer": at most {max_answer} characters and at most 3 sentences. \
For a question, answer it. For a complaint, suggest concrete next steps. \
For news, explain briefly what it means for a reader. For feedback, \
acknowledge it briefly.

Write "summary", "key_points" and "final_answer" in {language}. \
Write "category" and "sentiment" exactly as listed above, in English.

Example.
Text: "После последнего обновления системы батарея ноутбука садится за два \
часа. Раньше её хватало на весь день. Это вообще нормально?"
JSON:
{{"summary": "После обновления системы батарея ноутбука держит два часа \
вместо целого дня.", "category": "question", "sentiment": "negative", \
"key_points": ["Батарея стала садиться после обновления", "Сейчас её \
хватает примерно на два часа", "До обновления заряда хватало на весь \
день"], "final_answer": "Нет, это ненормально. Проверьте в настройках \
батареи, какие приложения стали расходовать заряд, и при необходимости \
откатите обновление или напишите в поддержку производителя."}}

Text:
\"\"\"
{text}
\"\"\"""",
)

PROMPTS = {variant.name: variant for variant in (V5_STRUCTURED,)}
DEFAULT_PROMPT = V5_STRUCTURED.name
