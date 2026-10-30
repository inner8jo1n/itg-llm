from dataclasses import dataclass

from itg_llm.schemas import Category


@dataclass(frozen=True)
class Route:
    """How to answer one category: instructions and format limits."""

    name: str
    instructions: str
    max_chars: int
    numbered_steps: bool = False


ROUTES = {
    Category.SUPPORT: Route(
        name="technical_steps",
        instructions=(
            "Give a structured technical answer: a numbered list of 2-4 "
            "concrete steps to diagnose or fix the problem, one short line "
            "each, most likely fix first. If key details are missing "
            "(device, app version, exact error text), end with one question "
            "asking for them."
        ),
        max_chars=450,
        numbered_steps=True,
    ),
    Category.COMPLAINT: Route(
        name="empathetic",
        instructions=(
            "Start with one sentence that acknowledges the author's "
            "frustration, calmly and without blaming anyone. Then give 2-3 "
            "concrete steps to get the problem resolved and say how to "
            "escalate if nothing changes. Respectful, careful tone, no "
            "promises on behalf of the company."
        ),
        max_chars=450,
    ),
    Category.SALES: Route(
        name="short_sales",
        instructions=(
            "Write a short persuasive answer in 1-2 sentences: name the main "
            "benefit for the author's case and one clear next step (request "
            "a quote, start a trial, talk to sales). Do not invent prices, "
            "discounts or terms that are not in the text."
        ),
        max_chars=220,
    ),
    Category.FEEDBACK: Route(
        name="thanks",
        instructions=(
            "Thank the author warmly in 1-2 sentences and mention the "
            "specific thing they praised or criticised."
        ),
        max_chars=200,
    ),
    Category.GENERAL_QUESTION: Route(
        name="direct_answer",
        instructions=(
            "Answer the question directly in the first sentence, then add "
            "up to 2 sentences of practical advice."
        ),
        max_chars=350,
    ),
    Category.NEWS: Route(
        name="explain_news",
        instructions=(
            "In 1-2 sentences explain what this news means for an ordinary "
            "reader and what to watch for. No congratulations or wishes."
        ),
        max_chars=250,
    ),
    Category.OTHER: Route(
        name="clarify",
        instructions=(
            "The request is unclear: reply in one friendly sentence and ask "
            "what exactly the author needs."
        ),
        max_chars=150,
    ),
}


def route_for(category: Category) -> Route:
    return ROUTES[category]
