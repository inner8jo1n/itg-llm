from dataclasses import dataclass

from itg_llm.routing import Route
from itg_llm.schemas import (
    Category,
    Classification,
    Meaning,
    RequestFields,
    Sentiment,
)
from itg_llm.utils import language

KEY_POINTS_COUNT = 3
MAX_SUMMARY_CHARS = 200
MAX_KEY_POINT_WORDS = 15

CATEGORY_HINTS = {
    Category.SUPPORT: "something does not work and the author needs "
    "technical help",
    Category.COMPLAINT: "the author is unhappy with a product, service, "
    "delivery or company and wants it put right",
    Category.SALES: "the author asks about buying: prices, plans, "
    "discounts, trials, terms",
    Category.FEEDBACK: "the author shares an opinion about a product, "
    "service or person (praise, thanks, criticism) and asks for nothing",
    Category.GENERAL_QUESTION: "the author asks a question or wants advice "
    "that is not about a broken product or a purchase",
    Category.NEWS: "news or information that asks nothing of the reader",
    Category.OTHER: "greetings, test messages, unclear requests and "
    "anything else",
}

LANGUAGE_NAMES = {"ru": "Russian", "en": "English"}

NEUTRAL_ROLE = """\
You are a neutral assistant that reads a text written by a user and helps \
its author.
You are not the company, shop or organisation mentioned in the text and \
never speak on its behalf: do not write "we", "our" or promise actions \
you cannot take.
Answer with a single JSON object and nothing else: no markdown, no code \
fences, no comments."""

CHECKER_ROLE = """\
You are a strict reviewer. You compare an answer with the text it \
answers and report every problem you find; you never rewrite the answer.
Answer with a single JSON object and nothing else: no markdown, no code \
fences, no comments."""


def answer_language(text: str) -> str:
    """The language name to put into a prompt for answering this text."""
    return LANGUAGE_NAMES.get(language(text), "the same language as the text")


@dataclass(frozen=True)
class PromptVariant:
    """A prompt for the first step of the chain, extracting the meaning."""

    name: str
    description: str
    system: str
    template: str

    def build_user_prompt(self, text: str) -> str:
        """Fill the template with the user's text and the limits."""
        text = text.strip()
        return self.template.format(
            count=KEY_POINTS_COUNT,
            max_summary=MAX_SUMMARY_CHARS,
            max_point_words=MAX_KEY_POINT_WORDS,
            language=answer_language(text),
            text=text,
        )


V7_EXTRACT = PromptVariant(
    name="v7_extract",
    description=(
        "Step 1 of the multi-step chain: summary, the author's goal, key "
        "points and the concrete facts an answer must not get wrong"
    ),
    system=NEUTRAL_ROLE,
    template="""\
Extract the meaning of the text below. Return a JSON object with exactly \
these keys and no others:
- "summary": one sentence, at most {max_summary} characters, stating \
what the text is about;
- "author_goal": one sentence, what the author wants to get or achieve; \
if they want nothing, say so;
- "key_points": exactly {count} strings, each at most {max_point_words} \
words, the most important ideas;
- "facts": every concrete detail an answer must not get wrong: numbers, \
dates, deadlines, order or ticket numbers, names, products, prices, \
places, things the author already tried. Copy them as written in the \
text. An empty list if there are none.

Write every value in {language}.

Text:
\"\"\"
{text}
\"\"\"""",
)

PROMPTS = {variant.name: variant for variant in (V7_EXTRACT,)}
DEFAULT_PROMPT = V7_EXTRACT.name

CLASSIFY_TEMPLATE = """\
Classify the request below using its extracted meaning and the text.

Summary: {summary}
Author's goal: {author_goal}

Return a JSON object with exactly these keys and no others:
- "category": exactly one of
{categories}
- "intent": what the author wants, 1-4 English words in snake_case, \
for example "request_refund", "ask_price", "fix_login", "say_thanks";
- "sentiment": the author's attitude, exactly one of {sentiments}.

Decide the category by the author's goal, not by the topic of the text.

Text:
\"\"\"
{text}
\"\"\""""

FIELDS_TEMPLATE = """\
Build the structured fields that an answer to this request needs.

Category: {category}, intent: {intent}, sentiment: {sentiment}
Summary: {summary}
Author's goal: {author_goal}
Facts from the text:
{facts}

Return a JSON object with exactly these keys and no others:
- "subject": what the request is about (a product, service, order or \
topic), a few words;
- "requested_action": one sentence, what the author asks for or expects; \
"nothing" if they ask for nothing;
- "details": the facts from the list above that matter for the answer, \
copied as they are; an empty list if none matter;
- "missing_info": what is not in the text but is needed to help the \
author (for example an app version or an order date); an empty list if \
nothing is missing.

Write every value in {language}.

Text:
\"\"\"
{text}
\"\"\""""

ANSWER_TEMPLATE = """\
Write the answer to the author of the text below.

Category: {category}, intent: {intent}, sentiment: {sentiment}
Summary: {summary}
Author's goal: {author_goal}
Subject: {subject}
Requested action: {requested_action}
Details the answer must respect:
{details}
Missing information:
{missing_info}

How to answer this kind of text:
{instructions}

Rules:
- at most {max_chars} characters;
- write in {language};
- use only facts from the text and the details above, never change them;
- if information is missing and it blocks the answer, ask for it.
{problems}
Return a JSON object with exactly one key: {{"final_answer": "..."}}.

Text:
\"\"\"
{text}
\"\"\""""

PROBLEMS_BLOCK = """
A reviewer found problems in your previous answer. Write a new answer \
that fixes all of them:
{problems}
"""

CHECK_TEMPLATE = """\
Check the answer below against the text it answers.

Author's goal: {author_goal}
Facts from the text:
{facts}
Details the answer needs:
{details}

The answer had to follow these instructions:
{instructions}
It had to be at most {max_chars} characters, so a short answer that \
leaves out other facts is expected and is not a problem.

Answer:
\"\"\"
{answer}
\"\"\"

Return a JSON object with exactly these keys and no others:
- "contradictions": every statement of the answer that contradicts the \
text or the facts: wrong numbers, dates or names, invented details, \
promises or statements made on behalf of a company ("we", "our"). \
An empty list if there are none;
- "missing_details": details from the "Details the answer needs" list \
that the answer ignores although the instructions require them. Never \
list other facts, information that is not in the text or details the \
instructions leave out. An empty list if there are none;
- "comment": one sentence, your verdict.

Write every value in {language}.

Text:
\"\"\"
{text}
\"\"\""""


def build_classify_prompt(text: str, meaning: Meaning) -> str:
    return CLASSIFY_TEMPLATE.format(
        summary=meaning.summary,
        author_goal=meaning.author_goal,
        categories=_hints(CATEGORY_HINTS),
        sentiments=", ".join(f'"{s.value}"' for s in Sentiment),
        text=text.strip(),
    )


def build_fields_prompt(
    text: str, meaning: Meaning, classification: Classification
) -> str:
    return FIELDS_TEMPLATE.format(
        **_classification(classification),
        summary=meaning.summary,
        author_goal=meaning.author_goal,
        facts=_bullets(meaning.facts),
        language=answer_language(text),
        text=text.strip(),
    )


def build_answer_prompt(
    text: str,
    meaning: Meaning,
    classification: Classification,
    fields: RequestFields,
    route: Route,
    problems: list[str] | None = None,
) -> str:
    """The answer prompt: every earlier step plus the route's instructions.

    problems are the self-check findings of a previous answer to fix.
    """
    return ANSWER_TEMPLATE.format(
        **_classification(classification),
        summary=meaning.summary,
        author_goal=meaning.author_goal,
        subject=fields.subject,
        requested_action=fields.requested_action,
        details=_bullets(fields.details),
        missing_info=_bullets(fields.missing_info),
        instructions=route.instructions,
        max_chars=route.max_chars,
        language=answer_language(text),
        problems=(
            PROBLEMS_BLOCK.format(problems=_bullets(problems))
            if problems
            else ""
        ),
        text=text.strip(),
    )


def build_check_prompt(
    text: str,
    meaning: Meaning,
    fields: RequestFields,
    route: Route,
    answer: str,
) -> str:
    """The self-check prompt: all facts for contradictions, but only the
    details chosen in step 3 and the route's limits for what is missing."""
    return CHECK_TEMPLATE.format(
        author_goal=meaning.author_goal,
        facts=_bullets(meaning.facts),
        details=_bullets(fields.details),
        instructions=route.instructions,
        max_chars=route.max_chars,
        answer=answer,
        language=answer_language(text),
        text=text.strip(),
    )


def _classification(classification: Classification) -> dict[str, str]:
    return {
        "category": classification.category.value,
        "intent": classification.intent,
        "sentiment": classification.sentiment.value,
    }


def _hints(hints: dict[Category, str]) -> str:
    return "\n".join(
        f'  - "{label.value}": {hint}' for label, hint in hints.items()
    )


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"
