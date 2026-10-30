import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Protocol

from pydantic import BaseModel

from itg_llm.guardrails import complete_validated
from itg_llm.prompts import (
    CHECKER_ROLE,
    DEFAULT_PROMPT,
    NEUTRAL_ROLE,
    PROMPTS,
    PromptVariant,
    build_answer_prompt,
    build_check_prompt,
    build_classify_prompt,
    build_fields_prompt,
)
from itg_llm.quality import check_quality
from itg_llm.report import priority
from itg_llm.routing import Route, route_for
from itg_llm.schemas import (
    AnalysisResult,
    Answer,
    Category,
    Classification,
    Meaning,
    RequestFields,
    SelfCheck,
)

logger = logging.getLogger(__name__)

MAX_REVISIONS = 1
STEPS = 5


class Completer(Protocol):
    """Anything that can answer a system + user prompt (LLMClient, fakes)."""

    def complete(
        self, system: str, user: str, json_mode: bool = False
    ) -> str: ...


def extract(
    text: str,
    client: Completer,
    prompt: PromptVariant = PROMPTS[DEFAULT_PROMPT],
) -> Meaning:
    """Step 1: summary, the author's goal, key points and facts."""
    return complete_validated(
        client, prompt.system, prompt.build_user_prompt(text), Meaning
    )


def classify(text: str, meaning: Meaning, client: Completer) -> Classification:
    """Step 2: category, intent and sentiment, decided by the goal."""
    return complete_validated(
        client,
        NEUTRAL_ROLE,
        build_classify_prompt(text, meaning),
        Classification,
    )


def build_fields(
    text: str,
    meaning: Meaning,
    classification: Classification,
    client: Completer,
) -> RequestFields:
    """Step 3: subject, requested action, details and missing info."""
    return complete_validated(
        client,
        NEUTRAL_ROLE,
        build_fields_prompt(text, meaning, classification),
        RequestFields,
    )


def answer(
    text: str,
    meaning: Meaning,
    classification: Classification,
    fields: RequestFields,
    route: Route,
    client: Completer,
    problems: list[str] | None = None,
) -> Answer:
    """Step 4: the answer, written with the route's instructions."""
    prompt = build_answer_prompt(
        text, meaning, classification, fields, route, problems
    )
    return complete_validated(
        client, NEUTRAL_ROLE, prompt, Answer, max_chars=route.max_chars
    )


def self_check(
    text: str,
    meaning: Meaning,
    fields: RequestFields,
    route: Route,
    final_answer: str,
    client: Completer,
) -> SelfCheck:
    """Step 5: contradictions with the text and lost details."""
    return complete_validated(
        client,
        CHECKER_ROLE,
        build_check_prompt(text, meaning, fields, route, final_answer),
        SelfCheck,
    )


@dataclass
class ChainResult:
    result: AnalysisResult
    route: Route
    check: SelfCheck
    revisions: int
    steps: list[dict] = field(default_factory=list)


@dataclass
class _Trace:
    """Times every step, logs its outcome and keeps it for the output."""

    name: str
    steps: list[dict] = field(default_factory=list)

    def run[Out: BaseModel](
        self,
        step: str,
        number: int,
        call: Callable[[], Out],
        describe: Callable[[Out], str],
        revision: int = 0,
    ) -> Out:
        started = time.perf_counter()
        output = call()
        ms = round((time.perf_counter() - started) * 1000)
        label = f"{number}/{STEPS} {step}"
        if revision:
            label += f" (revision {revision})"
        logger.info(
            "%s | %s: %s [%d ms]", self.name, label, describe(output), ms
        )
        self.steps.append(
            {
                "step": step,
                "revision": revision,
                "ms": ms,
                "output": output.model_dump(mode="json"),
            }
        )
        return output

    def decision(self, step: str, message: str, output: dict) -> None:
        logger.info("%s | %s: %s", self.name, step, message)
        self.steps.append(
            {"step": step, "revision": 0, "ms": 0, "output": output}
        )


def _describe_check(check: SelfCheck) -> str:
    verdict = (
        "passed"
        if check.passed
        else f"failed, {len(check.contradictions)} contradictions"
    )
    if check.missing_details:
        verdict += f", {len(check.missing_details)} missing details"
    return verdict


def _problems(check: SelfCheck) -> list[str]:
    return [
        *(f"contradiction: {item}" for item in check.contradictions),
        *(f"missing detail: {item}" for item in check.missing_details),
    ]


def process_text(
    text: str,
    client: Completer,
    prompt: PromptVariant = PROMPTS[DEFAULT_PROMPT],
    route_category: Category | None = None,
    name: str = "text",
    max_revisions: int = MAX_REVISIONS,
) -> ChainResult:
    """Run the text through the whole chain, every step feeding the next.

    extract -> classify -> route (code) -> build fields -> answer ->
    self-check; a contradiction found by the self-check sends the answer
    back to step 4 with the problems, at most max_revisions times; missing
    details alone do not. route_category forces a route instead of the
    classified category.
    """
    if not text.strip():
        raise ValueError("Text is empty")
    trace = _Trace(name)
    logger.info("%s | start: %d characters", name, len(text))

    meaning = trace.run(
        "extract",
        1,
        partial(extract, text, client, prompt),
        lambda m: f"goal: {m.author_goal} | {len(m.facts)} facts",
    )
    classification = trace.run(
        "classify",
        2,
        partial(classify, text, meaning, client),
        lambda c: f"{c.category.value} / {c.intent} / {c.sentiment.value}",
    )
    category = route_category or classification.category
    route = route_for(category)
    forced = route_category is not None
    trace.decision(
        "route",
        f"{category.value} -> {route.name}"
        + (" (forced)" if forced else " (by category)"),
        {"category": category.value, "route": route.name, "forced": forced},
    )
    fields = trace.run(
        "fields",
        3,
        partial(build_fields, text, meaning, classification, client),
        lambda f: (
            f"{f.subject} | needs: {f.requested_action} | "
            f"missing: {len(f.missing_info)}"
        ),
    )

    problems: list[str] = []
    revision = 0
    while True:
        reply = trace.run(
            "answer",
            4,
            partial(
                answer,
                text,
                meaning,
                classification,
                fields,
                route,
                client,
                problems,
            ),
            lambda a: f"{len(a.final_answer)} chars",
            revision,
        )
        check = trace.run(
            "check",
            5,
            partial(
                self_check,
                text,
                meaning,
                fields,
                route,
                reply.final_answer,
                client,
            ),
            _describe_check,
            revision,
        )
        if check.passed or revision >= max_revisions:
            break
        problems = _problems(check)
        revision += 1
        logger.info(
            "%s | self-check failed, revising the answer (%d/%d)",
            name,
            revision,
            max_revisions,
        )

    if not check.passed:
        logger.warning("%s | self-check still fails after revisions", name)
    result = AnalysisResult(
        **meaning.model_dump(),
        **classification.model_dump(),
        **fields.model_dump(),
        final_answer=reply.final_answer,
    )
    return ChainResult(result, route, check, revision, trace.steps)


def run(
    inputs: list[tuple[str, str]],
    client: Completer,
    prompt: PromptVariant = PROMPTS[DEFAULT_PROMPT],
    route_category: Category | None = None,
) -> list[dict]:
    """Process every input; a failed text is recorded, not fatal."""
    records = []
    for name, text in inputs:
        record: dict = {
            "name": name,
            "prompt": prompt.name,
            "text": text.strip(),
        }
        try:
            chain = process_text(
                text, client, prompt, route_category, name=name
            )
        except Exception as error:
            logger.error("Failed to process %s: %s", name, error)
            record["error"] = str(error)
        else:
            record["result"] = chain.result.model_dump(mode="json")
            record["route"] = chain.route.name
            record["priority"] = priority(chain.result).value
            record["check"] = {
                "passed": chain.check.passed,
                "revisions": chain.revisions,
                **chain.check.model_dump(mode="json"),
            }
            record["issues"] = [
                *check_quality(chain.result, text, chain.route),
                *(
                    f"self-check: missing detail: {detail}"
                    for detail in chain.check.missing_details
                ),
            ]
            record["steps"] = chain.steps
            for issue in record["issues"]:
                logger.warning("%s: %s", name, issue)
        records.append(record)
    return records
