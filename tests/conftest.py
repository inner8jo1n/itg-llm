import json

import pytest

MEANING = {
    "summary": "Пользователь не может войти в личный кабинет.",
    "author_goal": "Войти в личный кабинет.",
    "key_points": ["Сессия истекает", "Сброс пароля не помог", "Два браузера"],
    "facts": ["Ошибка «Сессия истекла»", "Пароль сбрасывал два раза"],
}
CLASSIFICATION = {
    "category": "support",
    "intent": "fix_login",
    "sentiment": "negative",
}
FIELDS = {
    "subject": "Личный кабинет",
    "requested_action": "Помочь войти в кабинет",
    "details": ["Ошибка «Сессия истекла»"],
    "missing_info": ["Версия браузера"],
}
ANSWER = {
    "final_answer": "1. Очистите cookies. 2. Войдите в режиме инкогнито."
}
CHECK_OK = {"contradictions": [], "missing_details": [], "comment": "Верно."}
CHECK_FAILED = {
    "contradictions": ["Ответ говорит от лица компании"],
    "missing_details": ["Пароль сбрасывали дважды"],
    "comment": "Нужно исправить.",
}

# The first line of every step's prompt, see prompts.py
STEP_PREFIXES = {
    "classify": "Classify the request",
    "fields": "Build the structured fields",
    "answer": "Write the answer",
    "check": "Check the answer",
}


class FakeLLM:
    """Plays the model in every step of the chain.

    Each step has its own list of replies used in order; the last reply of
    a list is repeated when it runs out. A prompt that matches no step
    prefix is the extraction step, so test prompt variants can be plain.
    """

    def __init__(
        self,
        extract=(MEANING,),
        classify=(CLASSIFICATION,),
        fields=(FIELDS,),
        answer=(ANSWER,),
        check=(CHECK_OK,),
    ):
        self.replies = {
            step: [_raw(reply) for reply in replies]
            for step, replies in {
                "extract": extract,
                "classify": classify,
                "fields": fields,
                "answer": answer,
                "check": check,
            }.items()
        }
        self.calls = []

    def complete(self, system, user, json_mode=False):
        step = next(
            (
                s
                for s, prefix in STEP_PREFIXES.items()
                if user.startswith(prefix)
            ),
            "extract",
        )
        self.calls.append(
            {
                "step": step,
                "system": system,
                "user": user,
                "json_mode": json_mode,
            }
        )
        replies = self.replies[step]
        return replies.pop(0) if len(replies) > 1 else replies[0]

    def steps(self):
        return [call["step"] for call in self.calls]


def _raw(reply):
    return reply if isinstance(reply, str) else json.dumps(reply)


@pytest.fixture
def make_llm():
    return FakeLLM


@pytest.fixture
def meaning():
    return dict(MEANING)


@pytest.fixture
def classification():
    return dict(CLASSIFICATION)


@pytest.fixture
def fields():
    return dict(FIELDS)


@pytest.fixture
def check_failed():
    return dict(CHECK_FAILED)


@pytest.fixture
def full_result():
    return {**MEANING, **CLASSIFICATION, **FIELDS, **ANSWER}
