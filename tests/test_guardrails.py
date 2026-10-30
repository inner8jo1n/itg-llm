import json

import pytest

from itg_llm.pipeline import process_text, run
from itg_llm.schemas import LLMOutputError


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "not JSON",
        '{"final_answer":',
        "{}",
        '{"final_answer": 42}',
        json.dumps({"final_answer": "я" * 451}),
    ],
)
def test_bad_answer_recovers_with_schema_and_original_context(make_llm, bad):
    llm = make_llm(answer=[bad, {"final_answer": "1. Проверьте вход."}])
    chain = process_text("Не могу войти", llm)
    assert chain.result.final_answer == "1. Проверьте вход."
    calls = [c for c in llm.calls if c["step"] == "answer"]
    assert len(calls) == 2
    assert "Fallback:" in calls[1]["user"]
    assert "Не могу войти" in calls[1]["user"]
    assert '"required"' in calls[1]["user"]
    assert "at most 450 characters" in calls[1]["user"]


@pytest.mark.parametrize("step", ["extract", "classify", "fields", "check"])
def test_fallback_covers_each_structured_step(make_llm, step):
    llm = make_llm()
    llm.replies[step].insert(0, "{}")
    assert process_text("Не могу войти", llm).check.passed
    assert llm.steps().count(step) == 2


def test_persistent_bad_output_is_bounded_and_logged(make_llm, caplog):
    llm = make_llm(answer=["broken"])
    with pytest.raises(LLMOutputError, match="invalid JSON"):
        process_text("Не могу войти", llm)
    assert llm.steps().count("answer") == 2
    assert "Answer validation failed (attempt 2/2)" in caplog.text


def test_persistent_long_answer_is_not_silently_truncated(make_llm):
    llm = make_llm(answer=[{"final_answer": "я" * 451}])
    with pytest.raises(LLMOutputError, match="451 chars"):
        process_text("Не могу войти", llm)


def test_raw_reply_size_limit(make_llm):
    llm = make_llm(extract=["x" * 32_001])
    with pytest.raises(LLMOutputError, match="exceeds 32000"):
        process_text("Не могу войти", llm)


def test_batch_continues_after_failed_recovery(make_llm):
    llm = make_llm()
    llm.replies["extract"] = ["broken", "broken", *llm.replies["extract"]]
    records = run([("bad", "Не могу войти"), ("ok", "Не могу войти")], llm)
    assert "invalid JSON" in records[0]["error"]
    assert records[1]["check"]["passed"]


def test_api_failure_isolated_to_one_record(make_llm):
    from itg_llm.llm_client import LLMAPIError

    delegate = make_llm()

    class PartialOutage:
        failed = False

        def complete(self, system, user, json_mode=False):
            if not self.failed:
                self.failed = True
                raise LLMAPIError(
                    "LLM API failed after 3 attempt(s): HTTP 503"
                )
            return delegate.complete(system, user, json_mode)

    records = run(
        [("outage", "Не могу войти"), ("ok", "Не могу войти")], PartialOutage()
    )
    assert "HTTP 503" in records[0]["error"]
    assert records[1]["check"]["passed"]


def test_answer_exactly_at_limit_needs_no_fallback(make_llm):
    llm = make_llm(answer=[{"final_answer": "я" * 450}])
    assert len(process_text("Не могу войти", llm).result.final_answer) == 450
    assert llm.steps().count("answer") == 1
