import json
from pathlib import Path

from itg_llm.check_replies import REPLIES_DIR, check_reply, main

GOOD = {
    "summary": "s",
    "category": "news",
    "sentiment": "neutral",
    "key_points": ["a", "b"],
    "final_answer": "f",
}


def test_check_reply_ok():
    assert check_reply(json.dumps(GOOD)) == (
        "OK: category=news, sentiment=neutral, 2 key points"
    )


def test_check_reply_error():
    assert check_reply("not json").startswith(
        "ERROR: Model returned invalid JSON"
    )


def test_main_prints_every_reply(tmp_path, capsys):
    (tmp_path / "b_bad.txt").write_text("[]", encoding="utf-8")
    (tmp_path / "a_good.txt").write_text(json.dumps(GOOD), encoding="utf-8")

    assert main(["--replies", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert out.index("=== a_good ===") < out.index("=== b_bad ===")
    assert "OK: category=news" in out
    assert "ERROR: Model returned a JSON list instead of an object" in out


def test_main_without_replies(tmp_path, capsys):
    assert main(["--replies", str(tmp_path)]) == 1
    assert "no .txt replies" in capsys.readouterr().err


def test_bundled_bad_replies_fail_except_the_fenced_one():
    root = Path(__file__).parent.parent
    outcomes = {
        path.stem: check_reply(path.read_text(encoding="utf-8"))
        for path in sorted((root / REPLIES_DIR).glob("*.txt"))
    }

    assert outcomes.pop("06_fenced_ok").startswith("OK")
    assert outcomes
    assert all(outcome.startswith("ERROR") for outcome in outcomes.values())
