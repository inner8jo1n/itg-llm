import json
from pathlib import Path

import pytest

from itg_llm.demo import RecordedClient, main

ROOT = Path(__file__).resolve().parents[1]


def test_five_demo_scenarios_run_full_pipeline_without_api(tmp_path, capsys):
    output = tmp_path / "demo.json"
    assert (
        main(
            [
                "--scenarios",
                str(ROOT / "demo/scenarios.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    data = json.loads(output.read_text())
    assert data["mode"] == "recorded_demo"
    assert len(data["records"]) == 5
    assert data["summary"]["accuracy"] == {
        "correct": 5,
        "total": 5,
        "mismatches": [],
    }
    for record in data["records"]:
        assert record["check"]["passed"]
        assert not record["issues"]
        assert [s["step"] for s in record["steps"]] == [
            "extract",
            "classify",
            "route",
            "fields",
            "answer",
            "check",
        ]
    assert "hand-authored replies" in capsys.readouterr().out


def test_demo_can_select_one_scenario(tmp_path):
    output = tmp_path / "one.json"
    assert (
        main(
            [
                "--scenarios",
                str(ROOT / "demo/scenarios.json"),
                "--scenario",
                "sales",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert [r["name"] for r in json.loads(output.read_text())["records"]] == [
        "sales",
    ]


@pytest.mark.parametrize("content", ["broken", "{}", "[{}]"])
def test_bad_demo_manifest_has_readable_error(tmp_path, capsys, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    assert main(["--scenarios", str(path)]) == 1
    assert "ERROR:" in capsys.readouterr().err


def test_unknown_demo_scenario(tmp_path, capsys):
    assert (
        main(
            [
                "--scenarios",
                str(ROOT / "demo/scenarios.json"),
                "--scenario",
                "unknown",
            ]
        )
        == 1
    )
    assert "No matching demo scenarios" in capsys.readouterr().err


def test_replay_exhaustion_is_explicit():
    with pytest.raises(ValueError, match="no more replies"):
        RecordedClient([]).complete("sys", "user")
