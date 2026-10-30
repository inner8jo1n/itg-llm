import argparse
import json
import sys
from collections import deque
from pathlib import Path

from itg_llm.main import format_record, save_json, setup_logging
from itg_llm.pipeline import run
from itg_llm.report import build_summary, format_summary


class RecordedClient:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = deque(replies)

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        if not self.replies:
            raise ValueError("Recorded demo has no more replies for this step")
        return json.dumps(self.replies.popleft(), ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run five recorded scenarios without an API key or network"
    )
    parser.add_argument(
        "--scenarios", type=Path, default=Path("demo/scenarios.json")
    )
    parser.add_argument(
        "--scenario", help="run only the scenario with this name"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/demo.json")
    )
    args = parser.parse_args(argv)
    setup_logging()
    print("Recorded demo: hand-authored replies; no model or API calls.")
    try:
        scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
        if not isinstance(scenarios, list) or not all(
            isinstance(scenario, dict) for scenario in scenarios
        ):
            raise ValueError("Demo scenarios must be a JSON list of objects")
        if args.scenario:
            scenarios = [s for s in scenarios if s["name"] == args.scenario]
        if not scenarios:
            raise ValueError("No matching demo scenarios")
        records = []
        labels = {}
        for scenario in scenarios:
            name = scenario["name"]
            labels[name] = scenario["expected_category"]
            records.extend(
                run(
                    [(name, scenario["text"])],
                    RecordedClient(scenario["replies"]),
                )
            )
        summary = build_summary(records, labels)
        save_json(
            {"mode": "recorded_demo", "summary": summary, "records": records},
            args.output,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for record in records:
        print(format_record(record), end="\n\n")
    print(format_summary(summary))
    print(f"Saved recorded demo to {args.output}")
    return 1 if any("error" in record for record in records) else 0


if __name__ == "__main__":
    sys.exit(main())
