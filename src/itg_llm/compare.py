import argparse
import logging
import sys
from pathlib import Path
from statistics import mean

from itg_llm.llm_client import LLMClient
from itg_llm.main import (
    EXAMPLES_DIR,
    Completer,
    load_inputs,
    run,
    save_json,
)
from itg_llm.prompts import PROMPTS, PromptVariant

OUTPUT_PATH = Path("results/comparison.json")

COLUMNS = [
    ("prompt", "Prompt"),
    ("answers", "Answers"),
    ("failed", "Failed"),
    ("with_issues", "With issues"),
    ("issues", "Issues"),
    ("avg_summary_chars", "Avg summary chars"),
    ("avg_answer_chars", "Avg answer chars"),
    ("answer_spread", "Answer length spread"),
]


def compare(
    inputs: list[tuple[str, str]],
    client: Completer,
    prompts: list[PromptVariant],
    runs: int,
) -> list[dict]:
    """Run every prompt on every input `runs` times."""
    records = []
    for prompt in prompts:
        for number in range(1, runs + 1):
            for record in run(inputs, client, prompt):
                records.append({**record, "run": number})
    return records


def summarize(records: list[dict]) -> list[dict]:
    """Per-prompt statistics, in the order the prompts first appear."""
    rows = []
    for name in dict.fromkeys(record["prompt"] for record in records):
        own = [record for record in records if record["prompt"] == name]
        ok = [record for record in own if "result" in record]
        rows.append(
            {
                "prompt": name,
                "answers": len(own),
                "failed": len(own) - len(ok),
                "with_issues": sum(1 for record in ok if record["issues"]),
                "issues": sum(len(record["issues"]) for record in ok),
                "avg_summary_chars": _avg_len(ok, "summary"),
                "avg_answer_chars": _avg_len(ok, "final_answer"),
                "answer_spread": _answer_spread(ok),
            }
        )
    return rows


def _avg_len(records: list[dict], field: str) -> int:
    if not records:
        return 0
    return round(mean(len(record["result"][field]) for record in records))


def _answer_spread(records: list[dict]) -> int:
    """Mean gap between the longest and shortest answer to one text.

    Shows how stable the answers are from run to run: 0 means every run
    gave an answer of the same length.
    """
    lengths: dict[str, list[int]] = {}
    for record in records:
        answer = record["result"]["final_answer"]
        lengths.setdefault(record["name"], []).append(len(answer))
    gaps = [max(group) - min(group) for group in lengths.values()]
    return round(mean(gaps)) if gaps else 0


def to_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(title for _, title in COLUMNS) + " |"
    divider = "|" + "---|" * len(COLUMNS)
    lines = [header, divider]
    for row in rows:
        cells = (str(row[key]) for key, _ in COLUMNS)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare prompt variants on the same example texts"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=PROMPTS,
        default=list(PROMPTS),
        help="prompt variants to compare (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="how many times to run each prompt on each text",
    )
    parser.add_argument("--examples", type=Path, default=EXAMPLES_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="JSON with every answer; the table goes next to it as .md",
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        client = LLMClient()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    inputs = load_inputs(None, None, args.examples)
    prompts = [PROMPTS[name] for name in args.prompts]
    records = compare(inputs, client, prompts, args.runs)
    save_json(records, args.output)

    table = to_markdown(summarize(records))
    table_path = args.output.with_suffix(".md")
    table_path.write_text(table, encoding="utf-8")
    print()
    print(table)
    print(f"Saved answers to {args.output} and the table to {table_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
