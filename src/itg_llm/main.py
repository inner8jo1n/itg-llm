import argparse
import json
import logging
import sys
from pathlib import Path

from itg_llm.llm_client import LLMClient
from itg_llm.pipeline import run
from itg_llm.prompts import DEFAULT_PROMPT, PROMPTS
from itg_llm.report import build_summary, format_summary
from itg_llm.schemas import Category

EXAMPLES_DIR = Path("examples")
LABELS_FILE = "labels.json"
OUTPUT_PATH = Path("results/output.json")
STEP_LABEL_WIDTH = 14


def load_inputs(
    text: str | None, file: Path | None, examples_dir: Path
) -> list[tuple[str, str]]:
    """Return (name, text) pairs: the given text, file or all examples."""
    if text is not None:
        return [("cli", text)]
    if file is not None:
        return [(file.stem, file.read_text(encoding="utf-8"))]
    paths = sorted(examples_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt examples in {examples_dir}")
    return [(path.stem, path.read_text(encoding="utf-8")) for path in paths]


def load_labels(examples_dir: Path) -> dict[str, str]:
    """Expected category of every example, if the folder has labels.json."""
    path = examples_dir / LABELS_FILE
    if not path.exists():
        return {}
    labels = json.loads(path.read_text(encoding="utf-8"))
    categories = {category.value for category in Category}
    if not isinstance(labels, dict) or not all(
        isinstance(name, str)
        and isinstance(category, str)
        and category in categories
        for name, category in labels.items()
    ):
        raise ValueError(
            f"Invalid labels in {path}: expected name -> category"
        )
    return labels


def format_record(record: dict) -> str:
    """Render one record as console text: every step of the chain in turn."""
    lines = [f"=== {record['name']} ({record['prompt']}) ==="]
    if "error" in record:
        lines.append(f"ERROR: {record['error']}")
        return "\n".join(lines)
    for step in record["steps"]:
        lines.extend(_format_step(step, record))
    check = record["check"]
    verdict = "passed" if check["passed"] else "FAILED"
    if check["revisions"]:
        verdict += f" after {check['revisions']} revision(s)"
    lines.append(f"Final answer: {record['result']['final_answer']}")
    lines.append(f"Self-check: {verdict}")
    if record.get("issues"):
        lines.append("Issues: " + "; ".join(record["issues"]))
    return "\n".join(lines)


def _format_step(step: dict, record: dict) -> list[str]:
    out = step["output"]
    name = step["step"]
    if step["revision"]:
        name += f" r{step['revision']}"
    label = f"[{name}]".ljust(STEP_LABEL_WIDTH)
    pad = " " * STEP_LABEL_WIDTH
    if step["step"] == "extract":
        return [
            f"{label}Summary: {out['summary']}",
            f"{pad}Goal: {out['author_goal']}",
            f"{pad}Key points: {_join(out['key_points'])}",
            f"{pad}Facts: {_join(out['facts'])}",
        ]
    if step["step"] == "classify":
        return [
            f"{label}{out['category']} / {out['intent']} / "
            f"{out['sentiment']} -> priority {record['priority']}"
        ]
    if step["step"] == "route":
        how = "forced" if out["forced"] else "by category"
        return [f"{label}{out['category']} -> {out['route']} ({how})"]
    if step["step"] == "fields":
        return [
            f"{label}Subject: {out['subject']}",
            f"{pad}Requested: {out['requested_action']}",
            f"{pad}Details: {_join(out['details'])}",
            f"{pad}Missing: {_join(out['missing_info'])}",
        ]
    if step["step"] == "answer":
        return [f"{label}{out['final_answer']}"]
    missing = [f"missing: {item}" for item in out["missing_details"]]
    if out["contradictions"]:
        contradictions = [
            f"contradiction: {item}" for item in out["contradictions"]
        ]
        return [f"{label}{_join(contradictions + missing)}"]
    if missing:
        return [f"{label}passed; {_join(missing)}"]
    return [f"{label}passed: {out['comment']}"]


def _join(items: list[str]) -> str:
    return "; ".join(items) or "-"


def save_json(data: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run texts through the chain: extract, classify, route, "
        "build fields, answer, self-check"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="a text to process")
    source.add_argument("--file", type=Path, help="a .txt file to process")
    parser.add_argument(
        "--examples",
        type=Path,
        default=EXAMPLES_DIR,
        help="folder with .txt examples (used when no text or file given)",
    )
    parser.add_argument(
        "--prompt",
        choices=PROMPTS,
        default=DEFAULT_PROMPT,
        help=f"extraction prompt (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "--route",
        choices=[category.value for category in Category],
        help="answer with this category's route instead of the classified one",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH, help="where to save JSON"
    )
    args = parser.parse_args(argv)

    setup_logging()
    using_examples = args.text is None and args.file is None
    try:
        inputs = load_inputs(args.text, args.file, args.examples)
        labels = load_labels(args.examples) if using_examples else {}
        client = LLMClient()
    except (RuntimeError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    route = Category(args.route) if args.route else None
    records = run(inputs, client, PROMPTS[args.prompt], route)
    summary = build_summary(records, labels)
    print()
    for record in records:
        print(format_record(record), end="\n\n")
    print(format_summary(summary), end="\n\n")
    try:
        save_json({"summary": summary, "records": records}, args.output)
    except OSError as error:
        print(f"ERROR: Cannot save results: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(records)} result(s) to {args.output}")
    return 1 if any("error" in record for record in records) else 0


if __name__ == "__main__":
    sys.exit(main())
