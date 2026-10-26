import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Protocol

from itg_llm.llm_client import LLMClient
from itg_llm.prompts import DEFAULT_PROMPT, PROMPTS, PromptVariant
from itg_llm.quality import check_quality
from itg_llm.report import build_summary, format_summary, priority
from itg_llm.schemas import AnalysisResult, parse_response

logger = logging.getLogger(__name__)

EXAMPLES_DIR = Path("examples")
OUTPUT_PATH = Path("results/output.json")


class Completer(Protocol):
    """Anything that can answer a system + user prompt (LLMClient, fakes)."""

    def complete(
        self, system: str, user: str, json_mode: bool = False
    ) -> str: ...


def process_text(
    text: str,
    client: Completer,
    prompt: PromptVariant = PROMPTS[DEFAULT_PROMPT],
) -> AnalysisResult:
    """Run one text through the model and return the validated result."""
    if not text.strip():
        raise ValueError("Text is empty")
    logger.info(
        "Processing text of %d characters with %s", len(text), prompt.name
    )
    raw = client.complete(
        prompt.system, prompt.build_user_prompt(text), json_mode=True
    )
    return parse_response(raw)


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


def run(
    inputs: list[tuple[str, str]],
    client: Completer,
    prompt: PromptVariant = PROMPTS[DEFAULT_PROMPT],
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
            result = process_text(text, client, prompt)
        except Exception as error:
            logger.error("Failed to process %s: %s", name, error)
            record["error"] = str(error)
        else:
            record["result"] = result.model_dump(mode="json")
            record["priority"] = priority(result).value
            record["issues"] = check_quality(result, text)
            for issue in record["issues"]:
                logger.warning("%s: %s", name, issue)
        records.append(record)
    return records


def format_record(record: dict) -> str:
    """Render one record as readable console text."""
    lines = [f"=== {record['name']} ({record['prompt']}) ==="]
    if "error" in record:
        lines.append(f"ERROR: {record['error']}")
        return "\n".join(lines)
    result = record["result"]
    lines.append(
        f"Category: {result['category']} | "
        f"Sentiment: {result['sentiment']} | "
        f"Priority: {record['priority']}"
    )
    lines.append(f"Summary: {result['summary']}")
    lines.append("Key points:")
    lines.extend(
        f"  {number}. {point}"
        for number, point in enumerate(result["key_points"], start=1)
    )
    lines.append(f"Final answer: {result['final_answer']}")
    if record.get("issues"):
        lines.append("Issues: " + "; ".join(record["issues"]))
    return "\n".join(lines)


def save_json(data: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise texts, extract key points and respond"
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
        help=f"prompt variant (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH, help="where to save JSON"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        client = LLMClient()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    inputs = load_inputs(args.text, args.file, args.examples)
    records = run(inputs, client, PROMPTS[args.prompt])
    summary = build_summary(records)
    print()
    for record in records:
        print(format_record(record), end="\n\n")
    print(format_summary(summary), end="\n\n")
    save_json({"summary": summary, "records": records}, args.output)
    print(f"Saved {len(records)} result(s) to {args.output}")
    return 1 if any("error" in record for record in records) else 0


if __name__ == "__main__":
    sys.exit(main())
