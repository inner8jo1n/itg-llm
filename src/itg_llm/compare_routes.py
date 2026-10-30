import argparse
import sys
from pathlib import Path

from itg_llm.llm_client import LLMAPIError, LLMClient
from itg_llm.main import save_json, setup_logging
from itg_llm.pipeline import (
    Completer,
    answer,
    build_fields,
    classify,
    extract,
)
from itg_llm.routing import ROUTES
from itg_llm.schemas import Category

OUTPUT_PATH = Path("results/routes.json")


def answer_with_every_route(
    text: str,
    client: Completer,
    categories: list[Category],
) -> dict:
    """Run steps 1-3 once, then answer the text with every given route."""
    meaning = extract(text, client)
    classification = classify(text, meaning, client)
    fields = build_fields(text, meaning, classification, client)
    answers = []
    for category in categories:
        route = ROUTES[category]
        try:
            reply = answer(
                text, meaning, classification, fields, route, client
            )
        except (ValueError, LLMAPIError) as error:
            answers.append(
                {
                    "category": category.value,
                    "route": route.name,
                    "error": str(error),
                }
            )
            continue
        answers.append(
            {
                "category": category.value,
                "route": route.name,
                "final_answer": reply.final_answer,
                "chars": len(reply.final_answer),
            }
        )
    return {
        "text": text.strip(),
        "meaning": meaning.model_dump(mode="json"),
        "classification": classification.model_dump(mode="json"),
        "fields": fields.model_dump(mode="json"),
        "answers": answers,
    }


def format_routes(data: dict) -> str:
    classification = data["classification"]
    lines = [
        f"Classified as {classification['category']} "
        f"(intent {classification['intent']})",
        "",
    ]
    for item in data["answers"]:
        marker = (
            "*" if item.get("category") == classification["category"] else " "
        )
        if "error" in item:
            lines.append(f"{marker} [{item['route']}] ERROR: {item['error']}")
        else:
            lines.append(
                f"{marker} [{item['route']}, {item['chars']} chars] "
                f"{item['final_answer']}"
            )
        lines.append("")
    lines.append("* the route chosen by the classification")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Answer one text with every route to compare them"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="a text to process")
    source.add_argument("--file", type=Path, help="a .txt file to process")
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=[category.value for category in Category],
        default=[category.value for category in Category],
        help="categories whose routes to try (default: all)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    setup_logging()
    try:
        text = (
            args.text
            if args.text is not None
            else args.file.read_text(encoding="utf-8")
        )
        client = LLMClient()
        categories = [Category(value) for value in args.routes]
        data = answer_with_every_route(text, client, categories)
        save_json(data, args.output)
    except (RuntimeError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print()
    print(format_routes(data))
    print(f"\nSaved to {args.output}")
    return 1 if any("error" in item for item in data["answers"]) else 0


if __name__ == "__main__":
    sys.exit(main())
