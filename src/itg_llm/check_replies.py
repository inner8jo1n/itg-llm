import argparse
import sys
from pathlib import Path

from itg_llm.schemas import LLMOutputError, parse_response

REPLIES_DIR = Path("examples/bad_replies")


def check_reply(raw: str) -> str:
    """Validate one saved model reply and describe the outcome."""
    try:
        result = parse_response(raw)
    except LLMOutputError as error:
        return f"ERROR: {error}"
    return (
        f"OK: category={result.category.value}, "
        f"sentiment={result.sentiment.value}, "
        f"{len(result.key_points)} key points"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate saved model replies without calling the API"
    )
    parser.add_argument("--replies", type=Path, default=REPLIES_DIR)
    args = parser.parse_args(argv)

    paths = sorted(args.replies.glob("*.txt"))
    if not paths:
        print(f"ERROR: no .txt replies in {args.replies}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"=== {path.stem} ===")
        print(check_reply(path.read_text(encoding="utf-8")), end="\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
