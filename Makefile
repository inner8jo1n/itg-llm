.PHONY: run run-text run-file compare check-replies test lint format fix typecheck check

# Every target takes an optional prompt variant: make run PROMPT=v3_role
PROMPT_ARG = $(if $(PROMPT),--prompt $(PROMPT))

# Process every text in examples/ and save results/output.json
run:
	uv run itg-llm $(PROMPT_ARG)

# make run-text TEXT="My order is late, what should I do?"
run-text:
	uv run itg-llm --text "$(TEXT)" $(PROMPT_ARG)

# make run-file FILE=examples/03_news.txt
run-file:
	uv run itg-llm --file "$(FILE)" $(PROMPT_ARG)

# Every prompt variant on every example, RUNS times each (default 2):
# results/comparison.json + .md table
RUNS ?= 2
compare:
	uv run itg-llm-compare --runs $(RUNS)

# Validate the broken model replies in examples/bad_replies, no API calls
check-replies:
	uv run itg-llm-check-replies

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run ty check

check:
	@./scripts/check.sh
