.PHONY: demo run run-text run-file compare compare-routes check-replies test lint format fix typecheck check

# The run targets take an optional prompt variant and a forced route:
# make run PROMPT=v7_extract ROUTE=sales
RUN_ARGS = $(if $(PROMPT),--prompt $(PROMPT)) $(if $(ROUTE),--route $(ROUTE))

# Process every text in examples/ and save results/output.json
run:
	uv run itg-llm $(RUN_ARGS)

# make run-text TEXT="My order is late, what should I do?"
run-text:
	uv run itg-llm --text "$(TEXT)" $(RUN_ARGS)

# make run-file FILE=examples/03_news.txt
run-file:
	uv run itg-llm --file "$(FILE)" $(RUN_ARGS)

# Every prompt variant on every example, RUNS times each (default 2):
# results/comparison.json + .md table
RUNS ?= 2
compare:
	uv run itg-llm-compare --runs $(RUNS)

# One text answered by every route: make compare-routes FILE=examples/02_complaint.txt
compare-routes:
	uv run itg-llm-compare-routes $(if $(TEXT),--text "$(TEXT)",--file "$(FILE)")

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

demo:
	uv run itg-llm-demo
