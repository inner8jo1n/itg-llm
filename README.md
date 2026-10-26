# itg-llm

LLM pipeline for text processing. For any text the model returns a JSON
object that is validated with a Pydantic schema:

| Field          | Type                                                    |
|----------------|---------------------------------------------------------|
| `summary`      | non-empty string, 1 sentence                            |
| `category`     | `question`, `complaint`, `news`, `feedback` or `other`  |
| `sentiment`    | `positive`, `neutral` or `negative`                     |
| `key_points`   | non-empty list of non-empty strings (3 expected)        |
| `final_answer` | non-empty string, a short helpful answer to the author  |

The code then uses `category` and `sentiment` to set a priority for every
text and to build a summary of the whole run.

## Setup

```bash
uv sync
cp .env.example .env   # then put your key into .env
```

`.env` variables:

| Variable          | Meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `OPENAI_API_KEY`  | API key (required)                                             |
| `OPENAI_BASE_URL` | empty for OpenAI, or the URL of a compatible API (OpenRouter, Ollama, ...) |
| `OPENAI_MODEL`    | model name, `gpt-4o-mini` by default                           |

## Usage

```bash
make run                                   # all texts from examples/
make run-text TEXT="My order is late"      # one text
make run-file FILE=examples/03_news.txt    # one file
make check-replies                         # validate broken replies, no API
```

or directly: `uv run itg-llm --help`.

Results are printed to the console and saved to `results/output.json`
(`--output` changes the path):

```json
{
  "summary": {
    "total": 6,
    "failed": [],
    "by_category": {"question": 2, "complaint": 1, "news": 1, "feedback": 1, "other": 1},
    "by_sentiment": {"positive": 1, "neutral": 2, "negative": 3},
    "by_priority": {"high": ["02_complaint"], "medium": ["01_question"], "low": ["03_news"]}
  },
  "records": [
    {
      "name": "02_complaint",
      "prompt": "v5_structured",
      "text": "...",
      "result": {
        "summary": "...",
        "category": "complaint",
        "sentiment": "negative",
        "key_points": ["...", "...", "..."],
        "final_answer": "..."
      },
      "priority": "high",
      "issues": []
    }
  ]
}
```

### Priority

`priority` is computed in code from the validated fields, not by the model:

| Category  | Sentiment | Priority |
|-----------|-----------|----------|
| complaint | negative  | high     |
| complaint | other     | medium   |
| question  | any       | medium   |
| other     | negative  | medium   |
| other     | other     | low      |

The run summary counts texts per category and sentiment and lists the
texts of every priority, so the most urgent ones are visible at once.

### Issues

`issues` lists soft problems of a valid answer: length limits, not exactly
3 key points, a language other than the text's, or an answer that speaks
as "we" (as if the assistant were the company). Issues are also logged as
warnings.

## Validation and errors

The API is called in JSON mode (`response_format={"type": "json_object"}`)
and the reply is validated with `schemas.parse_response`. A reply that is
wrapped in a ```` ```json ```` fence is unwrapped first; labels are
accepted in any case (`"Complaint"` becomes `complaint`).

A broken reply never crashes the run: the text gets an `"error"` field
instead of `"result"`, the error is logged and the other texts are still
processed (the exit code is 1). The error says exactly what is wrong:

```
Model returned an empty reply
Model returned invalid JSON: Expecting ',' delimiter (line 1, column 179)
Model returned a JSON list instead of an object
Model JSON does not match the schema: field 'sentiment' is missing; field 'final_answer' is missing
Model JSON does not match the schema: field 'key_points' must be a list, got string
Model JSON does not match the schema: field 'category' must be one of question, complaint, news, feedback or other, got 'rant'
```

`make check-replies` runs the saved broken replies from
`examples/bad_replies/` through the same validation without calling the
API.

## Prompts

The prompt is `v5_structured` in `prompts.py`. The allowed `category` and
`sentiment` values are taken from the schema enums, so the prompt and the
schema cannot drift apart. The answer language is detected in code
(`utils.language`) and named in the prompt explicitly ("in Russian" /
"in English").

### Day 2: comparing prompt variants

```bash
make compare           # every variant x every example x 2 runs
make compare RUNS=3
```

All answers go to `results/comparison.json`, the summary table to
`results/comparison.md` (one row per prompt variant): failed answers,
answers with issues, total issues, average summary and answer length and
the answer length spread between runs of the same text (lower is more
stable).

Results of day 2 (`gpt-4.1-nano`, 5 examples x 2 runs,
`results/day02_comparison.*`). The variants v1-v4 used the old answer
format (`response` instead of `final_answer`, no category and sentiment)
and were replaced by `v5_structured` on day 3.

| Prompt | Answers | Failed | With issues | Issues | Avg summary chars | Avg response chars | Response length spread |
|---|---|---|---|---|---|---|---|
| v1_basic | 10 | 2 | 1 | 1 | 140 | 234 | 10 |
| v2_strict | 10 | 0 | 2 | 2 | 108 | 149 | 20 |
| v3_role | 10 | 0 | 6 | 12 | 116 | 192 | 14 |
| v4_role_ru | 10 | 0 | 2 | 4 | 97 | 175 | 9 |

- **v1_basic** (no limits): the longest summaries and responses, goes over
  the summary limit, reacts to news with "good luck with the project".
  The 2 failed answers were API timeouts, not the prompt's fault.
- **v2_strict** (limits only): short and compact, but without a role it
  answers the complaint as the shop in both runs: "Sorry for the delay,
  we will check your order and contact you". That is the worst answer in
  the comparison: the assistant promises actions nobody will take.
- **v3_role** (limits + neutral role + English example): the best content
  (concrete next steps, no "we"), but the model copies the language of the
  example and answers 3 of 4 Russian texts in English in both runs.
- **v4_role_ru** (v3 + Russian example + explicit language rule): all
  Russian texts are answered in Russian, no "we", the shortest summaries
  and the most stable response length (spread 9). The same effect now goes
  the other way: the English text is answered in Russian.

**Chosen: `v4_role_ru`**, the base of `v5_structured`. Takeaways:
- limits in the prompt work: summaries and responses got shorter and
  more even, with no length violations from v2-v4;
- the role in the system prompt decides *whose voice* the answer has,
  limits alone do not fix that;
- a few-shot example is followed more strongly than an explicit
  instruction, including its language, which is why v5 names the answer
  language explicitly.

## Project structure

```
src/itg_llm/
  llm_client.py     # the only module that calls the LLM API
  prompts.py        # prompt (system prompt + user template) and limits
  schemas.py        # Pydantic schema, parsing and readable errors
  quality.py        # soft checks of a valid answer against the limits
  report.py         # priority from category + sentiment, run summary
  utils.py          # language detection
  main.py           # process_text + CLI
  compare.py        # runs prompt variants side by side
  check_replies.py  # validates saved replies without the API
examples/           # sample input texts
examples/bad_replies/  # broken model replies for make check-replies
results/            # pipeline output
tests/              # tests, the LLM is replaced with a fake
```

## Development

```bash
make test        # pytest
make lint        # ruff check
make format      # ruff format
make typecheck   # ty
make check       # all of the above
```
