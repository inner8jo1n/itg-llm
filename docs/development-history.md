# itg-llm

LLM pipeline for text processing: a multi-step chain where every step
uses the results of the steps before it.

| # | Step | Who | Input | Output (schema) |
|---|------|-----|-------|-----------------|
| 1 | extract | LLM | text | `Meaning`: `summary`, `author_goal`, `key_points`, `facts` |
| 2 | classify | LLM | text + summary + goal | `Classification`: `category`, `intent`, `sentiment` |
| - | route | code | category | `Route`: answer instructions + length limit (`routing.py`) |
| 3 | build fields | LLM | text + meaning + classification | `RequestFields`: `subject`, `requested_action`, `details`, `missing_info` |
| 4 | answer | LLM | everything above + route instructions | `Answer`: `final_answer` |
| 5 | self-check | LLM | text + facts + goal + details + route + answer | `SelfCheck`: `contradictions`, `missing_details`, `comment` |

The self-check looks for contradictions against **all** facts of the text,
but for missing details only against the `details` that step 3 chose for
the answer, and it knows the route's instructions and length limit, so a
short answer is not a problem. A contradiction (wrong numbers, dates or
names, invented details, statements on behalf of a company) sends the
answer back to step 4 together with the problems, and it is checked again
(at most once, `pipeline.MAX_REVISIONS`). A missing detail alone does not:
it becomes an issue of the record. The decision is made in code
(`SelfCheck.passed`).

Then the code sets a priority from `category` and `sentiment`, checks the
answer against its route (`quality.py`) and summarises the run.

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
make run ROUTE=sales                       # force one route for every text
make compare-routes FILE=examples/02_complaint.txt  # one text, every route
make check-replies                         # validate broken replies, no API
```

or directly: `uv run itg-llm --help`. A healthy text costs 5 LLM calls,
7 when the answer is revised. Day 6 recovery can add one fallback call
per step execution and up to two transport retries per call.

## How a text goes through the chain

The log shows every step with its outcome and time, and every decision:

```
08_support | 1/5 extract: goal: Войти в личный кабинет | 3 facts [812 ms]
08_support | 2/5 classify: support / fix_login / negative [640 ms]
08_support | route: support -> technical_steps (by category)
08_support | 3/5 fields: Личный кабинет | needs: ... | missing: 1 [701 ms]
08_support | 4/5 answer: 287 chars [905 ms]
08_support | 5/5 check: failed, 1 contradictions [733 ms]
08_support | self-check failed, revising the answer (1/1)
08_support | 4/5 answer (revision 1): 251 chars [880 ms]
08_support | 5/5 check (revision 1): passed [690 ms]
```

The console output shows the same chain with the step outputs:

```
=== 08_support (v7_extract) ===
[extract]     Summary: ...
              Goal: ...
              Key points: ...
              Facts: ...
[classify]    support / fix_login / negative -> priority high
[route]       support -> technical_steps (by category)
[fields]      Subject: ...
              Requested: ...
              Details: ...
              Missing: ...
[answer]      1. ... 2. ...
[check]       contradiction: ...
[answer r1]   1. ... 2. ...
[check r1]    passed: ...
Final answer: 1. ... 2. ...
Self-check: passed after 1 revision(s)
```

Results are saved to `results/output.json` (`--output` changes the path):
`summary` (totals by category, sentiment, route, priority, self-check
outcome and classification accuracy) and `records`; every record has
`result` (the fields of all steps), `route`, `priority`, `check`,
`issues` and `steps` (the output and time of every step, in order).

## Categories and routing

The categories are the `Category` enum in `schemas.py`; the classification
prompt lists them with a hint each (`prompts.CATEGORY_HINTS`) and asks to
decide by the author's goal from step 1, not by the topic. The routes are
an explicit table in `routing.py`:

| Category           | Route             | How the answer is written                                           | Max chars |
|--------------------|-------------------|---------------------------------------------------------------------|-----------|
| `support`          | `technical_steps` | numbered list of 2-4 concrete steps, asks for missing details        | 450 |
| `complaint`        | `empathetic`      | acknowledges the frustration, then steps to resolve and escalate     | 450 |
| `sales`            | `short_sales`     | 1-2 persuasive sentences: main benefit + next step, no invented prices | 220 |
| `feedback`         | `thanks`          | warm thanks naming what was praised or criticised                    | 200 |
| `general_question` | `direct_answer`   | direct answer first, then up to 2 sentences of advice                | 350 |
| `news`             | `explain_news`    | what the news means for a reader                                     | 250 |
| `other`            | `clarify`         | one friendly sentence asking what the author needs                   | 150 |

### How to see that the category changes the answer

- `make compare-routes FILE=examples/02_complaint.txt` runs steps 1-3
  once and answers the text with **every** route; the route chosen by the
  classification is marked with `*`. Saved to `results/routes.json`.
- `make run ROUTE=sales` answers every example with one forced route, to
  compare with a normal `make run`.
- `quality.py` checks every answer against its route: the length limit
  of the route, and numbered steps for `technical_steps`.

### Day 4 results

`gpt-4.1-nano`, 10 examples (`results/day04.json`, `results/routes.json`):

- category accuracy 7 of 9 (one text failed with an API timeout):
  `04_english` (advice for a team) and `10_other` (a test message) were
  taken for `feedback`;
- the routes clearly change the answer: for the same complaint
  `technical_steps` gave a numbered list, `clarify` one question of 75
  characters, `short_sales` 159 characters;
- 3 answers spoke for the company ("our support team", "our sales team").

Day 5 targets both problems: the category is decided by the author's goal
from the extraction step, and the self-check treats statements made on
behalf of a company as contradictions and sends the answer for revision.

### Day 5 results

`gpt-4.1-nano`, 10 examples (`results/day05.json`), first version of the
self-check:

- all 10 texts went through the whole chain, no errors;
- category accuracy 9 of 10 (day 4: 7 of 9): deciding by the author's
  goal fixed `04_english`; `10_other` (a test message) was taken for
  `support`;
- the self-check found no contradictions at all, but failed 9 of 10
  answers on `missing_details`, also after the revision. It got all facts
  of the text and was asked which ones the answer "should use", so it
  demanded every fact: the order number, payment date and delivery date in
  a complaint, the courier's name and the box in a 200-character thank
  you, even the price and discount in a sales answer whose route forbids
  inventing them. The check contradicted the routes, so the revision could
  not fix anything.

Fix: missing details are checked only against `details` from step 3, the
self-check gets the route's instructions and length limit, and only
contradictions send the answer for revision; missing details become
issues.

### Classification accuracy

`examples/labels.json` has the expected category of every example. When
`make run` processes the examples folder, the summary shows
`Category accuracy: N of 10` and lists every mismatch.

### Priority

`priority` is computed in code from the validated fields, not by the model:

| Category                 | Sentiment | Priority |
|--------------------------|-----------|----------|
| support, complaint       | negative  | high     |
| support, complaint       | other     | medium   |
| sales, general_question  | any       | medium   |
| news, feedback, other    | negative  | medium   |
| news, feedback, other    | other     | low      |

### Issues

`issues` lists soft problems of a valid answer: length limits of the
summary, key points and the route, not exactly 3 key points, no numbered
steps where the route needs them, a language other than the text's, or an
answer that speaks as "we" (as if the assistant were the company). Issues
are also logged as warnings.

## Validation and errors

Every LLM call uses JSON mode (`response_format={"type": "json_object"}`)
and every reply is validated with `schemas.parse_response(raw, schema)`
against the schema of its step. A reply that is wrapped in a
```` ```json ```` fence is unwrapped first; labels are accepted in any case
and spelling (`"General Question"` becomes `general_question`).

A broken reply in any step triggers one fallback prompt with the original
task, the validation error and the required JSON Schema. If that reply is
also invalid, the chain stops for that text, never the run: the text gets
an `"error"` field instead of `"result"`, the error is logged and the other
texts are still processed (the exit code is 1). The error says exactly
what is wrong:

```
Model returned an empty reply
Model returned invalid JSON: Unterminated string starting at (line 1, column 37)
Model returned a JSON list instead of an object
Model JSON does not match the schema: field 'intent' is missing; field 'sentiment' is missing
Model JSON does not match the schema: field 'key_points' must be a list, got string
Model JSON does not match the schema: field 'category' must be one of support, complaint, sales, feedback, general_question, news or other, got 'rant'
```

`make check-replies` runs the saved classification replies from
`examples/bad_replies/` through the same validation without calling the
API.

## Prompts

`prompts.py` holds the prompt of every step: `v7_extract` (step 1, the
variant chosen with `--prompt`) and the templates of steps 2-5, each
filled by a `build_*_prompt` function with the outputs of the earlier
steps. The answer and fields steps run as the neutral assistant, the
self-check as a strict reviewer (`CHECKER_ROLE`). The allowed labels come
from the schema enums, so the prompts and the schema cannot drift apart.
The answer language is detected in code (`utils.language`) and named in
the prompts explicitly ("in Russian" / "in English").

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
and were replaced on day 3.

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

**Chosen: `v4_role_ru`**, the base of the later prompts. Takeaways:
- limits in the prompt work: summaries and responses got shorter and
  more even, with no length violations from v2-v4;
- the role in the system prompt decides *whose voice* the answer has,
  limits alone do not fix that;
- a few-shot example is followed more strongly than an explicit
  instruction, including its language, which is why the answer language
  is now named explicitly.

## Project structure

```
src/itg_llm/
  llm_client.py       # API calls, timeout and bounded transient retries
  guardrails.py       # validated output, size limits and fallback prompt
  prompts.py          # prompts of all five steps and limits
  schemas.py          # Pydantic schemas, parsing and readable errors
  routing.py          # category -> route: answer instructions and limits
  pipeline.py         # the chain: extract -> classify -> route -> fields
                      #   -> answer -> self-check, with a trace of steps
  quality.py          # soft checks of an answer against its route
  report.py           # priority, run summary, classification accuracy
  utils.py            # language detection
  main.py             # CLI
  compare_routes.py   # one text answered by every route
  compare.py          # runs prompt variants side by side
  check_replies.py    # validates saved replies without the API
examples/             # sample input texts + labels.json
examples/bad_replies/ # broken model replies for make check-replies
results/              # pipeline output
tests/                # tests, the LLM is replaced with a fake
```

## Development

```bash
make test        # pytest
make lint        # ruff check
make format      # ruff format
make typecheck   # ty
make check       # all of the above
```

## Day 6: reliability and guardrails

- API connection errors, timeouts, HTTP 408/409/429 and 5xx: up to two
  retries, waiting 0.5 and 1 second. HTTP 400/401/403/404/422 fail without
  retry. Each request has a 30-second SDK timeout. SDK retries are
  disabled so the two retry layers cannot multiply attempts.
- Every chain step validates its JSON with Pydantic. An empty reply,
  invalid JSON, missing fields, wrong types or unknown labels triggers
  one fallback prompt with the original context, error and JSON Schema.
- `final_answer` must fit the selected route's character limit. An
  oversized answer gets the same fallback, then a readable error if it
  remains oversized; answers are never silently truncated.
- Each completion requests at most 2048 output tokens. Raw replies over
  32,000 characters are rejected before JSON parsing. Summary/key-point
  style limits remain soft checks in `quality.py`.
- Retry, fallback and exhausted recovery are logged. A failed text gets
  an error record; the batch continues. Route comparison also isolates
  API errors of individual answer routes.
- There is no fabricated successful answer if recovery fails, including
  when the self-check cannot be completed.

Offline verification (no API key or external requests needed):

```bash
uv run pytest -q tests/test_guardrails.py tests/test_llm_client.py
uv run pytest -q --junitxml=results/day06_tests.xml
make check
```

See [the detailed Day 6 implementation report](../results/day06_report.md)
for scenarios, limits and validation evidence. `results/day05.json` is a
historical real-model run, not evidence of the new recovery behavior.
