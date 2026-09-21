# Opik Trace Rerouter

A Streamlit app that takes Opik exports (CSV / Excel / JSON / JSONL) of **traces** and **spans** and
rebuilds the human-readable view you get inside the Opik app — conversations, execution trees,
waterfalls, feedback scores, tokens and cost — offline, from files.

## How Opik is put together

Opik's observability model is three nested levels, and the exports are flat tables that encode the
nesting in a few columns:

| Level | Identity | How nesting is encoded |
|---|---|---|
| **Thread** | `thread_id` | Traces sharing a `thread_id` are one conversation |
| **Trace** | `id` | One end-to-end request; carries `input`, `output`, `feedback_scores`, `usage`, `total_estimated_cost` |
| **Span** | `id` + `trace_id` | `parent_span_id` builds the execution tree; `type` is `llm` / `tool` / `general` / `guardrail` |

Two things make the raw export hard to read, and both are what this tool fixes:

1. **Everything structured is stringified.** `input`, `output`, `metadata`, `usage`,
   `feedback_scores` come back as JSON text (sometimes Python-repr text with single quotes).
2. **The tree is implicit.** Spans arrive as a flat list; the hierarchy only exists in
   `parent_span_id`, and the conversation only exists in `thread_id`.

`opik_core.py` reverses both: it re-parses the payloads, rebuilds the span tree, groups traces into
threads, and renders any payload shape into plain text (chat `messages` arrays, OpenAI/OpenRouter
`choices` responses, single-field dicts, plain strings).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload your exports in the sidebar. Upload the **traces** sheet and the **spans** sheet
together — the app detects which is which from the columns. `sample_data/` has a working pair
modelled on a Pinecone + OpenRouter RAG pipeline so you can see the UI before wiring in your own.

## Getting the data out of Opik

- **UI:** select rows → Actions → Export CSV (capped at the table page size, 100 rows).
- **SDK, for anything larger:**

```python
import opik, pandas as pd
client = opik.Opik(workspace="tagorej")
traces = client.search_traces(project_name="Streamlit", max_results=5000)
spans  = client.search_spans(project_name="Streamlit", max_results=20000)
pd.DataFrame([t.dict() for t in traces]).to_csv("traces.csv", index=False)
pd.DataFrame([s.dict() for s in spans]).to_csv("spans.csv", index=False)
```

- **REST:** `GET /v1/private/traces` and `GET /v1/private/spans`. Dump the JSON response
  straight to a `.json` file — the loader unwraps `{"content": [...]}`.

## What the three tabs give you

- **Threads** — the conversation view: every turn in a thread as user/assistant bubbles, with the
  execution tree for that turn folded underneath, plus a Markdown transcript download.
- **Traces** — searchable list on the left (free-text across name, input and output; errors-only
  filter), full detail on the right: Conversation / Execution / Raw tabs, a span waterfall coloured
  by span type, per-span input, output, model, provider, tokens, metadata and errors.
- **Analytics** — mean latency by trace name, time by span type, mean feedback scores, slowest traces.

If threads come out empty, that's the usual `thread_id` propagation gap rather than a parsing
problem — the traces were logged without a thread id, so Opik has nothing to group on either.

## Files

```
app.py          Streamlit UI
opik_core.py    loading, normalisation, span tree, thread assembly, humaniser, Markdown export
requirements.txt
sample_data/    traces_sample.csv, spans_sample.csv
```

`opik_core.py` has no Streamlit imports, so you can use the same parsing and `humanise()` in a
notebook or a CLI.
