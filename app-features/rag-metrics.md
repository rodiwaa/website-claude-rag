# RAG Metrics — LangSmith Feedback Logging

## Intent

Provide visibility into retrieval and generation quality for each query, surfaced directly in LangSmith without adding new dependencies or changing the app's user-facing behavior.

## What Was Added

A small instrumentation layer in `main.py` that attaches scalar metrics to the active LangSmith run after each query.

### Helper: `_log_metrics()`

Fire-and-forget function that calls `LangSmithClient().create_feedback()` for each metric. Errors are silently swallowed so metrics never block a user query.

### Retrieval Metrics (logged after Qdrant search)

| Key | Description |
|-----|-------------|
| `retrieval/hit_count` | Number of chunks returned from Qdrant |
| `retrieval/avg_score` | Mean cosine similarity score across returned chunks |
| `retrieval/top_score` | Highest similarity score in the result set |

### Generation Metrics (logged after LLM answer)

| Key | Description |
|-----|-------------|
| `generation/answer_length` | Character count of the LLM response |
| `generation/source_count` | Number of unique source documents cited |
| `generation/has_answer` | `1.0` if a real answer was returned |

## How It Works

LangGraph auto-creates a run per `invoke()` call. The `run_id` is extracted from the `RunnableConfig` passed to `query_node`. Metrics are attached to that run via `create_feedback()`, making them visible in the LangSmith UI under the **Feedback** tab.

## Viewing in LangSmith

1. Go to LangSmith → project `website-claude-rag` → Runs
2. Open any run → **Feedback** tab to see per-run scores
3. Use **Add to Dashboard** to chart metrics (e.g. `retrieval/avg_score`) over time

## Files Changed

- `main.py` — added `_log_metrics()` helper, `RunnableConfig` import, and metric logging calls in `query_node()` (~20 lines total)
