# LLM-as-a-Judge — Deeper RAG Metrics

## Intent

Extend the existing scalar metrics with semantic quality scores from the **RAG Triad**, evaluated by a judge LLM after each query. This catches issues that similarity scores cannot — hallucination, off-topic answers, irrelevant retrieval — without any impact on user-perceived latency.

## What Was Added

Three new `judge/*` metrics logged to LangSmith per query, scored 0.0–1.0 by a configurable judge LLM running in a background thread.

### Judge Metrics

| Key | Description |
|---|---|
| `judge/context_relevance` | Are the retrieved chunks relevant to the question? |
| `judge/faithfulness` | Is the answer grounded in the context (no hallucination)? |
| `judge/answer_relevance` | Does the answer actually address the user's question? |

These map directly to the **RAG Triad** industry standard:
- *Did I get the right information?* → `context_relevance`
- *Did I use the information correctly?* → `faithfulness`
- *Does the answer address the request?* → `answer_relevance`

## How It Works

### 1. Background thread (non-blocking)

After `query_node` logs its existing generation metrics, it calls `_run_judge()`. This immediately spawns a `daemon=True` thread and returns — the user gets their answer with no added latency. The thread makes three sequential judge LLM calls, parses each score, and logs them via the existing `_log_metrics()` helper.

```
query_node:
  1. embed question → Qdrant top-5        [unchanged]
  2. log retrieval metrics                [unchanged]
  3. build context string                 [unchanged]
  4. call DeepSeek LLM → answer           [unchanged]
  5. log generation metrics               [unchanged]
  6. _run_judge(...) → background thread  [NEW]
     ├── judge call → context_relevance → LangSmith
     ├── judge call → faithfulness → LangSmith
     └── judge call → answer_relevance → LangSmith
  7. return answer to user                [unchanged]
```

### 2. Judge prompts (versioned in LangSmith Hub)

Three prompts added to `prompts.py` following the same `_pull_template()` + `DEFAULTS` fallback pattern:

- `career-rag-judge-context-relevance` — inputs: `{question}`, `{context}`
- `career-rag-judge-faithfulness` — inputs: `{context}`, `{answer}`
- `career-rag-judge-answer-relevance` — inputs: `{question}`, `{answer}`

Each prompt asks the judge for a single decimal number (0.0–1.0) with an explicit rubric. `max_tokens=10` and `temperature=0.0` keep calls fast and deterministic.

### 3. Score parsing

`_parse_score()` extracts the first valid float in [0, 1] from the judge response using a regex. Values are clamped to [0.0, 1.0]. If parsing fails, the metric is omitted (not logged) rather than failing.

### 4. Judge model

Configurable via `JUDGE_MODEL` env var, defaulting to `openai/gpt-4o-mini`. Routed through LiteLLM — any LiteLLM-compatible model works.

```
JUDGE_MODEL=openai/gpt-4o-mini       # default, accurate and cheap
JUDGE_MODEL=groq/llama-3.3-70b-versatile  # fast, free tier
JUDGE_MODEL=deepseek/deepseek-chat        # same model as main LLM
```

## Full Metrics Picture After This Feature

| Key | Type | Source |
|---|---|---|
| `retrieval/hit_count` | float | Qdrant result count |
| `retrieval/avg_score` | float | Mean cosine similarity |
| `retrieval/top_score` | float | Max cosine similarity |
| `generation/answer_length` | float | Character count |
| `generation/source_count` | float | Unique sources cited |
| `generation/has_answer` | float | 1.0 if answer returned |
| `judge/context_relevance` | 0–1 | Judge LLM |
| `judge/faithfulness` | 0–1 | Judge LLM |
| `judge/answer_relevance` | 0–1 | Judge LLM |

## Files Changed

- `main.py` — added `JUDGE_MODEL` constant, `_parse_score()`, `_run_judge()`, wired into `query_node()`
- `prompts.py` — added 3 judge prompt defaults and 3 `get_judge_*` getter functions
- `push_prompts.py` — added 3 judge prompts to seed LangSmith Hub (run once)
- `.env.example` — added `JUDGE_MODEL` variable

## Viewing in LangSmith

Same as existing metrics:
1. LangSmith → project `website-claude-rag` → Runs
2. Open any run → **Feedback** tab
3. `judge/*` scores appear alongside `retrieval/*` and `generation/*` scores
4. Chart over time to track answer quality trends
