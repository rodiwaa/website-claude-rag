# LangSmith Prompt Hub Versioning

## What was implemented
Moved hardcoded LLM prompts in `website-claude-rag` to LangSmith Prompt Hub so they can be edited from the LangSmith UI without code changes.

## Prompts versioned
| Hub name | File | Function |
|----------|------|----------|
| `career-rag-query` | `main.py` | `query_node()` — RAG answer prompt; vars: `{context}`, `{question}` |
| `career-rag-intent` | `chainlit_app.py` | `classify_intent()` — intent classifier; var: `{text}` |

## Files changed
| File | Change |
|------|--------|
| `pyproject.toml` | Added `langsmith>=0.1.0` |
| `prompts.py` | New — `get_query_prompt()` and `get_intent_prompt()` with 5-min TTL cache + hardcoded fallback |
| `push_prompts.py` | New — one-time seed script (`uv run python push_prompts.py`) |
| `main.py` | Import `get_query_prompt`, replaced 5-line f-string with one call |
| `chainlit_app.py` | Import `get_intent_prompt`, replaced 4-line f-string with one call |

## Architecture
- `prompts.py` pulls from LangSmith Hub via `Client().pull_prompt(name)`
- Results cached in-process for 5 minutes (`CACHE_TTL = 300`)
- On any error (no API key, network down), falls back silently to hardcoded defaults
- Prompts stored as `ChatPromptTemplate` with a single `HumanMessagePromptTemplate`

## How to update a prompt
1. Go to LangSmith UI → Prompt Hub
2. Edit `career-rag-query` or `career-rag-intent`
3. App picks up the change within 5 minutes (next cache miss)

## Seeding prompts (one-time)
```bash
uv run python push_prompts.py
```
