# Project Recap: website-claude-rag

## Overview

A career/portfolio RAG (Retrieval-Augmented Generation) chatbot built to be deployed as a website chatbot. Users upload PDF documents (resumes, project write-ups, portfolio) which get indexed into Qdrant Cloud. The Chainlit UI then allows querying those documents conversationally.

---

## Architecture

### Two LangGraph Graphs

**Upload Graph** (`app`): `START → upload → markdown → chunk → embed_store → END`

| Node | What it does |
|------|-------------|
| `upload` | Validates file exists, records metadata (filename, upload timestamp) |
| `markdown` | Converts PDF → markdown using **docling** (`DocumentConverter`) |
| `chunk` | Splits markdown into ~500-char chunks with 50-char overlap |
| `embed_store` | Batch-embeds all chunks via OpenAI `text-embedding-3-small`, upserts to Qdrant Cloud |

**Query Graph** (`query_app`): `START → query → END`
- Embeds question, runs cosine similarity search (top-5) against Qdrant, answers with LLM

**List Graph** (`list_app`): `START → list_docs → END`
- Scrolls Qdrant collection, returns unique source files with chunk counts

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Graph orchestration | LangGraph ≥1.0.9 |
| UI | Chainlit ≥2.9.6 |
| PDF → Markdown | docling |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Vector DB | **Qdrant Cloud** (collection: `career_docs`, cosine distance) |
| LLM (answers) | DeepSeek via LiteLLM (`deepseek/deepseek-chat`) |
| Intent classifier | Groq `llama-3.1-8b-instant` (fast, cheap) |
| Telemetry | **LangSmith** (project: `website-claude-rag`) |
| Package manager | uv |

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | All LangGraph nodes, states, and compiled graphs |
| `chainlit_app.py` | Chainlit UI — handles file upload, intent classification, query, /list |
| `pyproject.toml` | Dependencies |
| `langgraph.json` | LangGraph CLI server config |
| `chainlit.md` | Chainlit welcome message |

---

## State Models

```python
class UploadState(BaseModel):
    messages, file_path, file_name, upload_time, raw_markdown, chunks

class QueryState(BaseModel):
    messages
```

---

## Chainlit UI Flow

1. **File attached** → always triggers upload pipeline (no intent check needed)
2. **Text message** → intent classifier (Groq) routes to:
   - `"upload"` → prompt user to attach a PDF
   - `"query"` → run query pipeline, return answer + sources
3. **`/list` command** → show all indexed documents

---

## Configuration (`.env`)

```
QDRANT_URL=<qdrant cloud url>
QDRANT_API_KEY=<qdrant api key>
OPENAI_API_KEY=<openai key for embeddings>
LANGSMITH_API_KEY=<langsmith key>
LLM_MODEL=deepseek/deepseek-chat        # override for answers
INTENT_MODEL=groq/llama-3.1-8b-instant  # override for intent
```

---

## Running the App

```bash
# Install deps
uv sync

# CLI upload (testing)
uv run python main.py path/to/resume.pdf

# Chainlit UI
uv run chainlit run chainlit_app.py
```

---

## Intended Deployment

This chatbot is meant to be embedded in a personal website. Visitors can ask questions about the owner's career, skills, and projects, with answers grounded in the uploaded PDF documents.
