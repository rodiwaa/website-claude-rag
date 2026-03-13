"""Career/Portfolio RAG: upload → markdown → chunk → embed_store | query → retrieve → answer."""

import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from litellm import completion
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from langchain_core.runnables import RunnableConfig
from langsmith import Client as LangSmithClient
from langsmith import get_current_run_tree

from prompts import get_query_prompt, DEFAULTS

load_dotenv()


def _log_metrics(run_id: str, metrics: dict[str, float]) -> None:
    """Fire-and-forget: log scalar metrics as LangSmith feedback."""
    try:
        print("_log_metrics")
        print("Fire-and-forget: log scalar metrics as LangSmith feedback.")
        ls = LangSmithClient()
        for key, score in metrics.items():
            ls.create_feedback(run_id=run_id, key=key, score=score)
    except Exception:
        pass  # never block the user query

os.environ["LANGSMITH_PROJECT"] = "website-claude-rag"


def _parse_score(text: str) -> float | None:
    match = re.search(r"\b([01](?:\.\d+)?|\d?\.\d+)\b", text.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


def _run_judge(run_id: str, question: str, context: str, answer: str) -> None:
    def _judge():
        try:
            evals = [
                (DEFAULTS["career-rag-judge-context-relevance"].format(question=question, context=context),
                 "judge/context_relevance"),
                (DEFAULTS["career-rag-judge-faithfulness"].format(context=context, answer=answer),
                 "judge/faithfulness"),
                (DEFAULTS["career-rag-judge-answer-relevance"].format(question=question, answer=answer),
                 "judge/answer_relevance"),
            ]
            metrics: dict[str, float] = {}
            for prompt, key in evals:
                try:
                    resp = completion(
                        model=JUDGE_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10,
                        temperature=0.0,
                    )
                    raw = resp.choices[0].message.content or ""
                    score = _parse_score(raw)
                    if score is not None:
                        metrics[key] = score
                except Exception as e:
                    print(f"[judge] failed for '{key}': {e}")
            if metrics:
                _log_metrics(run_id, metrics)
        except Exception:
            pass
    threading.Thread(target=_judge, daemon=True).start()

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COLLECTION_NAME = "career-docs"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5


_qdrant_client: QdrantClient | None = None
_openai_client: OpenAI | None = None


def get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"[qdrant] created collection '{COLLECTION_NAME}'")


def embed_text(text: str) -> list[float]:
    client = get_openai()
    response = client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding


# ── State ────────────────────────────────────────────────────────────────────

class UploadState(BaseModel):
    messages: Annotated[list, add_messages] = []
    file_path: str = ""
    file_name: str = ""
    upload_time: str = ""
    raw_markdown: str = ""
    chunks: list[dict] = []  # {text, source, chunk_index}


class QueryState(BaseModel):
    messages: Annotated[list, add_messages] = []


# ── Upload Nodes ─────────────────────────────────────────────────────────────

def upload_node(state: UploadState) -> dict:
    """Validate the file exists and record its path and metadata."""
    if not state.file_path:
        raise ValueError("No file path provided.")
    path = Path(state.file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    upload_time = datetime.now().isoformat()
    file_name = state.file_name or path.name
    print(f"[upload] loaded: {path} as '{file_name}'")
    return {
        "file_path": str(path),
        "file_name": file_name,
        "upload_time": upload_time,
        "messages": [{"role": "assistant", "content": f"Uploaded: {file_name}"}],
    }


def markdown_node(state: UploadState) -> dict:
    """Convert the uploaded PDF to markdown using docling."""
    converter = DocumentConverter()
    result = converter.convert(state.file_path)
    markdown = result.document.export_to_markdown()
    print(f"[markdown] converted {len(markdown)} chars")
    return {
        "raw_markdown": markdown,
        "messages": [{"role": "assistant", "content": "Converted to markdown."}],
    }


def chunk_node(state: UploadState) -> dict:
    """Split markdown into ~500-char chunks with overlap, attaching source metadata."""
    text = state.raw_markdown
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": state.file_name,
                "chunk_index": idx,
                "upload_time": state.upload_time,
            })
            idx += 1
        start = end - CHUNK_OVERLAP
    print(f"[chunk] created {len(chunks)} chunks from '{state.file_name}'")
    return {
        "chunks": chunks,
        "messages": [{"role": "assistant", "content": f"Split into {len(chunks)} chunks."}],
    }


def embed_store_node(state: UploadState) -> dict:
    """Batch-embed all chunks then upsert into Qdrant Cloud collection."""
    client = get_qdrant()
    ensure_collection(client)

    oai = get_openai()
    texts = [c["text"] for c in state.chunks]
    response = oai.embeddings.create(input=texts, model=EMBED_MODEL)
    vectors = [item.embedding for item in response.data]

    points = []
    for chunk, vector in zip(state.chunks, vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "upload_time": chunk["upload_time"],
            },
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[embed_store] upserted {len(points)} vectors for '{state.file_name}'")
    return {
        "messages": [
            {"role": "assistant", "content": f"Indexed {len(points)} chunks from '{state.file_name}'."}
        ],
    }


# ── Query Node ────────────────────────────────────────────────────────────────

def query_node(state: QueryState, config: RunnableConfig) -> dict:
    """Embed question, retrieve top-k chunks from Qdrant, answer with LLM."""
    question = state.messages[-1].content

    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        return {"messages": [{"role": "assistant", "content": "No documents indexed yet. Please upload a PDF first."}]}

    query_vector = embed_text(question)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    )
    results = response.points

    scores = [r.score for r in results if r.score is not None]
    run_tree = get_current_run_tree()
    run_id = str(run_tree.id) if run_tree else None
    print(f"[metrics] run_id={run_id}")
    if run_id:
        _log_metrics(str(run_id), {
            "retrieval/hit_count": float(len(results)),
            "retrieval/avg_score": sum(scores) / len(scores) if scores else 0.0,
            "retrieval/top_score": max(scores) if scores else 0.0,
        })

    if not results:
        return {"messages": [{"role": "assistant", "content": "No relevant content found. Try uploading more documents."}]}

    context_parts = []
    sources = []
    for hit in results:
        text = hit.payload.get("text", "")
        source = hit.payload.get("source", "unknown")
        context_parts.append(f"[{source}]\n{text}")
        if source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(context_parts)
    prompt = get_query_prompt(context=context, question=question)

    response = completion(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=800)
    answer = response.choices[0].message.content

    sources_str = ", ".join(f"`{s}`" for s in sources)
    full_answer = f"{answer}\n\n**Sources:** {sources_str}"

    if run_id:
        print(f"run id? {run_id}")
        _log_metrics(str(run_id), {
            "generation/answer_length": float(len(answer)),
            "generation/source_count": float(len(sources)),
            "generation/has_answer": 1.0,
        })
        _run_judge(
            run_id=str(run_id),
            question=question,
            context=context,
            answer=answer,
        )

    return {"messages": [{"role": "assistant", "content": full_answer}]}


def list_docs_node(state: QueryState) -> dict:
    """List all unique source documents indexed in Qdrant."""
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        return {"messages": [{"role": "assistant", "content": "No documents indexed yet."}]}

    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        limit=10000,
    )
    docs: dict[str, dict] = {}
    for point in results:
        source = point.payload.get("source", "unknown")
        upload_time = point.payload.get("upload_time", "")
        if source not in docs:
            docs[source] = {"source": source, "upload_time": upload_time, "chunks": 0}
        docs[source]["chunks"] += 1

    if not docs:
        return {"messages": [{"role": "assistant", "content": "No documents indexed yet."}]}

    lines = ["**Indexed documents:**\n"]
    for doc in sorted(docs.values(), key=lambda d: d["upload_time"]):
        lines.append(f"- `{doc['source']}` — {doc['chunks']} chunks (uploaded {doc['upload_time'][:10]})")
    return {"messages": [{"role": "assistant", "content": "\n".join(lines)}]}


# ── Upload Graph ──────────────────────────────────────────────────────────────

upload_graph = StateGraph(UploadState)
upload_graph.add_node("upload", upload_node)
upload_graph.add_node("markdown", markdown_node)
upload_graph.add_node("chunk", chunk_node)
upload_graph.add_node("embed_store", embed_store_node)

upload_graph.add_edge(START, "upload")
upload_graph.add_edge("upload", "markdown")
upload_graph.add_edge("markdown", "chunk")
upload_graph.add_edge("chunk", "embed_store")
upload_graph.add_edge("embed_store", END)

app = upload_graph.compile()


# ── Query Graph ───────────────────────────────────────────────────────────────

query_graph = StateGraph(QueryState)
query_graph.add_node("query", query_node)
query_graph.add_edge(START, "query")
query_graph.add_edge("query", END)

query_app = query_graph.compile()

list_graph = StateGraph(QueryState)
list_graph.add_node("list_docs", list_docs_node)
list_graph.add_edge(START, "list_docs")
list_graph.add_edge("list_docs", END)

list_app = list_graph.compile()


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if file_path:
        result = app.invoke({"file_path": file_path})
        print("\n── Summary ──")
        for msg in result["messages"]:
            print(f"  {msg.content}")
    else:
        print("Usage: python main.py <path-to-pdf>")
