"""Prompt loader with LangSmith Hub pull and TTL cache, falling back to hardcoded defaults."""

import time
from dotenv import load_dotenv

load_dotenv()

DEFAULTS = {
    "career-rag-query": (
        "You are a helpful assistant. Answer the user's question using only the context below.\n"
        "At the end of your answer, list the source documents used.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\nAnswer:"
    ),
    "career-rag-intent": (
        "Classify this user message as exactly one word: 'upload' or 'query'.\n"
        "'upload' = user wants to upload or add a document, resume, PDF, or portfolio.\n"
        "'query' = user is asking a question about their documents or career.\n"
        "Message: {text}\nAnswer:"
    ),
}

_cache: dict[str, tuple[str, float]] = {}
CACHE_TTL = 300  # seconds


def _pull_template(hub_name: str) -> str | None:
    cached = _cache.get(hub_name)
    if cached and (time.time() - cached[1]) < CACHE_TTL:
        return cached[0]
    try:
        from langsmith import Client

        client = Client()
        prompt_obj = client.pull_prompt(hub_name)
        template = prompt_obj.messages[0].prompt.template
        _cache[hub_name] = (template, time.time())
        return template
    except Exception as e:
        print(f"[prompts] could not pull '{hub_name}' from LangSmith: {e}")
        return None


def get_query_prompt(context: str, question: str) -> str:
    template = _pull_template("career-rag-query") or DEFAULTS["career-rag-query"]
    return template.format(context=context, question=question)


def get_intent_prompt(text: str) -> str:
    template = _pull_template("career-rag-intent") or DEFAULTS["career-rag-intent"]
    return template.format(text=text)
