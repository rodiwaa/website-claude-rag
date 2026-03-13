"""Prompt loader with LangSmith Hub pull and TTL cache, falling back to hardcoded defaults."""

import time
from dotenv import load_dotenv

load_dotenv()

DEFAULTS = {
    "career-rag-judge-context-relevance": (
        "You are an impartial evaluation judge.\n"
        "Score how relevant the retrieved context is to answering the question.\n\n"
        "Question: {question}\n\nRetrieved Context:\n{context}\n\n"
        "Score 0.0–1.0: 1.0=highly relevant, 0.5=partial, 0.0=irrelevant.\n"
        "Respond with ONLY a single decimal number. No explanation."
    ),
    "career-rag-judge-faithfulness": (
        "You are an impartial evaluation judge.\n"
        "Score whether the answer is fully grounded in the provided context (no hallucination).\n\n"
        "Retrieved Context:\n{context}\n\nAnswer: {answer}\n\n"
        "Score 0.0–1.0: 1.0=fully grounded, 0.5=partially, 0.0=contradicts context.\n"
        "Respond with ONLY a single decimal number. No explanation."
    ),
    "career-rag-judge-answer-relevance": (
        "You are an impartial evaluation judge.\n"
        "Score how well the answer addresses the user's question.\n\n"
        "Question: {question}\n\nAnswer: {answer}\n\n"
        "Score 0.0–1.0: 1.0=directly addresses, 0.5=partial, 0.0=does not address.\n"
        "Respond with ONLY a single decimal number. No explanation."
    ),
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


def get_judge_context_relevance_prompt(question: str, context: str) -> str:
    template = _pull_template("career-rag-judge-context-relevance") \
        or DEFAULTS["career-rag-judge-context-relevance"]
    return template.format(question=question, context=context)


def get_judge_faithfulness_prompt(context: str, answer: str) -> str:
    template = _pull_template("career-rag-judge-faithfulness") \
        or DEFAULTS["career-rag-judge-faithfulness"]
    return template.format(context=context, answer=answer)


def get_judge_answer_relevance_prompt(question: str, answer: str) -> str:
    template = _pull_template("career-rag-judge-answer-relevance") \
        or DEFAULTS["career-rag-judge-answer-relevance"]
    return template.format(question=question, answer=answer)
