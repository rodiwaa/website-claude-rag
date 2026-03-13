"""One-time script to seed prompts to LangSmith Prompt Hub.

Run with: uv run python push_prompts.py
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langsmith import Client

PROMPTS = {
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

client = Client()
for name, template_str in PROMPTS.items():
    prompt = ChatPromptTemplate.from_messages(
        [HumanMessagePromptTemplate.from_template(template_str)]
    )
    client.push_prompt(name, object=prompt)
    print(f"pushed '{name}'")
