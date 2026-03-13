"""One-time script to seed prompts to LangSmith Prompt Hub.

Run with: uv run python push_prompts.py
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langsmith import Client

PROMPTS = {
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
