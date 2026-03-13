"""Chainlit UI for career/portfolio PDF RAG — upload and query documents."""

import asyncio
import os
import tempfile
from pathlib import Path

import chainlit as cl
from langchain_core.messages import HumanMessage
from litellm import completion

from main import app, list_app, query_app
from prompts import get_intent_prompt

INTENT_MODEL = os.getenv("INTENT_MODEL", "groq/llama-3.1-8b-instant")


def classify_intent(text: str) -> str:
    """Returns 'upload' or 'query' using a fast Groq model."""
    prompt = get_intent_prompt(text=text)
    response = completion(model=INTENT_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=5)
    label = response.choices[0].message.content.strip().lower()
    return "upload" if "upload" in label else "query"


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "Welcome to Career RAG!\n\n"
            "Upload your resume, project docs, or portfolio PDFs and I will index them.\n"
            "Then ask me anything — I will answer using your documents.\n\n"
            "**Commands:**\n"
            "- Attach a PDF to upload and index it\n"
            "- Type a question to search your documents\n"
            "- Type `/list` to see all indexed documents"
        )
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    # /list command
    if msg.content.strip().lower() in ("/list", "list"):
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: list_app.invoke({"messages": [HumanMessage(content="list")]})
        )
        await cl.Message(content=result["messages"][-1].content).send()
        return

    # File attached → always run upload pipeline
    if msg.elements:
        file_el = msg.elements[0]
        file_path = file_el.path
        file_name = file_el.name

        await cl.Message(content=f"Processing `{file_name}`...").send()

        tmp_path = None
        try:
            suffix = Path(file_name).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(Path(file_path).read_bytes())

            await cl.Message(content="Converting to markdown and chunking...").send()

            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: app.invoke({"file_path": tmp_path, "file_name": file_name})
            )

            chunks = result.get("chunks", [])
            chunk_count = len(chunks)

            # Find final indexed message
            indexed_msg = next(
                (m.content for m in result["messages"] if "Indexed" in m.content),
                f"Indexed {chunk_count} chunks from '{file_name}'.",
            )
            await cl.Message(content=f"**Done!** {indexed_msg}").send()

        except Exception as e:
            await cl.Message(content=f"Error processing file: {e}").send()
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
        return

    # No file — classify intent
    intent = await asyncio.get_event_loop().run_in_executor(
        None, lambda: classify_intent(msg.content)
    )

    if intent == "upload":
        await cl.Message(content="Please attach a PDF to upload and index it.").send()
        return

    # Query intent
    await cl.Message(content="Searching your documents...").send()
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: query_app.invoke({"messages": [HumanMessage(content=msg.content)]})
    )
    answer = result["messages"][-1].content
    await cl.Message(content=answer).send()
