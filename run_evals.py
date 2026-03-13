"""Offline batch evaluations using LangSmith Datasets + langsmith.evaluate()."""

import json
import os
import random
import re

import langsmith
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import Client as LangSmithClient
from litellm import completion

load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "website-claude-rag"

from main import (  # noqa: E402
    COLLECTION_NAME,
    JUDGE_MODEL,
    TOP_K,
    embed_text,
    get_qdrant,
    query_app,
)
from prompts import DEFAULTS  # noqa: E402

DATASET_NAME = "career-rag-golden"
N_PAIRS = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_score(text: str) -> float | None:
    match = re.search(r"\b([01](?:\.\d+)?|\d?\.\d+)\b", text.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


def _judge_call(prompt: str) -> float | None:
    resp = completion(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or ""
    return _parse_score(raw)


# ── Step 1: Generate QA pairs from Qdrant chunks ──────────────────────────────

def generate_qa_pairs(n: int = N_PAIRS) -> list[dict]:
    client = get_qdrant()
    all_points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        limit=10000,
    )
    if not all_points:
        raise RuntimeError("No chunks found in Qdrant. Upload documents first.")

    sample = random.sample(all_points, min(n, len(all_points)))
    pairs = []
    for point in sample:
        chunk_text = point.payload.get("text", "")
        prompt = (
            "You are creating a factual QA dataset from a resume/portfolio.\n"
            "Given the excerpt below, generate ONE factual question and its exact answer.\n"
            "Rules:\n"
            "- Question must be answerable with a specific fact: year, company name, project name, technology, role title, duration, or location\n"
            "- Answer must be a short, specific fact copied directly from the text (not paraphrased)\n"
            "- Do NOT generate subjective, open-ended, or opinion questions (e.g. 'How did you...', 'What challenges...', 'Why did you...')\n"
            "- Examples of good questions: 'Which year did you join TCS?', 'What projects did you work on at JPMorgan Chase?', 'What technology stack was used in <project>?'\n"
            'Respond as JSON: {"question": "...", "answer": "..."}\n\n'
            f"Excerpt:\n{chunk_text}"
        )
        try:
            resp = completion(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content or ""
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            parsed = json.loads(raw)
            pairs.append({
                "question": parsed["question"],
                "expected_answer": parsed["answer"],
                "source_chunk": chunk_text,
            })
            print(f"  [qa] generated: {parsed['question'][:60]}...")
        except Exception as e:
            print(f"  [qa] skipped chunk (parse error): {e}")
    return pairs


# ── Step 2: Create or reuse LangSmith dataset ─────────────────────────────────

def create_or_update_dataset(client: LangSmithClient, pairs: list[dict]):
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        client.delete_dataset(dataset_id=existing[0].id)
        print(f"[dataset] deleted old dataset '{DATASET_NAME}' to reseed")

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Golden QA pairs (factual) generated from career/portfolio chunks.",
    )
    client.create_examples(
        inputs=[{"question": p["question"]} for p in pairs],
        outputs=[{"answer": p["expected_answer"]} for p in pairs],
        dataset_id=dataset.id,
    )
    print(f"[dataset] created '{DATASET_NAME}' and seeded {len(pairs)} examples")
    return dataset


# ── Step 3: Target function ───────────────────────────────────────────────────

def target(inputs: dict) -> dict:
    question = inputs["question"]

    query_vector = embed_text(question)
    qdrant = get_qdrant()
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    )
    context_parts = [hit.payload.get("text", "") for hit in response.points]
    context = "\n\n---\n\n".join(context_parts)

    result = query_app.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content

    return {"answer": answer, "context": context}


# ── Step 4: Evaluator functions ───────────────────────────────────────────────

def eval_context_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    prompt = DEFAULTS["career-rag-judge-context-relevance"].format(
        question=inputs["question"], context=outputs.get("context", "")
    )
    score = _judge_call(prompt)
    return {"key": "context_relevance", "score": score}


def eval_faithfulness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    prompt = DEFAULTS["career-rag-judge-faithfulness"].format(
        context=outputs.get("context", ""), answer=outputs.get("answer", "")
    )
    score = _judge_call(prompt)
    return {"key": "faithfulness", "score": score}


def eval_answer_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    prompt = DEFAULTS["career-rag-judge-answer-relevance"].format(
        question=inputs["question"], answer=outputs.get("answer", "")
    )
    score = _judge_call(prompt)
    return {"key": "answer_relevance", "score": score}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ls_client = LangSmithClient()

    print("[evals] generating QA pairs from Qdrant chunks...")
    pairs = generate_qa_pairs(n=N_PAIRS)
    print(f"[evals] got {len(pairs)} valid pairs")

    create_or_update_dataset(ls_client, pairs)

    print("[evals] running langsmith.evaluate()...")
    langsmith.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[eval_context_relevance, eval_faithfulness, eval_answer_relevance],
        experiment_prefix="career-rag",
    )
    print("[evals] done — check LangSmith UI for results")
