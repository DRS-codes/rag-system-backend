"""
Evaluation metrics, split into two families:

1. Retrieval metrics — do we fetch the right chunks? Computed against
   ground-truth doc_ids in the eval set, so they're cheap, deterministic,
   and don't require an LLM judge:
   - Precision@k: of the k chunks retrieved, what fraction come from a
     relevant document?
   - Recall@k: of all relevant documents, what fraction had at least one
     chunk retrieved?
   - MRR (Mean Reciprocal Rank): how high up the ranking was the first
     relevant hit? Rewards ranking quality, not just presence in the set.
   - NDCG@k: rank-aware, position-discounted relevance — penalizes a
     relevant hit at position 8 more gently than MRR does, but still less
     than a hit at position 1.

2. Generation metrics — given the retrieved chunks, is the answer any
   good? These require an LLM-as-judge since "is this answer supported by
   the context" isn't a string-matching problem:
   - Faithfulness: fraction of claims in the answer that are actually
     supported by the retrieved context (catches hallucination —
     specifically the failure mode where the model answers from
     parametric knowledge instead of the provided chunks).
   - Answer relevancy: does the answer actually address the question
     asked (catches the failure mode where the answer is faithful to the
     context but off-topic or evasive).

   Faithfulness and relevancy are measured independently on purpose: an
   answer can be faithful but not relevant (accurately restates an
   unrelated chunk) or relevant but not faithful (correctly on-topic but
   hallucinated). Optionally, RAGAS can be run for a second, differently-
   implemented opinion on the same two axes.
"""
from __future__ import annotations

import json
import math

from app.config import settings
from app.models.schemas import ScoredChunk


# ---------------- retrieval metrics ----------------

def precision_at_k(retrieved: list[ScoredChunk], relevant_doc_ids: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for sc in top if sc.chunk.doc_id in relevant_doc_ids)
    return hits / len(top)


def recall_at_k(retrieved: list[ScoredChunk], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 1.0
    top_doc_ids = {sc.chunk.doc_id for sc in retrieved[:k]}
    hits = len(top_doc_ids & relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def mrr(retrieved: list[ScoredChunk], relevant_doc_ids: set[str]) -> float:
    for i, sc in enumerate(retrieved):
        if sc.chunk.doc_id in relevant_doc_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[ScoredChunk], relevant_doc_ids: set[str], k: int) -> float:
    top = retrieved[:k]
    dcg = sum(
        (1.0 if sc.chunk.doc_id in relevant_doc_ids else 0.0) / math.log2(i + 2)
        for i, sc in enumerate(top)
    )
    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(retrieved: list[ScoredChunk], relevant_doc_ids: set[str], k: int) -> dict:
    return {
        "precision@k": round(precision_at_k(retrieved, relevant_doc_ids, k), 4),
        "recall@k": round(recall_at_k(retrieved, relevant_doc_ids, k), 4),
        "mrr": round(mrr(retrieved, relevant_doc_ids), 4),
        "ndcg@k": round(ndcg_at_k(retrieved, relevant_doc_ids, k), 4),
    }


# ---------------- generation metrics (LLM-as-judge) ----------------

_FAITHFULNESS_PROMPT = """You are evaluating whether an AI-generated answer is faithful to its source context.

Context:
{context}

Answer to evaluate:
{answer}

Break the answer into individual factual claims. For each claim, decide if it is directly supported by the context (yes/no).
Respond ONLY with JSON: {{"claims": [{{"claim": "...", "supported": true|false}}, ...]}}"""

_RELEVANCY_PROMPT = """You are evaluating whether an AI-generated answer actually addresses the question asked, regardless of factual accuracy.

Question: {question}
Answer: {answer}

Score relevancy from 0.0 (completely off-topic / evasive) to 1.0 (directly and fully addresses the question).
Respond ONLY with JSON: {{"relevancy": <float>, "reasoning": "<one sentence>"}}"""


def _judge_call(prompt: str) -> str:
    """
    Uses the same generation provider configured for answers, as the judge.

    Note if generation_provider is "ollama": small local models are
    noticeably less reliable at strictly following "respond ONLY with
    JSON" than Claude/GPT-4-class models. When the judge's response isn't
    valid JSON, faithfulness()/answer_relevancy() already catch that and
    return {"score": None, "error": ...} rather than crashing the eval run
    — but expect a higher null-score rate with a small local judge model,
    especially early on. If that rate is high enough to make aggregate
    scores unreliable, consider keeping Ollama for answer generation
    (cheap, frequent) while using a hosted model as the judge (occasional,
    only during eval runs) — that's a config change, not a code change:
    point ANTHROPIC_API_KEY/OPENAI_API_KEY at a real key and swap the
    provider check below to a separate JUDGE_PROVIDER setting if you want
    generation and judging to use different providers.
    """
    if settings.generation_provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    if settings.generation_provider == "ollama":
        import httpx

        payload = {
            "model": settings.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = httpx.post(f"{settings.ollama_base_url}/api/chat", json=payload, timeout=120.0)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_generation_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return resp.choices[0].message.content


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("json", 1)[-1] if raw.lower().startswith("json") else raw
    return json.loads(raw)


def faithfulness(answer: str, context_chunks: list[ScoredChunk]) -> dict:
    context = "\n\n".join(sc.chunk.text for sc in context_chunks)
    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    try:
        result = _parse_json_response(_judge_call(prompt))
        claims = result.get("claims", [])
        if not claims:
            return {"score": None, "claims": []}
        supported = sum(1 for c in claims if c.get("supported"))
        return {"score": round(supported / len(claims), 4), "claims": claims}
    except Exception as e:  # judge call/parsing failures shouldn't crash the eval run
        return {"score": None, "error": str(e)}


def answer_relevancy(question: str, answer: str) -> dict:
    prompt = _RELEVANCY_PROMPT.format(question=question, answer=answer)
    try:
        result = _parse_json_response(_judge_call(prompt))
        return {"score": round(float(result["relevancy"]), 4), "reasoning": result.get("reasoning", "")}
    except Exception as e:
        return {"score": None, "error": str(e)}
