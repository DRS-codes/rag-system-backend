"""
Answer generation. The prompt is deliberately strict about grounding:
the model is instructed to answer only from the provided chunks and to
say so explicitly when the chunks don't contain the answer, rather than
filling the gap from parametric knowledge. This is what the faithfulness
metric in the eval harness actually measures — whether the model honored
that instruction — so the prompt and the eval are two halves of the same
contract.
"""
from __future__ import annotations

from app.config import settings
from app.models.schemas import ScoredChunk

SYSTEM_PROMPT = """You are a retrieval-augmented assistant. Answer the user's question using ONLY the information in the provided context chunks.

Rules:
- If the context does not contain enough information to answer, say so explicitly. Do not use outside knowledge.
- Cite the chunk(s) you used by their [id] after each claim.
- Be concise. Do not repeat the context verbatim; synthesize it.
"""


def _build_context(chunks: list[ScoredChunk]) -> str:
    blocks = []
    for sc in chunks:
        blocks.append(f"[{sc.chunk.chunk_id}] (doc: {sc.chunk.doc_id})\n{sc.chunk.text}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, chunks: list[ScoredChunk]) -> str:
    context = _build_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    if not chunks:
        return "I don't have any relevant context to answer this question."

    if settings.generation_provider == "anthropic":
        return _generate_anthropic(user_prompt)
    if settings.generation_provider == "openai":
        return _generate_openai(user_prompt)
    if settings.generation_provider == "ollama":
        return _generate_ollama(user_prompt)
    raise ValueError(f"Unknown generation provider: {settings.generation_provider}")


def _generate_anthropic(user_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _generate_openai(user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_generation_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=800,
    )
    return resp.choices[0].message.content


def _generate_ollama(user_prompt: str) -> str:
    """
    Calls a local Ollama server's /api/chat endpoint. No API key, no
    per-token cost, but generation quality and latency depend entirely on
    the local model/hardware — a 3B model on CPU is not a drop-in
    replacement for Claude/GPT-4 class quality, especially on the
    grounding/citation instructions in SYSTEM_PROMPT. Use `/v1/evaluate`
    after switching to confirm faithfulness/relevancy hold up, not just
    that responses come back.
    """
    import httpx

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    # Local inference (esp. on CPU) is much slower than a hosted API — a
    # short timeout here would fail on legitimate slow-but-working requests.
    resp = httpx.post(f"{settings.ollama_base_url}/api/chat", json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json()["message"]["content"]
