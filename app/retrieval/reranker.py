"""
Re-ranking with a cross-encoder.

Bi-encoders (the embedding model used for vector search) encode query and
document independently, then compare vectors — fast enough to search
millions of chunks, but the model never actually looks at query and
document together. A cross-encoder feeds (query, chunk) pairs through the
model jointly and outputs a relevance score directly, which is
substantially more accurate but too slow to run over an entire corpus.

The standard production pattern, used here: cheap hybrid search narrows
a large corpus to hybrid_top_k candidates (e.g. 10-20), then the
cross-encoder re-scores just those candidates and we keep the top
final_top_k for generation. This buys most of the quality of "score
everything with a cross-encoder" at a fraction of the latency.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.models.schemas import ScoredChunk


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        rescored = [
            ScoredChunk(chunk=c.chunk, score=float(s), source="reranked")
            for c, s in zip(candidates, scores)
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(settings.reranker_model)
