"""
Fuse vector and keyword result lists via Reciprocal Rank Fusion (RRF).

RRF instead of a weighted score sum on purpose: BM25 scores and cosine
similarities live on incomparable scales (BM25 is an unbounded
corpus-dependent score; cosine sim is bounded [-1, 1]), so any fixed
linear weighting between them is a magic number that breaks when corpus
size or query type shifts. RRF sidesteps this by fusing on *rank*, not
raw score: score(d) = sum over lists of 1 / (k + rank(d)). It's the
standard choice in production hybrid search (Elastic, Weaviate, etc.)
for exactly this reason.
"""
from __future__ import annotations

from app.models.schemas import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[Chunk, float]]],
    k: int = 60,
) -> list[ScoredChunk]:
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for results in result_lists:
        for rank, (chunk, _orig_score) in enumerate(results):
            fused_scores.setdefault(chunk.chunk_id, 0.0)
            fused_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)
            chunk_by_id[chunk.chunk_id] = chunk

    ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        ScoredChunk(chunk=chunk_by_id[cid], score=score, source="hybrid")
        for cid, score in ranked
    ]
