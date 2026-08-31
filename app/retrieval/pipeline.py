from __future__ import annotations

from app.config import settings
from app.ingestion.chunking import get_chunker
from app.models.schemas import Chunk, ScoredChunk
from app.retrieval.embeddings import get_embedding_model
from app.retrieval.hybrid_search import reciprocal_rank_fusion
from app.retrieval.keyword_search import KeywordStore
from app.retrieval.reranker import get_reranker
from app.retrieval.vector_store import VectorStore


class RagPipeline:
    """
    Owns the vector store + keyword store and exposes ingest() / retrieve().
    A single process-wide instance is created at app startup (see main.py)
    and persisted to disk so re-ingestion isn't needed between restarts.
    """

    def __init__(self):
        self.embedder = get_embedding_model()
        self.vector_store = VectorStore(self.embedder.dim)
        self.keyword_store = KeywordStore()

        # Warm up the reranker here too, not just the embedder. get_reranker()
        # is @lru_cache'd, so without this it silently defers the CrossEncoder
        # download/load to the first /v1/query call that actually reranks —
        # which can take well over a minute on a cold HF cache and blow past
        # any reasonable client timeout. Loading it during __init__ means the
        # cost is paid once, at process startup, not on a user-facing request.
        self.reranker = get_reranker() if settings.rerank_enabled else None

    # ---------- ingestion ----------

    def ingest(self, doc_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        embed_fn = (lambda texts: self.embedder.embed(texts)) if settings.chunk_strategy == "semantic" else None
        chunker = get_chunker(
            settings.chunk_strategy,
            settings.chunk_size,
            settings.chunk_overlap,
            embed_fn=embed_fn,
            threshold=settings.semantic_chunk_threshold,
        )
        chunks = chunker.chunk_document(doc_id, text, metadata)
        if not chunks:
            return []

        embeddings = self.embedder.embed([c.text for c in chunks])
        self.vector_store.add(chunks, embeddings)
        self.keyword_store.add(chunks)
        return chunks

    # ---------- retrieval ----------

    def retrieve(self, query: str, final_top_k: int | None = None, rerank: bool | None = None) -> list[ScoredChunk]:
        final_top_k = final_top_k or settings.final_top_k
        do_rerank = settings.rerank_enabled if rerank is None else rerank

        query_vec = self.embedder.embed_one(query)
        vector_hits = self.vector_store.search(query_vec, settings.vector_top_k)
        keyword_hits = self.keyword_store.search(query, settings.keyword_top_k)

        fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=settings.rrf_k)
        candidates = fused[: settings.hybrid_top_k]

        # Reuse the warmed-up instance instead of calling get_reranker() again.
        # It's @lru_cache'd so a second call would be cheap anyway, but this
        # also makes it correct if rerank_enabled=False at startup (no
        # self.reranker was built) yet a per-request `rerank=True` override
        # asks for it here — in that edge case we still fall back to
        # get_reranker(), accepting the one-time lazy-load cost for that
        # specific override rather than always warming a model the default
        # config says isn't needed.
        if do_rerank and candidates:
            reranker = self.reranker or get_reranker()
            return reranker.rerank(query, candidates, final_top_k)

        return candidates[:final_top_k]

    # ---------- persistence ----------

    def save(self, directory: str) -> None:
        self.vector_store.save(directory)
        self.keyword_store.save(directory)

    def load(self, directory: str) -> None:
        self.vector_store = VectorStore.load(directory)
        self.keyword_store = KeywordStore.load(directory)

    @staticmethod
    def exists(directory: str) -> bool:
        return VectorStore.exists(directory) and KeywordStore.exists(directory)