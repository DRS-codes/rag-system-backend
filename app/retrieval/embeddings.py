"""
Embedding model choice.

This wraps two providers behind one interface so the rest of the pipeline
never cares which is active:

- local (default): sentence-transformers/all-MiniLM-L6-v2. 384 dims, ~80MB,
  runs on CPU in milliseconds per batch. Chosen as the default because it
  requires no API key, no network round-trip per query, and no per-token
  cost — the right trade for most internal-knowledge-base RAG systems
  where query latency and cost matter more than the last few points of
  retrieval quality. Weaker than large hosted models on nuanced semantic
  similarity and out-of-domain vocabulary (e.g. dense legal/medical text).

- openai (text-embedding-3-small, 1536 dims by default): meaningfully
  better retrieval quality on heterogeneous/long-tail text, at the cost
  of network latency, per-call cost, and an external dependency in the
  ingestion/query hot path. Swap to this when the eval harness shows the
  local model's retrieval recall is the bottleneck, not before —
  upgrading the embedding model is expensive to re-index against, so it
  should be evaluation-driven, not a default.

Both providers are normalized to unit vectors so cosine similarity and
inner-product search behave identically regardless of which is active,
and FAISS can use a plain IndexFlatIP.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import settings


class EmbeddingModel:
    def __init__(self):
        self.provider = settings.embedding_provider
        if self.provider == "local":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.local_embedding_model)
            self.dim = self._model.get_sentence_embedding_dimension()
        elif self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model_name = settings.openai_embedding_model
            # text-embedding-3-small default dim; text-embedding-3-large is 3072.
            self.dim = 1536
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        if self.provider == "local":
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")

        # openai
        resp = self._client.embeddings.create(model=self._model_name, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-8, None)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Process-wide singleton — loading the model is the expensive part."""
    return EmbeddingModel()
