"""
Vector store: FAISS IndexFlatIP over normalized embeddings (= cosine
similarity search). Flat index is exact (no ANN approximation error),
which matters for an eval harness — you want retrieval metrics to reflect
the embedding model's quality, not an approximate-search recall ceiling.
Swap to IndexIVFFlat / HNSW only once corpus size makes exact search a
measured latency problem; premature ANN just adds a second source of
recall loss that's hard to disentangle from chunking/embedding choices.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from app.models.schemas import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []  # position i in self.chunks <-> vector i in index

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        assert len(chunks) == embeddings.shape[0]
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.reshape(1, -1).astype("float32"), top_k)
        return [
            (self.chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "faiss.index"))
        with open(d / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(d / "meta.json", "w") as f:
            json.dump({"dim": self.dim, "count": len(self.chunks)}, f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        d = Path(directory)
        with open(d / "meta.json") as f:
            meta = json.load(f)
        store = cls(meta["dim"])
        store.index = faiss.read_index(str(d / "faiss.index"))
        with open(d / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store

    @staticmethod
    def exists(directory: str) -> bool:
        d = Path(directory)
        return (d / "faiss.index").exists() and (d / "chunks.pkl").exists()
