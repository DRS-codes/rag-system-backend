"""
Keyword search via BM25 (Okapi). This is what catches queries dense
embeddings often miss: exact identifiers, error codes, product SKUs,
acronyms, or rare proper nouns that get smoothed away in embedding space
but where an exact token match is exactly what the user needs. Hybrid
search exists because vector and BM25 fail on different query types —
neither alone is sufficient for a production system.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.models.schemas import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class KeywordStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.chunks:
            self._bm25 = None
            return
        corpus = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx if scores[i] > 0]

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "keyword_chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, directory: str) -> "KeywordStore":
        d = Path(directory)
        store = cls()
        with open(d / "keyword_chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        store._rebuild()
        return store

    @staticmethod
    def exists(directory: str) -> bool:
        return (Path(directory) / "keyword_chunks.pkl").exists()
