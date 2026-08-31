"""
Chunking strategies.

Chunking is where most naive RAG systems quietly lose quality: chunk too
large and retrieval precision drops (irrelevant text dilutes the embedding
and gets passed to the LLM); chunk too small and you lose the context a
question needs (an answer split across two chunks becomes unretrievable
as a single unit). This module implements three strategies with different
trade-offs, all sharing a token-aware base so chunk sizes are meaningful
regardless of the embedding model's context window.

- FixedSizeChunker: naive sliding window over tokens. Fast, predictable,
  but blind to sentence/paragraph boundaries — will happily cut a
  sentence in half. Useful as a baseline to prove the smarter strategies
  actually help (see the eval harness).

- RecursiveChunker: splits on a hierarchy of separators (paragraph ->
  sentence -> word), only falling back to a harder split when a unit is
  still too large. This is the strategy most production systems default
  to (same idea as LangChain's RecursiveCharacterTextSplitter) because it
  keeps chunks semantically coherent without needing an embedding call at
  chunk time.

- SemanticChunker: embeds consecutive sentences and cuts a new chunk
  wherever cosine similarity between adjacent sentences drops below a
  threshold — i.e. splits at genuine topic boundaries rather than a
  fixed length. Costs an embedding pass at ingestion time; pays off most
  on long, topically heterogeneous documents (e.g. a wiki page that
  covers five subtopics).
"""
from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod

import numpy as np
import tiktoken

from app.models.schemas import Chunk

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter — avoids pulling in a heavy NLP dependency
    # for something regex handles well enough for chunking purposes.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class BaseChunker(ABC):
    name: str

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, text: str) -> list[str]:
        ...

    def chunk_document(self, doc_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        pieces = self.split(text)
        chunks = []
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{uuid.uuid4().hex[:8]}",
                    doc_id=doc_id,
                    text=piece,
                    position=i,
                    token_count=count_tokens(piece),
                    metadata={**metadata, "chunk_strategy": self.name},
                )
            )
        return chunks


class FixedSizeChunker(BaseChunker):
    """Sliding window over raw tokens. No awareness of sentence boundaries."""
    name = "fixed"

    def split(self, text: str) -> list[str]:
        tokens = _ENCODER.encode(text)
        if not tokens:
            return []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        pieces = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.chunk_size]
            pieces.append(_ENCODER.decode(window))
            if start + self.chunk_size >= len(tokens):
                break
        return pieces


class RecursiveChunker(BaseChunker):
    """
    Splits on a separator hierarchy (paragraphs -> sentences -> words),
    greedily packing units up to chunk_size tokens, and only descending to
    a finer-grained separator when a unit alone exceeds chunk_size.
    Adjacent chunks share `chunk_overlap` tokens of trailing context.
    """
    name = "recursive"
    _separators = ["\n\n", "\n", ". ", " "]

    def split(self, text: str) -> list[str]:
        units = self._recursive_split(text.strip(), self._separators)
        return self._pack(units)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if count_tokens(text) <= self.chunk_size:
            return [text] if text else []
        if not separators:
            # Last resort: hard token cut.
            tokens = _ENCODER.encode(text)
            return [
                _ENCODER.decode(tokens[i : i + self.chunk_size])
                for i in range(0, len(tokens), self.chunk_size)
            ]

        sep, rest = separators[0], separators[1:]
        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) == 1:
            # Separator didn't help, fall through to the next one.
            return self._recursive_split(text, rest)

        out = []
        for part in parts:
            out.extend(self._recursive_split(part.strip(), rest))
        return out

    def _pack(self, units: list[str]) -> list[str]:
        """Greedily combine small units up to chunk_size, with token overlap."""
        if not units:
            return []
        chunks, current, current_tokens = [], [], 0
        for unit in units:
            ut = count_tokens(unit)
            if current and current_tokens + ut > self.chunk_size:
                chunks.append(" ".join(current))
                # carry trailing overlap into the next chunk
                overlap_units, overlap_tok = [], 0
                for u in reversed(current):
                    t = count_tokens(u)
                    if overlap_tok + t > self.chunk_overlap:
                        break
                    overlap_units.insert(0, u)
                    overlap_tok += t
                current, current_tokens = overlap_units.copy(), overlap_tok
            current.append(unit)
            current_tokens += ut
        if current:
            chunks.append(" ".join(current))
        return chunks


class SemanticChunker(BaseChunker):
    """
    Embeds sentences and cuts chunks at points where adjacent-sentence
    cosine similarity drops below `threshold` — i.e. at genuine topic
    shifts rather than an arbitrary length. Falls back to packing by
    chunk_size within a topic segment so chunks don't grow unbounded.

    Requires an embedding function at construction time since this is the
    one strategy that needs vectors during ingestion, not just at query
    time.
    """
    name = "semantic"

    def __init__(self, chunk_size: int, chunk_overlap: int, embed_fn, threshold: float = 0.55):
        super().__init__(chunk_size, chunk_overlap)
        self.embed_fn = embed_fn
        self.threshold = threshold

    def split(self, text: str) -> list[str]:
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        embeddings = np.array(self.embed_fn(sentences))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.clip(norms, 1e-8, None)
        sims = np.sum(normed[:-1] * normed[1:], axis=1)  # adjacent cosine sim

        segments, current = [], [sentences[0]]
        for i, sim in enumerate(sims):
            if sim < self.threshold:
                segments.append(current)
                current = []
            current.append(sentences[i + 1])
        if current:
            segments.append(current)

        # Within each topic segment, still respect chunk_size via the
        # recursive packer so one huge on-topic segment doesn't become
        # one huge chunk.
        packer = RecursiveChunker(self.chunk_size, self.chunk_overlap)
        pieces = []
        for seg in segments:
            pieces.extend(packer._pack(seg))
        return pieces


def get_chunker(strategy: str, chunk_size: int, chunk_overlap: int, embed_fn=None, threshold: float = 0.55) -> BaseChunker:
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size, chunk_overlap)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size, chunk_overlap)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("SemanticChunker requires an embed_fn")
        return SemanticChunker(chunk_size, chunk_overlap, embed_fn, threshold)
    raise ValueError(f"Unknown chunk strategy: {strategy}")
