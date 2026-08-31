from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrievable unit produced by the ingestion pipeline."""
    chunk_id: str
    doc_id: str
    text: str
    position: int                      # order within source document
    token_count: int
    metadata: dict = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    source: str                         # "vector" | "keyword" | "hybrid" | "reranked"


class IngestRequest(BaseModel):
    doc_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    num_chunks: int
    chunk_strategy: str


class FileIngestResult(BaseModel):
    filename: str
    doc_id: Optional[str] = None
    num_chunks: Optional[int] = None
    status: str                         # "success" | "error"
    error: Optional[str] = None
    duplicate_doc_id: bool = False      # true if this doc_id was already indexed (chunks appended, not replaced)


class FileIngestResponse(BaseModel):
    results: list[FileIngestResult]


class DocumentInfo(BaseModel):
    doc_id: str
    num_chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    rerank: Optional[bool] = None
    filters: dict = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieval_debug: dict = Field(default_factory=dict)


class EvalExample(BaseModel):
    """One row of an evaluation set: a question with known-good ground truth."""
    question: str
    ground_truth_doc_ids: list[str]     # which source docs *should* be retrieved
    reference_answer: str               # a correct, human-written answer


class EvalReport(BaseModel):
    num_examples: int
    retrieval: dict
    generation: dict
    per_example: list[dict]
