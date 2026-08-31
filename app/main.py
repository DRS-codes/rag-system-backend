from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.routes import router
from app.api.upload_page import UPLOAD_PAGE_HTML
from app.config import settings
from app.ingestion.loader import load_directory
from app.retrieval.pipeline import RagPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = RagPipeline()

    if RagPipeline.exists(settings.index_dir):
        pipeline.load(settings.index_dir)
        print(f"[startup] Loaded existing index from {settings.index_dir} "
              f"({len(pipeline.vector_store.chunks)} chunks)")
    else:
        print("[startup] No existing index found. Auto-ingesting data/sample_docs/ ...")
        for doc_id, text, metadata in load_directory("data/sample_docs"):
            chunks = pipeline.ingest(doc_id, text, metadata)
            print(f"  ingested {doc_id}: {len(chunks)} chunks")
        pipeline.save(settings.index_dir)

    app.state.pipeline = pipeline
    yield
    # no explicit teardown needed — FAISS index and BM25 are in-memory only


app = FastAPI(
    title="RAG System",
    description="Retrieval-augmented generation API with hybrid search, reranking, and a built-in evaluation harness.",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins = ["*"] if settings.cors_origins == "*" else [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your actual origin(s) before deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/v1")


@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    """
    A working file-upload + query test page. Exists because Swagger's
    /docs UI can't render a file picker for multi-file uploads (see
    app/api/upload_page.py docstring) — use this instead when you want to
    test /v1/ingest/file or /v1/query by hand without curl.
    """
    return UPLOAD_PAGE_HTML


@app.get("/")
def root():
    return {
        "service": "rag-system",
        "docs": "/docs",
        "upload_test_page": "/upload",
        "endpoints": ["/v1/health", "/v1/ingest", "/v1/ingest/file", "/v1/documents", "/v1/query", "/v1/evaluate"],
    }
