from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.evaluation.eval_harness import load_eval_set, run_eval, write_report
from app.generation.generator import generate_answer
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_file
from app.models.schemas import (
    Citation,
    DocumentInfo,
    DocumentListResponse,
    EvalReport,
    FileIngestResponse,
    FileIngestResult,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter()

UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB — read/write in chunks rather than
                                   # buffering the whole file in memory


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, request: Request):
    pipeline = request.app.state.pipeline
    chunks = pipeline.ingest(req.doc_id, req.text, req.metadata)
    pipeline.save(settings.index_dir)
    return IngestResponse(doc_id=req.doc_id, num_chunks=len(chunks), chunk_strategy=settings.chunk_strategy)


@router.post("/ingest/file", response_model=FileIngestResponse)
async def ingest_files(request: Request, files: list[UploadFile] = File(...)):
    """
    Accepts one or more uploaded files (multipart/form-data), runs each
    through the same loader registry used for server-side files
    (app/ingestion/loaders/), and ingests the extracted text.

    Two things worth knowing about how this differs from a plain
    "save the upload to disk" endpoint:

    1. The original file is written to UPLOAD_DIR (`./uploads` by default)
       and KEPT — not deleted after extraction. Its path is recorded in
       `metadata["stored_path"]` on every chunk. This means a frontend can
       later offer "view/download the original file," and re-ingestion or
       debugging has the real source to go back to, not just derived text.
    2. The write is streamed in fixed-size chunks (`_UPLOAD_CHUNK_SIZE`),
       with the size limit enforced *during* the write — an oversized
       file is rejected and its partial write deleted as soon as it
       crosses MAX_UPLOAD_SIZE_MB, rather than reading the entire file
       into memory first and only then checking whether it should have
       been rejected.

    Each file is processed independently — one bad/unsupported file in a
    batch doesn't abort the others; check `status`/`error` per result.

    Validation is by extension against SUPPORTED_EXTENSIONS, not by the
    browser-supplied `content_type` header — content_type is unreliable
    (depends on OS/browser MIME mappings, often wrong or missing for
    things like .py/.md) and, more fundamentally, the correct question
    isn't "is this an image" but "do we have a loader for this extension,"
    which is exactly what SUPPORTED_EXTENSIONS answers.
    """
    pipeline = request.app.state.pipeline
    existing_doc_ids = {c.doc_id for c in pipeline.vector_store.chunks}
    results: list[FileIngestResult] = []
    any_success = False
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    for upload in files:
        raw_filename = upload.filename or "unnamed"
        # .name strips any directory components (e.g. "../../etc/cron.d/x")
        # — the multipart filename is fully attacker-controlled, and using
        # it unsanitized in a path join would allow writing outside
        # UPLOAD_DIR entirely.
        filename = Path(raw_filename).name or "unnamed"
        suffix = Path(filename).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            results.append(FileIngestResult(
                filename=filename,
                status="error",
                error=f"Unsupported file type: {suffix or '(none)'}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            ))
            continue

        # Prefix with a short random id so two different uploads of the
        # same filename never collide on disk (the doc_id/index-level
        # duplicate handling below is separate from this filesystem
        # collision concern).
        dest_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
        bytes_written = 0
        rejected_for_size = False

        try:
            with dest_path.open("wb") as buffer:
                while chunk := await upload.read(_UPLOAD_CHUNK_SIZE):
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        rejected_for_size = True
                        break
                    buffer.write(chunk)

            if rejected_for_size:
                dest_path.unlink(missing_ok=True)
                results.append(FileIngestResult(
                    filename=filename,
                    status="error",
                    error=f"File exceeds MAX_UPLOAD_SIZE_MB={settings.max_upload_size_mb} "
                          f"(stopped after {bytes_written / (1024*1024):.1f}MB)",
                ))
                continue

            text, metadata = load_file(dest_path)
            doc_id = Path(filename).stem
            metadata["original_filename"] = filename
            metadata["stored_path"] = str(dest_path)
            is_duplicate = doc_id in existing_doc_ids

            chunks = pipeline.ingest(doc_id, text, metadata)
            existing_doc_ids.add(doc_id)
            any_success = True

            results.append(FileIngestResult(
                filename=filename,
                doc_id=doc_id,
                num_chunks=len(chunks),
                status="success",
                duplicate_doc_id=is_duplicate,
            ))
        except Exception as e:
            # Extraction failed (corrupt/password-protected file, etc.) —
            # don't leave a half-usable file behind on disk for something
            # that was never successfully ingested.
            dest_path.unlink(missing_ok=True)
            results.append(FileIngestResult(filename=filename, status="error", error=str(e)))

    if any_success:
        pipeline.save(settings.index_dir)

    return FileIngestResponse(results=results)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(request: Request):
    """Lists every doc_id currently indexed, with chunk counts — useful for
    confirming an upload landed, or spotting an accidental duplicate ingest
    before it skews retrieval."""
    pipeline = request.app.state.pipeline
    counts: dict[str, int] = {}
    for c in pipeline.vector_store.chunks:
        counts[c.doc_id] = counts.get(c.doc_id, 0) + 1

    documents = [DocumentInfo(doc_id=doc_id, num_chunks=n) for doc_id, n in sorted(counts.items())]
    return DocumentListResponse(documents=documents, total_chunks=sum(counts.values()))


@router.get("/documents/{doc_id}/original")
def get_original_file(doc_id: str, request: Request):
    """
    Serves back the original file for a doc_id that was ingested via
    /ingest/file — the counterpart to persisting uploads in UPLOAD_DIR
    instead of deleting them after extraction. Lets a frontend offer
    "view/download the original document" next to a citation.

    Docs ingested via POST /ingest (raw text, no file) have no original
    to serve — 404 in that case, distinctly from "doc_id doesn't exist
    at all," so a frontend can tell the two apart if it cares to.
    """
    pipeline = request.app.state.pipeline
    doc_chunks = [c for c in pipeline.vector_store.chunks if c.doc_id == doc_id]
    if not doc_chunks:
        raise HTTPException(status_code=404, detail=f"No document found with doc_id '{doc_id}'")

    stored_path = doc_chunks[0].metadata.get("stored_path")
    if not stored_path or not Path(stored_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"No original file stored for doc_id '{doc_id}' (it may have been ingested as raw text via /ingest)",
        )

    original_filename = doc_chunks[0].metadata.get("original_filename", doc_id)
    return FileResponse(stored_path, filename=original_filename)


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request):
    pipeline = request.app.state.pipeline
    retrieved = pipeline.retrieve(req.question, final_top_k=req.top_k, rerank=req.rerank)

    if not retrieved:
        raise HTTPException(status_code=404, detail="No indexed documents to search. Ingest documents first.")

    answer = generate_answer(req.question, retrieved)
    citations = [
        Citation(chunk_id=sc.chunk.chunk_id, doc_id=sc.chunk.doc_id, text=sc.chunk.text, score=sc.score)
        for sc in retrieved
    ]
    return QueryResponse(
        question=req.question,
        answer=answer,
        citations=citations,
        retrieval_debug={"num_candidates": len(retrieved), "reranked": settings.rerank_enabled if req.rerank is None else req.rerank},
    )


@router.post("/evaluate", response_model=EvalReport)
def evaluate(request: Request, eval_set_path: str = "data/eval_set.json", k: int | None = None, judge_generation: bool = True):
    pipeline = request.app.state.pipeline
    try:
        eval_set = load_eval_set(eval_set_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Eval set not found at {eval_set_path}")

    report = run_eval(pipeline, eval_set, k=k, judge_generation=judge_generation)
    write_report(report, out_dir="./eval_reports")
    return report
