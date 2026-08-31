"""PDF text extraction via pypdf. No OCR — a scanned/image-only PDF will
extract empty or near-empty text; that needs an OCR step ahead of this
(planned for the images phase), not something this loader does silently.
"""
from __future__ import annotations

from pathlib import Path

PDF_EXTENSIONS = {".pdf"}


def load_pdf(path: Path) -> tuple[str, dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)

    metadata = {
        "source_path": str(path),
        "filename": path.name,
        "format": "pdf",
        "num_pages": len(reader.pages),
    }
    return text, metadata
