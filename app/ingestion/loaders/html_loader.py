"""
HTML — strips tags/scripts/styles and pulls visible text, using Python's
built-in html.parser backend (no lxml dependency needed). Not a full
readability/boilerplate-removal pass (no nav/footer stripping beyond
script/style) — good enough for a saved article or internal wiki export,
less good for a page dense with navigation chrome.
"""
from __future__ import annotations

from pathlib import Path

HTML_EXTENSIONS = {".html", ".htm"}


def load_html(path: Path) -> tuple[str, dict]:
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else None
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    metadata = {"source_path": str(path), "filename": path.name, "format": "html"}
    if title:
        metadata["title"] = title
    return text, metadata
