"""
JSON files — pretty-printed rather than minified, so the structure stays
legible to both the chunker (which splits on newlines/brackets more
sanely) and to a human reading a retrieved chunk in a citation.
"""
from __future__ import annotations

import json
from pathlib import Path

JSON_EXTENSIONS = {".json"}


def load_json(path: Path) -> tuple[str, dict]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    data = json.loads(raw)
    text = json.dumps(data, indent=2, ensure_ascii=False)

    metadata = {"source_path": str(path), "filename": path.name, "format": "json"}
    if isinstance(data, dict):
        metadata["top_level_keys"] = list(data.keys())[:20]
    elif isinstance(data, list):
        metadata["num_items"] = len(data)

    return text, metadata
