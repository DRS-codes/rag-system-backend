"""
The loader registry. Every format module above exports a set/dict of
extensions it handles plus a `load_x(path) -> (text, metadata)` function.
This file is the single place that wires extension -> loader — adding a
new format later means writing one new module and adding two lines here,
nothing else in the codebase (pipeline.py, main.py, routes.py) needs to
change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .code_loader import CODE_EXTENSIONS, load_code
from .docx_loader import DOCX_EXTENSIONS, load_docx
from .html_loader import HTML_EXTENSIONS, load_html
from .json_loader import JSON_EXTENSIONS, load_json
from .pdf_loader import PDF_EXTENSIONS, load_pdf
from .spreadsheet_loader import SPREADSHEET_EXTENSIONS, load_spreadsheet
from .text_loaders import TEXT_EXTENSIONS, load_text_or_whatsapp

LoaderFn = Callable[[Path], tuple[str, dict]]

_REGISTRY: dict[str, LoaderFn] = {}
for ext in TEXT_EXTENSIONS:
    _REGISTRY[ext] = load_text_or_whatsapp
for ext in PDF_EXTENSIONS:
    _REGISTRY[ext] = load_pdf
for ext in DOCX_EXTENSIONS:
    _REGISTRY[ext] = load_docx
for ext in SPREADSHEET_EXTENSIONS:
    _REGISTRY[ext] = load_spreadsheet
for ext in HTML_EXTENSIONS:
    _REGISTRY[ext] = load_html
for ext in JSON_EXTENSIONS:
    _REGISTRY[ext] = load_json
for ext in CODE_EXTENSIONS:
    _REGISTRY[ext] = load_code

SUPPORTED_EXTENSIONS = set(_REGISTRY.keys())


def load_file(path: str | Path) -> tuple[str, dict]:
    path = Path(path)
    loader_fn = _REGISTRY.get(path.suffix.lower())
    if loader_fn is None:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return loader_fn(path)


def load_directory(dir_path: str | Path) -> list[tuple[str, str, dict]]:
    """Returns list of (doc_id, text, metadata) for every supported file in a directory."""
    dir_path = Path(dir_path)
    out = []
    for path in sorted(dir_path.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text, metadata = load_file(path)
        doc_id = path.stem
        out.append((doc_id, text, metadata))
    return out
