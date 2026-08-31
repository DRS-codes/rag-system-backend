"""
Backward-compatible facade. The actual per-format loaders live in
app/ingestion/loaders/ (a registry keyed by extension) — this file just
re-exports the two functions the rest of the app calls, so main.py and
pipeline.py don't need to know the registry exists.
"""
from app.ingestion.loaders import (  # noqa: F401
    SUPPORTED_EXTENSIONS,
    load_directory,
    load_file,
)

__all__ = ["load_file", "load_directory", "SUPPORTED_EXTENSIONS"]
