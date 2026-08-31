"""
Source code files. Phase 1 reads them as plain text — no AST-aware
splitting yet (that's Phase 2, so functions/classes don't get cut in
half by a token-count chunker). What Phase 1 *does* do is tag every
code chunk with its language in metadata, which is what a future
AST-aware chunker would key off of to pick the right parser per file.
"""
from __future__ import annotations

from pathlib import Path

CODE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "shell",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".css": "css",
    ".scss": "scss",
}


def load_code(path: Path) -> tuple[str, dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    language = CODE_EXTENSIONS.get(path.suffix.lower(), "unknown")
    metadata = {
        "source_path": str(path),
        "filename": path.name,
        "format": "code",
        "language": language,
    }
    return text, metadata
