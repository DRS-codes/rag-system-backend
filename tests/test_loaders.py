import json
import zipfile
from pathlib import Path

import pytest

from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_directory, load_file
from app.ingestion.loaders.code_loader import load_code
from app.ingestion.loaders.json_loader import load_json
from app.ingestion.loaders.pdf_loader import load_pdf
from app.ingestion.loaders.text_loaders import _looks_like_whatsapp_export, load_text_or_whatsapp

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "format_samples"


# ---------- WhatsApp export detection/parsing ----------

WHATSAPP_SAMPLE = """12/08/23, 9:41 PM - John Doe: Hey, how's it going?
12/08/23, 9:42 PM - Jane Smith: Pretty good! Working on the project.
This is a continuation line, no timestamp.
12/08/23, 9:43 PM - John Doe: Nice, does it handle that correctly?
"""

PLAIN_TEXT_SAMPLE = """This is just a regular text file.
It has multiple lines.
None of them look like chat timestamps.
"""


def test_whatsapp_detection_positive():
    assert _looks_like_whatsapp_export(WHATSAPP_SAMPLE) is True


def test_whatsapp_detection_negative():
    assert _looks_like_whatsapp_export(PLAIN_TEXT_SAMPLE) is False


def test_whatsapp_parsing_extracts_participants_and_folds_continuations(tmp_path):
    p = tmp_path / "chat.txt"
    p.write_text(WHATSAPP_SAMPLE)
    text, meta = load_text_or_whatsapp(p)

    assert meta["format"] == "whatsapp_export"
    assert meta["message_count"] == 3
    assert set(meta["participants"]) == {"John Doe", "Jane Smith"}
    # the continuation line should be folded into Jane's message, not a separate one
    assert "This is a continuation line" in text
    assert text.count("Jane Smith (") == 1


def test_plain_txt_not_misdetected_as_whatsapp(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text(PLAIN_TEXT_SAMPLE)
    text, meta = load_text_or_whatsapp(p)
    assert meta["format"] == "plain_text"
    assert text == PLAIN_TEXT_SAMPLE


# ---------- JSON loader ----------

def test_json_loader_pretty_prints_and_tags_top_level_keys(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"a": 1, "b": [1, 2, 3]}))
    text, meta = load_json(p)
    assert meta["format"] == "json"
    assert meta["top_level_keys"] == ["a", "b"]
    assert json.loads(text) == {"a": 1, "b": [1, 2, 3]}  # round-trips


def test_json_loader_handles_top_level_list(tmp_path):
    p = tmp_path / "items.json"
    p.write_text(json.dumps([{"id": 1}, {"id": 2}, {"id": 3}]))
    _, meta = load_json(p)
    assert meta["num_items"] == 3


# ---------- code loader ----------

def test_code_loader_tags_language_by_extension(tmp_path):
    p = tmp_path / "script.py"
    p.write_text("def f(): pass\n")
    text, meta = load_code(p)
    assert meta["language"] == "python"
    assert meta["format"] == "code"
    assert "def f()" in text


def test_code_loader_unknown_extension_tagged_unknown(tmp_path, monkeypatch):
    # .py etc. are pre-registered; an extension not in CODE_EXTENSIONS at all
    # wouldn't reach load_code via the registry, but load_code itself should
    # still degrade gracefully if called directly.
    p = tmp_path / "file.weirdext"
    p.write_text("some content")
    _, meta = load_code(p)
    assert meta["language"] == "unknown"


# ---------- PDF loader ----------
# Uses the real sample file (a hand-built, text-based PDF — not a mock)
# so this exercises actual pypdf extraction, same as production.

def test_pdf_loader_extracts_text_and_page_count():
    text, meta = load_pdf(SAMPLES_DIR / "incident_runbook.pdf")
    assert meta["format"] == "pdf"
    assert meta["num_pages"] == 1
    assert "acknowledged within 15 minutes" in text
    assert "Incident Response Runbook" in text


def test_pdf_loader_via_registry_dispatch():
    text, meta = load_file(SAMPLES_DIR / "incident_runbook.pdf")
    assert meta["format"] == "pdf"
    assert "blameless" in text


# ---------- registry-level behavior ----------

def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        load_file("nonexistent.exe")


def test_registry_includes_all_new_formats():
    expected = {".csv", ".xlsx", ".xls", ".html", ".htm", ".json", ".docx", ".py", ".txt", ".md"}
    assert expected.issubset(SUPPORTED_EXTENSIONS)


def test_load_directory_skips_unsupported_files(tmp_path):
    (tmp_path / "doc.txt").write_text("hello")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")  # unsupported, should be skipped silently
    results = load_directory(tmp_path)
    doc_ids = [r[0] for r in results]
    assert "doc" in doc_ids
    assert "image" not in doc_ids
    assert len(results) == 1
