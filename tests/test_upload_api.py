"""
Tests for the /v1/ingest/file and /v1/documents endpoints. Uses FastAPI's
TestClient, which runs the app in-process (real request/response cycle
through routing, validation, and the actual RagPipeline — not a mock),
but does trigger real model loading (sentence-transformers) on the
lifespan startup, so these are slower than the pure-logic tests in
test_loaders.py and require the full requirements.txt installed.
"""
import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_pdf_ingests_successfully(client):
    with open("data/format_samples/incident_runbook.pdf", "rb") as f:
        resp = client.post(
            "/v1/ingest/file",
            files={"files": ("incident_runbook.pdf", f, "application/pdf")},
        )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert results[0]["doc_id"] == "incident_runbook"
    assert results[0]["num_chunks"] > 0
    assert results[0]["duplicate_doc_id"] is False


def test_upload_same_file_twice_flags_duplicate(client):
    with open("data/format_samples/service_config.json", "rb") as f:
        client.post("/v1/ingest/file", files={"files": ("service_config.json", f, "application/json")})

    with open("data/format_samples/service_config.json", "rb") as f:
        resp = client.post("/v1/ingest/file", files={"files": ("service_config.json", f, "application/json")})

    result = resp.json()["results"][0]
    assert result["status"] == "success"
    assert result["duplicate_doc_id"] is True


def test_upload_unsupported_extension_returns_error_not_500(client):
    fake_file = io.BytesIO(b"not a real executable, just bytes")
    resp = client.post(
        "/v1/ingest/file",
        files={"files": ("virus.exe", fake_file, "application/octet-stream")},
    )
    assert resp.status_code == 200  # per-file errors, not a request-level failure
    result = resp.json()["results"][0]
    assert result["status"] == "error"
    assert "Unsupported file type" in result["error"]


def test_upload_multiple_files_mixed_success_and_failure(client):
    files = [
        ("files", ("service_config.json", open("data/format_samples/service_config.json", "rb"), "application/json")),
        ("files", ("bad.exe", io.BytesIO(b"junk"), "application/octet-stream")),
    ]
    resp = client.post("/v1/ingest/file", files=files)
    results = resp.json()["results"]
    assert len(results) == 2
    statuses = {r["filename"]: r["status"] for r in results}
    assert statuses["service_config.json"] == "success"
    assert statuses["bad.exe"] == "error"


def test_documents_endpoint_lists_ingested_docs(client):
    with open("data/format_samples/pricing_utils.py", "rb") as f:
        client.post("/v1/ingest/file", files={"files": ("pricing_utils.py", f, "text/x-python")})

    resp = client.get("/v1/documents")
    assert resp.status_code == 200
    body = resp.json()
    doc_ids = [d["doc_id"] for d in body["documents"]]
    assert "pricing_utils" in doc_ids
    assert body["total_chunks"] >= 1


# ---------- original file persistence + retrieval ----------

def test_uploaded_original_is_kept_on_disk_and_downloadable(client):
    with open("data/format_samples/incident_runbook.pdf", "rb") as f:
        client.post("/v1/ingest/file", files={"files": ("incident_runbook.pdf", f, "application/pdf")})

    resp = client.get("/v1/documents/incident_runbook/original")
    assert resp.status_code == 200
    assert resp.content.startswith(b"%PDF")  # got the real original file back, not derived text
    assert "attachment" in resp.headers.get("content-disposition", "") or resp.headers.get("content-type")


def test_original_file_404_for_raw_text_ingest(client):
    client.post("/v1/ingest", json={"doc_id": "raw_text_doc", "text": "some raw text, no file involved"})
    resp = client.get("/v1/documents/raw_text_doc/original")
    assert resp.status_code == 404


def test_original_file_404_for_unknown_doc_id(client):
    resp = client.get("/v1/documents/totally_made_up_doc_id/original")
    assert resp.status_code == 404


def test_upload_filename_with_path_traversal_is_sanitized(client):
    """A malicious filename shouldn't be able to write outside UPLOAD_DIR —
    this exercises the same Path(filename).name sanitization used in
    routes.py, via a real request rather than calling the helper directly."""
    fake_file = io.BytesIO(b"# just a python comment\nx = 1\n")
    resp = client.post(
        "/v1/ingest/file",
        files={"files": ("../../../tmp/evil_traversal_test.py", fake_file, "text/x-python")},
    )
    assert resp.status_code == 200
    result = resp.json()["results"][0]
    assert result["status"] == "success"
    # doc_id should be derived from the sanitized basename, not the traversal path
    assert result["doc_id"] == "evil_traversal_test"
    # and critically: nothing should have been written to /tmp directly
    assert not __import__("pathlib").Path("/tmp/evil_traversal_test.py").exists()
