"""
A minimal, dependency-free HTML page for testing file upload and query
by hand, served directly by FastAPI (no separate frontend build needed).

This exists specifically because Swagger UI's /docs page cannot render a
proper file-picker for `list[UploadFile]` parameters — it falls back to a
generic string-array text editor instead (a long-standing Swagger UI
limitation for arrays of binary files, not a bug in this app's code; see
https://github.com/swagger-api/swagger-ui/issues/4600, open since 2018).
This page hits the exact same /v1/ingest/file and /v1/query endpoints
with real multipart/JSON requests via fetch(), so it's a faithful test of
the actual API — just with a working file picker.
"""

UPLOAD_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RAG System — Upload &amp; Query</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; color: #1a1a1a; }
  h2 { margin-top: 32px; }
  .note { background: #f0f4ff; border-left: 3px solid #4a6cf7; padding: 10px 14px; font-size: 14px; margin-bottom: 20px; }
  input[type="file"] { display: block; margin: 12px 0; }
  input[type="text"] { width: 100%; padding: 8px; font-size: 14px; box-sizing: border-box; }
  button { background: #4a6cf7; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; margin-top: 8px; }
  button:hover { background: #3a5ce5; }
  pre { background: #f4f4f4; padding: 12px; white-space: pre-wrap; word-break: break-word; border-radius: 4px; font-size: 13px; max-height: 400px; overflow-y: auto; }
  .doc-list { font-size: 14px; }
</style>
</head>
<body>

<h1>RAG System — manual test page</h1>
<div class="note">
  Swagger's <code>/docs</code> page can't render a file picker for multi-file
  uploads (a known Swagger UI limitation, not a bug here) — this page hits
  the real API directly instead.
</div>

<h2>1. Upload documents</h2>
<input type="file" id="fileInput" multiple>
<button onclick="doUpload()">Upload</button>
<pre id="uploadResult">(results appear here)</pre>

<h2>2. See what's indexed</h2>
<button onclick="doListDocs()">Refresh document list</button>
<pre id="docsResult" class="doc-list">(click above)</pre>

<h2>3. Ask a question</h2>
<input type="text" id="questionInput" placeholder="e.g. What is my education qualification?">
<button onclick="doQuery()">Ask</button>
<pre id="queryResult">(answer appears here)</pre>

<script>
async function doUpload() {
  const input = document.getElementById('fileInput');
  const out = document.getElementById('uploadResult');
  if (!input.files.length) { out.textContent = 'Choose at least one file first.'; return; }

  const formData = new FormData();
  for (const file of input.files) formData.append('files', file);

  out.textContent = 'Uploading...';
  try {
    const resp = await fetch('/v1/ingest/file', { method: 'POST', body: formData });
    const data = await resp.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e;
  }
}

async function doListDocs() {
  const out = document.getElementById('docsResult');
  out.textContent = 'Loading...';
  try {
    const resp = await fetch('/v1/documents');
    const data = await resp.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e;
  }
}

async function doQuery() {
  const question = document.getElementById('questionInput').value.trim();
  const out = document.getElementById('queryResult');
  if (!question) { out.textContent = 'Type a question first.'; return; }

  out.textContent = 'Thinking...';
  try {
    const resp = await fetch('/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await resp.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e;
  }
}
</script>
</body>
</html>"""
