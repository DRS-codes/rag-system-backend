# API Reference — for frontend development

Everything a frontend needs: every endpoint, exact request/response shapes,
error formats, TypeScript types, and the UX gotchas that aren't obvious
from the schema alone (which calls are slow, which endpoints partially
succeed, how errors are shaped differently depending on *why* they failed).

---

## Base URL & CORS

```
http://localhost:8000
```

All API endpoints are under the `/v1` prefix (e.g. `/v1/query`). Two
non-API pages exist outside that prefix — see [Non-API routes](#non-api-routes)
at the bottom.

**CORS is enabled** (`app/main.py`, via `CORSMiddleware`) so a frontend
running on a different origin/port (e.g. `http://localhost:3000` for a
React dev server) can call this API directly from the browser. Controlled
by `CORS_ORIGINS` in `.env` — defaults to `*` (any origin) for local dev.
**Before deploying, set it to your actual frontend origin(s)**:
```
CORS_ORIGINS=http://localhost:3000,https://myapp.example.com
```

No authentication currently exists on any endpoint — anyone who can reach
the server can call every route. Fine for local dev; not something to
expose publicly as-is.

---

## Quick reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/health` | Liveness check |
| `POST` | `/v1/ingest` | Ingest raw text (no file) |
| `POST` | `/v1/ingest/file` | Upload one or more files |
| `GET` | `/v1/documents/{doc_id}/original` | Download the original file for a doc_id |
| `GET` | `/v1/documents` | List what's currently indexed |
| `POST` | `/v1/query` | Ask a question, get a grounded answer + citations |
| `POST` | `/v1/evaluate` | Run the eval harness, get metrics |

---

## `GET /v1/health`

Liveness check — no auth, no dependencies checked (doesn't verify the
embedding model loaded or the vector store is healthy, just that the
process is up).

**Response `200`:**
```json
{ "status": "ok" }
```

---

## `POST /v1/ingest`

Ingest raw text directly (no file involved) — useful for text a user
pastes/types in the UI rather than uploads.

**Request body:**
```json
{
  "doc_id": "my_doc",
  "text": "The full text to index...",
  "metadata": { "source": "manual" }
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `doc_id` | string | yes | Used as the citation source name and for dedup checks. No uniqueness enforced — see [Duplicate doc_id](#duplicate-doc_id-behavior) below. |
| `text` | string | yes | Raw text, any length (gets chunked server-side). |
| `metadata` | object | no | Freeform, attached to every chunk from this doc. Defaults to `{}`. |

**Response `200`:**
```json
{
  "doc_id": "my_doc",
  "num_chunks": 4,
  "chunk_strategy": "recursive"
}
```

**Errors:** `422` if `doc_id`/`text` are missing or wrong type (standard
FastAPI validation error — see [Error shapes](#error-shapes)).

---

## `POST /v1/ingest/file`

Upload one or more files (`multipart/form-data`). This is the endpoint
for anything a user actually uploads — PDF, docx, csv, xlsx, html, json,
code files, `.txt`/`.md`, or WhatsApp-style chat exports (auto-detected).
See the main README's format table for the full supported list.

**The original file is kept, not deleted.** After extraction, the
uploaded file is persisted server-side (`UPLOAD_DIR`, `./uploads` by
default) — that's what makes `GET /v1/documents/{doc_id}/original` below
work. Validation is by file extension against the actual set of formats
this backend can parse, not by the browser's `content_type` guess (which
is unreliable and doesn't match this system's format list anyway).

**Request:** `multipart/form-data`, field name **must be `files`**
(repeated for multiple files):

```js
const formData = new FormData();
for (const file of fileInputElement.files) {
  formData.append('files', file);   // field name must be exactly "files"
}

const res = await fetch('http://localhost:8000/v1/ingest/file', {
  method: 'POST',
  body: formData,
  // do NOT set Content-Type manually — the browser sets the multipart
  // boundary automatically; setting it yourself breaks the request
});
```

**Response `200`** — always `200` even if individual files failed, see
[Partial success](#partial-success-on-ingestfile) below:

```json
{
  "results": [
    {
      "filename": "resume.pdf",
      "doc_id": "resume",
      "num_chunks": 3,
      "status": "success",
      "error": null,
      "duplicate_doc_id": false
    },
    {
      "filename": "virus.exe",
      "doc_id": null,
      "num_chunks": null,
      "status": "error",
      "error": "Unsupported file type: .exe. Supported: ['.c', '.cpp', ...]",
      "duplicate_doc_id": false
    }
  ]
}
```

**Per-result fields:**

| Field | Type | Notes |
|---|---|---|
| `filename` | string | Original filename as uploaded |
| `doc_id` | string \| null | Filename minus extension; `null` if this file errored before extraction. This is also the id you use to build the download URL: `/v1/documents/{doc_id}/original` |
| `num_chunks` | number \| null | `null` on error |
| `status` | `"success"` \| `"error"` | **Check this per file** — don't infer success from the outer HTTP status |
| `error` | string \| null | Human-readable reason, only present when `status: "error"` |
| `duplicate_doc_id` | boolean | `true` if this `doc_id` was already indexed before this upload — see below |

**Rejection reasons you'll see in `error`:**
- Unsupported extension (not in the supported-formats list)
- File exceeds `MAX_UPLOAD_SIZE_MB` (20MB by default) — enforced *during*
  upload (streamed in 1MB chunks with a running total), not after
  buffering the whole file, so a huge file is rejected without the server
  ever holding all of it in memory at once
- Extraction failure (e.g. a corrupt/password-protected PDF)

**Filenames are sanitized server-side** — only the base filename is used
(any `../` path-traversal attempt in a filename is stripped down to just
the final component before it touches the filesystem). Not something a
frontend needs to handle, just worth knowing nothing unusual happens if a
user's file happens to have an odd name.

---

## `GET /v1/documents/{doc_id}/original`

Downloads the **original file** for a `doc_id`, exactly as uploaded — not
the extracted text, the actual file bytes. This is what makes "view/
download the source document" possible next to a citation in the UI.

```js
// direct link/download — no fetch() needed for a simple download button:
<a href="http://localhost:8000/v1/documents/resume/original" download>Download original</a>

// or fetch it if you need to do something with the bytes first:
const res = await fetch(`http://localhost:8000/v1/documents/${docId}/original`);
if (res.ok) {
  const blob = await res.blob();
  // e.g. render inline: URL.createObjectURL(blob)
}
```

**Response `200`:** the raw file, with `Content-Disposition` set to the
original filename (so a browser download defaults to the right name) and
an appropriate `Content-Type` inferred from the file.

**Response `404`** (plain-string `detail`, see [Error shapes](#error-shapes)) in two distinct cases:
- `doc_id` doesn't exist at all
- `doc_id` exists but was ingested via `POST /v1/ingest` (raw text, no
  file involved) — there's no original to serve. Worth handling
  differently in the UI ("no source file for this entry" vs. "document
  not found") if it matters to your design, since both come back as 404
  but the `detail` message text differs.

---

## `GET /v1/documents`

Lists every document currently in the index — the way to populate a
"your documents" panel in the UI, or to confirm an upload actually landed.

**Response `200`:**
```json
{
  "documents": [
    { "doc_id": "employee_handbook", "num_chunks": 4 },
    { "doc_id": "resume", "num_chunks": 3 }
  ],
  "total_chunks": 7
}
```

Sorted alphabetically by `doc_id`. There's currently no per-document
metadata returned here (upload time, file type, size) — only `doc_id` and
`num_chunks`. If the UI needs more than that, that's a backend addition,
not something to fake on the frontend. (You *can* infer whether a
document has a downloadable original by just trying
`GET /v1/documents/{doc_id}/original` and handling a 404 — there's no
separate "has_original: true/false" flag in this response to check first.)

---

## `POST /v1/query`

Ask a question against everything currently indexed. This is the
generation call — expect real latency (see [Latency expectations](#latency-expectations-what-to-show-a-spinner-for)).

**Request body:**
```json
{
  "question": "What is my education qualification?",
  "top_k": null,
  "rerank": null,
  "filters": {}
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | yes | |
| `top_k` | number \| null | no | Override how many chunks reach the LLM (default from server config, `FINAL_TOP_K`, usually 4). Leave `null`/omit to use the server default. |
| `rerank` | boolean \| null | no | Override whether cross-encoder reranking runs for this query. `null`/omit uses server default (`RERANK_ENABLED`). |
| `filters` | object | no | **Accepted but not currently used by retrieval** — present in the schema for future metadata filtering, don't build UI around it yet. |

**Response `200`:**
```json
{
  "question": "What is my education qualification?",
  "answer": "Based on the retrieved context, you hold a B.Tech in Computer Science [resume::a1b2c3d4]...",
  "citations": [
    {
      "chunk_id": "resume::a1b2c3d4",
      "doc_id": "resume",
      "text": "Education: B.Tech in Computer Science, XYZ University, 2020...",
      "score": 0.83
    }
  ],
  "retrieval_debug": {
    "num_candidates": 4,
    "reranked": true
  }
}
```

**`citations`** is the key field for building a "sources" UI — each entry
has the exact chunk text used, its `doc_id` (so you can link back to
"resume.pdf" in the document list), and a relevance `score`. The `answer`
text itself contains inline `[chunk_id]`-style citation markers from the
LLM — you can leave them as plain text, or parse them and cross-reference
against the `citations` array to build clickable inline citations (the
`chunk_id` format is stable: `{doc_id}::{8-char hex}`).

**Errors:**
- **`404`** — no documents indexed at all:
  ```json
  { "detail": "No indexed documents to search. Ingest documents first." }
  ```
  Show this as "upload something first," not a generic error.
- **`422`** — `question` missing/wrong type.
- **`500`** — generation call failed (bad/missing API key for the
  configured provider, Ollama not running, network issue). Comes back as
  FastAPI's generic unhandled-exception shape (not a clean `detail`
  string) — treat any `500` here as "something went wrong, try again" and
  don't try to parse specifics out of it.

---

## `POST /v1/evaluate`

Runs the eval harness end-to-end and returns aggregate + per-question
metrics. This is a **developer/debug tool**, not something end users of a
consumer frontend would typically trigger — if you're building an
internal/admin view, this is the endpoint for an "eval dashboard."

**Query parameters** (not a JSON body — these go in the URL):

| Param | Type | Default | Notes |
|---|---|---|---|
| `eval_set_path` | string | `"data/eval_set.json"` | Server-side file path, not an upload |
| `k` | number \| null | server default | Overrides `top_k` for this run |
| `judge_generation` | boolean | `true` | If `false`, skips the LLM-as-judge step (faster, retrieval metrics only) |

```js
await fetch('http://localhost:8000/v1/evaluate?judge_generation=true', { method: 'POST' });
```

**Response `200`:**
```json
{
  "num_examples": 10,
  "retrieval": {
    "precision@k": 0.85,
    "recall@k": 0.9,
    "mrr": 0.88,
    "ndcg@k": 0.87
  },
  "generation": {
    "faithfulness": 0.92,
    "answer_relevancy": 0.89,
    "judged_examples": 10
  },
  "per_example": [
    {
      "question": "...",
      "ground_truth_doc_ids": ["..."],
      "retrieved_doc_ids": ["..."],
      "retrieval": { "precision@k": 1.0, "recall@k": 1.0, "mrr": 1.0, "ndcg@k": 1.0 },
      "answer": "...",
      "reference_answer": "...",
      "faithfulness": { "score": 0.95, "claims": [...] },
      "answer_relevancy": { "score": 0.9, "reasoning": "..." }
    }
  ]
}
```

**This is slow** — one full generation call *and* two judge calls per eval
question. 10 questions with a hosted LLM can take 30–60+ seconds; with a
local Ollama model, longer. **Disable the trigger button while this is in
flight and show a progress/loading state**, not a spinner that implies
sub-second completion.

`generation.faithfulness`/`generation.answer_relevancy` can be `null` if
every judge call failed to parse (more likely with a small local Ollama
model) — handle `null` distinctly from `0`, they mean different things
("couldn't measure" vs. "measured and it's bad").

---

## Error shapes

FastAPI produces **two different `detail` shapes** depending on the error
type — a frontend needs to handle both:

**1. Explicit errors raised by this app** (e.g. the `404` on empty-index
query) — `detail` is a plain string:
```json
{ "detail": "No indexed documents to search. Ingest documents first." }
```

**2. Request validation errors** (missing/wrong-type fields, `422`) —
`detail` is an array of field-level errors:
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "question"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

A safe frontend pattern:
```js
function extractErrorMessage(errorBody) {
  if (typeof errorBody.detail === 'string') return errorBody.detail;
  if (Array.isArray(errorBody.detail)) {
    return errorBody.detail.map(e => `${e.loc?.join('.')}: ${e.msg}`).join('; ');
  }
  return 'Something went wrong.';
}
```

**3. Unhandled exceptions (`500`)** — generation/eval failures (bad API
key, provider unreachable) aren't currently caught and reformatted, so
they surface as FastAPI's default error response. Don't try to parse
specifics from a `500` — show a generic retry message.

---

## Partial success on `/ingest/file`

The endpoint **always returns `200`** even when some or all uploaded
files failed — check `results[i].status` per file, not the outer HTTP
status. This is deliberate: uploading 5 files where 1 has an unsupported
extension shouldn't fail the other 4. Build the upload UI to show a
per-file result list, not a single pass/fail toast.

## Duplicate doc_id behavior

`doc_id` is derived from the filename (minus extension) — uploading
`resume.pdf` twice creates **two overlapping sets of chunks** under the
same `doc_id`, not a replacement. The second upload's result has
`duplicate_doc_id: true`, which the frontend should surface as a warning
("this looks like a re-upload — chunks will be added, not replaced") since
there's currently no update/delete-by-doc_id endpoint to do the "right"
thing automatically.

## Latency expectations (what to show a spinner for)

| Call | Typical latency | UI treatment |
|---|---|---|
| `/health` | instant | none needed |
| `/documents` | instant (in-memory) | none needed |
| `/ingest` (raw text) | fast — chunking + local embedding | brief spinner |
| `/ingest/file` | depends on file size; embedding is local/fast, extraction (esp. PDF/docx) adds a bit | spinner, per-file if uploading many |
| `/query` | **1–10+ seconds** — retrieval is fast, but the generation call (hosted LLM or local Ollama) dominates | dedicated loading state, disable the ask button while in flight |
| `/evaluate` | **30s–several minutes** depending on eval set size and provider | progress indicator, explicitly not a quick action |

---

## TypeScript types

Mirrors `app/models/schemas.py` exactly — copy directly into the frontend.

```typescript
interface Chunk {
  chunk_id: string;
  doc_id: string;
  text: string;
  position: number;
  token_count: number;
  metadata: Record<string, unknown>;
}

interface ScoredChunk {
  chunk: Chunk;
  score: number;
  source: "vector" | "keyword" | "hybrid" | "reranked";
}

// POST /v1/ingest
interface IngestRequest {
  doc_id: string;
  text: string;
  metadata?: Record<string, unknown>;
}
interface IngestResponse {
  doc_id: string;
  num_chunks: number;
  chunk_strategy: string;
}

// POST /v1/ingest/file
interface FileIngestResult {
  filename: string;
  doc_id: string | null;
  num_chunks: number | null;
  status: "success" | "error";
  error: string | null;
  duplicate_doc_id: boolean;
}
interface FileIngestResponse {
  results: FileIngestResult[];
}

// GET /v1/documents
interface DocumentInfo {
  doc_id: string;
  num_chunks: number;
}
interface DocumentListResponse {
  documents: DocumentInfo[];
  total_chunks: number;
}

// POST /v1/query
interface QueryRequest {
  question: string;
  top_k?: number | null;
  rerank?: boolean | null;
  filters?: Record<string, unknown>;
}
interface Citation {
  chunk_id: string;
  doc_id: string;
  text: string;
  score: number;
}
interface QueryResponse {
  question: string;
  answer: string;
  citations: Citation[];
  retrieval_debug: Record<string, unknown>;
}

// POST /v1/evaluate
interface EvalReport {
  num_examples: number;
  retrieval: {
    "precision@k": number;
    "recall@k": number;
    mrr: number;
    "ndcg@k": number;
  };
  generation: {
    faithfulness: number | null;
    answer_relevancy: number | null;
    judged_examples: number;
  } | Record<string, never>;  // empty object if judge_generation=false
  per_example: Record<string, unknown>[];
}

// Error shapes
interface SimpleError {
  detail: string;
}
interface ValidationError {
  detail: Array<{ type: string; loc: (string | number)[]; msg: string; input: unknown }>;
}
```

---

## Suggested frontend flow

A minimal end-to-end UI needs three screens/panels, in this order of
build priority:

1. **Upload panel** — file picker (multi-file) → `POST /v1/ingest/file` →
   render per-file results (success/error/duplicate) from the response.
   Refresh the document list after any successful upload.
2. **Document list panel** — `GET /v1/documents` on load and after every
   upload. Simple table: `doc_id`, `num_chunks`.
3. **Query panel** — text input → `POST /v1/query` → render `answer`, and
   render `citations` as an expandable "sources" section (each citation's
   `text` is the exact chunk the model saw — good for a "why did it say
   that" affordance).

An eval dashboard (`POST /v1/evaluate` + rendering the metrics table) is a
reasonable v2/admin-only addition, not part of the core user-facing flow.

---

## Non-API routes

Not part of the `/v1` API surface, but relevant while developing:

| Route | Purpose |
|---|---|
| `GET /` | Service info — lists available endpoints as JSON |
| `GET /docs` | Auto-generated Swagger UI. **Cannot properly test `/v1/ingest/file`** — Swagger UI has a long-standing limitation rendering file pickers for multi-file (array) upload params, falls back to a broken text-array editor. Don't build UX expectations around this page; it's for browsing request/response schemas, not testing uploads. |
| `GET /upload` | A minimal built-in test page with a real file picker, for manually verifying the API works before your frontend is ready. Good for isolating "is this a backend problem or a frontend problem" while debugging. |
