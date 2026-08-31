# RAG System

A retrieval-augmented generation pipeline built to show the parts a toy
"stuff docs into a vector DB" implementation skips: a real chunking
strategy comparison, hybrid (vector + keyword) search, cross-encoder
reranking, and — the part that actually matters in production — an
evaluation harness that measures retrieval quality and answer
faithfulness numerically, so you can tell whether a change helped or hurt
instead of eyeballing a few demo queries.

**Building a frontend against this API?** See
[`API_REFERENCE.md`](./API_REFERENCE.md) — every endpoint's exact
request/response shape, error formats, TypeScript types, and latency/UX
notes (which calls need a spinner, which partially succeed, etc.).

## Architecture

```
Ingestion:  raw doc -> chunker (fixed | recursive | semantic) -> chunks
                                                                     |
                                                    embed (local | OpenAI)
                                                                     |
                                            FAISS vector index  +  BM25 index

Query:      question -> [vector search top-20] -+
                      -> [BM25 search top-20]   -+-> Reciprocal Rank Fusion
                                                       -> top-10 candidates
                                                       -> cross-encoder rerank
                                                       -> top-4 final chunks
                                                       -> LLM generates answer
                                                          (grounded, cited)

Eval:       eval_set.json (question + ground-truth doc_ids + reference answer)
                -> run every question through the live pipeline
                -> retrieval metrics (precision@k, recall@k, MRR, NDCG@k)
                -> generation metrics (faithfulness, answer relevancy) via LLM-as-judge
                -> eval_report.json + eval_report.md
```

## Why each hard part is implemented the way it is

**Chunking** (`app/ingestion/chunking.py`) — three strategies, not one:
- `fixed`: naive token sliding window. Included mainly as a baseline so
  the eval harness can *prove* the smarter strategies help, rather than
  assuming they do.
- `recursive` (default): splits on paragraph → sentence → word
  separators, packing units up to `chunk_size` tokens with overlap. Keeps
  chunks semantically coherent without an embedding call at ingest time.
- `semantic`: embeds sentences and cuts at points where adjacent-sentence
  similarity drops below a threshold — splits at real topic boundaries.
  Costs an embedding pass at ingestion; worth it on long, topically mixed
  documents.

**Embedding model** (`app/retrieval/embeddings.py`) — defaults to a local
`sentence-transformers` model (no API key, no per-query network hop, no
per-token cost). OpenAI embeddings are a config flip away for when the
eval harness shows retrieval recall is actually the bottleneck — that's
an evaluation-driven upgrade, not a default, because re-embedding a whole
corpus against a new model is expensive.

**Hybrid search** (`app/retrieval/hybrid_search.py`) — vector and BM25
results are fused with Reciprocal Rank Fusion, not a weighted score sum.
BM25 scores and cosine similarities live on incomparable scales, so a
fixed linear weight between them is a magic number that breaks as corpus
size shifts. RRF fuses on rank instead, which is the standard approach in
production hybrid search systems.

**Reranking** (`app/retrieval/reranker.py`) — a cross-encoder re-scores
the top ~10 hybrid candidates before the final top-k is picked. Bi-encoder
vector search never lets the model see query and document together;
cross-encoders do, at a cost that's fine over 10 candidates but not over
a whole corpus — so it sits after hybrid search narrows the field, not
before.

**Evaluation** (`app/evaluation/`) — this is the part a toy RAG demo
skips entirely:
- Retrieval metrics (precision@k, recall@k, MRR, NDCG@k) are computed
  against ground-truth `doc_ids` in `data/eval_set.json` — cheap,
  deterministic, no LLM call needed.
- Generation metrics (faithfulness, answer relevancy) use an LLM-as-judge
  since "is this answer supported by the retrieved context" isn't a
  string-matching problem. They're measured independently because an
  answer can be faithful-but-irrelevant or relevant-but-hallucinated —
  conflating them into one score hides which failure mode you actually have.
- Optional RAGAS integration (`app/evaluation/ragas_eval.py`, gated by
  `RAGAS_ENABLED`) gives a second, independently-implemented opinion on
  the same axes — a sanity check against the custom judge scoring its own
  provider's generations leniently.

## Supported file formats

Ingestion goes through a registry (`app/ingestion/loaders/`) — one small
module per format, each converting its file type into plain text +
metadata, which then flows through the same chunk → embed → index pipeline
regardless of source format.

| Format | Extensions | Notes |
|---|---|---|
| Plain text / Markdown | `.txt`, `.md` | |
| WhatsApp chat export | `.txt` | Auto-detected (not a separate extension) — a `.txt` file is sniffed for chat-timestamp patterns; if it looks like an export, messages are parsed with sender/timestamp and multi-line messages are folded together instead of split into fragments |
| PDF | `.pdf` | Text-layer extraction only, no OCR — scanned/image-only PDFs will extract empty or near-empty text |
| Word | `.docx` | Paragraphs + tables (tables flattened to `\|`-joined rows) |
| Spreadsheets | `.csv`, `.xlsx`, `.xls` | Converted to a readable text table. This makes them **semantically retrievable** ("what does this sheet say about X") but **cannot answer computed/aggregate questions** ("what's the average of column Y") correctly — that needs a query engine (pandas/SQL agent), which is a separate, not-yet-built capability, not something this loader fakes |
| HTML | `.html`, `.htm` | Scripts/styles stripped, visible text extracted; no boilerplate/nav removal beyond that |
| JSON | `.json` | Pretty-printed for legibility, not minified |
| Source code | `.py`, `.js`, `.jsx`, `.ts`, `.tsx`, `.java`, `.c`, `.h`, `.cpp`, `.hpp`, `.go`, `.rs`, `.rb`, `.php`, `.cs`, `.swift`, `.kt`, `.sh`, `.sql`, `.yaml`, `.yml`, `.toml`, `.css`, `.scss` | Read as plain text, tagged with `language` in metadata. Chunked with the same token-based chunker as prose for now — no AST-aware splitting yet, so a function can still get cut mid-body by a chunk boundary |

Sample files for every new format live in `data/format_samples/` (not
auto-ingested — copy what you want into `data/sample_docs/` to index it, or
`POST` it via `/v1/ingest`).

**Adding a new format:** write one module in `app/ingestion/loaders/`
exporting an extension set and a `load_x(path) -> (text, metadata)`
function, then register it in `app/ingestion/loaders/__init__.py`. Nothing
else in the codebase needs to change — `pipeline.py`, `main.py`, and
`routes.py` only ever call `load_file`/`load_directory`.



```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set GENERATION_PROVIDER to anthropic / openai / ollama.
# ollama is free and runs locally — no API key, but needs `ollama serve`
# running and the model pulled first: `ollama pull llama3.2:3b`.
# For anthropic/openai, set the matching API key instead.
```

The default config (`EMBEDDING_PROVIDER=local`) needs no API key for
embeddings or retrieval — only generation (`/query`, `/evaluate`) needs a
key, since that's the one step that has to call an LLM.

CORS is open (`CORS_ORIGINS=*`) by default so a frontend on any port can
call this API during development. Restrict `CORS_ORIGINS` in `.env` to
your actual frontend origin(s) before deploying anywhere it's reachable
by others.

## Running

```bash
uvicorn app.main:app --reload
```

On first startup, if no index exists in `./storage`, the app auto-ingests
everything in `data/sample_docs/` (three docs about a fictional API
product — auth, rate limits, data retention) so there's something to
query immediately. Swap those for your own docs, or `POST /v1/ingest`
your own text.

Interactive API docs: `http://localhost:8000/docs`

> **Uploading files via `/docs`:** Swagger UI cannot render a proper file
> picker for `/v1/ingest/file` — it's a multi-file endpoint
> (`list[UploadFile]`), and Swagger UI has a long-standing limitation
> where arrays of binary files fall back to a plain text/string-array
> editor instead of a file chooser (open since 2018:
> [swagger-ui#4600](https://github.com/swagger-api/swagger-ui/issues/4600)).
> That's not a bug in this app — the endpoint itself works fine. For
> actually testing uploads by hand, use **`http://localhost:8000/upload`**
> instead: a small standalone page with a real file picker that hits the
> same `/v1/ingest/file` and `/v1/query` endpoints directly via `fetch()`.
> (Single-file endpoints, if you ever add one, render fine in Swagger —
> it's specifically the *array* of files that breaks.)

### Ingest a document (raw text)

```bash
curl -X POST localhost:8000/v1/ingest -H "Content-Type: application/json" -d '{
  "doc_id": "my_doc",
  "text": "Your document text here...",
  "metadata": {"source": "manual"}
}'
```

### Upload a file (the usual path for real files — PDF, docx, csv, etc.)

No need to place files on the server's filesystem — upload directly:

```bash
curl -X POST localhost:8000/v1/ingest/file \
  -F "files=@/path/to/resume.pdf"
```

Multiple files in one request (each processed independently — one
unsupported/corrupt file doesn't fail the others):

```bash
curl -X POST localhost:8000/v1/ingest/file \
  -F "files=@handbook.docx" \
  -F "files=@catalog.csv" \
  -F "files=@notes.md"
```

Response is a per-file breakdown:

```json
{
  "results": [
    {"filename": "handbook.docx", "doc_id": "handbook", "num_chunks": 4, "status": "success", "duplicate_doc_id": false},
    {"filename": "catalog.csv", "doc_id": "catalog", "num_chunks": 2, "status": "success", "duplicate_doc_id": false}
  ]
}
```

`doc_id` is derived from the filename (minus extension). Uploading a file
with the same name twice succeeds but sets `duplicate_doc_id: true` —
chunks are **added**, not replaced, so re-uploading the same file doubles
its chunks in the index. There's no update/delete-by-doc_id yet; if you
need to replace a document, that's a `rm -rf storage/` + full re-ingest for
now, not a targeted operation.

Rejected files (wrong extension, or over `MAX_UPLOAD_SIZE_MB` — 20MB by
default) come back as `"status": "error"` with a reason, not an HTTP
failure for the whole request — check each result's `status`. The size
limit is enforced while the file streams in, not after buffering the
whole thing in memory first.

**The original file is kept**, not discarded after extraction — it's
saved to `UPLOAD_DIR` (`./uploads` by default) and can be fetched back:

```bash
curl localhost:8000/v1/documents/handbook/original -o handbook_downloaded.docx
```

404s if the `doc_id` doesn't exist, or if it was ingested via raw-text
`POST /ingest` rather than a file upload (nothing to serve in that case).

### See what's indexed

```bash
curl localhost:8000/v1/documents
```

Lists every `doc_id` currently in the index with its chunk count — the
quickest way to confirm an upload landed, or to notice you've accidentally
double-ingested something.

### Query

```bash
curl -X POST localhost:8000/v1/query -H "Content-Type: application/json" -d '{
  "question": "What happens to my old API key after I rotate it?"
}'
```

Returns a grounded answer plus the chunks (with scores) it was generated from.

### Run the evaluation harness

```bash
curl -X POST "localhost:8000/v1/evaluate?judge_generation=true"
```

or directly, without the server:

```bash
python -m scripts.run_eval
```

Writes `eval_reports/eval_report.json` and `eval_reports/eval_report.md`
with aggregate + per-question metrics. Use this whenever you change
chunk size, chunk strategy, embedding model, or reranking on/off — the
report tells you whether the change actually helped.

## Tests

```bash
pytest tests/ -v
```

Covers chunking (size limits, overlap, uniqueness), hybrid fusion (RRF
ranking behavior, BM25 exact-match), and retrieval metrics (precision,
recall, MRR, NDCG correctness against hand-constructed cases).

## Key trade-offs and what to change first

- **`GENERATION_PROVIDER=ollama`** trades cost for quality and speed — free
  and local, but a small model (e.g. `llama3.2:3b`) is not a drop-in
  replacement for Claude/GPT-4-class generation, and is noticeably less
  reliable at strictly following the eval harness's "respond ONLY with
  JSON" judge prompts, so expect more `faithfulness`/`answer_relevancy`
  scores to come back `null` than with a hosted judge model. Run
  `/v1/evaluate` after switching to see the actual gap on your own eval
  set rather than assuming it either does or doesn't matter.
- **Judge and generation currently share one provider setting.** If Ollama
  answers are good enough but its JSON-following makes eval scores too
  noisy to trust, the cheap fix is to keep `GENERATION_PROVIDER=ollama` for
  `/v1/query` and temporarily point `.env` at a hosted provider only when
  running `/v1/evaluate` — the code doesn't yet split "judge model" from
  "generation model" as separate settings, so this is a manual swap for
  now, not a config flag.

- **FAISS is `IndexFlatIP`** (exact search), not an approximate index.
  Deliberate: for an eval harness to isolate chunking/embedding quality,
  you don't want ANN approximation error as a second, confounding source
  of recall loss. Move to `IndexIVFFlat`/HNSW once corpus size makes
  exact search a measured latency problem, not before.
- **Local embeddings by default.** Upgrade to OpenAI/a larger model only
  once `/v1/evaluate` shows retrieval recall, not chunking or reranking,
  is the bottleneck — re-indexing against a new embedding model is
  expensive to redo speculatively.
- **RRF constant (`RRF_K=60`) and hybrid candidate pool
  (`HYBRID_TOP_K=10`)** are standard starting values, not tuned for any
  particular corpus — re-tune against your own eval set once you have
  real query logs.
- **Judge model = same provider as generation, by default.** This risks
  a model being lenient toward its own outputs; the RAGAS cross-check
  exists specifically to catch that. For anything beyond a demo, prefer
  a different, stronger model as the judge.

## Project layout

```
app/
  config.py                settings (env-driven)
  models/schemas.py        pydantic models shared across the app
  ingestion/
    chunking.py            fixed / recursive / semantic chunkers
    loader.py               facade re-exporting the loaders/ registry
    loaders/                 one module per file format (registry pattern)
      __init__.py             extension -> loader function registry
      text_loaders.py          .txt/.md + WhatsApp export detection
      pdf_loader.py             .pdf
      docx_loader.py            .docx
      spreadsheet_loader.py     .csv/.xlsx/.xls (text representation, not a query engine)
      html_loader.py            .html/.htm
      json_loader.py            .json
      code_loader.py            .py/.js/.java/... (20+ languages, tagged by language)
  retrieval/
    embeddings.py           local (sentence-transformers) / OpenAI embeddings
    vector_store.py         FAISS wrapper + persistence
    keyword_search.py       BM25 wrapper + persistence
    hybrid_search.py        Reciprocal Rank Fusion
    reranker.py              cross-encoder reranking
    pipeline.py              ties ingestion + retrieval together
  generation/generator.py   grounded answer generation (Anthropic/OpenAI/Ollama)
  evaluation/
    metrics.py               precision/recall/MRR/NDCG + faithfulness/relevancy
    eval_harness.py          runs eval_set.json end-to-end, writes reports
    ragas_eval.py             optional RAGAS cross-check
  api/routes.py              /ingest /query /evaluate /health
  main.py                    FastAPI app + startup auto-indexing
data/
  sample_docs/                3 sample docs (fictional API product docs)
  eval_set.json                10 question/ground-truth/reference-answer rows
tests/                         pytest unit tests
Dockerfile
requirements.txt
.env.example
```

## A note on this build environment

This project was written and syntax-verified (every file parses cleanly)
in a sandbox without network access, so the dependencies in
`requirements.txt` could not be installed or run live here — you'll want
to run `pytest tests/ -v` yourself after `pip install -r requirements.txt`
to confirm end-to-end behavior in your own environment before relying on it.
