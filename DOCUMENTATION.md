# RAG System — File-by-File Documentation

This walks through every file in the project **in the order it's most useful
to read them** — not alphabetically, but following the actual flow of data
through the system: config → data shapes → ingestion → retrieval → generation
→ evaluation → API → tests. By the end you should be able to trace a request
from "user asks a question" to "answer with citations" through actual code,
and know exactly which file to open when you want to change something.

---

## 0. The mental model first

Before the files, the one diagram worth keeping in your head:

```
INGEST TIME (once per document):
  raw text → chunk it → embed each chunk → store in FAISS (vectors) + BM25 (keywords)

QUERY TIME (once per question):
  question → embed it → search FAISS (top 20) AND search BM25 (top 20)
           → merge both result lists into one ranked list (Reciprocal Rank Fusion)
           → take the top 10 → re-score with a cross-encoder (reranking)
           → take the top 4 → hand those 4 chunks + the question to an LLM
           → LLM answers using ONLY those chunks

EVAL TIME (whenever you want to know if a change helped or hurt):
  for each question in a test set with known-correct answers:
    run it through the exact same query-time pipeline above
    compare retrieved chunks against known-correct docs → retrieval scores
    compare generated answer against the context it saw → faithfulness score
```

Every file below implements one box in that diagram. Keep this picture open
while reading.

---

## 1. `app/config.py` — the control panel

**What it does:** defines every tunable setting as a Python class
(`Settings`), using `pydantic-settings`. Values come from environment
variables or a `.env` file, with the defaults you see in the code as
fallback.

**Why it exists as its own file:** nothing else in the codebase hardcodes a
number, a model name, or a strategy choice. `app/ingestion/chunking.py`
doesn't know the chunk size is `512` — it receives `settings.chunk_size` as
an argument. This means changing `CHUNK_STRATEGY=semantic` in `.env` and
restarting is the entire "how do I switch strategies" workflow — no code
edits.

**Read this file when:** you want to know what's configurable, or you're
about to add a new tunable and need to know where it goes (here, plus a line
in `.env.example`).

**Key fields to know by name** (you'll see these referenced everywhere):
- `chunk_strategy`, `chunk_size`, `chunk_overlap` — ingestion behavior
- `vector_top_k`, `keyword_top_k` — how many candidates each search pulls before fusion
- `hybrid_top_k` — how many fused candidates go to the reranker
- `final_top_k` — how many chunks actually reach the LLM
- `rerank_enabled` — on/off switch for the cross-encoder step
- `embedding_provider`, `generation_provider` — `local`/`openai` and `anthropic`/`openai` respectively

---

## 2. `app/models/schemas.py` — the shapes data takes

**What it does:** every Pydantic model that flows between layers of the
system. Nothing else in `app/` invents its own dict shape — everything
passes one of these objects around, which is what makes FastAPI's automatic
request/response validation and the `/docs` page work correctly.

**Read top to bottom, this is the lifecycle of one piece of text:**

| Class | Represents |
|---|---|
| `Chunk` | One chunk of a document after splitting — has `chunk_id`, `doc_id`, `text`, `position`, `token_count`, `metadata` |
| `ScoredChunk` | A `Chunk` plus a relevance `score` and which stage produced it (`"vector"` / `"keyword"` / `"hybrid"` / `"reranked"`) — this `source` field is genuinely useful for debugging *why* a chunk got picked |
| `IngestRequest` / `IngestResponse` | What you POST to `/v1/ingest` and what you get back |
| `QueryRequest` / `QueryResponse` | What you POST to `/v1/query` and what you get back — note `QueryResponse.citations` is a list of `Citation`, so every answer is traceable to specific chunks |
| `EvalExample` | One row of a test set: a `question`, the `ground_truth_doc_ids` that *should* be retrieved, and a `reference_answer` a human would give |
| `EvalReport` | The output of a full eval run: aggregate `retrieval` metrics, aggregate `generation` metrics, and `per_example` detail |
| `FileIngestResult` / `FileIngestResponse` | Per-file outcome of a `POST /ingest/file` call — `status`, `error` if any, and `duplicate_doc_id` if this filename was already indexed |
| `DocumentInfo` / `DocumentListResponse` | What `GET /documents` returns — every indexed `doc_id` with its chunk count |

**Read this file when:** you're not sure what fields something has, or
you're extending the API and need to add a field — do it here first, it
propagates everywhere automatically.

---

## 3. `app/ingestion/loaders/` — turning files into text (registry pattern)

**What it does, as a whole:** this is a package, not a single file, because
Phase 1 widened format support from 3 extensions to 30+. The pattern: each
module handles exactly one format family and exports (a) a set of file
extensions it owns, and (b) a `load_x(path) -> (text, metadata)` function.
`__init__.py` builds a dict mapping every extension to its loader function
and exposes two calls — `load_file(path)` and `load_directory(dir_path)` —
which is all the rest of the app (`pipeline.py`, `main.py`) ever calls.
Nothing downstream knows or cares that the registry exists.

**The modules, each independently testable and swappable:**

- **`text_loaders.py`** — `.txt`/`.md`. The interesting part: a `.txt` file
  is sniffed for WhatsApp-export-style timestamp lines
  (`_looks_like_whatsapp_export`); if it matches, `_parse_whatsapp_export`
  parses sender/timestamp/message per line and **folds multi-line
  continuations into the message they belong to** rather than treating
  every newline as a new unit — so a long message doesn't get chunked into
  disconnected fragments later.

- **`pdf_loader.py`** — same `pypdf` text-layer extraction as before, just
  moved into its own module. Still no OCR — a scanned PDF extracts empty.

- **`docx_loader.py`** — `python-docx`. Pulls paragraph text plus table
  cells (flattened to `|`-joined rows, appended after the body rather than
  interleaved in original position).

- **`spreadsheet_loader.py`** — `.csv`/`.xlsx`/`.xls` via `pandas`. Converts
  rows to a readable text table so spreadsheets become semantically
  retrievable through the *same* chunk+embed pipeline as everything else.
  The docstring is explicit about the limit: this makes "what does this
  sheet say about X" answerable, but **not** "what's the average of column
  Y" — an LLM reading a text dump of rows will guess at arithmetic, not
  compute it correctly. Real spreadsheet Q&A needs a query engine (pandas/
  SQL agent) routed to separately from vector search — a distinct,
  not-yet-built capability, not something this loader pretends to solve.

- **`html_loader.py`** — `BeautifulSoup` with the built-in `html.parser`
  backend (no `lxml` needed). Strips `<script>`/`<style>`, extracts visible
  text. No nav/footer/boilerplate removal beyond that — a saved article
  works better than a page dense with navigation chrome.

- **`json_loader.py`** — pretty-prints (not minifies) so structure stays
  legible in a retrieved chunk; tags `top_level_keys` or `num_items` in
  metadata depending on whether the root is a dict or list.

- **`code_loader.py`** — reads source as plain text, tags `language` in
  metadata from a 20+-extension map (`.py`→`python`, `.rs`→`rust`, etc.).
  Deliberately does **not** do AST-aware splitting yet — that's a Phase 2
  item — so a function can still get cut mid-body by the token-based
  chunker. The language tag is what a future AST chunker would key off of
  to pick the right parser per file.

**`app/ingestion/loader.py`** — kept as a thin backward-compatible facade:
`from app.ingestion.loaders import load_file, load_directory,
SUPPORTED_EXTENSIONS`. This exists purely so nothing that already imported
`app.ingestion.loader` had to change when the registry was introduced.

**Read this package when:** you're adding support for a new file format
(write one module, add two lines to `__init__.py`), or debugging why a
particular file didn't extract the text you expected (find its module,
that's the one place its extraction logic lives).

---

## 4. `app/ingestion/chunking.py` — the hard part, part 1

This is the longest, most-commented file in the project on purpose — chunking
strategy is where naive RAG systems quietly lose quality, so the *reasoning*
behind each choice is written into the module docstring and each class.

**What it does:** defines a `BaseChunker` interface with one required method,
`split(text) -> list[str]`, and three implementations:

- **`FixedSizeChunker`** — a sliding window over raw tokens. Cuts sentences
  in half without hesitation. Exists mainly as a deliberately-dumb baseline
  so you can prove (via the eval harness) that the smarter strategies
  actually help, rather than assuming it.

- **`RecursiveChunker`** (the default) — tries to split on paragraph breaks
  first; if a paragraph is still too big, splits on sentences; if a sentence
  is still too big, splits on words. Then greedily *packs* those pieces back
  up to `chunk_size` tokens, carrying a bit of trailing text as overlap into
  the next chunk. This is the same idea LangChain's
  `RecursiveCharacterTextSplitter` uses, and it's the standard default in
  production RAG because it keeps chunks readable without costing an
  embedding call at ingest time.

- **`SemanticChunker`** — splits text into sentences, embeds each one, and
  cuts a new chunk exactly where cosine similarity between two consecutive
  sentences drops below `SEMANTIC_CHUNK_THRESHOLD`. In plain terms: it finds
  where the topic actually changes, instead of cutting at a fixed length. It
  needs an embedding function passed in at construction (`embed_fn`), which
  is why it's the one strategy the pipeline constructs slightly differently
  (see `pipeline.py` below).

**The factory function** `get_chunker(strategy, ...)` at the bottom is what
`pipeline.py` calls — it's the single place that maps the `CHUNK_STRATEGY`
config string to an actual class instance.

**Also in this file:** `count_tokens()` (uses `tiktoken`, the same tokenizer
family OpenAI models use, so token counts are a realistic proxy regardless
of which embedding model you end up using) and `_split_sentences()`, a
lightweight regex sentence splitter used by the semantic strategy.

**Read this file when:** retrieval quality feels off and you suspect chunks
are too big/small/badly-cut — this is usually the first place to look, and
the eval harness (`app/evaluation/`) is how you'd confirm a change here
actually helped.

---

## 5. `app/retrieval/embeddings.py` — the hard part, part 2

**What it does:** `EmbeddingModel` is a class that wraps either a local
`sentence-transformers` model or the OpenAI embeddings API behind one
interface — `.embed(list[str]) -> np.ndarray` and `.embed_one(str) ->
np.ndarray`. Both providers are normalized to unit-length vectors so cosine
similarity works identically regardless of which one is active.

**Why local is the default:** no API key needed for retrieval, no
per-query network round trip, no per-token cost. The docstring in this file
explicitly argues this is the *right default* for most internal-knowledge-
base use cases, and that upgrading to OpenAI embeddings should be a decision
you make *after* the eval harness shows retrieval recall is actually your
bottleneck — because re-embedding an entire corpus against a new model is
expensive to redo speculatively.

**`get_embedding_model()`** at the bottom is a cached singleton (`lru_cache`)
— the model is loaded once per process, not once per request, because
loading it is the slow part.

**Read this file when:** you're switching embedding providers, or debugging
why retrieval quality differs between two environments (check
`EMBEDDING_PROVIDER` matches).

---

## 6. `app/retrieval/vector_store.py` — where embeddings live

**What it does:** wraps a FAISS `IndexFlatIP` (inner product on normalized
vectors = cosine similarity) plus a parallel Python list of `Chunk` objects,
so vector index position `i` always corresponds to `self.chunks[i]`.

**Why `IndexFlatIP` and not an approximate index:** it's *exact* search — no
approximation error. The docstring explains this is deliberate: if you're
running an eval harness to measure retrieval quality, you don't want ANN
approximation error muddying whether a low score is from bad chunking, a
weak embedding model, or just the index skipping the right answer. Move to
`IndexIVFFlat`/HNSW only once corpus size makes exact search a *measured*
latency problem.

**`.save(directory)` / `.load(directory)`** persist the index (via
`faiss.write_index`) and the chunk list (via `pickle`) to disk, which is how
the app avoids re-ingesting on every restart.

**Read this file when:** you're scaling past a small corpus and need to
think about ANN indexes, or debugging a mismatch between chunk count and
index size.

---

## 7. `app/retrieval/keyword_search.py` — the other half of hybrid search

**What it does:** wraps `BM25Okapi` from `rank_bm25`. `tokenize()` lowercases
and strips to alphanumeric tokens; `KeywordStore.add()` builds/rebuilds the
BM25 index over all chunks; `.search(query, top_k)` returns chunks ranked by
BM25 score.

**Why this exists alongside vector search:** the docstring is direct about
it — embeddings are good at *meaning*, bad at *exact tokens*. A dense
embedding model will happily conflate a specific error code, SKU, or
acronym with something semantically similar but wrong. BM25 catches exact
matches embeddings miss. Neither alone is sufficient; that's the entire
argument for hybrid search.

**Read this file when:** a query with a specific identifier/code/name isn't
retrieving the right chunk — check whether keyword search alone (bypass
vector) finds it, which tells you whether the gap is in embeddings or
somewhere else.

---

## 8. `app/retrieval/hybrid_search.py` — merging the two result lists

**What it does:** one function, `reciprocal_rank_fusion(result_lists, k=60)`.
For every chunk, it sums `1 / (k + rank + 1)` across every list it appears
in, then sorts by that summed score.

**Why RRF and not "just add the scores together":** BM25 scores are
unbounded and corpus-size-dependent; cosine similarities are bounded
`[-1, 1]`. Any fixed weight between them (e.g. `0.5 * bm25 + 0.5 * cosine`)
is a magic number that breaks the moment corpus size or query style shifts.
RRF sidesteps the whole problem by fusing on **rank position**, not raw
score — a chunk ranked #1 in both lists always beats one ranked #1 in only
one list, regardless of what the underlying scores looked like. This is the
standard approach in production hybrid search systems (Elasticsearch,
Weaviate, etc.) for exactly this reason.

**Read this file when:** you're tuning `RRF_K` — smaller `k` weights top
ranks more heavily; larger `k` flattens the curve so lower ranks still
matter.

---

## 9. `app/retrieval/reranker.py` — the hard part, part 3

**What it does:** `Reranker` wraps a `CrossEncoder` model (from
`sentence-transformers`). `.rerank(query, candidates, top_k)` scores every
`(query, chunk_text)` pair *jointly* through the model and returns the
top-k by that score.

**Why this is a separate stage from vector/keyword search, not a
replacement for them:** the docstring explains the bi-encoder vs.
cross-encoder distinction plainly — a bi-encoder (what embeddings use)
encodes query and document *separately*, so it's fast enough to search
millions of chunks, but the model never actually looks at query and document
together. A cross-encoder does look at them together and is meaningfully
more accurate, but far too slow to run over an entire corpus. So the
pattern is: cheap hybrid search narrows the field to ~10 candidates, then
the expensive-but-accurate cross-encoder re-scores just those.

**`get_reranker()`** — same cached-singleton pattern as the embedding model.

**Read this file when:** deciding whether `RERANK_ENABLED=true` is worth the
extra latency for your use case — the eval harness is how you'd measure
that trade-off concretely instead of guessing.

---

## 10. `app/retrieval/pipeline.py` — where 4–9 come together

**What it does:** `RagPipeline` is the one class that owns a `VectorStore`
and a `KeywordStore` and exposes exactly two operations you actually call
from the rest of the app: `.ingest(doc_id, text, metadata)` and
`.retrieve(query, final_top_k, rerank)`.

**Trace `.ingest()`:** picks a chunker via `get_chunker()` (passing an embed
function only if strategy is `"semantic"`) → chunks the text → embeds every
chunk in one batch call → adds to both the vector store and keyword store.

**Trace `.retrieve()`:** embeds the query once → searches vector store (top
`vector_top_k`) → searches keyword store (top `keyword_top_k`) → fuses both
lists with RRF → takes the top `hybrid_top_k` → if reranking is on, reranks
down to `final_top_k`; if off, just truncates to `final_top_k`.

**`.save()` / `.load()` / `.exists()`** — thin wrappers that persist/restore
both stores together, since they always need to stay in sync.

**Read this file when:** you want to understand the whole retrieval flow in
one place without jumping between six files — this is the "connective
tissue" file. It's also the one file you'd touch to add a new retrieval
stage (e.g. a query-rewriting step before embedding).

---

## 11. `app/generation/generator.py` — turning chunks into an answer

**What it does:** `generate_answer(question, chunks)` builds a prompt with
`SYSTEM_PROMPT` (instructs the model to answer *only* from the given
context, cite chunk IDs, and explicitly say when the context doesn't have
the answer) plus the formatted chunks, then calls either Anthropic or OpenAI
depending on `settings.generation_provider`.

**Why the system prompt is strict about grounding:** this isn't just good
practice — it's the other half of a contract with the evaluation harness.
The faithfulness metric (see `metrics.py` below) measures whether the model
actually honored "answer only from context." The prompt and the eval exist
as a matched pair; changing one without the other breaks the measurement.

**`_build_context()`** formats chunks as `[chunk_id] (doc: doc_id)\n{text}`
blocks separated by `---`, which is what lets the model's citations in its
answer actually correspond to real chunk IDs you can look up.

**Read this file when:** answers feel too hedgy, too willing to guess beyond
the context, or citation formatting needs to change — the prompt lives
here, in one place, not scattered across call sites.

---

## 12. `app/evaluation/metrics.py` — the hard part, part 4 (and the point of the whole project)

This file has two halves, and the docstring at the top is worth reading in
full — it explains *why* each metric exists, not just what it computes.

**Half 1 — retrieval metrics** (no LLM needed, pure math against known
ground truth):
- `precision_at_k` — of the k chunks you retrieved, what fraction came from
  a document that was actually relevant?
- `recall_at_k` — of all the relevant documents that exist, what fraction
  had at least one chunk show up in your top k?
- `mrr` (Mean Reciprocal Rank) — how far down the list was the *first*
  relevant hit? A hit at position 1 scores 1.0, position 2 scores 0.5,
  position 10 scores 0.1. Rewards ranking quality specifically.
- `ndcg_at_k` — like MRR but accounts for *multiple* relevant hits and
  discounts by position more gradually (log-scale) rather than by strict
  reciprocal rank.

These four together tell you different things: precision tells you if
you're retrieving junk, recall tells you if you're missing things entirely,
MRR/NDCG tell you if the *good* chunks are near the top or buried.

**Half 2 — generation metrics** (require an LLM-as-judge, since "is this
answer supported by the context" isn't something you can string-match):
- `faithfulness(answer, context_chunks)` — asks a judge LLM to break the
  answer into individual factual claims and mark each one supported/
  unsupported by the given context. Returns the fraction supported. This is
  specifically catching **hallucination** — the model answering from its
  own training knowledge instead of the retrieved chunks.
- `answer_relevancy(question, answer)` — asks a judge LLM to score, 0 to 1,
  whether the answer actually addresses the question at all, independent of
  whether it's factually grounded. This catches a *different* failure mode:
  an answer that's perfectly faithful to the context but off-topic or
  evasive.

**Why these two are measured separately, not combined into one score:** the
docstring is explicit — an answer can be faithful-but-irrelevant (accurately
restates an unrelated chunk) or relevant-but-unfaithful (correctly on-topic
but made-up). Combining them into one number would hide which failure mode
you're actually looking at.

**`_judge_call()` / `_parse_json_response()`** — plumbing that sends the
judge prompt to whichever provider is configured and parses the JSON
response, with the judge prompts to Markdown-code-fence stripping baked in
since LLMs like to wrap JSON in ```json fences even when told not to.

**Read this file when:** you want to understand exactly what a score in an
eval report means, or you're adding a new metric.

---

## 13. `app/evaluation/eval_harness.py` — running metrics.py against real questions

**What it does:** `load_eval_set(path)` reads a JSON file of `EvalExample`
rows. `run_eval(pipeline, eval_set, k, judge_generation)` is the main loop:
for every example, it calls `pipeline.retrieve()` (the *exact same* method
the live API uses — not a special eval-only code path), computes retrieval
metrics against `ground_truth_doc_ids`, optionally generates an answer and
computes faithfulness/relevancy, and collects everything into an
`EvalReport`. `write_report()` writes both a machine-readable
`eval_report.json` and a human-readable `eval_report.md`.

**Why it calls the real pipeline instead of a mock:** the whole point of an
eval harness is to measure what the live system actually does. If eval used
a different code path than production, a passing eval wouldn't tell you
anything about production behavior.

**Read this file when:** you want to change what a report looks like, add a
new field to the CSV/JSON output, or understand exactly how aggregate scores
are computed from per-example scores (it's a simple mean, skipping any
example where the judge call failed rather than treating a failure as 0).

---

## 14. `app/evaluation/ragas_eval.py` — an optional second opinion

**What it does:** `run_ragas_eval()` runs the same eval set through the
pipeline, but hands the results to the third-party `ragas` library's
`faithfulness` and `answer_relevancy` metrics instead of the custom ones in
`metrics.py`.

**Why it's opt-in (`RAGAS_ENABLED=false` by default), not the default path:**
two reasons stated in the docstring — it pulls in extra dependencies
(`datasets`, its own LLM calls), and its real value is as a *sanity check*:
if your custom judge (using the same LLM provider as generation) is scoring
its own outputs leniently, RAGAS's independent implementation is a way to
catch that bias. Not something you need on every run.

**Read this file when:** you suspect the custom faithfulness/relevancy
scores are too optimistic and want a second, independently-coded metric to
compare against.

---

## 15. `app/api/routes.py` — the HTTP surface

**What it does:** seven endpoints, each a thin wrapper around
`RagPipeline`/generation/eval functions you've already read about:

- `GET /health` — liveness check, returns `{"status": "ok"}`
- `POST /ingest` — calls `pipeline.ingest()` with raw text supplied in the
  request body, then `pipeline.save()` so it persists across restarts
- `POST /ingest/file` — accepts one or more uploaded files
  (`multipart/form-data`), the actual on-ramp for real files (PDF, docx,
  csv, etc.) without needing filesystem access to the server. Each file is
  streamed to disk in `UPLOAD_DIR` (`./uploads` by default) in 1MB chunks
  — `MAX_UPLOAD_SIZE_MB` is enforced *during* the write (the running total
  is checked every chunk, and a partial write is deleted the moment it's
  exceeded) rather than after buffering the whole file into memory first.
  Unlike the very first version of this endpoint, **the original file is
  kept, not deleted** after extraction — its path is recorded in
  `metadata["stored_path"]` on every chunk, which is what
  `GET /documents/{doc_id}/original` reads back. The uploaded filename is
  sanitized to its base name (`Path(filename).name`) before it's ever used
  in a path join, specifically to block path-traversal (a filename like
  `"../../etc/cron.d/x"` is fully attacker-controlled since it's just a
  multipart header). Files are processed independently in a loop: one
  unsupported extension or corrupt file returns `"status": "error"` for
  *that* result without aborting the rest of the batch. Extension is
  checked against `SUPPORTED_EXTENSIONS` before anything is written.
  `doc_id` is derived from the sanitized filename's stem; uploading the
  same filename twice succeeds but flags `duplicate_doc_id: true` — chunks
  are *appended*, not replaced, since there's no update/delete-by-doc_id
  yet.
- `GET /documents/{doc_id}/original` — serves the original file back
  (via `FileResponse`), the counterpart to persisting uploads instead of
  deleting them. Looks up `stored_path` from that doc_id's chunk metadata;
  404s distinctly for "doc_id doesn't exist" vs. "doc_id exists but was
  ingested via raw-text `/ingest`, so there's no original file to serve."
- `GET /documents` — lists every `doc_id` currently in the vector store
  with its chunk count, by walking `pipeline.vector_store.chunks` and
  counting. Exists specifically so upload success/duplicates are checkable
  without needing eval reports or server logs.
- `POST /query` — calls `pipeline.retrieve()`, then `generate_answer()`,
  packages the result with citations
- `POST /evaluate` — calls `load_eval_set()` + `run_eval()` + `write_report()`

**Notice what this file does *not* contain:** no chunking logic, no
retrieval logic, no prompt text, no file-format-specific extraction code.
Every route is a thin translation layer because the real work lives in the
modules above — `/ingest/file`'s only format-aware line is the call to
`load_file()`, the same one `data/sample_docs/` auto-ingestion uses.

**Read this file when:** adding a new endpoint, or checking exactly what
request/response shape an endpoint expects (cross-reference with
`schemas.py`).

---

## 16. `app/main.py` — where the app actually starts

**What it does:** defines the FastAPI `app` object and a `lifespan` context
manager that runs once at startup: it constructs a `RagPipeline`, and either
loads an existing saved index from `./storage` (if one exists) or
auto-ingests everything in `data/sample_docs/` and saves a fresh index. The
pipeline is stashed on `app.state.pipeline`, which is how `routes.py`
accesses it (via `request.app.state.pipeline`).

**Why auto-ingestion on first run:** so `POST /v1/query` works immediately
after `uvicorn app.main:app` with zero setup — there's always something
indexed to query against.

**Also serves `GET /upload`** — a standalone HTML test page
(`app/api/upload_page.py`, just a Python string constant, no templating
engine needed for something this small) with a real `<input type="file"
multiple>` picker that calls `/v1/ingest/file`, `/v1/documents`, and
`/v1/query` via `fetch()`. Exists because Swagger's `/docs` page cannot
render a file picker for a multi-file (`list[UploadFile]`) endpoint — a
long-standing Swagger UI limitation, not a bug in this app — so this page
is the actual way to test uploads by hand in a browser.

**Read this file when:** you want to change startup behavior (e.g. always
re-ingest instead of loading a saved index), or you're wiring in a new
router.

---

## 17. `data/sample_docs/*.md` — the demo corpus

Three short fictional docs about a made-up API product ("Nimbus"):
`authentication.md`, `rate_limits.md`, `data_retention.md`. They exist
purely so the system has something realistic to ingest and query out of the
box — topically distinct enough that retrieval should cleanly separate them,
which makes it obvious when something's wrong (if a rate-limits question
pulls an authentication chunk, that's a clear signal, not a subtle one).

**Read these when:** you want to see what "good" input documents look like
in terms of length/structure, or you're replacing them with your own corpus.

**Also see `data/format_samples/`** — one sample file per new format added
in Phase 1 (`.csv`, `.html`, `.json`, `.py`, a WhatsApp-style `.txt` export,
and a hand-built `.docx`). Not auto-ingested on startup — copy the ones you
want into `data/sample_docs/` to actually index and query them, or use
them to sanity-check a loader change (`load_file(path)` on any one of them
should return readable text + sensible metadata).

---

## 18. `data/eval_set.json` — the test set

Ten questions, each with `ground_truth_doc_ids` (which sample doc should
answer it) and a `reference_answer` (what a correct answer looks like). This
is the input to `run_eval()`.

**Read this when:** writing your own eval set for your own documents — copy
this structure. The quality of your eval report is entirely bounded by the
quality of this file: garbage or ambiguous ground truth gives you garbage
metrics.

---

## 19. `tests/test_chunking.py`, `test_retrieval.py`, `test_metrics.py` — unit tests

These don't need an LLM key or network access (that's deliberate — they
test pure logic):

- **`test_chunking.py`** — asserts `FixedSizeChunker` respects size limits
  and actually produces overlapping text between adjacent chunks; asserts
  `RecursiveChunker` collapses a small doc to few chunks but splits a large
  one; asserts chunk IDs are unique.
- **`test_retrieval.py`** — asserts BM25 tokenization behaves as expected
  and finds exact-term matches; asserts RRF actually boosts chunks that rank
  highly in *both* lists over ones that only appear in one.
- **`test_metrics.py`** — hand-constructs small retrieval result lists and
  checks precision/recall/MRR/NDCG against manually-computed expected
  values, so you can trust the metric implementations themselves are
  correct before trusting what they report about your pipeline.

**Read/run these when:** you change any of the logic in the files they
cover — they're your first line of defense against a subtle bug (e.g. an
off-by-one in ranking) silently corrupting eval numbers.

---

## 20. `scripts/run_eval.py` — eval without starting a server

**What it does:** the same ingest-then-eval flow as the API's `/evaluate`
endpoint, but runnable directly with `python -m scripts.run_eval` — no
`uvicorn` needed. This is the fast iteration loop: change a setting in
`.env`, run this, read the printed metrics and the written report, repeat.

**Read this when:** you're tuning config values and don't want the overhead
of starting/stopping a server each time.

---

## 21. `requirements.txt`, `.env.example`, `Dockerfile`

- **`requirements.txt`** — every Python dependency, pinned with `>=` minimum
  versions (not exact pins) so `pip install` resolves to whatever's current
  on PyPI rather than failing on a specific version.
- **`.env.example`** — template for `.env`, documents every setting from
  `config.py` with inline comments on what each does and its default.
  Copy it, fill in an API key, rename to `.env`.
- **`Dockerfile`** — standard Python slim image, installs deps, copies the
  app, runs `uvicorn` on port 8000. Nothing project-specific to know beyond
  "this is how you'd containerize it."

---

## How to actually use this document

You don't need to have read every file before using the system — the
`README.md` covers setup/run/test commands. Come back to this doc when:

- You're changing behavior and need to know *which single file* owns that
  behavior (use the file list above as an index).
- Eval numbers look wrong and you need to trace *which stage* of the
  pipeline (chunking? embedding? fusion? reranking? generation?) is the
  likely cause — section 0's diagram plus the "read this file when" notes
  above are built for exactly that triage.
- You're extending the system (new file format, new chunking strategy, new
  metric) and want to follow the existing pattern rather than invent a new
  one.
