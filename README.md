# DocuQuery

A local Python RAG prototype for querying PDF policies, SOPs and technical documents. It combines Qdrant dense search, BM25, reciprocal rank fusion and cross-encoder reranking, then asks Groq to answer using retrieved passages with source labels.

## Setup

From the project root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`. `GROQ_MODEL` optionally overrides the default `openai/gpt-oss-20b`. Dependencies are listed but not locked. The first retrieval run downloads the embedding and reranking models.

Start Docker and Qdrant:

```sh
./script_qdrant.sh start
```

Qdrant listens on localhost:6333 and stores data in `qdrant_storage/`. Stop it with `./script_qdrant.sh stop`.

Index the bundled corpus once before querying:

```sh
cd backend
../.venv/bin/python -c 'from app.ingestion.pipeline import ingest_directory; ingest_directory("../data/raw")'
../.venv/bin/uvicorn app.main:app --reload
```

Ingestion creates the collection if needed. Restarting the API reads the local PDFs for BM25 lazily; it does not re-embed them. Keep the local corpus and Qdrant collection synchronized.

## API

Open [API documentation](http://127.0.0.1:8000/docs).

- `GET /` and `GET /health`: process liveness; these do not verify Qdrant or Groq.
- `POST /api/query`: JSON body `{"question": "How much parental leave can an employee take?"}`. Returns `question`, `answer`, and `sources` containing document ID, filename, page and reranker score. Sources correspond in order to `[SOURCE 1]`, `[SOURCE 2]`, etc.
- `POST /api/documents`: multipart `file` (PDF, at most 20 MiB) and `department`. Supported departments: HR, IT, Security, Compliance, Finance, Engineering, Operations, Sales. Returns 201 on success, 400 for invalid input, 409 for an existing filename and 413 for an oversized upload. Uploads are embedded into Qdrant and refresh the BM25 index.

Blank questions are rejected. PDFs without extractable text are rejected during ingestion; use OCR first. Queries with no evidence return an insufficient-information message without calling Groq. With retrieved evidence, source grounding and insufficient-information handling rely on the model prompt and are not guaranteed.

Run the manual example from the project root:

```sh
.venv/bin/python backend/app/rag/pipeline.py
```

`backend/testing_pipeline.py` also supports individual retrieval stages through its `RUN_*` flags. It reads the local corpus but expects Qdrant to be indexed already.

## Pipeline and code map

```text
PDF → page text → metadata → overlapping chunks → embeddings → Qdrant
Question → dense search + cached BM25 → RRF → reranker → context → Groq
```

| Path | Responsibility |
| --- | --- |
| `backend/app/ingestion/` | PDF extraction, metadata, 500-character chunks with 100-character overlap |
| `backend/app/embedding/` | Lazy-loaded BGE small embeddings, normalized 384-dimensional vectors |
| `backend/app/vectorstore/` | Qdrant collection, deterministic point IDs, upserts and search |
| `backend/app/retrieval/` | Dense baseline, BM25, fusion and diversification experiment |
| `backend/app/reranking/` | Lazy-loaded cross-encoder |
| `backend/app/generation/` | Source-labelled context and Groq generation |
| `backend/app/rag/` | Query orchestration (10 candidates per retriever, 10 fused, 3 final) |
| `backend/app/api/` | Query and PDF upload endpoints |
| `backend/app/services/` | Cached BM25 corpus and refresh |
| `backend/evaluate_retrieval.py` | Retrieval benchmark runner |
| `backend/tests/` | Regression and evaluation tests |
| `data/raw/` | Bundled PDF corpus and benchmark fixtures |

## Retrieval evaluation

With Qdrant running, from the project root:

```sh
.venv/bin/python backend/evaluate_retrieval.py
PYTHONPATH=backend .venv/bin/python -m unittest discover -s backend/tests -v
```

Evaluation uses a separate temporary collection and deletes it afterward. It saves per-query results, Recall@1/3/5, MRR@5 and diagnostics to `data/retrieval_evaluation.json`. Use `--help` to override paths. ACL queries and queries without expected documents are excluded from relevance averages. This benchmark does not validate authorization or answer quality.

Baseline fusion and reranking use 10 candidates per retriever. Diversification uses depth 30, caps candidates at two chunks per document, then reranks up to 10 into five results. This changes both candidate depth and diversity; results cannot be attributed to diversity alone.

### Latest local benchmark

110 queries evaluated; 4 ACL queries excluded (17 October 2026).

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | --- | --- | --- | --- |
| Dense | 0.668 | 0.886 | 0.950 | 0.801 |
| BM25 | 0.855 | 0.900 | 0.995 | 0.917 |
| RRF | 0.891 | 0.977 | 0.995 | 0.955 |
| RRF + Reranker | 0.900 | 0.968 | 0.995 | 0.957 |
| Diversified + Reranker | 0.891 | 0.982 | 1.000 | 0.955 |

Diversification recovers COM-005 for CHALLENGE-08, raising its Recall@5
from 0.5 to 1.0. It improves reciprocal rank on three queries and regresses
on Q-IT-002. Aggregate MRR@5 decreases slightly, so the original reranker
remains the comparison baseline. Version-aware retrieval remains future work.

## Current limits

- Users authenticate with expiring bearer tokens. Department checks and document-ID allowlists apply to dense and BM25 retrieval. Use one API worker: the BM25 cache is process-local.
- API-managed documents have UUIDs and support vector/file deletion. Version selection and archived-policy filtering remain future work. Legacy vectors are excluded until documents are imported.
- Filesystem, Qdrant and BM25 changes are not transactional; after partial upload failures, reconcile the stored PDF and vector collection before retrying. Upload serialization and cache refresh apply within one process only.
- No OCR, frontend, answer-quality benchmark or deployment setup. Prompt citations are not independently validated.
- The saved benchmark is a historical development result; it has not been rerun for these fixes.

Development notes: [Progress](PROGRESS.MD) · [Steps](STEPS.md) · [Pipeline explanation](STEP_EXPLANATION.md).


## Authenticated API

All `/api/query` and `/api/documents` routes require `Authorization: Bearer <access_token>`.
`/`, `/health`, `/docs`, and `/api/auth/login` remain public.

Initialize an administrator with a prompted password (no default credentials):

```bash
PYTHONPATH=backend .venv/bin/python backend/manage.py admin
```

The database defaults to `data/docuquery.db` (SQLite). To use PostgreSQL, create an empty
PostgreSQL database and set `DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/docuquery`
in `.env` before starting the server or running management commands. Tables are created
on startup; migrating an existing SQLite database to PostgreSQL is not automatic.

| Endpoint | Behavior |
| --- | --- |
| `POST /api/auth/login` | JSON `username`, `password`; returns a bearer token valid for one hour |
| `GET /api/auth/me` | Current user's ID, username, role, department |
| `POST /api/auth/users` | Admin only; JSON `username`, `password` (12+ characters), `role`, `department` |
| `POST /api/documents` | Multipart PDF and department; returns UUID and chunk count |
| `GET /api/documents?offset=0&limit=50` | Department-visible metadata, status, and total; max limit 100 |
| `GET /api/documents/{id}` | Document metadata; inaccessible IDs return 404 |
| `DELETE /api/documents/{id}` | Deletes vectors and PDF; returns 204; also retries incomplete deletions |
| `POST /api/query` | JSON `question`; retrieval restricted to visible ready documents |

Admins access every department and create users. Editors upload, read, query, and delete
within their department. Viewers read and query within their department. Roles and department
membership are assigned by admins when creating users. Document access is department-wide;
`owner_id` records the uploader, but does not grant extra access.

Example login:

```bash
curl -X POST http://127.0.0.1:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"your-password"}'
```

Existing PDFs must be registered and reindexed before authenticated queries can retrieve them:

```bash
PYTHONPATH=backend .venv/bin/python backend/import_documents.py
```

This copies PDFs directly under supported department folders into UUID subfolders and retains
originals. Re-running skips existing records; it never resets the vector collection.
Failed uploads remain listed with `failed` status and can be deleted before retrying.
Deletion revokes query access first; if storage cleanup fails, retry DELETE on the same ID.
Files, Qdrant, and SQL storage are not a distributed transaction. A process crash can leave an
`indexing` record; stop the interrupted worker and delete that record through the API before retrying.
Use a single API worker to keep the BM25 cache current. Query requests already in progress
may complete with their initial access snapshot. Production login throttling and a managed
schema-migration workflow remain deployment work.
