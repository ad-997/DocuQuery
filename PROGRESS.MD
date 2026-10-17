# DocuQuery progress

Updated 18 October 2026.

## Implemented

- Page-wise PDF extraction and metadata preservation.
- Fixed-size overlapping chunks with parameter validation and empty-text handling.
- Normalized BGE embeddings and Qdrant storage with deterministic IDs.
- Dense, cached BM25, RRF and cross-encoder retrieval.
- Retrieval evaluation with document recall, MRR, per-query diagnostics and diversification comparison.
- Source-labelled context and Groq answer generation.
- FastAPI query, PDF upload and liveness endpoints.
- Upload size/type/department validation and duplicate-filename protection.
- Lazy model loading and BM25 initialization.
- Dependency list, environment example and regression tests.

## Verified in this maintenance pass

- API routing and query-to-pipeline arguments using controlled test doubles.
- Empty corpus/evidence handling, chunk validation and PDF extraction fallback.
- Upload validation, failed-ingestion file cleanup and duplicate protection.
- Isolated Qdrant local-mode upsert/search and repeat-ingestion point count.
- Existing benchmark evaluation and diversification tests.

Live Groq generation and the full corpus benchmark were not rerun. See the README for setup and historical benchmark results.

## Remaining

- Production authentication throttling and session administration.
- Managed schema migrations and multi-worker metadata/cache coordination.
- Version-aware retrieval and archived-document filtering.
- Atomic recovery across files, vectors and BM25; multiple-worker cache coordination.
- OCR and chunking experiments.
- Answer grounding/citation validation and full RAG evaluation.
- Frontend and deployment work when required.

## API implementation update

- Document list/detail/delete, UUID upload identities, persistent lifecycle records, and vector cleanup.
- Login/me, admin-only user creation, scrypt password hashes, hashed expiring bearer sessions.
- Admin/editor/viewer roles with department-scoped document APIs and retrieval allowlists.
- SQLAlchemy storage with local SQLite default and PostgreSQL configuration support.
- Administrator bootstrap and non-destructive legacy PDF import commands.
- API authorization, lifecycle, session expiry, and both retrieval filters covered by isolated tests.

PostgreSQL server integration and live model/Groq calls have not been run in this pass.
