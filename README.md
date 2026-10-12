# DocuQuery

An early Python prototype for finding relevant passages in PDFs using sentence embeddings. The goal is to make policies, SOPs and technical documents searchable even when a question uses different words from the source.

**Status: retrieval prototype in development.** This is the beginning of a RAG project. It does not yet generate answers, enforce access permissions or provide a document-search API.

## Implemented

- PDF text extraction with `pypdf`, retaining page numbers.
- Document metadata derived from the filename and parent folder.
- Character-based chunks with defaults of 500 characters and 100 characters of overlap.
- Normalized embeddings using `BAAI/bge-small-en-v1.5` through Sentence Transformers.
- In-memory top-k retrieval using the dot product of normalized vectors (cosine similarity).
- A FastAPI application with a root health message.
- A script that retrieves the top three chunks for a sample parental-leave question.

## Pipeline

```text
PDF → extract page text → attach metadata → chunk text
    → embed chunks → embed query → rank by similarity → print matches
```

The script prints the similarity score, page number, filename and passage. Scores rank matches; they are not measured answer accuracy or calibrated confidence.

## Try the prototype locally

Use a Python environment compatible with Sentence Transformers and PyTorch. Dependencies are currently unpinned; the commands below are a manual setup recipe, not a verified reproducible build.

```sh
git clone https://github.com/ad-997/DocuQuery.git
cd DocuQuery
python3 -m venv .venv
source .venv/bin/activate
python -m pip install fastapi uvicorn pypdf sentence-transformers numpy
cd backend
python testing_pipeline.py
```

Run from `backend/`: the script uses the relative path `../data/raw/HR/hr-006_parental_leave_guide.pdf`. The first embedding run downloads the model. Change `file_path` and `query` in the script to explore another document or question.

For the separate FastAPI scaffold, from `backend/`:

```sh
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the health message and `http://127.0.0.1:8000/docs` for the API schema. Retrieval currently runs through the script, not an HTTP endpoint.

## Code map

| Path | Responsibility |
| --- | --- |
| `backend/app/ingestion/pdf_loader.py` | Extract page text |
| `backend/app/ingestion/metadata.py` | Derive document metadata |
| `backend/app/ingestion/chunker.py` | Split text into overlapping chunks |
| `backend/app/embedding/embedder.py` | Encode chunks and queries |
| `backend/app/retrieval/dense.py` | Rank chunks by vector similarity |
| `backend/testing_pipeline.py` | Run the current single-document example |
| `data/raw/` | Example PDFs, manifests and evaluation-query fixtures |

## Current limits

- The chunker's `return chunks` is inside the page loop, so the current implementation only chunks the first page. Multi-page retrieval needs a fix before it can be claimed.
- The loader expects extractable text; scanned PDFs need OCR, and empty extraction needs handling.
- Retrieval covers the document loaded by the script and keeps embeddings in memory.
- Chunk-size/overlap validation, dependency locking and automated regression checks are still needed.
- Evaluation fixtures are present, but retrieval quality has not been measured.

## Next steps

First, correct multi-page chunking and add checks for page coverage and empty PDFs. Then make dependencies reproducible and evaluate retrieval against the supplied questions.

Later work: Qdrant persistence, BM25/hybrid retrieval, reranking, answer generation with citations, authentication and role-based access, a React interface and deployment.

Development notes: [Progress](PROGRESS.MD) · [Steps](STEPS.md) · [Pipeline explanation](STEP_EXPLANATION.md).
