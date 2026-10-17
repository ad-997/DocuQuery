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

## Retrieval evaluation

The evaluation script compares Dense, BM25, RRF, RRF + Reranker, and
RRF + Diversification + Reranker against the labelled PDF corpus. It uses a
separate temporary Qdrant collection and deletes it after the run.
Start Qdrant on localhost:6333 and install `qdrant-client` and `rank-bm25`
in addition to the dependencies above, then run from the project root:

```sh
.venv/bin/python backend/evaluate_retrieval.py
PYTHONPATH=backend .venv/bin/python -m unittest discover -s backend/tests -v
```

The report is saved to `data/retrieval_evaluation.json`, including per-query
rankings, Recall@1/3/5, MRR@5, improvements, regressions, and candidate diagnostics.
Use `--help` for data and output paths. ACL queries and queries without expected
documents are counted separately and excluded from retrieval quality averages.
This benchmark does not verify access permissions or answer quality.

The original RRF and reranker baselines use 10 candidates from each retriever.
The diversification experiment uses 30, retains at most two chunks per document,
and reranks up to 10 candidates into a final top five. It changes both candidate
depth and document diversity, so differences cannot be attributed to diversity alone.
The document cap applies before reranking; it does not guarantee every relevant
document survives into the final five.

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

- No answer generation, permission enforcement, or search API yet.
- Scanned PDFs need OCR; dependency locking and ingestion validation remain future work.
- Document versions and current versus archived policies are not filtered yet.
- The small labelled corpus is a development benchmark, not a general accuracy claim.

Development notes: [Progress](PROGRESS.MD) · [Steps](STEPS.md) · [Pipeline explanation](STEP_EXPLANATION.md).
