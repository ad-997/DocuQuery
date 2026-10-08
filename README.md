# DocuQuery

DocuQuery is an enterprise RAG knowledge platform designed to help employees search internal documents such as policies, manuals, SOPs, and technical documentation.

The project is currently under development.

## Current Features

- FastAPI backend setup
- PDF text extraction using `pypdf`
- Page-wise document processing
- Basic metadata extraction
- Fixed-size text chunking with overlap
- Sentence Transformer embeddings
- Query embeddings
- Basic in-memory dense retrieval using cosine similarity

## Current Pipeline

```text
PDF
↓
Load text page-by-page
↓
Extract metadata
↓
Split text into chunks
↓
Generate chunk embeddings
↓
Generate query embedding
↓
Compare similarity
↓
Return top matching chunks
```

## Project Structure

```text
DocuQuery/
├── backend/
│   ├── app/
│   │   ├── embedding/
│   │   ├── ingestion/
│   │   └── retrieval/
│   └── testing_pipeline.py
│
├── data/
│   └── raw/
│
└── README.md
```

## Tech Stack

- Python
- FastAPI
- Uvicorn
- PyPDF
- Sentence Transformers
- NumPy

## Planned Features

- Qdrant vector database
- Persistent document embeddings
- BM25 sparse retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion (RRF)
- Reranking
- LLM-based answer generation
- Source and page citations
- JWT authentication
- Role-Based Access Control
- RAG evaluation
- React frontend
- Docker deployment

## Status

Currently working on the retrieval pipeline.

The next major step is integrating Qdrant for vector storage and retrieval.