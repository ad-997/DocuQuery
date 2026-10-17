from pathlib import Path

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.metadata import extract_metadata
from app.ingestion.chunker import chunk_pages
from app.embedding.embedder import embed_chunks
from app.vectorstore.qdrant_store import upsert_chunks


# =========================================================
# LOAD + CHUNK ONE DOCUMENT
# =========================================================

def load_document_chunks(file_path):
    metadata = extract_metadata(file_path)

    pages = load_pdf(file_path)

    chunks = chunk_pages(
        pages,
        metadata,
        chunk_size=500,
        overlap=100
    )

    return chunks


# =========================================================
# LOAD + CHUNK ENTIRE CORPUS
# DOES NOT EMBED OR WRITE TO QDRANT
# =========================================================

def load_corpus_chunks(directory):
    directory = Path(directory)

    all_chunks = []

    for file_path in directory.rglob("*.pdf"):
        chunks = load_document_chunks(file_path)

        all_chunks.extend(chunks)

    return all_chunks


# =========================================================
# INGEST ONE DOCUMENT INTO QDRANT
# =========================================================

def ingest_document(file_path):
    chunks = load_document_chunks(file_path)

    embeddings = embed_chunks(chunks)

    upsert_chunks(
        chunks,
        embeddings
    )

    return chunks


# =========================================================
# INGEST ENTIRE DIRECTORY INTO QDRANT
# =========================================================

def ingest_directory(directory):
    directory = Path(directory)

    all_chunks = []

    for file_path in directory.rglob("*.pdf"):
        print(f"Ingesting: {file_path}")

        chunks = ingest_document(file_path)

        all_chunks.extend(chunks)

    return all_chunks