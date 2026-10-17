from pathlib import Path

from ..embedding.embedder import embed_chunks
from ..vectorstore.qdrant_store import upsert_chunks

from .pdf_loader import load_pdf
from .metadata import extract_metadata
from .chunker import chunk_pages


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

    if not directory.is_dir():
        raise ValueError(f"Corpus directory does not exist: {directory}")

    all_chunks = []

    for file_path in sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"):
        chunks = load_document_chunks(file_path)

        all_chunks.extend(chunks)

    return all_chunks


# =========================================================
# INGEST ONE DOCUMENT INTO QDRANT
# =========================================================

def ingest_document(file_path):
    chunks = load_document_chunks(file_path)

    if not chunks:
        raise ValueError("PDF contains no extractable text; OCR may be required")

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

    if not directory.is_dir():
        raise ValueError(f"Corpus directory does not exist: {directory}")

    all_chunks = []

    for file_path in sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"):
        print(f"Ingesting: {file_path}")

        chunks = ingest_document(file_path)

        all_chunks.extend(chunks)

    return all_chunks
