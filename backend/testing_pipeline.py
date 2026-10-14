from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.ingestion.metadata import extract_metadata

from app.embedding.embedder import embed_chunks, embed_query

from app.retrieval.sparse import search_bm25
from app.retrieval.rrf import reciprocal_rank_fusion

from app.reranking.reranker import rerank

from app.vectorstore.qdrant_store import (
    create_collection,
    upsert_chunks,
    search_qdrant
)


# =========================================================
# CONFIG
# =========================================================

file_path = "/Users/ad/DocuQuery/data/raw/HR/hr-006_parental_leave_guide.pdf"

queries = [
    "How much parental leave can an employee take?",
    "Who is eligible for parental leave?",
    "When does the parental leave policy become effective?"
]

# Choose what you want to test
RUN_DENSE = False
RUN_BM25 = False
RUN_HYBRID = False
RUN_RERANKER = True


# =========================================================
# INGESTION
# =========================================================

create_collection()

metadata = extract_metadata(file_path)

pages = load_pdf(file_path)

chunks = chunk_pages(
    pages,
    metadata,
    chunk_size=500,
    overlap=100
)

chunk_embeddings = embed_chunks(chunks)

upsert_chunks(
    chunks,
    chunk_embeddings
)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))


# =========================================================
# TEST FUNCTIONS
# =========================================================

def test_dense(query):
    query_embedding = embed_query(query)

    results = search_qdrant(
        query_embedding,
        top_k=10
    )

    print("\n------- DENSE RESULTS -------")

    for i, result in enumerate(results, start=1):
        print(f"\nDENSE RESULT {i}")
        print("Score:", result.score)
        print("Document:", result.payload["filename"])
        print("Page:", result.payload["page"])
        print("Text:")
        print(result.payload["text"])

    return results


def test_bm25(query):
    results = search_bm25(
        query,
        chunks,
        top_k=10
    )

    print("\n------- BM25 RESULTS -------")

    for i, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\nBM25 RESULT {i}")
        print("Score:", result["score"])
        print("Document:", chunk["filename"])
        print("Page:", chunk["page"])
        print("Text:")
        print(chunk["text"])

    return results


def test_hybrid(query):
    query_embedding = embed_query(query)

    dense_results = search_qdrant(
        query_embedding,
        top_k=10
    )

    bm25_results = search_bm25(
        query,
        chunks,
        top_k=10
    )

    results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=10
    )

    print("\n------- HYBRID / RRF RESULTS -------")

    for i, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\nHYBRID RESULT {i}")
        print("RRF Score:", result["score"])
        print("Document:", chunk["filename"])
        print("Page:", chunk["page"])
        print("Text:")
        print(chunk["text"])

    return results


def test_reranker(query):
    query_embedding = embed_query(query)

    dense_results = search_qdrant(
        query_embedding,
        top_k=10
    )

    bm25_results = search_bm25(
        query,
        chunks,
        top_k=10
    )

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=10
    )

    results = rerank(
        query,
        hybrid_results,
        top_k=3
    )

    print("\n------- RERANKED RESULTS -------")

    for i, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print(f"\nRERANKED RESULT {i}")
        print("QUERY:", query)
        print("Reranker Score:", result["score"])
        print("Document:", chunk["filename"])
        print("Page:", chunk["page"])
        print("Text:")
        print(chunk["text"])

    return results


# =========================================================
# RUN TESTS
# =========================================================

for query in queries:

    print("\n\n========================================")
    print("QUERY:", query)
    print("========================================")

    if RUN_DENSE:
        test_dense(query)

    if RUN_BM25:
        test_bm25(query)

    if RUN_HYBRID:
        test_hybrid(query)

    if RUN_RERANKER:
        test_reranker(query)