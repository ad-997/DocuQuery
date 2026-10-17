from app.ingestion.pipeline import load_corpus_chunks

from app.generation.generator import generate_answer

from app.generation.context_builder import build_context

from app.embedding.embedder import embed_query

from app.retrieval.sparse import search_bm25
from app.retrieval.rrf import reciprocal_rank_fusion

from app.reranking.reranker import rerank

from app.vectorstore.qdrant_store import search_qdrant


# =========================================================
# CONFIG
# =========================================================

DATA_DIRECTORY = "/Users/ad/DocuQuery/data/raw"

queries = [
    "How much parental leave can an employee take?",
    "Who is eligible for parental leave?",
    "When does the parental leave policy become effective?",
    "Does a legal hold allow financial close records to be deleted after the normal seven-year retention period?",
    "If financial close records have reached the end of their seven-year retention period but are currently under a legal hold, what should the company do and why?",
    "How did the annual leave allowance change from the 2024 policy to the current policy?"
    
]


# Choose what you want to test

RUN_DENSE = False
RUN_BM25 = False
RUN_HYBRID = False
RUN_RERANKER = True


# =========================================================
# LOAD CORPUS FOR BM25
# =========================================================

print("\n========================================")
print("LOADING DOCUMENT CORPUS")
print("========================================")

# Important:
# This reads PDFs and creates chunks for BM25.
# It DOES NOT generate embeddings.
# It DOES NOT insert anything into Qdrant.

chunks = load_corpus_chunks(
    DATA_DIRECTORY
)

print("Total corpus chunks:", len(chunks))


# =========================================================
# DENSE RETRIEVAL TEST
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

        print(
            "Score:",
            result.score
        )

        print(
            "Document:",
            result.payload["filename"]
        )

        print(
            "Page:",
            result.payload["page"]
        )

        print("Text:")

        print(
            result.payload["text"]
        )

    return results


# =========================================================
# BM25 RETRIEVAL TEST
# =========================================================

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

        print(
            "Score:",
            result["score"]
        )

        print(
            "Document:",
            chunk["filename"]
        )

        print(
            "Page:",
            chunk["page"]
        )

        print("Text:")

        print(
            chunk["text"]
        )

    return results


# =========================================================
# HYBRID / RRF TEST
# =========================================================

def test_hybrid(query):

    query_embedding = embed_query(query)

    # -------------------------------------
    # Dense retrieval
    # -------------------------------------

    dense_results = search_qdrant(
        query_embedding,
        top_k=10
    )

    # -------------------------------------
    # BM25 retrieval
    # -------------------------------------

    bm25_results = search_bm25(
        query,
        chunks,
        top_k=10
    )

    # -------------------------------------
    # Reciprocal Rank Fusion
    # -------------------------------------

    results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=10
    )

    print("\n------- HYBRID / RRF RESULTS -------")

    for i, result in enumerate(results, start=1):

        chunk = result["chunk"]

        print(f"\nHYBRID RESULT {i}")

        print(
            "RRF Score:",
            result["score"]
        )

        print(
            "Document:",
            chunk["filename"]
        )

        print(
            "Page:",
            chunk["page"]
        )

        print("Text:")

        print(
            chunk["text"]
        )

    return results


# =========================================================
# RERANKER TEST
# =========================================================

def test_reranker(query):

    query_embedding = embed_query(query)

    # -------------------------------------
    # Dense candidates
    # -------------------------------------

    dense_results = search_qdrant(
        query_embedding,
        top_k=10
    )

    # -------------------------------------
    # BM25 candidates
    # -------------------------------------

    bm25_results = search_bm25(
        query,
        chunks,
        top_k=10
    )

    # -------------------------------------
    # RRF
    # -------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=10
    )

    # -------------------------------------
    # Cross-encoder reranking
    # -------------------------------------

    results = rerank(
        query,
        hybrid_results,
        top_k=3
    )

    print("\n------- RERANKED RESULTS -------")

    for i, result in enumerate(results, start=1):

        chunk = result["chunk"]

        print(f"\nRERANKED RESULT {i}")

        print(
            "QUERY:",
            query
        )

        print(
            "Reranker Score:",
            result["score"]
        )

        print(
            "Document:",
            chunk["filename"]
        )

        print(
            "Page:",
            chunk["page"]
        )

        print("Text:")

        print(
            chunk["text"]
        )

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

        reranked_results = test_reranker(
            query
        )

        # -------------------------------------
        # BUILD LLM CONTEXT
        # -------------------------------------

        context = build_context(
            reranked_results
        )

        print("\n------- LLM CONTEXT -------")

        print(context)
        
    answer = generate_answer(
    query,
    context
)

    print("\n------- GENERATED ANSWER -------")
    print(answer)