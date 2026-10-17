import json

from app.embedding.embedder import embed_query
from app.ingestion.pipeline import ingest_directory

from app.evaluation.retrieval_metrics import (
    document_recall_at_k,
    reciprocal_rank
)

from app.vectorstore.qdrant_store import (
    reset_collection,
    search_qdrant
)

from app.reranking.reranker import rerank

from app.retrieval.sparse import search_bm25
from app.retrieval.rrf import reciprocal_rank_fusion


# =========================================
# CONFIGURATION
# =========================================

DATA_DIRECTORY = "/Users/ad/DocuQuery/data/raw"

EVALUATION_FILE = (
    "/Users/ad/DocuQuery/data/raw/evaluation_queries.json"
)


# =========================================
# RESET QDRANT FOR CLEAN EVALUATION
# =========================================

print("\n========================================")
print("RESETTING QDRANT COLLECTION")
print("========================================")

reset_collection()


# =========================================
# INGEST ENTIRE DOCUMENT CORPUS
# =========================================

print("\n========================================")
print("INGESTING DOCUMENT CORPUS")
print("========================================")

chunks = ingest_directory(DATA_DIRECTORY)

print("\n========================================")
print("CORPUS INGESTION COMPLETE")
print("========================================")

print("Total corpus chunks:", len(chunks))


# =========================================
# LOAD EVALUATION DATASET
# =========================================

with open(EVALUATION_FILE, "r") as file:
    evaluation_queries = json.load(file)


print("\n========================================")
print("EVALUATION DATASET LOADED")
print("========================================")

print(
    "Total evaluation queries:",
    len(evaluation_queries)
)


# =========================================================
# DENSE RETRIEVAL EVALUATION
# =========================================================

dense_recall_1_scores = []
dense_recall_3_scores = []
dense_recall_5_scores = []
dense_rr_scores = []

dense_evaluated_queries = 0
skipped_acl_queries = 0


for item in evaluation_queries:

    query_id = item["query_id"]

    # Security evaluation will be handled separately
    if query_id.startswith("ACL-"):
        skipped_acl_queries += 1
        continue

    dense_evaluated_queries += 1

    query = item["question"]
    expected_document_ids = item["expected_document_ids"]

    # -------------------------------------
    # CREATE QUERY EMBEDDING
    # -------------------------------------

    query_embedding = embed_query(query)

    # -------------------------------------
    # DENSE SEARCH
    # -------------------------------------

    dense_results = search_qdrant(
        query_embedding,
        top_k=5
    )

    # -------------------------------------
    # NORMALIZE RESULT FORMAT
    # -------------------------------------

    results = []

    for result in dense_results:

        results.append({
            "document_id": result.payload["document_id"],
            "page": result.payload["page"],
            "text": result.payload["text"],
            "score": result.score
        })

    # -------------------------------------
    # METRICS
    # -------------------------------------

    recall_1 = document_recall_at_k(
        results,
        expected_document_ids,
        k=1
    )

    recall_3 = document_recall_at_k(
        results,
        expected_document_ids,
        k=3
    )

    recall_5 = document_recall_at_k(
        results,
        expected_document_ids,
        k=5
    )

    rr = reciprocal_rank(
        results,
        expected_document_ids
    )

    dense_recall_1_scores.append(recall_1)
    dense_recall_3_scores.append(recall_3)
    dense_recall_5_scores.append(recall_5)
    dense_rr_scores.append(rr)


# =========================================
# CALCULATE DENSE FINAL METRICS
# =========================================

dense_recall_1 = (
    sum(dense_recall_1_scores)
    / dense_evaluated_queries
)

dense_recall_3 = (
    sum(dense_recall_3_scores)
    / dense_evaluated_queries
)

dense_recall_5 = (
    sum(dense_recall_5_scores)
    / dense_evaluated_queries
)

dense_mrr = (
    sum(dense_rr_scores)
    / dense_evaluated_queries
)


print("\n\n========================================")
print("DENSE RETRIEVAL RESULTS")
print("========================================")

print(f"Evaluated Queries: {dense_evaluated_queries}")
print(f"Skipped ACL Queries: {skipped_acl_queries}")
print(f"Document Recall@1: {dense_recall_1:.3f}")
print(f"Document Recall@3: {dense_recall_3:.3f}")
print(f"Document Recall@5: {dense_recall_5:.3f}")
print(f"MRR:               {dense_mrr:.3f}")


# =========================================================
# BM25 RETRIEVAL EVALUATION
# =========================================================

bm25_recall_1_scores = []
bm25_recall_3_scores = []
bm25_recall_5_scores = []
bm25_rr_scores = []

bm25_evaluated_queries = 0


for item in evaluation_queries:

    query_id = item["query_id"]

    if query_id.startswith("ACL-"):
        continue

    bm25_evaluated_queries += 1

    query = item["question"]
    expected_document_ids = item["expected_document_ids"]

    # -------------------------------------
    # BM25 SEARCH
    # -------------------------------------

    bm25_results = search_bm25(
        query,
        chunks,
        top_k=5
    )

    # -------------------------------------
    # NORMALIZE RESULT FORMAT
    # -------------------------------------

    results = []

    for result in bm25_results:

        chunk = result["chunk"]

        results.append({
            "document_id": chunk["document_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": result["score"]
        })

    # -------------------------------------
    # METRICS
    # -------------------------------------

    recall_1 = document_recall_at_k(
        results,
        expected_document_ids,
        k=1
    )

    recall_3 = document_recall_at_k(
        results,
        expected_document_ids,
        k=3
    )

    recall_5 = document_recall_at_k(
        results,
        expected_document_ids,
        k=5
    )

    rr = reciprocal_rank(
        results,
        expected_document_ids
    )

    bm25_recall_1_scores.append(recall_1)
    bm25_recall_3_scores.append(recall_3)
    bm25_recall_5_scores.append(recall_5)
    bm25_rr_scores.append(rr)


# =========================================
# CALCULATE BM25 FINAL METRICS
# =========================================

bm25_recall_1 = (
    sum(bm25_recall_1_scores)
    / bm25_evaluated_queries
)

bm25_recall_3 = (
    sum(bm25_recall_3_scores)
    / bm25_evaluated_queries
)

bm25_recall_5 = (
    sum(bm25_recall_5_scores)
    / bm25_evaluated_queries
)

bm25_mrr = (
    sum(bm25_rr_scores)
    / bm25_evaluated_queries
)


print("\n\n========================================")
print("BM25 RETRIEVAL RESULTS")
print("========================================")

print(f"Evaluated Queries: {bm25_evaluated_queries}")
print(f"Document Recall@1: {bm25_recall_1:.3f}")
print(f"Document Recall@3: {bm25_recall_3:.3f}")
print(f"Document Recall@5: {bm25_recall_5:.3f}")
print(f"MRR:               {bm25_mrr:.3f}")


# =========================================================
# HYBRID / RRF RETRIEVAL EVALUATION
# =========================================================

rrf_recall_1_scores = []
rrf_recall_3_scores = []
rrf_recall_5_scores = []
rrf_rr_scores = []

rrf_evaluated_queries = 0


for item in evaluation_queries:

    query_id = item["query_id"]

    if query_id.startswith("ACL-"):
        continue

    rrf_evaluated_queries += 1

    query = item["question"]
    expected_document_ids = item["expected_document_ids"]

    # -------------------------------------
    # DENSE CANDIDATES
    # -------------------------------------

    query_embedding = embed_query(query)

    dense_results = search_qdrant(
        query_embedding,
        top_k=10
    )

    # -------------------------------------
    # BM25 CANDIDATES
    # -------------------------------------

    bm25_results = search_bm25(
        query,
        chunks,
        top_k=10
    )

    # -------------------------------------
    # RECIPROCAL RANK FUSION
    # -------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=5
    )

    # -------------------------------------
    # NORMALIZE RESULT FORMAT
    # -------------------------------------

    results = []

    for result in hybrid_results:

        chunk = result["chunk"]

        results.append({
            "document_id": chunk["document_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": result["score"]
        })

    # -------------------------------------
    # METRICS
    # -------------------------------------

    recall_1 = document_recall_at_k(
        results,
        expected_document_ids,
        k=1
    )

    recall_3 = document_recall_at_k(
        results,
        expected_document_ids,
        k=3
    )

    recall_5 = document_recall_at_k(
        results,
        expected_document_ids,
        k=5
    )

    rr = reciprocal_rank(
        results,
        expected_document_ids
    )

    rrf_recall_1_scores.append(recall_1)
    rrf_recall_3_scores.append(recall_3)
    rrf_recall_5_scores.append(recall_5)
    rrf_rr_scores.append(rr)


# =========================================
# CALCULATE RRF FINAL METRICS
# =========================================

rrf_recall_1 = (
    sum(rrf_recall_1_scores)
    / rrf_evaluated_queries
)

rrf_recall_3 = (
    sum(rrf_recall_3_scores)
    / rrf_evaluated_queries
)

rrf_recall_5 = (
    sum(rrf_recall_5_scores)
    / rrf_evaluated_queries
)

rrf_mrr = (
    sum(rrf_rr_scores)
    / rrf_evaluated_queries
)


print("\n\n========================================")
print("HYBRID / RRF RETRIEVAL RESULTS")
print("========================================")

print(f"Evaluated Queries: {rrf_evaluated_queries}")
print(f"Document Recall@1: {rrf_recall_1:.3f}")
print(f"Document Recall@3: {rrf_recall_3:.3f}")
print(f"Document Recall@5: {rrf_recall_5:.3f}")
print(f"MRR:               {rrf_mrr:.3f}")


# =========================================================
# FINAL RETRIEVAL COMPARISON
# =========================================================

print("\n\n========================================")
print("RETRIEVAL COMPARISON")
print("========================================")

print(
    f"{'Method':<12}"
    f"{'R@1':>10}"
    f"{'R@3':>10}"
    f"{'R@5':>10}"
    f"{'MRR':>10}"
)

print("-" * 52)

print(
    f"{'Dense':<12}"
    f"{dense_recall_1:>10.3f}"
    f"{dense_recall_3:>10.3f}"
    f"{dense_recall_5:>10.3f}"
    f"{dense_mrr:>10.3f}"
)

print(
    f"{'BM25':<12}"
    f"{bm25_recall_1:>10.3f}"
    f"{bm25_recall_3:>10.3f}"
    f"{bm25_recall_5:>10.3f}"
    f"{bm25_mrr:>10.3f}"
)

print(
    f"{'RRF':<12}"
    f"{rrf_recall_1:>10.3f}"
    f"{rrf_recall_3:>10.3f}"
    f"{rrf_recall_5:>10.3f}"
    f"{rrf_mrr:>10.3f}"
)

# =========================================================
# RERANKER EVALUATION
# =========================================================

reranker_recall_1_scores = []
reranker_recall_3_scores = []
reranker_recall_5_scores = []
reranker_rr_scores = []

reranker_evaluated_queries = 0


for item in evaluation_queries:

    query_id = item["query_id"]

    if query_id.startswith("ACL-"):
        continue

    reranker_evaluated_queries += 1

    query = item["question"]
    expected_document_ids = item["expected_document_ids"]

    # -------------------------------------
    # DENSE CANDIDATES
    # -------------------------------------

    query_embedding = embed_query(query)

    dense_results = search_qdrant(
        query_embedding,
        top_k=10
    )

    # -------------------------------------
    # BM25 CANDIDATES
    # -------------------------------------

    bm25_results = search_bm25(
        query,
        chunks,
        top_k=10
    )

    # -------------------------------------
    # RRF CANDIDATES
    # -------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=10
    )

    # -------------------------------------
    # CROSS-ENCODER RERANKING
    # -------------------------------------

    reranked_results = rerank(
        query,
        hybrid_results,
        top_k=5
    )

    # -------------------------------------
    # NORMALIZE RESULT FORMAT
    # -------------------------------------

    results = []

    for result in reranked_results:

        chunk = result["chunk"]

        results.append({
            "document_id": chunk["document_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": result["score"]
        })

    # -------------------------------------
    # METRICS
    # -------------------------------------

    recall_1 = document_recall_at_k(
        results,
        expected_document_ids,
        k=1
    )

    recall_3 = document_recall_at_k(
        results,
        expected_document_ids,
        k=3
    )

    recall_5 = document_recall_at_k(
        results,
        expected_document_ids,
        k=5
    )

    rr = reciprocal_rank(
        results,
        expected_document_ids
    )

    reranker_recall_1_scores.append(recall_1)
    reranker_recall_3_scores.append(recall_3)
    reranker_recall_5_scores.append(recall_5)
    reranker_rr_scores.append(rr)


# =========================================
# CALCULATE RERANKER FINAL METRICS
# =========================================

reranker_recall_1 = (
    sum(reranker_recall_1_scores)
    / reranker_evaluated_queries
)

reranker_recall_3 = (
    sum(reranker_recall_3_scores)
    / reranker_evaluated_queries
)

reranker_recall_5 = (
    sum(reranker_recall_5_scores)
    / reranker_evaluated_queries
)

reranker_mrr = (
    sum(reranker_rr_scores)
    / reranker_evaluated_queries
)


print("\n\n========================================")
print("RERANKER RESULTS")
print("========================================")

print(f"Evaluated Queries: {reranker_evaluated_queries}")
print(f"Document Recall@1: {reranker_recall_1:.3f}")
print(f"Document Recall@3: {reranker_recall_3:.3f}")
print(f"Document Recall@5: {reranker_recall_5:.3f}")
print(f"MRR:               {reranker_mrr:.3f}")

print(
    f"{'Reranker':<12}"
    f"{reranker_recall_1:>10.3f}"
    f"{reranker_recall_3:>10.3f}"
    f"{reranker_recall_5:>10.3f}"
    f"{reranker_mrr:>10.3f}"
)