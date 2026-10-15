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


# =========================================
# DENSE RETRIEVAL EVALUATION
# =========================================

recall_1_scores = []
recall_3_scores = []
recall_5_scores = []
rr_scores = []

evaluated_queries = 0
skipped_acl_queries = 0


for item in evaluation_queries:

    query_id = item["query_id"]


    # =====================================
    # SKIP ACL / SECURITY QUERIES
    # =====================================

    if query_id.startswith("ACL-"):

        skipped_acl_queries += 1

        print("\n========================================")
        print("SKIPPING SECURITY QUERY:", query_id)
        print("========================================")

        continue


    evaluated_queries += 1


    # =====================================
    # GET QUERY INFORMATION
    # =====================================

    query = item["question"]

    expected_document_ids = (
        item["expected_document_ids"]
    )


    # =====================================
    # CREATE QUERY EMBEDDING
    # =====================================

    query_embedding = embed_query(query)


    # =====================================
    # DENSE SEARCH USING QDRANT
    # =====================================

    dense_results = search_qdrant(
        query_embedding,
        top_k=5
    )


    # =====================================
    # CONVERT QDRANT RESULTS
    # =====================================

    results = []

    for result in dense_results:

        results.append({
            "document_id": (
                result.payload["document_id"]
            ),

            "page": (
                result.payload["page"]
            ),

            "text": (
                result.payload["text"]
            ),

            "score": (
                result.score
            )
        })


    # =====================================
    # CALCULATE METRICS
    # =====================================

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


    reciprocal_rank_score = reciprocal_rank(
        results,
        expected_document_ids
    )


    # =====================================
    # SAVE SCORES
    # =====================================

    recall_1_scores.append(
        recall_1
    )

    recall_3_scores.append(
        recall_3
    )

    recall_5_scores.append(
        recall_5
    )

    rr_scores.append(
        reciprocal_rank_score
    )


    # =====================================
    # PRINT QUERY RESULT
    # =====================================

    print("\n========================================")
    print("Query:", query_id)
    print("========================================")

    print(query)

    print(
        "\nExpected documents:",
        expected_document_ids
    )


    print("\nRetrieved documents:")


    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            rank,
            result["document_id"],
            result["score"]
        )


    print("\nDocument Recall@1:", recall_1)
    print("Document Recall@3:", recall_3)
    print("Document Recall@5:", recall_5)

    print(
        "Reciprocal Rank:",
        reciprocal_rank_score
    )


# =========================================
# CALCULATE FINAL METRICS
# =========================================

dense_recall_1 = (
    sum(recall_1_scores)
    / evaluated_queries
)

dense_recall_3 = (
    sum(recall_3_scores)
    / evaluated_queries
)

dense_recall_5 = (
    sum(recall_5_scores)
    / evaluated_queries
)

dense_mrr = (
    sum(rr_scores)
    / evaluated_queries
)


# =========================================
# FINAL RESULTS
# =========================================

print("\n\n========================================")
print("DENSE RETRIEVAL RESULTS")
print("========================================")

print(
    f"Total Dataset Queries: "
    f"{len(evaluation_queries)}"
)

print(
    f"Evaluated Queries: "
    f"{evaluated_queries}"
)

print(
    f"Skipped ACL Queries: "
    f"{skipped_acl_queries}"
)

print()

print(
    f"Document Recall@1: "
    f"{dense_recall_1:.3f}"
)

print(
    f"Document Recall@3: "
    f"{dense_recall_3:.3f}"
)

print(
    f"Document Recall@5: "
    f"{dense_recall_5:.3f}"
)

print(
    f"MRR:               "
    f"{dense_mrr:.3f}"
)