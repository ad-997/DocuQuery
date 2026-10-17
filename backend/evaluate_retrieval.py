"""Compare retrieval strategies and save reproducible per-query diagnostics."""
import argparse
import json
from pathlib import Path

from app.evaluation.retrieval_metrics import document_recall_at_k, reciprocal_rank
from app.retrieval.diversify import diversify_results
from app.retrieval.rrf import reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("Dense", "BM25", "RRF", "Reranker", "Diversified+Reranker")


def score_results(results, expected):
    return {
        **{f"recall_{k}": document_recall_at_k(results, expected, k) for k in (1, 3, 5)},
        "rr": reciprocal_rank(results, expected),
    }


def evaluate_queries(queries, chunks, embed_query, search_dense, search_sparse, rerank):
    """Keep original 10-candidate baselines; diversify a separate depth-30 pool."""
    records = {method: {} for method in METHODS}
    diagnostics = {}
    skipped = {"acl": 0, "no_expected_documents": 0}
    for item in queries:
        query_id = item["query_id"]
        if query_id.startswith("ACL-"):
            skipped["acl"] += 1
            continue
        expected = item["expected_document_ids"]
        if not expected:
            skipped["no_expected_documents"] += 1
            continue
        query = item["question"]
        dense = search_dense(embed_query(query), top_k=30)
        sparse = search_sparse(query, chunks, top_k=30)
        baseline = reciprocal_rank_fusion(dense[:10], sparse[:10], top_k=10)
        deep = reciprocal_rank_fusion(dense, sparse, top_k=30)
        diverse = diversify_results(deep, top_k=10, max_chunks_per_document=2)
        ranked = {
            "Dense": [{"chunk": result.payload, "score": result.score} for result in dense[:5]],
            "BM25": sparse[:5],
            "RRF": baseline[:5],
            "Reranker": rerank(query, baseline, top_k=5),
            "Diversified+Reranker": rerank(query, diverse, top_k=5),
        }
        for method, results in ranked.items():
            normalized = [{**r["chunk"], "score": float(r["score"])} for r in results]
            records[method][query_id] = {
                "question": query, "expected": expected,
                "retrieved": [r["document_id"] for r in normalized],
                "results": normalized, **score_results(normalized, expected),
            }
        diagnostics[query_id] = {
            "rrf_top_30": [r["chunk"]["document_id"] for r in deep],
            "diversified_candidates": [r["chunk"]["document_id"] for r in diverse],
        }
        print(f"Evaluated {query_id}", flush=True)
    if not records["Dense"]:
        raise ValueError("No eligible queries with expected documents")
    metrics = {
        method: {key: sum(r[key] for r in rows.values()) / len(rows)
                 for key in ("recall_1", "recall_3", "recall_5", "rr")}
        for method, rows in records.items()
    }
    comparisons = {}
    for before, after in (("RRF", "Reranker"), ("Reranker", "Diversified+Reranker")):
        comparisons[f"{before} → {after}"] = {
            "improvements": [q for q in records[before] if records[after][q]["rr"] > records[before][q]["rr"]],
            "regressions": [q for q in records[before] if records[after][q]["rr"] < records[before][q]["rr"]],
            "recall_improvements": [q for q in records[before] if records[after][q]["recall_5"] > records[before][q]["recall_5"]],
            "recall_regressions": [q for q in records[before] if records[after][q]["recall_5"] < records[before][q]["recall_5"]],
        }
    return {"evaluated_queries": len(records["Dense"]), "skipped": skipped,
            "metrics": metrics, "queries": records, "comparisons": comparisons,
            "candidate_diagnostics": diagnostics,
            "configuration": {"baseline_depth": 10, "diversified_depth": 30,
                              "diversified_candidates": 10, "max_chunks_per_document": 2,
                              "final_top_k": 5, "mrr_cutoff": 5}}


def print_report(report):
    print(f"\nEvaluated queries: {report['evaluated_queries']}; skipped: {report['skipped']}")
    print(f"{'Method':<24}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR@5':>8}")
    for method, metrics in report["metrics"].items():
        print(f"{method:<24}" + "".join(f"{metrics[key]:>8.3f}" for key in ("recall_1", "recall_3", "recall_5", "rr")))
    for label, comparison in report["comparisons"].items():
        print(f"\n{label}")
        for kind, ids in comparison.items():
            print(f"  {kind}: {len(ids)} {ids}")
            for query_id in ids:
                before, after = label.split(" → ")
                old, new = (report["queries"][m][query_id] for m in (before, after))
                print(f"    {query_id}: {new['question']}\n      Expected: {new['expected']}\n      Before: {old['retrieved']}\n      After: {new['retrieved']}\n      RR: {old['rr']:.3f} → {new['rr']:.3f}; R@5: {old['recall_5']:.3f} → {new['recall_5']:.3f}")
    for method, rows in report["queries"].items():
        failures = {q: r for q, r in rows.items() if r["recall_5"] < 1}
        print(f"\n{method} Top-5 failures: {len(failures)}")
        for query_id, row in failures.items():
            print(f"  {query_id}: expected {row['expected']}, retrieved {row['retrieved']}, R@5={row['recall_5']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--evaluation-file", type=Path, default=ROOT / "data/raw/evaluation_queries.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/retrieval_evaluation.json")
    args = parser.parse_args()
    queries = json.loads(args.evaluation_file.read_text())
    # Heavy models and database access only load when explicitly running the CLI.
    from app.embedding.embedder import embed_query
    from app.ingestion.pipeline import ingest_directory
    from app.reranking.reranker import rerank
    from app.retrieval.sparse import search_bm25
    from app.vectorstore.qdrant_store import client
    from qdrant_client.models import Distance, VectorParams

    # Isolate evaluation from the application's existing collection.
    from app.vectorstore import qdrant_store
    original_collection = qdrant_store.COLLECTION_NAME
    from uuid import uuid4
    collection = f"docuquery_eval_{uuid4().hex}"
    client.create_collection(collection, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    qdrant_store.COLLECTION_NAME = collection
    try:
        chunks = ingest_directory(args.data_directory)
        if not chunks:
            raise ValueError("No PDF chunks found in data directory")
        report = evaluate_queries(queries, chunks, embed_query, qdrant_store.search_qdrant, search_bm25, rerank)
        report["configuration"].update({"data_directory": str(args.data_directory.resolve()),
                                          "evaluation_file": str(args.evaluation_file.resolve()),
                                          "corpus_chunks": len(chunks)})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print_report(report)
        print(f"\nSaved report: {args.output}")
    finally:
        qdrant_store.COLLECTION_NAME = original_collection
        client.delete_collection(collection)


if __name__ == "__main__":
    main()
