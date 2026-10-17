import unittest
from types import SimpleNamespace

from evaluate_retrieval import evaluate_queries
from app.retrieval.diversify import diversify_results
from app.evaluation.retrieval_metrics import document_recall_at_k, reciprocal_rank


def result(document, text=None):
    return {"chunk": {"document_id": document, "page": 1, "text": text or document}, "score": 1.0}


class RetrievalTests(unittest.TestCase):
    def test_crowded_document_allows_second_source(self):
        results = [result("COM-001", str(i)) for i in range(5)] + [result("COM-005")]
        selected = diversify_results(results)
        self.assertEqual([r["chunk"]["document_id"] for r in selected], ["COM-001", "COM-001", "COM-005"])
        self.assertEqual(len(results), 6)

    def test_limits_and_case(self):
        self.assertEqual(diversify_results([result("A")], top_k=0), [])
        self.assertEqual(len(diversify_results([result("A"), result("a")], max_chunks_per_document=1)), 1)
        with self.assertRaises(ValueError):
            diversify_results([], max_chunks_per_document=0)

    def test_metrics_count_documents_once_but_keep_chunk_rank(self):
        rows = [{"document_id": d} for d in ("a", "A", "B")]
        self.assertEqual(document_recall_at_k(rows, ["A", "b"], 2), 0.5)
        self.assertEqual(document_recall_at_k(rows, ["A", "b"], 3), 1)
        self.assertEqual(reciprocal_rank(rows, ["b"]), 1 / 3)

    def test_baseline_depth_and_failure_comparison(self):
        candidates = [result("A", str(i)) for i in range(15)] + [result("B")]
        dense = [SimpleNamespace(payload=r["chunk"], score=1) for r in candidates]
        calls = []
        def rerank(query, rows, top_k):
            calls.append(len(rows))
            return rows[:top_k]
        queries = [
            {"query_id": "CHALLENGE-08", "question": "hold?", "expected_document_ids": ["A", "B"]},
            {"query_id": "ACL-1", "question": "secret?", "expected_document_ids": ["A"]},
            {"query_id": "EMPTY", "question": "none?", "expected_document_ids": []},
        ]
        report = evaluate_queries(queries, [], lambda q: q, lambda q, top_k: dense,
                                  lambda q, chunks, top_k: candidates, rerank)
        self.assertEqual(calls, [10, 3])
        self.assertEqual(report["metrics"]["Reranker"]["recall_5"], 0.5)
        self.assertEqual(report["metrics"]["Diversified+Reranker"]["recall_5"], 1)
        self.assertEqual(report["comparisons"]["Reranker → Diversified+Reranker"]["recall_improvements"], ["CHALLENGE-08"])
        self.assertEqual(report["skipped"], {"acl": 1, "no_expected_documents": 1})

    def test_no_eligible_queries(self):
        with self.assertRaisesRegex(ValueError, "No eligible"):
            evaluate_queries([], [], None, None, None, None)


if __name__ == "__main__":
    unittest.main()
