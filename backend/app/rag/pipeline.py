"""Retrieve evidence and generate an answer with source labels."""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.embedding.embedder import embed_query
from app.generation.context_builder import build_context
from app.generation.generator import generate_answer
from app.reranking.reranker import rerank
from app.retrieval.rrf import reciprocal_rank_fusion
from app.vectorstore.qdrant_store import search_qdrant


def run_rag_pipeline(question, bm25_retriever, dense_k=10, bm25_k=10,
                     hybrid_k=10, final_k=3, allowed_document_ids=None):
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")
    if min(dense_k, bm25_k, hybrid_k, final_k) < 0:
        raise ValueError("Retrieval limits must be nonnegative")
    options = {} if allowed_document_ids is None else {'allowed_document_ids': allowed_document_ids}
    if allowed_document_ids is not None and not allowed_document_ids:
        return {'question': question, 'answer': generate_answer(question, ''), 'sources': []}
    dense_results = search_qdrant(embed_query(question), top_k=dense_k, **options)
    sparse_results = bm25_retriever.search(question, top_k=bm25_k, **options)
    hybrid = reciprocal_rank_fusion(dense_results, sparse_results, top_k=hybrid_k)
    results = rerank(question, hybrid, top_k=final_k)
    answer = generate_answer(question, build_context(results))
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"document_id": r["chunk"]["document_id"],
             "filename": r["chunk"]["filename"],
             "page": r["chunk"]["page"], "score": r["score"]}
            for r in results
        ],
    }


if __name__ == "__main__":
    from app.services.corpus_service import corpus_service
    result = run_rag_pipeline(
        "How much parental leave can an employee take?", corpus_service.retriever
    )
    print(result["answer"])
    for source in result["sources"]:
        print(source)
