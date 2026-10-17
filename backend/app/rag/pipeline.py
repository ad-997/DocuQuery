"""Retrieve evidence and generate a grounded answer."""

import sys
from pathlib import Path

# Allow running this file directly from any directory.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.embedding.embedder import embed_query
from app.generation.context_builder import build_context
from app.generation.generator import generate_answer
from app.ingestion.pipeline import load_corpus_chunks
from app.reranking.reranker import rerank
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.sparse import search_bm25
from app.vectorstore.qdrant_store import search_qdrant


def run_rag_pipeline(question, chunks):
    query_embedding = embed_query(question)
    dense_results = search_qdrant(query_embedding, top_k=10)
    bm25_results = search_bm25(question, chunks, top_k=10)
    hybrid_results = reciprocal_rank_fusion(dense_results, bm25_results, top_k=10)
    results = rerank(question, hybrid_results, top_k=3)

    context = build_context(results)
    answer = generate_answer(question, context)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "document_id": result["chunk"]["document_id"],
                "filename": result["chunk"]["filename"],
                "page": result["chunk"]["page"],
                "score": result["score"],
            }
            for result in results
        ],
    }


if __name__ == "__main__":
    data_directory = Path(__file__).resolve().parents[3] / "data/raw"
    chunks = load_corpus_chunks(data_directory)
    questions = [
        "How much parental leave can an employee take?",
        "Who is eligible for parental leave?",
        "When does the parental leave policy become effective?",
        "Does a legal hold allow financial close records to be deleted after seven years?",
        "How did the annual leave allowance change from 2024 to the current policy?",
        "What is the cafeteria menu on Mars next Tuesday?",
    ]

    for number, question in enumerate(questions, start=1):
        print("\n" + "=" * 72)
        print(f"QUESTION {number}/{len(questions)}")
        print(question)
        print("=" * 72, flush=True)

        result = run_rag_pipeline(question, chunks)

        print("\nLLM ANSWER")
        print("-" * 72)
        print(result["answer"])

        print("\nRETRIEVED SOURCES")
        print("-" * 72)
        for index, source in enumerate(result["sources"], start=1):
            print(f"[SOURCE {index}] {source['filename']} | Page {source['page']}")
            print(f"  Document: {source['document_id']} | Reranker score: {source['score']:.3f}")
        print(flush=True)
