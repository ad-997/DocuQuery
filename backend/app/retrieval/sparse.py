"""Reusable BM25 index and a convenience wrapper for scripts."""
import re
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        tokenized = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized) if any(tokenized) else None

    def search(self, query, top_k=10, allowed_document_ids=None):
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if allowed_document_ids is not None:
            allowed = set(allowed_document_ids)
            return BM25Retriever([chunk for chunk in self.chunks if chunk["document_id"] in allowed]).search(query, top_k)
        tokens = tokenize(query)
        if self.bm25 is None or not tokens or top_k == 0:
            return []
        scores = self.bm25.get_scores(tokens)
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {"index": int(i), "score": float(scores[i]), "chunk": self.chunks[i]}
            for i in indices
        ]


def search_bm25(query, chunks, top_k=10):
    return BM25Retriever(chunks).search(query, top_k=top_k)
