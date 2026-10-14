import re
import numpy as np

from rank_bm25 import BM25Okapi

# rank_bm25 expects tokenized docs, we're tokenizing it ourself

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower()) # split into a list

def search_bm25(query, chunks, top_k=10):
    tokenized_chunk = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]
    
    #builds a search index over our chunks
    bm25 = BM25Okapi(tokenized_chunk)
    
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(-scores)[:top_k]
    results = []
    for index in top_indices:
        results.append({
            "index": int(index),
            "score": float(scores[index]),
            "chunk": chunks[index]
        })
    return results