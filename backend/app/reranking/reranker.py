from functools import lru_cache

from sentence_transformers import CrossEncoder

@lru_cache(maxsize=1)
def get_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, results, top_k=3):
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if not results or top_k == 0:
        return []
    pairs = [
        [query, result["chunk"]["text"]] for result in results
    ]
    
    scores = get_model().predict(pairs)
    
    reranked = []
    
    for result, score in zip(results, scores):
        reranked.append({
            "chunk":result["chunk"],
            "score":float(score)
        })
        
    reranked.sort(
        key=lambda result: result["score"],
        reverse=True
    )
    
    return reranked[:top_k]
