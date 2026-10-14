from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, results, top_k=3):
    pairs = [
        [query, result["chunk"]["text"]] for result in results
    ]
    
    scores = model.predict(pairs)
    
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