import numpy as np

def search_chunks(query_embeddings, chunk_embeddings, chunks, top_k=3):
    scores = chunk_embeddings @ query_embeddings # Dot Product
    top_indices = np.argsort(-scores)[:top_k]
    results = []

    # print(scores) will show scores in an array [0.6710, 0.6716, 0.4995]
    # print(top_indices) will print top indices

    for index in top_indices:
        results.append({
            "score": float(scores[index]),
            "chunk": chunks[index]
        })

    return results

