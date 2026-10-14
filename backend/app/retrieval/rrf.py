def reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_k=3):
    fused = {}
    
    #Dense rankings
    for rank, result in enumerate(dense_results, start=1):
        chunk = result.payload
        
        key = (
            chunk["document_id"],
            chunk["page"],
            chunk["text"]
        )
        
        if key not in fused:
            fused[key] = {
                "chunk":chunk,
                "score":0
            }
        
        fused[key]["score"] += 1/(k+rank)

    # BM25 Rankings
    for rank, result in enumerate(bm25_results, start=1):
        chunk = result["chunk"]
        
        key = (
            chunk["document_id"],
            chunk["page"],
            chunk["text"]
        )
        
        if key not in fused:
            fused[key] = {
                "chunk": chunk,
                "score": 0
            }
        
        fused[key]["score"] += 1 / (k + rank)
    
    ranked_results = sorted(
        fused.values(),
        key=lambda result: result["score"],
        reverse=True
    )
    
    return ranked_results[:top_k]