"""Limit document crowding while preserving the fused ranking order."""


def diversify_results(results, top_k=10, max_chunks_per_document=2):
    if top_k < 0 or max_chunks_per_document < 1:
        raise ValueError("top_k must be nonnegative and document limit positive")
    selected = []
    counts = {}
    for result in results:
        if len(selected) >= top_k:
            break
        document_id = result["chunk"]["document_id"].lower()
        if counts.get(document_id, 0) >= max_chunks_per_document:
            continue
        selected.append(result)
        counts[document_id] = counts.get(document_id, 0) + 1
    return selected
