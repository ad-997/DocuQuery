def normalize_document_id(document_id):
    return document_id.lower()


def hit_at_k(results, expected_document_ids, k):
    expected = {
        normalize_document_id(document_id)
        for document_id in expected_document_ids
    }

    retrieved = {
        normalize_document_id(result["document_id"])
        for result in results[:k]
    }

    return 1 if expected & retrieved else 0


def document_recall_at_k(results, expected_document_ids, k):
    expected = {
        normalize_document_id(document_id)
        for document_id in expected_document_ids
    }

    if not expected:
        return None

    retrieved = {
        normalize_document_id(result["document_id"])
        for result in results[:k]
    }

    relevant_retrieved = expected & retrieved

    return len(relevant_retrieved) / len(expected)


def reciprocal_rank(results, expected_document_ids):
    expected = {
        normalize_document_id(document_id)
        for document_id in expected_document_ids
    }

    if not expected:
        return None

    for rank, result in enumerate(results, start=1):

        document_id = normalize_document_id(
            result["document_id"]
        )

        if document_id in expected:
            return 1 / rank

    return 0