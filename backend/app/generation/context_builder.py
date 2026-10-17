def build_context(results):
    context_parts = []
    for index, result in enumerate(results, start=1):
        chunk = result["chunk"]
        document_id = chunk["document_id"]
        filename = chunk["filename"]
        page = chunk["page"]
        text = chunk["text"]
        
        source_block = (
            f"[SOURCE {index}]\n"
            f"Document ID: {document_id}\n"
            f"Filename: {filename}\n"
            f"Page: {page}\n\n"
            f"{text}"
        )
        
        context_parts.append(source_block)
        
    return "\n\n".join(context_parts)