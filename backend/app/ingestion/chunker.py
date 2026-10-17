

# Fixed Size Chunks with Overlap
def chunk_pages(pages, metadata, chunk_size=500, overlap=100):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    chunks = []

    for page in pages:
        text = page['text']
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if not chunk_text.strip():
                start += chunk_size - overlap
                continue

            chunks.append({
                "document_id":metadata["document_id"],
                "filename":metadata["filename"],
                "department":metadata["department"],
                "file_type":metadata["file_type"],
                "page":page_number,
                "text":chunk_text
            })
            start += chunk_size - overlap
    return chunks
