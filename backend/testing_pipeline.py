from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.ingestion.metadata import extract_metadata

from app.embedding.embedder import embed_chunks, embed_query

from app.vectorstore.qdrant_store import (
    reset_collection,
    create_collection,
    upsert_chunks,
    search_qdrant
)


file_path = "/Users/ad/DocuQuery/data/raw/HR/hr-006_parental_leave_guide.pdf"


# -----------------------
# INGESTION
# -----------------------

create_collection()
metadata = extract_metadata(file_path)

pages = load_pdf(file_path)

chunks = chunk_pages(
    pages,
    metadata,
    chunk_size=500,
    overlap=100
)

chunk_embeddings = embed_chunks(chunks)

print("\nTotal pages:", len(pages))
print("Total chunks:", len(chunks))

upsert_chunks(
    chunks,
    chunk_embeddings
)


# -----------------------
# RETRIEVAL
# -----------------------

queries = [
    "How much parental leave can an employee take?",
    "Who is eligible for parental leave?",
    "When does the parental leave policy become effective?"
]


for query in queries:

    print("\n====================================")
    print("QUERY:", query)
    print("====================================")

    query_embedding = embed_query(query)

    results = search_qdrant(
        query_embedding,
        top_k=3
    )

    for i, result in enumerate(results, start=1):
        print(f"\n----- RESULT {i} -----")

        print("Score:", result.score)
        print("Document:", result.payload["filename"])
        print("Page:", result.payload["page"])
        print("Department:", result.payload["department"])

        print("\nText:")
        print(result.payload["text"])