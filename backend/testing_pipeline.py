from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.ingestion.metadata import extract_metadata
from app.embedding.embedder import embed_chunks, embed_query
from app.retrieval.dense import search_chunks


file_path = "../data/raw/HR/hr-006_parental_leave_guide.pdf"

metadata = extract_metadata(file_path)
pages = load_pdf(file_path)

chunks = chunk_pages(
    pages,
    metadata,
    chunk_size=500,
    overlap=100
)

chunk_embeddings = embed_chunks(chunks)

query = "How much parental leave can an employee take?"

query_embedding = embed_query(query)

results = search_chunks(
    query_embedding,
    chunk_embeddings,
    chunks,
    top_k=3
)

for i, result in enumerate(results, start=1):
    print(f"\n---- RESULT {i} ---")
    print("Score", result["score"])
    print("Page", result["chunk"]["page"])
    print("Document: ", result["chunk"]["filename"])
    print(result["chunk"]["text"])