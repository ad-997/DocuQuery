from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.ingestion.metadata import extract_metadata


file_path = "../data/raw/HR/hr-006_parental_leave_guide.pdf"

metadata = extract_metadata(file_path)
pages = load_pdf(file_path)

chunks = chunk_pages(
    pages,
    metadata,
    chunk_size=500,
    overlap=100
)

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk)