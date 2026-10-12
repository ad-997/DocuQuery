1. Installed the dependencies required so far. Additional dependencies will be added as the project progresses.

2. Set up a basic **FastAPI** server and ran it using **Uvicorn**.

3. Created `pdf_loader.py` inside the ingestion module. It takes a PDF as input and converts it into a Python list where each item contains:
   - Page number
   - Text extracted from that page

4. Created a basic `testing_pipeline.py` used to test whether each individual part of the pipeline works before connecting everything to FastAPI.

5. Created `metadata.py` to extract basic document metadata from the file path, including:
   - Document ID
   - Filename
   - Department
   - File type

6. Created `chunker.py`, which takes the page text returned by `load_pdf()` and splits it into smaller overlapping chunks.

   Current baseline chunking strategy:

   `PDF page text → 500-character chunk → next 500-character chunk with 100-character overlap`

   Each chunk also preserves its page number and document metadata.

7. Created `embedder.py`, which:
   - Converts document chunks into embedding vectors
   - Converts the user's query into an embedding vector

   Tested both using `testing_pipeline.py`.

8. Created `dense.py`, which performs basic in-memory dense retrieval by comparing the normalized query embedding against all normalized chunk embeddings.

9. Since the embeddings are normalized, the dot product acts like cosine similarity.

10. Sorted similarity scores in descending order and returned the top `k` matching chunks.

11. Added **Qdrant** as the vector database and ran Qdrant locally using Docker.

12. Created a Qdrant collection called `docuquery_chunks`.

   The collection currently uses:
   - Vector size: `384`
   - Distance metric: `COSINE`

13. Created `qdrant_store.py` to handle Qdrant-related functionality.

14. Added `upsert_chunks()` to store all document chunks in Qdrant.

15. Each Qdrant point contains:
   - A deterministic UUID
   - The chunk embedding vector
   - Payload metadata including:
     - Document ID
     - Filename
     - Department
     - File type
     - Page number
     - Original chunk text

16. Used deterministic UUIDs so that re-ingesting the same chunk produces the same point ID instead of blindly creating duplicates.

17. Used Qdrant `upsert` so existing points can be updated when the same ID is written again.

18. Added `search_qdrant()` to perform dense vector retrieval directly through Qdrant.

19. Query flow now works as:

   `User query → query embedding → Qdrant vector search → top-k matching chunks`

20. Verified that Qdrant returns:
   - Similarity score
   - Matching chunk text
   - Filename
   - Page number
   - Department

## Current Pipeline

```text
PDF
↓
Load text page-by-page
↓
Extract metadata
↓
Split into chunks
↓
Generate embeddings
↓
Store vectors + metadata in Qdrant

User query
↓
Generate query embedding
↓
Search Qdrant
↓
Return top-k matching chunks
```

## Current Status

Completed:
- PDF ingestion
- Basic metadata extraction
- Baseline chunking
- Embeddings
- Manual dense retrieval
- Qdrant vector storage
- Qdrant dense retrieval

Next:
- BM25 sparse retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion (RRF)
- Reranking
- LLM answer generation
- Source/page citations
- JWT authentication
- RBAC
- Evaluation
- React frontend
- Dockerized full stack