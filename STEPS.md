1. Installed the dependencies required so far. Additional dependencies are being added incrementally as the project progresses.

2. Set up a basic **FastAPI** backend and ran it using **Uvicorn**.

3. Created `pdf_loader.py` inside the ingestion module. It takes a PDF as input and converts it into a Python list where each item contains:
   - Page number
   - Text extracted from that page

4. Created `testing_pipeline.py` as a manual testing environment. It is used to test individual parts of the retrieval pipeline before connecting them to FastAPI endpoints.

5. Created `metadata.py` to extract document metadata from the file path, including:
   - Document ID
   - Filename
   - Department
   - File type

6. Created `chunker.py`, which takes the page text returned by `load_pdf()` and splits it into smaller overlapping chunks.

   Current baseline chunking strategy:

   `PDF page text → 500-character chunk → next chunk with 100-character overlap`

   Each chunk preserves:
   - Document ID
   - Filename
   - Department
   - File type
   - Page number
   - Chunk text

7. Created `embedder.py` using the `BAAI/bge-small-en-v1.5` embedding model.

   It:
   - Converts document chunks into embedding vectors
   - Converts user queries into query embedding vectors
   - Normalizes embeddings

   The current embedding size is:

   `384 dimensions`

8. Created `dense.py` to understand how dense retrieval works manually before introducing a vector database.

   It compares the normalized query embedding against all normalized chunk embeddings.

9. Because both document and query embeddings are normalized, the dot product can be used as cosine similarity.

10. Sorted similarity scores in descending order and returned the top `k` most similar chunks.

11. Added **Qdrant** as the vector database and ran it locally using Docker.

12. Created a Qdrant collection called:

   `docuquery_chunks`

   Current configuration:
   - Vector size: `384`
   - Distance metric: `COSINE`

13. Created `qdrant_store.py` to handle Qdrant operations.

14. Added collection-management functionality including:
   - Create collection
   - Reset collection
   - Check whether the collection already exists

15. Added `upsert_chunks()` to store document chunks inside Qdrant.

16. Each Qdrant point currently contains:
   - Deterministic UUID
   - Embedding vector
   - Payload containing:
     - Document ID
     - Filename
     - Department
     - File type
     - Page number
     - Original chunk text

17. Used deterministic UUIDs so that re-ingesting the same chunk generates the same point ID instead of blindly creating duplicate points.

18. Used Qdrant `upsert` so an existing point can be updated when the same ID is written again.

19. Added `search_qdrant()` to perform dense vector retrieval directly through Qdrant.

20. Dense retrieval now works as:

   `User query → query embedding → Qdrant vector search → top-k semantically similar chunks`

21. Verified that Qdrant returns:
   - Similarity score
   - Chunk text
   - Document ID
   - Filename
   - Page number
   - Department

22. Created `sparse.py` and added **BM25 sparse retrieval**.

   BM25 provides lexical retrieval based on words and terms appearing in the query and document chunks.

   This gives the system two different retrieval approaches:

   `Dense retrieval → semantic similarity`

   `BM25 retrieval → keyword / lexical relevance`

23. Added a simple tokenizer for BM25 that:
   - Converts text to lowercase
   - Extracts individual words
   - Creates tokenized document chunks and queries

24. Used `BM25Okapi` to calculate BM25 relevance scores and return the highest-ranked chunks.

25. Tested dense retrieval and BM25 independently using the same queries in `testing_pipeline.py`.

26. Created `rrf.py` and implemented **Reciprocal Rank Fusion (RRF)**.

27. RRF combines the rankings from:
   - Dense retrieval
   - BM25 retrieval

   rather than directly adding their raw scores.

28. RRF is used because dense similarity scores and BM25 scores exist on different scales and should not be directly compared.

29. Hybrid retrieval now works approximately as:

```text
Dense Retrieval ──┐
                  ├── RRF ──→ Hybrid Ranking
BM25 Retrieval ───┘
```

30. Created a `reranking` module and implemented a **Cross-Encoder reranker**.

31. The reranker takes:
   - User query
   - Candidate chunk

   and evaluates them together to produce a more precise relevance score.

32. Retrieval now first finds a wider candidate set and then reranks those candidates.

   Current retrieval architecture:

```text
User Query
   ↓
Dense Retrieval
   +
BM25 Retrieval
   ↓
Reciprocal Rank Fusion
   ↓
Top Candidate Chunks
   ↓
Cross-Encoder Reranker
   ↓
Best Final Chunks
```

33. Updated `testing_pipeline.py` so that individual retrieval stages can be tested independently.

   It can currently test:
   - Dense retrieval only
   - BM25 only
   - Hybrid / RRF only
   - Reranking
   - Entire retrieval pipeline

34. Created `pipeline.py` inside the ingestion module.

35. Added `ingest_document()` to handle ingestion of a single PDF.

36. Added `ingest_directory()` to recursively discover and ingest PDFs from the entire enterprise document corpus.

37. The ingestion pipeline now supports documents from multiple departments including:
   - HR
   - IT
   - Security
   - Compliance
   - Finance
   - Engineering
   - Operations
   - Sales

38. Full-corpus ingestion now works as:

```text
Document Corpus
↓
Discover PDFs
↓
Load pages
↓
Extract metadata
↓
Chunk
↓
Generate embeddings
↓
Store in Qdrant
```

39. Created an `evaluation` module.

40. Added `retrieval_metrics.py` for measuring retrieval performance.

41. Implemented **Document Recall@K**.

   This measures how many expected documents were retrieved within the first `K` results.

42. Implemented **Reciprocal Rank** and **Mean Reciprocal Rank (MRR)**.

   Reciprocal Rank rewards systems that place the first correct document higher in the result list.

43. Created `evaluate_retrieval.py` as a separate benchmark runner.

44. The evaluation script:
   - Resets Qdrant for clean experiments
   - Ingests the complete document corpus
   - Loads `evaluation_queries.json`
   - Runs retrieval queries
   - Compares retrieved documents against expected documents
   - Calculates retrieval metrics

45. Separated normal retrieval evaluation from ACL/security evaluation.

   Queries beginning with:

   `ACL-`

   are currently excluded from standard retrieval metrics because they are intended to test authorization and restricted-document leakage rather than retrieval relevance.

46. Completed the first **dense retrieval baseline evaluation** across the enterprise corpus.

47. The evaluation has already revealed several real retrieval problems, including:
   - Semantically similar but incorrect documents
   - Current vs archived policy confusion
   - Multiple relevant-document retrieval issues
   - Repeated chunks occupying multiple result positions
   - Restricted documents being retrievable before RBAC/filtering is implemented

48. Retrieval evaluation is now being expanded so we can compare:

```text
Dense
↓
BM25
↓
Dense + BM25 + RRF
↓
RRF + Cross-Encoder Reranker
```

using the same labelled evaluation dataset.

## Current Pipeline

```text
Enterprise PDF Corpus
↓
Load PDFs page-by-page
↓
Extract document metadata
↓
Split pages into overlapping chunks
↓
Generate embeddings
↓
Store vectors + metadata in Qdrant


User Query
↓
Generate query embedding
↓
┌─────────────────┐
│ Dense Retrieval │
└────────┬────────┘
         │
         ├─────────────┐
         │             │
┌────────▼────────┐    │
│ BM25 Retrieval  │    │
└────────┬────────┘    │
         │             │
         └──────┬──────┘
                ↓
               RRF
                ↓
       Candidate Chunks
                ↓
     Cross-Encoder Reranker
                ↓
        Best Final Chunks
                ↓
       Retrieval Evaluation
```

## Current Status

### Completed

- PDF ingestion
- Page-wise text extraction
- Document metadata extraction
- Baseline fixed-size chunking
- Chunk overlap
- Page metadata preservation
- Document embeddings
- Query embeddings
- Manual in-memory dense retrieval
- Cosine similarity / dot-product retrieval
- Top-K retrieval
- Qdrant Docker setup
- Qdrant vector collection
- Vector storage in Qdrant
- Metadata payload storage
- Deterministic point IDs
- Qdrant upsert
- Dense retrieval through Qdrant
- BM25 sparse retrieval
- Hybrid dense + sparse retrieval
- Reciprocal Rank Fusion
- Cross-Encoder reranking
- Full-corpus recursive ingestion
- Reusable document ingestion pipeline
- Retrieval evaluation framework
- Document Recall@K
- Reciprocal Rank / MRR
- Dense retrieval baseline evaluation
- Separation of ACL/security queries from relevance evaluation

### Currently Working On

- BM25 retrieval benchmark
- Dense vs BM25 comparison
- RRF hybrid retrieval benchmark
- Reranker benchmark
- Retrieval strategy comparison

### Next

- Improve and evaluate chunking strategies
- Handle duplicate / repetitive chunks
- Version-aware retrieval
- Current vs archived document handling
- Metadata filtering
- LLM answer generation
- Context construction
- Source and page citations
- Insufficient-context handling
- JWT authentication
- Role-Based Access Control
- Retrieval-level permission filtering
- Restricted-document leakage evaluation
- Full RAG answer evaluation
- React frontend
- PostgreSQL integration
- Automated tests
- Full Docker Compose setup
- Logging and monitoring
- Deployment

### Later

- Document update/delete lifecycle
- Better deterministic chunk IDs
- Background ingestion workers
- Asynchronous ingestion
- Kafka-based ingestion pipeline
- Experiment tracking
- Production monitoring