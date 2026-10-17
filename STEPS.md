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
   - Creates a separate temporary Qdrant collection for clean experiments
   - Deletes that temporary collection after the run
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

48. Expanded retrieval evaluation to compare:

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

49. Completed the **BM25, hybrid RRF and reranker benchmarks** using the same labelled questions.

    Saved the results in `data/retrieval_evaluation.json`, including:
    - Recall@1, Recall@3 and Recall@5
    - MRR@5
    - Per-query retrieved documents
    - Improvements and regressions between methods

50. Created `diversify.py` inside the retrieval module to experiment with repeated chunks crowding out other relevant documents.

    It limits how many candidate chunks can come from the same document while preserving the fused ranking order.

51. Added a separate **diversification + reranking experiment** to `evaluate_retrieval.py`.

    The original baseline uses 10 candidates from each retriever. The experiment uses 30, keeps at most two chunks per document, and reranks up to 10 candidates into five results.

    Because both candidate depth and diversity change, the results do not measure diversification alone. This remains a benchmark experiment rather than the default RAG path.

52. Created `test_retrieval_evaluation.py` inside `backend/tests/` to check retrieval metrics, diversification and benchmark comparisons.

53. Created `context_builder.py` inside a new `generation` module.

    It converts retrieved passages into context blocks containing:
    - A label such as `[SOURCE 1]`
    - Document ID
    - Filename
    - Page number
    - Passage text

54. Created `generator.py` to connect the retrieved context to **Groq** for answer generation.

    The prompt instructs the model to:
    - Answer using only the supplied context
    - Cite the supporting source labels
    - Acknowledge conflicting sources
    - Say when the context is insufficient

    These instructions guide the model; answer accuracy and citation grounding still need evaluation.

55. Created `pipeline.py` inside the `rag` module to connect retrieval and generation in one reusable function, `run_rag_pipeline()`.

    The default query flow became:

    `Question → dense + BM25 → RRF → reranker → context → generated answer`

    It requests 10 candidates per retriever, keeps 10 fused candidates and selects three final passages.

56. Updated `testing_pipeline.py` to build context and generate an answer after reranking.

    It still supports testing the retrieval stages independently through its `RUN_*` flags.

57. Added `BM25Retriever` in `sparse.py` so the tokenized corpus and BM25 index can be built once and reused across questions.

    Kept `search_bm25()` as a convenience wrapper for existing callers. Updated the manual and evaluation scripts to reuse an index.

58. Created `corpus_service.py` inside a new `services` module.

    It loads the local PDF chunks and caches their BM25 retriever on first use. Refreshing it rebuilds the lexical index without re-embedding the corpus or writing to Qdrant.

59. Created `query.py` inside a new `api` module and added:

    `POST /api/query`

    It validates the question, passes the cached BM25 retriever to the RAG pipeline and returns the answer with source metadata.

60. Created `documents.py` inside the API module and added:

    `POST /api/documents`

    It accepts a PDF and department, saves the file, ingests its chunks into Qdrant and refreshes BM25.

61. Updated `main.py` to register the query and document routers and added `GET /health` alongside the existing root endpoint.

    These endpoints report process liveness; they do not check Qdrant or Groq availability.

62. Added validation and error handling around the existing pipeline:
    - Reject blank questions
    - Validate PDF filenames and departments
    - Limit uploads to 20 MiB
    - Reject duplicate filenames
    - Remove the saved upload when ingestion fails
    - Reject PDFs with no extractable text
    - Validate chunk size and overlap to prevent invalid chunking loops
    - Handle empty BM25 corpora and empty reranking inputs

63. Changed `embedder.py` and `reranker.py` to load their models on first use instead of during application import.

    Added an empty-context response in `generator.py` so a question with no retrieved evidence does not call Groq.

64. Fixed the query endpoint's pipeline arguments, restored the BM25 wrapper used by scripts, and corrected the manual script's generation indentation.

    Also replaced the machine-specific corpus path with a project-relative path and corrected the Qdrant helper's storage path and usage message.

65. Added `requirements.txt` and `.env.example` to document dependencies and Groq configuration.

    Updated the Markdown files to describe the implemented RAG/API flow and remaining limitations. Dependencies are listed but not locked.

66. Created `test_pipeline_regressions.py` to cover API wiring, upload validation and cleanup, empty-input handling, PDF text fallback and isolated Qdrant upsert/search behavior.

    The maintenance checks passed all 14 tests, Python compilation, shell syntax validation and the diff whitespace check. Live Groq generation and the full retrieval benchmark were not rerun during this pass.

## Current Pipeline

```text
Enterprise PDF Corpus
          ↓
    Page Text + Metadata
          ↓
   Overlapping Chunks
          ↓
      Embeddings
          ↓
        Qdrant

User Question
      ├── Query Embedding → Qdrant Dense Search ──┐
      └── Cached BM25 Search ────────────────────┤
                                                ↓
                                               RRF
                                                ↓
                                      Cross-Encoder Reranker
                                                ↓
                                       Source-Labelled Context
                                                ↓
                                          Groq Generation
                                                ↓
                                       Answer + Source Metadata
```

The query and upload flows are available through FastAPI. The evaluation runner separately compares retrieval strategies against the labelled corpus.

## Current Status

The project has grown from an in-memory semantic-search experiment into a local RAG prototype with retrieval evaluation, answer generation and API endpoints.

The next work is document lifecycle handling, version-aware retrieval, authentication and retrieval permissions, followed by answer-quality and citation evaluation. OCR, multiple-worker cache coordination and recovery across files, Qdrant and BM25 are also unresolved.

See [README](README.md) for current setup commands and API details, and [Progress](PROGRESS.MD) for the remaining work.
