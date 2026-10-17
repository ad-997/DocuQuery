# DocuQuery

Enterprise RAG Knowledge Platform.

## Progress

### Completed
- [x] Project structure
- [x] FastAPI backend setup
- [x] PDF loading using PyPDF
- [x] Page-wise text extraction
- [x] Fixed-size chunking
- [x] Chunk overlap
- [x] Page metadata preserved during chunking
- [x] Document metadata extraction
- [x] Generate document chunk embeddings
- [x] Generate query embeddings
- [x] Basic in-memory dense retrieval
- [x] Cosine similarity using normalized embeddings
- [x] Top-K retrieval
- [x] Run Qdrant locally using Docker
- [x] Create Qdrant vector collection
- [x] Store chunk vectors in Qdrant
- [x] Store document metadata as Qdrant payload
- [x] Deterministic point IDs for document chunks
- [x] Upsert chunks into Qdrant
- [x] Dense retrieval using Qdrant
- [x] BM25 sparse retrieval
- [x] Hybrid dense + sparse retrieval
- [x] Reciprocal Rank Fusion (RRF)
- [x] Cross-encoder reranking
- [x] Full-corpus PDF ingestion pipeline
- [x] Recursive ingestion of documents across departments
- [x] Clean Qdrant collection reset for evaluation runs
- [x] Load labelled retrieval evaluation dataset
- [x] Document-level Recall@K evaluation
- [x] Mean Reciprocal Rank (MRR) evaluation
- [x] Dense retrieval benchmark across the enterprise corpus
- [x] Separate ACL/security queries from standard retrieval evaluation

### In Progress
- [x] BM25 retrieval evaluation
- [x] Dense vs BM25 retrieval comparison
- [x] RRF hybrid retrieval evaluation
- [x] Reranker evaluation
- [x] Retrieval benchmark comparison across all retrieval strategies
- [x] Per-query improvement, regression, and Top-5 failure reports
- [x] Candidate diversification experiment with preserved reranker baseline
- [x] Retrieval evaluation regression tests

### Next
- [ ] Improve and evaluate chunking strategies
- [ ] Handle duplicate / highly repetitive chunks
- [ ] Document version-aware retrieval
- [ ] Current vs archived document filtering
- [ ] Metadata-based retrieval filtering
- [ ] LLM answer generation
- [ ] Grounded context construction
- [ ] Source and page citations
- [ ] Hallucination / insufficient-context handling
- [ ] JWT authentication
- [ ] Role-Based Access Control (RBAC)
- [ ] Retrieval-level permission filtering
- [ ] Restricted-document leakage evaluation
- [ ] Full RAG answer evaluation
- [ ] React frontend
- [ ] PostgreSQL integration
- [ ] Full Docker / Docker Compose setup
- [ ] Automated tests
- [ ] Logging and error handling
- [ ] Deployment

### Later / Advanced
- [ ] Async document ingestion
- [ ] Background ingestion workers
- [ ] Kafka-based ingestion pipeline
- [ ] Document update / deletion lifecycle
- [ ] Document versioning
- [ ] Retrieval experiment tracking
- [ ] Production monitoring