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

### Next
- [ ] BM25 sparse retrieval
- [ ] Hybrid retrieval
- [ ] Reciprocal Rank Fusion (RRF)
- [ ] Reranking
- [ ] Improve and evaluate chunking strategies
- [ ] LLM answer generation
- [ ] Source and page citations
- [ ] JWT authentication
- [ ] Role-Based Access Control (RBAC)
- [ ] Retrieval-level permission filtering
- [ ] RAG evaluation
- [ ] React frontend
- [ ] Full Docker / Docker Compose setup
- [ ] Deployment