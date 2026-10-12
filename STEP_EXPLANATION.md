### Steps 1–10

At this stage, we first built a basic **in-memory semantic search system**.

Each document is split into smaller chunks, and every chunk is converted into an embedding vector. The user's query is also converted into an embedding using the same model.

We then compare the query embedding against all chunk embeddings using cosine similarity and return the chunks with the highest similarity scores.

```text
User Query
↓
Query Embedding
↓
Compare against Chunk Embeddings
↓
Rank by Similarity
↓
Return Top-K Relevant Chunks
```

This helped us understand how dense semantic retrieval works before introducing a vector database.

We then integrated **Qdrant** to persist document embeddings and metadata and to perform vector similarity search through the database instead of manually comparing every embedding in Python.

The retrieval flow is now:

```text
Document Chunks
↓
Generate Embeddings
↓
Store Vectors + Metadata in Qdrant

User Query
↓
Generate Query Embedding
↓
Search Qdrant
↓
Return Top-K Relevant Chunks
```
