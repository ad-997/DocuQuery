### Steps 1–10

At this stage, we built a basic **in-memory semantic search system**.

Each document is split into smaller chunks, and every chunk is converted into an embedding vector. The user's query is also converted into an embedding using the same model.

We then compare the query embedding against all chunk embeddings using cosine similarity and return the chunks with the highest similarity scores.

In simple terms:

```text
User Query
↓
Query Embedding
↓
Compare against all Chunk Embeddings
↓
Rank by similarity
↓
Return Top-K relevant chunks
```

This works well for learning and small datasets, but all embeddings currently live in memory and must be regenerated and searched again whenever the pipeline runs.

The next step is **Qdrant**, which gives us persistent vector storage and allows us to perform vector similarity search without manually comparing every embedding in Python.
