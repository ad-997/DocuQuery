# How the pipeline works

## Ingestion

PDFs are read page by page. Metadata comes from each file's name and department directory. Text is split into overlapping character windows, preserving page references. Empty documents cannot be ingested without OCR.

BGE converts chunks into normalized 384-dimensional vectors. Qdrant stores each vector with its text and metadata. Deterministic IDs make identical repeat ingestion update points. They do not provide a complete document-update lifecycle: obsolete chunks can remain after a document changes.

## Retrieval

A question is embedded for dense search. A cached BM25 index independently retrieves lexical matches from the local PDFs. These branches run separately, then reciprocal rank fusion combines their rank positions because their raw scores have different scales.

A cross-encoder scores each fused candidate together with the question. The normal RAG path requests 10 dense and 10 BM25 candidates, retains 10 fused candidates and returns the top three reranked passages. Reranker scores indicate relative relevance; they are not confidence probabilities.

## Generation

Selected passages become `[SOURCE n]` blocks with document ID, filename and page. Groq receives the question, context and instructions to answer only from that evidence, cite sources and acknowledge conflicts or insufficient information. Empty evidence returns a fixed message without a provider call. For nonempty evidence, correctness and citation grounding still require evaluation.

## API and cache

The query endpoint uses a reusable BM25 retriever. Uploads validate department, filename and size, save a PDF, ingest it into Qdrant and refresh the lexical cache. A process-local lock serializes uploads. The application is intended for one local process; cache coordination and atomic recovery across storage systems remain future work.

Health endpoints report that the API is running. Models load on first retrieval use, so health checks do not require model downloads or provider credentials.

## Evaluation

The benchmark compares dense, BM25, RRF, RRF plus reranking and a diversified reranking experiment. Document Recall@K measures the fraction of expected documents present in the first K chunk results. MRR@5 rewards the first relevant result's position. Duplicate chunks still occupy rank positions.

Diversification limits document crowding before reranking, while also increasing retrieval depth from 10 to 30. It does not guarantee every relevant document reaches the final five. ACL questions are excluded because relevance evaluation does not establish permission enforcement.

---

### **Steps 11–12 — Moving Vector Search to Qdrant**

The next step was introducing Qdrant as the vector database.

Instead of keeping vectors only in Python memory, we now store each chunk inside Qdrant as:

```text
Point
├── ID
├── Embedding vector
└── Payload
    ├── document_id
    ├── filename
    ├── department
    ├── page
    └── text
```

We also created deterministic point IDs so that re-ingesting the same chunks updates existing points rather than creating unnecessary duplicates.

The retrieval flow became:

```text
Documents
↓
Chunk
↓
Embed
↓
Store vectors + metadata in Qdrant
```

Then at query time:

```text
User Question
↓
Query Embedding
↓
Qdrant Vector Search
↓
Top-K semantically similar chunks
```

This removed the need to manually compare the query against every vector using NumPy.

Our original in-memory dense retrieval is still useful as a learning and comparison baseline, while Qdrant now acts as the real vector-search layer.

---

### **Step 13 — Adding BM25 Sparse Retrieval**

Dense retrieval understands semantic meaning well, but semantic similarity does not always mean that a chunk contains the exact information needed to answer the question.

For example, dense retrieval may understand that:

```text
"employee vacation entitlement"
```

is related to:

```text
"annual leave allowance"
```

But sometimes exact words, identifiers, product names, policy names or technical terms matter more.

For this reason, we added BM25.

BM25 performs lexical retrieval based on terms appearing in the query and document chunks.

We now have two independent retrieval systems:

```text
Dense Retrieval
→ semantic meaning

BM25 Retrieval
→ lexical / keyword relevance
```

This gives us two different views of which chunks may be relevant.

---

### **Step 14 — Hybrid Retrieval with Reciprocal Rank Fusion**

Dense and BM25 scores cannot be directly added together because their score ranges mean completely different things.

For example:

```text
Dense score = 0.78
BM25 score  = 6.42
```

The value `6.42` is not automatically better than `0.78`.

Instead, we combine the **rankings** produced by the two systems using Reciprocal Rank Fusion.

The flow became:

```text
                 ┌── Dense Retrieval ──┐
User Question ───┤                     ├── RRF
                 └── BM25 Retrieval ───┘
                                            ↓
                                      Hybrid Ranking
```

RRF gives additional importance to chunks that appear near the top of multiple retrieval systems.

For example:

```text
Dense:
1. Chunk A
2. Chunk C
3. Chunk B

BM25:
1. Chunk B
2. Chunk A
3. Chunk D
```

RRF may produce:

```text
1. Chunk A
2. Chunk B
3. Chunk C
4. Chunk D
```

This gives us a more robust hybrid retrieval system than relying only on embeddings or only on keywords.

---

### **Step 15 — Cross-Encoder Reranking**

Hybrid retrieval is good at finding a strong candidate set, but it still does not deeply compare every candidate against the question.

We therefore added a cross-encoder reranker.

The first-stage retrievers are designed to be fast:

```text
Dense
+
BM25
↓
RRF
↓
Top candidate chunks
```

The reranker then performs a more expensive but more accurate comparison between:

```text
Question + Chunk
```

and assigns a relevance score.

The complete retrieval pipeline is now:

```text
User Question
      ↓

┌───────────────┐
│ Dense Search  │
└───────┬───────┘
        │
        ├─────────────┐
        │             │
┌───────▼───────┐     │
│ BM25 Search   │     │
└───────┬───────┘     │
        │             │
        └─────┬───────┘
              ↓
             RRF
              ↓
      Candidate Chunks
              ↓
     Cross-Encoder Reranker
              ↓
       Best Final Chunks
```

The purpose of the reranker is to answer a more precise question:

> Out of the chunks already retrieved, which ones actually match the user's question most strongly?

---

### **Step 16 — Full-Corpus Ingestion**

Initially, retrieval was tested using a single PDF such as the parental leave guide.

That was useful for developing the pipeline, but it was not a realistic test because retrieving parental-leave information from a database containing only a parental-leave document is easy.

We therefore created a reusable ingestion pipeline that recursively loads the entire enterprise corpus:

```text
HR
IT
Finance
Security
Compliance
Engineering
Operations
Sales
...
↓
PDF loading
↓
Metadata extraction
↓
Chunking
↓
Embedding
↓
Qdrant
```

Now retrieval must find the correct information among many similar policies, SOPs, archived documents, guides and restricted documents.

This makes retrieval testing much closer to a real enterprise knowledge system.

---

### **Step 17 — Retrieval Evaluation**

Instead of judging retrieval quality manually, we started building a proper evaluation pipeline.

The dataset already contains labelled evaluation questions containing information such as:

```text
Question
Expected document
Expected answer
Expected page
Task type
Allowed roles
```

We currently evaluate retrieval using metrics such as:

#### Document Recall@K

Measures how many expected documents appear within the first K retrieved results.

For example:

```text
Expected:
HR-001
HR-003

Top 5 contains:
HR-003
HR-001
```

then:

```text
Document Recall@5 = 2 / 2 = 1.0
```

#### Mean Reciprocal Rank (MRR)

Measures how high the first correct result appears.

```text
Correct at rank 1 → 1.0
Correct at rank 2 → 0.5
Correct at rank 3 → 0.33
```

The goal is to benchmark every retrieval strategy using exactly the same dataset:

```text
Dense
↓
BM25
↓
Dense + BM25 + RRF
↓
RRF + Cross-Encoder Reranker
```

This allows us to measure whether every additional retrieval component actually improves performance instead of assuming that a more complicated architecture is automatically better.

---

### **Current State**

DocuQuery has now evolved from:

```text
PDF
↓
Embedding
↓
Similarity Search
```

into:

```text
Enterprise Document Corpus
          ↓
       Ingestion
          ↓
       Chunking
          ↓
      Embeddings
          ↓
        Qdrant
          ↓
   ┌──────┴──────┐
   ↓             ↓
 Dense          BM25
   └──────┬──────┘
          ↓
         RRF
          ↓
      Reranker
          ↓
    Best Evidence
          ↓
 Retrieval Evaluation
```

The next major step after completing retrieval evaluation will be:

```text
Best Retrieved Evidence
          ↓
     LLM Generation
          ↓
Grounded Answer
          ↓
Source + Page Citations
```

At that point, DocuQuery will move from being a retrieval system into a complete Retrieval-Augmented Generation system. 
