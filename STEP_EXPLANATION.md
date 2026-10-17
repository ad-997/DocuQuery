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

See [README](README.md) for commands, saved benchmark numbers and limitations, and [Progress](PROGRESS.MD) for outstanding work.
