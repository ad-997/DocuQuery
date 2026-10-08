from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings

def embed_query(query):
    query_embeddings = model.encode(
        query,
        normalize_embeddings=True
    )
    return query_embeddings