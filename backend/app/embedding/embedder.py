from functools import lru_cache

from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = get_model().encode(
        texts,
        normalize_embeddings=True
    )
    return embeddings

def embed_query(query):
    query_embeddings = get_model().encode(
        query,
        normalize_embeddings=True
    )
    return query_embeddings
