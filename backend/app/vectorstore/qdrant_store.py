import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "docuquery_chunks"

client = QdrantClient(url="http://localhost:6333")

def create_collection():
    if client.collection_exists(COLLECTION_NAME):
        print("Collection already exists")
        return
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE    
        )
    )

def reset_collection():
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    create_collection()
    
def upsert_chunks(chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must have same length")
    
    points = []
    
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{chunk['document_id']}:{index}"
            )
        )
    
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={
                "document_id":chunk["document_id"],
                "filename":chunk["filename"],
                "department":chunk["department"],
                "file_type":chunk["file_type"],
                "page":chunk["page"],
                "text":chunk["text"]
            }
        )
        points.append(point)
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )
    
    print(f"Inserted {len(points)} chunks into  Qdrant")
    

def search_qdrant(query_embedding, top_k=3):
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        with_payload=True,
        limit=top_k
    ).points
    
    return results