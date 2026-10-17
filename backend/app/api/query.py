import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.auth import current_user
from app.api.documents import visible_documents
from app.rag.pipeline import run_rag_pipeline
from app.services.corpus_service import corpus_service


router = APIRouter(
    prefix="/api",
    tags=["RAG"]
)


class QueryRequest(BaseModel):
    question: str


@router.post("/query")
def query_document(request: QueryRequest, user=Depends(current_user)):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = run_rag_pipeline(
            question=question,
            bm25_retriever=corpus_service.retriever,
            allowed_document_ids=[row["id"] for row in visible_documents(user)]
        )

        return result

    except Exception:
        logging.exception("Query processing failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to process query. Check the server logs."
        )