"""Authenticated document lifecycle and department-scoped access."""
import logging
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from uuid import uuid4
import time
from app.api.auth import current_user
from app.services.database import engine, documents
from app.vectorstore.qdrant_store import delete_document_chunks
from pypdf.errors import PdfReadError

from app.ingestion.pipeline import ingest_document
from app.services.corpus_service import DATA_DIRECTORY, corpus_service

router = APIRouter(prefix="/api/documents", tags=["Documents"])
DEPARTMENTS = {name.lower(): name for name in (
    "HR", "IT", "Security", "Compliance", "Finance", "Engineering", "Operations", "Sales"
)}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_upload_lock = Lock()


@router.post("", status_code=201)
def upload_document(file: UploadFile = File(...), department: str = Form(...), user=Depends(current_user)):
    filename = (file.filename or "").replace("\\", "/")
    safe_filename = Path(filename).name
    if not safe_filename or Path(safe_filename).suffix.lower() != ".pdf":
        raise HTTPException(400, "A PDF filename is required.")
    department_name = DEPARTMENTS.get(department.strip().lower())
    if department_name is None:
        raise HTTPException(400, "Invalid department.")
    if user['role'] not in ('admin', 'editor'):
        raise HTTPException(403, "Upload permission required.")
    if user['role'] != 'admin' and department_name != user['department']:
        raise HTTPException(403, "Department access denied.")
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "PDF must be 20 MiB or smaller.")
    if not contents:
        raise HTTPException(400, "PDF is empty.")

    document_id = str(uuid4())
    # UUID parent makes IDs collision-safe without changing original filenames.
    destination = DATA_DIRECTORY / department_name / document_id / safe_filename
    # Serialize uploads within this single-process prototype.
    with _upload_lock:
        try:
            with engine.begin() as connection:
                duplicate = connection.execute(select(documents.c.id).where(
                    documents.c.filename == safe_filename, documents.c.department == department_name)).first()
                if duplicate:
                    raise HTTPException(409, "Document already exists.")
                connection.execute(documents.insert().values(id=document_id, filename=safe_filename,
                    department=department_name, path=str(destination), owner_id=user['id'],
                    chunks_created=0, created_at=time.time(), status='indexing'))
        except IntegrityError:
            raise HTTPException(409, "Document already exists.")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(contents)
        except FileExistsError:
            raise HTTPException(409, "Document already exists.")
        except OSError:
            with engine.begin() as connection:
                connection.execute(documents.update().where(documents.c.id == document_id).values(status='failed'))
            logging.exception("Saving upload failed")
            raise HTTPException(500, "Failed to save PDF.")
        try:
            chunks = ingest_document(destination)
        except (PdfReadError, ValueError):
            destination.unlink(missing_ok=True)
            with engine.begin() as connection:
                connection.execute(documents.update().where(documents.c.id == document_id).values(status='failed'))
            raise HTTPException(400, "Invalid PDF or no extractable text; OCR may be required.")
        except Exception:
            destination.unlink(missing_ok=True)
            with engine.begin() as connection:
                connection.execute(documents.update().where(documents.c.id == document_id).values(status='failed'))
            logging.exception("Document ingestion failed")
            raise HTTPException(500, "Document ingestion failed. Check the server logs.")
        try:
            corpus_service.refresh()
        except Exception:
            with engine.begin() as connection:
                connection.execute(documents.update().where(documents.c.id == document_id).values(status='failed'))
            logging.exception('Corpus refresh failed')
            raise HTTPException(503, 'Corpus refresh failed; delete the failed document before retrying.')
        with engine.begin() as connection:
            connection.execute(documents.update().where(documents.c.id == document_id).values(
                chunks_created=len(chunks), status='ready'))
    return {
        "id": document_id,
        "message": "Document uploaded successfully.",
        "filename": safe_filename,
        "department": department_name,
        "chunks_created": len(chunks),
    }


def visible_documents(user, ready_only=True):
    query = select(documents)
    if ready_only:
        query = query.where(documents.c.status == 'ready')
    if user['role'] != 'admin':
        query = query.where(documents.c.department == user['department'])
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(query.order_by(documents.c.created_at, documents.c.id)).mappings()]


def document_record(document_id, user):
    with engine.connect() as connection:
        row = connection.execute(select(documents).where(documents.c.id == document_id)).mappings().first()
    if row is None or (user['role'] != 'admin' and row['department'] != user['department']):
        raise HTTPException(404, 'Document not found.')
    return dict(row)


def public_document(row):
    return {key: value for key, value in row.items() if key != 'path'}


@router.get('')
def list_documents(offset: int = 0, limit: int = 50, user=Depends(current_user)):
    if offset < 0 or not 1 <= limit <= 100:
        raise HTTPException(422, 'offset must be nonnegative and limit must be between 1 and 100.')
    rows = visible_documents(user, ready_only=False)
    return {'items': [public_document(row) for row in rows[offset:offset + limit]], 'total': len(rows)}


@router.get('/{document_id}')
def get_document(document_id: str, user=Depends(current_user)):
    return public_document(document_record(document_id, user))


@router.delete('/{document_id}', status_code=204)
def delete_document(document_id: str, user=Depends(current_user)):
    if user['role'] not in ('admin', 'editor'):
        raise HTTPException(403, 'Delete permission required.')
    with _upload_lock:
        row = document_record(document_id, user)
        # Revoke retrieval access first; partial failures remain hidden and retryable.
        with engine.begin() as connection:
            connection.execute(documents.update().where(documents.c.id == document_id).values(status='deleting'))
        try:
            delete_document_chunks(document_id)
            Path(row['path']).unlink(missing_ok=True)
            corpus_service.refresh()
        except Exception:
            logging.exception('Document deletion failed')
            raise HTTPException(503, 'Deletion incomplete; document access has been revoked.')
        with engine.begin() as connection:
            connection.execute(documents.delete().where(documents.c.id == document_id))
    return Response(status_code=204)
