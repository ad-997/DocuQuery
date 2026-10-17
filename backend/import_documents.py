"""Copy legacy department PDFs into the managed API corpus and reindex them.

Original PDFs are retained. Old vectors are excluded by the API's ID allowlist.
"""
from fastapi import UploadFile
from sqlalchemy import select
from app.services.database import initialize, engine, users, documents
from app.api.auth import public_user
from app.api.documents import DATA_DIRECTORY, DEPARTMENTS, upload_document


def main():
    initialize()
    with engine.connect() as connection:
        admin = connection.execute(select(users).where(users.c.role == 'admin')).mappings().first()
    if admin is None:
        raise SystemExit('Create an administrator with backend/manage.py first.')
    for department in DEPARTMENTS.values():
        for path in sorted((DATA_DIRECTORY / department).glob('*')):
            if not path.is_file() or path.suffix.lower() != '.pdf':
                continue
            with engine.connect() as connection:
                existing = connection.execute(select(documents.c.id, documents.c.status).where(
                    documents.c.filename == path.name, documents.c.department == department)).first()
            if existing:
                print(f'Skipped {path.name}: {existing.status} ({existing.id})')
                continue
            with path.open('rb') as stream:
                result = upload_document(UploadFile(file=stream, filename=path.name), department, public_user(admin))
            print(f'Imported {path.name}: {result["id"]}')


if __name__ == '__main__':
    main()
