from pathlib import Path

def extract_metadata(file_path):
    path = Path(file_path)
    filename = path.name
    from uuid import UUID
    try:
        UUID(path.parent.name)
        managed = True
    except ValueError:
        managed = False
    department = path.parent.parent.name if managed else path.parent.name
    document_id = path.parent.name if managed else filename.split("_")[0]

    return {
        "document_id":document_id,
        "filename":filename,
        "department":department,
        "file_type":path.suffix
    }