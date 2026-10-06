from pathlib import Path

def extract_metadata(file_path):
    path = Path(file_path)
    filename = path.name
    department = path.parent.name
    document_id = filename.split("_")[0]

    return {
        "document_id":document_id,
        "filename":filename,
        "department":department,
        "file_type":path.suffix
    }