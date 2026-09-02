from pathlib import Path
import hashlib


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def ingest_document(path: Path) -> dict:
    return {
        "filename": path.name,
        "filepath": str(path),
        "sha256": calculate_sha256(path),
    }
