from pathlib import Path
import hashlib


def calculate_sha256(
    path: Path | str,
) -> str:
    """
    Calculate the SHA-256 checksum of a file.
    """

    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Source file not found: {path}"
        )

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def inspect_source_file(
    path: Path | str,
) -> dict:
    """
    Inspect a physical source file.

    Returns filesystem metadata needed during ingestion.

    Path storage conventions for PostgreSQL are handled
    separately by src.paths.
    """

    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Source file not found: {path}"
        )

    return {
        "filename": path.name,
        "path": path,
        "sha256": calculate_sha256(path),
    }
